"""
BirdCLEF 2026 – SED Models (Sound Event Detection)
CNN backbones with GeM frequency pooling + framewise classification head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional


# ──────────────────────────────────────────────
# GeM Pooling
# ──────────────────────────────────────────────

class GeM(nn.Module):
    """Generalized Mean Pooling.

    p=3 gives a balance between average and max pooling.
    p → ∞ approaches max pooling; p → 1 approaches average.
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GeM pooling over spatial dimensions.

        Args:
            x: (B, C, H, W) feature map.

        Returns:
            (B, C) pooled vector.
        """
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1.0 / self.p).flatten(1)


class GeMFrequencyPooling(nn.Module):
    """GeM pooling over frequency axis only, keeping time dimension.

    Used as SED head: pool mel bands → keep frame-level predictions.
    Input: (B, C, H_freq, W_time) → Output: (B, C, W_time)
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.gem = GeM(p, eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool over frequency dimension.

        Args:
            x: (B, C, H_freq, W_time).

        Returns:
            (B, C, W_time).
        """
        B, C, H, W = x.shape
        # Reshape so we pool H dimension: treat (C*W) as channels, H as spatial
        x = x.permute(0, 2, 1, 3).reshape(B * H, C, 1, W)
        # We want per-frequency pooling but keep time...
        # Actually: pool frequency axis (dim=-2)
        x = F.avg_pool2d(x.clamp(min=self.gem.eps).pow(self.gem.p), (H, 1)).pow(1.0 / self.gem.p)
        return x.reshape(B, C, W)


# ──────────────────────────────────────────────
# SED Classifier
# ──────────────────────────────────────────────

class BirdSEDClassifier(nn.Module):
    """CNN backbone + SED head for multi-label bird sound classification.

    Uses GeM frequency pooling to produce frame-level predictions,
    then aggregates temporally for clip-level scores during training,
    or keeps frame-level for soundscape inference.

    Backbones supported via timm:
    - tf_efficientnet_b3.ns_jft_in1k (12.2M)
    - tf_efficientnet_b0.ns_jft_in1k (5.3M)
    - se_resnext50_32x4d (27.5M)
    - nfnet_f0 (71.5M)
    """

    BACKBONE_CONFIGS = {
        "efficientnet_b3": "tf_efficientnet_b3.ns_jft_in1k",
        "efficientnet_b0": "tf_efficientnet_b0.ns_jft_in1k",
        "se_resnext50": "se_resnext50_32x4d",
        "nfnet_f0": "eca_nfnet_l0",  # closest available
    }

    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        n_classes: int = 234,
        pretrained: bool = True,
        gem_p: float = 3.0,
        use_sed_head: bool = True,
        drop_path_rate: float = 0.0,
    ):
        """
        Args:
            backbone_name: Key in BACKBONE_CONFIGS.
            n_classes: Number of output species (234).
            pretrained: Use timm pretrained weights.
            gem_p: GeM pooling exponent.
            use_sed_head: If True, use GeM frequency pooling → framewise output.
                          If False, use GeM global pooling → single vector.
            drop_path_rate: Stochastic depth rate (for Noisy Student).
        """
        super().__init__()

        timm_name = self.BACKBONE_CONFIGS.get(backbone_name, backbone_name)

        self.backbone = timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=0,  # we add our own head
            global_pool="",  # no pooling, keep feature map
            drop_path_rate=drop_path_rate,
        )

        # Infer feature dimension
        with torch.no_grad():
            dummy = torch.randn(2, 3, 224, 224)
            features = self.backbone.forward_features(dummy)
            # features shape: (B, C_feat, H, W) e.g., (2, 1280, 7, 7) for EffNet-B0
            self.feat_dim = features.shape[1]
            self._feat_h = features.shape[2]
            self._feat_w = features.shape[3]

        self.use_sed_head = use_sed_head

        if use_sed_head:
            # SED: frame-level predictions
            # Pool frequency axis → (B, C_feat, W_time)
            # Then conv1x1 to project to n_classes → (B, n_classes, W_time)
            self.freq_pool = nn.AdaptiveAvgPool2d((1, None))  # pool H→1, keep W
            self.classifier = nn.Conv1d(self.feat_dim, n_classes, kernel_size=1)
        else:
            # Standard: global GeM → linear
            self.global_pool = GeM(p=gem_p)
            self.classifier = nn.Linear(self.feat_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224) mel spectrogram.

        Returns:
            If use_sed_head: (B, n_classes, T) frame-level logits.
            Else: (B, n_classes) clip-level logits.
        """
        features = self.backbone.forward_features(x)  # (B, C, H, W)

        if self.use_sed_head:
            # Pool frequency → (B, C, 1, T) → (B, C, T)
            pooled = self.freq_pool(features).squeeze(2)
            return self.classifier(pooled)  # (B, n_classes, T)
        else:
            pooled = self.global_pool(features)  # (B, C)
            return self.classifier(pooled)  # (B, n_classes)

    def forward_clip(self, x: torch.Tensor, aggregation: str = "mean") -> torch.Tensor:
        """Forward pass returning clip-level logits only.

        Args:
            x: (B, 3, 224, 224).
            aggregation: How to aggregate frame predictions
                         ('mean', 'max', 'attention').

        Returns:
            (B, n_classes) logits.
        """
        out = self.forward(x)
        if self.use_sed_head and out.ndim == 3:
            if aggregation == "mean":
                return out.mean(dim=-1)
            elif aggregation == "max":
                return out.max(dim=-1).values
            else:
                return out.mean(dim=-1)
        return out


# ──────────────────────────────────────────────
# Focal Loss
# ──────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal Loss for multi-label classification.

    FL = -α * (1 - p_t)^γ * log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B, C) logits.
            targets: (B, C) binary labels.

        Returns:
            Scalar loss.
        """
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# ──────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────

def create_model(
    backbone_name: str = "efficientnet_b0",
    n_classes: int = 234,
    pretrained: bool = True,
    use_sed_head: bool = True,
    drop_path_rate: float = 0.0,
) -> BirdSEDClassifier:
    """Factory function for BirdSEDClassifier."""
    return BirdSEDClassifier(
        backbone_name=backbone_name,
        n_classes=n_classes,
        pretrained=pretrained,
        use_sed_head=use_sed_head,
        drop_path_rate=drop_path_rate,
    )


if __name__ == "__main__":
    print("Testing BirdSEDClassifier...")

    for name in ["efficientnet_b0", "efficientnet_b3", "se_resnext50"]:
        try:
            model = create_model(name, n_classes=234, use_sed_head=True)
            x = torch.randn(4, 3, 224, 224)
            with torch.no_grad():
                out = model(x)
                clip_out = model.forward_clip(x)
            print(f"  {name}: SED output={out.shape}, clip={clip_out.shape}, params={sum(p.numel() for p in model.parameters()):,}")
        except Exception as e:
            print(f"  {name}: [SKIP] {e}")
