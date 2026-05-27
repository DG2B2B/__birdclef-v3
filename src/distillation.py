"""
BirdCLEF 2026 – Knowledge Distillation (Teacher → Student)
Compresses the teacher ensemble into lightweight ONNX-friendly models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List
import gc
import time

from src.models_sed import create_model, BirdSEDClassifier
from src.dataset import BirdCLEFDataset, SpecAugment, build_label_map, load_norm_stats
from src.validation import macro_auc, load_folds


# ──────────────────────────────────────────────
# Distillation Loss
# ──────────────────────────────────────────────

class DistillationLoss(nn.Module):
    """Combined BCE + KL divergence loss for knowledge distillation.

    loss = α * BCE(pred, labels) + (1-α) * T² * KL(softmax(pred/T), softmax(teacher/T))
    """

    def __init__(self, alpha: float = 0.7, T: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.T = T
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student_logits: (B, C).
            teacher_logits: (B, C) aggregated teacher logits (pre-softmax).
            labels: (B, C) binary ground truth.

        Returns:
            Scalar loss.
        """
        # BCE with hard labels
        bce_loss = self.bce(student_logits, labels)

        # KL divergence with soft teacher targets
        T = self.T
        student_soft = F.log_softmax(student_logits / T, dim=1)
        teacher_soft = F.softmax(teacher_logits / T, dim=1)
        kl_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (T * T)

        return self.alpha * bce_loss + (1 - self.alpha) * kl_loss


# ──────────────────────────────────────────────
# Teacher logits generation
# ──────────────────────────────────────────────

def generate_teacher_logits(
    teacher_models: List[BirdSEDClassifier],
    dataset: BirdCLEFDataset,
    batch_size: int = 64,
    device: str = "cuda",
) -> np.ndarray:
    """Pre-compute teacher ensemble logits for the entire dataset.

    Averages logits across all teacher models.

    Args:
        teacher_models: List of trained teacher models (one per fold or architecture).
        dataset: BirdCLEFDataset.
        batch_size: Batch size.
        device: 'cuda' or 'cpu'.

    Returns:
        (N, C) array of averaged teacher logits.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    all_logits = []
    for x, _ in loader:
        x = x.to(device)
        batch_logits = torch.zeros((x.size(0), dataset.n_classes), device=device)

        for model in teacher_models:
            model.eval()
            with torch.no_grad():
                out = model.forward_clip(x)
                batch_logits += out

        batch_logits /= len(teacher_models)
        all_logits.append(batch_logits.cpu().numpy())

    return np.concatenate(all_logits, axis=0)


# ──────────────────────────────────────────────
# Student dataset with teacher logits
# ──────────────────────────────────────────────

class DistillationDataset(torch.utils.data.Dataset):
    """Dataset that returns (x, labels, teacher_logits)."""

    def __init__(self, base_dataset: BirdCLEFDataset, teacher_logits: np.ndarray):
        self.base = base_dataset
        self.teacher_logits = torch.from_numpy(teacher_logits.astype(np.float32))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        t_logits = self.teacher_logits[idx]
        return x, y, t_logits


# ──────────────────────────────────────────────
# Student training
# ──────────────────────────────────────────────

STUDENT_CONFIG = {
    "backbone": "efficientnet_b0",
    "n_classes": 234,
    "pretrained": True,
    "use_sed_head": False,  # students use global pooling (simpler, faster)

    "n_folds": 5,
    "epochs": 40,
    "batch_size": 64,
    "lr": 5e-4,
    "weight_decay": 1e-4,
    "distill_alpha": 0.7,
    "distill_T": 2.0,
    "warmup_epochs": 2,
    "early_stopping_patience": 8,
    "seed": 42,
    "use_amp": True,
    "num_workers": 2,
    "log_interval": 50,
    "sr": 32000,
    "use_pcen": False,
    "target_frames": 224,
    "spec_augment": True,
    "output_dir": "models/students",
    "norm_stats_path": "data/norm_stats.pkl",
    "folds_path": "data/folds.pkl",
    "sample_submission": "C:/Users/Utilisateur/Mon Drive/Kaggle/__birdclef-V2/birdclef-2026-data/sample_submission.csv",
}


def train_student(
    student_backbone: str,
    teacher_model_paths: List[str | Path],
    enriched_csv: str | Path,
    config: Optional[dict] = None,
) -> List[float]:
    """Train a student model with knowledge distillation.

    Args:
        student_backbone: Student architecture name (e.g., 'efficientnet_b0').
        teacher_model_paths: Paths to trained teacher checkpoints.
        enriched_csv: Path to enriched training CSV (original + pseudo-labeled).
        config: Override config dict.

    Returns:
        List of validation AUCs per fold.
    """
    cfg = STUDENT_CONFIG.copy()
    if config:
        cfg.update(config)
    cfg["backbone"] = student_backbone

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load teachers
    from src.models_sed import create_model as create_teacher

    teachers = []
    for tp in teacher_model_paths:
        ckpt = torch.load(tp, map_location=device, weights_only=False)
        t_cfg = ckpt.get("config", {})
        t_model = create_teacher(
            backbone_name=t_cfg.get("backbone", "efficientnet_b0"),
            n_classes=cfg["n_classes"],
            pretrained=False,
            use_sed_head=t_cfg.get("use_sed_head", True),
        )
        t_model.load_state_dict(ckpt["model_state_dict"])
        t_model = t_model.to(device)
        t_model.eval()
        teachers.append(t_model)
    print(f"Loaded {len(teachers)} teacher models")

    # ── Training loop ──
    folds_data = load_folds(cfg["folds_path"])
    folds_df = folds_data["folds"]

    sample_sub = pd.read_csv(cfg["sample_submission"])
    species_list = [c for c in sample_sub.columns if c != "row_id"]
    label_map = build_label_map(species_list=species_list)

    mean, std = load_norm_stats(cfg["norm_stats_path"])
    spec_aug = SpecAugment(p=0.5) if cfg["spec_augment"] else None

    aucs = []

    for fold_idx in range(cfg["n_folds"]):
        print(f"\n{'='*50}")
        print(f"Student {student_backbone} – Fold {fold_idx}")

        train_files = set(folds_df[folds_df["fold"] != fold_idx]["filename"].tolist())
        val_files = set(folds_df[folds_df["fold"] == fold_idx]["filename"].tolist())

        df_all = pd.read_csv(enriched_csv)
        df_all["filename"] = df_all["source_file"].apply(
            lambda p: "/".join(Path(str(p)).parts[-2:]) if pd.notna(p) else ""
        )

        is_orig = df_all.get("is_pseudo", 0) == 0
        train_df = pd.concat([
            df_all[is_orig & df_all["filename"].isin(train_files)],
            df_all[~is_orig],
        ], ignore_index=True)
        val_df = df_all[is_orig & df_all["filename"].isin(val_files)]

        # Datasets
        train_ds = BirdCLEFDataset(
            csv_path=enriched_csv, n_classes=cfg["n_classes"],
            sr=cfg["sr"], mean=mean, std=std, spec_augment=spec_aug,
            label_map=label_map, target_frames=cfg["target_frames"],
        )
        train_ds.df = train_df.reset_index(drop=True)

        val_ds = BirdCLEFDataset(
            csv_path=enriched_csv, n_classes=cfg["n_classes"],
            sr=cfg["sr"], mean=mean, std=std, label_map=label_map,
            target_frames=cfg["target_frames"],
        )
        val_ds.df = val_df.reset_index(drop=True)

        # Generate teacher logits
        print("  Generating teacher logits...")
        teacher_logits_train = generate_teacher_logits(teachers, train_ds, device=str(device))
        teacher_logits_val = generate_teacher_logits(teachers, val_ds, device=str(device))

        # Distillation datasets
        distill_train = DistillationDataset(train_ds, teacher_logits_train)
        distill_val = DistillationDataset(val_ds, teacher_logits_val)

        train_loader = DataLoader(distill_train, batch_size=cfg["batch_size"],
                                  shuffle=True, num_workers=cfg["num_workers"],
                                  pin_memory=True, drop_last=True)
        val_loader = DataLoader(distill_val, batch_size=cfg["batch_size"] * 2,
                                shuffle=False, num_workers=cfg["num_workers"],
                                pin_memory=True)

        # Student model
        student = create_model(
            backbone_name=student_backbone,
            n_classes=cfg["n_classes"],
            pretrained=cfg["pretrained"],
            use_sed_head=cfg["use_sed_head"],
        )
        student = student.to(device)

        criterion = DistillationLoss(alpha=cfg["distill_alpha"], T=cfg["distill_T"])
        optimizer = AdamW(student.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

        steps_per_epoch = len(train_loader)
        total_steps = cfg["epochs"] * steps_per_epoch
        warmup_steps = cfg["warmup_epochs"] * steps_per_epoch
        warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
        main = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, main], milestones=[warmup_steps])

        scaler = torch.cuda.amp.GradScaler() if cfg["use_amp"] and torch.cuda.is_available() else None

        best_auc = 0.0
        patience = 0
        output_dir = Path(cfg["output_dir"]) / student_backbone
        output_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(cfg["epochs"]):
            student.train()
            train_loss = 0.0
            t0 = time.time()

            for x, y, t_logits in train_loader:
                x, y, t_logits = x.to(device), y.to(device), t_logits.to(device)

                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        out = student.forward_clip(x)
                        loss = criterion(out, t_logits, y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    out = student.forward_clip(x)
                    loss = criterion(out, t_logits, y)
                    loss.backward()
                    optimizer.step()

                optimizer.zero_grad()
                scheduler.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validate
            student.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for x, y, _ in val_loader:
                    x = x.to(device)
                    out = student.forward_clip(x)
                    all_preds.append(torch.sigmoid(out).cpu().numpy())
                    all_labels.append(y.numpy())

            val_auc = macro_auc(np.concatenate(all_labels), np.concatenate(all_preds))

            elapsed = time.time() - t0
            print(f"  E{epoch+1} | Loss {train_loss:.4f} | AUC {val_auc:.4f} | {elapsed:.0f}s")

            if val_auc > best_auc:
                best_auc = val_auc
                patience = 0
                torch.save({
                    "epoch": epoch + 1, "model_state_dict": student.state_dict(),
                    "val_auc": val_auc, "config": cfg,
                }, output_dir / f"{student_backbone}_fold{fold_idx}.pth")
            else:
                patience += 1

            if patience >= cfg["early_stopping_patience"]:
                break

        aucs.append(best_auc)
        del student, train_loader, val_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nStudent {student_backbone}: Mean AUC = {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
    return aucs


if __name__ == "__main__":
    print("Distillation module ready.")
    print("  Supported students: efficientnet_b0, efficientvit_b0, mnasnet_100")
    print("  Loss = 0.7*BCE + 0.3*KL(T=2.0)")
