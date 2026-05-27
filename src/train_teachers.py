"""
BirdCLEF 2026 – Teacher Training Script
5-fold GroupKFold training with FocalLoss, AdamW, CosineAnnealing, early stopping.
"""

import os
import gc
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from pathlib import Path
from typing import Optional
import time
from datetime import datetime

from src.models_sed import BirdSEDClassifier, FocalLoss, create_model
from src.dataset import BirdCLEFDataset, SpecAugment, MixUp, build_label_map, load_norm_stats
from src.validation import macro_auc, load_folds

# ──────────────────────────────────────────────
# Config (override with configs/train_teachers.yaml)
# ──────────────────────────────────────────────

CONFIG = {
    # Paths
    "train_csv": "data/train_segments/segments.csv",
    "folds_path": "data/folds.pkl",
    "norm_stats_path": "data/norm_stats.pkl",
    "taxonomy_csv": "C:/Users/Utilisateur/Mon Drive/Kaggle/__birdclef-V2/birdclef-2026-data/taxonomy.csv",
    "output_dir": "models/teachers",
    "sample_submission": "C:/Users/Utilisateur/Mon Drive/Kaggle/__birdclef-V2/birdclef-2026-data/sample_submission.csv",

    # Model
    "backbone": "efficientnet_b0",
    "n_classes": 234,
    "pretrained": True,
    "use_sed_head": True,
    "drop_path_rate": 0.0,

    # Training
    "n_folds": 5,
    "fold": None,  # None = train all folds, int = single fold
    "epochs": 80,
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "focal_alpha": 0.25,
    "focal_gamma": 2.0,
    "warmup_epochs": 4,  # ~5% of 80
    "early_stopping_patience": 10,
    "seed": 42,
    "use_amp": True,  # mixed precision

    # Data
    "sr": 32000,
    "use_pcen": False,
    "target_frames": 224,

    # Augmentation
    "spec_augment": True,
    "mixup": True,
    "mixup_alpha": 0.5,

    # Logging
    "log_interval": 50,
    "save_best_only": True,
    "num_workers": 2,
}


# ──────────────────────────────────────────────
# Training utilities
# ──────────────────────────────────────────────

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def mixup_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply MixUp to a batch."""
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    lam = alpha  # fixed blending

    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y


# ──────────────────────────────────────────────
# Train one fold
# ──────────────────────────────────────────────

def train_fold(
    fold_idx: int,
    config: dict,
) -> float:
    """Train a single fold. Returns best validation macro-AUC."""
    device = get_device()
    set_seed(config["seed"])

    output_dir = Path(config["output_dir"]) / config["backbone"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    folds_data = load_folds(config["folds_path"])
    folds_df = folds_data["folds"]

    # Build label map
    sample_sub = pd.read_csv(config["sample_submission"])
    species_list = [c for c in sample_sub.columns if c != "row_id"]
    label_map = build_label_map(species_list=species_list)

    # Invert: filename → label
    train_meta = pd.read_csv("C:/Users/Utilisateur/Mon Drive/Kaggle/__birdclef-V2/birdclef-2026-data/train.csv")
    filename_to_label = dict(zip(train_meta["filename"], train_meta["primary_label"]))

    # Build split DataFrames
    train_files = folds_df[folds_df["fold"] != fold_idx]["filename"].tolist()
    val_files = folds_df[folds_df["fold"] == fold_idx]["filename"].tolist()

    # Convert filenames to .npy paths
    segments_csv = pd.read_csv(config["train_csv"])
    segments_csv["filename"] = segments_csv["source_file"].apply(
        lambda p: "/".join(Path(p).parts[-2:])
    )

    train_df = segments_csv[segments_csv["filename"].isin(train_files)].copy()
    val_df = segments_csv[segments_csv["filename"].isin(val_files)].copy()

    print(f"\n{'='*50}")
    print(f"Fold {fold_idx}: train={len(train_df)}, val={len(val_df)}")
    print(f"  Species in train: {train_df['primary_label'].nunique()}")
    print(f"  Species in val:   {val_df['primary_label'].nunique()}")

    # ── Load norm stats ──
    mean, std = None, None
    if Path(config["norm_stats_path"]).exists():
        mean, std = load_norm_stats(config["norm_stats_path"])

    # ── Augmentations ──
    spec_aug = SpecAugment(p=0.5) if config["spec_augment"] else None

    # ── Datasets ──
    train_ds = BirdCLEFDataset(
        csv_path=...,
        n_classes=config["n_classes"],
        sr=config["sr"],
        use_pcen=config["use_pcen"],
        mean=mean,
        std=std,
        spec_augment=spec_aug,
        label_map=label_map,
        target_frames=config["target_frames"],
    )
    train_ds.df = train_df.reset_index(drop=True)

    val_ds = BirdCLEFDataset(
        csv_path=...,
        n_classes=config["n_classes"],
        sr=config["sr"],
        use_pcen=config["use_pcen"],
        mean=mean,
        std=std,
        label_map=label_map,
        target_frames=config["target_frames"],
    )
    val_ds.df = val_df.reset_index(drop=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"] * 2,
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    # ── Model ──
    model = create_model(
        backbone_name=config["backbone"],
        n_classes=config["n_classes"],
        pretrained=config["pretrained"],
        use_sed_head=config["use_sed_head"],
        drop_path_rate=config["drop_path_rate"],
    )
    model = model.to(device)

    # ── Loss & optimizer ──
    criterion = FocalLoss(alpha=config["focal_alpha"], gamma=config["focal_gamma"])
    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    steps_per_epoch = len(train_loader)
    total_steps = config["epochs"] * steps_per_epoch
    warmup_steps = config["warmup_epochs"] * steps_per_epoch

    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
    )
    main_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_steps]
    )

    scaler = torch.cuda.amp.GradScaler() if config["use_amp"] and torch.cuda.is_available() else None

    # ── Training loop ──
    best_val_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "val_auc": [], "lr": []}

    for epoch in range(config["epochs"]):
        # ── Train ──
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            # MixUp
            if config["mixup"]:
                x, y = mixup_batch(x, y, alpha=config["mixup_alpha"])

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    out = model.forward_clip(x)
                    loss = criterion(out, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model.forward_clip(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

            optimizer.zero_grad()
            scheduler.step()

            train_loss += loss.item()

            if batch_idx % config["log_interval"] == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch+1}/{config['epochs']} | Batch {batch_idx}/{steps_per_epoch} | Loss {loss.item():.4f} | LR {current_lr:.2e}")

        train_loss /= len(train_loader)
        history["train_loss"].append(train_loss)

        # ── Validate ──
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model.forward_clip(x)
                all_preds.append(torch.sigmoid(out).cpu().numpy())
                all_labels.append(y.cpu().numpy())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)

        val_auc = macro_auc(y_true, y_pred)
        history["val_auc"].append(val_auc)
        history["lr"].append(scheduler.get_last_lr()[0])

        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1} | Train Loss {train_loss:.4f} | Val AUC {val_auc:.4f} | Time {elapsed:.0f}s | LR {scheduler.get_last_lr()[0]:.2e}")

        # ── Checkpoint ──
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            patience_counter = 0

            checkpoint_path = output_dir / f"{config['backbone']}_fold{fold_idx}.pth"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auc": val_auc,
                    "config": config,
                },
                checkpoint_path,
            )
            print(f"  [SAVED] {checkpoint_path} (AUC={val_auc:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config["early_stopping_patience"]:
            print(f"  [STOP] Early stopping at epoch {epoch+1} (best: {best_val_auc:.4f} at epoch {best_epoch})")
            break

    # ── Cleanup ──
    del model, optimizer, scheduler, train_loader, val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_val_auc


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main(config: Optional[dict] = None):
    """Run full 5-fold training or single fold."""
    cfg = CONFIG.copy()
    if config:
        cfg.update(config)

    print(f"[{datetime.now()}] Starting teacher training")
    print(f"  Backbone: {cfg['backbone']}")
    print(f"  Classes:  {cfg['n_classes']}")
    print(f"  Epochs:   {cfg['epochs']}")
    print(f"  Device:   {get_device()}")

    folds_to_run = [cfg["fold"]] if cfg["fold"] is not None else range(cfg["n_folds"])
    aucs = []

    for fold_idx in folds_to_run:
        auc = train_fold(fold_idx, cfg)
        aucs.append(auc)

    if len(aucs) > 1:
        print(f"\n{'='*50}")
        print(f"Cross-validation results ({cfg['backbone']}):")
        for i, auc in enumerate(aucs):
            print(f"  Fold {i}: {auc:.4f}")
        print(f"  Mean:    {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")

    return aucs


if __name__ == "__main__":
    main()
