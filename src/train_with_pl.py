"""
BirdCLEF 2026 – Training with Pseudo-Labels (Noisy Student)
Re-trains teacher models on original + pseudo-labeled data.
Supports Stochastic Depth, MixUp 1:1 blending, and iterative fine-tuning.
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

from src.models_sed import create_model, FocalLoss
from src.dataset import (
    BirdCLEFDataset, SpecAugment, build_label_map, load_norm_stats,
)
from src.validation import macro_auc, load_folds


# ──────────────────────────────────────────────
# Default config for pseudo-label training
# ──────────────────────────────────────────────

PL_CONFIG = {
    "backbone": "efficientnet_b0",
    "n_classes": 234,
    "pretrained": True,
    "use_sed_head": True,
    "drop_path_rate": 0.15,  # Noisy Student regularization

    "n_folds": 5,
    "fold": None,
    "epochs": 40,            # fewer epochs for re-training
    "batch_size": 64,
    "lr": 5e-4,              # lower LR for fine-tuning
    "weight_decay": 1e-4,
    "focal_alpha": 0.25,
    "focal_gamma": 2.0,
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
    "mixup": True,
    "mixup_alpha": 0.5,      # fixed blending (1st place 2025)

    # Paths
    "norm_stats_path": "data/norm_stats.pkl",
    "folds_path": "data/folds.pkl",
    "sample_submission": "C:/Users/Utilisateur/Mon Drive/Kaggle/__birdclef-V2/birdclef-2026-data/sample_submission.csv",
    "output_dir": "models/teachers_pl",
    "pretrained_weights": None,  # path to base model weights for fine-tuning
}


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mixup_batch(x, y, alpha=0.5):
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    lam = alpha
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y


def train_fold_with_pl(
    fold_idx: int,
    enriched_csv: str | Path,
    config: dict,
) -> float:
    """Train one fold on enriched dataset."""
    device = get_device()
    set_seed(config["seed"])

    output_dir = Path(config["output_dir"]) / config["backbone"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    folds_data = load_folds(config["folds_path"])
    folds_df = folds_data["folds"]
    train_files = set(folds_df[folds_df["fold"] != fold_idx]["filename"].tolist())
    val_files = set(folds_df[folds_df["fold"] == fold_idx]["filename"].tolist())

    # Load enriched CSV and split
    df_all = pd.read_csv(enriched_csv)
    df_all["filename"] = df_all["source_file"].apply(
        lambda p: "/".join(Path(str(p)).parts[-2:]) if pd.notna(p) else ""
    )

    # For original data: use fold split
    # For pseudo-labeled data: use all as training
    is_original = df_all.get("is_pseudo", 0) == 0

    train_orig = df_all[is_original & df_all["filename"].isin(train_files)]
    val_orig = df_all[is_original & df_all["filename"].isin(val_files)]
    train_pseudo = df_all[~is_original]

    train_df = pd.concat([train_orig, train_pseudo], ignore_index=True)
    val_df = val_orig

    print(f"\nFold {fold_idx}:")
    print(f"  Train: {len(train_df)} ({len(train_orig)} original + {len(train_pseudo)} pseudo)")
    print(f"  Val:   {len(val_df)}")

    # ── Norm stats ──
    mean, std = None, None
    if Path(config["norm_stats_path"]).exists():
        mean, std = load_norm_stats(config["norm_stats_path"])

    # ── Label map ──
    sample_sub = pd.read_csv(config["sample_submission"])
    species_list = [c for c in sample_sub.columns if c != "row_id"]
    label_map = build_label_map(species_list=species_list)

    # ── Augmentations ──
    spec_aug = SpecAugment(p=0.5) if config["spec_augment"] else None

    # ── Datasets ──
    train_ds = BirdCLEFDataset(
        csv_path=enriched_csv, n_classes=config["n_classes"],
        sr=config["sr"], use_pcen=config["use_pcen"],
        mean=mean, std=std, spec_augment=spec_aug,
        label_map=label_map, target_frames=config["target_frames"],
    )
    train_ds.df = train_df.reset_index(drop=True)

    val_ds = BirdCLEFDataset(
        csv_path=enriched_csv, n_classes=config["n_classes"],
        sr=config["sr"], use_pcen=config["use_pcen"],
        mean=mean, std=std, label_map=label_map,
        target_frames=config["target_frames"],
    )
    val_ds.df = val_df.reset_index(drop=True)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"],
                              shuffle=True, num_workers=config["num_workers"],
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"] * 2,
                            shuffle=False, num_workers=config["num_workers"],
                            pin_memory=True)

    # ── Model ──
    model = create_model(
        backbone_name=config["backbone"],
        n_classes=config["n_classes"],
        pretrained=config["pretrained"],
        use_sed_head=config["use_sed_head"],
        drop_path_rate=config["drop_path_rate"],
    )

    # Load pretrained weights if provided
    if config.get("pretrained_weights") and Path(config["pretrained_weights"]).exists():
        ckpt = torch.load(config["pretrained_weights"], map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"  Loaded pretrained weights from {config['pretrained_weights']}")

    model = model.to(device)

    # ── Loss & optimizer ──
    criterion = FocalLoss(alpha=config["focal_alpha"], gamma=config["focal_gamma"])
    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    steps_per_epoch = len(train_loader)
    total_steps = config["epochs"] * steps_per_epoch
    warmup_steps = config["warmup_epochs"] * steps_per_epoch

    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    main = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, main], milestones=[warmup_steps])

    scaler = torch.cuda.amp.GradScaler() if config["use_amp"] and torch.cuda.is_available() else None

    # ── Training ──
    best_val_auc = 0.0
    patience_counter = 0

    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

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
                print(f"  E{epoch+1}/{config['epochs']} B{batch_idx}/{steps_per_epoch} | Loss {loss.item():.4f}")

        train_loss /= len(train_loader)

        # ── Validate ──
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model.forward_clip(x)
                all_preds.append(torch.sigmoid(out).cpu().numpy())
                all_labels.append(y.cpu().numpy())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)
        val_auc = macro_auc(y_true, y_pred)

        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1} | Loss {train_loss:.4f} | Val AUC {val_auc:.4f} | {elapsed:.0f}s")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            ckpt_path = output_dir / f"{config['backbone']}_fold{fold_idx}_pl.pth"
            torch.save({
                "epoch": epoch + 1, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": val_auc, "config": config,
            }, ckpt_path)
            print(f"  [SAVED] {ckpt_path} (AUC={val_auc:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= config["early_stopping_patience"]:
            print(f"  [STOP] Best AUC={best_val_auc:.4f}")
            break

    del model, optimizer, scheduler, train_loader, val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_val_auc


def main():
    print(f"[{datetime.now()}] Noisy Student training with pseudo-labels")
    # This is meant to be called from a Kaggle notebook with the enriched CSV
    print("Usage: see notebooks/train_pl.py or call train_fold_with_pl()")


if __name__ == "__main__":
    main()
