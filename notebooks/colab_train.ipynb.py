# BirdCLEF 2026 — Colab Training Notebook
# ==========================================
# Colle ce script dans une cellule Colab et execute "Run All".
# Checkpoints sauvegardes automatiquement sur Google Drive.
# Si le runtime crash, relance — le script reprend ou le fold suivant.

# ╔══════════════════════════════════════════════════════════════╗
# ║ CELL 1: Setup (5-8 min)                                     ║
# ╚══════════════════════════════════════════════════════════════╝

#@title 1. Mount Drive & Clone Repo
import os, sys, time, json, pickle, shutil
from pathlib import Path

# ── Mount Google Drive ──
from google.colab import drive
drive.mount('/content/drive')

DRIVE_REPO = Path("/content/drive/MyDrive/BirdCLEF2026")
DRIVE_REPO.mkdir(parents=True, exist_ok=True)

# ── Clone repo (if not already) ──
REPO_DIR = Path("/content/__birdclef-v3")
if not REPO_DIR.exists():
    !git clone https://github.com/<USER>/__birdclef-v3.git {REPO_DIR}
    %cd {REPO_DIR}
    !git checkout sprint-1-preprocessing
else:
    %cd {REPO_DIR}
    !git pull

# ── Setup Kaggle API ──
KAGGLE_JSON = REPO_DIR / "kaggle.json"
if KAGGLE_JSON.exists():
    !mkdir -p ~/.kaggle
    !cp {KAGGLE_JSON} ~/.kaggle/kaggle.json
    !chmod 600 ~/.kaggle/kaggle.json

# ── Install deps ──
!pip install -q timm librosa scikit-learn pandas onnx onnxruntime iterative-stratification scipy pyyaml tqdm soundfile

# ── Download competition data (if not already) ──
DATA_DIR = Path("/content/birdclef-2026-data")
if not (DATA_DIR / "train_audio").exists():
    print("Downloading competition data (~16 GB, ~20 min)...")
    !kaggle competitions download -c birdclef-2026 -p /content/birdclef-data
    !unzip -q /content/birdclef-data/birdclef-2026.zip -d {DATA_DIR}

# ── Sync norm stats & folds from Drive (if available) ──
for f in ["norm_stats.pkl", "folds.pkl"]:
    src = DRIVE_REPO / f
    dst = REPO_DIR / "data" / f
    if src.exists() and not dst.exists():
        shutil.copy(src, dst)
        print(f"  Copied {f} from Drive")

print("Setup complete.")

# ╔══════════════════════════════════════════════════════════════╗
# ║ CELL 2: Configure — CHOOSE YOUR JOB HERE                    ║
# ╚══════════════════════════════════════════════════════════════╝

#@title 2. Job Configuration
#@markdown **Choose the model and fold to train.**
BACKBONE = "efficientnet_b3"  #@param ["efficientnet_b0", "efficientnet_b3", "se_resnext50", "nfnet_f0", "rare_b0", "student_b0", "student_vit", "student_mnas"]
FOLD = 3  #@param {type:"slider", min:0, max:4, step:1}
EPOCHS = 80  #@param {type:"slider", min:10, max:80, step:10}
BATCH_SIZE = 64  #@param {type:"slider", min:16, max:128, step:16}
LR = 0.001  #@param ["0.001", "0.0005", "0.0001"]
USE_SED_HEAD = True  #@param {type:"boolean"}
PSEUDO_LABEL_ITER = 0  #@param {type:"slider", min:0, max:4, step:1}  # 0=baseline, 1-4=PL iteration
USE_PRETRAINED = True  #@param {type:"boolean"}

print(f"Job: {BACKBONE} fold {FOLD} | Epochs: {EPOCHS} | BS: {BATCH_SIZE} | LR: {LR}")
print(f"SED head: {USE_SED_HEAD} | PL iter: {PSEUDO_LABEL_ITER}")

# ╔══════════════════════════════════════════════════════════════╗
# ║ CELL 3: Preprocessing (compute norm stats if needed)        ║
# ╚══════════════════════════════════════════════════════════════╝

#@title 3. Preprocessing
sys.path.insert(0, str(REPO_DIR))

import numpy as np
import pandas as pd
from src.features import compute_and_save_norm_stats
from src.dataset import build_label_map

# Compute norm stats if not on Drive
NORM_STATS_PATH = REPO_DIR / "data" / "norm_stats.pkl"
if not NORM_STATS_PATH.exists():
    print("Computing norm stats from training data...")
    train_audio = DATA_DIR / "train_audio"
    all_files = []
    for sp_dir in train_audio.iterdir():
        if sp_dir.is_dir():
            all_files.extend(list(sp_dir.glob("*.ogg")))
    
    compute_and_save_norm_stats(
        [str(f) for f in all_files],
        sr=32000, n_samples=2000, use_pcen=False,
        output_path=str(NORM_STATS_PATH), seed=42,
    )
    # Copy to Drive
    shutil.copy(NORM_STATS_PATH, DRIVE_REPO / "norm_stats.pkl")

# Generate folds if not on Drive
FOLDS_PATH = REPO_DIR / "data" / "folds.pkl"
if not FOLDS_PATH.exists():
    print("Generating GroupKFold splits...")
    from src.validation import create_folds
    create_folds(
        str(DATA_DIR / "train.csv"),
        n_splits=5,
        output_path=str(FOLDS_PATH),
    )
    shutil.copy(FOLDS_PATH, DRIVE_REPO / "folds.pkl")

print("Preprocessing OK")

# ╔══════════════════════════════════════════════════════════════╗
# ║ CELL 4: Segment training audio                              ║
# ╚══════════════════════════════════════════════════════════════╝

#@title 4. Segment Audio (first time only, ~30-60 min)
SEGMENTS_CSV = REPO_DIR / "data" / "train_segments" / "segments.csv"
if not SEGMENTS_CSV.exists():
    print("Segmenting training audio...")
    from src.preprocessing import process_train_audio
    process_train_audio(
        str(DATA_DIR / "train_audio"),
        str(REPO_DIR / "data" / "train_segments"),
        sr=32000, duration=5,
        max_files_per_class=None,  # all files
    )
    print("Segmentation done")
else:
    print(f"Segments already exist: {SEGMENTS_CSV}")

# ╔══════════════════════════════════════════════════════════════╗
# ║ CELL 5: Training                                            ║
# ╚══════════════════════════════════════════════════════════════╝

#@title 5. Train
import torch
from src.train_teachers import train_fold
from src.train_with_pl import train_fold_with_pl, PL_CONFIG

# Determine output path
PL_SUFFIX = f"_pl{PSEUDO_LABEL_ITER}" if PSEUDO_LABEL_ITER > 0 else ""
OUTPUT_DIR = REPO_DIR / "models" / "teachers" / BACKBONE
CHECKPOINT_NAME = f"{BACKBONE}_fold{FOLD}{PL_SUFFIX}.pth"

# Skip if already done
CHECKPOINT_PATH = OUTPUT_DIR / CHECKPOINT_NAME
DRIVE_CHECKPOINT = DRIVE_REPO / "models" / BACKBONE / CHECKPOINT_NAME

if DRIVE_CHECKPOINT.exists():
    print(f"Checkpoint already on Drive: {DRIVE_CHECKPOINT}")
    # Verify it's valid
    try:
        ckpt = torch.load(DRIVE_CHECKPOINT, map_location="cpu", weights_only=False)
        print(f"  Epoch: {ckpt.get('epoch')}, AUC: {ckpt.get('val_auc')}")
        print("[SKIP] Already done")
    except:
        print("[CORRUPT] Will retrain")
else:
    config = {
        "backbone": BACKBONE,
        "fold": FOLD,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": float(LR),
        "n_classes": 234,
        "use_sed_head": USE_SED_HEAD,
        "pretrained": USE_PRETRAINED,
        "train_csv": str(SEGMENTS_CSV),
        "folds_path": str(FOLDS_PATH),
        "norm_stats_path": str(NORM_STATS_PATH),
        "sample_submission": str(DATA_DIR / "sample_submission.csv"),
        "output_dir": str(REPO_DIR / "models" / "teachers"),
        "num_workers": 2,
        "use_amp": True,
        "log_interval": 20,
    }
    
    if PSEUDO_LABEL_ITER > 0:
        # Use PL training
        enriched_csv = REPO_DIR / "data" / "pseudo_labels" / f"train_enriched_iter{PSEUDO_LABEL_ITER}.csv"
        if not enriched_csv.exists():
            enriched_csv = DRIVE_REPO / "data" / "pseudo_labels" / f"train_enriched_iter{PSEUDO_LABEL_ITER}.csv"
        config["output_dir"] = str(REPO_DIR / "models" / "teachers_pl" / f"pl_iter{PSEUDO_LABEL_ITER}")
        config["epochs"] = 40
        config["lr"] = 5e-4
        config["drop_path_rate"] = 0.15
        auc = train_fold_with_pl(FOLD, enriched_csv, config)
    else:
        auc = train_fold(FOLD, config)
    
    print(f"\n[DONE] {BACKBONE} fold {FOLD}: AUC = {auc:.4f}")
    
    # Copy checkpoint to Drive
    DRIVE_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(CHECKPOINT_PATH, DRIVE_CHECKPOINT)
    print(f"[SAVED] Checkpoint → Drive: {DRIVE_CHECKPOINT}")

# ╔══════════════════════════════════════════════════════════════╗
# ║ CELL 6: Notify                                              ║
# ╚══════════════════════════════════════════════════════════════╝

#@title 6. Done!
from datetime import datetime
print(f"\n{'='*50}")
print(f"TRAINING COMPLETE")
print(f"  Model:  {BACKBONE}")
print(f"  Fold:   {FOLD}")
print(f"  PL iter:{PSEUDO_LABEL_ITER}")
print(f"  Time:   {datetime.now()}")
print(f"  Drive:  {DRIVE_CHECKPOINT}")
print(f"{'='*50}")

# Optional: shut down runtime to save GPU quota
# import os; os.system("kill -9 -1")
