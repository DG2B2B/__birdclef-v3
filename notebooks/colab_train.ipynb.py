# BirdCLEF 2026 — Colab Training Notebook v2
# ============================================
# AVANT DE LANCER :
#   1. Upload kaggle.json dans Colab (icone dossier a gauche → upload)
#   2. Upload le repo entier en .zip dans Colab
#   3. Change BACKBONE et FOLD ci-dessous → Run All

# ╔══════════════════════════════════════════════════════════════╗
# ║ CELL 1: Config                                              ║
# ╚══════════════════════════════════════════════════════════════╝

BACKBONE = "efficientnet_b0"   # efficientnet_b0, efficientnet_b3
FOLD = 0                        # 0 a 4
EPOCHS = 80
BATCH_SIZE = 64
LR = 0.001
USE_SED_HEAD = True
PSEUDO_LABEL_ITER = 0           # 0 = baseline
USE_PRETRAINED = True

REPO_ZIP = "birdclef-v3.zip"    # nom du zip que tu as upload dans Colab

print(f"Job: {BACKBONE} fold {FOLD} | {EPOCHS} epochs | BS={BATCH_SIZE}")

# ╔══════════════════════════════════════════════════════════════╗
# ║ CELL 2: Setup                                               ║
# ╚══════════════════════════════════════════════════════════════╝

import os, sys, time, json, pickle, shutil, zipfile
from pathlib import Path

# ── Mount Drive (optionnel, pour sauvegarde checkpoints) ──
try:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_DIR = Path("/content/drive/MyDrive/BirdCLEF2026")
    DRIVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Drive mounted: {DRIVE_DIR}")
except:
    DRIVE_DIR = None
    print("Drive not mounted (checkpoints will be lost on runtime reset!)")

# ── Install deps ──
!pip install -q timm librosa scikit-learn pandas onnx onnxruntime scipy pyyaml tqdm soundfile

# ── Unzip repo ──
REPO_DIR = Path("/content/__birdclef-v3")
if not REPO_DIR.exists():
    print(f"Unzipping {REPO_ZIP}...")
    with zipfile.ZipFile(REPO_ZIP, 'r') as zf:
        zf.extractall(REPO_DIR)
    print("Repo extracted")

%cd {REPO_DIR}
sys.path.insert(0, str(REPO_DIR))

# ── Setup Kaggle API ──
KAGGLE_JSON = Path("/content/kaggle.json")
if KAGGLE_JSON.exists():
    !mkdir -p ~/.kaggle
    !cp /content/kaggle.json ~/.kaggle/kaggle.json
    !chmod 600 ~/.kaggle/kaggle.json
    print("Kaggle API configured")
else:
    print("WARNING: kaggle.json not found — upload it in Colab sidebar")

# ── Download data ──
DATA_DIR = Path("/content/birdclef-data")
if not (DATA_DIR / "train_audio").exists():
    print("Downloading competition data (15 min)...")
    !kaggle competitions download -c birdclef-2026 -p /content/birdclef-data
    !unzip -q -o /content/birdclef-data/birdclef-2026.zip -d {DATA_DIR}

# ── Sync norm stats & folds from Drive ──
os.makedirs(REPO_DIR / "data", exist_ok=True)
if DRIVE_DIR:
    for f in ["norm_stats.pkl", "folds.pkl"]:
        src = DRIVE_DIR / f
        dst = REPO_DIR / "data" / f
        if src.exists():
            shutil.copy(src, dst)
            print(f"  Loaded {f} from Drive")

print("\nSetup complete. Starting training...")

# ╔══════════════════════════════════════════════════════════════╗
# ║ CELL 3: Preprocessing (si necessaire)                       ║
# ╚══════════════════════════════════════════════════════════════╝

import numpy as np
import pandas as pd
import torch
from src.features import compute_and_save_norm_stats
from src.validation import create_folds

NORM_STATS_PATH = REPO_DIR / "data" / "norm_stats.pkl"
FOLDS_PATH = REPO_DIR / "data" / "folds.pkl"

if not NORM_STATS_PATH.exists():
    print("Computing norm stats (2-3 min)...")
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
    if DRIVE_DIR:
        shutil.copy(NORM_STATS_PATH, DRIVE_DIR / "norm_stats.pkl")

if not FOLDS_PATH.exists():
    print("Generating GroupKFold splits...")
    create_folds(str(DATA_DIR / "train.csv"), n_splits=5, output_path=str(FOLDS_PATH))
    if DRIVE_DIR:
        shutil.copy(FOLDS_PATH, DRIVE_DIR / "folds.pkl")

print("Preprocessing OK")

# ╔══════════════════════════════════════════════════════════════╗
# ║ CELL 4: Segment train audio (skip si deja fait)             ║
# ╚══════════════════════════════════════════════════════════════╝

SEGMENTS_CSV = REPO_DIR / "data" / "train_segments" / "segments.csv"
if not SEGMENTS_CSV.exists():
    print("Segmenting training audio (~20-40 min, first time only)...")
    from src.preprocessing import process_train_audio
    process_train_audio(
        str(DATA_DIR / "train_audio"),
        str(REPO_DIR / "data" / "train_segments"),
        sr=32000, duration=5,
    )
else:
    print(f"Segments already exist ({SEGMENTS_CSV.stat().st_size/1e6:.1f} MB)")

# ╔══════════════════════════════════════════════════════════════╗
# ║ CELL 5: Training                                            ║
# ╚══════════════════════════════════════════════════════════════╝

from src.train_teachers import train_fold

OUTPUT_DIR = REPO_DIR / "models" / "teachers" / BACKBONE
CHECKPOINT_NAME = f"{BACKBONE}_fold{FOLD}.pth"
CHECKPOINT_PATH = OUTPUT_DIR / CHECKPOINT_NAME

# Check if already done (on Drive)
if DRIVE_DIR:
    drive_ckpt = DRIVE_DIR / "models" / BACKBONE / CHECKPOINT_NAME
    if drive_ckpt.exists():
        try:
            ckpt = torch.load(drive_ckpt, map_location="cpu", weights_only=False)
            print(f"[SKIP] Already done: epoch={ckpt.get('epoch')}, AUC={ckpt.get('val_auc'):.4f}")
            print("Change FOLD to train next fold.")
        except:
            print("[CORRUPT] Retraining...")
            drive_ckpt = None
    else:
        drive_ckpt = None
else:
    drive_ckpt = None

if drive_ckpt is None or not drive_ckpt.exists():
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
    
    auc = train_fold(FOLD, config)
    print(f"\n[DONE] {BACKBONE} fold {FOLD} — AUC = {auc:.4f}")
    
    # Copy to Drive
    if DRIVE_DIR:
        drive_ckpt_path = DRIVE_DIR / "models" / BACKBONE / CHECKPOINT_NAME
        drive_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(CHECKPOINT_PATH, drive_ckpt_path)
        print(f"[SAVED] → Drive: {drive_ckpt_path}")

# ╔══════════════════════════════════════════════════════════════╗
# ║ CELL 6: Done                                                ║
# ╚══════════════════════════════════════════════════════════════╝

from datetime import datetime
print(f"\n{'='*50}")
print(f"DONE — {BACKBONE} fold {FOLD}")
print(f"Time: {datetime.now()}")
if DRIVE_DIR:
    print(f"Checkpoint: {DRIVE_DIR}/models/{BACKBONE}/{CHECKPOINT_NAME}")
print(f"{'='*50}")
print("\nNext: change FOLD to {0} and Run All again".format(FOLD + 1))
