"""
BirdCLEF 2026 – Kaggle Dataset Preparation Script
==================================================
Prépare le dataset à uploader sur Kaggle pour la soumission Perch.

Ce script est exécuté LOCALEMENT (ou sur Colab avec accès internet).
Il télécharge les artefacts nécessaires et crée l'arborescence
prête à être uploadée comme dataset Kaggle.

Usage:
    python notebooks/prepare_kaggle_dataset.py

Output:
    kaggle_dataset/
    ├── perch_v2.onnx           # Depuis rishikeshjani/perch-onnx-for-birdclef-2026
    ├── labels.csv              # Mapping Perch → espèces
    ├── classifier_ssm.onnx     # Votre classifieur entraîné
    ├── config.pkl              # Paramètres du classifieur
    ├── weights.json            # Poids d'ensemble
    └── calibrators.pkl         # Optionnel
"""

import os
import sys
import shutil
import json
import pickle
import urllib.request
from pathlib import Path
import zipfile
import subprocess

# ── Configuration ──
OUTPUT_DIR = Path("kaggle_dataset")

# Sources
PERCH_DATASET = "rishikeshjani/perch-onnx-for-birdclef-2026"  # Kaggle dataset slug
MODELS_DIR = Path("models")  # Where trained models are stored

# Fichiers à inclure
REQUIRED_FILES = [
    "perch_v2.onnx",
    "labels.csv",
]

OPTIONAL_FILES = [
    "classifier_ssm.onnx",
    "config.pkl",
    "weights.json",
    "calibrators.pkl",
]


def create_dataset_dir():
    """Create output directory structure."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Output directory: {OUTPUT_DIR.absolute()}")


def download_kaggle_dataset(dataset_slug: str):
    """Download a Kaggle dataset using kagglehub or kaggle CLI."""
    print(f"\n[INFO] Downloading dataset: {dataset_slug}")

    # Try kagglehub first
    try:
        import kagglehub
        path = kagglehub.dataset_download(dataset_slug)
        print(f"  Downloaded to: {path}")
        return Path(path)
    except ImportError:
        pass

    # Fallback: kaggle CLI
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_slug, "-p", str(OUTPUT_DIR)],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            # Unzip
            zip_path = OUTPUT_DIR / f"{dataset_slug.split('/')[-1]}.zip"
            if zip_path.exists():
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(OUTPUT_DIR)
                zip_path.unlink()
            return OUTPUT_DIR
        else:
            print(f"  [ERROR] kaggle CLI failed: {result.stderr}")
    except FileNotFoundError:
        print("  [ERROR] kaggle CLI not found. Install with: pip install kaggle")
    except Exception as e:
        print(f"  [ERROR] {e}")

    return None


def copy_from_models():
    """Copy trained model artifacts to the dataset directory."""
    if not MODELS_DIR.exists():
        print(f"\n[WARN] Models directory not found: {MODELS_DIR}")
        return

    for fname in OPTIONAL_FILES:
        src = MODELS_DIR / fname
        if src.exists():
            dst = OUTPUT_DIR / fname
            shutil.copy2(src, dst)
            print(f"  [OK] Copied {fname} ({src.stat().st_size / 1e6:.1f} MB)")
        else:
            print(f"  [MISS] {fname} not found (will skip)")


def verify():
    """Verify dataset integrity."""
    print(f"\n{'=' * 60}")
    print("Verification")
    print(f"{'=' * 60}")

    total_size = 0
    missing = []

    for fname in REQUIRED_FILES + OPTIONAL_FILES:
        fpath = OUTPUT_DIR / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / 1e6
            total_size += size_mb
            print(f"  ✅ {fname} ({size_mb:.1f} MB)")
        else:
            if fname in REQUIRED_FILES:
                print(f"  ❌ {fname} (REQUIRED — MISSING!)")
                missing.append(fname)
            else:
                print(f"  ⬜ {fname} (optional — skipped)")

    print(f"\n  Total size: {total_size:.1f} MB")
    if total_size > 20000:
        print(f"  ⚠️  WARNING: Dataset > 20 GB Kaggle limit!")
    elif total_size > 1000:
        print(f"  ⚠️  Dataset > 1 GB — consider compression")
    else:
        print(f"  ✅ Size OK for Kaggle upload")

    if missing:
        print(f"\n  ❌ MISSING REQUIRED FILES: {missing}")
        print(f"  Run: kagglehub dataset download {PERCH_DATASET}")
        return False

    return True


def create_dataset_metadata():
    """Create dataset-metadata.json for Kaggle upload."""
    metadata = {
        "title": "BirdCLEF 2026 — Perch Models",
        "id": "birdclef-perch-models",
        "licenses": [{"name": "CC0-1.0"}],
    }
    meta_path = OUTPUT_DIR / "dataset-metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  [OK] Created dataset-metadata.json")


def print_upload_instructions():
    """Print instructions for uploading to Kaggle."""
    print(f"\n{'=' * 60}")
    print("UPLOAD INSTRUCTIONS")
    print(f"{'=' * 60}")
    print(f"""
1. Go to: https://www.kaggle.com/datasets
2. Click "New Dataset"
3. Upload the contents of: {OUTPUT_DIR.absolute()}
4. Set title: "BirdCLEF 2026 — Perch Models"
5. Set license: CC0 (Public Domain)
6. Click "Create"

OR use Kaggle CLI:
    kaggle datasets create -p {OUTPUT_DIR} --dir-mode skip

7. In your submission notebook, add this dataset as input.
    """)


def main():
    print("=" * 60)
    print("BirdCLEF 2026 — Kaggle Dataset Preparation")
    print("=" * 60)

    create_dataset_dir()

    # Download Perch ONNX
    perch_path = download_kaggle_dataset(PERCH_DATASET)
    if perch_path:
        for fname in REQUIRED_FILES:
            src = perch_path / fname
            if src.exists():
                dst = OUTPUT_DIR / fname
                if not dst.exists():
                    shutil.copy2(src, dst)
                    print(f"  [OK] Copied {fname} from Perch dataset")
            else:
                # Search recursively
                for found in perch_path.rglob(fname):
                    dst = OUTPUT_DIR / fname
                    if not dst.exists():
                        shutil.copy2(found, dst)
                        print(f"  [OK] Copied {fname} from {found}")

    # Copy trained models
    copy_from_models()

    # Create metadata
    create_dataset_metadata()

    # Verify
    ok = verify()

    if ok:
        print_upload_instructions()
        print("\n[DONE] Dataset ready for Kaggle upload!")
    else:
        print("\n[FAIL] Missing required files. Fix before uploading.")


if __name__ == "__main__":
    main()
