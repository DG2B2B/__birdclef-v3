"""
Hyperparameter Grid Search — Generate notebook variants
=========================================================
Creates multiple EoS notebook variants by varying:
  - Model_3 weight (direct blend)
  - Neighborhood smoothing alpha
  - Species boost strength

Usage:
  python scripts/grid_search_variants.py
  → Creates _push_grid/ folder with notebook variants
  → Push with: for d in _push_grid/*/; do kaggle kernels push -p "$d"; done
"""

import json
import shutil
from pathlib import Path

ROOT = Path(r"C:\Users\Utilisateur\Mon Drive\Kaggle\__birdclef-v3")
BASE_NB = ROOT / "notebooks" / "eos_submit_v8.ipynb"
GRID_DIR = ROOT / "_push_grid"

# Clean grid dir
if GRID_DIR.exists():
    shutil.rmtree(GRID_DIR)
GRID_DIR.mkdir()

# ── Grid parameters ──
MODEL3_WEIGHTS = [0.05, 0.10, 0.15]
SMOOTHING_ALPHAS = [0.10, 0.15, 0.20]
BOOST_STRENGTHS = [0.05, 0.10, 0.15]

# Only vary Model_3 weight for now (most impactful)
# Smoothing and boost are secondary

print(f"Loading base notebook: {BASE_NB.name}")
with open(BASE_NB, "r", encoding="utf-8") as f:
    base_nb = json.load(f)

variant_num = 0

for w3 in MODEL3_WEIGHTS:
    w4 = round(1.0 - w3, 2)
    
    variant_num += 1
    variant_id = f"eos-submit-v6-w{w3}"
    variant_title = f"EoS v6 w3={w3}"
    
    # Deep copy
    nb = json.loads(json.dumps(base_nb))
    
    # Modify solutions dict
    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell["source"])
        if "solutions = {" in src and "'type_add'" in src:
            src = src.replace(
                f"'weight':0.10",
                f"'weight':{w3}"
            )
            src = src.replace(
                f"'weight':0.90",
                f"'weight':{w4}"
            )
            nb["cells"][i]["source"] = [src]
            break
    
    # Create variant folder
    variant_dir = GRID_DIR / variant_id
    variant_dir.mkdir()
    
    # Save notebook
    nb_path = variant_dir / f"{variant_id}.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=True)
    
    # Create kernel metadata
    metadata = {
        "id": f"hellodave2035/{variant_id}",
        "title": variant_title,
        "code_file": f"{variant_id}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [
            "hellodave2035/birdclef-perch-models",
            "hideyukizushi/sgkfk-202604041716",
            "tuckerarrants/bc2026-distilled-sed-public",
            "tuckerarrants/birdclef-2026-waveform-cache",
            "hellodave2035/birdclef-species-boost",
        ],
        "competition_sources": ["birdclef-2026"],
        "kernel_sources": [],
        "model_sources": [
            "google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu/1"
        ],
    }
    meta_path = variant_dir / "kernel-metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  [{variant_num}] {variant_id}: Model_3={w3}, Model_4={w4} → {variant_dir}")

print(f"\nCreated {variant_num} variants in {GRID_DIR}")
print(f"To push all: for d in _push_grid/*/; do kaggle kernels push -p \"$d\"; done")
print(f"Or push individually: kaggle kernels push -p _push_grid/{variant_id}")
