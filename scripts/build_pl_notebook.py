"""
Build Pseudo-Label Training Notebook
=====================================
Extracts the Model_4 training cell from EoS V2, modifies it for PL training,
packages as a standalone Kaggle GPU notebook.

Key modifications:
  1. MODE = "train" (was "submit")
  2. Load pseudo-labels from Kaggle dataset
  3. Merge with original soundscape labels
  4. Export trained model as Kaggle dataset
"""
import json
import copy
from pathlib import Path

ROOT = Path(r"C:\Users\Utilisateur\Mon Drive\Kaggle\__birdclef-v3")
SRC_NB = ROOT / "notebooks" / "eos_submit_v2.ipynb"
DST_NB = ROOT / "notebooks" / "train_pl_v1.ipynb"

print(f"Loading {SRC_NB.name}...")
with open(SRC_NB, "r", encoding="utf-8") as f:
    src_nb = json.load(f)

# ── Extract Model_4 cell (Cell 9) ──
model4_cell = None
for i, cell in enumerate(src_nb["cells"]):
    src = "".join(cell["source"])
    if src.strip().startswith("if 'Model_4' in _ensemble_models:"):
        model4_cell = cell
        print(f"Found Model_4 cell at index {i}, length={len(src)} chars")
        break

if model4_cell is None:
    raise ValueError("Model_4 cell not found!")

model4_src = "".join(model4_cell["source"])

# ── Modifications ──
print("Applying modifications...")

# 1. Remove the outer if-block
model4_src = model4_src.replace(
    "if 'Model_4' in _ensemble_models:",
    "# V9 PL Training — Model_4 cell adapted for pseudo-label retraining"
)
# Remove the first indentation level (4 spaces) from all lines
lines = model4_src.split("\n")
dedented_lines = []
for line in lines:
    if line.startswith("    "):
        dedented_lines.append(line[4:])
    else:
        dedented_lines.append(line)
model4_src = "\n".join(dedented_lines)

# 2. Switch to train mode
model4_src = model4_src.replace(
    'MODE = "submit"',
    'MODE = "train"  # V9 PL: training mode for pseudo-label retraining'
)

# 3. Change dryrun to include all files for serious training
model4_src = model4_src.replace(
    '"dryrun_n_files": 20 if MODE == "train" else 0',
    '"dryrun_n_files": 0  # V9 PL: no dry-run, train on all data'
)

# 4. Add pseudo-label loading right after Y_SC is built
# Find the Y_SC creation and add PL merging after it
y_sc_marker = "Y_FULL = Y_SC[full_rows[\"index\"].to_numpy()]"
pl_injection = '''Y_FULL = Y_SC[full_rows["index"].to_numpy()]

# ═══════════════════════════════════════════════════════════
# V9 PL: Load and merge pseudo-labels
# ═══════════════════════════════════════════════════════════
PL_CSV = Path("/kaggle/input/birdclef-pl-v1/pseudo_labels_filtered.csv")
if PL_CSV.exists():
    print("Loading pseudo-labels...")
    pl_df = pd.read_csv(PL_CSV)
    pl_species = [c for c in pl_df.columns if c != "row_id"]
    
    # Align pseudo-labels with sc (soundscape segments)
    pl_dict = {}
    for _, row in pl_df.iterrows():
        pl_dict[row["row_id"]] = row[pl_species].values
    
    # Add pseudo-labeled rows to sc
    pl_rows = []
    for rid, probs in pl_dict.items():
        # Parse row_id: {soundscape}_{end_sec}
        parts = str(rid).rsplit("_", 1)
        if len(parts) == 2:
            fname = parts[0] + ".ogg"
            end_sec = int(parts[1])
            pl_rows.append({
                "filename": fname,
                "start": f"00:{max(0,end_sec-5):02d}:00",
                "end": f"00:{end_sec:02d}:00",
                "label_list": [],  # will be filled from probs
                "end_sec": end_sec,
                "row_id": rid,
                "site": "pl",
                "hour_utc": -1,
            })
    
    if pl_rows:
        pl_sc = pd.DataFrame(pl_rows)
        # Mark pseudo-label rows
        pl_sc["is_pseudo"] = True
        sc["is_pseudo"] = False
        sc = pd.concat([sc, pl_sc], ignore_index=True)
        
        # Extend Y_SC with pseudo-label probabilities
        pl_y = np.zeros((len(pl_rows), N_CLASSES), dtype=np.float32)
        for i, rid in enumerate([r["row_id"] for r in pl_rows]):
            if rid in pl_dict:
                pl_y[i] = pl_dict[rid]
        Y_SC = np.vstack([Y_SC, pl_y.astype(np.uint8)])  # binarize for training
        
        print(f"Added {len(pl_rows)} pseudo-labeled segments to training data")
        print(f"Total training segments: {len(sc)} (original + pseudo)")
else:
    print("No pseudo-labels found — training on original data only")'''

if y_sc_marker in model4_src:
    model4_src = model4_src.replace(y_sc_marker, pl_injection)
    print("  Injected PL loading after Y_FULL creation")
else:
    print("  WARNING: Could not find Y_FULL marker — PL injection may need manual review")

# 5. Add model export cell at the very end
export_cell_code = '''
# ═══════════════════════════════════════════════════════════
# V9 PL: Export trained model
# ═══════════════════════════════════════════════════════════
import torch
import json

EXPORT_DIR = Path("/kaggle/working/pl_model_export")
EXPORT_DIR.mkdir(exist_ok=True)

# Save ProtoSSM state dict
if "model" in dir() and model is not None:
    torch.save(model.state_dict(), EXPORT_DIR / "proto_ssm_pl.pt")
    print(f"Saved ProtoSSM to {EXPORT_DIR / 'proto_ssm_pl.pt'}")

# Save ResidualSSM state dict
if "residual_model" in dir():
    torch.save(residual_model.state_dict(), EXPORT_DIR / "residual_ssm_pl.pt")
    print(f"Saved ResidualSSM to {EXPORT_DIR / 'residual_ssm_pl.pt'}")

# Save training metadata
meta = {
    "n_pseudo_segments": len(pl_rows) if "pl_rows" in dir() else 0,
    "mode": "train_pl_v1",
    "training_date": "2026-06-01",
}
with open(EXPORT_DIR / "training_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print("Training metadata saved")
print(f"\\nModel export complete. Upload {EXPORT_DIR} as Kaggle dataset.")
'''

# ── Build the new notebook ──
print("Building notebook...")

# Clean encoding
def clean(s):
    for old, new in [("\u2018","'"),("\u2019","'"),("\u201c",'"'),("\u201d",'"'),("\u2013","--"),("\u2014","--"),("\u00a0"," ")]:
        s = s.replace(old, new)
    return s.encode("ascii", errors="replace").decode("ascii")

model4_src = clean(model4_src)
export_cell_code = clean(export_cell_code)

# Create cells
cells = []

# Cell 0: Config + info
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["# BirdCLEF 2026 — Pseudo-Label Training v1\n\n"
               "Retrains ProtoSSM + ResidualSSM on original soundscape labels + high-confidence pseudo-labels.\n\n"
               "**Required Kaggle datasets:**\n"
               "- `birdclef-2026` (competition)\n"
               "- `hellodave2035/birdclef-perch-models` (ONNX Runtime wheel)\n"
               "- `hellodave2035/birdclef-pl-v1` (pseudo-labels CSV)\n"
               "- `hideyukizushi/sgkfk-202604041716` (pre-trained weights for init)\n"
               "- `google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu/1` (Perch TF model)\n\n"
               "**Output:** Trained model checkpoints to upload as Kaggle dataset.\n"
               "**Runtime:** ~5-8h on GPU T4 x2.\n"]
})

# Cell 1: Install deps
cells.append({
    "cell_type": "code",
    "metadata": {},
    "source": ["# Install ONNX Runtime from bundled wheel\n"
               "import subprocess, sys, os\n"
               "from pathlib import Path\n\n"
               "INPUT_ROOT = Path(\"/kaggle/input\")\n"
               "ONNX_WHL = Path(\"/kaggle/input/datasets/hellodave2035/birdclef-perch-models/onnxruntime-1.24.4-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl\")\n"
               "if ONNX_WHL.exists():\n"
               "    subprocess.run([sys.executable, \"-m\", \"pip\", \"install\", \"-q\", \"--no-deps\", str(ONNX_WHL)], check=True)\n"
               "    print(\"ONNX Runtime installed\")\n\n"
               "# Install TF wheels\n"
               "def find_wheel(pattern):\n"
               "    for p in INPUT_ROOT.rglob(pattern):\n"
               "        return p\n"
               "    return None\n\n"
               "_w = find_wheel(\"tensorboard-2.20.0-*.whl\")\n"
               "if _w:\n"
               "    subprocess.run([sys.executable, \"-m\", \"pip\", \"install\", \"-q\", \"--no-deps\", str(_w)], check=True)\n"
               "_w = find_wheel(\"tensorflow-2.20.0-*.whl\")\n"
               "if _w:\n"
               "    subprocess.run([sys.executable, \"-m\", \"pip\", \"install\", \"-q\", \"--no-deps\", str(_w)], check=True)\n"
               "print(\"TF 2.20 installed\")\n\n"
               "print(\"Dependencies ready\")"]
})

# Cell 2: Imports + seed (from Model_4)
cells.append({
    "cell_type": "code",
    "metadata": {},
    "source": [clean("""import random, os, numpy as np, torch
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(4)
print("Global random seed set to 4")""")]
})

# Cell 3: The main training cell (modified Model_4)
cells.append({
    "cell_type": "code",
    "metadata": {},
    "source": [model4_src]
})

# Cell 4: Export
cells.append({
    "cell_type": "code",
    "metadata": {},
    "source": [export_cell_code]
})

# ── Assemble notebook ──
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

print(f"Saving to {DST_NB.name} ({len(cells)} cells)...")
with open(DST_NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=True)

print(f"\nDone! {DST_NB.name} ready (local only).")
print(f"Cells: {len(cells)}")
print("NOT PUSHED — awaiting user GO.")
