"""
V9 fix: Add set_index before to_csv to prevent column shift bug.
V8 produced 0.89 because reset_index() made row_id a column,
but to_csv(index=True) then wrote a numeric index as first column,
shifting all 234 species columns by one position.

Fix: After post-processing, set row_id back as index.
"""
import json, shutil
from pathlib import Path

ROOT = Path(r"C:\Users\Utilisateur\Mon Drive\Kaggle\__birdclef-v3")

# ── Fix V8 → V9 ──
SRC = ROOT / "notebooks" / "eos_submit_v8.ipynb"
DST = ROOT / "notebooks" / "eos_submit_v9.ipynb"

with open(SRC, "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    if "Neighborhood smoothing applied" in src:
        # Add set_index at the end, before the cell ends
        src = src.rstrip() + "\n\n# Restore row_id as index for proper CSV format\nsubmission = submission.set_index(\"row_id\")\n"
        nb["cells"][i]["source"] = [src]
        print(f"V9: Added set_index to cell {i}")
        break

with open(DST, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=True)
print(f"V9 saved: {DST.name}")

# ── Regenerate grid from V9 ──
GRID_DIR = ROOT / "_push_grid"
if GRID_DIR.exists():
    shutil.rmtree(GRID_DIR)
GRID_DIR.mkdir()

for w3 in [0.05, 0.10, 0.15]:
    w4 = round(1.0 - w3, 2)
    name = f"eos-submit-v6-w{w3}"
    slug = f"hellodave2035/eos-v6-w3-0-{str(int(w3*100)).zfill(2)}" if w3 != 0.1 else "hellodave2035/eos-v6-w3-0-1"
    
    nb_grid = json.loads(json.dumps(nb))
    for i, cell in enumerate(nb_grid["cells"]):
        src = "".join(cell["source"])
        if "solutions = {" in src and "'type_add'" in src:
            src = src.replace("'weight':0.10", f"'weight':{w3}")
            # The second replace might match partially - be careful
            src2 = ""
            for line in src.split("\n"):
                if "'weight':0.90" in line:
                    line = line.replace("'weight':0.90", f"'weight':{w4}")
                src2 += line + "\n"
            src = src2
            nb_grid["cells"][i]["source"] = [src]
            break
    
    folder = GRID_DIR / name
    folder.mkdir()
    with open(folder / f"{name}.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb_grid, f, indent=1, ensure_ascii=True)
    
    meta = {
        "id": slug,
        "title": f"EoS v6 w3={w3}",
        "code_file": f"{name}.ipynb",
        "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
        "dataset_sources": ["hellodave2035/birdclef-perch-models","hideyukizushi/sgkfk-202604041716","tuckerarrants/bc2026-distilled-sed-public","tuckerarrants/birdclef-2026-waveform-cache","hellodave2035/birdclef-species-boost"],
        "competition_sources": ["birdclef-2026"], "kernel_sources": [],
        "model_sources": ["google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu/1"],
    }
    with open(folder / "kernel-metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Grid {name}: w3={w3}, w4={w4}")

print(f"\nV9 + {3} grid variants ready in {GRID_DIR}")
print("Awaiting GO for Kaggle push.")
