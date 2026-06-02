"""V18 = V17 + adjusted params: more Model_2, stronger smoothing, more Platt species."""
import json, shutil
from pathlib import Path

SRC = Path(r"c:\Users\Utilisateur\Mon Drive\Kaggle\__birdclef-v3\notebooks\eos_submit_v17.ipynb")
DST = Path(r"c:\Users\Utilisateur\Mon Drive\Kaggle\__birdclef-v3\notebooks\eos_submit_v18.ipynb")

shutil.copy(str(SRC), str(DST))
with open(str(DST), "r", encoding="utf-8") as f:
    nb = json.load(f)

changes = []

# --- Change 1: solutions dict weights ---
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    if "solutions = {" not in src or "type_add" not in src:
        continue
    src = src.replace("'weight':0.05", "'weight':0.10")   # Model_2: 0.05→0.10
    src = src.replace("'weight':0.85", "'weight':0.80")   # Model_4: 0.85→0.80
    nb["cells"][i]["source"] = [src]
    changes.append("Model_2: 0.05→0.10, Model_4: 0.85→0.80")
    break

# --- Change 2: taxon smoothing alphas ---
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    if "alpha_texture=0.30" not in src:
        continue
    src = src.replace("alpha_texture=0.30", "alpha_texture=0.35")
    src = src.replace("alpha_event=0.10", "alpha_event=0.12")
    nb["cells"][i]["source"] = [src]
    changes.append("Smoothing: tex 0.30→0.35, evt 0.10→0.12")
    break

# --- Change 3: Platt min_pos ---
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    if "MIN_POS = 50" not in src:
        continue
    src = src.replace("MIN_POS = 50", "MIN_POS = 30")
    nb["cells"][i]["source"] = [src]
    changes.append("Platt min_pos: 50→30")
    break

# Save
with open(str(DST), "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=True)

print("V18 changes:")
for c in changes:
    print(f"  {c}")

# Push
import subprocess
push_dir = Path(r"c:\Users\Utilisateur\Mon Drive\Kaggle\__birdclef-v3\_push_v18")
push_dir.mkdir(parents=True, exist_ok=True)
shutil.copy(str(DST), push_dir / "eos_submit_v18.ipynb")
meta = {
    "id": "hellodave2035/eos-submit-v18",
    "title": "EoS Submit v18",
    "code_file": "eos_submit_v18.ipynb",
    "language": "python", "kernel_type": "notebook",
    "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
    "dataset_sources": ["hellodave2035/birdclef-perch-models", "hideyukizushi/sgkfk-202604041716",
                        "tuckerarrants/bc2026-distilled-sed-public", "tuckerarrants/birdclef-2026-waveform-cache"],
    "competition_sources": ["birdclef-2026"], "kernel_sources": [],
    "model_sources": ["google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu/1"],
}
with open(push_dir / "kernel-metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

result = subprocess.run(["kaggle", "kernels", "push", "-p", str(push_dir)], capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    print(result.stderr)

shutil.rmtree(push_dir, ignore_errors=True)
print(f"\nV18 saved + pushed: {DST.name}")
