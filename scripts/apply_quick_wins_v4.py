"""
Fix V3 → V4: Correct the post-processing cell placement.
V3 had the neighborhood smoothing + species boost inserted into Cell 5 (Model_2 code)
instead of Cell 14 (the blend result cell). This caused an IndentationError.

V4: Post-processing goes in Cell 14, right after `submission = f_add()`.
"""
import json
import copy
from pathlib import Path

ROOT = Path(r"C:\Users\Utilisateur\Mon Drive\Kaggle\__birdclef-v3")
SRC_NB = ROOT / "notebooks" / "eos_submit_v2.ipynb"  # Start fresh from V2
DST_NB = ROOT / "notebooks" / "eos_submit_v4.ipynb"

print(f"Loading {SRC_NB.name}...")
with open(SRC_NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Found {len(nb['cells'])} cells")

# ═══════════════════════════════════════════════════════
# MOD 1: Fix solutions dict (Cell 2) — same as V3
# ═══════════════════════════════════════════════════════
print("\n--- MOD 1: Fix solutions dict ---")
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    if "solutions = {" in src and "'type_add'" in src:
        src = src.replace("'type_add' :'direct'", "'type_add' :'rank.1'")
        src = src.replace(
            "{'Model':'Model_3','subm':'subm_3.csv','weight':0.0,'xSED':['    ,    '],'LB':'0.928'}",
            "{'Model':'Model_3','subm':'subm_3.csv','weight':0.15,'xSED':['    ,    '],'LB':'0.928'}"
        )
        src = src.replace(
            "{'Model':'Model_4','subm':'subm_4.csv','weight':1.0,'xSED':[0.595,0.405],'LB':'0.947'}",
            "{'Model':'Model_4','subm':'subm_4.csv','weight':0.85,'xSED':[0.595,0.405],'LB':'0.947'}"
        )
        nb["cells"][i]["source"] = [src]
        print(f"  OK cell {i}")
        break

# ═══════════════════════════════════════════════════════
# MOD 2: Fix rank_1_add2 weight inversion (Cell 11)
# ═══════════════════════════════════════════════════════
print("\n--- MOD 2: Fix rank_1_add2 ---")
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    if "wSED       = PROTOSSM_W / (PROTOSSM_W + SED_W)" in src:
        src = src.replace(
            "wSED       = PROTOSSM_W / (PROTOSSM_W + SED_W)",
            "wSED       = SED_W      / (PROTOSSM_W + SED_W)   # FIXED: was inverted"
        )
        src = src.replace(
            "wPROTO     = SED_W      / (PROTOSSM_W + SED_W)",
            "wPROTO     = PROTOSSM_W / (PROTOSSM_W + SED_W)   # FIXED: was inverted"
        )
        nb["cells"][i]["source"] = [src]
        print(f"  OK cell {i}")
        break

# ═══════════════════════════════════════════════════════
# MOD 3: Post-processing in CORRECT cell (Cell 14: submission = f_add())
# ═══════════════════════════════════════════════════════
print("\n--- MOD 3: Post-processing in Cell 14 ---")

# Find the cell with "submission = f_add()" (should be cell 14)
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    if src.strip().startswith("submission = f_add()") or "submission = f_add()" in src.split("\n")[0]:
        print(f"  Found submission = f_add() in cell {i}")

        post_code = """
# ═══════════════════════════════════════════════════════════
# V4 POST-PROCESSING: Neighborhood Smoothing + Species Boost
# ═══════════════════════════════════════════════════════════
import numpy as np

species_cols = [c for c in submission.columns if c != "row_id"]

# 1. Neighborhood temporal smoothing
def _neighborhood_smooth(df, cols, alpha=0.25):
    arr = df[cols].values.copy()
    smoothed = arr.copy()
    ss_map = {}
    for idx, row_id in enumerate(df["row_id"]):
        parts = str(row_id).split("_")
        ss = "_".join(parts[:-1])
        if ss not in ss_map:
            ss_map[ss] = []
        ss_map[ss].append(idx)
    for ss, indices in ss_map.items():
        indices.sort(key=lambda j: int(str(df.loc[j, "row_id"]).split("_")[-1]))
        if len(indices) < 3:
            continue
        for pos in range(1, len(indices) - 1):
            i_cur, i_prev, i_next = indices[pos], indices[pos-1], indices[pos+1]
            smoothed[i_cur] = (1 - 2*alpha) * arr[i_cur] + alpha * (arr[i_prev] + arr[i_next])
    result = df.copy()
    result[cols] = np.clip(smoothed, 0.0, 1.0)
    return result

submission = _neighborhood_smooth(submission, species_cols, alpha=0.25)
print("Neighborhood smoothing applied")

# 2. Species prevalence weights (if config available)
import json
from pathlib import Path
boost_cfg_path = None
for candidate in [
    Path("/kaggle/input/birdclef-species-boost/species_boost_config.json"),
    Path("/kaggle/input/datasets/hellodave2035/birdclef-species-boost/species_boost_config.json"),
]:
    if candidate.exists():
        boost_cfg_path = candidate
        break

if boost_cfg_path is not None:
    with open(boost_cfg_path) as f:
        boost_cfg = json.load(f)
    species_list = list(boost_cfg["species_weights"].keys())
    weights = np.array([boost_cfg["species_weights"][sp] for sp in species_list])
    weight_strength = 0.2
    effective_weights = 1.0 + weight_strength * (weights - 1.0)
    subm_arr = submission[species_cols].values
    subm_arr = subm_arr * effective_weights[np.newaxis, :]
    row_sums = subm_arr.sum(axis=1, keepdims=True)
    subm_arr = subm_arr / (row_sums + 1e-8) * (row_sums.mean() + 1e-8)
    submission[species_cols] = np.clip(subm_arr, 0.0, 1.0)
    print("Species weights applied (strength=0.2)")
else:
    print("Species boost config not found - skipping (non-critical)")
"""

        src = src + post_code
        nb["cells"][i]["source"] = [src]
        print(f"  OK - added post-processing to cell {i}")
        break
else:
    print("  ERROR: Could not find submission = f_add() cell!")

# ═══════════════════════════════════════════════════════
# MOD 4: Clean encoding for Kaggle
# ═══════════════════════════════════════════════════════
print("\n--- MOD 4: Clean encoding ---")
for cell in nb["cells"]:
    src = "".join(cell["source"])
    replacements = {
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2013": "--", "\u2014": "--",
        "\u00a0": " ",
        "\u2026": "...",
        "\u00b0": "deg",
        "\u00d7": "x",
        "\u2192": "->",
    }
    for old, new in replacements.items():
        src = src.replace(old, new)
    src = src.encode("ascii", errors="replace").decode("ascii")
    cell["source"] = [src]
print("  OK")

# ═══════════════════════════════════════════════════════
# Save
# ═══════════════════════════════════════════════════════
print(f"\nSaving to {DST_NB.name}...")
with open(DST_NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=True)
print(f"Done! {DST_NB.name} created.")

# Quick validation
print("\n--- Validation ---")
with open(DST_NB, "r", encoding="utf-8") as f:
    vnb = json.load(f)
src14 = "".join(vnb["cells"][14]["source"])
assert "neighborhood_smooth" in src14, "FAIL: post-processing not in cell 14!"
assert "submission = f_add()" in src14, "FAIL: blend call missing!"
src2 = "".join(vnb["cells"][2]["source"])
assert "rank.1" in src2, "FAIL: rank.1 not set!"
src11 = "".join(vnb["cells"][11]["source"])
assert "FIXED" in src11, "FAIL: weight bug not fixed!"
print("All checks passed!")
