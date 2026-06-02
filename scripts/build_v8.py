"""
V8 — Clean rebuild from V2 with all fixes
===========================================
Instead of incremental string patches (which break indentation),
this script rebuilds everything from V2 in one pass.

Changes:
  1. Cell 2: solutions dict (direct mode, Model_3=0.10)
  2. Cell 11: fix rank_1_add2 weight bug + dry-run fallback
  3. NEW Cell 15: post-processing (smoothing + species weights)
  4. Cell 16 (was 15): to_csv
  5. Cell 17 (was 16): dry-run check
"""
import json
import copy
from pathlib import Path

ROOT = Path(r"C:\Users\Utilisateur\Mon Drive\Kaggle\__birdclef-v3")
SRC_NB = ROOT / "notebooks" / "eos_submit_v2.ipynb"
DST_NB = ROOT / "notebooks" / "eos_submit_v8.ipynb"

print(f"Loading {SRC_NB.name}...")
with open(SRC_NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

# ═══════════════════════════════════════════════════════
# MOD 1: Fix solutions dict (Cell 2)
# ═══════════════════════════════════════════════════════
print("MOD 1: Fix solutions dict...")
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    if "solutions = {" in src and "'type_add'" in src:
        src = src.replace("'type_add' :'direct'", "'type_add' :'direct'")  # unchanged
        src = src.replace(
            "{'Model':'Model_3','subm':'subm_3.csv','weight':0.0,'xSED':['    ,    '],'LB':'0.928'}",
            "{'Model':'Model_3','subm':'subm_3.csv','weight':0.10,'xSED':['    ,    '],'LB':'0.928'}"
        )
        src = src.replace(
            "{'Model':'Model_4','subm':'subm_4.csv','weight':1.0,'xSED':[0.595,0.405],'LB':'0.947'}",
            "{'Model':'Model_4','subm':'subm_4.csv','weight':0.90,'xSED':[0.595,0.405],'LB':'0.947'}"
        )
        nb["cells"][i]["source"] = [src]
        print(f"  OK cell {i}")
        break

# ═══════════════════════════════════════════════════════
# MOD 2: Fix rank_1_add2 weight bug + dry-run fallback (Cell 11)
# ═══════════════════════════════════════════════════════
print("MOD 2: Fix rank_1_add2...")
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    if "wSED       = PROTOSSM_W / (PROTOSSM_W + SED_W)" not in src:
        continue
    
    # Fix weight inversion
    src = src.replace(
        "wSED       = PROTOSSM_W / (PROTOSSM_W + SED_W)",
        "wSED       = SED_W      / (PROTOSSM_W + SED_W)   # FIXED: was inverted"
    )
    src = src.replace(
        "wPROTO     = SED_W      / (PROTOSSM_W + SED_W)",
        "wPROTO     = PROTOSSM_W / (PROTOSSM_W + SED_W)   # FIXED: was inverted"
    )
    
    # Fix dry-run row_id mismatch
    old_mismatch = '    missing    = sorted(set(SED["row_id"]) - set(PROTO["row_id"]))\n    if missing:\n        raise ValueError(f"row_id mismatch: {len(missing)} rows in ProtoSSM not in SED; first={missing[:3]}")\n    PROTO      = PROTO.set_index("row_id").loc[SED["row_id"]].reset_index()'
    new_mismatch = '    missing_sed  = sorted(set(SED["row_id"]) - set(PROTO["row_id"]))\n    missing_proto = sorted(set(PROTO["row_id"]) - set(SED["row_id"]))\n    if missing_sed or missing_proto:\n        print(f"row_id mismatch in dry-run: SED missing {len(missing_sed)}, PROTO missing {len(missing_proto)}")\n        if len(SED) >= len(PROTO):\n            print("Falling back to SED only")\n            return SED\n        else:\n            print("Falling back to PROTO only")\n            subm = PROTO.copy()\n            subm[cols] = np.clip(PROTO[cols].to_numpy("float32"), EPS, 1.0 - EPS)\n            return subm\n    PROTO      = PROTO.set_index("row_id").loc[SED["row_id"]].reset_index()'
    
    if old_mismatch in src:
        src = src.replace(old_mismatch, new_mismatch)
    
    nb["cells"][i]["source"] = [src]
    print(f"  OK cell {i}")
    break

# ═══════════════════════════════════════════════════════
# MOD 3: Insert NEW post-processing cell between Cell 14 and Cell 15
# Cell 14 = "submission = f_add()"
# Cell 15 = "submission.to_csv(...)"
# ═══════════════════════════════════════════════════════
print("MOD 3: Insert post-processing cell...")

# Find Cell 14 (submission = f_add())
cell14_idx = None
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    if src.strip().startswith("submission = f_add()"):
        cell14_idx = i
        break

if cell14_idx is None:
    print("  ERROR: Could not find submission = f_add() cell!")
else:
    print(f"  Found submission cell at index {cell14_idx}")
    
    # Create new post-processing cell
    pp_code = """# V8 POST-PROCESSING: Neighborhood Smoothing + Species Boost
import numpy as np

# Ensure row_id is a column (direct_add2 returns it as index)
submission = submission.reset_index()
species_cols = [c for c in submission.columns if c != "row_id"]

# ---- 1. Neighborhood temporal smoothing ----
arr = submission[species_cols].values.copy()
smoothed = arr.copy()

# Group rows by soundscape
ss_map = {}
for idx, row_id in enumerate(submission["row_id"]):
    ss = "_".join(str(row_id).split("_")[:-1])
    ss_map.setdefault(ss, []).append(idx)

alpha = 0.15
for ss, indices in ss_map.items():
    indices.sort(key=lambda j: int(str(submission.loc[j, "row_id"]).split("_")[-1]))
    if len(indices) < 3:
        continue
    for pos in range(1, len(indices) - 1):
        i_cur, i_prev, i_next = indices[pos], indices[pos-1], indices[pos+1]
        smoothed[i_cur] = (1 - 2*alpha) * arr[i_cur] + alpha * (arr[i_prev] + arr[i_next])

submission[species_cols] = np.clip(smoothed, 0.0, 1.0)
print("Neighborhood smoothing applied (alpha=0.15)")

# ---- 2. Species prevalence weights (if config available) ----
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
    weight_strength = 0.1
    effective_weights = 1.0 + weight_strength * (weights - 1.0)
    subm_arr = submission[species_cols].values
    subm_arr = subm_arr * effective_weights[np.newaxis, :]
    row_sums = subm_arr.sum(axis=1, keepdims=True)
    subm_arr = subm_arr / (row_sums + 1e-8) * (row_sums.mean() + 1e-8)
    submission[species_cols] = np.clip(subm_arr, 0.0, 1.0)
    print("Species weights applied (strength=0.1)")
else:
    print("Species boost config not found - skipping")
"""
    
    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [pp_code]
    }
    
    # Insert after cell 14
    nb["cells"].insert(cell14_idx + 1, new_cell)
    print(f"  Inserted post-processing cell at index {cell14_idx + 1}")

# ═══════════════════════════════════════════════════════
# MOD 4: Clean encoding
# ═══════════════════════════════════════════════════════
print("MOD 4: Clean encoding...")
for cell in nb["cells"]:
    src = "".join(cell["source"])
    for old, new in [
        ("\u2018", "'"), ("\u2019", "'"),
        ("\u201c", '"'), ("\u201d", '"'),
        ("\u2013", "--"), ("\u2014", "--"),
        ("\u00a0", " "),
    ]:
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

# Validate
with open(DST_NB, "r", encoding="utf-8") as f:
    vnb = json.load(f)

src2 = "".join(vnb["cells"][2]["source"])
src_blend = "".join(vnb["cells"][11]["source"])
pp_src = "".join(vnb["cells"][15]["source"])  # new cell at index 15

checks = [
    ("direct mode", "'direct'" in src2),
    ("Model_3=0.10", "weight':0.10" in src2),
    ("Model_4=0.90", "weight':0.90" in src2),
    ("weight bug fixed", "FIXED" in src_blend),
    ("dry-run fallback", "Falling back" in src_blend),
    ("post-processing cell", "Neighborhood smoothing" in pp_src),
    ("smoothing alpha=0.15", "alpha = 0.15" in pp_src),
    ("weights strength=0.1", "weight_strength = 0.1" in pp_src),
    ("reset_index", "reset_index()" in pp_src),
]
all_ok = True
for name, ok in checks:
    status = "OK" if ok else "FAIL"
    if not ok:
        all_ok = False
    print(f"  {status}: {name}")

print(f"\n{'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
print(f"Total cells: {len(vnb['cells'])} (V2 had 17, V8 has 18)")
print(f"V8 ready at: {DST_NB}")
