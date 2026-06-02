"""
Phase 1 — Apply Quick Wins to EoS V2 notebook
==============================================
Creates eos_submit_v3.ipynb from eos_submit_v2.ipynb with:
  1. Fix rank_1_add2 weight inversion bug
  2. Switch to rank.1 mode
  3. Activate Model_3 with weight 0.15
  4. Add neighborhood temporal smoothing
  5. Integrate species boost config (optional)
"""

import json
import copy
from pathlib import Path

ROOT = Path(r"C:\Users\Utilisateur\Mon Drive\Kaggle\__birdclef-v3")
SRC_NB = ROOT / "notebooks" / "eos_submit_v2.ipynb"
DST_NB = ROOT / "notebooks" / "eos_submit_v3.ipynb"

print(f"Loading {SRC_NB.name}...")
with open(SRC_NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Found {len(nb['cells'])} cells")

# ═══════════════════════════════════════════════════════
# MOD 1: Fix solutions dict (Cell 2)
# ═══════════════════════════════════════════════════════
print("\n--- MOD 1: Fix solutions dict ---")

# Find cell with solutions dict
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    if "solutions = {" in src and "'type_add'" in src:
        print(f"  Found solutions dict in cell {i}")

        # Replace type_add
        src = src.replace("'type_add' :'direct'", "'type_add' :'rank.1'")

        # Replace Model_3 weight 0.0 → 0.15
        src = src.replace(
            "{'Model':'Model_3','subm':'subm_3.csv','weight':0.0,'xSED':['    ,    '],'LB':'0.928'}",
            "{'Model':'Model_3','subm':'subm_3.csv','weight':0.15,'xSED':['    ,    '],'LB':'0.928'}"
        )

        # Replace Model_4 weight 1.0 → 0.85
        src = src.replace(
            "{'Model':'Model_4','subm':'subm_4.csv','weight':1.0,'xSED':[0.595,0.405],'LB':'0.947'}",
            "{'Model':'Model_4','subm':'subm_4.csv','weight':0.85,'xSED':[0.595,0.405],'LB':'0.947'}"
        )

        # Update cell source (preserve list-of-strings format)
        nb["cells"][i]["source"] = [src]
        print(f"  ✓ Updated cell {i}")
        break
else:
    print("  ⚠ Could not find solutions dict cell!")

# ═══════════════════════════════════════════════════════
# MOD 2: Fix rank_1_add2 weight inversion bug (Cell 11)
# ═══════════════════════════════════════════════════════
print("\n--- MOD 2: Fix rank_1_add2 weight bug ---")

for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    if "wSED       = PROTOSSM_W / (PROTOSSM_W + SED_W)" in src:
        print(f"  Found rank_1_add2 in cell {i}")

        # Fix the inversion
        src = src.replace(
            "wSED       = PROTOSSM_W / (PROTOSSM_W + SED_W)",
            "wSED       = SED_W      / (PROTOSSM_W + SED_W)   # FIXED: was inverted"
        )
        src = src.replace(
            "wPROTO     = SED_W      / (PROTOSSM_W + SED_W)",
            "wPROTO     = PROTOSSM_W / (PROTOSSM_W + SED_W)   # FIXED: was inverted"
        )

        nb["cells"][i]["source"] = [src]
        print(f"  ✓ Fixed cell {i}")
        break
else:
    print("  ⚠ Could not find rank_1_add2 cell!")

# ═══════════════════════════════════════════════════════
# MOD 3: Add neighborhood smoothing + species boost (Cell 14)
# ═══════════════════════════════════════════════════════
print("\n--- MOD 3: Add post-processing ---")

# Find the submit cell (contains "submission.to_csv" or "submission = f_add()")
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    if "submission = f_add()" in src or "submission.to_csv" in src:
        print(f"  Found submit cell {i}")

        # Find insertion point: right after "submission = f_add()" or before to_csv
        # We'll insert before the to_csv call
        post_processing_code = """

# ═══════════════════════════════════════════════════════════
# POST-PROCESSING: Neighborhood Smoothing + Species Boost
# ═══════════════════════════════════════════════════════════

import numpy as np

# ── Identify species columns ──
species_cols = [c for c in submission.columns if c != "row_id"]

# ── 1. Neighborhood temporal smoothing ──
# For each soundscape, smooth predictions across adjacent 5s windows:
#   s[t] = (1 - 2*alpha)*s[t] + alpha*(s[t-1] + s[t+1])
def neighborhood_smooth(df, cols, alpha=0.25):
    arr = df[cols].values.copy()
    smoothed = arr.copy()
    df_temp = df.copy()
    df_temp["soundscape"] = df_temp["row_id"].apply(lambda x: "_".join(str(x).split("_")[:-1]))
    for ss, idx in df_temp.groupby("soundscape").groups.items():
        idx = sorted(idx, key=lambda i: int(str(df_temp.loc[i, "row_id"]).split("_")[-1]))
        if len(idx) < 3:
            continue
        for pos in range(1, len(idx) - 1):
            i_cur, i_prev, i_next = idx[pos], idx[pos-1], idx[pos+1]
            smoothed[i_cur] = (1 - 2*alpha) * arr[i_cur] + alpha * (arr[i_prev] + arr[i_next])
    result = df.copy()
    result[cols] = np.clip(smoothed, 0.0, 1.0)
    return result

submission = neighborhood_smooth(submission, species_cols, alpha=0.25)
print("Neighborhood smoothing applied (alpha=0.25)")

# ── 2. Species prevalence weights (optional, conservatif) ──
# Charger depuis le dataset Kaggle si disponible
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
    
    # Simple weight application (no external imports needed)
    species_list = list(boost_cfg["species_weights"].keys())
    weights = np.array([boost_cfg["species_weights"][sp] for sp in species_list])
    weight_strength = 0.2  # Conservateur: 20% de l'effet
    
    # Interpolate weights
    effective_weights = 1.0 + weight_strength * (weights - 1.0)
    
    # Apply to submission
    subm_arr = submission[species_cols].values
    subm_arr = subm_arr * effective_weights[np.newaxis, :]
    # Row-wise renormalization
    row_sums_before = subm_arr.sum(axis=1, keepdims=True)
    subm_arr = subm_arr / (row_sums_before + 1e-8) * (row_sums_before.mean() + 1e-8)
    submission[species_cols] = np.clip(subm_arr, 0.0, 1.0)
    print(f"Species weights applied (strength={weight_strength}, {len(species_list)} species)")
else:
    print("Species boost config not found — skipping (non-critical)")

print("Post-processing complete")
"""

        # Insert before to_csv or before the dry-run override
        if "submission.to_csv" in src:
            src = src.replace(
                "submission.to_csv",
                post_processing_code + "\nsubmission.to_csv"
            )
        elif "submission = f_add()" in src and "submission.to_csv" not in src:
            # Insert after the blend
            src = src.replace(
                "submission = f_add()",
                "submission = f_add()" + post_processing_code
            )
        else:
            # Append at the end
            src = src + post_processing_code

        nb["cells"][i]["source"] = [src]
        print(f"  ✓ Added post-processing to cell {i}")
        break
else:
    print("  ⚠ Could not find submit cell!")

# ═══════════════════════════════════════════════════════
# MOD 4: Also fix rank_1_add3 if present (same bug pattern)
# ═══════════════════════════════════════════════════════
print("\n--- MOD 4: Fix rank_1_add3 (if present) ---")

for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    if "def rank_1_add3():" in src:
        print(f"  Found rank_1_add3 in cell {i}")
        # Check if it has the same inversion pattern
        if "wSED       = PROTO_w1 + PROTO_w2" in src or "wPROTO" in src:
            # The 3-way version has different variable names, let's check
            # For now, just flag it
            print(f"  ⚠ rank_1_add3 found — verify weights manually if using 3-way blend")
        break

# ═══════════════════════════════════════════════════════
# Save
# ═══════════════════════════════════════════════════════
print(f"\nSaving to {DST_NB.name}...")
with open(DST_NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"✓ Done! {DST_NB.name} created with Phase 1 Quick Wins.")

print("\n" + "=" * 60)
print("SUMMARY OF CHANGES:")
print("=" * 60)
print("  1. type_add: 'direct' → 'rank.1'")
print("  2. Model_3 weight: 0.0 → 0.15")
print("  3. Model_4 weight: 1.0 → 0.85")
print("  4. rank_1_add2: weight inversion FIXED")
print("  5. Post-processing: neighborhood smoothing + species weights")
print("=" * 60)
