"""
V6 CORRECTIVE — Rollback rank.1 + conservative ensemble
=========================================================
Root cause analysis of 0.947 → 0.855 drop:
  PRIMARY:   'rank.1' mode destroys probability calibration.
             Rank averaging maps all species to uniform [0,1] range,
             equalizing zero-shot species (legitimately near 0) with
             common species. This creates massive false positives.
  SECONDARY: Model_3 (0.928) at 15% via rank blending adds noise.
  TERTIARY:  Neighborhood smoothing (alpha=0.25) + species weights
             (strength=0.2) compound the rank averaging damage.

V6 fix:
  1. REVERT type_add to 'direct' (linear probability blending)
  2. Keep Model_3 at conservative 0.10 weight
  3. Reduce smoothing alpha: 0.25 → 0.15
  4. Reduce species weights strength: 0.2 → 0.1
  5. Keep all bug fixes (weight inversion, dry-run fallback)
"""
import json
from pathlib import Path

ROOT = Path(r"C:\Users\Utilisateur\Mon Drive\Kaggle\__birdclef-v3")
SRC_NB = ROOT / "notebooks" / "eos_submit_v5.ipynb"
DST_NB = ROOT / "notebooks" / "eos_submit_v6.ipynb"

print(f"Loading {SRC_NB.name}...")
with open(SRC_NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

# ═══════════════════════════════════════════════════════
# MOD 1: REVERT to 'direct' mode, conservative weights
# ═══════════════════════════════════════════════════════
print("\n--- MOD 1: Revert to direct mode, conservative weights ---")
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    if "solutions = {" in src and "'type_add'" in src:
        # Revert type_add
        src = src.replace("'type_add' :'rank.1'", "'type_add' :'direct'")
        # Conservative weights: 0.10 / 0.90
        src = src.replace(
            "{'Model':'Model_3','subm':'subm_3.csv','weight':0.15,'xSED':['    ,    '],'LB':'0.928'}",
            "{'Model':'Model_3','subm':'subm_3.csv','weight':0.10,'xSED':['    ,    '],'LB':'0.928'}"
        )
        src = src.replace(
            "{'Model':'Model_4','subm':'subm_4.csv','weight':0.85,'xSED':[0.595,0.405],'LB':'0.947'}",
            "{'Model':'Model_4','subm':'subm_4.csv','weight':0.90,'xSED':[0.595,0.405],'LB':'0.947'}"
        )
        nb["cells"][i]["source"] = [src]
        print(f"  OK: type_add='direct', Model_3=0.10, Model_4=0.90")
        break

# ═══════════════════════════════════════════════════════
# MOD 2: Reduce smoothing alpha 0.25 → 0.15
# ═══════════════════════════════════════════════════════
print("\n--- MOD 2: Reduce smoothing alpha ---")
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    if "neighborhood_smooth" in src and "alpha=0.25" in src:
        src = src.replace("alpha=0.25", "alpha=0.15")
        nb["cells"][i]["source"] = [src]
        print(f"  OK: alpha 0.25 → 0.15 in cell {i}")
        break

# ═══════════════════════════════════════════════════════
# MOD 3: Reduce species weights strength 0.2 → 0.1
# ═══════════════════════════════════════════════════════
print("\n--- MOD 3: Reduce species weights strength ---")
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    if "weight_strength = 0.2" in src:
        src = src.replace("weight_strength = 0.2", "weight_strength = 0.1")
        nb["cells"][i]["source"] = [src]
        print(f"  OK: weight_strength 0.2 → 0.1 in cell {i}")
        break

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

# Validate
with open(DST_NB, "r", encoding="utf-8") as f:
    vnb = json.load(f)

src2 = "".join(vnb["cells"][2]["source"])
src14 = "".join(vnb["cells"][14]["source"])

checks = [
    ("direct mode", "'direct'" in src2),
    ("Model_3=0.10", "weight':0.10" in src2 or "weight\":0.10" in src2),
    ("Model_4=0.90", "weight':0.90" in src2 or "weight\":0.90" in src2),
    ("smoothing alpha=0.15", "alpha=0.15" in src14),
    ("weights strength=0.1", "weight_strength = 0.1" in src14),
]
for name, ok in checks:
    print(f"  {'OK' if ok else 'FAIL'}: {name}")

print(f"\nDone! {DST_NB.name} created.")
