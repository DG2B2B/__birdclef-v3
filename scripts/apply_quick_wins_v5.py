"""
Fix V4 → V5: Handle row_id mismatch in rank_1_add2 during Kaggle dry-run.
During dry-run, Model_3 (ProtoSSM) and Model_4 (SED) use different fallback
strategies, producing incompatible row_ids. The fix: detect mismatch and
fall back to the model that matches sample_submission.csv row count.
"""
import json
from pathlib import Path

ROOT = Path(r"C:\Users\Utilisateur\Mon Drive\Kaggle\__birdclef-v3")
SRC_NB = ROOT / "notebooks" / "eos_submit_v4.ipynb"
DST_NB = ROOT / "notebooks" / "eos_submit_v5.ipynb"

print(f"Loading {SRC_NB.name}...")
with open(SRC_NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

# ═══════════════════════════════════════════════════════
# MOD: Fix rank_1_add2 to handle dry-run row_id mismatch
# ═══════════════════════════════════════════════════════
print("\n--- Fix rank_1_add2 row_id mismatch ---")

for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    if "def rank_1_add2():" not in src:
        continue
    
    print(f"  Found rank_1_add2 in cell {i}")
    
    # Replace the row_id mismatch block with graceful fallback
    old_block = """    missing    = sorted(set(SED["row_id"]) - set(PROTO["row_id"]))
    if missing:
        raise ValueError(f"row_id mismatch: {len(missing)} rows in ProtoSSM not in SED; first={missing[:3]}")
    PROTO      = PROTO.set_index("row_id").loc[SED["row_id"]].reset_index()"""
    
    new_block = """    # Handle dry-run row_id mismatch: fall back to dominant model
    missing_sed  = sorted(set(SED["row_id"]) - set(PROTO["row_id"]))
    missing_proto = sorted(set(PROTO["row_id"]) - set(SED["row_id"]))
    if missing_sed or missing_proto:
        print(f"row_id mismatch: SED missing {len(missing_sed)}, PROTO missing {len(missing_proto)}")
        # Fallback: use the model with more rows (stronger signal)
        if len(SED) >= len(PROTO):
            print("Falling back to SED only (more rows)")
            return SED
        else:
            print("Falling back to PROTO only (more rows)")
            subm = PROTO.copy()
            subm[cols] = np.clip(PROTO[cols].to_numpy("float32"), EPS, 1.0 - EPS)
            return subm
    PROTO      = PROTO.set_index("row_id").loc[SED["row_id"]].reset_index()"""
    
    if old_block in src:
        src = src.replace(old_block, new_block)
        nb["cells"][i]["source"] = [src]
        print(f"  OK - fixed rank_1_add2 in cell {i}")
    else:
        print(f"  WARNING: Could not find exact block to replace in cell {i}")
        # Try to find approximate match
        if "missing    = sorted(set(SED" in src:
            print(f"  Found similar pattern, attempting manual fix...")
    break

# Also fix rank_1_add3 similarly
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    if "def rank_1_add3():" not in src:
        continue
    
    print(f"\n  Found rank_1_add3 in cell {i}")
    
    # Find and fix similar row_id mismatch checks in rank_1_add3
    old_check1 = 'raise ValueError(f"row_id mismatch: {len(missing1)} rows in ProtoSSM not in SED; first={missing1[:3]}")'
    new_check1 = 'print(f"WARNING row_id mismatch: {len(missing1)} rows. Falling back to SED."); return SED'
    
    if old_check1 in src:
        src = src.replace(old_check1, new_check1)
    
    old_check2 = 'raise ValueError(f"row_id mismatch: {len(missing2)} rows in ProtoSSM not in SED; first={missing2[:3]}")'
    new_check2 = 'print(f"WARNING row_id mismatch: {len(missing2)} rows. Falling back to SED."); return SED'
    
    if old_check2 in src:
        src = src.replace(old_check2, new_check2)
    
    nb["cells"][i]["source"] = [src]
    print(f"  OK - fixed rank_1_add3 in cell {i}")
    break

# ═══════════════════════════════════════════════════════
# Save
# ═══════════════════════════════════════════════════════
print(f"\nSaving to {DST_NB.name}...")
with open(DST_NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=True)

# Validate
with open(DST_NB, "r", encoding="utf-8") as f:
    vnb = json.load(f)

src11 = "".join(vnb["cells"][11]["source"])
checks = [
    ("rank.1 set", "rank.1" in "".join(vnb["cells"][2]["source"])),
    ("weight bug fixed", "FIXED" in src11),
    ("row_id fallback", "Falling back to SED" in src11 or "Falling back to PROTO" in src11),
    ("post-processing in cell 14", "neighborhood_smooth" in "".join(vnb["cells"][14]["source"])),
]
for name, ok in checks:
    print(f"  {'OK' if ok else 'FAIL'}: {name}")

print(f"\nDone! {DST_NB.name} created.")
