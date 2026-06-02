"""
V7 — Fix KeyError: 'row_id' in post-processing
================================================
Root cause: direct_add2() uses index_col="row_id", so the returned DataFrame
has row_id as the INDEX, not a column. The _neighborhood_smooth function
expects it as a column.

Fix: Add submission.reset_index() at the start of post-processing.
"""
import json
from pathlib import Path

ROOT = Path(r"C:\Users\Utilisateur\Mon Drive\Kaggle\__birdclef-v3")
SRC_NB = ROOT / "notebooks" / "eos_submit_v6.ipynb"
DST_NB = ROOT / "notebooks" / "eos_submit_v7.ipynb"

print(f"Loading {SRC_NB.name}...")
with open(SRC_NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

# ── Fix: add reset_index() before post-processing ──
# Also fix the _neighborhood_smooth function to be robust to both index and column

for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    if "_neighborhood_smooth" not in src:
        continue

    print(f"  Found post-processing in cell {i}")

    # Fix 1: Add reset_index before calling _neighborhood_smooth
    src = src.replace(
        "submission = _neighborhood_smooth(submission, species_cols, alpha=0.15)",
        "# Ensure row_id is a column (direct_add2 returns it as index)\n"
        "submission = submission.reset_index()\n"
        "submission = _neighborhood_smooth(submission, species_cols, alpha=0.15)"
    )

    # Fix 2: Make _neighborhood_smooth robust (also accept index)
    # Replace df["row_id"] with a fallback
    src = src.replace(
        'for idx, row_id in enumerate(df["row_id"]):',
        'row_id_col = "row_id" if "row_id" in df.columns else df.index.name\n'
        'row_ids = df[row_id_col] if row_id_col in df.columns else df.index\n'
        'for idx, row_id in enumerate(row_ids):'
    )

    # Fix 3: Also fix the sorting line that accesses row_id
    src = src.replace(
        'indices.sort(key=lambda j: int(str(df.loc[j, "row_id"]).split("_")[-1]))',
        'indices.sort(key=lambda j: int(str(row_ids[j]).split("_")[-1]))'
    )

    nb["cells"][i]["source"] = [src]
    print(f"  OK - fixed row_id access in cell {i}")
    break

# ── Save ──
print(f"\nSaving to {DST_NB.name}...")
with open(DST_NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=True)

print("Done! V7 ready (local only — awaiting go for Kaggle push).")
