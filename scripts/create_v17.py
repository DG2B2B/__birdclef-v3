"""V17 = V16 + fix direct_add2/direct_add3 dry-run row_id mismatch + Platt NaN guard."""
import json, shutil
from pathlib import Path

SRC = Path(r"c:\Users\Utilisateur\Mon Drive\Kaggle\__birdclef-v3\notebooks\eos_submit_v16.ipynb")
DST = Path(r"c:\Users\Utilisateur\Mon Drive\Kaggle\__birdclef-v3\notebooks\eos_submit_v17.ipynb")

shutil.copy(str(SRC), str(DST))
with open(str(DST), "r", encoding="utf-8") as f:
    nb = json.load(f)

# ============================================================
# FIX 1: Add dry-run row_id mismatch fallback to direct_add2 and direct_add3
# ============================================================
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    if "def direct_add2():" not in src:
        continue

    print(f"Found blend functions in cell {i}")

    # Fix direct_add2
    old_d2 = """def direct_add2():
    print(f'Ensemble: {_ensemble_models},   LB: {_lbs},   weights: {_weights}')
    df0 = pd.read_csv(_files_subm[0], index_col = "row_id")              ;display(df0)
    df1 = pd.read_csv(_files_subm[1], index_col = "row_id")[df0.columns] ;display(df1)
    dfs = df0 * _weights[0] + _weights[1] * df1
    return dfs"""

    new_d2 = """def direct_add2():
    print(f'Ensemble: {_ensemble_models},   LB: {_lbs},   weights: {_weights}')
    df0 = pd.read_csv(_files_subm[0], index_col = "row_id")
    df1 = pd.read_csv(_files_subm[1], index_col = "row_id")[df0.columns]
    # Handle dry-run row_id mismatch: fall back to model with more rows
    common = df0.index.intersection(df1.index)
    if len(common) < len(df0.index) or len(common) < len(df1.index):
        print(f"row_id mismatch: df0={len(df0)}, df1={len(df1)}, common={len(common)}")
        if len(df0) >= len(df1):
            print("Falling back to Model_3 only")
            return df0
        else:
            print("Falling back to Model_4 only")
            return df1
    dfs = df0 * _weights[0] + _weights[1] * df1
    return dfs"""

    if old_d2 in src:
        src = src.replace(old_d2, new_d2)
        print("  Fixed direct_add2")
    else:
        print("  WARNING: direct_add2 pattern not matched")

    # Fix direct_add3
    old_d3 = """def direct_add3():
    print(f'Ensemble: {_ensemble_models},   LB: {_lbs},   weights: {_weights}')
    df0 = pd.read_csv(_files_subm[0], index_col = "row_id")              ;display(df0)
    df1 = pd.read_csv(_files_subm[1], index_col = "row_id")[df0.columns] ;display(df1)
    df2 = pd.read_csv(_files_subm[2], index_col = "row_id")[df0.columns] ;display(df2)
    dfs = \\
        df0 * _weights[0] + \\
        df1 * _weights[1] + \\
        df2 * _weights[2]
    return dfs"""

    new_d3 = """def direct_add3():
    print(f'Ensemble: {_ensemble_models},   LB: {_lbs},   weights: {_weights}')
    df0 = pd.read_csv(_files_subm[0], index_col = "row_id")
    df1 = pd.read_csv(_files_subm[1], index_col = "row_id")[df0.columns]
    df2 = pd.read_csv(_files_subm[2], index_col = "row_id")[df0.columns]
    # Handle dry-run row_id mismatch: fall back to model with most rows
    common = df0.index.intersection(df1.index).intersection(df2.index)
    if len(common) < max(len(df0.index), len(df1.index), len(df2.index)):
        print(f"row_id mismatch: df0={len(df0)}, df1={len(df1)}, df2={len(df2)}, common={len(common)}")
        # Fall back to the model with the most rows (strongest signal)
        sizes = [(len(df0), df0, "Model_2"), (len(df1), df1, "Model_3"), (len(df2), df2, "Model_4")]
        _, best_df, best_name = max(sizes, key=lambda x: x[0])
        print(f"Falling back to {best_name} only")
        return best_df
    dfs = (
        df0 * _weights[0] +
        df1 * _weights[1] +
        df2 * _weights[2]
    )
    return dfs"""

    if old_d3 in src:
        src = src.replace(old_d3, new_d3)
        print("  Fixed direct_add3")
    else:
        print("  WARNING: direct_add3 pattern not matched")

    nb["cells"][i]["source"] = [src]
    break

# ============================================================
# FIX 2: Add NaN guard to Platt calibration
# ============================================================
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    if "Platt calibration" not in src:
        continue

    print(f"\nFound Platt calibration cell {i}")

    # Add NaN check before applying calibration
    old_apply = """            # Apply to submission (both in probability space - MATCHING!)
            _subm_arr[:, _s_idx] = _lr.predict_proba(
                _subm_arr[:, _s_idx].reshape(-1, 1)
            )[:, 1]"""

    new_apply = """            # Apply to submission (both in probability space - MATCHING!)
            _col_vals = _subm_arr[:, _s_idx]
            if np.isnan(_col_vals).any():
                continue  # skip species with NaN in submission (dry-run artifact)
            _subm_arr[:, _s_idx] = _lr.predict_proba(
                _col_vals.reshape(-1, 1)
            )[:, 1]"""

    if old_apply in src:
        src = src.replace(old_apply, new_apply)
        print("  Added NaN guard to Platt calibration")
    else:
        print("  WARNING: Platt apply pattern not matched")

    # Also add NaN guard for the training data
    old_train = """            # Train on 100% OOF data (no fold loop - test set is independent)
            _lr = LogisticRegression(C=100.0, max_iter=1000, class_weight="balanced")
            _lr.fit(oof_base[:, _s_idx].reshape(-1, 1), _y_sp)"""

    new_train = """            # Train on 100% OOF data (no fold loop - test set is independent)
            _oof_col = oof_base[:, _s_idx]
            if np.isnan(_oof_col).any():
                continue  # skip species with NaN in OOF data
            _lr = LogisticRegression(C=100.0, max_iter=1000, class_weight="balanced")
            _lr.fit(_oof_col.reshape(-1, 1), _y_sp)"""

    if old_train in src:
        src = src.replace(old_train, new_train)
        print("  Added NaN guard to Platt training")
    else:
        print("  WARNING: Platt train pattern not matched")

    nb["cells"][i]["source"] = [src]
    break

# Save
with open(str(DST), "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=True)

print(f"\nV17 saved: {DST.name} ({len(nb['cells'])} cells)")
print("NOT PUSHED")
