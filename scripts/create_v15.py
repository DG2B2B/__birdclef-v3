"""V15 = V14 + simplified Platt calibration (100% OOF, no fold loop)."""
import json, shutil
from pathlib import Path

SRC = Path(r"c:\Users\Utilisateur\Mon Drive\Kaggle\__birdclef-v3\notebooks\eos_submit_v14.ipynb")
DST = Path(r"c:\Users\Utilisateur\Mon Drive\Kaggle\__birdclef-v3\notebooks\eos_submit_v15.ipynb")

shutil.copy(str(SRC), str(DST))
with open(str(DST), "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find Platt calibration cell
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    if "logit->prob" not in src and "V14: Per-species Platt" not in src:
        continue
    if "Platt calibration" not in src:
        continue

    print(f"Found Platt cell at index {i}")

    # Replace the 5-fold loop + calibration block with simplified version
    old_block = """        # Conservative Platt: only calibrate species with >50 positives
        MIN_POS = 50
        _n_pos = _y.sum(axis=0)
        _calibrated = 0
        _calibrators = {}

        _unique_folds = sorted(np.unique(fold_id))

        # Apply calibration to submission (already in probability space)
        _subm_arr = submission[species_cols].values.copy()

        for _s_idx in range(_N_CLASSES):
            if _n_pos[_s_idx] < MIN_POS:
                continue

            _y_sp = _y[:, _s_idx]
            if _y_sp.sum() == 0 or _y_sp.sum() == len(_y_sp):
                continue

            # 5-fold Platt calibration (now in probability space)
            for _fold in _unique_folds:
                _tr = np.where(fold_id != _fold)[0]
                _va = np.where(fold_id == _fold)[0]
                if _y_sp[_tr].sum() == 0 or (_y_sp[_tr] == 1).sum() == 0:
                    continue
                _lr = LogisticRegression(C=100.0, max_iter=1000, class_weight="balanced")
                _lr.fit(oof_base[_tr, _s_idx].reshape(-1, 1), _y_sp[_tr])

            _calibrators[_species_cols[_s_idx]] = _lr
            _calibrated += 1

            # Apply to submission (both in probability space - MATCHING!)
            _subm_arr[:, _s_idx] = _lr.predict_proba(
                _subm_arr[:, _s_idx].reshape(-1, 1)
            )[:, 1]"""

    new_block = """        # Conservative Platt: only calibrate species with >50 positives
        MIN_POS = 50
        _n_pos = _y.sum(axis=0)
        _calibrated = 0

        # Apply calibration to submission (already in probability space)
        _subm_arr = submission[species_cols].values.copy()

        for _s_idx in range(_N_CLASSES):
            if _n_pos[_s_idx] < MIN_POS:
                continue

            _y_sp = _y[:, _s_idx]
            if _y_sp.sum() == 0 or _y_sp.sum() == len(_y_sp):
                continue

            # Train on 100% OOF data (no fold loop - test set is independent)
            _lr = LogisticRegression(C=100.0, max_iter=1000, class_weight="balanced")
            _lr.fit(oof_base[:, _s_idx].reshape(-1, 1), _y_sp)
            _calibrated += 1

            # Apply to submission (both in probability space - MATCHING!)
            _subm_arr[:, _s_idx] = _lr.predict_proba(
                _subm_arr[:, _s_idx].reshape(-1, 1)
            )[:, 1]"""

    if old_block in src:
        src = src.replace(old_block, new_block)
        nb["cells"][i]["source"] = [src]
        print(f"  Simplified Platt: 100% OOF, no fold loop")
    else:
        print("  WARNING: Could not find exact block to replace")
        # Try fuzzy match
        if "5-fold Platt calibration" in src:
            src = src.replace("5-fold Platt calibration (now in probability space)", "Platt calibration on 100% OOF (no fold loop)")
            nb["cells"][i]["source"] = [src]
            print("  Fuzzy fix applied")
    break

# Save
with open(str(DST), "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=True)

print(f"\nV15 saved: {DST.name} ({len(nb['cells'])} cells)")
print("NOT PUSHED")
