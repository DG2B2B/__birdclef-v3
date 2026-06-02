"""V14 = V13 + fix Platt calibration logit→probability mismatch."""
import json, shutil
from pathlib import Path

SRC = Path(r"c:\Users\Utilisateur\Mon Drive\Kaggle\__birdclef-v3\notebooks\eos_submit_v13.ipynb")
DST = Path(r"c:\Users\Utilisateur\Mon Drive\Kaggle\__birdclef-v3\notebooks\eos_submit_v14.ipynb")

shutil.copy(str(SRC), str(DST))

with open(str(DST), "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find the Platt calibration cell (contains "Per-species Platt calibration")
for i, cell in enumerate(nb["cells"]):
    cell_src = "".join(cell["source"])
    if "Per-species Platt calibration" not in cell_src:
        continue

    print(f"Found Platt calibration cell at index {i}")

    # Replace the entire cell with corrected version
    fixed_cell = """# V14: Per-species Platt calibration (FIXED: logit->prob conversion)
# Only calibrates species with >50 positive OOF samples
# CRITICAL FIX: oof_base contains LOGITS, not probabilities.
# We convert to probs via sigmoid BEFORE training, so both
# training and inference operate in probability space [0,1].
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression

# Try to load OOF data
oof_paths = [
    Path("/kaggle/input/datasets/hideyukizushi/sgkfk-202604041716/perch_cache/full_oof_meta_features.npz"),
    Path("/kaggle/working/perch_cache/full_oof_meta_features.npz"),
]

oof_loaded = False
for oof_path in oof_paths:
    if oof_path.exists():
        oof = np.load(str(oof_path), allow_pickle=True)
        if "oof_base" in oof and "fold_id" in oof:
            oof_loaded = True
            print(f"Loaded OOF data from {oof_path}")
            break

if oof_loaded:
    oof_base_logits = oof["oof_base"]  # (708, 234) Perch base LOGITS
    fold_id = oof["fold_id"]           # (708,) 1-indexed folds

    # --- FIX: Convert logits to probabilities ---
    oof_base = 1.0 / (1.0 + np.exp(-np.clip(oof_base_logits, -30, 30)))

    # Load ground truth labels (aligned with OOF)
    from pathlib import Path as _Path
    labels_csv = _Path("/kaggle/input/competitions/birdclef-2026/train_soundscapes_labels.csv")
    if labels_csv.exists():
        import pandas as _pd
        _labels_df = _pd.read_csv(labels_csv)
        _sample_sub = _pd.read_csv(_Path("/kaggle/input/competitions/birdclef-2026/sample_submission.csv"))
        _species_cols = [c for c in _sample_sub.columns if c != "row_id"]
        _N_CLASSES = len(_species_cols)
        _sp_to_idx = {sp: i for i, sp in enumerate(_species_cols)}

        # Build label matrix
        def _union_labels(series):
            out = set()
            for x in series:
                if _pd.notna(x):
                    for t in str(x).split(";"):
                        t = t.strip()
                        if t: out.add(t)
            return sorted(out)

        _sc = (_labels_df.groupby(["filename", "start", "end"])["primary_label"]
               .apply(_union_labels).reset_index(name="label_list"))
        _sc["end_sec"] = _pd.to_timedelta(_sc["end"]).dt.total_seconds().astype(int)
        _sc["row_id"] = _sc["filename"].str.replace(".ogg", "", regex=False) + "_" + _sc["end_sec"].astype(str)

        _N_WINDOWS = 12
        _windows_per_file = _sc.groupby("filename").size()
        _full_files = sorted(_windows_per_file[_windows_per_file == _N_WINDOWS].index.tolist())

        _all_row_ids = []
        for _fname in _full_files:
            _stem = _fname.replace(".ogg", "")
            for _t in range(5, 65, 5):
                _all_row_ids.append(f"{_stem}_{_t}")

        _sc_dict = {}
        for _, _row in _sc.iterrows():
            _sc_dict[_row["row_id"]] = _row["label_list"]

        _y = np.zeros((len(_all_row_ids), _N_CLASSES), dtype=np.float32)
        for _i, _rid in enumerate(_all_row_ids):
            if _rid in _sc_dict:
                for _lbl in _sc_dict[_rid]:
                    if _lbl in _sp_to_idx:
                        _y[_i, _sp_to_idx[_lbl]] = 1.0

        # Align with OOF
        _N_OOF = len(oof_base)
        if len(_y) >= _N_OOF:
            _y = _y[:_N_OOF]
        else:
            _pad = np.zeros((_N_OOF - len(_y), _N_CLASSES), dtype=np.float32)
            _y = np.vstack([_y, _pad])

        # Conservative Platt: only calibrate species with >50 positives
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
            )[:, 1]

        submission[species_cols] = np.clip(_subm_arr, 0.0, 1.0)
        print(f"Platt calibration applied to {_calibrated}/{_N_CLASSES} species (min_pos={MIN_POS})")
    else:
        print("Labels CSV not found - skipping Platt calibration")
else:
    print("OOF data not found - skipping Platt calibration")
"""

    nb["cells"][i]["source"] = [fixed_cell]
    print(f"Fixed Platt calibration cell {i}")
    break

# Save
with open(str(DST), "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=True)

# Verify
with open(str(DST), "r", encoding="utf-8") as f:
    vnb = json.load(f)

for i, cell in enumerate(vnb["cells"]):
    src = "".join(cell["source"])
    if "logit->prob" in src or "LOGITS" in src:
        print(f"  OK: Cell {i} has logit fix")
    if "Convert logits to probabilities" in src:
        print(f"  OK: Cell {i} has sigmoid conversion")

print(f"\nV14 saved: {DST.name} ({len(vnb['cells'])} cells)")
print("NOT PUSHED")
