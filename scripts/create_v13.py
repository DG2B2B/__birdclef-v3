"""V13 = V12 + taxon-aware temporal smoothing + per-species Platt calibration."""
import json, shutil
from pathlib import Path

SRC = Path(r"c:\Users\Utilisateur\Mon Drive\Kaggle\__birdclef-v3\notebooks\eos_submit_v12.ipynb")
DST = Path(r"c:\Users\Utilisateur\Mon Drive\Kaggle\__birdclef-v3\notebooks\eos_submit_v13.ipynb")

shutil.copy(str(SRC), str(DST))

with open(str(DST), "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find submission = f_add() cell
for i, cell in enumerate(nb["cells"]):
    cell_src = "".join(cell["source"])
    if cell_src.strip().startswith("submission = f_add()"):

        # === Cell A: Taxon-aware temporal smoothing ===
        taxon_cell_code = """# V13: Taxon-aware temporal smoothing
# Stronger for texture taxa (Amphibia/Insecta), lighter for Aves
import numpy as np
from pathlib import Path

submission = submission.reset_index()
species_cols = [c for c in submission.columns if c != "row_id"]

# Load taxonomy
tax_paths = [Path(p) for p in [
    "/kaggle/input/competitions/birdclef-2026/taxonomy.csv",
    "/kaggle/input/birdclef-2026/taxonomy.csv",
] if Path(p).exists()]
tax = pd.read_csv(tax_paths[0]) if tax_paths else None

TEXTURE_TAXA = {"Amphibia", "Insecta"}

if tax is not None:
    label_to_class = dict(zip(tax["primary_label"].astype(str), tax["class_name"].astype(str)))
    texture_cols = [c for c in species_cols if label_to_class.get(c, "") in TEXTURE_TAXA]
    event_cols = [c for c in species_cols if label_to_class.get(c, "") not in TEXTURE_TAXA]
else:
    texture_cols = []
    event_cols = species_cols

n_tex, n_evt = len(texture_cols), len(event_cols)
print(f"Taxon smoothing: {n_tex} texture (alpha=0.30), {n_evt} event (alpha=0.10)")

def taxon_smooth(df, texture_cols, event_cols, alpha_texture=0.30, alpha_event=0.10):
    all_cols = texture_cols + event_cols
    arr = df[all_cols].values.copy()
    smoothed = arr.copy()
    alpha_map = {}
    for j in range(len(texture_cols)):
        alpha_map[j] = alpha_texture
    for j in range(len(event_cols)):
        alpha_map[len(texture_cols) + j] = alpha_event

    ss_map = {}
    for idx, row_id in enumerate(df["row_id"]):
        ss = "_".join(str(row_id).split("_")[:-1])
        ss_map.setdefault(ss, []).append(idx)

    for ss, indices in ss_map.items():
        indices.sort(key=lambda j: int(str(df.loc[j, "row_id"]).split("_")[-1]))
        if len(indices) < 3:
            continue
        for pos in range(1, len(indices) - 1):
            i_cur = indices[pos]
            i_prev = indices[pos - 1]
            i_next = indices[pos + 1]
            for j in range(arr.shape[1]):
                a = alpha_map.get(j, 0.15)
                smoothed[i_cur, j] = (1 - a) * arr[i_cur, j] + 0.5 * a * (arr[i_prev, j] + arr[i_next, j])

    result = df.copy()
    result[all_cols] = np.clip(smoothed, 0.0, 1.0)
    return result

submission = taxon_smooth(submission, texture_cols, event_cols, alpha_texture=0.30, alpha_event=0.10)
print("Taxon-aware temporal smoothing applied")
submission = submission.set_index("row_id")
"""

        # === Cell B: Per-species Platt calibration ===
        calib_cell_code = """# V13: Per-species Platt calibration (conservative)
# Only calibrates species with >50 positive OOF samples
# Uses pre-computed OOF meta-features from Model_3/Model_4 training
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
    oof_base = oof["oof_base"]  # (708, 234) Perch base logits
    fold_id = oof["fold_id"]    # (708,) 1-indexed folds

    # Load ground truth labels (aligned with OOF)
    # The OOF corresponds to the same 708 fully-labeled training windows
    from pathlib import Path as _Path
    labels_csv = _Path("/kaggle/input/competitions/birdclef-2026/train_soundscapes_labels.csv")
    if labels_csv.exists():
        import pandas as _pd
        _labels_df = _pd.read_csv(labels_csv)
        _sample_sub = _pd.read_csv(_Path("/kaggle/input/competitions/birdclef-2026/sample_submission.csv"))
        _species_cols = [c for c in _sample_sub.columns if c != "row_id"]
        _N_CLASSES = len(_species_cols)
        _sp_to_idx = {sp: i for i, sp in enumerate(_species_cols)}

        # Build label matrix using same logic as Model_3/4
        def _union_labels(series):
            out = set()
            for x in series:
                if _pd.notna(x):
                    for t in str(x).split(";"):
                        t = t.strip()
                        if t:
                            out.add(t)
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

        # Apply calibration to submission
        _subm_arr = submission[species_cols].values.copy()

        for _s_idx in range(_N_CLASSES):
            if _n_pos[_s_idx] < MIN_POS:
                continue

            _y_sp = _y[:, _s_idx]
            if _y_sp.sum() == 0 or _y_sp.sum() == len(_y_sp):
                continue

            # 5-fold Platt calibration
            _oof_cal = np.zeros(_N_OOF)
            for _fold in _unique_folds:
                _tr = np.where(fold_id != _fold)[0]
                _va = np.where(fold_id == _fold)[0]
                if _y_sp[_tr].sum() == 0 or (_y_sp[_tr] == 1).sum() == 0:
                    _oof_cal[_va] = oof_base[_va, _s_idx]
                    continue
                _lr = LogisticRegression(C=100.0, max_iter=1000, class_weight="balanced")
                _lr.fit(oof_base[_tr, _s_idx].reshape(-1, 1), _y_sp[_tr])
                _oof_cal[_va] = _lr.predict_proba(oof_base[_va, _s_idx].reshape(-1, 1))[:, 1]
                _calibrators[_species_cols[_s_idx]] = _lr

            _calibrated += 1

            # Apply to submission
            _subm_arr[:, _s_idx] = _lr.predict_proba(_subm_arr[:, _s_idx].reshape(-1, 1))[:, 1]

        submission[species_cols] = np.clip(_subm_arr, 0.0, 1.0)
        print(f"Platt calibration applied to {_calibrated}/{_N_CLASSES} species (min_pos={MIN_POS})")
    else:
        print("Labels CSV not found - skipping Platt calibration")
else:
    print("OOF data not found - skipping Platt calibration")
"""

        # Insert both new cells after submission = f_add()
        nb["cells"].insert(i + 1, {
            "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [taxon_cell_code]
        })
        nb["cells"].insert(i + 2, {
            "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [calib_cell_code]
        })
        print(f"Inserted taxon smoothing + Platt calibration after cell {i}")
        break

# Save
with open(str(DST), "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=True)

print(f"V13 saved: {DST.name} ({len(nb['cells'])} cells)")
print("NOT PUSHED")
