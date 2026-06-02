"""
XGBoost Stacker for BirdCLEF 2026 — v2
========================================
Builds a meta-learner on top of Perch OOF logits using the 708 OOF
training soundscapes segments.

Alignment strategy: Replicates the EXACT EoS V2 notebook logic:
  1. Find fully-labeled soundscapes (all 12 windows have labels)
  2. Sort them alphabetically
  3. Generate row_ids: {stem}_{5}, {stem}_{10}, ..., {stem}_{60}
  4. Match with labels from train_soundscapes_labels.csv

Author: BirdCLEF 2026 Stacking Sprint
Date: 2026-05-31
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from scipy.optimize import minimize
import json
import warnings
import re

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════
ROOT = Path(r"C:\Users\Utilisateur\Mon Drive\Kaggle\__birdclef-v3")
OOF_FILE = ROOT / "sgkf_dataset_v3" / "perch_cache" / "full_oof_meta_features.npz"
LABELS_CSV = Path(r"C:\Users\Utilisateur\Mon Drive\Kaggle\__birdclef-V2\birdclef-2026-data\train_soundscapes_labels.csv")
SAMPLE_SUB = Path(r"C:\Users\Utilisateur\Mon Drive\Kaggle\__birdclef-V2\birdclef-2026-data\sample_submission.csv")
OUTPUT_DIR = ROOT / "models" / "xgb_stacker"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════
# 1. Load OOF data
# ═══════════════════════════════════════════════════════
print("=" * 60)
print("1. Loading OOF meta-features...")
oof = np.load(OOF_FILE, allow_pickle=True)
oof_base = oof["oof_base"]     # (708, 234) — Perch direct logits
oof_prior = oof["oof_prior"]   # (708, 234) — SSM-corrected predictions
fold_id = oof["fold_id"]       # (708,) — fold assignment (1-indexed)

N_SAMPLES, N_CLASSES = oof_base.shape
print(f"   oof_base:  {oof_base.shape}  range [{oof_base.min():.4f}, {oof_base.max():.4f}]")
print(f"   oof_prior: {oof_prior.shape}  range [{oof_prior.min():.4f}, {oof_prior.max():.4f}]")
print(f"   fold_id:   {fold_id.shape}  folds {np.unique(fold_id)}")
print(f"   {N_SAMPLES} segments x {N_CLASSES} species")
assert not np.isnan(oof_base).any(), "NaN in oof_base!"
assert not np.isnan(oof_prior).any(), "NaN in oof_prior!"

# ═══════════════════════════════════════════════════════
# 2. Build ground-truth labels (EoS V2 exact replication)
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. Building ground-truth labels (EoS V2 logic)...")

# Load species list from sample submission (canonical order)
sample_sub = pd.read_csv(SAMPLE_SUB)
species_cols = sample_sub.columns[1:].tolist()
assert len(species_cols) == N_CLASSES
species_to_idx = {sp: i for i, sp in enumerate(species_cols)}

# Load soundscapes labels
labels_df = pd.read_csv(LABELS_CSV)

# ── Replicate EoS V2: soundscape_labels → union_labels → windows_per_file → full_files ──
def union_labels(series):
    out = set()
    for x in series:
        if pd.notna(x):
            for t in str(x).split(";"):
                t = t.strip()
                if t: out.add(t)
    return sorted(out)

sc = (labels_df
      .groupby(["filename", "start", "end"])["primary_label"]
      .apply(union_labels)
      .reset_index(name="label_list"))

sc["end_sec"] = pd.to_timedelta(sc["end"]).dt.total_seconds().astype(int)
sc["row_id"] = sc["filename"].str.replace(".ogg", "", regex=False) + "_" + sc["end_sec"].astype(str)

N_WINDOWS = 12
windows_per_file = sc.groupby("filename").size()
# EoS V2: only keep files with EXACTLY 12 labeled windows
full_files = sorted(windows_per_file[windows_per_file == N_WINDOWS].index.tolist())

print(f"   Fully-labeled files (all 12 windows): {len(full_files)}")

# ── Generate row_ids in EoS V2 exact order ──
# In the notebook: for path in paths → row_ids = [f"{stem}_{t}" for t in range(5,65,5)]
all_row_ids = []
for fname in full_files:
    stem = fname.replace(".ogg", "")
    for t in range(5, 65, 5):
        all_row_ids.append(f"{stem}_{t}")

print(f"   Total row_ids generated: {len(all_row_ids)}")

# ── Build label matrix ──
# Map row_id → list of species
sc_dict = {}
for _, row in sc.iterrows():
    sc_dict[row["row_id"]] = row["label_list"]

y = np.zeros((len(all_row_ids), N_CLASSES), dtype=np.float32)
for i, rid in enumerate(all_row_ids):
    if rid in sc_dict:
        for lbl in sc_dict[rid]:
            if lbl in species_to_idx:
                y[i, species_to_idx[lbl]] = 1.0

# ── Align with OOF ──
if len(all_row_ids) == N_SAMPLES:
    print("   ✓ Row count matches OOF exactly!")
elif len(all_row_ids) > N_SAMPLES:
    print(f"   ⚠ Trimming from {len(all_row_ids)} to {N_SAMPLES}")
    y = y[:N_SAMPLES]
else:
    print(f"   ⚠ Padding from {len(all_row_ids)} to {N_SAMPLES}")
    pad = np.zeros((N_SAMPLES - len(y), N_CLASSES), dtype=np.float32)
    y = np.vstack([y, pad])

# ── Alignment quality check ──
print(f"\n   Alignment check (AUC oof_base vs y, top species):")
n_pos_per_sp = y.sum(axis=0)
top_idx = np.argsort(-n_pos_per_sp)[:15]
aucs = []
for idx in top_idx:
    if 0 < y[:, idx].sum() < N_SAMPLES:
        try:
            auc_val = roc_auc_score(y[:, idx], oof_base[:, idx])
            aucs.append(auc_val)
            marker = "✓" if auc_val > 0.80 else ("~" if auc_val > 0.65 else "✗")
            print(f"     {marker} {species_cols[idx]}: n_pos={y[:, idx].sum():.0f}, AUC={auc_val:.4f}")
        except ValueError:
            pass

mean_auc = np.mean(aucs) if aucs else 0.0
print(f"   Mean alignment AUC: {mean_auc:.4f} {'✓' if mean_auc > 0.80 else '⚠'}")
print(f"   y shape: {y.shape}, positive/class: min={y.sum(0).min():.0f} max={y.sum(0).max():.0f} mean={y.sum(0).mean():.1f}")

# Save alignment for later use
np.savez(OUTPUT_DIR / "oof_labels.npz", y=y, oof_base=oof_base, oof_prior=oof_prior, fold_id=fold_id, row_ids=np.array(all_row_ids[:N_SAMPLES]))
print("   Saved oof_labels.npz")

# ═══════════════════════════════════════════════════════
# 3. Build XGBoost features
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. Building XGBoost features...")

# Context features (global statistics, same for all species)
context_features = np.column_stack([
    oof_base.mean(axis=1),
    oof_prior.mean(axis=1),
    oof_base.max(axis=1),
    oof_prior.max(axis=1),
    oof_base.std(axis=1),
    oof_prior.std(axis=1),
])

# One-hot encode fold_id (values 1-5 → 0-4)
fold_onehot = np.eye(5)[fold_id - 1]

print(f"   Context features: {context_features.shape[1]}")
print(f"   Fold one-hot: {fold_onehot.shape[1]}")
print(f"   Per-species: base + prior = 2 → Total = {2 + context_features.shape[1] + fold_onehot.shape[1]}")

# ═══════════════════════════════════════════════════════
# 4. Train XGBoost per species (5-fold OOF)
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4. Training XGBoost per species (5-fold cross-val)...")

unique_folds = sorted(np.unique(fold_id))
xgb_models = {}
oof_xgb = np.zeros_like(oof_base)

for s_idx in range(N_CLASSES):
    sp_name = species_cols[s_idx]
    y_sp = y[:, s_idx]
    n_pos = y_sp.sum()

    # Skip zero-shot (no positive labels ever)
    if n_pos == 0:
        oof_xgb[:, s_idx] = oof_base[:, s_idx]
        xgb_models[sp_name] = None
        continue

    # Build per-species feature matrix
    X_sp = np.column_stack([
        oof_base[:, s_idx],
        oof_prior[:, s_idx],
        context_features,
        fold_onehot,
    ])

    oof_preds = np.zeros(N_SAMPLES)

    for fold in unique_folds:
        train_idx = np.where(fold_id != fold)[0]
        val_idx = np.where(fold_id == fold)[0]

        X_tr, y_tr = X_sp[train_idx], y_sp[train_idx]
        X_val, y_val = X_sp[val_idx], y_sp[val_idx]

        if y_tr.sum() == 0:
            oof_preds[val_idx] = oof_base[val_idx, s_idx]
            continue

        scale_pos_weight = (y_tr == 0).sum() / max(y_tr.sum(), 1)

        model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=min(scale_pos_weight, 50.0),
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42 + fold,
            verbosity=0,
        )
        model.fit(X_tr, y_tr)
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

        if fold == unique_folds[-1]:
            xgb_models[sp_name] = model

    oof_xgb[:, s_idx] = oof_preds

    # Periodic progress
    if s_idx < 10 or (s_idx + 1) % 25 == 0:
        try:
            auc_base = roc_auc_score(y_sp, oof_base[:, s_idx])
            auc_xgb = roc_auc_score(y_sp, oof_preds)
            delta = auc_xgb - auc_base
            print(f"   [{s_idx:3d}] {sp_name}: base={auc_base:.4f} xgb={auc_xgb:.4f} (Δ{delta:+.4f}) n={n_pos:.0f}")
        except ValueError:
            print(f"   [{s_idx:3d}] {sp_name}: n_pos={n_pos:.0f} (AUC undefined)")

# ═══════════════════════════════════════════════════════
# 5. Evaluate OOF
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("5. OOF Evaluation...")

def macro_auc(y_true, y_pred):
    aucs = []
    for c in range(y_true.shape[1]):
        if y_true[:, c].sum() == 0:
            continue
        try:
            aucs.append(roc_auc_score(y_true[:, c], y_pred[:, c]))
        except ValueError:
            pass
    return float(np.mean(aucs)) if aucs else 0.5

auc_base_all = macro_auc(y, oof_base)
auc_xgb_all = macro_auc(y, oof_xgb)
print(f"   Perch base OOF AUC:  {auc_base_all:.6f}")
print(f"   XGBoost OOF AUC:     {auc_xgb_all:.6f}  (Δ {auc_xgb_all - auc_base_all:+.6f})")

# ═══════════════════════════════════════════════════════
# 6. Optimize late fusion weights
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("6. Optimizing late fusion weights...")

def objective(w):
    blended = w[0] * oof_base + w[1] * oof_xgb
    return -macro_auc(y, blended)

best_score = -np.inf
best_w = np.array([0.5, 0.5])

for _ in range(20):
    w0 = np.random.dirichlet(np.ones(2))
    result = minimize(
        objective, w0, method="SLSQP",
        bounds=[(0, 1), (0, 1)],
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if -result.fun > best_score:
        best_score = -result.fun
        best_w = result.x

print(f"   Optimal weights: Base={best_w[0]:.4f}, XGBoost={best_w[1]:.4f}")
print(f"   Ensemble OOF AUC:  {best_score:.6f}  (Δ {best_score - auc_base_all:+.6f})")

# ═══════════════════════════════════════════════════════
# 7. Save models & artifacts
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("7. Saving models and artifacts...")

models_dir = OUTPUT_DIR / "models_json"
models_dir.mkdir(exist_ok=True)

n_saved = 0
for sp_name, model in xgb_models.items():
    if model is not None:
        model.save_model(str(models_dir / f"{sp_name}.json"))
        n_saved += 1

print(f"   Saved {n_saved}/{N_CLASSES} XGBoost models → {models_dir}")

weights = {
    "base_perch_weight": float(best_w[0]),
    "xgb_stacker_weight": float(best_w[1]),
    "oof_auc_base": float(auc_base_all),
    "oof_auc_xgb": float(auc_xgb_all),
    "oof_auc_ensemble": float(best_score),
    "n_models_trained": n_saved,
    "n_zero_shot_skipped": N_CLASSES - n_saved,
    "n_samples": N_SAMPLES,
    "description": "XGBoost meta-learner on Perch+SSM OOF logits. Late fusion with EoS Perch base.",
}

with open(OUTPUT_DIR / "fusion_weights.json", "w") as f:
    json.dump(weights, f, indent=2)
print(f"   Saved fusion_weights.json")

np.savez(
    OUTPUT_DIR / "oof_predictions.npz",
    oof_base=oof_base,
    oof_xgb=oof_xgb,
    y_true=y,
    fold_id=fold_id,
    species_cols=np.array(species_cols),
)
print(f"   Saved oof_predictions.npz")

# ═══════════════════════════════════════════════════════
# 8. Summary
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("8. SUMMARY")
print("=" * 60)
print(f"   Samples: {N_SAMPLES} | Species: {N_CLASSES} | Trained: {n_saved}")
print(f"   Perch base AUC:  {auc_base_all:.6f}")
print(f"   XGBoost OOF AUC: {auc_xgb_all:.6f}  (Δ {auc_xgb_all - auc_base_all:+.6f})")
print(f"   Ensemble AUC:    {best_score:.6f}  (Δ {best_score - auc_base_all:+.6f})")
print(f"   Fusion: {best_w[0]:.1%} Perch + {best_w[1]:.1%} XGBoost")
print(f"   Output: {OUTPUT_DIR}")
print("=" * 60)

if best_score > auc_base_all + 0.0005:
    print("   ✓ POSITIVE GAIN — worth integrating into submission!")
elif auc_xgb_all > auc_base_all + 0.0005:
    print("   ~ XGBoost alone shows gain, ensemble may help")
else:
    print("   ⚠ MARGINAL/NO GAIN — verify before integrating")
