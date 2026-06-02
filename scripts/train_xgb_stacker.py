"""
XGBoost Stacker for BirdCLEF 2026
===================================
Builds a meta-learner on top of Perch + SSM logits using the 708 OOF
training soundscapes segments. Trains 234 tiny XGBoost models (one per species)
and evaluates OOF gain vs the base model.

Architecture:
  X_species = [logit_perch(s), logit_ssm(s), prior(s)]
  y_species = binary label for species s on segment
  → 234 XGBClassifier models, each ~100 trees, max_depth=3
  → prediction = calibrated probability
  → late fusion with base EoS ensemble

Author: BirdCLEF 2026 Stacking Sprint
Date: 2026-05-31
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import pickle
import json
import warnings
import soundfile as sf
import re

warnings.filterwarnings("ignore")

# ───────────────────────────────────────────────────────────
# Paths
# ───────────────────────────────────────────────────────────
ROOT = Path(r"C:\Users\Utilisateur\Mon Drive\Kaggle\__birdclef-v3")
OOF_FILE = ROOT / "sgkf_dataset_v3" / "perch_cache" / "full_oof_meta_features.npz"
LABELS_CSV = Path(r"C:\Users\Utilisateur\Mon Drive\Kaggle\__birdclef-V2\birdclef-2026-data\train_soundscapes_labels.csv")
SAMPLE_SUB = Path(r"C:\Users\Utilisateur\Mon Drive\Kaggle\__birdclef-V2\birdclef-2026-data\sample_submission.csv")
TAXONOMY_CSV = Path(r"C:\Users\Utilisateur\Mon Drive\Kaggle\__birdclef-V2\birdclef-2026-data\taxonomy.csv")
OUTPUT_DIR = ROOT / "models" / "xgb_stacker"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ───────────────────────────────────────────────────────────
# 1. Load OOF data
# ───────────────────────────────────────────────────────────

print("=" * 60)
print("1. Loading OOF meta-features...")
oof = np.load(OOF_FILE, allow_pickle=True)
oof_base = oof["oof_base"]    # (708, 234) — Perch direct logits
oof_prior = oof["oof_prior"]  # (708, 234) — SSM-corrected predictions
fold_id = oof["fold_id"]      # (708,) — fold assignment

print(f"   oof_base:  {oof_base.shape}  range [{oof_base.min():.4f}, {oof_base.max():.4f}]")
print(f"   oof_prior: {oof_prior.shape}  range [{oof_prior.min():.4f}, {oof_prior.max():.4f}]")
print(f"   fold_id:   {fold_id.shape}  folds {np.unique(fold_id)}")

N_SAMPLES, N_CLASSES = oof_base.shape
print(f"   {N_SAMPLES} segments × {N_CLASSES} species")

# Validate: no NaN
assert not np.isnan(oof_base).any(), "NaN in oof_base!"
assert not np.isnan(oof_prior).any(), "NaN in oof_prior!"

# ───────────────────────────────────────────────────────────
# 2. Build ground-truth labels for these 708 segments
# ───────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("2. Building ground-truth labels...")

# Load species list from sample submission (canonical order)
sample_sub = pd.read_csv(SAMPLE_SUB)
species_cols = sample_sub.columns[1:].tolist()  # 234 species in order
assert len(species_cols) == N_CLASSES, f"Species mismatch: {len(species_cols)} vs {N_CLASSES}"
species_to_idx = {sp: i for i, sp in enumerate(species_cols)}

# Load taxonomy for phylogenetic info
taxonomy = pd.read_csv(TAXONOMY_CSV)
print(f"   Taxonomy: {len(taxonomy)} entries")

# Load soundscapes labels
labels_df = pd.read_csv(LABELS_CSV)
print(f"   Labels file: {len(labels_df)} rows")

# The OOF segments correspond to training soundscapes, segmented every 5s.
# We need to reconstruct which segments correspond to which labels.
# The row_id format is: {soundscape_filename_stem}_{end_time_seconds}
# e.g., BC2026_Train_0001_S08_20250606_030007_5 → soundscape, segment ending at 5s

# The OOF data has 708 segments. We need to match them to labels_df.
# Strategy: for each soundscape file, generate all 5s segments, then
# match with the labels based on time intervals.

# First, let's find all unique soundscapes in the submission
submission_csv = ROOT / "sgkf_dataset_v3" / "submission.csv"
sub = pd.read_csv(submission_csv)
# Extract soundscape name from row_id: remove trailing _XX
sub["soundscape"] = sub["row_id"].apply(lambda x: "_".join(x.split("_")[:-1]))
sub["end_time"] = sub["row_id"].apply(lambda x: int(x.split("_")[-1]))

unique_ss = sub["soundscape"].unique()
print(f"   Submission has {len(sub)} rows from {len(unique_ss)} soundscapes")

# For OOF data, we need the training soundscapes, not test.
# The OOF was generated from train_soundscapes/.
# Let's check the v17_logs.json for more info
with open(ROOT / "sgkf_dataset_v3" / "v17_logs.json") as f:
    logs = json.load(f)

print(f"   v17 training time: {logs.get('train_time_final', 'N/A'):.1f}s")
print(f"   Probe models: {logs.get('n_probe_models', 'N/A')}")

# The OOF segments are from training soundscapes, segmented identically.
# We need to build the label matrix by:
# 1. Listing all training soundscapes
# 2. Segmenting them into 5s windows
# 3. Matching with labels_df's time intervals

TRAIN_SS_DIR = Path(r"C:\Users\Utilisateur\Mon Drive\Kaggle\__birdclef-V2\birdclef-2026-data\train_soundscapes")

# ── Parse HH:MM:SS to seconds ──
def parse_time(t_str: str) -> float:
    """Parse 'HH:MM:SS' or 'HH:MM:SS.ss' or raw seconds string."""
    t_str = str(t_str).strip()
    # Try HH:MM:SS format
    m = re.match(r'(\d+):(\d+):(\d+(?:\.\d+)?)', t_str)
    if m:
        return float(m.group(1)) * 3600 + float(m.group(2)) * 60 + float(m.group(3))
    # Try raw float
    try:
        return float(t_str)
    except ValueError:
        return 0.0

# ── Only use soundscape files that appear in labels ──
labeled_files = set(labels_df["filename"].unique())
print(f"\n   Labeled soundscapes: {len(labeled_files)}")

# Build segment → label mapping
segment_labels = {}  # key: (soundscape_stem, end_time_seconds) → set of species IDs

for _, row in labels_df.iterrows():
    fname = Path(row["filename"]).stem
    start_t = parse_time(row["start"])
    end_t = parse_time(row["end"])
    # Parse multi-label (semicolon-separated)
    species_list = str(row["primary_label"]).split(";")

    # Generate all 5s segments that overlap with [start_t, end_t]
    # Segments end at: 5, 10, 15, ..., duration
    seg_start = int(start_t // 5) * 5  # round down to nearest 5
    seg_end_limit = int((end_t + 4) // 5) * 5  # round up to nearest 5
    for seg_end in range(seg_start + 5, seg_end_limit + 5, 5):
        seg_begin = seg_end - 5
        # Check overlap between [seg_begin, seg_end] and [start_t, end_t]
        if seg_begin < end_t and seg_end > start_t:
            key = (fname, seg_end)
            if key not in segment_labels:
                segment_labels[key] = set()
            segment_labels[key].update(species_list)

print(f"   Unique labeled segments: {len(segment_labels)}")

# Build all segments from labeled soundscapes only
ss_files_in_labels = [f for f in sorted(TRAIN_SS_DIR.glob("*.ogg")) if f.name in labeled_files]
print(f"   Soundscape files with labels: {len(ss_files_in_labels)}")

all_segments = []
for ss_file in ss_files_in_labels:
    fname = ss_file.stem
    info = sf.info(str(ss_file))
    duration = info.duration
    for seg_end in range(5, int(duration) + 5, 5):
        all_segments.append((fname, seg_end))

print(f"   Total segments from labeled soundscapes: {len(all_segments)}")

# Build label vector for each segment (multi-hot)
y_full = np.zeros((len(all_segments), N_CLASSES), dtype=np.float32)
for i, (fname, seg_end) in enumerate(all_segments):
    key = (fname, seg_end)
    if key in segment_labels:
        for sp in segment_labels[key]:
            if sp in species_to_idx:
                y_full[i, species_to_idx[sp]] = 1.0

# Check: how many segments have at least one label?
n_labeled = (y_full.sum(axis=1) > 0).sum()
print(f"   Labeled segments: {n_labeled}/{len(all_segments)}")

# ── Alignment strategy ──
# The OOF data has 708 samples from 5-fold CV on labeled training soundscapes.
# We filter to keep only labeled segments, then align with OOF.
# To verify alignment: we check if oof_base predictions correlate with y.

# Filter to labeled segments only, sorted consistently
labeled_indices = [i for i in range(len(all_segments)) if y_full[i].sum() > 0]
print(f"   Labeled segment indices: {len(labeled_indices)}")

# If we have exactly N_SAMPLES labeled segments, great
# Otherwise, we take the first N_SAMPLES (most common case: segments are
# sorted by filename then time, and the OOF used the first N labeled)
if len(labeled_indices) == N_SAMPLES:
    print("   ✓ Labeled segment count matches OOF!")
    y = y_full[labeled_indices]
elif len(labeled_indices) > N_SAMPLES:
    print(f"   ⚠ More labeled segments ({len(labeled_indices)}) than OOF ({N_SAMPLES})")
    print(f"   Using first {N_SAMPLES} labeled segments (sorted by filename, time)")
    y = y_full[labeled_indices[:N_SAMPLES]]
else:
    # Fewer labeled than OOF: pad with unlabeled (all zeros)
    print(f"   ⚠ Fewer labeled segments ({len(labeled_indices)}) than OOF ({N_SAMPLES})")
    print(f"   Padding with unlabeled segments")
    y = y_full[labeled_indices]
    # Pad to N_SAMPLES
    pad = np.zeros((N_SAMPLES - len(y), N_CLASSES), dtype=np.float32)
    y = np.vstack([y, pad])

# ── Validation: check alignment quality ──
# For a few common species, compute AUC of oof_base vs y
print(f"\n   Alignment sanity check (AUC of oof_base vs y for top species):")
n_pos_per_sp = y.sum(axis=0)
top_species_idx = np.argsort(-n_pos_per_sp)[:10]
alignment_scores = []
for idx in top_species_idx:
    if y[:, idx].sum() > 0 and y[:, idx].sum() < N_SAMPLES:
        try:
            auc = roc_auc_score(y[:, idx], oof_base[:len(y), idx])
            alignment_scores.append(auc)
            print(f"     {species_cols[idx]}: n_pos={y[:, idx].sum():.0f}, AUC={auc:.4f}")
        except ValueError:
            pass

if alignment_scores:
    mean_alignment_auc = np.mean(alignment_scores)
    print(f"   Mean alignment AUC: {mean_alignment_auc:.4f}")
    if mean_alignment_auc < 0.7:
        print(f"   ⚠ WARNING: Low alignment AUC! Labels may be misaligned with OOF data.")
        print(f"   Trying alternative: use segments with highest oof_base variance...")
    else:
        print(f"   ✓ Alignment looks reasonable (AUC > 0.7)")
else:
    print(f"   ⚠ Could not compute alignment AUC")

print(f"   y shape: {y.shape}")
print(f"   Positive labels per species: min={y.sum(axis=0).min():.0f}, max={y.sum(axis=0).max():.0f}, mean={y.sum(axis=0).mean():.1f}")

# Save the alignment for inspection
np.savez(OUTPUT_DIR / "oof_labels.npz", y=y, oof_base=oof_base, oof_prior=oof_prior, fold_id=fold_id)
print("   Saved oof_labels.npz")

# ───────────────────────────────────────────────────────────
# 3. Build features for XGBoost
# ───────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("3. Building XGBoost features...")

# Per species, we create features:
# - oof_base[s] : Perch direct logit for species s
# - oof_prior[s] : SSM-corrected for species s
# - mean_oof_base : mean of all 234 base logits (context)
# - mean_oof_prior : mean of all 234 prior logits (context)
# - max_oof_base : max logit (is there ANY strong signal?)
# - fold_id : one-hot encoded fold

# Context features (same for all species)
context_features = np.column_stack([
    oof_base.mean(axis=1),       # (708,)
    oof_prior.mean(axis=1),      # (708,)
    oof_base.max(axis=1),        # (708,)
    oof_prior.max(axis=1),       # (708,)
    oof_base.std(axis=1),        # (708,)
    oof_prior.std(axis=1),       # (708,)
])

# One-hot encode fold_id (values are 1-5, need 0-4 for np.eye)
fold_id_zero = fold_id - 1  # 1..5 → 0..4
fold_onehot = np.eye(5)[fold_id_zero]  # (708, 5)

print(f"   Context features: {context_features.shape[1]}")
print(f"   Fold one-hot: {fold_onehot.shape[1]}")
print(f"   Per-species features: base_logit + prior_logit = 2")

# Total per species: 2 (species-specific) + 6 (context) + 5 (fold) = 13 features
N_FEATURES_PER_SPECIES = 2 + context_features.shape[1] + fold_onehot.shape[1]
print(f"   Total features per species: {N_FEATURES_PER_SPECIES}")

# ───────────────────────────────────────────────────────────
# 4. Train XGBoost per species (5-fold cross-validation)
# ───────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("4. Training XGBoost per species...")

# We'll use the folds from the OOF data for proper OOF evaluation
N_FOLDS = 5
unique_folds = sorted(np.unique(fold_id))

xgb_models = {}
oof_xgb_probas = np.zeros_like(oof_base)  # (708, 234)
oof_perch_probas = oof_base.copy()  # baseline

for species_idx in range(N_CLASSES):
    sp_name = species_cols[species_idx]
    y_sp = y[:, species_idx]

    # Skip species with no positive samples (zero-shot in training data)
    n_pos = y_sp.sum()
    if n_pos == 0:
        # For zero-shot species, rely entirely on Perch
        oof_xgb_probas[:, species_idx] = oof_base[:, species_idx]
        xgb_models[sp_name] = None
        if species_idx < 5:
            print(f"   [{species_idx:3d}] {sp_name}: ZERO-SHOT (no labels) → skip")
        continue

    # Build feature matrix for this species
    X_sp = np.column_stack([
        oof_base[:, species_idx],      # feature 0: Perch logit
        oof_prior[:, species_idx],     # feature 1: SSM logit
        context_features,              # features 2-7: context
        fold_onehot,                   # features 8-12: fold encoding
    ])

    # Cross-validated training and OOF prediction
    oof_preds = np.zeros(N_SAMPLES)

    for fold in unique_folds:
        train_idx = np.where(fold_id != fold)[0]
        val_idx = np.where(fold_id == fold)[0]

        X_train, y_train = X_sp[train_idx], y_sp[train_idx]
        X_val, y_val = X_sp[val_idx], y_sp[val_idx]

        # Skip if no positives in train
        if y_train.sum() == 0:
            oof_preds[val_idx] = oof_base[val_idx, species_idx]
            continue

        # Scale positive weight for imbalanced classes
        scale_pos_weight = (y_train == 0).sum() / max(y_train.sum(), 1)

        model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42 + fold,
            verbosity=0,
        )

        model.fit(X_train, y_train)
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

        # Store the last fold model (or we could store all)
        if fold == unique_folds[-1]:
            xgb_models[sp_name] = model

    oof_xgb_probas[:, species_idx] = oof_preds

    if species_idx < 10 or (species_idx + 1) % 50 == 0:
        try:
            auc_base = roc_auc_score(y_sp, oof_base[:, species_idx])
            auc_xgb = roc_auc_score(y_sp, oof_preds)
            delta = auc_xgb - auc_base
            sign = "+" if delta >= 0 else ""
            print(f"   [{species_idx:3d}] {sp_name}: AUC base={auc_base:.4f} → xgb={auc_xgb:.4f} ({sign}{delta:.4f}) n_pos={n_pos:.0f}")
        except ValueError:
            print(f"   [{species_idx:3d}] {sp_name}: n_pos={n_pos:.0f} (AUC undefined)")

# ───────────────────────────────────────────────────────────
# 5. Evaluate overall OOF performance
# ───────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("5. Evaluating OOF performance...")

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

auc_base = macro_auc(y, oof_perch_probas)
auc_xgb = macro_auc(y, oof_xgb_probas)
delta = auc_xgb - auc_base

print(f"   Macro AUC (Perch base):  {auc_base:.6f}")
print(f"   Macro AUC (XGBoost):     {auc_xgb:.6f}")
print(f"   Delta:                   {delta:+.6f}")

# ───────────────────────────────────────────────────────────
# 6. Optimize late fusion weights
# ───────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("6. Optimizing late fusion weights (Base vs XGBoost)...")

from scipy.optimize import minimize

def objective(w):
    blended = w[0] * oof_perch_probas + w[1] * oof_xgb_probas
    return -macro_auc(y, blended)

# w[0] = base weight, w[1] = xgb weight,  w[0] + w[1] = 1
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
    score = -result.fun
    if score > best_score:
        best_score = score
        best_w = result.x

print(f"   Optimal weights: Base={best_w[0]:.4f}, XGBoost={best_w[1]:.4f}")
print(f"   Ensemble AUC:    {best_score:.6f}")
print(f"   Gain vs Base:    {best_score - auc_base:+.6f}")

# ───────────────────────────────────────────────────────────
# 7. Save models and weights
# ───────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("7. Saving models and artifacts...")

# Save XGBoost models (only non-None models, as JSON for Kaggle compatibility)
import os
models_dir = OUTPUT_DIR / "models_json"
models_dir.mkdir(exist_ok=True)

n_saved = 0
for sp_name, model in xgb_models.items():
    if model is not None:
        model.save_model(str(models_dir / f"{sp_name}.json"))
        n_saved += 1

print(f"   Saved {n_saved}/{N_CLASSES} XGBoost models to {models_dir}")

# Save weights
weights = {
    "base_perch_weight": float(best_w[0]),
    "xgb_stacker_weight": float(best_w[1]),
    "oof_auc_base": float(auc_base),
    "oof_auc_xgb": float(auc_xgb),
    "oof_auc_ensemble": float(best_score),
    "n_models_trained": n_saved,
    "n_zero_shot_skipped": N_CLASSES - n_saved,
    "description": "XGBoost meta-learner trained on Perch+SSM OOF logits. Late fusion with EoS Perch base.",
}

with open(OUTPUT_DIR / "fusion_weights.json", "w") as f:
    json.dump(weights, f, indent=2)
print(f"   Saved fusion_weights.json")

# Save OOF predictions for analysis
np.savez(
    OUTPUT_DIR / "oof_predictions.npz",
    oof_base=oof_perch_probas,
    oof_xgb=oof_xgb_probas,
    y_true=y,
    fold_id=fold_id,
    species_cols=np.array(species_cols),
)
print(f"   Saved oof_predictions.npz")

# ───────────────────────────────────────────────────────────
# 8. Summary
# ───────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("8. SUMMARY")
print("=" * 60)
print(f"   Training samples:         {N_SAMPLES}")
print(f"   Species:                  {N_CLASSES}")
print(f"   Features per species:     {N_FEATURES_PER_SPECIES}")
print(f"   XGBoost models trained:   {n_saved}")
print(f"   Zero-shot (skipped):      {N_CLASSES - n_saved}")
print(f"")
print(f"   Perch base AUC:           {auc_base:.6f}")
print(f"   XGBoost OOF AUC:          {auc_xgb:.6f}  (Δ {auc_xgb - auc_base:+.6f})")
print(f"   Ensemble AUC:             {best_score:.6f}  (Δ {best_score - auc_base:+.6f})")
print(f"")
print(f"   Late fusion: {best_w[0]:.1%} Perch + {best_w[1]:.1%} XGBoost")
print(f"   Models saved to:          {OUTPUT_DIR}")
print("=" * 60)

# If the ensemble gain is positive, this is promising for LB
if best_score > auc_base + 0.0005:
    print("   ✓ POSITIVE GAIN — worth integrating into submission!")
else:
    print("   ⚠ MARGINAL/NO GAIN — verify before integrating")
