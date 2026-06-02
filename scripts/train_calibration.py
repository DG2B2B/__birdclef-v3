"""
BirdCLEF 2026 — Species Calibration & Rare Boost
==================================================
Post-processes Perch/SSM probabilities with:
  1. Per-species Platt calibration (logistic regression on logits)
  2. Prevalence-based weight boost for rare species
  3. Phylogenetic prior for zero-shot species
  4. Outputs calibration config for submission notebook

Uses the 708 OOF samples (correctly aligned via EoS V2 logic).
Much more sample-efficient than XGBoost stacker.

Author: BirdCLEF 2026 Stacking Sprint
Date: 2026-05-31
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import json
import warnings
import re

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════
ROOT = Path(r"C:\Users\Utilisateur\Mon Drive\Kaggle\__birdclef-v3")
OOF_LABELS = ROOT / "models" / "xgb_stacker" / "oof_labels.npz"
TAXONOMY_CSV = Path(r"C:\Users\Utilisateur\Mon Drive\Kaggle\__birdclef-V2\birdclef-2026-data\taxonomy.csv")
SAMPLE_SUB = Path(r"C:\Users\Utilisateur\Mon Drive\Kaggle\__birdclef-V2\birdclef-2026-data\sample_submission.csv")
OUTPUT_DIR = ROOT / "models" / "calibration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════
# 1. Load aligned OOF data
# ═══════════════════════════════════════════════════════
print("=" * 60)
print("1. Loading aligned OOF data...")
data = np.load(OOF_LABELS, allow_pickle=True)
oof_base = data["oof_base"]     # (708, 234) Perch logits
oof_prior = data["oof_prior"]   # (708, 234) SSM-corrected
y = data["y"]                   # (708, 234) ground truth
fold_id = data["fold_id"]       # (708,) 1-indexed

N_SAMPLES, N_CLASSES = oof_base.shape
print(f"   {N_SAMPLES} samples x {N_CLASSES} species")

# Load species list
sample_sub = pd.read_csv(SAMPLE_SUB)
species_cols = sample_sub.columns[1:].tolist()

# Load taxonomy
taxonomy = pd.read_csv(TAXONOMY_CSV)
taxonomy_dict = taxonomy.set_index("primary_label").to_dict(orient="index")

# ═══════════════════════════════════════════════════════
# 2. Per-species statistics
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. Per-species statistics...")

species_stats = {}
for s_idx, sp_name in enumerate(species_cols):
    y_sp = y[:, s_idx]
    n_pos = int(y_sp.sum())
    n_neg = int((y_sp == 0).sum())

    # Base AUC
    base_auc = 0.5
    if n_pos > 0 and n_pos < N_SAMPLES:
        try:
            base_auc = roc_auc_score(y_sp, oof_base[:, s_idx])
        except ValueError:
            pass

    species_stats[sp_name] = {
        "index": s_idx,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "prevalence": n_pos / N_SAMPLES if N_SAMPLES > 0 else 0.0,
        "base_auc": float(base_auc),
        "is_zero_shot": n_pos == 0,
        "is_rare": 0 < n_pos < 10,
    }

# Count categories
n_zero_shot = sum(1 for s in species_stats.values() if s["is_zero_shot"])
n_rare = sum(1 for s in species_stats.values() if s["is_rare"])
n_common = N_CLASSES - n_zero_shot - n_rare
print(f"   Zero-shot (0 labels): {n_zero_shot}")
print(f"   Rare (< 10 labels):   {n_rare}")
print(f"   Common (≥ 10 labels):  {n_common}")

# ═══════════════════════════════════════════════════════
# 3. Per-species Platt calibration
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. Per-species Platt calibration...")

# For each species with sufficient labels, fit logistic regression
# on the Perch logit → probability
# This is a 1-feature model: p = sigmoid(a * logit + b)
calibrators = {}
oof_calibrated = np.zeros_like(oof_base)
unique_folds = sorted(np.unique(fold_id))

for s_idx, sp_name in enumerate(species_cols):
    stats = species_stats[sp_name]
    y_sp = y[:, s_idx]

    if stats["is_zero_shot"]:
        # Zero-shot: use Perch as-is
        oof_calibrated[:, s_idx] = oof_base[:, s_idx]
        calibrators[sp_name] = None
        continue

    if stats["n_pos"] < 5:
        # Too few samples for calibration, also slight temperature adjustment
        oof_calibrated[:, s_idx] = oof_base[:, s_idx]
        calibrators[sp_name] = None
        continue

    # 5-fold cross-validated calibration
    oof_cal = np.zeros(N_SAMPLES)
    lr_models = []

    for fold in unique_folds:
        train_idx = np.where(fold_id != fold)[0]
        val_idx = np.where(fold_id == fold)[0]

        X_tr = oof_base[train_idx, s_idx].reshape(-1, 1)
        y_tr = y_sp[train_idx]
        X_val = oof_base[val_idx, s_idx].reshape(-1, 1)

        if y_tr.sum() == 0 or (y_tr == 1).sum() == 0:
            oof_cal[val_idx] = oof_base[val_idx, s_idx]
            continue

        lr = LogisticRegression(C=10.0, max_iter=1000, class_weight="balanced")
        lr.fit(X_tr, y_tr)
        oof_cal[val_idx] = lr.predict_proba(X_val)[:, 1]
        lr_models.append(lr)

    oof_calibrated[:, s_idx] = oof_cal
    calibrators[sp_name] = lr_models[-1]  # store last fold model

    if s_idx < 5 or stats["base_auc"] < 0.65:
        cal_auc = roc_auc_score(y_sp, oof_cal) if stats["n_pos"] > 1 else 0.5
        delta = cal_auc - stats["base_auc"]
        print(f"   [{s_idx:3d}] {sp_name}: base={stats['base_auc']:.4f} → cal={cal_auc:.4f} (Δ{delta:+.4f}) n={stats['n_pos']}")

# ═══════════════════════════════════════════════════════
# 4. Prevalence-based weights
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4. Computing prevalence-based species weights...")

# Weight = 1/sqrt(count + 1), normalized
counts = np.array([species_stats[sp]["n_pos"] for sp in species_cols], dtype=np.float32)
raw_weights = 1.0 / np.sqrt(counts + 1.0)
# Normalize so mean weight = 1.0
species_weights = raw_weights / raw_weights.mean()

print(f"   Weight range: [{species_weights.min():.3f}, {species_weights.max():.3f}]")
print(f"   Top-10 boosted species:")
top_boosted = np.argsort(-species_weights)[:10]
for idx in top_boosted:
    sp = species_cols[idx]
    print(f"     {sp}: weight={species_weights[idx]:.3f}, n_pos={counts[idx]:.0f}")

# ═══════════════════════════════════════════════════════
# 5. Phylogenetic prior for zero-shot species
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("5. Phylogenetic prior for zero-shot species...")

# For each zero-shot species, find the closest relative with training data
# Strategy: same genus → same family → same order → same class

def find_closest_trained(sp_name, taxonomy_dict, species_stats):
    """Find the closest species with training data based on taxonomy."""
    if sp_name not in taxonomy_dict:
        return None, "no_taxonomy"

    tax = taxonomy_dict[sp_name]
    genus = tax.get("genus", "")
    family = tax.get("family", "")
    order = tax.get("order", "")
    class_name = tax.get("class_name", "")

    # First: try same genus
    same_genus = []
    for other_sp, other_tax in taxonomy_dict.items():
        if other_sp == sp_name:
            continue
        if other_tax.get("genus") == genus and species_stats[other_sp]["n_pos"] >= 5:
            same_genus.append(other_sp)

    if same_genus:
        # Pick the one with most training samples
        best = max(same_genus, key=lambda s: species_stats[s]["n_pos"])
        return best, f"same_genus({genus})"

    # Second: try same family
    same_family = []
    for other_sp, other_tax in taxonomy_dict.items():
        if other_sp == sp_name:
            continue
        if other_tax.get("family") == family and species_stats[other_sp]["n_pos"] >= 5:
            same_family.append(other_sp)

    if same_family:
        best = max(same_family, key=lambda s: species_stats[s]["n_pos"])
        return best, f"same_family({family})"

    return None, "no_close_relative"

# Build phylogenetic mapping
phylo_prior = {}
for sp_name in species_cols:
    stats = species_stats[sp_name]
    if stats["is_zero_shot"] or stats["n_pos"] < 3:
        closest, reason = find_closest_trained(sp_name, taxonomy_dict, species_stats)
        if closest is not None:
            phylo_prior[sp_name] = {
                "proxy_species": closest,
                "reason": reason,
                "proxy_n_pos": species_stats[closest]["n_pos"],
                "proxy_base_auc": species_stats[closest]["base_auc"],
            }

print(f"   Found phylogenetic proxies for {len(phylo_prior)} species:")
for sp_name, info in list(phylo_prior.items())[:10]:
    print(f"     {sp_name} → {info['proxy_species']} ({info['reason']})")

# ═══════════════════════════════════════════════════════
# 6. Evaluate
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("6. Evaluation...")

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

auc_base = macro_auc(y, oof_base)
auc_cal = macro_auc(y, oof_calibrated)

# Apply species weights to calibrated predictions
oof_weighted = oof_calibrated * species_weights[np.newaxis, :]
# Row-wise softmax-like normalization (keep in [0,1])
oof_weighted = oof_weighted / oof_weighted.sum(axis=1, keepdims=True) * oof_calibrated.sum(axis=1, keepdims=True)
oof_weighted = np.clip(oof_weighted, 0, 1)
auc_weighted = macro_auc(y, oof_weighted)

print(f"   Perch base AUC:           {auc_base:.6f}")
print(f"   Calibrated AUC:           {auc_cal:.6f}  (Δ {auc_cal - auc_base:+.6f})")
print(f"   Calibrated + weighted AUC: {auc_weighted:.6f}  (Δ {auc_weighted - auc_base:+.6f})")

# ═══════════════════════════════════════════════════════
# 7. Save artifacts for submission notebook
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("7. Saving artifacts...")

# Calibrator coefficients (intercept + coef for each species)
calibration_coeffs = {}
for sp_name, lr in calibrators.items():
    if lr is not None:
        calibration_coeffs[sp_name] = {
            "intercept": float(lr.intercept_[0]),
            "coef": float(lr.coef_[0][0]),
        }

config = {
    "calibration_coeffs": calibration_coeffs,
    "species_weights": {sp: float(w) for sp, w in zip(species_cols, species_weights)},
    "phylo_prior": phylo_prior,
    "species_stats": species_stats,
    "n_samples": N_SAMPLES,
    "n_classes": N_CLASSES,
    "description": "Per-species Platt calibration + prevalence weights + phylogenetic priors",
}

with open(OUTPUT_DIR / "calibration_config.json", "w") as f:
    json.dump(config, f, indent=2)
print(f"   Saved calibration_config.json ({len(calibration_coeffs)} calibrators, {len(species_weights)} weights)")

# Save the inference function as a Python module
inference_code = '''
"""BirdCLEF 2026 — Calibrated Inference Module
Import this in the submission notebook to apply calibration + species weights.
"""

import numpy as np
import json
from pathlib import Path
from scipy.special import expit  # sigmoid

def load_calibration(config_path: str | Path) -> dict:
    """Load calibration config from JSON file."""
    with open(config_path) as f:
        return json.load(f)

def apply_calibration(logits: np.ndarray, config: dict) -> np.ndarray:
    """Apply per-species Platt calibration to raw Perch logits.
    
    Args:
        logits: (N, 234) raw Perch logits
        config: Calibration config dict
    
    Returns:
        (N, 234) calibrated probabilities
    """
    probs = expit(logits)  # sigmoid conversion
    cal = config["calibration_coeffs"]
    
    for sp_name, coeffs in cal.items():
        idx = config["species_stats"][sp_name]["index"]
        # p = sigmoid(a * logit + b)
        a = coeffs["coef"]
        b = coeffs["intercept"]
        probs[:, idx] = expit(a * logits[:, idx] + b)
    
    return probs

def apply_species_weights(probs: np.ndarray, config: dict, boost_strength: float = 1.0) -> np.ndarray:
    """Apply prevalence-based species weights.
    
    Args:
        probs: (N, 234) probabilities
        config: Calibration config dict
        boost_strength: How strongly to apply weights (0=off, 1=full)
    
    Returns:
        (N, 234) adjusted probabilities
    """
    weights = np.array([config["species_weights"][sp] for sp in config["species_stats"]])
    # Interpolate between original (boost_strength=0) and weighted (boost_strength=1)
    effective_weights = 1.0 + boost_strength * (weights - 1.0)
    
    probs_adj = probs * effective_weights[np.newaxis, :]
    # Row-wise normalization to keep probabilities well-behaved
    row_sums = probs_adj.sum(axis=1, keepdims=True)
    row_sums_orig = probs.sum(axis=1, keepdims=True)
    # Blend normalization: keep original scale
    probs_adj = probs_adj / (row_sums + 1e-8) * (row_sums_orig + 1e-8)
    
    return np.clip(probs_adj, 0.0, 1.0)

def apply_phylo_prior(probs: np.ndarray, config: dict) -> np.ndarray:
    """Apply phylogenetic prior: for zero-shot species, use proxy species probability.
    
    This borrows the calibrated probability from the closest related species
    that has training data, scaled by a conservatism factor.
    
    Args:
        probs: (N, 234) probabilities
        config: Calibration config dict
    
    Returns:
        (N, 234) adjusted probabilities
    """
    phylo = config.get("phylo_prior", {})
    stats = config["species_stats"]
    
    for sp_name, info in phylo.items():
        idx_target = stats[sp_name]["index"]
        idx_proxy = stats[info["proxy_species"]]["index"]
        # Blend: 70% original + 30% proxy, to add signal without overriding
        proxy_prob = probs[:, idx_proxy]
        # Conservatism factor: scale down proxy probability
        conservatism = 0.7
        probs[:, idx_target] = 0.3 * probs[:, idx_target] + 0.7 * (proxy_prob * conservatism)
    
    return np.clip(probs, 0.0, 1.0)
'''

with open(OUTPUT_DIR / "calibrated_inference.py", "w") as f:
    f.write(inference_code)
print(f"   Saved calibrated_inference.py")

print("\n" + "=" * 60)
print("8. SUMMARY")
print("=" * 60)
print(f"   Samples: {N_SAMPLES} | Species: {N_CLASSES}")
print(f"   Calibrators trained: {len(calibration_coeffs)}")
print(f"   Phylogenetic proxies: {len(phylo_prior)}")
print(f"")
print(f"   Perch base AUC:           {auc_base:.6f}")
print(f"   + Calibration:            {auc_cal:.6f}  (Δ {auc_cal - auc_base:+.6f})")
print(f"   + Cal + Weights:          {auc_weighted:.6f}  (Δ {auc_weighted - auc_base:+.6f})")
print(f"")
print(f"   Artifacts saved to: {OUTPUT_DIR}")
print("=" * 60)
