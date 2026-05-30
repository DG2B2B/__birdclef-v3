"""
BirdCLEF 2026 – Ensemble Blend Notebook
========================================
Fusionne les prédictions de multiples sources :
  - Perch direct logits (mapped to BirdCLEF 234 species)
  - Classifieur SSM entraîné sur embeddings Perch
  - Modèles publics (tonylica/birdclef-2026-model, etc.)
  
Optimise les poids d'ensemble via scipy.optimize sur validation OOF.
Applique la calibration Platt par espèce (optionnelle).

Input: Fichiers de prédictions OOF (out-of-fold) pour chaque modèle
Output: weights.json + calibrators.pkl + soumission fusionnée
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════
# CELL 1: Configuration
# ══════════════════════════════════════════════════════════════════

class CFG:
    OUTPUT_DIR = "/kaggle/working"
    SAMPLE_SUB = "/kaggle/input/birdclef-2026/sample_submission.csv"

    # Sources de prédictions (chemins vers les .npy OOF)
    # Chaque fichier contient (N_val, 234) probabilités
    PRED_SOURCES = {
        "perch_direct": "/kaggle/working/oof_perch_direct.npy",
        "ssm_classifier": "/kaggle/working/oof_ssm_classifier.npy",
        # "public_model_1": "/kaggle/input/birdclef-2026-model/oof_model1.npy",
    }

    # Labels OOF correspondants
    OOF_LABELS = "/kaggle/working/oof_labels.npy"

    # Calibration
    DO_CALIBRATION = True

cfg = CFG()


# ══════════════════════════════════════════════════════════════════
# CELL 2: Load Predictions
# ══════════════════════════════════════════════════════════════════

print("=" * 60)
print("Loading prediction sources...")
print("=" * 60)

sample_sub = pd.read_csv(cfg.SAMPLE_SUB)
SPECIES_COLS = [c for c in sample_sub.columns if c != "row_id"]
N_SPECIES = len(SPECIES_COLS)

all_preds = {}
for name, path in cfg.PRED_SOURCES.items():
    if os.path.exists(path):
        preds = np.load(path)
        all_preds[name] = preds
        print(f"  [{name}] shape={preds.shape}, range=[{preds.min():.4f}, {preds.max():.4f}]")
    else:
        print(f"  [{name}] NOT FOUND: {path}")

if not all_preds:
    raise ValueError("No prediction files found!")

# Load labels
if os.path.exists(cfg.OOF_LABELS):
    y_true = np.load(cfg.OOF_LABELS)
    print(f"\n  Labels: shape={y_true.shape}")
else:
    print("\n  [WARN] No OOF labels — skipping weight optimization")
    y_true = None


# ══════════════════════════════════════════════════════════════════
# CELL 3: Weight Optimization
# ══════════════════════════════════════════════════════════════════

def macro_auc_loss(weights, preds_dict, y_true):
    """Negative macro AUC for optimization.
    
    Args:
        weights: Array of weights (will be normalized to sum=1).
        preds_dict: Dict of name → (N, C) predictions.
        y_true: (N, C) ground truth labels.
    
    Returns:
        Negative macro AUC (to minimize).
    """
    weights = np.maximum(weights, 0)  # non-negative
    weights = weights / weights.sum()  # normalize
    
    # Weighted average
    ensemble = np.zeros_like(list(preds_dict.values())[0])
    for w, preds in zip(weights, preds_dict.values()):
        ensemble += w * preds
    
    # Macro AUC
    aucs = []
    for c in range(N_SPECIES):
        if y_true[:, c].sum() == 0 or y_true[:, c].sum() == len(y_true):
            continue
        try:
            aucs.append(roc_auc_score(y_true[:, c], ensemble[:, c]))
        except ValueError:
            pass
    
    return -np.mean(aucs) if aucs else 0.0


if y_true is not None and len(all_preds) > 1:
    print("\n" + "=" * 60)
    print("Optimizing ensemble weights...")
    print("=" * 60)
    
    n_models = len(all_preds)
    initial_weights = np.ones(n_models) / n_models
    
    # Bounds: weights >= 0
    bounds = [(0, None)] * n_models
    
    # Constraint: sum(weights) = 1
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    
    result = minimize(
        macro_auc_loss,
        initial_weights,
        args=(all_preds, y_true),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    
    opt_weights = np.maximum(result.x, 0)
    opt_weights = opt_weights / opt_weights.sum()
    
    print(f"\n  Optimized weights:")
    for name, w in zip(all_preds.keys(), opt_weights):
        print(f"    {name}: {w:.4f}")
    
    # Evaluate
    ensemble_probs = np.zeros_like(list(all_preds.values())[0])
    for w, preds in zip(opt_weights, all_preds.values()):
        ensemble_probs += w * preds
    
    final_auc = -macro_auc_loss(opt_weights, all_preds, y_true)
    print(f"\n  Ensemble macro AUC: {final_auc:.4f}")
    
    # Individual AUCs for comparison
    print(f"\n  Individual AUCs:")
    for name, preds in all_preds.items():
        ind_auc = -macro_auc_loss(np.array([1.0]), {name: preds}, y_true)
        print(f"    {name}: {ind_auc:.4f}")
    
else:
    print("\n[INFO] Single model or no labels — using uniform weights")
    opt_weights = np.ones(len(all_preds)) / len(all_preds)


# ══════════════════════════════════════════════════════════════════
# CELL 4: Calibration (Platt Scaling per species)
# ══════════════════════════════════════════════════════════════════

calibrators = None

if cfg.DO_CALIBRATION and y_true is not None:
    print("\n" + "=" * 60)
    print("Calibrating predictions (Platt scaling per species)...")
    print("=" * 60)
    
    # Build ensemble predictions for calibration
    ensemble_probs = np.zeros_like(list(all_preds.values())[0])
    for w, preds in zip(opt_weights, all_preds.values()):
        ensemble_probs += w * preds
    
    calibrators = []
    improved = 0
    degraded = 0
    
    for c in range(N_SPECIES):
        # Skip species with too few positives
        if y_true[:, c].sum() < 5:
            calibrators.append(None)
            continue
        
        try:
            cal = CalibratedClassifierCV(
                method="sigmoid", cv=3
            )
            # Reshape for sklearn: (N, 1)
            cal.fit(ensemble_probs[:, c:c+1], y_true[:, c])
            calibrators.append(cal)
            
            # Check if calibration helps
            cal_probs = cal.predict_proba(ensemble_probs[:, c:c+1])[:, 1]
            orig_auc = roc_auc_score(y_true[:, c], ensemble_probs[:, c])
            cal_auc = roc_auc_score(y_true[:, c], cal_probs)
            
            if cal_auc > orig_auc + 1e-5:
                improved += 1
            elif cal_auc < orig_auc - 1e-5:
                degraded += 1
                calibrators[-1] = None  # Don't use if it hurts
        
        except Exception:
            calibrators.append(None)
    
    print(f"  Calibrated: {sum(1 for c in calibrators if c is not None)}/{N_SPECIES}")
    print(f"  Improved: {improved}, Degraded: {degraded}")

    # Save calibrators
    cal_path = os.path.join(cfg.OUTPUT_DIR, "calibrators.pkl")
    with open(cal_path, "wb") as f:
        pickle.dump(calibrators, f)
    print(f"  Saved to {cal_path}")


# ══════════════════════════════════════════════════════════════════
# CELL 5: Save Ensemble Artifacts
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Saving ensemble artifacts...")
print("=" * 60)

# Weights
weights_dict = {
    "models": list(all_preds.keys()),
    "weights": opt_weights.tolist(),
    "n_models": len(all_preds),
}
weights_path = os.path.join(cfg.OUTPUT_DIR, "ensemble_weights.json")
with open(weights_path, "w") as f:
    json.dump(weights_dict, f, indent=2)
print(f"  Weights: {weights_path}")

# Calibrators (already saved above)

# Ensemble config
config = {
    "n_species": N_SPECIES,
    "models": list(all_preds.keys()),
    "weights": opt_weights.tolist(),
    "calibration": cfg.DO_CALIBRATION,
    "n_calibrated": sum(1 for c in calibrators if c is not None) if calibrators else 0,
}
config_path = os.path.join(cfg.OUTPUT_DIR, "ensemble_config.json")
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

print(f"\n{'=' * 60}")
print(f"ENSEMBLE READY")
print(f"{'=' * 60}")
print(f"Models: {list(all_preds.keys())}")
print(f"Weights: {dict(zip(all_preds.keys(), opt_weights.round(4)))}")
if y_true is not None:
    print(f"Ensemble AUC: {final_auc:.4f}")
