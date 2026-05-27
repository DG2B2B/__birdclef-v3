"""
BirdCLEF 2026 – Fusion, Stacking & Calibration
Model ensembling, weight optimization, and Platt calibration.
Runs on CPU (numpy/scikit-learn only).
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from typing import Optional, List, Tuple
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score


# ──────────────────────────────────────────────
# Macro AUC
# ──────────────────────────────────────────────

def macro_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    n_classes = y_true.shape[1]
    aucs = []
    for c in range(n_classes):
        if y_true[:, c].sum() == 0:
            continue
        try:
            aucs.append(roc_auc_score(y_true[:, c], y_pred[:, c]))
        except ValueError:
            aucs.append(0.5)
    return float(np.mean(aucs)) if aucs else 0.5


# ──────────────────────────────────────────────
# Weighted average fusion
# ──────────────────────────────────────────────

def weighted_average(
    probs_list: List[np.ndarray],
    weights: np.ndarray,
) -> np.ndarray:
    """Ensemble predictions via weighted average.

    Args:
        probs_list: List of (N, C) probability arrays.
        weights: Array of shape (M,) summing to 1.

    Returns:
        (N, C) weighted average.
    """
    result = np.zeros_like(probs_list[0])
    for w, p in zip(weights, probs_list):
        result += w * p
    return result


def optimize_weights(
    probs_list: List[np.ndarray],
    y_true: np.ndarray,
    n_restarts: int = 10,
) -> np.ndarray:
    """Find optimal ensemble weights maximizing macro-AUC.

    Uses scipy.optimize with random restarts.
    Constraint: Σ w_i = 1, w_i ≥ 0.

    Args:
        probs_list: List of (N, C) OOF probability arrays from M models.
        y_true: (N, C) ground truth labels.
        n_restarts: Number of random initializations.

    Returns:
        Optimal weights (M,).
    """
    M = len(probs_list)
    best_weights = None
    best_score = -np.inf

    def objective(w):
        pred = weighted_average(probs_list, w)
        return -macro_auc(y_true, pred)

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0, 1)] * M

    for _ in range(n_restarts):
        w0 = np.random.dirichlet(np.ones(M))
        result = minimize(
            objective, w0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-8},
        )
        score = -result.fun
        if score > best_score:
            best_score = score
            best_weights = result.x

    print(f"Optimal weights: {np.round(best_weights, 4)}")
    print(f"Ensemble AUC:    {best_score:.4f}")

    return best_weights


# ──────────────────────────────────────────────
# Stacking (meta-model)
# ──────────────────────────────────────────────

def train_stacking_model(
    probs_list: List[np.ndarray],
    y_true: np.ndarray,
    C: float = 1.0,
) -> LogisticRegression:
    """Train a logistic regression meta-model on OOF predictions.

    Stacking: use each model's OOF predictions as features
    for a logistic regression that predicts the final label.

    IMPORTANT: Must use OOF predictions to avoid data leakage!

    Args:
        probs_list: List of (N, C) OOF probabilities.
        y_true: (N, C) binary labels.
        C: Regularization strength.

    Returns:
        Fitted LogisticRegression model.
    """
    N, C = y_true.shape
    # Features: concatenate all model predictions → (N, M*C)
    X = np.concatenate(probs_list, axis=1)  # (N, M*C)

    # Train one LR per class
    meta_models = []
    for c in range(C):
        if y_true[:, c].sum() == 0:
            # No positives for this class → predict 0
            lr = LogisticRegression(C=C, max_iter=1000)
            lr.classes_ = np.array([0, 1])
            lr.coef_ = np.zeros((1, X.shape[1]))
            lr.intercept_ = np.array([-10.0])  # always predict 0
            meta_models.append(lr)
            continue

        lr = LogisticRegression(C=C, max_iter=1000)
        lr.fit(X, y_true[:, c])
        meta_models.append(lr)

    # Wrap in a simple class
    class StackingEnsemble:
        def __init__(self, models):
            self.models = models

        def predict_proba(self, X):
            preds = []
            for lr in self.models:
                if hasattr(lr, "predict_proba"):
                    p = lr.predict_proba(X)[:, 1]
                else:
                    p = np.zeros(len(X))
                preds.append(p)
            return np.column_stack(preds)

    return StackingEnsemble(meta_models)


# ──────────────────────────────────────────────
# Platt Calibration
# ──────────────────────────────────────────────

def calibrate_platt(
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> CalibratedClassifierCV:
    """Calibrate predictions per class using Platt scaling.

    Args:
        y_pred: (N, C) uncalibrated probabilities.
        y_true: (N, C) binary labels.

    Returns:
        Fitted CalibratedClassifierCV.
    """
    # Platt scaling: fit a sigmoid per class
    # Using sklearn's CalibratedClassifierCV with method='sigmoid'
    calibrated = CalibratedClassifierCV(
        estimator=LogisticRegression(C=1.0, max_iter=1000),
        method="sigmoid",
        cv=3,
    )

    # Train one calibrator per class
    # For simplicity, we calibrate the final ensemble output
    calibrators = []
    for c in range(y_true.shape[1]):
        if y_true[:, c].sum() == 0:
            calibrators.append(None)
            continue
        cal = CalibratedClassifierCV(
            estimator=LogisticRegression(C=1.0, max_iter=1000),
            method="sigmoid",
            cv=3,
        )
        cal.fit(y_pred[:, c:c+1], y_true[:, c])
        calibrators.append(cal)

    return calibrators


def apply_calibration(
    y_pred: np.ndarray,
    calibrators: list,
) -> np.ndarray:
    """Apply per-class Platt calibration."""
    calibrated = np.zeros_like(y_pred)
    for c, cal in enumerate(calibrators):
        if cal is not None:
            calibrated[:, c] = cal.predict_proba(y_pred[:, c:c+1])[:, 1]
        else:
            calibrated[:, c] = y_pred[:, c]
    return calibrated


# ──────────────────────────────────────────────
# Save / Load
# ──────────────────────────────────────────────

def save_fusion_artifacts(
    weights: np.ndarray,
    output_dir: str | Path,
    calibrators: Optional[list] = None,
    model_names: Optional[List[str]] = None,
):
    """Save ensemble weights and calibrators."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Weights
    weights_dict = {
        "weights": weights.tolist(),
        "model_names": model_names or [f"model_{i}" for i in range(len(weights))],
    }
    with open(output_dir / "weights.json", "w") as f:
        json.dump(weights_dict, f, indent=2)

    # Calibrators
    if calibrators is not None:
        with open(output_dir / "calibrators.pkl", "wb") as f:
            pickle.dump(calibrators, f)

    print(f"Fusion artifacts saved to {output_dir}")


def load_fusion_artifacts(
    model_dir: str | Path,
) -> Tuple[np.ndarray, Optional[list]]:
    """Load weights and calibrators."""
    model_dir = Path(model_dir)
    with open(model_dir / "weights.json", "r") as f:
        data = json.load(f)
    weights = np.array(data["weights"])

    calibrators = None
    cal_path = model_dir / "calibrators.pkl"
    if cal_path.exists():
        with open(cal_path, "rb") as f:
            calibrators = pickle.load(f)

    return weights, calibrators


# ──────────────────────────────────────────────
# End-to-end fusion pipeline
# ──────────────────────────────────────────────

def run_fusion_pipeline(
    oof_probs: List[np.ndarray],  # List of (N, 234) OOF arrays
    y_true: np.ndarray,            # (N, 234) ground truth
    model_names: Optional[List[str]] = None,
    output_dir: str | Path = "models/fusion",
    use_stacking: bool = False,
    use_calibration: bool = True,
) -> dict:
    """Complete fusion pipeline.

    Args:
        oof_probs: OOF predictions from each student model.
        y_true: Ground truth labels.
        model_names: Names of each model.
        output_dir: Where to save artifacts.
        use_stacking: If True, use stacking instead of weighted average.
        use_calibration: If True, apply Platt calibration.

    Returns:
        Dict with results.
    """
    print(f"Fusion pipeline: {len(oof_probs)} models, {y_true.shape[1]} classes")

    # 1. Weighted average or stacking
    if use_stacking:
        print("Using stacking (logistic regression meta-model)")
        ensemble = train_stacking_model(oof_probs, y_true)
        X_meta = np.concatenate(oof_probs, axis=1)
        final_preds = ensemble.predict_proba(X_meta)
    else:
        print("Optimizing ensemble weights...")
        weights = optimize_weights(oof_probs, y_true)
        final_preds = weighted_average(oof_probs, weights)

    base_auc = macro_auc(y_true, final_preds)
    print(f"Base ensemble AUC: {base_auc:.4f}")

    # 2. Calibration
    calibrators = None
    if use_calibration:
        print("Applying Platt calibration...")
        calibrators = calibrate_platt(final_preds, y_true)
        final_preds_cal = apply_calibration(final_preds, calibrators)
        cal_auc = macro_auc(y_true, final_preds_cal)
        print(f"Calibrated AUC:    {cal_auc:.4f} (delta={cal_auc - base_auc:+.4f})")

        if cal_auc > base_auc:
            print("[OK] Calibration improves AUC, keeping it")
            final_preds = final_preds_cal
        else:
            print("[SKIP] Calibration degrades AUC, discarding")
            calibrators = None

    # 3. Save
    if not use_stacking:
        save_fusion_artifacts(weights, output_dir, calibrators, model_names)

    return {
        "auc": macro_auc(y_true, final_preds),
        "weights": weights if not use_stacking else None,
        "calibrators": calibrators,
    }


if __name__ == "__main__":
    print("Fusion module ready.")
    print("  - Weighted average with scipy.optimize")
    print("  - Stacking (logistic regression meta-model)")
    print("  - Platt calibration (per-class sigmoid)")
