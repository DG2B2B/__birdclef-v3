"""
BirdCLEF 2026 – Validation & GroupKFold
Group-aware cross-validation to prevent noise leakage between train/val.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score


# ──────────────────────────────────────────────
# GroupKFold
# ──────────────────────────────────────────────

def create_folds(
    train_csv: str | Path,
    n_splits: int = 5,
    group_col: str = "author",
    loc_decimals: int = 1,
    seed: int = 42,
    output_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Create GroupKFold splits and save them.

    Groups = author + rounded_lat_lon to keep same recordist/location
    in the same fold, preventing background-noise leakage.

    Args:
        train_csv: Path to train.csv.
        n_splits: Number of folds (default 5).
        group_col: Column for recordist identity.
        loc_decimals: Decimal rounding for lat/lon grouping.
        seed: Random seed for GroupKFold.
        output_path: Where to save the folds pickle.

    Returns:
        DataFrame with added 'fold' column (0..4).
    """
    df = pd.read_csv(train_csv)

    # Build group key
    df["lat_round"] = df["latitude"].round(loc_decimals).astype(str)
    df["lon_round"] = df["longitude"].round(loc_decimals).astype(str)
    df["group"] = (
        df[group_col].astype(str) + "_" + df["lat_round"] + "_" + df["lon_round"]
    )

    gkf = GroupKFold(n_splits=n_splits)
    df["fold"] = -1

    for fold_idx, (_, val_idx) in enumerate(gkf.split(df, groups=df["group"])):
        df.loc[val_idx, "fold"] = fold_idx

    # Drop helper columns
    df = df.drop(columns=["lat_round", "lon_round", "group"])

    # Save
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump({"folds": df[["filename", "primary_label", "fold"]].copy(), "n_splits": n_splits, "seed": seed}, f)
        print(f"Folds saved to {output_path}")

    # Print fold stats
    for f in range(n_splits):
        fold_df = df[df["fold"] == f]
        print(f"  Fold {f}: {len(fold_df)} clips, {fold_df['primary_label'].nunique()} species")

    return df


def load_folds(path: str | Path) -> dict:
    """Load saved folds dict."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ──────────────────────────────────────────────
# Macro-AUC metric
# ──────────────────────────────────────────────

def macro_auc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Compute macro-averaged ROC-AUC.

    Args:
        y_true: Binary multi-label matrix (N, C).
        y_pred: Predicted probabilities (N, C).

    Returns:
        Macro AUC score (float).
    """
    n_classes = y_true.shape[1]
    aucs = []
    for c in range(n_classes):
        if y_true[:, c].sum() == 0:
            continue  # skip classes with no positives in this split
        try:
            aucs.append(roc_auc_score(y_true[:, c], y_pred[:, c]))
        except ValueError:
            aucs.append(0.5)
    return float(np.mean(aucs)) if aucs else 0.5


def per_class_auc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list[str]] = None,
) -> dict[str, float]:
    """Compute ROC-AUC for each class."""
    n_classes = y_true.shape[1]
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    results = {}
    for c in range(n_classes):
        name = class_names[c]
        if y_true[:, c].sum() == 0:
            results[name] = float("nan")
        else:
            try:
                results[name] = float(roc_auc_score(y_true[:, c], y_pred[:, c]))
            except ValueError:
                results[name] = 0.5
    return results


# ──────────────────────────────────────────────
# Replay protocol (fast testing)
# ──────────────────────────────────────────────

def create_replay_fold(
    train_csv: str | Path,
    n_classes_subset: int = 30,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a single train/val split on a subset of classes for fast iteration.

    Args:
        train_csv: Path to train.csv.
        n_classes_subset: Number of species to include.
        seed: Random seed.

    Returns:
        (train_df, val_df) DataFrames.
    """
    df = pd.read_csv(train_csv)
    rng = np.random.RandomState(seed)

    # Pick a random subset of species
    all_species = sorted(df["primary_label"].unique())
    subset = rng.choice(all_species, size=min(n_classes_subset, len(all_species)), replace=False)
    df_sub = df[df["primary_label"].isin(subset)].copy()

    # Split 80/20 by author groups
    groups = df_sub["author"].unique()
    rng.shuffle(groups)
    split = int(0.8 * len(groups))
    train_authors = set(groups[:split])
    val_authors = set(groups[split:])

    train_df = df_sub[df_sub["author"].isin(train_authors)]
    val_df = df_sub[df_sub["author"].isin(val_authors)]

    print(f"Replay fold: train={len(train_df)}, val={len(val_df)}, species={len(subset)}")
    return train_df, val_df


if __name__ == "__main__":
    print("Creating GroupKFold splits...")
    create_folds(
        "C:/Users/Utilisateur/Mon Drive/Kaggle/__birdclef-V2/birdclef-2026-data/train.csv",
        n_splits=5,
        output_path="data/folds.pkl",
    )
