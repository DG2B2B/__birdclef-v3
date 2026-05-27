"""
Phase 6 – Iterative Pseudo-Labeling Pipeline
Run this on Kaggle GPU after training baseline teachers.

Usage:
    python notebooks/run_pseudo_labeling.py --iteration 1

Power schedule (from 1st place BirdCLEF 2025):
    Iter 1: power=1.0  (standard thresholding)
    Iter 2: power=0.65 (moderate cleaning)
    Iter 3: power=0.55 (strong cleaning)
    Iter 4: power=0.6  (fine-tune)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.pseudo_label import run_pseudo_label_iteration, POWER_SCHEDULE
from src.train_with_pl import train_fold_with_pl, PL_CONFIG

# ── Paths (adjust for Kaggle environment) ──
PATHS = {
    "soundscape_dir": "/kaggle/input/birdclef-2026/train_soundscapes",
    "original_csv": "data/train_segments/segments.csv",
    "norm_stats": "data/norm_stats.pkl",
    "sample_submission": "/kaggle/input/birdclef-2026/sample_submission.csv",
    "folds": "data/folds.pkl",
    "output_dir": "data/pseudo_labels",
    "teacher_models": [
        "models/teachers/efficientnet_b0/efficientnet_b0_fold0.pth",
        "models/teachers/efficientnet_b0/efficientnet_b0_fold1.pth",
        "models/teachers/efficientnet_b0/efficientnet_b0_fold2.pth",
        "models/teachers/efficientnet_b0/efficientnet_b0_fold3.pth",
        "models/teachers/efficientnet_b0/efficientnet_b0_fold4.pth",
    ],
}

# ── Species list ──
sample_sub = pd.read_csv(PATHS["sample_submission"])
SPECIES_LIST = [c for c in sample_sub.columns if c != "row_id"]


def run_full_iteration(iteration: int):
    """Run one complete pseudo-labeling iteration:
    1. Generate pseudo-labels on soundscapes
    2. Create enriched dataset
    3. Retrain all folds with Noisy Student
    """
    power_schedule = dict(POWER_SCHEDULE)
    power_val, threshold = power_schedule.get(iteration, (1.0, 0.9))

    print(f"\n{'#'*60}")
    print(f"# PSEUDO-LABEL ITERATION {iteration}")
    print(f"# Power={power_val}, Threshold={threshold}")
    print(f"{'#'*60}")

    # Step 1-2: Generate pseudo-labels + enrich dataset
    pseudo_csv, enriched_csv = run_pseudo_label_iteration(
        iteration=iteration,
        model_paths=PATHS["teacher_models"],
        soundscape_dir=PATHS["soundscape_dir"],
        original_csv=PATHS["original_csv"],
        norm_stats_path=PATHS["norm_stats"],
        species_list=SPECIES_LIST,
        output_dir=PATHS["output_dir"],
        power=power_val,
        threshold=threshold,
        original_ratio=0.5,
        batch_size=64,
    )

    # Step 3: Retrain all folds with Noisy Student
    config = PL_CONFIG.copy()
    config["norm_stats_path"] = PATHS["norm_stats"]
    config["folds_path"] = PATHS["folds"]
    config["sample_submission"] = PATHS["sample_submission"]
    config["output_dir"] = f"models/teachers_pl_iter{iteration}"
    config["drop_path_rate"] = 0.15  # Noisy Student

    # Use iteration-1 model weights as starting point
    if iteration > 1:
        prev_dir = f"models/teachers_pl_iter{iteration-1}/{config['backbone']}"
        config["pretrained_weights"] = f"{prev_dir}/{config['backbone']}_fold0_pl.pth"

    aucs = []
    for fold_idx in range(config["n_folds"]):
        auc = train_fold_with_pl(fold_idx, enriched_csv, config)
        aucs.append(auc)
        print(f"  Fold {fold_idx} AUC: {auc:.4f}")

    mean_auc = sum(aucs) / len(aucs)
    print(f"\nIteration {iteration} results: Mean AUC = {mean_auc:.4f}")
    for i, a in enumerate(aucs):
        print(f"  Fold {i}: {a:.4f}")

    return aucs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, required=True, choices=[1, 2, 3, 4])
    args = parser.parse_args()
    run_full_iteration(args.iteration)
