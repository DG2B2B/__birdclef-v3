"""
Phase 1b — Generate Pseudo-Labels from Submission
===================================================
Filters high-confidence predictions from submission.csv to create
pseudo-labels for test-time self-training.

Usage:
  1. Upload eos_submit_v3.ipynb to Kaggle, run it, download submission.csv
  2. Run this script: python filter_pseudo_labels.py --input submission.csv
  3. Upload pseudo_labels_filtered.csv as Kaggle dataset "birdclef-pl-v1"

Filtering criteria (conservative):
  - max_prob > 0.95 (very high confidence)
  - gap between 1st and 2nd highest prob > 0.4 (no ambiguity)

Author: BirdCLEF 2026 Last Push
Date: 2026-05-31
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse

# ═══════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════
MAX_PROB_THRESHOLD = 0.95      # Keep only rows where max probability > this
GAP_THRESHOLD = 0.4            # Keep only rows where (1st - 2nd) > this
MIN_ROWS_TO_KEEP = 50          # If filtering yields fewer than this, relax thresholds

def filter_pseudo_labels(
    submission_path: str | Path,
    output_path: str | Path = "pseudo_labels_filtered.csv",
    max_prob_threshold: float = MAX_PROB_THRESHOLD,
    gap_threshold: float = GAP_THRESHOLD,
    min_rows: int = MIN_ROWS_TO_KEEP,
) -> pd.DataFrame:
    """Filter high-confidence predictions for pseudo-labeling.

    Args:
        submission_path: Path to submission.csv from Kaggle
        output_path: Where to save filtered pseudo-labels
        max_prob_threshold: Minimum max probability to keep
        gap_threshold: Minimum gap between 1st and 2nd highest
        min_rows: Minimum rows to keep (relaxes thresholds if needed)

    Returns:
        Filtered DataFrame in submission format
    """
    print(f"Loading {submission_path}...")
    df = pd.read_csv(submission_path)
    species_cols = [c for c in df.columns if c != "row_id"]
    n_total = len(df)
    n_species = len(species_cols)
    print(f"  {n_total} rows x {n_species} species")

    # Compute per-row statistics
    arr = df[species_cols].values.astype(np.float32)

    # Get top-2 probabilities and their indices per row
    # For efficiency, use partition instead of full sort
    top2_indices = np.argpartition(-arr, 1, axis=1)[:, :2]
    top2_values = np.take_along_axis(arr, top2_indices, axis=1)

    max_probs = top2_values[:, 0]
    second_probs = top2_values[:, 1]
    gaps = max_probs - second_probs

    # ── Filter ──
    mask = (max_probs > max_prob_threshold) & (gaps > gap_threshold)
    n_filtered = mask.sum()

    print(f"\n  Filtering with max_prob > {max_prob_threshold} AND gap > {gap_threshold}")
    print(f"  Kept: {n_filtered}/{n_total} rows ({100*n_filtered/n_total:.1f}%)")

    # Relax thresholds if too few rows
    if n_filtered < min_rows:
        print(f"\n  ⚠ Only {n_filtered} rows kept (min={min_rows}), relaxing thresholds...")
        relaxed_prob = max_prob_threshold
        relaxed_gap = gap_threshold
        while n_filtered < min_rows and relaxed_prob > 0.7:
            relaxed_prob -= 0.05
            relaxed_gap = max(0.1, relaxed_gap - 0.05)
            mask = (max_probs > relaxed_prob) & (gaps > relaxed_gap)
            n_filtered = mask.sum()
            print(f"    max_prob > {relaxed_prob:.2f}, gap > {relaxed_gap:.2f} → {n_filtered} rows")

    # ── Create filtered pseudo-labels ──
    # For high-confidence rows, keep ALL probabilities (not just the max)
    # This preserves the full probability distribution for multi-label training
    df_filtered = df[mask].copy()

    # ── Statistics ──
    print(f"\n  Final pseudo-labels: {len(df_filtered)} rows")
    print(f"  Max prob distribution: mean={max_probs[mask].mean():.4f}, "
          f"min={max_probs[mask].min():.4f}, max={max_probs[mask].max():.4f}")
    print(f"  Gap distribution: mean={gaps[mask].mean():.4f}, "
          f"min={gaps[mask].min():.4f}, max={gaps[mask].max():.4f}")

    # Count how many species are represented
    species_presence = (df_filtered[species_cols].values > 0.5).sum(axis=0)
    n_species_represented = (species_presence > 0).sum()
    print(f"  Species represented (>0.5 prob): {n_species_represented}/{n_species}")

    # ── Save ──
    df_filtered.to_csv(output_path, index=False)
    print(f"\n  ✓ Saved to {output_path}")

    return df_filtered

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter pseudo-labels from submission.csv")
    parser.add_argument("--input", "-i", default="submission.csv", help="Path to submission.csv")
    parser.add_argument("--output", "-o", default="pseudo_labels_filtered.csv", help="Output path")
    parser.add_argument("--max-prob", "-p", type=float, default=MAX_PROB_THRESHOLD, help="Max prob threshold")
    parser.add_argument("--gap", "-g", type=float, default=GAP_THRESHOLD, help="Gap threshold")
    args = parser.parse_args()

    filter_pseudo_labels(
        submission_path=args.input,
        output_path=args.output,
        max_prob_threshold=args.max_prob,
        gap_threshold=args.gap,
    )
