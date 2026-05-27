"""
check_progress.py — Scan models/ to determine training completion status.
Generates report of completed / in-progress / remaining jobs.
"""

import os
import json
import torch
from pathlib import Path
from datetime import datetime

REPO = Path(os.environ.get("REPO", os.path.expanduser("~/__birdclef-v3")))
MODELS_DIR = REPO / "models"
DONE_DIR = REPO / ".done"

# Expected jobs definition
ALL_JOBS = {
    "Phase 3 — Baseline teachers": [
        ("efficientnet_b0", f"efficientnet_b0_fold{i}.pth", 5.3, "Kaggle K1/K2/Colab")
        for i in range(5)
    ] + [
        ("efficientnet_b3", f"efficientnet_b3_fold{i}.pth", 12.2, "Kaggle K3/Colab")
        for i in range(5)
    ] + [
        ("se_resnext50", f"se_resnext50_fold{i}.pth", 27.5, "A6000")
        for i in range(5)
    ] + [
        ("nfnet_f0", f"nfnet_f0_fold{i}.pth", 71.5, "A6000")
        for i in range(5)
    ],
    "Phase 6 iter 1 — PL": [
        (f"teachers_pl/pl_iter1/efficientnet_b0", f"efficientnet_b0_fold{i}_pl.pth", 5.3, "Kaggle")
        for i in range(5)
    ] + [
        (f"teachers_pl/pl_iter1/efficientnet_b3", f"efficientnet_b3_fold{i}_pl.pth", 12.2, "Kaggle")
        for i in range(5)
    ] + [
        (f"teachers_pl/pl_iter1/se_resnext50", f"se_resnext50_fold{i}_pl.pth", 27.5, "A6000")
        for i in range(5)
    ] + [
        (f"teachers_pl/pl_iter1/nfnet_f0", f"nfnet_f0_fold{i}_pl.pth", 71.5, "A6000")
        for i in range(5)
    ],
    "Phase 6 iter 2-4 — PL": [],  # dynamic
    "Phase 7 — Rare species": [
        ("teachers/rare_b0", f"rare_b0_fold{i}.pth", 5.3, "Kaggle")
        for i in range(5)
    ],
    "Phase 8 — Students": [
        ("students/efficientnet_b0", f"efficientnet_b0_fold{i}.pth", 5.3, "Kaggle")
        for i in range(5)
    ] + [
        ("students/efficientvit_b0", f"efficientvit_b0_fold{i}.pth", 3.5, "Kaggle")
        for i in range(5)
    ] + [
        ("students/mnasnet_100", f"mnasnet_100_fold{i}.pth", 4.4, "Kaggle")
        for i in range(5)
    ],
}


def check_checkpoint(path: Path) -> str:
    """Check if a checkpoint file is valid and return its status."""
    if not path.exists():
        return "MISSING"
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        epoch = ckpt.get("epoch", "?")
        auc = ckpt.get("val_auc", "?")
        size_mb = path.stat().st_size / 1e6
        return f"DONE (epoch={epoch}, AUC={auc:.4f}, {size_mb:.0f}MB)" if isinstance(auc, float) else f"DONE (epoch={epoch})"
    except Exception as e:
        return f"CORRUPT ({e})"


def main():
    print("=" * 70)
    print(f"BirdCLEF 2026 — Training Progress Report")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    grand_total = 0
    grand_done = 0

    for phase_name, jobs in ALL_JOBS.items():
        if not jobs:
            continue

        done_count = 0
        print(f"\n── {phase_name} ({len(jobs)} jobs) ──")

        for subdir, filename, params_m, route in jobs:
            ckpt_path = MODELS_DIR / subdir / filename
            status = check_checkpoint(ckpt_path)
            marker = "[OK]" if status.startswith("DONE") else "[  ]"
            print(f"  {marker} {subdir}/{filename} ({params_m:.0f}M) [{route}] → {status}")

            if status.startswith("DONE"):
                done_count += 1

        grand_total += len(jobs)
        grand_done += done_count
        pct = 100 * done_count / len(jobs) if jobs else 0
        print(f"  → {done_count}/{len(jobs)} complete ({pct:.0f}%)")

    print(f"\n{'=' * 70}")
    print(f"OVERALL: {grand_done}/{grand_total} jobs complete ({100*grand_done/grand_total:.0f}%)")

    # Check data artifacts
    print(f"\n── Data artifacts ──")
    for f in ["folds.pkl", "norm_stats.pkl"]:
        path = REPO / "data" / f
        status = "FOUND" if path.exists() else "MISSING"
        print(f"  [{status}] data/{f}")

    # Pseudo-label CSVs
    pl_dir = REPO / "data" / "pseudo_labels"
    if pl_dir.exists():
        for csv_file in sorted(pl_dir.glob("*.csv")):
            print(f"  [FOUND] data/pseudo_labels/{csv_file.name} ({csv_file.stat().st_size/1e6:.1f}MB)")

    # Remaining jobs
    remaining = grand_total - grand_done
    if remaining > 0:
        print(f"\n⚠️  {remaining} jobs remaining. Run launch scripts to resume.")
    else:
        print(f"\n🎉  All jobs complete! Ready for fusion + ONNX + submission.")


if __name__ == "__main__":
    main()
