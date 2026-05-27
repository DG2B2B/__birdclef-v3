#!/bin/bash
# resume.sh — Crash recovery: download latest checkpoints and resume
set -e

REPO="$HOME/__birdclef-v3"
REMOTE="gdrive:BirdCLEF2026/checkpoints/"

cd $REPO

echo "=== BirdCLEF 2026 — Crash Recovery ==="
echo "Date: $(date)"

# 1. Download all checkpoints from cloud
echo "[1/3] Downloading checkpoints from $REMOTE..."
rclone copy "$REMOTE/models/" models/ --progress

# 2. Download data artifacts
echo "[2/3] Downloading data artifacts..."
rclone copy "$REMOTE/data/" data/ --progress

# 3. Check progress
echo "[3/3] Checking which folds are complete..."
python scripts/check_progress.py

echo ""
echo "=== Recovery complete ==="
echo "Review scripts/check_progress.py output to see remaining jobs."
echo "Then run: bash scripts/launch_a6000_jobs.sh"
