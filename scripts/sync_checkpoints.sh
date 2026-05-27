#!/bin/bash
# sync_checkpoints.sh — Stream checkpoints to cloud storage every 5 min
# Add to crontab: */5 * * * * bash /home/user/__birdclef-v3/scripts/sync_checkpoints.sh

REPO="$HOME/__birdclef-v3"
REMOTE="gdrive:BirdCLEF2026/checkpoints/"  # Change to S3/Dropbox as needed
LOGFILE="$REPO/logs/sync.log"

mkdir -p "$REPO/logs"

echo "[$(date)] Syncing checkpoints..." >> $LOGFILE

# Sync all .pth files modified in last 10 minutes
find "$REPO/models/" -name "*.pth" -mmin -10 -type f | while read f; do
    relpath="${f#$REPO/}"
    rclone copyto "$f" "$REMOTE$relpath" --progress 2>> $LOGFILE
    echo "  Synced: $relpath" >> $LOGFILE
done

# Sync CSV pseudo-labels
find "$REPO/data/pseudo_labels/" -name "*.csv" -mmin -10 -type f | while read f; do
    relpath="${f#$REPO/}"
    rclone copyto "$f" "$REMOTE$relpath" 2>> $LOGFILE
done

# Sync folds and norm stats (once)
rclone copyto "$REPO/data/folds.pkl" "$REMOTE/data/folds.pkl" 2>> $LOGFILE
rclone copyto "$REPO/data/norm_stats.pkl" "$REMOTE/data/norm_stats.pkl" 2>> $LOGFILE

# Sync weights.json
if [ -f "$REPO/models/fusion/weights.json" ]; then
    rclone copyto "$REPO/models/fusion/weights.json" "$REMOTE/models/fusion/weights.json" 2>> $LOGFILE
fi

echo "[$(date)] Sync done" >> $LOGFILE
