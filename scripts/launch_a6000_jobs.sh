#!/bin/bash
# launch_a6000_jobs.sh — Lance tous les jobs A6000 avec parallelisme
# Chaque job tourne dans un tmux window dedie.
# Usage: bash scripts/launch_a6000_jobs.sh

set -e
SESSION="birdclef"
REPO="$HOME/__birdclef-v3"

cd $REPO
source ~/miniconda3/etc/profile.d/conda.sh
conda activate birdclef

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4

# ── Kill existing session if any ──
tmux kill-session -t $SESSION 2>/dev/null || true
tmux new-session -d -s $SESSION -n monitor

# ── Helper: launch a training job in a tmux window ──
launch_job() {
    local window_name=$1
    local backbone=$2
    local fold=$3
    local extra_args=${4:-""}

    tmux new-window -t $SESSION -n "$window_name" \
        "echo '[$(date)] Starting $backbone fold $fold' && \
         python src/train_teachers.py \
            --backbone $backbone \
            --fold $fold \
            --epochs 80 \
            --batch_size 64 \
            --lr 1e-3 \
            $extra_args \
            2>&1 | tee logs/${backbone}_fold${fold}.log && \
         echo '[$(date)] DONE $backbone fold $fold' && \
         touch .done/${backbone}_fold${fold}.done"
}

mkdir -p logs .done

echo "=== Launching A6000 Phase 3 Jobs ==="

# ═══ PHASE 3: Baseline Teachers ═══
# Vague 1: SE-ResNeXt50 (fold 0 + fold 1) + NFNet-F0 (fold 0 + fold 1)
#       = 4 jobs en parallele (VRAM: 4 × ~10-14G = ~48G → OK)
echo "[WAVE 1] SE + NF folds 0-1"

launch_job "SE_f0" "se_resnext50" 0
tmux select-window -t $SESSION:1
sleep 5
launch_job "SE_f1" "se_resnext50" 1
sleep 5
launch_job "NF_f0" "nfnet_f0" 0
sleep 5
launch_job "NF_f1" "nfnet_f0" 1
sleep 5

echo "[WAVE 1] Launched 4 jobs. Waiting for slots..."
echo "Monitor: tmux attach -t $SESSION"
echo "Logs:    tail -f logs/se_resnext50_fold0.log"

# ═══ Check completion and launch next waves ═══
# This script runs indefinitely, launching next jobs when slots free up

declare -A REMAINING_FOLDS
REMAINING_FOLDS["se_resnext50"]="2 3 4"
REMAINING_FOLDS["nfnet_f0"]="2 3 4"

ACTIVE_JOBS=4
MAX_JOBS=4  # 4 concurrent pour A6000 48GB

while [ $ACTIVE_JOBS -gt 0 ] || [ ${#REMAINING_FOLDS[@]} -gt 0 ]; do
    # Check for completed jobs
    for arch in "${!REMAINING_FOLDS[@]}"; do
        folds=${REMAINING_FOLDS[$arch]}
        new_folds=""
        for f in $folds; do
            if [ -f ".done/${arch}_fold${f}.done" ]; then
                ACTIVE_JOBS=$((ACTIVE_JOBS - 1))
                echo "[$(date)] $arch fold $f DONE (active: $ACTIVE_JOBS/$MAX_JOBS)"
            else
                new_folds="$new_folds $f"
            fi
        done
        REMAINING_FOLDS[$arch]=$new_folds
    done

    # Launch new jobs if slots available
    while [ $ACTIVE_JOBS -lt $MAX_JOBS ]; do
        launched=0
        for arch in "${!REMAINING_FOLDS[@]}"; do
            folds=${REMAINING_FOLDS[$arch]}
            if [ -n "$folds" ]; then
                next_fold=$(echo $folds | awk '{print $1}')
                REMAINING_FOLDS[$arch]=$(echo $folds | cut -d' ' -f2-)
                launch_job "${arch:0:4}_f${next_fold}" "$arch" "$next_fold"
                ACTIVE_JOBS=$((ACTIVE_JOBS + 1))
                launched=1
                break
            fi
        done
        if [ $launched -eq 0 ]; then
            break  # no more jobs to launch
        fi
        sleep 5
    done

    sleep 30  # Check every 30 seconds
done

echo ""
echo "=== All A6000 baseline jobs complete ==="
echo "Ready for Phase 6 pseudo-labeling."
echo "Run: bash scripts/launch_pseudo_labeling.sh"
