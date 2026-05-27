"""
BirdCLEF 2026 – Submission Notebook (Inference-Only, CPU)
=========================================================
This is the final notebook that runs on Kaggle CPU.
It loads ONNX models, processes test soundscapes, and generates submission.csv.

Expected runtime: < 60 minutes on 8-core CPU with 16 GB RAM.
"""

# # ==============================================================
# #  CELL 1: Imports & Configuration                             # 
# # ==============================================================

import os
os.environ["KAGGLE_NO_INTERNET"] = "1"

import numpy as np
import pandas as pd
import onnxruntime as ort
import librosa
import pickle
import json
import gc
import time
from pathlib import Path

# ── Paths (Kaggle environment) ──
TEST_DIR = "/kaggle/input/birdclef-2026/test_soundscapes"
MODEL_DIR = "/kaggle/input/birdclef-2026-model"  # your uploaded dataset
SAMPLE_SUB = "/kaggle/input/birdclef-2026/sample_submission.csv"
OUTPUT_PATH = "/kaggle/working/submission.csv"

# ── Audio params ──
SR = 32000
SEGMENT_LENGTH = 5  # seconds
N_MELS = 224
HOP_LENGTH = 320
WIN_LENGTH = 800
N_FFT = 4096
FMIN = 0
FMAX = 16000
TOP_DB = 80.0
TARGET_FRAMES = 224

# ── Inference params ──
BATCH_SIZE = 64
USE_SLIDING_WINDOW = True   # overlap for smoother predictions
SLIDING_STEP = 2.5           # seconds (50% overlap)
SMOOTHING_KERNEL = np.array([0.1, 0.2, 0.4, 0.2, 0.1])  # temporal smoothing

# # ==============================================================
# #  CELL 2: Load artifacts                                      # 
# # ==============================================================

print("Loading models and artifacts...")

# ONNX Runtime session options (CPU optimized)
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.intra_op_num_threads = 8
sess_options.inter_op_num_threads = 4
providers = ["CPUExecutionProvider"]

# Load ONNX models
model_files = sorted(Path(MODEL_DIR).glob("*.onnx"))
models = []
print(f"Found {len(model_files)} ONNX models: {[f.name for f in model_files]}")
for mf in model_files:
    sess = ort.InferenceSession(str(mf), sess_options, providers=providers)
    models.append(sess)

# Load ensemble weights
with open(f"{MODEL_DIR}/weights.json", "r") as f:
    weight_data = json.load(f)
weights = np.array(weight_data["weights"], dtype=np.float32)

# Load normalization stats
with open(f"{MODEL_DIR}/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
NORM_MEAN = scaler["mean"]
NORM_STD = scaler["std"]

# Load calibrators (optional)
calibrators = None
cal_path = Path(MODEL_DIR) / "calibrators.pkl"
if cal_path.exists():
    with open(cal_path, "rb") as f:
        calibrators = pickle.load(f)
    print("Calibrators loaded")

# Load sample submission to get species list
sample_sub = pd.read_csv(SAMPLE_SUB)
SPECIES_COLS = [c for c in sample_sub.columns if c != "row_id"]
N_SPECIES = len(SPECIES_COLS)
print(f"Species: {N_SPECIES}")

# # ==============================================================
# #  CELL 3: Feature extraction                                  # 
# # ==============================================================

def extract_logmel(y: np.ndarray) -> np.ndarray:
    """Extract normalized log-mel spectrogram → (3, 224, 224)."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=N_MELS,
        hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        n_fft=N_FFT, fmin=FMIN, fmax=FMAX,
    )
    logmel = librosa.power_to_db(mel, ref=np.max, top_db=TOP_DB)

    # Pad or truncate to target_frames
    if logmel.shape[1] >= TARGET_FRAMES:
        logmel = logmel[:, :TARGET_FRAMES]
    else:
        pad_w = TARGET_FRAMES - logmel.shape[1]
        logmel = np.pad(logmel, ((0, 0), (0, pad_w)), mode="constant")

    # Normalize
    logmel = (logmel - NORM_MEAN) / (NORM_STD + 1e-8)

    # 3-channel
    return np.stack([logmel] * 3, axis=0).astype(np.float32)


# # ==============================================================
# #  CELL 4: Inference                                           # 
# # ==============================================================

def predict_soundscape(file_path: str) -> np.ndarray:
    """Predict probabilities for all 5s windows in a soundscape.

    Returns:
        (N_segments, N_SPECIES) array of probabilities.
    """
    y, _ = librosa.load(file_path, sr=SR, mono=True)
    total_samples = len(y)

    segments = []
    if USE_SLIDING_WINDOW:
        step_samples = int(SR * SLIDING_STEP)
        for start in range(0, total_samples - SR * SEGMENT_LENGTH + 1, step_samples):
            seg = y[start : start + SR * SEGMENT_LENGTH]
            feat = extract_logmel(seg)
            segments.append(feat)
    else:
        for start in range(0, total_samples, SR * SEGMENT_LENGTH):
            seg = y[start : start + SR * SEGMENT_LENGTH]
            if len(seg) < SR * SEGMENT_LENGTH:
                break
            feat = extract_logmel(seg)
            segments.append(feat)

    if not segments:
        return np.zeros((0, N_SPECIES), dtype=np.float32)

    segments = np.stack(segments)  # (N, 3, 224, 224)
    all_probs = []

    for i in range(0, len(segments), BATCH_SIZE):
        batch = segments[i : i + BATCH_SIZE]
        batch_probs = np.zeros((len(batch), N_SPECIES), dtype=np.float32)

        for w, sess in zip(weights, models):
            out = sess.run(None, {"input": batch})[0]  # (B, T, C) or (B, C)
            if out.ndim == 3:
                out = out.mean(axis=1)  # aggregate frame predictions
            batch_probs += w * out

        # Apply calibration if available
        if calibrators is not None:
            for c, cal in enumerate(calibrators):
                if cal is not None:
                    batch_probs[:, c] = cal.predict_proba(
                        batch_probs[:, c:c+1]
                    )[:, 1]

        all_probs.append(batch_probs)

    probs = np.concatenate(all_probs, axis=0)

    # Temporal smoothing
    if USE_SLIDING_WINDOW and len(SMOOTHING_KERNEL) > 1 and len(probs) > len(SMOOTHING_KERNEL):
        from scipy.ndimage import convolve1d
        kernel = SMOOTHING_KERNEL / SMOOTHING_KERNEL.sum()
        smoothed = np.zeros_like(probs)
        for c in range(N_SPECIES):
            smoothed[:, c] = convolve1d(probs[:, c], kernel, mode="reflect")
        probs = smoothed

    return probs


# # ==============================================================
# #  CELL 5: Main inference loop                                 # 
# # ==============================================================

print("Starting inference on test soundscapes...")
t_start = time.time()

test_files = sorted([f for f in os.listdir(TEST_DIR) if f.endswith(".ogg")])
print(f"Found {len(test_files)} test soundscapes")

results = []

for idx, file in enumerate(test_files):
    file_path = os.path.join(TEST_DIR, file)
    soundscape_id = os.path.splitext(file)[0]

    probs = predict_soundscape(file_path)

    for seg_idx in range(probs.shape[0]):
        end_time = (seg_idx + 1) * SEGMENT_LENGTH
        row_id = f"{soundscape_id}_{end_time}"
        row = [row_id] + probs[seg_idx].tolist()
        results.append(row)

    # Progress
    if (idx + 1) % 10 == 0:
        elapsed = time.time() - t_start
        print(f"  [{idx+1}/{len(test_files)}] {file} ({probs.shape[0]} segments) - {elapsed:.0f}s")
        gc.collect()

# # ==============================================================
# #  CELL 6: Generate submission.csv                             # 
# # ==============================================================

print(f"\nBuilding submission DataFrame ({len(results)} rows)...")
sub = pd.DataFrame(results, columns=["row_id"] + SPECIES_COLS)

# Align with sample_submission (fill missing row_ids with 0)
sub = sample_sub[["row_id"]].merge(sub, on="row_id", how="left").fillna(0.0)

# Validate
assert sub.shape == sample_sub.shape, f"Shape mismatch: {sub.shape} vs {sample_sub.shape}"
assert set(sub.columns) == set(sample_sub.columns), "Column mismatch"
assert sub.isnull().sum().sum() == 0, "NaN values detected"
assert (sub.drop("row_id", axis=1) >= 0).all().all(), "Negative values detected"

# Save
sub.to_csv(OUTPUT_PATH, index=False)
print(f"[OK] Submission saved: {OUTPUT_PATH}")
print(f"     Shape: {sub.shape}")
print(f"     Total time: {(time.time() - t_start):.0f}s ({(time.time() - t_start)/60:.1f} min)")

# # ==============================================================
# #  CELL 7: Verification                                        # 
# # ==============================================================

# Quick sanity checks
sub_check = pd.read_csv(OUTPUT_PATH)
print(f"\nVerification:")
print(f"  Rows:    {len(sub_check)}")
print(f"  Columns: {len(sub_check.columns)}")
print(f"  NaN:     {sub_check.isnull().sum().sum()}")
print(f"  Min:     {sub_check.drop('row_id', axis=1).min().min():.6f}")
print(f"  Max:     {sub_check.drop('row_id', axis=1).max().max():.6f}")
print(f"  Mean:    {sub_check.drop('row_id', axis=1).mean().mean():.6f}")
print(f"\n[DONE] Ready to submit!")
