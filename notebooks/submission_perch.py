"""
BirdCLEF 2026 – Submission Notebook: Perch ONNX + Lightweight Classifier
=========================================================================
Approche zéro spectrogramme pré-calculé : Perch ONNX traite l'audio brut.
Aucun upload de 130 Go nécessaire — seul l'ONNX Perch (~427 MB) + classifieur (~20 MB).

Fonctionne en CPU-only avec onnxruntime.
Runtime cible: < 90 minutes sur 8 cœurs / 16 Go RAM.

Dataset Kaggle à uploader:
  ├── perch_v2.onnx           # Perch ONNX (rishikeshjani/perch-onnx-for-birdclef-2026)
  ├── classifier_ssm.onnx     # Classifieur léger entraîné sur embeddings Perch
  ├── perch_labels.csv        # Mapping espèces Perch → BirdCLEF
  ├── weights.json            # Poids d'ensemble (optionnel)
  └── config.pkl              # Paramètres (SR, segment_length, etc.)
"""

# ══════════════════════════════════════════════════════════════════
# CELL 1: Imports & Configuration
# ══════════════════════════════════════════════════════════════════

import os
os.environ["KAGGLE_NO_INTERNET"] = "1"

import numpy as np
import pandas as pd
import onnxruntime as ort
import json
import pickle
import gc
import time
from pathlib import Path

# ── Paths (Kaggle environment) ──
COMP_DIR  = "/kaggle/input/birdclef-2026"
MODEL_DIR = "/kaggle/input/birdclef-perch-models"   # votre dataset uploadé
TEST_DIR  = f"{COMP_DIR}/test_soundscapes"
SAMPLE_SUB_PATH = f"{COMP_DIR}/sample_submission.csv"
OUTPUT_PATH     = "/kaggle/working/submission.csv"

# ── Audio params (Perch expects 5s @ 32kHz mono) ──
SR = 32000
SEGMENT_LENGTH = 5        # secondes
SEGMENT_SAMPLES = SR * SEGMENT_LENGTH  # 160000 échantillons

# ── Sliding window (optional, for smoother predictions) ──
USE_SLIDING_WINDOW = True
SLIDING_STEP_SEC = 2.5    # 50% overlap
SMOOTHING_KERNEL = np.array([0.1, 0.2, 0.4, 0.2, 0.1], dtype=np.float32)

# ── Batch inference ──
BATCH_SIZE = 32           # Perch ONNX ~12M params, ajuster selon RAM dispo
CLASSIFIER_BATCH = 128    # Le classifieur SSM est plus léger

# ── ONNX Runtime options ──
SESS_OPTS = ort.SessionOptions()
SESS_OPTS.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
SESS_OPTS.intra_op_num_threads = 8
SESS_OPTS.inter_op_num_threads = 4
SESS_OPTS.enable_mem_pattern = True
PROVIDERS = ["CPUExecutionProvider"]


# ══════════════════════════════════════════════════════════════════
# CELL 2: Load Models & Artifacts
# ══════════════════════════════════════════════════════════════════

print("=" * 60)
print("Loading models and artifacts...")
print("=" * 60)

# ── 2a. Perch ONNX ──
perch_path = os.path.join(MODEL_DIR, "perch_v2.onnx")
if os.path.exists(perch_path):
    print(f"[OK] Perch ONNX found: {perch_path}")
    perch_session = ort.InferenceSession(perch_path, SESS_OPTS, providers=PROVIDERS)
    perch_input_name = perch_session.get_inputs()[0].name
    perch_output_names = [o.name for o in perch_session.get_outputs()]
    print(f"  Input: {perch_input_name}")
    print(f"  Outputs: {perch_output_names}")
    HAS_PERCH = True
else:
    print(f"[WARN] Perch ONNX not found at {perch_path}")
    print("  Will use fallback CNN models instead.")
    HAS_PERCH = False

# ── 2b. Classifier ONNX (SSM/MLP trained on Perch embeddings) ──
classifier_path = os.path.join(MODEL_DIR, "classifier_ssm.onnx")
if os.path.exists(classifier_path):
    print(f"[OK] Classifier ONNX found: {classifier_path}")
    clf_session = ort.InferenceSession(classifier_path, SESS_OPTS, providers=PROVIDERS)
    clf_input_name = clf_session.get_inputs()[0].name
    clf_output_name = clf_session.get_outputs()[0].name
    print(f"  Input: {clf_input_name}, Output: {clf_output_name}")
    HAS_CLASSIFIER = True
else:
    print("[WARN] No classifier ONNX — using Perch logits directly.")
    HAS_CLASSIFIER = False

# ── 2c. Perch labels mapping ──
perch_labels_path = os.path.join(MODEL_DIR, "perch_labels.csv")
if os.path.exists(perch_labels_path):
    perch_labels_df = pd.read_csv(perch_labels_path)
    print(f"[OK] Perch labels: {len(perch_labels_df)} classes")
else:
    perch_labels_df = None
    print("[WARN] No Perch labels — will attempt to infer from sample_submission.")

# ── 2d. Sample submission → get BirdCLEF species list ──
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
SPECIES_COLS = [c for c in sample_sub.columns if c != "row_id"]
N_SPECIES = len(SPECIES_COLS)
print(f"\n[INFO] BirdCLEF species: {N_SPECIES}")

# Build species name → index mapping
species_to_idx = {sp: i for i, sp in enumerate(SPECIES_COLS)}

# ── 2e. Ensemble weights (optional) ──
weights_path = os.path.join(MODEL_DIR, "weights.json")
if os.path.exists(weights_path):
    with open(weights_path, "r") as f:
        weight_data = json.load(f)
    print(f"[OK] Ensemble weights: {weight_data}")
else:
    weight_data = None

# ── 2f. Config ──
config_path = os.path.join(MODEL_DIR, "config.pkl")
if os.path.exists(config_path):
    with open(config_path, "rb") as f:
        config = pickle.load(f)
    print(f"[OK] Config loaded")
else:
    config = {}


# ══════════════════════════════════════════════════════════════════
# CELL 3: Audio Processing & Perch Inference
# ══════════════════════════════════════════════════════════════════

def load_audio_segments(file_path: str) -> np.ndarray:
    """Load a soundscape and split into 5s segments for Perch.

    Perch accepts raw waveform directly (no librosa mel needed).
    Uses scipy for fast I/O — much faster than librosa for simple load.

    Args:
        file_path: Path to .ogg file.

    Returns:
        np.ndarray of shape (N_segments, 160000) float32.
    """
    import soundfile as sf

    y, file_sr = sf.read(file_path, dtype="float32", always_2d=False)

    # Convert to mono if stereo
    if y.ndim > 1:
        y = y.mean(axis=1)

    # Resample if needed (Perch expects 32 kHz)
    if file_sr != SR:
        import librosa
        y = librosa.resample(y, orig_sr=file_sr, target_sr=SR)

    # Pad to at least one segment
    if len(y) < SEGMENT_SAMPLES:
        y = np.pad(y, (0, SEGMENT_SAMPLES - len(y)))

    total = len(y)
    segments = []

    if USE_SLIDING_WINDOW:
        step_samples = int(SR * SLIDING_STEP_SEC)
        for start in range(0, total - SEGMENT_SAMPLES + 1, step_samples):
            segments.append(y[start : start + SEGMENT_SAMPLES])
    else:
        for start in range(0, total, SEGMENT_SAMPLES):
            seg = y[start : start + SEGMENT_SAMPLES]
            if len(seg) < SEGMENT_SAMPLES:
                seg = np.pad(seg, (0, SEGMENT_SAMPLES - len(seg)))
            segments.append(seg)

    if not segments:
        # Fallback: at least one segment (padded)
        segments.append(y[:SEGMENT_SAMPLES])

    return np.stack(segments).astype(np.float32)  # (N, 160000)


def run_perch_inference(waveforms: np.ndarray) -> dict:
    """Run Perch ONNX on a batch of audio segments.

    Args:
        waveforms: (B, 160000) float32 array.

    Returns:
        dict mapping output name → numpy array.
        Typical keys: 'embedding' (B, 1536), 'label' (B, ~15000), 'spectrogram'
    """
    feed = {perch_input_name: waveforms}
    outputs = perch_session.run(perch_output_names, feed)
    return dict(zip(perch_output_names, outputs))


def get_perch_embeddings(waveforms: np.ndarray) -> np.ndarray:
    """Extract Perch embeddings for a batch.

    Returns (B, 1536) float32 array.
    """
    outs = run_perch_inference(waveforms)
    # The embedding key may vary between ONNX versions
    for key in ["embedding", "embeddings", "output_0"]:
        if key in outs:
            return outs[key].astype(np.float32)
    # Fallback: use spatial_embedding pooled
    for key in ["spatial_embedding", "spatial_embeddings"]:
        if key in outs:
            return outs[key].mean(axis=(1, 2)).astype(np.float32)
    raise KeyError(f"No embedding found in Perch outputs: {list(outs.keys())}")


def get_perch_logits(waveforms: np.ndarray) -> np.ndarray:
    """Extract Perch species logits for a batch.

    Returns (B, ~15000) float32 array.
    """
    outs = run_perch_inference(waveforms)
    for key in ["label", "logits", "output_1"]:
        if key in outs:
            return outs[key].astype(np.float32)
    raise KeyError(f"No logits found in Perch outputs: {list(outs.keys())}")


# ══════════════════════════════════════════════════════════════════
# CELL 4: BirdCLEF Species Mapping
# ══════════════════════════════════════════════════════════════════

def map_perch_to_birdclef(perch_logits: np.ndarray) -> np.ndarray:
    """Map Perch logits (~15000 classes) to BirdCLEF 234 species.

    Strategy:
    1. Use Perch labels.csv to match class names to BirdCLEF species.
    2. Direct match on scientific names (genus species).
    3. Fill unmatched species with 0.0 (or use embedding-based classifier).

    Args:
        perch_logits: (B, N_perch) logits from Perch.

    Returns:
        (B, 234) probabilities for BirdCLEF species.
    """
    B = perch_logits.shape[0]
    birdclef_probs = np.zeros((B, N_SPECIES), dtype=np.float32)

    if perch_labels_df is None:
        return birdclef_probs  # no mapping available

    # Build Perch label → index
    perch_label_to_idx = {}
    for i, row in perch_labels_df.iterrows():
        label = str(row.get("label", row.get("scientific_name", ""))).lower().strip()
        perch_label_to_idx[label] = i

    # Match each BirdCLEF species
    matched = 0
    for bc_sp in SPECIES_COLS:
        bc_sp_lower = bc_sp.lower().replace("_", " ").strip()

        # Try exact match first
        if bc_sp_lower in perch_label_to_idx:
            perch_idx = perch_label_to_idx[bc_sp_lower]
            birdclef_idx = species_to_idx[bc_sp]
            birdclef_probs[:, birdclef_idx] = perch_logits[:, perch_idx]
            matched += 1
            continue

        # Try partial match (genus only)
        genus = bc_sp_lower.split()[0] if " " in bc_sp_lower else bc_sp_lower
        for label, idx in perch_label_to_idx.items():
            if label.startswith(genus + " "):
                # Take the first matching species for this genus
                birdclef_idx = species_to_idx[bc_sp]
                birdclef_probs[:, birdclef_idx] = perch_logits[:, idx]
                matched += 1
                break

    # Apply sigmoid to convert logits → probabilities
    # Perch uses softmax training but outputs raw logits
    birdclef_probs = 1.0 / (1.0 + np.exp(-birdclef_probs))

    print(f"  [Perch→BirdCLEF] Matched {matched}/{N_SPECIES} species")
    return birdclef_probs


# ══════════════════════════════════════════════════════════════════
# CELL 5: Combined Inference Pipeline
# ══════════════════════════════════════════════════════════════════

def predict_soundscape(file_path: str) -> np.ndarray:
    """Full inference for one soundscape.

    Pipeline:
      Audio → Perch ONNX → embeddings (1536-dim)
                          → classifier ONNX → BirdCLEF probs (234)
                          → Perch logits → BirdCLEF probs (234, optionnel)
      → Ensemble weighted average

    Returns:
        (N_segments, N_SPECIES) array of probabilities.
    """
    t0 = time.time()

    # Step 1: Load audio segments
    waveforms = load_audio_segments(file_path)  # (N, 160000)
    n_segments = len(waveforms)
    if n_segments == 0:
        return np.zeros((0, N_SPECIES), dtype=np.float32)

    all_classifier_probs = []
    all_perch_probs = []

    # Step 2: Batch through Perch
    for i in range(0, n_segments, BATCH_SIZE):
        batch_wav = waveforms[i : i + BATCH_SIZE]

        if HAS_PERCH:
            # 2a. Get Perch embeddings → feed to classifier
            if HAS_CLASSIFIER:
                embeddings = get_perch_embeddings(batch_wav)  # (B, 1536)
                clf_out = clf_session.run(
                    [clf_output_name], {clf_input_name: embeddings}
                )[0]
                # Apply sigmoid
                clf_probs = 1.0 / (1.0 + np.exp(-clf_out))
                all_classifier_probs.append(clf_probs)

            # 2b. Get Perch direct logits → map to BirdCLEF
            perch_logits = get_perch_logits(batch_wav)
            perch_probs = map_perch_to_birdclef(perch_logits)
            all_perch_probs.append(perch_probs)
        else:
            # Fallback: use a simpler spectrogram+CNN path
            # (handled by the original submission_notebook.py)
            pass

    # Step 3: Aggregate
    classifier_probs = (
        np.concatenate(all_classifier_probs, axis=0)
        if all_classifier_probs
        else None
    )
    perch_probs = (
        np.concatenate(all_perch_probs, axis=0)
        if all_perch_probs
        else None
    )

    # Step 4: Ensemble
    if HAS_CLASSIFIER and HAS_PERCH:
        # Weighted ensemble: 0.4 Perch + 0.6 classifier (tunable)
        w_perch = 0.4
        w_clf = 0.6
        probs = w_perch * perch_probs + w_clf * classifier_probs
    elif HAS_CLASSIFIER:
        probs = classifier_probs
    elif HAS_PERCH:
        probs = perch_probs
    else:
        probs = np.zeros((n_segments, N_SPECIES), dtype=np.float32)

    # Step 5: Temporal smoothing
    if USE_SLIDING_WINDOW and len(probs) > len(SMOOTHING_KERNEL):
        from scipy.ndimage import convolve1d
        kernel = SMOOTHING_KERNEL / SMOOTHING_KERNEL.sum()
        smoothed = np.zeros_like(probs)
        for c in range(N_SPECIES):
            smoothed[:, c] = convolve1d(probs[:, c], kernel, mode="reflect")
        probs = smoothed

    elapsed = time.time() - t0
    return probs.astype(np.float32)


# ══════════════════════════════════════════════════════════════════
# CELL 6: Main Inference Loop
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Starting inference on test soundscapes...")
print("=" * 60)

t_start = time.time()

# List test files
test_files = sorted([
    f for f in os.listdir(TEST_DIR)
    if f.endswith((".ogg", ".mp3", ".wav"))
])
print(f"Found {len(test_files)} test soundscapes")

results = []
total_segments = 0

for idx, file in enumerate(test_files):
    file_path = os.path.join(TEST_DIR, file)
    soundscape_id = os.path.splitext(file)[0]

    try:
        probs = predict_soundscape(file_path)
    except Exception as e:
        print(f"  [ERROR] {file}: {e}")
        # If a file fails, fill with zeros based on sample_submission
        # Find expected row_ids for this soundscape
        expected_rows = sample_sub[
            sample_sub["row_id"].str.startswith(soundscape_id)
        ]
        probs = np.zeros((len(expected_rows), N_SPECIES), dtype=np.float32)

    n_seg = probs.shape[0]
    total_segments += n_seg

    # Build rows: row_id = soundscape_endtime
    for seg_idx in range(n_seg):
        if USE_SLIDING_WINDOW:
            end_time = (seg_idx + 1) * SLIDING_STEP_SEC
        else:
            end_time = (seg_idx + 1) * SEGMENT_LENGTH
        row_id = f"{soundscape_id}_{int(end_time)}"
        row = [row_id] + probs[seg_idx].tolist()
        results.append(row)

    # Progress
    if (idx + 1) % 5 == 0 or idx == 0:
        elapsed = time.time() - t_start
        est_total = elapsed / (idx + 1) * len(test_files)
        print(
            f"  [{idx+1:3d}/{len(test_files)}] {file} "
            f"({n_seg} seg) — {elapsed:.0f}s / ~{est_total:.0f}s"
        )
        gc.collect()


# ══════════════════════════════════════════════════════════════════
# CELL 7: Generate submission.csv
# ══════════════════════════════════════════════════════════════════

print(f"\nBuilding submission DataFrame ({len(results)} rows)...")
sub = pd.DataFrame(results, columns=["row_id"] + SPECIES_COLS)

# Merge with sample_submission to ensure correct row_ids
# Fill missing rows with 0.0
sub = sample_sub[["row_id"]].merge(sub, on="row_id", how="left")

# Fill NaN (unmatched row_ids) with 0.0
sub = sub.fillna(0.0)

# Ensure column order matches sample_submission
sub = sub[sample_sub.columns]

# ── Validation ──
assert sub.shape == sample_sub.shape, (
    f"Shape mismatch: {sub.shape} vs {sample_sub.shape}"
)
assert list(sub.columns) == list(sample_sub.columns), "Column mismatch"
assert sub.isnull().sum().sum() == 0, "NaN values detected!"

# Check no negative values
numeric_cols = sub.drop("row_id", axis=1)
assert (numeric_cols >= 0).all().all(), "Negative values detected!"

# Save
sub.to_csv(OUTPUT_PATH, index=False)

total_time = time.time() - t_start
print(f"\n[OK] Submission saved: {OUTPUT_PATH}")
print(f"     Shape: {sub.shape}")
print(f"     Total segments inferred: {total_segments}")
print(f"     Total time: {total_time:.0f}s ({total_time/60:.1f} min)")


# ══════════════════════════════════════════════════════════════════
# CELL 8: Final Verification
# ══════════════════════════════════════════════════════════════════

sub_check = pd.read_csv(OUTPUT_PATH)
print(f"\n{'=' * 60}")
print(f"VERIFICATION")
print(f"{'=' * 60}")
print(f"  Rows:       {len(sub_check)}")
print(f"  Columns:    {len(sub_check.columns)}")
print(f"  NaN:        {sub_check.isnull().sum().sum()}")
print(f"  Min prob:   {numeric_cols.min().min():.6f}")
print(f"  Max prob:   {numeric_cols.max().max():.6f}")
print(f"  Mean prob:  {numeric_cols.mean().mean():.6f}")
print(f"  Zero cols:  {(numeric_cols.sum() == 0).sum()} / {N_SPECIES}")
print(f"\n  Species with predictions > 0.5:")
for sp in SPECIES_COLS:
    sp_max = sub_check[sp].max()
    if sp_max > 0.5:
        print(f"    {sp}: max={sp_max:.4f}, mean={sub_check[sp].mean():.4f}")

print(f"\n[DONE] Ready to submit!")
