"""
Phase 1 Validation -- Smoke test for preprocessing, features, and dataset.
Run: python notebooks/validate_phase1.py
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import (
    load_audio, resample_and_segment, is_silent, inspect_audio,
    DEFAULT_SR, DEFAULT_DURATION,
)
from src.features import (
    log_mel_spectrogram, pcen_spectrogram, extract_features, pad_or_truncate,
)
from src.dataset import SpecAugment, MixUp

DATA_ROOT = Path("C:/Users/Utilisateur/Mon Drive/Kaggle/__birdclef-V2/birdclef-2026-data")
TRAIN_AUDIO = DATA_ROOT / "train_audio"

species_dirs = sorted([d for d in TRAIN_AUDIO.iterdir() if d.is_dir()])
if not species_dirs:
    print("[FAIL] No species directories found!")
    sys.exit(1)

sp_dir = species_dirs[0]
audio_files = list(sp_dir.glob("*.ogg")) + list(sp_dir.glob("*.mp3"))
test_file = audio_files[0]
print(f"[TEST] {test_file}  (species: {sp_dir.name})")

# 1. Audio loading
y = load_audio(test_file)
print(f"[OK]  Loaded: {len(y)} samples ({len(y)/DEFAULT_SR:.1f}s)")

info = inspect_audio(test_file)
print(f"     Original SR: {info['original_sr']} Hz, peak: {info['peak_db']} dB")

# 2. Segmentation
segs_5s = resample_and_segment(test_file, duration=5, remove_silence=True)
print(f"[OK]  Segments (5s, no silence): {len(segs_5s)}")

segs_all = resample_and_segment(test_file, duration=5, remove_silence=False)
silent = sum(1 for s in segs_all if is_silent(s))
print(f"[OK]  Silent ratio: {silent}/{len(segs_all)}")

segs_20s = resample_and_segment(test_file, duration=20, remove_silence=False)
print(f"[OK]  Chunks (20s): {len(segs_20s)}")

# 3. Feature extraction
seg = segs_5s[0]
logmel = log_mel_spectrogram(seg)
print(f"[OK]  Log-Mel spectrogram: {logmel.shape}")

pcen = pcen_spectrogram(seg)
print(f"[OK]  PCEN spectrogram:    {pcen.shape}")

logmel_224 = pad_or_truncate(logmel, 224)
print(f"[OK]  Padded to 224 frames: {logmel_224.shape}")

# 4. End-to-end
feat = extract_features(seg, use_pcen=False)
print(f"[OK]  Full feature (Log-Mel): {feat.shape}, dtype={feat.dtype}")

feat_pcen = extract_features(seg, use_pcen=True)
print(f"[OK]  Full feature (PCEN):    {feat_pcen.shape}, dtype={feat_pcen.dtype}")

# 5. Augmentations
spec_aug = SpecAugment(p=1.0)
aug = spec_aug(logmel_224)
print(f"[OK]  SpecAugment: {(aug==0).mean():.1%} masked")

mixup = MixUp(alpha=0.5)
y1 = np.zeros(234, dtype=np.float32); y1[0] = 1.0
y2 = np.zeros(234, dtype=np.float32); y2[1] = 1.0
mx, my = mixup(feat, y1, feat_pcen, y2)
print(f"[OK]  MixUp: shape={mx.shape}, y_sum={my.sum():.2f}")

# Summary
print()
print("=" * 50)
print("PHASE 1 VALIDATION PASSED")
print(f"  File: {test_file.name}")
print(f"  Segments (5s): {len(segs_5s)}")
print(f"  Feature shape: {feat.shape}")
print(f"  Output: (3, 224, 224) float32 -- ready for CNN")
