"""
BirdCLEF 2026 – Prétraitement Audio
Resampling, segmentation, silence filtering, and audio cleaning.
"""

import librosa
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import warnings
import soundfile as sf

warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

DEFAULT_SR = 32000
DEFAULT_DURATION = 5  # seconds
SILENCE_THRESHOLD_DB = -40  # dB


# ──────────────────────────────────────────────
# Core functions
# ──────────────────────────────────────────────

def load_audio(file_path: str | Path, sr: int = DEFAULT_SR) -> np.ndarray:
    """Load audio file, convert to mono, resample to `sr`.

    Args:
        file_path: Path to .ogg or .mp3 file.
        sr: Target sample rate (default 32 000 Hz).

    Returns:
        1D numpy array (float32).
    """
    y, orig_sr = librosa.load(str(file_path), sr=sr, mono=True)
    return y.astype(np.float32)


def segment_audio(
    y: np.ndarray,
    sr: int = DEFAULT_SR,
    duration: float = DEFAULT_DURATION,
    overlap: float = 0.0,
) -> List[np.ndarray]:
    """Split audio into fixed-duration segments.

    Args:
        y: Audio signal (1D).
        sr: Sample rate.
        duration: Segment length in seconds (default 5 s).
        overlap: Overlap between consecutive segments in seconds (0 = no overlap).

    Returns:
        List of segments as 1D numpy arrays.
    """
    seg_samples = int(sr * duration)
    step = int(seg_samples * (1 - overlap))
    segments = []

    for start in range(0, len(y) - seg_samples + 1, step):
        seg = y[start : start + seg_samples]
        segments.append(seg)

    return segments


def is_silent(y: np.ndarray, threshold_db: float = SILENCE_THRESHOLD_DB) -> bool:
    """Return True if the maximum amplitude of the segment is below threshold."""
    if len(y) == 0:
        return True
    max_val = np.max(np.abs(y))
    if max_val == 0:
        return True
    max_db = 20 * np.log10(max_val + 1e-10)
    return max_db < threshold_db


def filter_silence(
    segments: List[np.ndarray], threshold_db: float = SILENCE_THRESHOLD_DB
) -> List[np.ndarray]:
    """Remove segments whose max amplitude is below `threshold_db`."""
    return [seg for seg in segments if not is_silent(seg, threshold_db)]


def resample_and_segment(
    audio_path: str | Path,
    sr: int = DEFAULT_SR,
    duration: float = DEFAULT_DURATION,
    overlap: float = 0.0,
    remove_silence: bool = True,
    silence_threshold_db: float = SILENCE_THRESHOLD_DB,
) -> List[np.ndarray]:
    """Load, resample, segment, and optionally filter silent segments.

    Args:
        audio_path: Path to the audio file (.ogg or .mp3).
        sr: Target sample rate.
        duration: Segment duration in seconds.
        overlap: Overlap fraction (0 to 1).
        remove_silence: If True, drop silent segments.
        silence_threshold_db: dB threshold for silence detection.

    Returns:
        List of 1D numpy arrays (segments).
    """
    y = load_audio(audio_path, sr)
    segments = segment_audio(y, sr, duration, overlap)
    if remove_silence:
        segments = filter_silence(segments, silence_threshold_db)
    return segments


# ──────────────────────────────────────────────
# Batch processing helpers
# ──────────────────────────────────────────────

def process_train_audio(
    train_audio_dir: str | Path,
    output_dir: str | Path,
    sr: int = DEFAULT_SR,
    duration: float = DEFAULT_DURATION,
    max_files_per_class: Optional[int] = None,
) -> Path:
    """Process all training audio: segment each file, save as .npy.

    Args:
        train_audio_dir: Root directory containing subdirs per species.
        output_dir: Where to save segmented .npy files.
        sr: Sample rate.
        duration: Segment length in seconds.
        max_files_per_class: Cap on number of source files per class.

    Returns:
        Path to the generated segments CSV.
    """
    train_audio_dir = Path(train_audio_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    species_dirs = sorted([d for d in train_audio_dir.iterdir() if d.is_dir()])

    for sp_dir in species_dirs:
        species_id = sp_dir.name
        audio_files = sorted(sp_dir.glob("*.ogg")) + sorted(sp_dir.glob("*.mp3"))
        if max_files_per_class:
            audio_files = audio_files[:max_files_per_class]

        for af in audio_files:
            try:
                segments = resample_and_segment(
                    af, sr=sr, duration=duration, remove_silence=True
                )
            except Exception as e:
                print(f"[WARN] Skipping {af}: {e}")
                continue

            for i, seg in enumerate(segments):
                out_name = f"{species_id}_{af.stem}_{i:03d}.npy"
                out_path = output_dir / out_name
                np.save(out_path, seg)
                rows.append(
                    {
                        "filepath": str(out_path),
                        "primary_label": species_id,
                        "source_file": str(af),
                        "segment_idx": i,
                    }
                )

    # Save CSV
    import pandas as pd

    csv_path = output_dir / "segments.csv"
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} segments to {csv_path}")
    return csv_path


def process_soundscape(
    file_path: str | Path,
    sr: int = DEFAULT_SR,
    duration: float = DEFAULT_DURATION,
) -> List[np.ndarray]:
    """Process a single soundscape: load, segment (no silence removal).

    Soundscapes must preserve all segments to keep row_id alignment.
    """
    y = load_audio(file_path, sr)
    # No silence removal for soundscapes — we need every 5 s window
    return segment_audio(y, sr, duration, overlap=0.0)


# ──────────────────────────────────────────────
# Metadata inspection
# ──────────────────────────────────────────────

def inspect_audio(file_path: str | Path) -> dict:
    """Return duration, sample rate, and peak dB for a single audio file."""
    y, orig_sr = librosa.load(str(file_path), sr=None, mono=True)
    duration_s = len(y) / orig_sr
    max_val = np.max(np.abs(y))
    peak_db = 20 * np.log10(max_val + 1e-10)
    return {
        "path": str(file_path),
        "original_sr": orig_sr,
        "duration_s": round(duration_s, 2),
        "peak_db": round(peak_db, 1),
    }


if __name__ == "__main__":
    # Quick smoke test
    import sys

    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        segs = resample_and_segment(test_file)
        print(f"{test_file}: {len(segs)} segments of {DEFAULT_DURATION}s")
    else:
        print("Usage: python preprocessing.py <audio_file>")
