"""
BirdCLEF 2026 – Feature Extraction
Log-Mel spectrograms, PCEN, and normalization.
"""

import numpy as np
import librosa
from pathlib import Path
from typing import Tuple, Optional


# ──────────────────────────────────────────────
# Spectrogram parameters (matching 1st place 2025)
# ──────────────────────────────────────────────

N_MELS = 224
HOP_LENGTH = 320   # 10 ms at 32 kHz
WIN_LENGTH = 800   # 25 ms at 32 kHz
N_FFT = 4096
FMIN = 0
FMAX = 16000
TOP_DB = 80.0


def log_mel_spectrogram(
    y: np.ndarray,
    sr: int = 32000,
    n_mels: int = N_MELS,
    hop_length: int = HOP_LENGTH,
    win_length: int = WIN_LENGTH,
    n_fft: int = N_FFT,
    fmin: float = FMIN,
    fmax: float = FMAX,
    top_db: float = TOP_DB,
) -> np.ndarray:
    """Compute log-mel spectrogram from raw audio.

    Args:
        y: 1D audio signal (float32).
        sr: Sample rate (Hz).
        n_mels: Number of mel bands.
        hop_length: Hop length in samples.
        win_length: Window length in samples.
        n_fft: FFT size.
        fmin: Minimum frequency (Hz).
        fmax: Maximum frequency (Hz).
        top_db: Threshold for dB clipping.

    Returns:
        Array of shape (n_mels, time_frames) in dB.
    """
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
        fmin=fmin,
        fmax=fmax,
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max, top_db=top_db)
    return log_mel.astype(np.float32)


def pcen_spectrogram(
    y: np.ndarray,
    sr: int = 32000,
    n_mels: int = N_MELS,
    hop_length: int = HOP_LENGTH,
    win_length: int = WIN_LENGTH,
    n_fft: int = N_FFT,
    fmin: float = FMIN,
    fmax: float = FMAX,
) -> np.ndarray:
    """Compute PCEN (Per-Channel Energy Normalization) spectrogram.

    PCEN helps handle constant background noise (wind, water in Pantanal).
    Recommended as alternative to log-mel for this competition.
    """
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
        fmin=fmin,
        fmax=fmax,
    )
    # PCEN parameters from librosa defaults
    pcen = librosa.pcen(
        mel_spec,
        sr=sr,
        hop_length=hop_length,
        gain=0.98,
        bias=2,
        power=0.5,
        time_constant=0.4,
        eps=1e-6,
    )
    return pcen.astype(np.float32)


# ──────────────────────────────────────────────
# Normalization
# ──────────────────────────────────────────────

def pad_or_truncate(spec: np.ndarray, target_frames: int = 224) -> np.ndarray:
    """Pad or truncate spectrogram time axis to exactly `target_frames`.

    Args:
        spec: (n_mels, time_frames) array.
        target_frames: Desired number of time frames.

    Returns:
        (n_mels, target_frames) array.
    """
    n_mels, t = spec.shape
    if t >= target_frames:
        return spec[:, :target_frames]
    else:
        pad_width = target_frames - t
        return np.pad(spec, ((0, 0), (0, pad_width)), mode="constant", constant_values=0)


def normalize_spectrogram(
    spec: np.ndarray, mean: float, std: float
) -> np.ndarray:
    """Apply (x - mean) / std normalization."""
    return (spec - mean) / (std + 1e-8)


def to_3channel(spec_2d: np.ndarray) -> np.ndarray:
    """Duplicate a (224, 224) spectrogram to 3 channels → (3, 224, 224).

    This makes it compatible with ImageNet-pretrained CNNs.
    """
    return np.stack([spec_2d] * 3, axis=0).astype(np.float32)


# ──────────────────────────────────────────────
# End-to-end extraction
# ──────────────────────────────────────────────

def extract_features(
    y: np.ndarray,
    sr: int = 32000,
    use_pcen: bool = False,
    mean: Optional[float] = None,
    std: Optional[float] = None,
    target_frames: int = 224,
) -> np.ndarray:
    """Complete feature extraction pipeline: audio → (3, 224, 224).

    Args:
        y: 1D audio array.
        sr: Sample rate.
        use_pcen: If True, use PCEN instead of log-mel.
        mean: Pre-computed global mean for normalization.
        std: Pre-computed global std for normalization.
        target_frames: Target number of time frames (default 224).

    Returns:
        Array of shape (3, target_frames, n_mels) = (3, 224, 224) by default.
    """
    if use_pcen:
        spec = pcen_spectrogram(y, sr)
    else:
        spec = log_mel_spectrogram(y, sr)

    spec = pad_or_truncate(spec, target_frames)

    if mean is not None and std is not None:
        spec = normalize_spectrogram(spec, mean, std)

    return to_3channel(spec)


# ──────────────────────────────────────────────
# Normalization stats computer
# ──────────────────────────────────────────────

def compute_norm_stats(
    segments: list[np.ndarray],
    sr: int = 32000,
    use_pcen: bool = False,
) -> Tuple[float, float]:
    """Compute global mean and std of spectrograms from a list of audio segments.

    Args:
        segments: List of 1D audio arrays.
        sr: Sample rate.
        use_pcen: If True, use PCEN.

    Returns:
        (mean, std) floats.
    """
    all_values = []
    for seg in segments:
        if use_pcen:
            spec = pcen_spectrogram(seg, sr)
        else:
            spec = log_mel_spectrogram(seg, sr)
        all_values.append(spec.ravel())

    all_values = np.concatenate(all_values)
    return float(all_values.mean()), float(all_values.std())


def compute_and_save_norm_stats(
    audio_files: list[str | Path],
    sr: int = 32000,
    n_samples: int = 5000,
    use_pcen: bool = False,
    output_path: str | Path = "data/norm_stats.pkl",
    seed: int = 42,
) -> Tuple[float, float]:
    """Pick N random files, extract one random segment each,
    compute mean/std, save to pickle.

    Args:
        audio_files: List of audio file paths.
        sr: Sample rate.
        n_samples: Number of audio files to sample.
        use_pcen: If True, compute PCEN stats.
        output_path: Where to save (mean, std) as pickle.
        seed: Random seed.

    Returns:
        (mean, std) tuple.
    """
    import pickle
    import random

    random.seed(seed)
    sampled = random.sample(audio_files, min(n_samples, len(audio_files)))

    from src.preprocessing import resample_and_segment

    segments = []
    for f in sampled:
        try:
            segs = resample_and_segment(f, sr=sr, duration=5, remove_silence=True)
            if segs:
                segments.append(random.choice(segs))
        except Exception:
            continue

    mean_val, std_val = compute_norm_stats(segments, sr, use_pcen)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({"mean": mean_val, "std": std_val}, f)

    print(f"Norm stats saved to {output_path}: mean={mean_val:.4f}, std={std_val:.4f}")
    return mean_val, std_val


if __name__ == "__main__":
    # Quick smoke test
    import sys

    if len(sys.argv) > 1:
        from src.preprocessing import load_audio

        y = load_audio(sys.argv[1])
        feat_logmel = extract_features(y, use_pcen=False)
        feat_pcen = extract_features(y, use_pcen=True)
        print(f"Log-Mel shape: {feat_logmel.shape}, range: [{feat_logmel.min():.2f}, {feat_logmel.max():.2f}]")
        print(f"PCEN shape:   {feat_pcen.shape}, range: [{feat_pcen.min():.2f}, {feat_pcen.max():.2f}]")
    else:
        print("Usage: python features.py <audio_file>")
