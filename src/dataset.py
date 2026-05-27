"""
BirdCLEF 2026 – PyTorch Dataset
Multi-label dataset with SpecAugment, Background Mix, and MixUp augmentations.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Callable
import pickle
import random

from src.features import extract_features, pad_or_truncate, to_3channel


# ──────────────────────────────────────────────
# Augmentations
# ──────────────────────────────────────────────

class SpecAugment:
    """Frequency and time masking for spectrograms.

    Applied directly on the 2D spectrogram (n_mels, time_frames).
    """

    def __init__(
        self,
        freq_mask_param: int = 10,
        time_mask_param: int = 20,
        n_freq_masks: int = 1,
        n_time_masks: int = 1,
        p: float = 0.5,
    ):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.p = p

    def __call__(self, spec: np.ndarray) -> np.ndarray:
        """Apply SpecAugment to a 2D spectrogram.

        Args:
            spec: (n_mels, time_frames) array.

        Returns:
            Augmented (n_mels, time_frames) array.
        """
        if random.random() > self.p:
            return spec

        spec = spec.copy()
        n_mels, t = spec.shape

        # Frequency masking
        for _ in range(self.n_freq_masks):
            f = random.randint(0, max(1, self.freq_mask_param))
            f0 = random.randint(0, max(1, n_mels - f))
            spec[f0 : f0 + f, :] = 0

        # Time masking
        for _ in range(self.n_time_masks):
            t_len = random.randint(0, max(1, self.time_mask_param))
            t0 = random.randint(0, max(1, t - t_len))
            spec[:, t0 : t0 + t_len] = 0

        return spec


class MixUp:
    """MixUp augmentation for multi-label classification.

    Blends two samples with a fixed or random alpha.
    """

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def __call__(
        self,
        x1: np.ndarray,
        y1: np.ndarray,
        x2: np.ndarray,
        y2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mix two samples.

        Args:
            x1, x2: Spectrograms (3, H, W).
            y1, y2: Multi-hot label vectors.

        Returns:
            (mixed_x, mixed_y).
        """
        lam = self.alpha  # fixed blending (as per 1st place 2025)
        mixed_x = lam * x1 + (1 - lam) * x2
        mixed_y = lam * y1 + (1 - lam) * y2
        return mixed_x, mixed_y


class BackgroundMix:
    """Mix background noise (rain, insects, wind) into the audio.

    Operates on raw audio before spectrogram extraction.
    The background files should be pre-collected noise samples.
    """

    def __init__(
        self,
        noise_dir: Optional[str | Path] = None,
        noise_ratio: float = 0.3,
        p: float = 0.3,
    ):
        self.noise_dir = Path(noise_dir) if noise_dir else None
        self.noise_ratio = noise_ratio
        self.p = p
        self._noise_cache: list[np.ndarray] = []

        if self.noise_dir and self.noise_dir.exists():
            self._load_noise()

    def _load_noise(self) -> None:
        """Preload noise samples from directory."""
        import librosa

        for f in self.noise_dir.glob("*.ogg"):
            try:
                y, _ = librosa.load(str(f), sr=32000, mono=True)
                self._noise_cache.append(y.astype(np.float32))
            except Exception:
                pass
        for f in self.noise_dir.glob("*.mp3"):
            try:
                y, _ = librosa.load(str(f), sr=32000, mono=True)
                self._noise_cache.append(y.astype(np.float32))
            except Exception:
                pass
        print(f"[BackgroundMix] Loaded {len(self._noise_cache)} noise samples")

    def __call__(self, y: np.ndarray) -> np.ndarray:
        """Mix background noise into audio signal.

        Args:
            y: 1D audio array (float32).

        Returns:
            Augmented audio.
        """
        if random.random() > self.p or not self._noise_cache:
            return y

        noise = random.choice(self._noise_cache)
        # Trim or repeat noise to match y length
        if len(noise) < len(y):
            repeats = len(y) // len(noise) + 1
            noise = np.tile(noise, repeats)
        noise = noise[: len(y)]

        # Normalize noise power
        noise_rms = np.sqrt(np.mean(noise**2)) + 1e-8
        y_rms = np.sqrt(np.mean(y**2)) + 1e-8
        noise = noise / noise_rms * y_rms * self.noise_ratio

        return y + noise


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class BirdCLEFDataset:
    """Multi-label dataset for BirdCLEF 2026.

    Reads a CSV with columns: filepath, primary_label[, secondary_labels].
    Each sample is a pre-segmented audio clip (5 s).
    """

    def __init__(
        self,
        csv_path: str | Path,
        n_classes: int = 234,
        sr: int = 32000,
        use_pcen: bool = False,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        spec_augment: Optional[SpecAugment] = None,
        background_mix: Optional[BackgroundMix] = None,
        mix_up: Optional[MixUp] = None,
        mixup_indices: Optional[list[int]] = None,
        target_frames: int = 224,
        label_col: str = "primary_label",
        label_map: Optional[dict[str, int]] = None,
    ):
        """
        Args:
            csv_path: Path to CSV with 'filepath' and label columns.
            n_classes: Total number of species (234).
            sr: Sample rate.
            use_pcen: If True, use PCEN instead of log-mel.
            mean, std: Normalization stats.
            spec_augment: SpecAugment transform or None.
            background_mix: BackgroundMix transform or None.
            mix_up: MixUp transform or None.
            mixup_indices: Shuffled indices for MixUp pairing.
            target_frames: Target time frames for spectrogram.
            label_col: Column name for primary label.
            label_map: Dict mapping species ID → integer index.
        """
        import pandas as pd

        self.df = pd.read_csv(csv_path)
        self.n_classes = n_classes
        self.sr = sr
        self.use_pcen = use_pcen
        self.mean = mean
        self.std = std
        self.spec_augment = spec_augment
        self.background_mix = background_mix
        self.mix_up = mix_up
        self.mixup_indices = mixup_indices
        self.target_frames = target_frames
        self.label_col = label_col
        self.label_map = label_map or {}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        """Returns (spectrogram, label) as numpy arrays.
        
        Use torch.from_numpy() in training loop to convert to tensors.
        """
        row = self.df.iloc[idx]
        seg_path = row["filepath"]

        # Load pre-segmented audio
        y = np.load(seg_path).astype(np.float32)

        # Background mix (on raw audio)
        if self.background_mix is not None:
            y = self.background_mix(y)

        # Extract spectrogram + normalize
        spec = extract_features(
            y, sr=self.sr, use_pcen=self.use_pcen,
            mean=self.mean, std=self.std, target_frames=self.target_frames,
        )  # shape: (3, target_frames, n_mels)

        # SpecAugment (on the 2D representation before 3-channel)
        if self.spec_augment is not None:
            spec_2d = spec[0]  # one channel is enough for SpecAugment
            spec_2d = self.spec_augment(spec_2d)
            spec = np.stack([spec_2d] * 3, axis=0)

        # Multi-hot label
        label = np.zeros(self.n_classes, dtype=np.float32)
        primary = row[self.label_col]
        if self.label_map:
            if primary in self.label_map:
                label[self.label_map[primary]] = 1.0
        else:
            try:
                label[int(primary)] = 1.0
            except (ValueError, KeyError):
                pass

        return spec, label


# ──────────────────────────────────────────────
# Label map builder
# ──────────────────────────────────────────────

def build_label_map(
    taxonomy_csv: str | Path,
    species_list: Optional[list[str]] = None,
) -> dict[str, int]:
    """Build a mapping from species ID to integer index.

    Args:
        taxonomy_csv: Path to taxonomy.csv.
        species_list: Optional explicit list of species IDs (from sample_submission).

    Returns:
        Dict mapping species_id → index (0..233).
    """
    import pandas as pd

    if species_list is not None:
        return {sp: i for i, sp in enumerate(sorted(species_list))}

    df = pd.read_csv(taxonomy_csv)
    species = sorted(df["primary_label"].unique().tolist())
    return {sp: i for i, sp in enumerate(species)}


def load_norm_stats(path: str | Path) -> tuple[float, float]:
    """Load (mean, std) from pickle file."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["mean"], data["std"]


if __name__ == "__main__":
    print("BirdCLEFDataset ready.")
    print(f"  - SpecAugment: freq_mask=10, time_mask=20")
    print(f"  - MixUp: alpha=0.5 (fixed)")
    print(f"  - BackgroundMix: p=0.3, ratio=0.3")
