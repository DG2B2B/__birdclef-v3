"""
BirdCLEF 2026 – Pseudo-Labeling with Power Transform
Iterative self-training inspired by 1st place BirdCLEF 2025.
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Optional, List, Tuple
import pickle
import gc
from datetime import datetime

from src.preprocessing import process_soundscape, DEFAULT_SR, DEFAULT_DURATION
from src.features import extract_features, load_norm_stats
from src.models_sed import BirdSEDClassifier, create_model
from src.dataset import build_label_map


# ──────────────────────────────────────────────
# Power Transform
# ──────────────────────────────────────────────

def power_transform(probs: np.ndarray, power: float) -> np.ndarray:
    """Apply power transform to clean noisy pseudo-labels.

    p_transformed = sigmoid(logit(p) * power) = p^power / (p^power + (1-p)^power)
    Simplified: p_transformed = p ** power

    This pushes probabilities toward 0 or 1, cleaning ambiguous predictions.
    Lower power = stronger cleaning.

    Args:
        probs: Array of probabilities in [0, 1].
        power: Power value (1.0 = no change, <1 = sharpen).

    Returns:
        Transformed probabilities.
    """
    if power == 1.0:
        return probs
    return probs ** power


# ──────────────────────────────────────────────
# Pseudo-label generation
# ──────────────────────────────────────────────

def generate_pseudo_labels(
    model_paths: List[str | Path],
    soundscape_dir: str | Path,
    output_csv: str | Path,
    norm_stats_path: str | Path,
    species_list: List[str],
    label_map: dict[str, int],
    threshold: float = 0.9,
    power: float = 1.0,
    batch_size: int = 64,
    device: Optional[str] = None,
) -> pd.DataFrame:
    """Generate pseudo-labels for all soundscapes using teacher ensemble.

    For each soundscape:
      1. Split into 5s segments
      2. Extract features
      3. Predict with all teacher models
      4. Average probabilities across models
      5. Apply Power Transform
      6. Apply confidence threshold -> binary labels

    Args:
        model_paths: List of .pth checkpoint paths (one per ensemble model).
        soundscape_dir: Directory containing .ogg soundscape files.
        output_csv: Where to save pseudo-labeled CSV.
        norm_stats_path: Path to norm_stats.pkl.
        species_list: Ordered list of 234 species IDs.
        label_map: species_id -> index mapping.
        threshold: Confidence threshold for binary labels.
        power: Power transform exponent.
        batch_size: Batch size for inference.
        device: 'cuda' or 'cpu'.

    Returns:
        DataFrame with columns: filepath, [234 species probs], is_pseudo.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load normalization stats
    mean, std = load_norm_stats(norm_stats_path)

    # Load models
    models = []
    for mp in model_paths:
        checkpoint = torch.load(mp, map_location=device, weights_only=False)
        cfg = checkpoint.get("config", {})
        backbone = cfg.get("backbone", "efficientnet_b0")
        n_classes = cfg.get("n_classes", 234)
        use_sed = cfg.get("use_sed_head", True)

        model = create_model(
            backbone_name=backbone,
            n_classes=n_classes,
            pretrained=False,
            use_sed_head=use_sed,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()
        models.append(model)
        print(f"  Loaded {backbone} (AUC={checkpoint.get('val_auc', '?')})")

    # List soundscapes
    soundscape_dir = Path(soundscape_dir)
    soundscape_files = sorted(soundscape_dir.glob("*.ogg"))
    print(f"  Processing {len(soundscape_files)} soundscapes...")

    rows = []
    n_total_segments = 0

    for sf in soundscape_files:
        # Segment without silence removal
        segments = process_soundscape(sf, sr=DEFAULT_SR, duration=DEFAULT_DURATION)

        if not segments:
            continue

        # Extract features in batch
        features = []
        for seg in segments:
            feat = extract_features(seg, sr=DEFAULT_SR, use_pcen=False, mean=mean, std=std)
            features.append(feat)

        features = np.stack(features)  # (N, 3, 224, 224)

        # Ensemble prediction
        all_probs = []
        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch = torch.from_numpy(features[i : i + batch_size]).to(device)
                batch_probs = torch.zeros((len(batch), len(species_list)), device=device)

                for model in models:
                    out = model.forward_clip(batch)
                    batch_probs += torch.sigmoid(out)

                batch_probs /= len(models)
                all_probs.append(batch_probs.cpu().numpy())

        probs = np.concatenate(all_probs, axis=0)  # (N, 234)

        # Power Transform
        probs_transformed = power_transform(probs, power)

        # Threshold -> binary (soft labels kept as probabilities for training)
        for seg_idx, seg_probs in enumerate(probs_transformed):
            row = {"filepath": f"{sf}_{seg_idx:04d}", "is_pseudo": 1}
            for sp_idx, sp_id in enumerate(species_list):
                row[sp_id] = float(seg_probs[sp_idx])
            rows.append(row)

        n_total_segments += len(segments)
        if len(soundscape_files) % 50 == 0:
            print(f"    {len(soundscape_files)}/{len(soundscape_files)} done, {n_total_segments} segments")

    # Build DataFrame
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"  Saved {len(df)} pseudo-labeled segments to {output_csv}")

    # Cleanup
    del models
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return df


# ──────────────────────────────────────────────
# Dataset merging (original + pseudo-labeled)
# ──────────────────────────────────────────────

def create_enriched_dataset(
    original_csv: str | Path,
    pseudo_csv: str | Path,
    output_csv: str | Path,
    original_ratio: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """Merge original labeled data with pseudo-labeled data.

    Args:
        original_csv: Original segments CSV (from phase 1 preprocessing).
        pseudo_csv: Pseudo-labeled segments CSV.
        output_csv: Output path.
        original_ratio: Fraction of original data to keep (0.5 = 50/50 split).
        seed: Random seed.

    Returns:
        Combined DataFrame.
    """
    df_orig = pd.read_csv(original_csv)
    df_pseudo = pd.read_csv(pseudo_csv)

    # Add is_pseudo flag if not present
    if "is_pseudo" not in df_orig.columns:
        df_orig["is_pseudo"] = 0

    # Balance: sample original to match pseudo count if needed
    rng = np.random.RandomState(seed)
    target_orig = int(len(df_pseudo) * original_ratio / (1 - original_ratio))
    if len(df_orig) > target_orig:
        df_orig = df_orig.sample(n=target_orig, random_state=seed)

    df_combined = pd.concat([df_orig, df_pseudo], ignore_index=True)
    df_combined = df_combined.sample(frac=1, random_state=seed).reset_index(drop=True)

    df_combined.to_csv(output_csv, index=False)
    print(f"Enriched dataset: {len(df_orig)} original + {len(df_pseudo)} pseudo = {len(df_combined)} total")
    print(f"  Saved to {output_csv}")

    return df_combined


# ──────────────────────────────────────────────
# Iteration runner
# ──────────────────────────────────────────────

POWER_SCHEDULE = [
    # (iteration, power_value, threshold)
    (1, 1.0, 0.9),
    (2, 0.65, 0.9),
    (3, 0.55, 0.9),
    (4, 0.6, 0.9),
]

def run_pseudo_label_iteration(
    iteration: int,
    model_paths: List[str | Path],
    soundscape_dir: str | Path,
    original_csv: str | Path,
    norm_stats_path: str | Path,
    species_list: List[str],
    output_dir: str | Path,
    power: float = 1.0,
    threshold: float = 0.9,
    original_ratio: float = 0.5,
    batch_size: int = 64,
) -> Tuple[str, str]:
    """Run one pseudo-label iteration.

    Returns:
        (pseudo_csv_path, enriched_csv_path).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_map = build_label_map(species_list=species_list)

    print(f"\n{'='*60}")
    print(f"Iteration {iteration}: Power={power}, Threshold={threshold}")
    print(f"{'='*60}")

    # Step 1: Generate pseudo-labels
    pseudo_csv = output_dir / f"pseudo_labels_iter{iteration}.csv"
    print(f"\n[Step 1] Generating pseudo-labels...")
    generate_pseudo_labels(
        model_paths=model_paths,
        soundscape_dir=soundscape_dir,
        output_csv=pseudo_csv,
        norm_stats_path=norm_stats_path,
        species_list=species_list,
        label_map=label_map,
        threshold=threshold,
        power=power,
        batch_size=batch_size,
    )

    # Step 2: Create enriched dataset
    enriched_csv = output_dir / f"train_enriched_iter{iteration}.csv"
    print(f"\n[Step 2] Creating enriched dataset...")
    create_enriched_dataset(
        original_csv=original_csv,
        pseudo_csv=pseudo_csv,
        output_csv=enriched_csv,
        original_ratio=original_ratio,
    )

    return str(pseudo_csv), str(enriched_csv)


if __name__ == "__main__":
    print("Pseudo-labeling module ready.")
    print(f"  Power schedule: {POWER_SCHEDULE}")
    print(f"  Species: 234 (206 with data + 28 zero-shot)")
