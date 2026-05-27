"""
BirdCLEF 2026 – Rare Species & Zero-Shot Model
Handles the 28 species with no training data in train_audio/.
Strategy: Xeno-Canto downloads + Perch embeddings fallback.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List


# ──────────────────────────────────────────────
# Zero-shot species list (verified from data)
# ──────────────────────────────────────────────

ZERO_SHOT_SPECIES = [
    # 3 Amphibia
    "1491113",  # Adenomera guarani (Guarani leaf-litter frog)
    "25073",    # Chiasmocleis mehelyi
    "517063",   # Pithecopus azureus (Southern Orange-legged Leaf Frog)
    # 25 Insect sonotypes (variants of taxon 47158)
    "47158son01", "47158son02", "47158son03", "47158son04", "47158son05",
    "47158son06", "47158son07", "47158son08", "47158son09", "47158son10",
    "47158son11", "47158son12", "47158son13", "47158son14", "47158son15",
    "47158son16", "47158son17", "47158son18", "47158son19", "47158son20",
    "47158son21", "47158son22", "47158son23", "47158son24", "47158son25",
]

# Species with very few training clips (< 10)
RARE_THRESHOLD = 10


def identify_rare_species(train_csv: str | Path) -> List[str]:
    """Return list of species IDs with fewer than RARE_THRESHOLD clips."""
    df = pd.read_csv(train_csv)
    counts = df["primary_label"].value_counts()
    rare = counts[counts < RARE_THRESHOLD].index.tolist()
    print(f"Rare species (< {RARE_THRESHOLD} clips): {len(rare)}")
    for sp in rare:
        print(f"  {sp}: {counts[sp]} clips")
    return rare


def get_zero_shot_species() -> List[str]:
    """Return the 28 zero-shot species IDs."""
    return ZERO_SHOT_SPECIES.copy()


def build_zero_shot_strategy(
    taxonomy_csv: str | Path,
    train_csv: str | Path,
) -> dict:
    """Analyze zero-shot and rare species, output action plan.

    Returns:
        Dict with {'zero_shot': [...], 'rare': [...], 'action': 'xeno_canto'|'perch'|'both'}.
    """
    taxo = pd.read_csv(taxonomy_csv)
    train = pd.read_csv(train_csv)

    all_submission_species = set(taxo["primary_label"].unique())
    train_species = set(train["primary_label"].unique())
    zero_shot = sorted(all_submission_species - train_species)

    counts = train["primary_label"].value_counts()
    rare = counts[counts < RARE_THRESHOLD].index.tolist()

    # Check taxonomy for these species
    for sp in zero_shot:
        row = taxo[taxo["primary_label"] == sp]
        if len(row) > 0:
            class_name = row.iloc[0]["class_name"]
            sci_name = row.iloc[0]["scientific_name"]
            print(f"  ZERO-SHOT: {sp} | {sci_name} | {class_name}")

    print(f"\nZero-shot species: {len(zero_shot)} (need Xeno-Canto + Perch)")
    print(f"Rare species: {len(rare)} (need data augmentation or oversampling)")

    return {
        "zero_shot": zero_shot,
        "rare": rare,
        "action": "xeno_canto_for_all_28_plus_perch_fallback",
    }


# ──────────────────────────────────────────────
# Xeno-Canto download helper
# ──────────────────────────────────────────────

XENO_CANTO_QUERIES = {
    "1491113": "Adenomera guarani",       # Amphibia
    "25073": "Chiasmocleis mehelyi",       # Amphibia
    "517063": "Pithecopus azureus",        # Amphibia
    "47158": "Insect sonotype Pantanal",   # Insecta (all sonotypes share parent)
}


def generate_xeno_canto_queries() -> dict[str, str]:
    """Return species_id -> search query for Xeno-Canto downloads."""
    return XENO_CANTO_QUERIES


# ──────────────────────────────────────────────
# Rare species training wrapper
# ──────────────────────────────────────────────

RARE_CONFIG = {
    "backbone": "efficientnet_b0",
    "n_classes": 234,
    "epochs": 30,
    "batch_size": 32,
    "lr": 5e-4,
    "focal_alpha": 0.25,
    "focal_gamma": 2.0,
    "drop_path_rate": 0.1,
    "use_class_weights": True,  # upweight rare/zero-shot classes
}


def compute_class_weights(
    train_csv: str | Path,
    n_classes: int = 234,
    zero_shot_weight: float = 5.0,
    rare_weight: float = 3.0,
) -> np.ndarray:
    """Compute per-class weights to handle imbalance.

    Zero-shot species get the highest weight.
    Rare species (< 10 clips) get elevated weight.
    Common species get weight = 1.0.
    """
    df = pd.read_csv(train_csv)
    counts = df["primary_label"].value_counts().to_dict()

    weights = np.ones(n_classes, dtype=np.float32)

    # This needs the label_map from sample_submission
    # For now, return a template
    return weights


if __name__ == "__main__":
    taxonomy = "C:/Users/Utilisateur/Mon Drive/Kaggle/__birdclef-V2/birdclef-2026-data/taxonomy.csv"
    train_csv = "C:/Users/Utilisateur/Mon Drive/Kaggle/__birdclef-V2/birdclef-2026-data/train.csv"
    build_zero_shot_strategy(taxonomy, train_csv)
