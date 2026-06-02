
"""BirdCLEF 2026 — Calibrated Inference Module
Import this in the submission notebook to apply calibration + species weights.
"""

import numpy as np
import json
from pathlib import Path
from scipy.special import expit  # sigmoid

def load_calibration(config_path: str | Path) -> dict:
    """Load calibration config from JSON file."""
    with open(config_path) as f:
        return json.load(f)

def apply_calibration(logits: np.ndarray, config: dict) -> np.ndarray:
    """Apply per-species Platt calibration to raw Perch logits.
    
    Args:
        logits: (N, 234) raw Perch logits
        config: Calibration config dict
    
    Returns:
        (N, 234) calibrated probabilities
    """
    probs = expit(logits)  # sigmoid conversion
    cal = config["calibration_coeffs"]
    
    for sp_name, coeffs in cal.items():
        idx = config["species_stats"][sp_name]["index"]
        # p = sigmoid(a * logit + b)
        a = coeffs["coef"]
        b = coeffs["intercept"]
        probs[:, idx] = expit(a * logits[:, idx] + b)
    
    return probs

def apply_species_weights(probs: np.ndarray, config: dict, boost_strength: float = 1.0) -> np.ndarray:
    """Apply prevalence-based species weights.
    
    Args:
        probs: (N, 234) probabilities
        config: Calibration config dict
        boost_strength: How strongly to apply weights (0=off, 1=full)
    
    Returns:
        (N, 234) adjusted probabilities
    """
    weights = np.array([config["species_weights"][sp] for sp in config["species_stats"]])
    # Interpolate between original (boost_strength=0) and weighted (boost_strength=1)
    effective_weights = 1.0 + boost_strength * (weights - 1.0)
    
    probs_adj = probs * effective_weights[np.newaxis, :]
    # Row-wise normalization to keep probabilities well-behaved
    row_sums = probs_adj.sum(axis=1, keepdims=True)
    row_sums_orig = probs.sum(axis=1, keepdims=True)
    # Blend normalization: keep original scale
    probs_adj = probs_adj / (row_sums + 1e-8) * (row_sums_orig + 1e-8)
    
    return np.clip(probs_adj, 0.0, 1.0)

def apply_phylo_prior(probs: np.ndarray, config: dict) -> np.ndarray:
    """Apply phylogenetic prior: for zero-shot species, use proxy species probability.
    
    This borrows the calibrated probability from the closest related species
    that has training data, scaled by a conservatism factor.
    
    Args:
        probs: (N, 234) probabilities
        config: Calibration config dict
    
    Returns:
        (N, 234) adjusted probabilities
    """
    phylo = config.get("phylo_prior", {})
    stats = config["species_stats"]
    
    for sp_name, info in phylo.items():
        idx_target = stats[sp_name]["index"]
        idx_proxy = stats[info["proxy_species"]]["index"]
        # Blend: 70% original + 30% proxy, to add signal without overriding
        proxy_prob = probs[:, idx_proxy]
        # Conservatism factor: scale down proxy probability
        conservatism = 0.7
        probs[:, idx_target] = 0.3 * probs[:, idx_target] + 0.7 * (proxy_prob * conservatism)
    
    return np.clip(probs, 0.0, 1.0)
