
# ═══════════════════════════════════════════════════════════
# CELL: Species Boost Post-Processing
# Apply heuristic species weights + phylogenetic proxies
# ═══════════════════════════════════════════════════════════

import json
import numpy as np
from pathlib import Path

def load_species_boost(config_path: str | Path) -> dict:
    """Load species boost configuration."""
    with open(config_path) as f:
        return json.load(f)

def apply_species_boost(probs: np.ndarray, config: dict, 
                        weight_strength: float = 0.3,
                        proxy_strength: float = 0.2) -> np.ndarray:
    """Apply heuristic species weight boost and phylogenetic proxies.
    
    Args:
        probs: (N, 234) probability array from Perch+SSM ensemble
        config: Species boost config dict
        weight_strength: 0=off, 1=full weight boost (default 0.3, conservative)
        proxy_strength: How much to blend phylogenetic proxy (default 0.2)
    
    Returns:
        (N, 234) adjusted probabilities
    """
    species_list = list(config["species_weights"].keys())
    n_species = len(species_list)
    
    # ── 1. Apply species prevalence weights ──
    weights = np.array([config["species_weights"][sp] for sp in species_list])
    # Interpolate: weight=1 means no change, weight>1 means boost
    effective_weights = 1.0 + weight_strength * (weights - 1.0)
    probs = probs * effective_weights[np.newaxis, :]
    
    # ── 2. Apply phylogenetic proxies for zero-shot species ──
    phylo = config.get("phylo_proxy", {})
    species_to_idx = {sp: i for i, sp in enumerate(species_list)}
    
    for target_sp, info in phylo.items():
        idx_target = species_to_idx[target_sp]
        idx_proxy = species_to_idx[info["proxy_species"]]
        # Blend: keep (1-proxy_strength) of original + proxy_strength of proxy
        proxy_prob = probs[:, idx_proxy]
        # Conservatism: scale down proxy (it's a different species after all)
        conservatism = 0.5
        probs[:, idx_target] = (
            (1.0 - proxy_strength) * probs[:, idx_target] + 
            proxy_strength * proxy_prob * conservatism
        )
    
    # ── 3. Row-wise renormalization ──
    # Keep the overall probability mass consistent
    row_sums_before = probs.sum(axis=1, keepdims=True)
    probs = probs / (row_sums_before + 1e-8) * (row_sums_before.mean() + 1e-8)
    
    return np.clip(probs, 0.0, 1.0)

def apply_taxon_temperatures(logits: np.ndarray, config: dict) -> np.ndarray:
    """Apply taxon-based temperature scaling to logits before sigmoid.
    
    Lower temperature (0.95) for Amphibia/Insecta → sharper predictions.
    Higher temperature (1.10) for Aves → softer, less overconfident.
    """
    from scipy.special import expit
    species_list = list(config["temperatures"].keys())
    temps = np.array([config["temperatures"][sp] for sp in species_list])
    return expit(logits / temps[np.newaxis, :])
