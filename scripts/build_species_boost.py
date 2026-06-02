"""
BirdCLEF 2026 — Species Boost Config (Heuristic, No Learning)
==============================================================
Generates lightweight post-processing config for the submission notebook:
  1. Species prevalence weights (from FULL train.csv, not OOF soundscapes)
  2. Genus-level phylogenetic proxy for 28 zero-shot species
  3. Taxon-based temperature scaling (Amphibia/Insecta vs Aves)

All heuristics — zero overfitting risk, zero inference cost.

Author: BirdCLEF 2026 Stacking Sprint
Date: 2026-05-31
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import warnings

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════
ROOT = Path(r"C:\Users\Utilisateur\Mon Drive\Kaggle\__birdclef-v3")
TRAIN_CSV = Path(r"C:\Users\Utilisateur\Mon Drive\Kaggle\__birdclef-V2\birdclef-2026-data\train.csv")
TAXONOMY_CSV = Path(r"C:\Users\Utilisateur\Mon Drive\Kaggle\__birdclef-V2\birdclef-2026-data\taxonomy.csv")
SAMPLE_SUB = Path(r"C:\Users\Utilisateur\Mon Drive\Kaggle\__birdclef-V2\birdclef-2026-data\sample_submission.csv")
OUTPUT_DIR = ROOT / "models" / "species_boost"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════
# 1. Load data
# ═══════════════════════════════════════════════════════
print("=" * 60)
print("1. Loading data...")

train = pd.read_csv(TRAIN_CSV)
taxonomy = pd.read_csv(TAXONOMY_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)
species_cols = sample_sub.columns[1:].tolist()
N_CLASSES = len(species_cols)

print(f"   Training clips: {len(train)}")
print(f"   Species in taxonomy: {len(taxonomy)}")
print(f"   Species in submission: {N_CLASSES}")

# ═══════════════════════════════════════════════════════
# 2. Species prevalence from FULL training data
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. Computing species prevalence from full train.csv...")

train_counts = train["primary_label"].value_counts().to_dict()
species_to_idx = {sp: i for i, sp in enumerate(species_cols)}

counts = np.array([train_counts.get(sp, 0) for sp in species_cols], dtype=np.float32)
n_zero_shot = int((counts == 0).sum())
n_rare = int(((counts > 0) & (counts < 10)).sum())
n_common = int((counts >= 10).sum())

print(f"   Zero-shot (0 clips):     {n_zero_shot}")
print(f"   Rare (1-9 clips):        {n_rare}")
print(f"   Common (≥ 10 clips):     {n_common}")
print(f"   Count range: [{counts[counts > 0].min():.0f}, {counts.max():.0f}]")

# Heuristic weight: 1 / sqrt(count + 1)
# Zero-shot species (count=0) get the highest weight
# Very common species (count=499) get lower weight
raw_weights = 1.0 / np.sqrt(counts + 1.0)
# Normalize so mean = 1.0
species_weights = raw_weights / raw_weights.mean()

print(f"\n   Weight range: [{species_weights.min():.3f}, {species_weights.max():.3f}]")
print(f"   Top-10 boosted species:")
top_boosted = np.argsort(-species_weights)[:10]
for idx in top_boosted:
    sp = species_cols[idx]
    print(f"     {sp}: w={species_weights[idx]:.3f}, clips={counts[idx]:.0f}")

print(f"\n   Bottom-5 (most penalized) species:")
bottom_boosted = np.argsort(species_weights)[:5]
for idx in bottom_boosted:
    sp = species_cols[idx]
    print(f"     {sp}: w={species_weights[idx]:.3f}, clips={counts[idx]:.0f}")

# ═══════════════════════════════════════════════════════
# 3. Genus-level phylogenetic proxy for zero-shot species
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. Building genus-level phylogenetic proxies...")

# Parse genus from scientific_name (first word)
taxonomy["genus"] = taxonomy["scientific_name"].apply(
    lambda x: str(x).split()[0] if pd.notna(x) and len(str(x).split()) > 0 else ""
)

# Build genus → list of (species, count) mapping
genus_to_species = {}
for _, row in taxonomy.iterrows():
    sp = row["primary_label"]
    genus = row["genus"]
    if genus and genus != "nan":
        if genus not in genus_to_species:
            genus_to_species[genus] = []
        cnt = train_counts.get(sp, 0)
        genus_to_species[genus].append((sp, cnt))

# For each zero-shot or rare species, find proxy
phylo_proxy = {}
for sp in species_cols:
    cnt = train_counts.get(sp, 0)
    if cnt >= 5:
        continue  # Enough data, no proxy needed

    # Get genus
    tax_row = taxonomy[taxonomy["primary_label"] == sp]
    if len(tax_row) == 0:
        continue

    genus = tax_row.iloc[0]["genus"]
    class_name = tax_row.iloc[0]["class_name"]

    # Look for species in same genus with training data
    candidates = genus_to_species.get(genus, [])
    candidates = [(s, c) for s, c in candidates if s != sp and c >= 5]
    candidates.sort(key=lambda x: -x[1])  # most samples first

    if candidates:
        proxy_sp, proxy_cnt = candidates[0]
        phylo_proxy[sp] = {
            "proxy_species": str(proxy_sp),
            "proxy_count": int(proxy_cnt),
            "method": f"same_genus({genus})",
            "target_count": int(cnt),
            "target_class": str(class_name),
        }
    else:
        # Fallback: use class-level prior
        same_class = [
            (s, c) for s, c in zip(species_cols, counts)
            if s != sp and c >= 5 and taxonomy[taxonomy["primary_label"] == s].iloc[0]["class_name"] == class_name
        ]
        if same_class:
            same_class.sort(key=lambda x: -x[1])
            proxy_sp, proxy_cnt = same_class[0]
            phylo_proxy[sp] = {
                "proxy_species": str(proxy_sp),
                "proxy_count": int(proxy_cnt),
                "method": f"same_class({class_name})",
                "target_count": int(cnt),
                "target_class": str(class_name),
            }

print(f"   Found phylogenetic proxies for {len(phylo_proxy)} species:")
for sp, info in list(phylo_proxy.items())[:15]:
    print(f"     {sp} (cnt={info['target_count']}, {info['target_class']}) → {info['proxy_species']} ({info['method']})")
if len(phylo_proxy) > 15:
    print(f"     ... and {len(phylo_proxy) - 15} more")

# ═══════════════════════════════════════════════════════
# 4. Taxon-based temperature scaling
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4. Computing taxon-based temperatures...")

# From EoS V2 notebook: temperatures for Amphibia/Insecta vs Aves
# Amphibia and Insecta are texture-based sounds (not vocal)
# Lower temperature = sharper predictions for texture taxa
TEXTURE_TAXA = {"Amphibia", "Insecta"}
DEFAULT_TEMP = 1.10  # Aves: slightly soften (reduce overconfidence)
TEXTURE_TEMP = 0.95  # Amphibia/Insecta: slightly sharpen

temperatures = np.ones(N_CLASSES, dtype=np.float32)
for i, sp in enumerate(species_cols):
    tax_row = taxonomy[taxonomy["primary_label"] == sp]
    if len(tax_row) > 0:
        cls = tax_row.iloc[0]["class_name"]
        temperatures[i] = TEXTURE_TEMP if cls in TEXTURE_TAXA else DEFAULT_TEMP

print(f"   Aves temperature: {DEFAULT_TEMP}")
print(f"   Amphibia/Insecta temperature: {TEXTURE_TEMP}")
print(f"   Texture taxa count: {(temperatures == TEXTURE_TEMP).sum()}")

# ═══════════════════════════════════════════════════════
# 5. Save config
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("5. Saving config...")

config = {
    "species_weights": {sp: float(w) for sp, w in zip(species_cols, species_weights)},
    "species_counts": {sp: int(c) for sp, c in zip(species_cols, counts)},
    "phylo_proxy": phylo_proxy,
    "temperatures": {sp: float(t) for sp, t in zip(species_cols, temperatures)},
    "n_classes": N_CLASSES,
    "n_zero_shot": n_zero_shot,
    "n_rare": n_rare,
    "n_common": n_common,
    "total_training_clips": len(train),
    "description": "Heuristic species weights + phylogenetic proxies for BirdCLEF 2026 submission. Zero overfitting risk.",
}

with open(OUTPUT_DIR / "species_boost_config.json", "w") as f:
    json.dump(config, f, indent=2)
print(f"   Saved species_boost_config.json")

# ═══════════════════════════════════════════════════════
# 6. Generate submission notebook code snippet
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("6. Generating submission notebook code...")

submission_code = '''
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
'''

with open(OUTPUT_DIR / "species_boost_inference.py", "w", encoding="utf-8") as f:
    f.write(submission_code)
print(f"   Saved species_boost_inference.py")

# ═══════════════════════════════════════════════════════
# 7. Summary
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("7. SUMMARY")
print("=" * 60)
print(f"   Total training clips: {len(train)}")
print(f"   Species: {N_CLASSES} ({n_common} common, {n_rare} rare, {n_zero_shot} zero-shot)")
print(f"   Phylogenetic proxies: {len(phylo_proxy)}")
print(f"   Weight range: [{species_weights.min():.3f}, {species_weights.max():.3f}]")
print(f"")
print(f"   Usage in submission notebook:")
print(f"     probs = apply_species_boost(probs, config, weight_strength=0.3)")
print(f"   Or with temperatures:")
print(f"     probs = apply_taxon_temperatures(logits, config)")
print(f"")
print(f"   Config saved to: {OUTPUT_DIR}")
print("=" * 60)
