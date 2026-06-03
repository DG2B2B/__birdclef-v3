"""
validate_oof_v22.py — Phase 0 : Validation OOF pour BirdCLEF 2026
==================================================================
Répond à 4 questions critiques avant toute soumission Kaggle :
  1. Direct vs Rank blending — lequel gagne sur OOF ?
  2. Post-processing — quelles couches aident vraiment ?
  3. Poids d'ensemble optimaux Model_3 / Model_4 ?
  4. (Optionnel) Model_2 diversité

Utilise les données OOF existantes :
  - full_oof_meta_features.npz (708 segments, 234 espèces)
  - oof_labels.npz (ground truth aligné)

Auteur : BirdCLEF 2026 Sprint Final
Date : 2026-06-02
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy.special import expit as sigmoid
import json
import warnings
import sys

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════
ROOT = Path(__file__).resolve().parent.parent  # __birdclef-v3/

# Sources OOF
OOF_META   = ROOT / "sgkf_dataset_v3" / "perch_cache" / "full_oof_meta_features.npz"
OOF_LABELS = ROOT / "models" / "xgb_stacker" / "oof_labels.npz"

# Taxonomy (pour smoothing taxonomique)
TAXONOMY_PATHS = [
    ROOT / "kaggle_dataset" / "taxonomy.csv",
    ROOT / "data" / "taxonomy.csv",
]
# Fallback: competition data
COMP_DATA = Path(r"C:\Users\Utilisateur\Mon Drive\Kaggle\__birdclef-V2\birdclef-2026-data")

N_WINDOWS = 12  # fenêtres par fichier soundscape
N_CLASSES = 234

# ═══════════════════════════════════════════════════════════
# 1. CHARGEMENT DES DONNÉES
# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("PHASE 0 — VALIDATION OOF PRÉ-SOUMISSION")
print("=" * 70)

# --- 1a. OOF predictions ---
print("\n[1] Chargement des prédictions OOF...")
oof_meta = np.load(OOF_META, allow_pickle=True)
oof_labels = np.load(OOF_LABELS, allow_pickle=True)

# oof_base = Perch logits bruts → proxy Model_3
# oof_prior = SSM-corrected logits → proxy Model_4
logits_A = oof_meta["oof_base"].astype(np.float64)   # (708, 234)
logits_B = oof_meta["oof_prior"].astype(np.float64)  # (708, 234)
fold_id  = oof_meta["fold_id"]                        # (708,)

# Ground truth
y_true = oof_labels["y"].astype(np.float64)           # (708, 234)
row_ids = oof_labels["row_ids"]                        # (708,)

print(f"   logits_A (Perch base):    {logits_A.shape}  range [{logits_A.min():.2f}, {logits_A.max():.2f}]")
print(f"   logits_B (SSM corrected): {logits_B.shape}  range [{logits_B.min():.2f}, {logits_B.max():.2f}]")
print(f"   y_true:                   {y_true.shape}  positives={int(y_true.sum())}")
print(f"   Folds: {np.unique(fold_id)}")

# Convertir logits → probabilités
probs_A = sigmoid(logits_A)
probs_B = sigmoid(logits_B)
print(f"   probs_A range: [{probs_A.min():.4f}, {probs_A.max():.4f}]")
print(f"   probs_B range: [{probs_B.min():.4f}, {probs_B.max():.4f}]")

# --- 1b. Taxonomy ---
print("\n[2] Chargement de la taxonomie...")
taxonomy = None
for tp in TAXONOMY_PATHS:
    if tp.exists():
        taxonomy = pd.read_csv(tp)
        break
if taxonomy is None:
    # Try competition data folder
    tp = COMP_DATA / "taxonomy.csv"
    if tp.exists():
        taxonomy = pd.read_csv(tp)

TEXTURE_TAXA = {"Amphibia", "Insecta"}

if taxonomy is not None:
    label_to_class = dict(zip(
        taxonomy["primary_label"].astype(str),
        taxonomy["class_name"].astype(str)
    ))
    # Charger les noms de colonnes depuis sample_submission
    sample_paths = [
        ROOT / "kaggle_dataset" / "sample_submission.csv",
        COMP_DATA / "sample_submission.csv",
    ]
    sample_sub = None
    for sp in sample_paths:
        if sp.exists():
            sample_sub = pd.read_csv(sp)
            break
    if sample_sub is not None:
        species_cols = sample_sub.columns[1:].tolist()
        texture_cols = [c for c in species_cols if label_to_class.get(c, "") in TEXTURE_TAXA]
        event_cols = [c for c in species_cols if label_to_class.get(c, "") not in TEXTURE_TAXA]
        texture_idx = [i for i, c in enumerate(species_cols) if label_to_class.get(c, "") in TEXTURE_TAXA]
        event_idx = [i for i, c in enumerate(species_cols) if label_to_class.get(c, "") not in TEXTURE_TAXA]
        
        # Genus groups for taxonomic smoothing
        species_to_genus = {}
        species_to_class_map = {}
        for _, r in taxonomy.iterrows():
            sp = str(r["primary_label"])
            sci = str(r.get("scientific_name", ""))
            cls = str(r.get("class_name", ""))
            if sci:
                genus = sci.split()[0] if " " in sci else sci
                species_to_genus[sp] = genus
            species_to_class_map[sp] = cls
        
        genus_groups = {}
        class_groups = {}
        for c in species_cols:
            g = species_to_genus.get(c, c)
            cl = species_to_class_map.get(c, c)
            genus_groups.setdefault(g, []).append(c)
            class_groups.setdefault(cl, []).append(c)
        
        multi_genus = {g: [species_cols.index(c) for c in members]
                       for g, members in genus_groups.items() if len(members) > 1}
        multi_class = {cl: [species_cols.index(c) for c in members]
                       for cl, members in class_groups.items() if len(members) > 1}
        print(f"   Taxonomy loaded: {len(species_cols)} espèces")
        print(f"   Texture: {len(texture_idx)}, Event: {len(event_idx)}")
        print(f"   Multi-genus groups: {len(multi_genus)}, Multi-class groups: {len(multi_class)}")
    else:
        print("   ⚠ sample_submission.csv non trouvé — smoothing taxonomique désactivé")
        species_cols = None
        texture_idx, event_idx = [], []
        multi_genus, multi_class = {}, {}
else:
    print("   ⚠ taxonomy.csv non trouvé — smoothing taxonomique désactivé")
    species_cols = None
    texture_idx, event_idx = [], []
    multi_genus, multi_class = {}, {}


# ═══════════════════════════════════════════════════════════
# 2. FONCTIONS DE BLENDING
# ═══════════════════════════════════════════════════════════

def direct_blend(pA, pB, wA):
    """Moyenne pondérée directe des probabilités."""
    return wA * pA + (1.0 - wA) * pB


def rank_blend(pA, pB, wA):
    """Blending par rangs percentiles (rank.1)."""
    EPS = 1e-8
    rA = pd.DataFrame(pA).rank(axis=0, pct=True).to_numpy(np.float64)
    rB = pd.DataFrame(pB).rank(axis=0, pct=True).to_numpy(np.float64)
    return wA * rA + (1.0 - wA) * rB


# ═══════════════════════════════════════════════════════════
# 3. FONCTIONS DE POST-PROCESSING (reproduites du notebook)
# ═══════════════════════════════════════════════════════════

def taxon_temporal_smooth(probs, texture_idx, event_idx, alpha_texture=0.15, alpha_event=0.05):
    """
    CELL 15 du notebook : lissage temporel conscient des taxons.
    Texture (amphibiens/insectes) → alpha plus fort (appels continus).
    Event (oiseaux) → alpha plus faible (appels transitoires).
    """
    N = len(probs)
    assert N % N_WINDOWS == 0, f"Expected multiple of {N_WINDOWS}, got {N}"
    n_files = N // N_WINDOWS
    view = probs.reshape(n_files, N_WINDOWS, -1).copy()
    
    # Appliquer alpha_texture aux colonnes texture
    if len(texture_idx) > 0:
        x = view[:, :, texture_idx]
        prev = np.concatenate([x[:, :1, :], x[:, :-1, :]], axis=1)
        nxt = np.concatenate([x[:, 1:, :], x[:, -1:, :]], axis=1)
        view[:, :, texture_idx] = (1.0 - alpha_texture) * x + 0.5 * alpha_texture * (prev + nxt)
    
    # Appliquer alpha_event aux colonnes event
    if len(event_idx) > 0:
        x = view[:, :, event_idx]
        prev = np.concatenate([x[:, :1, :], x[:, :-1, :]], axis=1)
        nxt = np.concatenate([x[:, 1:, :], x[:, -1:, :]], axis=1)
        view[:, :, event_idx] = (1.0 - alpha_event) * x + 0.5 * alpha_event * (prev + nxt)
    
    return np.clip(view.reshape(N, -1), 0.0, 1.0)


def delta_shift_smooth(probs, alpha=0.15):
    """V17: Delta shift smoothing — identique à taxon_temporal_smooth mais uniforme."""
    N = len(probs)
    assert N % N_WINDOWS == 0
    n_files = N // N_WINDOWS
    view = probs.reshape(n_files, N_WINDOWS, -1)
    prev = np.concatenate([view[:, :1, :], view[:, :-1, :]], axis=1)
    nxt = np.concatenate([view[:, 1:, :], view[:, -1:, :]], axis=1)
    smoothed = (1.0 - alpha) * view + 0.5 * alpha * (prev + nxt)
    return np.clip(smoothed.reshape(N, -1), 0.0, 1.0)


def taxonomic_smooth(probs, multi_genus, multi_class, alpha_genus=0.10, alpha_class=0.03):
    """
    CELL 17 du notebook : lissage taxonomique.
    Moyenne partielle au sein du même genre / classe.
    """
    p = probs.copy()
    
    for members in multi_genus.values():
        mean_val = p[:, members].mean(axis=1, keepdims=True)
        p[:, members] = (1.0 - alpha_genus) * p[:, members] + alpha_genus * mean_val
    
    for members in multi_class.values():
        mean_val = p[:, members].mean(axis=1, keepdims=True)
        p[:, members] = (1.0 - alpha_class) * p[:, members] + alpha_class * mean_val
    
    return np.clip(p, 0.0, 1.0)


def rank_aware_scale(probs, power=0.4):
    """V17: Rank-aware scaling. Scale par (file_max)^power."""
    N = len(probs)
    assert N % N_WINDOWS == 0
    n_files = N // N_WINDOWS
    view = probs.reshape(n_files, N_WINDOWS, -1)
    file_max = view.max(axis=1, keepdims=True)
    scale = np.power(file_max, power)
    return np.clip(view * scale, 0.0, 1.0).reshape(N, -1)


def file_confidence_scale(probs, top_k=2):
    """Scale chaque fenêtre par la moyenne des top-K max par fichier."""
    N = len(probs)
    assert N % N_WINDOWS == 0
    n_files = N // N_WINDOWS
    C = probs.shape[1]
    view = probs.reshape(n_files, N_WINDOWS, C)
    sorted_view = np.sort(view, axis=1)
    top_k_mean = sorted_view[:, -top_k:, :].mean(axis=1, keepdims=True)
    return np.clip(view * top_k_mean, 0.0, 1.0).reshape(N, -1)


# ═══════════════════════════════════════════════════════════
# 4. MÉTRIQUES
# ═══════════════════════════════════════════════════════════

def macro_auc(y_true, y_pred):
    """Macro AUC sur les espèces avec au moins 1 positif dans y_true."""
    keep = y_true.sum(axis=0) > 0
    if keep.sum() < 2:
        return 0.0, 0
    aucs = []
    for c in range(y_true.shape[1]):
        if keep[c]:
            yt = y_true[:, c]
            yp = y_pred[:, c]
            if yt.sum() > 0 and yt.sum() < len(yt):  # besoin des 2 classes
                try:
                    aucs.append(roc_auc_score(yt, yp))
                except ValueError:
                    pass
    return np.mean(aucs) if aucs else 0.0, len(aucs)


def macro_auc_all(y_true, y_pred):
    """Macro AUC sur TOUTES les espèces (inclut les zéros = AUC=0.5 par défaut)."""
    aucs = []
    for c in range(y_true.shape[1]):
        yt = y_true[:, c]
        yp = y_pred[:, c]
        if yt.sum() > 0 and yt.sum() < len(yt):
            try:
                aucs.append(roc_auc_score(yt, yp))
            except ValueError:
                aucs.append(0.5)
        else:
            aucs.append(0.5)  # espèce jamais présente → AUC neutre
    return np.mean(aucs)


def prob_stats(probs):
    """Statistiques descriptives des probabilités."""
    return {
        "mean": float(probs.mean()),
        "median": float(np.median(probs)),
        "min": float(probs.min()),
        "max": float(probs.max()),
        "pct_gt_05": float((probs > 0.5).mean() * 100),
        "pct_gt_01": float((probs > 0.1).mean() * 100),
        "pct_zero": float((probs < 1e-6).mean() * 100),
    }


# ═══════════════════════════════════════════════════════════
# 5. QUESTION 1 — DIRECT VS RANK BLENDING
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("QUESTION 1 — DIRECT vs RANK BLENDING")
print("=" * 70)

weights_to_test = np.arange(0.05, 1.0, 0.05)  # wA de 0.05 à 0.95

results_q1 = []
for wA in weights_to_test:
    wB = 1.0 - wA
    
    # Direct
    p_dir = direct_blend(probs_A, probs_B, wA)
    auc_dir, n_dir = macro_auc(y_true, p_dir)
    
    # Rank
    p_rank = rank_blend(probs_A, probs_B, wA)
    auc_rank, n_rank = macro_auc(y_true, p_rank)
    
    results_q1.append({
        "wA": round(wA, 2),
        "wB": round(wB, 2),
        "direct_auc": round(auc_dir, 5),
        "rank_auc": round(auc_rank, 5),
        "delta": round(auc_dir - auc_rank, 5),
        "n_species": n_dir,
    })

df_q1 = pd.DataFrame(results_q1)
best_direct = df_q1.loc[df_q1["direct_auc"].idxmax()]
best_rank = df_q1.loc[df_q1["rank_auc"].idxmax()]

print(f"\nMeilleur DIRECT : wA={best_direct['wA']:.2f} → AUC={best_direct['direct_auc']:.5f}")
print(f"Meilleur RANK   : wA={best_rank['wA']:.2f} → AUC={best_rank['rank_auc']:.5f}")
print(f"Écart direct - rank : {best_direct['direct_auc'] - best_rank['rank_auc']:.5f}")
print(f"\nRecommandation : {'DIRECT' if best_direct['direct_auc'] >= best_rank['rank_auc'] else 'RANK'} blending")

print("\nTop 5 configurations :")
print(df_q1.sort_values("direct_auc", ascending=False).head(10).to_string(index=False))


# ═══════════════════════════════════════════════════════════
# 6. QUESTION 2 — POST-PROCESSING : QUELLES COUCHES AIDENT ?
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("QUESTION 2 — POST-PROCESSING LAYERS")
print("=" * 70)

# Prendre le meilleur blend direct comme baseline
best_wA = best_direct["wA"]
baseline = direct_blend(probs_A, probs_B, best_wA)
baseline_auc, _ = macro_auc(y_true, baseline)
print(f"\nBaseline (direct blend, wA={best_wA:.2f}) : AUC = {baseline_auc:.5f}")

# Test de chaque couche individuellement
layers = {}

# 2a. Taxon temporal smoothing
if len(texture_idx) > 0 and len(event_idx) > 0:
    print("\n--- Taxon Temporal Smoothing ---")
    for at in [0.10, 0.15, 0.20, 0.25]:
        for ae in [0.03, 0.05, 0.08]:
            p = taxon_temporal_smooth(baseline, texture_idx, event_idx, alpha_texture=at, alpha_event=ae)
            auc, n = macro_auc(y_true, p)
            layers[f"taxon_smooth_T{at}_E{ae}"] = {
                "auc": round(auc, 5), "delta": round(auc - baseline_auc, 5),
                "n_species": n, "stats": prob_stats(p)
            }

# 2b. Delta shift smoothing (uniforme)
print("--- Delta Shift Smoothing ---")
for alpha in [0.10, 0.15, 0.20, 0.25]:
    p = delta_shift_smooth(baseline, alpha=alpha)
    auc, n = macro_auc(y_true, p)
    layers[f"delta_shift_a{alpha}"] = {
        "auc": round(auc, 5), "delta": round(auc - baseline_auc, 5),
        "n_species": n, "stats": prob_stats(p)
    }

# 2c. Taxonomic smoothing
if len(multi_genus) > 0:
    print("--- Taxonomic Smoothing ---")
    for ag in [0.05, 0.10, 0.15]:
        for ac in [0.02, 0.03, 0.05]:
            p = taxonomic_smooth(baseline, multi_genus, multi_class, alpha_genus=ag, alpha_class=ac)
            auc, n = macro_auc(y_true, p)
            layers[f"tax_smooth_G{ag}_C{ac}"] = {
                "auc": round(auc, 5), "delta": round(auc - baseline_auc, 5),
                "n_species": n, "stats": prob_stats(p)
            }

# 2d. Rank-aware scaling
print("--- Rank-Aware Scaling ---")
for power in [0.3, 0.4, 0.5]:
    p = rank_aware_scale(baseline, power=power)
    auc, n = macro_auc(y_true, p)
    layers[f"rank_aware_p{power}"] = {
        "auc": round(auc, 5), "delta": round(auc - baseline_auc, 5),
        "n_species": n, "stats": prob_stats(p)
    }

# 2e. File confidence scaling
print("--- File Confidence Scaling ---")
for tk in [1, 2]:
    p = file_confidence_scale(baseline, top_k=tk)
    auc, n = macro_auc(y_true, p)
    layers[f"file_conf_top{tk}"] = {
        "auc": round(auc, 5), "delta": round(auc - baseline_auc, 5),
        "n_species": n, "stats": prob_stats(p)
    }

# Trier par AUC
sorted_layers = sorted(layers.items(), key=lambda x: x[1]["auc"], reverse=True)
print(f"\n{'Layer':<30s} {'AUC':>8s} {'Δ vs baseline':>14s} {'#sp':>5s}")
print("-" * 65)
for name, info in sorted_layers:
    delta_str = f"{info['delta']:+.5f}"
    print(f"{name:<30s} {info['auc']:>8.5f} {delta_str:>14s} {info['n_species']:>5d}")

# Identifier les couches qui aident
helpful = [(n, i) for n, i in sorted_layers if i["delta"] >= 0.001]
harmful = [(n, i) for n, i in sorted_layers if i["delta"] <= -0.001]
neutral = [(n, i) for n, i in sorted_layers if -0.001 < i["delta"] < 0.001]

print(f"\n✅ Aident (Δ ≥ +0.001) : {len(helpful)} couches")
for n, i in helpful:
    print(f"   {n}: +{i['delta']:.5f}")

print(f"\n❌ Nuient (Δ ≤ -0.001) : {len(harmful)} couches")
for n, i in harmful:
    print(f"   {n}: {i['delta']:.5f}")

print(f"\n➖ Neutres : {len(neutral)} couches")


# ═══════════════════════════════════════════════════════════
# 7. QUESTION 3 — COMBINAISON OPTIMALE DES COUCHES GAGNANTES
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("QUESTION 3 — COMBINAISON OPTIMALE (best layers stacked)")
print("=" * 70)

# Prendre les 2-3 meilleures couches et les empiler dans l'ordre
best_layer_names = [n for n, i in sorted_layers if i["delta"] > 0][:3]
print(f"Couches à empiler : {best_layer_names}")

p_stacked = baseline.copy()
for name in best_layer_names:
    if name.startswith("taxon_smooth"):
        parts = name.split("_")
        at = float(parts[2][1:])  # T0.15
        ae = float(parts[3][1:])  # E0.05
        p_stacked = taxon_temporal_smooth(p_stacked, texture_idx, event_idx, alpha_texture=at, alpha_event=ae)
    elif name.startswith("delta_shift"):
        alpha = float(name.split("a")[1])
        p_stacked = delta_shift_smooth(p_stacked, alpha=alpha)
    elif name.startswith("tax_smooth"):
        parts = name.split("_")
        ag = float(parts[2][1:])
        ac = float(parts[3][1:])
        p_stacked = taxonomic_smooth(p_stacked, multi_genus, multi_class, alpha_genus=ag, alpha_class=ac)
    elif name.startswith("rank_aware"):
        power = float(name.split("p")[1])
        p_stacked = rank_aware_scale(p_stacked, power=power)
    elif name.startswith("file_conf"):
        tk = int(name.split("top")[1])
        p_stacked = file_confidence_scale(p_stacked, top_k=tk)

stacked_auc, stacked_n = macro_auc(y_true, p_stacked)
print(f"\nAUC empilée : {stacked_auc:.5f} (Δ vs baseline = {stacked_auc - baseline_auc:+.5f})")


# ═══════════════════════════════════════════════════════════
# 8. QUESTION 4 — RATIO INTERNE MODEL_4 (ProtoSSM / SED)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("QUESTION 4 — RATIO INTERNE (simulé sur OOF)")
print("=" * 70)
print("⚠ Test limité : on simule un blend ProtoSSM/SED en utilisant")
print("  probs_A comme proxy ProtoSSM et probs_B comme proxy SED.")
print("  Les vrais ratios ne peuvent être validés qu'avec les composants")
print("  internes de Model_4, non disponibles en OOF.")
print()

# Simuler : ratio ProtoSSM (probs_A) vs SED (probs_B) dans un blend interne
# Puis re-blender avec l'autre modèle
ratios = np.arange(0.40, 0.80, 0.05)
print(f"{'Ratio P/S':>12s} {'AUC':>8s}")
print("-" * 25)
for r in ratios:
    # Simuler un Model_4' = r * ProtoSSM_proxy + (1-r) * SED_proxy
    internal = direct_blend(probs_B, probs_A, r)  # probs_B = "SED", probs_A = "ProtoSSM"
    # Ensemble final : Model_3 (probs_A) + Model_4' (internal)
    final = direct_blend(probs_A, internal, best_wA)
    auc, n = macro_auc(y_true, final)
    print(f"   {r:.2f}/{1-r:.2f}    {auc:.5f}")


# ═══════════════════════════════════════════════════════════
# 9. RÉSUMÉ ET RECOMMANDATIONS
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("RÉSUMÉ — RECOMMANDATIONS POUR V22")
print("=" * 70)

print(f"""
┌─────────────────────────────────────────────────────────────┐
│ Q1 — BLENDING                                                │
│   Meilleur DIRECT : wA={best_direct['wA']:.2f} → AUC={best_direct['direct_auc']:.5f}          │
│   Meilleur RANK   : wA={best_rank['wA']:.2f} → AUC={best_rank['rank_auc']:.5f}          │
│   ➜ Recommandation : {'DIRECT' if best_direct['direct_auc'] >= best_rank['rank_auc'] else 'RANK'}                           │
├─────────────────────────────────────────────────────────────┤
│ Q2 — POST-PROCESSING                                        │
│   Baseline AUC : {baseline_auc:.5f}                                   │
│   Couches aidantes ({len(helpful)}) : {" + ".join([n for n,_ in helpful]) if helpful else 'AUCUNE'} │
│   Couches nuisibles ({len(harmful)}) : {" + ".join([n for n,_ in harmful]) if harmful else 'AUCUNE'} │
├─────────────────────────────────────────────────────────────┤
│ Q3 — STACKING                                               │
│   Meilleures couches empilées → AUC = {stacked_auc:.5f}                    │
│   Δ vs baseline = {stacked_auc - baseline_auc:+.5f}                              │
├─────────────────────────────────────────────────────────────┤
│ Q4 — RATIO INTERNE (simulé)                                 │
│   Test sur OOF avec proxys — voir tableau ci-dessus         │
└─────────────────────────────────────────────────────────────┘
""")

# Sauvegarder les résultats
output = {
    "baseline": {
        "method": "direct",
        "wA": round(float(best_wA), 2),
        "auc": round(float(baseline_auc), 5),
    },
    "q1_direct_vs_rank": df_q1.to_dict(orient="records"),
    "q2_post_processing": {name: info for name, info in sorted_layers},
    "q3_stacked_auc": round(float(stacked_auc), 5),
    "q3_stacked_delta": round(float(stacked_auc - baseline_auc), 5),
    "recommendation": {
        "blending": "DIRECT" if best_direct["direct_auc"] >= best_rank["rank_auc"] else "RANK",
        "best_w_model3": round(float(best_direct["wA"]), 2),
        "best_w_model4": round(float(1.0 - best_direct["wA"]), 2),
        "keep_layers": [n for n, _ in helpful],
        "drop_layers": [n for n, _ in harmful],
        "expected_oof_auc": round(float(max(stacked_auc, baseline_auc)), 5),
    }
}

output_path = ROOT / "scripts" / "validate_oof_v22_results.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n✅ Résultats sauvegardés dans : {output_path}")
print("✅ Phase 0 terminée. Prêt pour V22.")
