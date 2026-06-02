# SPRINT LAST PUSH — BirdCLEF 2026 : 0.947 → 0.955+

> **Statut** : 🟡 V6 en cours d'exécution Kaggle — attente score  
> **Début** : 31 mai 2026 — J-3 avant deadline  
> **Dernière màj** : 1 juin 2026 — Root cause analysis + V6 corrective  
> **Deadline** : 3 juin 2026  
> **INTENT_MODE** : PRODUCTION_EXECUTION

---

## 1. Chronologie des Versions

| Version | Changement clé | Score LB | Statut |
|---------|---------------|----------|--------|
| EoS V2 | Model_4 seul, `direct` blend, poids [0.0, 1.0] | **0.947** | Baseline |
| EoS V3 | `rank.1`, Model_3=0.15, smoothing, species weights | ❌ IndentationError | Bug code |
| EoS V4 | Correction indentation (post-processing dans bonne cellule) | ❌ row_id mismatch | Dry-run |
| EoS V5 | Fallback dry-run row_id mismatch | **0.855** 🔴 | ÉCHEC |
| **EoS V6** | **Retour `direct`, Model_3=0.10, alpha=0.15, boost=0.10** | ⏳ En cours | — |

---

## 2. Root Cause Analysis — Pourquoi V5 a scoré 0.855

### Diagnostic

Le passage de `'direct'` à `'rank.1'` (rank averaging) est le coupable unique.

### Mécanisme de l'échec

Le **rank averaging** (`rank.1`) transforme les probabilités en rangs percentiles **par espèce** via `pd.DataFrame.rank(axis=0, pct=True)`. Conséquences :

1. **Destruction de la calibration** : Une espèce avec `max_prob = 0.001` (quasi-absente) voit ses valeurs étalées uniformément sur [0, 1]. Une espèce avec `max_prob = 0.999` subit le même traitement.

2. **Faux positifs massifs sur les espèces absentes** : 163 espèces n'ont aucun label positif dans les 708 segments OOF. Leurs probabilités Perch sont légitimement proches de 0. Le rank averaging les gonfle artificiellement.

3. **Effet cascade** : Les 28 espèces zero-shot + 25 espèces rares reçoivent des rangs gonflés → le neighborhood smoothing les propage → les species weights les amplifient → **macro AUC s'effondre**.

4. **Pourquoi QWEN recommandait ça** : Le rank averaging fonctionne quand les modèles ont des calibrations TRÈS différentes (ex: CNN vs Transformer). Ici, Model_3 et Model_4 partagent le même pipeline Perch — leurs probabilités sont déjà comparables. Le `direct` blending est strictement supérieur.

### Leçons

| Leçon | Détail |
|-------|--------|
| **Rank averaging n'est pas un silver bullet** | Valider sur OOF avant de submit |
| **Toujours dry-run avant de submit** | Les bugs V3→V4→V5 auraient été détectés |
| **708 échantillons OOF insuffisants pour apprendre** | XGBoost, Platt calibration : tous overfit |
| **Le post-processing heuristique est roi** | Smoothing, species weights : pas d'apprentissage, pas d'overfit |

---

## 3. V6 — Correctif (en cours)

| Paramètre | V5 (cassé) | V6 (correctif) | Justification |
|-----------|-----------|-----------------|---------------|
| `type_add` | `'rank.1'` 🔴 | **`'direct'`** ✅ | Préserve la calibration |
| Model_3 weight | 0.15 | **0.10** | Diversité conservative |
| Model_4 weight | 0.85 | **0.90** | Modèle dominant protégé |
| Smoothing alpha | 0.25 | **0.15** | Moins agressif |
| Species boost | 0.20 | **0.10** | Très conservateur |

**Attendu** : Retour à ≥ 0.947, avec possible +0.002-0.005 (Model_3 diversité + smoothing doux).

---

## 4. Plan d'Action Post-V6

### Si V6 ≥ 0.947 ✅

| # | Action | Durée | Gain estimé |
|---|--------|-------|-------------|
| 4.1 | Grid search poids Model_3 [0.05, 0.10, 0.15] | 3×15 min | +0.001-0.003 |
| 4.2 | Grid search smoothing alpha [0.10, 0.15, 0.20] | 3×15 min | +0.000-0.002 |
| 4.3 | Grid search species boost [0.05, 0.10, 0.15] | 3×15 min | +0.000-0.002 |
| 4.4 | Lancer PL retraining sur GPU (si temps) | 5-8h | +0.002-0.005 |
| 4.5 | Soumission finale V7 avec best params | — | — |

### Si V6 < 0.947 🔴

| # | Action |
|---|--------|
| 4.1 | Revert à V2 (Model_4 seul, aucun ajout) |
| 4.2 | Tester Model_3 seul en direct avec V2 comme base |
| 4.3 | Tester smoothing seul (sans Model_3, sans species weights) |
| 4.4 | Identifier le composant qui dégrade → soumettre le sous-ensemble gagnant |

---

## 5. Artefacts

| Artefact | Chemin local | Kaggle |
|----------|-------------|--------|
| Sprint doc | `SPRINT_LAST_PUSH.md` | — |
| Species boost dataset | `models/species_boost/` | `hellodave2035/birdclef-species-boost` |
| EoS V2 (baseline) | `notebooks/eos_submit_v2.ipynb` | `hellodave2035/eos-submit-v2` |
| EoS V5 (échec) | `notebooks/eos_submit_v5.ipynb` | `hellodave2035/eos-submit-v5` |
| **EoS V6 (en cours)** | `notebooks/eos_submit_v6.ipynb` | `hellodave2035/eos-submit-v6` |
| Filter PL script | `scripts/filter_pseudo_labels.py` | — |
| Build scripts | `scripts/apply_quick_wins*.py`, `scripts/create_v6_corrective.py` | — |
| XGBoost/Calib experiments | `models/xgb_stacker/`, `models/calibration/` | — |

---

## 6. Bilan à J-2

- **12h perdues** sur des bugs évitables (indentation, dry-run, rank averaging)
- **Leçon clé** : toujours dry-run avant submit. Ne pas appliquer une technique sans valider ses hypothèses.
- **30h GPU Kaggle** restantes, réservées pour du PL si V6 est stable
- **Baseline 0.947 solide** — V6 devrait y revenir
- **Vrai edge** : grid search hyperparam + PL (1 cycle) si temps disponible
