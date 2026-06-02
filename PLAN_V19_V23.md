# PLAN D'ACTION V19 → V23 — BirdCLEF 2026 (4 tentatives restantes)

> **Date**: 2 juin 2026 — J-1 avant deadline  
> **Score V19**: ~0.948 (V6 en attente confirmation)  
> **Objectif**: 0.955+  
> **INTENT_MODE**: PRODUCTION_EXECUTION

---

## 0. DIAGNOSTIC V19 — Ce qui cloche

### 0.1 BUG CRITIQUE dans `rank_1_add2()` et `rank_1_add3()` (CELL 11)

Le calcul des poids de rang est **INVERSÉ**. Chaque modèle reçoit un poids de rang proportionnel à la somme des poids des AUTRES modèles :

```python
# rank_1_add2 — BUG (lignes 6082-6083)
wSED   = PROTOSSM_W / (PROTOSSM_W + SED_W)   # poids du 1er modèle → rang du 2e
wPROTO = SED_W      / (PROTOSSM_W + SED_W)   # poids du 2e modèle → rang du 1er

# rank_1_add3 — BUG (lignes 6119-6121)
wSED     = (PROTO_w1 + PROTO_w2) / total      # poids des 2 autres → rang du 3e
wPROTO_1 = (SED_w + PROTO_w2)    / total      # poids des 2 autres → rang du 1er
wPROTO_2 = (SED_w + PROTO_w1)    / total      # poids des 2 autres → rang du 2e
```

**Conséquence**: Avec `_weights = [0.03, 0.10, 0.87]`:
- Model_4 (LB 0.947, meilleur) → poids de rang = **13%** 😱
- Model_2 (LB 0.917, pire)    → poids de rang = **97%** 😱

C'est la **cause racine** de l'échec V5 (score 0.855). Le meilleur modèle était quasi-ignoré dans le blend.

**Correction**: Chaque modèle doit avoir un poids de rang proportionnel à son PROPRE poids :
```python
# CORRECTION
wSED     = SED_W     / total    # Model_4 → 87%
wPROTO_1 = PROTO_w1  / total    # Model_2 → 3%
wPROTO_2 = PROTO_w2  / total    # Model_3 → 10%
```

⚠️ **Ce bug n'explique pas tout**: le SPRINT_LAST_PUSH identifie aussi un problème fondamental du rank averaging sur des modèles partageant le même backbone Perch (probabilités déjà calibrées similairement). Même corrigé, le rank blending n'est pas garanti de battre le `direct`.

### 0.2 Model_2 : poids 3%, LB 0.917, contribution quasi-nulle

Avec `direct` blending, Model_2 apporte 3% de son signal. Vu son LB faible (0.917 vs 0.947-0.948 pour l'ensemble), sa contribution est au mieux neutre, au pire légèrement négative (bruit).

**Recommandation**: Dropper Model_2 (poids = 0 ou retirer de la liste).

### 0.3 Le paramètre `xSED` dans `solutions` est MORT

```python
{'Model':'Model_4', 'xSED':[0.595, 0.405], ...}
```

Ce paramètre est parsé dans `_xsed` (CELL 3) mais **jamais utilisé** dans le code de blend de Model_4. Les ratios sont hardcodés :
- 2-way (ProtoSSM/SED): `0.595 / 0.405` → ligne ~5887
- 3-way (+BirdNET): `0.50 / 0.30 / 0.20` → ligne ~5886

Pour changer ces ratios, il faut modifier directement les lignes de code.

### 0.4 BirdNET est conditionnel → souvent désactivé

Si le modèle BirdNET `.tflite` n'est pas attaché au notebook, Model_4 fonctionne en mode 2-way dégradé. Le 3-way blend (ProtoSSM + SED + BirdNET) est potentiellement plus performant car BirdNET couvre les espèces non-mappées par Perch.

### 0.5 Over-smoothing : 4 couches de lissage en cascade

1. Model_2: Gaussian smoothing (sigma=0.65) dans les logits
2. Model_3: Delta-shift smooth (alpha=0.20) + rank-aware scaling
3. CELL 15: Taxon-aware temporal smoothing (alpha_texture=0.25, alpha_event=0.08)
4. CELL 17: Taxonomic smoothing genus+class (alpha_genus=0.15, alpha_class=0.05)

**Risque**: Les détections brèves (oiseaux à appels pulsés) sont noyées.

---

## 1. LES DEUX PISTES DE TRAVAIL

### PISTE A 🔵 — Sécurité : Nettoyage + Optimisation Conservative

**Principe**: Modifications minimales, risque faible, gain modéré mais certain.

| # | Action | Code à changer | Impact |
|---|--------|---------------|--------|
| A1 | **Drop Model_2** | `solutions['Models']` → retirer Model_2 | Supprime bruit |
| A2 | **Grid search poids Model_3 vs Model_4** | Tester [0.05/0.95], [0.10/0.90], [0.15/0.85], [0.20/0.80] | +0.001-0.003 |
| A3 | **Réduire over-smoothing** | CELL 15: alpha_texture=0.15, alpha_event=0.05; CELL 17: alpha_genus=0.10, alpha_class=0.03 | +0.001-0.002 |
| A4 | **Simplifier la calibration** | CELL 16: réduire C à 10.0, ajouter validation split | +0.000-0.001 |
| A5 | **Garder `direct` blending** | Aucun changement à `type_add` | Stabilité |

**Gain estimé**: +0.003 à +0.006  
**Risque**: ⭐ FAIBLE (toutes les modifications sont réversibles)

### PISTE B 🟠 — Ambition : Upgrade Model_4 + Activation BirdNET

**Principe**: Améliorer le modèle dominant (87% du poids) plutôt que l'ensemble.

| # | Action | Code à changer | Impact |
|---|--------|---------------|--------|
| B1 | **Activer BirdNET dans Model_4** | S'assurer que le modèle `.tflite` est attaché au dataset Kaggle | +0.002-0.005 |
| B2 | **Optimiser le ratio 3-way ProtoSSM/SED/BirdNET** | Modifier les ratios hardcodés ligne ~5886: tester [0.50/0.25/0.25], [0.45/0.35/0.20], [0.55/0.25/0.20] | +0.002-0.004 |
| B3 | **Optimiser les seuils des 5 gates** | Ajuster les hard thresholds des gates 1-5 (noise, temporal, spike, mirroring) | +0.001-0.003 |
| B4 | **Augmenter le poids de Model_3** | Passer à 0.15-0.20 (LB 0.928 a de la valeur diversifiante) | +0.001-0.002 |
| B5 | **Corriger le bug rank_1_add3 + tester rank blending corrigé** | Fix poids + comparer rank vs direct en OOF | +0.000-0.003 |

**Gain estimé**: +0.004 à +0.010  
**Risque**: ⭐⭐ MOYEN (BirdNET peut ne pas être attaché, les gates sont sensibles)

---

## 2. SÉQUENCE DES 4 TENTATIVES

```
Tentative 1 (V20)        Tentative 2 (V21)        Tentative 3 (V22)        Tentative 4 (V23)
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ PISTE A          │     │ PISTE B          │     │ COMPARAISON      │     │ FINE-TUNE        │
│                  │     │                  │     │                  │     │                  │
│ • Drop Model_2   │     │ • BirdNET ON     │     │ • V20 vs V21     │     │ • Grid search    │
│ • Grid weight M3 │     │ • Optim ratios   │     │ • Si V21>V20:    │     │   poids final    │
│ • Reduce smooth  │     │ • Tune gates     │     │   adopter B      │     │ • Optim seuils   │
│ • Keep direct    │     │ • Fix rank bug   │     │ • Si V20>V21:    │     │ • Dernière calib │
│                  │     │                  │     │   adopter A      │     │ • SUBMIT FINAL   │
│ Score cible:     │     │ Score cible:     │     │ • Si ≈: keep A   │     │ Score cible:     │
│ 0.950-0.952      │     │ 0.951-0.955     │     │   (plus simple)  │     │ 0.953-0.958      │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Détail par tentative

#### TENTATIVE 1 — V20 (Piste A)

**Fichier**: `notebooks/eos_submit_v20.py` (copie de V19 modifiée)

**Modifications**:
```python
# CELL 2 — Remplacer tout le bloc solutions
solutions = {
 'type_add' :'direct',
 'Models'   : [
  # Model_2 RETIRÉ — LB 0.917 trop faible, poids 3% = bruit
  {'Model':'Model_3','subm':'subm_3.csv','weight':0.10,'xSED':['    ,    '],'LB':'0.928'},
  {'Model':'Model_4','subm':'subm_4.csv','weight':0.90,'xSED':[0.595,0.405],'LB':'0.947'}
 ]
}
```

```python
# CELL 15 — Réduire over-smoothing
# alpha_texture: 0.25 → 0.15, alpha_event: 0.08 → 0.05
submission = taxon_smooth(submission, texture_cols, event_cols, alpha_texture=0.15, alpha_event=0.05)
```

```python
# CELL 17 — Réduire taxonomic smoothing
# alpha_genus: 0.15 → 0.10, alpha_class: 0.05 → 0.03
alpha_genus = 0.10
alpha_class = 0.03
```

**À NE PAS CHANGER**: `type_add='direct'`, Model_4 inchangé

**Validation**: Vérifier que le notebook tourne en dry-run (3 lignes output).

#### TENTATIVE 2 — V21 (Piste B)

**Fichier**: `notebooks/eos_submit_v21.py` (copie de V19 modifiée)

**Prérequis**: Attacher le dataset BirdNET (shadiakiki1/birdnet-analyzer) au notebook Kaggle.

**Modifications**:

1. **Activer BirdNET** — Déjà géré par le code (auto-détection), mais vérifier que le dataset est bien attaché.

2. **Optimiser les ratios de blend interne Model_4** (ligne ~5886-5888):
```python
# Remplacer les ratios hardcodés
if rank_birdnet is not None:
    pred = (rank_proto * 0.50) + (rank_sed * 0.25) + (rank_birdnet * 0.25)  # V21: plus de BirdNET
else:
    pred = (rank_proto * 0.60) + (rank_sed * 0.40)  # V21: ajusté
```

3. **Corriger le bug `rank_1_add3`** (pour les tests OOF, pas pour le submit):
```python
# CORRECTION du calcul des poids de rang
wSED     = SED_w     / (PROTO_w1 + PROTO_w2 + SED_w)     # poids propre → rang propre
wPROTO_1 = PROTO_w1  / (PROTO_w1 + PROTO_w2 + SED_w)
wPROTO_2 = PROTO_w2  / (PROTO_w1 + PROTO_w2 + SED_w)
```

4. **Ajuster les gates** — Optionnel, seulement si le temps le permet:
   - Gate 1 (noise suppression): seuil `p_sed < 0.05` → `p_sed < 0.08`
   - Gate 3 (SED spike): `rank_sed > 0.95` → `rank_sed > 0.92`

#### TENTATIVE 3 — V22 (Décision)

**Comparer V20 et V21**:
- Si V21 > V20 de +0.002 ou plus → adopter Piste B comme base
- Si V20 > V21 → adopter Piste A comme base
- Si égalité (écart < 0.001) → garder Piste A (plus simple, moins de risques)

**Combinaison possible**: Si les deux pistes apportent des gains indépendants, V22 = V21 + les améliorations de V20 (drop Model_2, réduction smoothing).

#### TENTATIVE 4 — V23 (Fine-tune Final)

Sur la base gagnante (V20 ou V21), appliquer:
1. **Grid search fin des poids** avec granularité 0.02 (9 combinaisons si 2 modèles)
2. **Optimisation des alphas de smoothing** par taxon
3. **Vérification OOF** sur le cache disque (`full_oof_meta_features.npz`)
4. **Dernière soumission**

---

## 3. TABLEAU DES MODIFICATIONS PAR VERSION

| Version | Base | Model_2 | Poids M3/M4 | Blend | Smoothing | BirdNET | Gates |
|---------|------|---------|-------------|-------|-----------|---------|-------|
| **V19** | — | 0.03 | 0.10/0.87 | direct | Standard | Auto | Standard |
| **V20** | V19 | **DROP** | **0.10/0.90** | direct | **Réduit** | Auto | Standard |
| **V21** | V19 | 0.03 | 0.10/0.87 | direct | Standard | **FORCÉ ON** | **Optimisé** |
| **V22** | best(V20,V21) | — | — | — | — | — | — |
| **V23** | V22 | — | **Grid search** | — | **Per-taxon** | — | — |

---

## 4. CHECK-LIST AVANT CHAQUE SOUMISSION

- [ ] Le notebook tourne en dry-run (3 lignes) sans erreur
- [ ] `submission.csv` a le bon nombre de colonnes (235)
- [ ] Les colonnes sont dans l'ordre du `sample_submission.csv`
- [ ] Pas de NaN, pas de valeurs hors [0, 1]
- [ ] `submission.csv` est bien dans `/kaggle/working/`
- [ ] Le notebook termine en < 90 minutes

---

## 5. RÉFÉRENCES CROISÉES

| Document | Contenu clé |
|----------|------------|
| `SPRINT_LAST_PUSH.md` | Historique V2→V6, root cause du fail V5 (rank averaging) |
| `OPusfeedback.md` | Analyse exhaustive des 3 modèles et leviers d'amélioration |
| `SUBMISSION_MASTER_GUIDE.md` | Format submission, pièges dry-run, row_id |
| `TODO_BIRDCLEF3.md` | Plan complet original (Phase 0-8) |

---

## Annexe A — Le bug rank_1_add3 en détail

```python
# ═══════════════════════════════════════════════════════
# CODE ACTUEL (BUG) — lignes 6106-6125 de eos_submit_v19.py
# ═══════════════════════════════════════════════════════
def rank_1_add3():
    PROTO_w1 = _weights[0]  # 0.03 (Model_2)
    PROTO_w2 = _weights[1]  # 0.10 (Model_3)
    SED_w    = _weights[2]  # 0.87 (Model_4)
    ...
    # BUG: poids croisés → meilleur modèle reçoit le plus petit poids de rang
    wSED     = (PROTO_w1 + PROTO_w2) / ((PROTO_w1+PROTO_w2) + SED_W)  # = 0.13
    wPROTO_1 = (SED_w    + PROTO_w2) / ((PROTO_w1+PROTO_w2) + SED_W)  # = 0.97
    wPROTO_2 = (SED_w    + PROTO_w1) / ((PROTO_w1+PROTO_w2) + SED_W)  # = 0.90
    ...
    pred = wSED * rSED + rPROTO_1 * wPROTO_1 + rPROTO_2 * wPROTO_2
    #     = 0.13*rSED  + 0.97*rPROTO_1       + 0.90*rPROTO_2
    #     = Model_4 à 13% + Model_2 à 97% + Model_3 à 90%  🔴🔴🔴

# ═══════════════════════════════════════════════════════
# CODE CORRIGÉ
# ═══════════════════════════════════════════════════════
    wSED     = SED_w     / (PROTO_w1 + PROTO_w2 + SED_w)  # = 0.87
    wPROTO_1 = PROTO_w1  / (PROTO_w1 + PROTO_w2 + SED_w)  # = 0.03
    wPROTO_2 = PROTO_w2  / (PROTO_w1 + PROTO_w2 + SED_w)  # = 0.10
    ...
    pred = wSED * rSED + rPROTO_1 * wPROTO_1 + rPROTO_2 * wPROTO_2
    #     = 0.87*rSED  + 0.03*rPROTO_1       + 0.10*rPROTO_2
    #     = Model_4 à 87% + Model_2 à 3% + Model_3 à 10%  ✅✅✅
```

**Note importante**: Même corrigé, le rank blending reste risqué car les 3 modèles partagent le backbone Perch (calibrations similaires). Le SPRINT_LAST_PUSH a documenté que le rank averaging « détruit la calibration » sur les espèces rares. La correction du bug rend le rank blending _utilisable_ mais pas nécessairement _supérieur_ au direct blending.

---

## Annexe B — Pondérations OOF pour validation locale

Si vous avez accès au cache OOF (`full_oof_meta_features.npz`), vous pouvez tester les configurations localement :

```python
import numpy as np
from sklearn.metrics import roc_auc_score

def evaluate_ensemble_weights(weights, oof_preds_list, y_true):
    """Évalue une combinaison de poids sur les prédictions OOF."""
    blended = sum(w * p for w, p in zip(weights, oof_preds_list))
    keep = y_true.sum(axis=0) > 0
    return roc_auc_score(y_true[:, keep], blended[:, keep], average='macro')

# Exemple pour 2 modèles
for w3 in [0.05, 0.10, 0.15, 0.20]:
    w4 = 1.0 - w3
    auc = evaluate_ensemble_weights([w3, w4], [oof_m3, oof_m4], y_oof)
    print(f"Model_3={w3:.2f} Model_4={w4:.2f} → OOF AUC={auc:.5f}")
```
