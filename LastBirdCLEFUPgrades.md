LastBirdCLEFUPgrades.md
=======================

## KIMI 

User: voici mon notebook pour la compétition BirdCleféàé6. Mon score plafonne a 0.947. Que faire pour améliorer mes résultats?
Kimi: Je vois que c'est un notebook BirdClef 2026 très sophistiqué avec un ensemble de modèles (ProtoSSM + SED + BirdNET) qui plafonne à 0.947. Voici une analyse des goulots d'étranglement et des pistes concrètes pour dépasser ce plafond.

---

## 🔍 Diagnostic rapide de tes points faibles

### 1. **Le poids de l'ensemble est mal optimisé**

```python
# === CELL 2 ===
solutions = {
 'type_add' :'direct',  # ← Tu utilises "direct" (moyenne pondérée)
 'Models'   : [
  {'Model':'Model_3','subm':'subm_3.csv','weight':0.10,'xSED':['    ,    '],'LB':'0.928'},
  {'Model':'Model_4','subm':'subm_4.csv','weight':0.90,'xSED':[0.595,0.405],'LB':'0.947'}
 ]
}
```

**Problème** : Tu mets 90% sur Model_4 (LB 0.947) et seulement 10% sur Model_3 (LB 0.928). Mais `type_add='direct'` fait une moyenne pondérée des **probabilités**, ce qui est sous-optimal pour les compétitions de ranking.

**Fix immédiat** :
```python
'type_add': 'rank.1'  # ← Passe en blending de rangs (percentile)
```

Le rank-blending est presque toujours meilleur quand les modèles ont des distributions de scores différentes.

---

### 2. **Model_2 est complètement retiré — peut-être trop tôt ?**

```python
# Model_2 RETIRÉ — LB 0.917 trop faible, poids 3% = bruit négligeable
```

**Reconsidère** : Même un modèle à 0.917 peut apporter de la **diversité** utile en ensemble. Le bruit n'est pas négligeable si :
- Il détecte des espèces que les autres ratent
- Il a une architecture complémentaire (SED vs frame-level vs embedding)

**Test** : Réactive Model_2 avec un poids faible (0.05-0.10) en rank-blending, pas en direct.

---

### 3. **Le SED Model_4 a un problème de fusion interne**

Dans Model_4, tu fais :
```python
# 50% ProtoSSM / 25% SED / 25% BirdNET (rank blend)
pred = (rank_proto * 0.50) + (rank_sed * 0.25) + (rank_birdnet * 0.25)
```

Mais si BirdNET n'est pas attaché (`USE_BIRDNET=False`), tu retombes sur :
```python
pred = (rank_proto * 0.60) + (rank_sed * 0.40)
```

**Problème** : Ce 60/40 est **arbitraire** et n'a pas été optimisé sur OOF. Tu as des fonctions pour ça (`optimize_ensemble_weight`, `sweep_ensemble_weight`) mais elles ne sont pas utilisées pour la fusion finale ProtoSSM/SED.

---

### 4. **La post-processing "Gate" est trop agressive**

```python
# Gate 1: Noise suppression
fake_only = (p_proto > 0.50) & (p_sed < 0.05)
pred = np.where(fake_only, (1.0 - 0.08) * pred + 0.08 * rank_proto, pred)
```

Ces gates **figent** les décisions avec des seuils durs (0.50, 0.05, 0.88, 0.75...) qui n'ont pas été validés sur OOF. Un modèle qui fait 0.947 a probablement besoin de **moins** de post-processing, pas plus.

---

## 🚀 Actions concrètes pour gagner des points

### A. Optimisation de l'ensemble final (gain potentiel : +0.003-0.008)

**1. Passe en rank-blending au niveau supérieur** :

```python
# === CELL 2 ===
solutions = {
 'type_add': 'rank.1',  # ← CHANGE ICI
 'Models': [
  {'Model':'Model_3','subm':'subm_3.csv','weight':0.15,'xSED':['    ,    '],'LB':'0.928'},
  {'Model':'Model_4','subm':'subm_4.csv','weight':0.85,'xSED':[0.595,0.405],'LB':'0.947'}
 ]
}
```

**2. Ou mieux : fais un grid search du poids optimal** :

```python
# Ajoute ça dans CELL 11 ou avant CELL 12
import numpy as np
from sklearn.metrics import roc_auc_score

def find_optimal_weight(subm1_path, subm2_path, y_true, weight_range=np.arange(0.1, 0.9, 0.05)):
    """Grid search du meilleur poids de rank-blending sur OOF."""
    df1 = pd.read_csv(subm1_path)
    df2 = pd.read_csv(subm2_path)
    cols = [c for c in df1.columns if c != "row_id"]
    
    # Align
    df2 = df2.set_index("row_id").loc[df1["row_id"]].reset_index()
    p1 = df1[cols].values
    p2 = df2[cols].values
    
    best_auc, best_w = 0, 0.5
    for w in weight_range:
        # Rank blend
        r1 = pd.DataFrame(p1).rank(pct=True).values
        r2 = pd.DataFrame(p2).rank(pct=True).values
        blended = w * r1 + (1-w) * r2
        auc = roc_auc_score(y_true, blended, average='macro')
        if auc > best_auc:
            best_auc, best_w = auc, w
    return best_w, best_auc
```

### B. Amélioration du SED (Model_2) — le plus gros levier

Ton SED est un EfficientNet-B0 avec distillation Perch. Les limitations :

| Problème | Solution |
|---------|----------|
| Backbone B0 trop petit | Passe à `tf_efficientnet_b2` ou `eca_nfnet_l0` |
| Pas d'audio augmentation forte | Ajoute SpecAugment agressif, MixUp à 0.7, CutMix |
| Distillation fixe | Fais de la **self-distillation** ou distille un modèle plus fort |
| Inférence : blend clip + frame_max arbitraire | Optimise `0.5*clip + 0.5*fmax` sur OOF |

**Dans le code SED (Cell 5)** :
```python
# Au lieu de :
p_blend = 0.5 * p_clip + 0.5 * p_fmax

# Fais un grid search :
def optimize_clip_fmax_ratio(model, val_wavs, Y_val, mel_transform):
    ratios = np.arange(0.1, 0.9, 0.1)
    best_auc, best_r = 0, 0.5
    for r in ratios:
        preds = r * predict_clip + (1-r) * predict_fmax
        auc = macro_auc(Y_val, preds)
        if auc > best_auc:
            best_auc, best_r = auc, r
    return best_r
```

### C. ProtoSSM v5 — sous-exploité

Tu as des configs V17/V18 mais :
- `d_model=320, n_ssm_layers=4` — as-tu assez de données pour entraîner ça ?
- Le **Mixup+CutMix** est défini mais pas utilisé dans `train_proto_ssm_single`
- La **SWA** (Stochastic Weight Averaging) est configurée mais pas testée sur OOF

**Vérifie** : Est-ce que `ProtoSSM_PATH` et `ResidualSSM_PATH` pointent vers des modèles entraînés avec les **dernières configs V18** ?

```python
# Dans CELL 13, vérifie :
print(f"ProtoSSM config used: d_model={ssm_cfg['d_model']}, n_layers={ssm_cfg['n_ssm_layers']}")
# Si ça affiche d_model=128, tu utilises une vieille config !
```

### D. BirdNET — opportunité manquée

Ton code BirdNET est conditionné à la présence du modèle :
```python
if _bn_model_path is None:
    USE_BIRDNET = False
```

**Si tu n'as pas attaché BirdNET**, tu perds un modèle complémentaire qui :
- Fonctionne à 48kHz (meilleure résolution fréquentielle)
- A été entraîné sur 6000+ espèces (meilleure généralisation)
- Détecte bien les sons "texturaux" (insectes, amphibiens)

**Action** : Attache le dataset `shadiakiki1/birdnet-analyzer` et vérifie que `_find_birdnet_model()` le trouve.

---

## 📊 Plan d'attaque priorisé

| Priorité | Action | Gain estimé | Difficulté |
|---------|--------|-------------|------------|
| 1 | Passer `type_add='rank.1'` | +0.002-0.005 | ⭐ Facile |
| 2 | Grid search poids Model_3/Model_4 | +0.001-0.003 | ⭐ Facile |
| 3 | Attacher BirdNET + vérifier activation | +0.003-0.008 | ⭐⭐ Moyen |
| 4 | Optimiser ratio clip/fmax dans SED | +0.001-0.003 | ⭐⭐ Moyen |
| 5 | Upgrader SED backbone (B0→B2) | +0.002-0.005 | ⭐⭐⭐ Dur |
| 6 | Réduire post-processing agressif (Gates) | +0.001-0.004 | ⭐⭐ Moyen |
| 7 | Réactiver Model_2 avec poids faible | +0.000-0.003 | ⭐ Facile |

---

## 🎯 Le plus important

Ton score 0.947 suggère que tu es **très proche du plafond de performance** de tes modèles actuels. Les gains viendront de :

1. **Meilleure fusion** (rank > direct, poids optimisés)
2. **Plus de diversité** (BirdNET actif, Model_2 réactivé)
3. **Moins de post-processing empirique** (les Gates à seuils fixes tuent plus qu'ils n'aident à ce niveau)



============================ /// ==============================

## QWEN

Félicitations pour ce score de **0.947**, c'est un excellent résultat qui vous place dans le haut du classement. À ce niveau de performance, les gains marginaux ne viennent plus de l'ajout de modèles aléatoires, mais d'une **optimisation chirurgicale** de l'ensemble, du blending et, surtout, de la prévention du surapprentissage (overfitting) au Public LB.

Voici une analyse détaillée de votre notebook et des pistes concrètes, classées par priorité, pour franchir le cap des 0.950+.

---

### 🚨 Priorité 1 : Maîtriser le Post-Processing (Risque majeur de Private LB Shakeup)
Votre notebook applique **4 couches successives** de post-traitement à la fin (Cells 15, 16, 17) :
1. Lissage temporel conscient des taxons (V20)
2. Calibration de Platt par espèce
3. Lissage taxonomique au niveau du genre
4. Lissage taxonomique au niveau de la classe

**Le danger** : Empiler autant de règles heuristiques (surtout avec des `alpha` ajustés manuellement) est la cause n°1 des effondrements de score entre le Public et le Private LB.
**Action** :
*   **Validez chaque étape sur votre OOF (Out-of-Fold)** : Avant d'appliquer un lissage ou une calibration, calculez le Macro AUC sur vos données OOF *avec* et *sans* cette étape. Si une étape n'améliore pas le score OOF d'au moins +0.001, **supprimez-la**.
*   **Calibration de Platt** : Votre code utilise `C=100.0`. C'est très élevé (presque pas de régularisation). Essayez une grille `C=[1.0, 10.0, 100.0]` et validez sur l'OOF.
*   **Sonotype Mirroring (Gate 4)** : Vérifiez que les paires hardcoded (`47158son15`, `47158son16`, etc.) sont toujours valides dans la taxonomie *spécifique* de BirdCLEF 2026. Une erreur ici tue le score d'espèces rares.

---

### 🎯 Priorité 2 : Optimisation du Blending et de l'Ensemble
Actuellement, votre ensemble est dominé à **90% par Model_4** (LB 0.947) et 10% par Model_3 (LB 0.928). Model_2 a été retiré.

**Actions** :
1.  **Optimisation des poids par Grid Search OOF** : Ne gardez pas `0.90 / 0.10` figé. Utilisez vos prédictions OOF de Model_3 et Model_4 pour faire une recherche de grille (ou une optimisation type `scipy.optimize.minimize`) sur les poids du *rank blending*. Les poids optimaux sur l'OOF sont souvent différents de ceux du Public LB.
2.  **Réintroduisez un "Model_2" diversifié** : Puisque Model_3 et Model_4 sont tous deux basés sur Perch/ProtoSSM, leurs erreurs sont corrélées. Pour qu'un ensemble fonctionne, il faut de la diversité. Entraînez un nouveau modèle avec une architecture radicalement différente :
    *   Un modèle **AST** (Audio Spectrogram Transformer) ou **BEATs**.
    *   Un modèle purement CNN (ex: `tf_efficientnet_b3` ou `convnext_base`) *sans* distillation Perch, pour qu'il apprenne des features spectrogrammes brutes différentes.
    *   Même si ce nouveau modèle fait seulement 0.930 en solo, s'il est non-corrélé, il peut pousser l'ensemble à 0.950+.

---

### ⚙️ Priorité 3 : Améliorations Architecturales (Model_4 & SED)

**Pour le modèle SED (Distilled SED)** :
*   **Backbone** : Vous utilisez `tf_efficientnet_b0.ns_jft_in1k`. C'est un modèle très léger. Si la mémoire le permet, passez à `tf_efficientnet_b3` ou `b4`. Le gain de capacité est souvent directement traduit en +0.005 sur le LB.
*   **Pooling** : Vous utilisez `GeMFreqPool` avec `p_init=3.0`. Essayez de rendre ce paramètre `p` entièrement entraînable sans le clamper trop tôt, ou testez un `GeM` 2D (temps et fréquence) au lieu d'uniquement fréquence.

**Pour le ProtoSSM** :
*   **Mixup/CutMix** : Votre code a une fonction `mixup_cutmix` mais elle semble principalement configurée pour l'embedding. Assurez-vous que le `mixup_alpha` (actuellement 0.4) est bien optimal. Testez `0.2` et `0.6` sur l'OOF.
*   **Focal Loss** : `focal_gamma` est à 2.5. Pour les classes très rares, un gamma plus élevé (3.0 ou 4.0) peut aider, mais au détriment des classes communes. Une approche par classe (gamma plus élevé pour les classes < 20 échantillons) pourrait être bénéfique.

---

### 🔍 Priorité 4 : Débogage des "Gates" et de BirdNET

1.  **BirdNET est-il actif ?** : Votre code contient cette ligne : `if not USE_BIRDNET: ... zero submission saved`. Si le modèle TFLite n'est pas trouvé, vous retombez sur un blend 60/40 au lieu de 50/25/25.
    *   *Action* : Vérifiez les logs. Si BirdNET n'est pas chargé, corrigez le chemin d'accès ou uploadez le dataset `shadiakiki1/birdnet-analyzer`. Même un BirdNET faible apporte un signal orthogonal précieux pour les espèces non mappées.
2.  **Les "Gates" (Portes) de décision** :
    *   `Gate 1 (Noise suppression)` : `p_proto > 0.50 & p_sed < 0.05`. Ce seuil de 0.05 pour SED est très agressif. SED peut être silencieux sur des appels très brefs mais réels. Assouplissez ce seuil ou validez-le sur l'OOF.
    *   `Gate 3 (SED spike)` : Le seuil a été abaissé de 0.95 à 0.92. C'est bien, mais assurez-vous que cela n'injecte pas trop de faux positifs pour les espèces de fond (bruit).

---

### 📋 Plan d'action concret pour votre prochain submit

Ne changez pas tout en même temps. Procédez par étapes validées sur l'OOF :

1.  **Étape 1 (Sécurité)** : Désactivez temporairement les lissages taxonomiques (Cell 17) et le lissage temporel (Cell 15). Soumettez. Si le score monte ou reste stable, c'est que ces post-processings *overfittaient* le Public LB. Gardez la version la plus simple qui fonctionne.
2.  **Étape 2 (Blending)** : Exécutez un script de grid search sur vos prédictions OOF pour trouver les poids exacts `(w_proto, w_sed)` qui maximisent le Macro AUC OOF. Mettez à jour `_weights` dans la `CELL 2`.
3.  **Étape 3 (Diversité)** : Ajoutez un 3ème modèle non-Perch (ex: AST ou EfficientNet-B3) dans l'ensemble. Même avec un poids de 15%, la décorrélation des erreurs fera miraculeusement monter le score.
4.  **Étape 4 (BirdNET)** : Forcez le chargement de BirdNET. Vérifiez que `_bn_active` est bien `True` dans les logs.

**Conseil final** : À 0.947, votre ennemi n'est plus le sous-apprentissage (underfitting), mais le **surapprentissage au Public LB (overfitting)**. Toute modification qui améliore le Public LB de +0.001 mais *détériore* votre score OOF local doit être immédiatement rejetée. Faites confiance à votre validation croisée (OOF).