# RUN DISTRIBUTION — BirdCLEF 2026 Plan d'Exécution (PIVOT PERCH)

> **Statut**: 🔄 PIVOT vers approche Perch — 0 VM GPU nécessaire, 0 upload de specs  
> **Date**: 30 mai 2026 — J-4 avant deadline  
> **Deadline competition**: 3 juin 2026 (J-4)  
> **Décision clé**: Abandon CNN from scratch → Perch ONNX + SSM léger sur Kaggle gratuit

---

## 🎯 Pourquoi le Pivot ?

| Contrainte | Approche originale (CNN) | Nouvelle approche (Perch) |
|---|---|---|
| VM GPU | A6000 payante (~2-5$/h) | ❌ Pas besoin |
| Upload Kaggle | 130 Go de specs .npy | ~500 Mo ONNX ✅ |
| Entraînement | 4×5 folds CNN lourds (jours) | 5-fold SSM léger (3-5h) ✅ |
| 28 espèces zero-shot | Données XC à collecter | Déjà dans Perch ✅ |
| Score potentiel | 0.92+ (si tout réussi) | 0.93-0.95 (notebooks publics existants) ✅ |

---

## Vue d'ensemble — ÉTAT RÉEL AU 30 MAI

```
Ressources disponibles:
  [VM-GC]   VM Google Cloud CPU — ⚠️ Inaccessible (arrêtée/expirée)
            Preprocessing terminé: ~235 000 specs .npy (~135 GB)
            Données sur GDrive ? (à vérifier)
            
  [K1]      Kaggle account principal T4 16 GB — 30h/semaine — ✅ RESET dispo
            Utilisable pour entraînement Perch+SSM
            
  [C1]      Google Colab T4 16 GB — ~25h/semaine gratuites — backup

  [Kaggle]  Ressources gratuites sur la plateforme:
            - Google Perch ONNX (rishikeshjani, 427 MB)
            - Modèles publics pré-entraînés (tonylica, 775 MB)
            - Notebooks publics 0.934-0.947 LB
```

---

### Chronologie réelle (30 mai 2026)

```
28 mai 22:44  → VM GCloud démarrée, preprocessing 14 workers
29 mai ~01:45 → Preprocessing terminé (~235k .npy)
29 mai        → Décision: les 130 Go ne passeront jamais sur Kaggle
29 mai        → Recherche: Perch + modèles publics sur Kaggle
30 mai        → PIVOT acté: approche Perch ONNX + SSM léger
30 mai        → Notebooks créés: submission_perch.py, train_perch_ssm.py, ensemble_blend.py
J-4 à J-1     → Exécution sur Kaggle K1/Colab
```

---

## 📦 Stratégie Perch (3 Piliers)

### Pilier 1 ⭐ — Forker l'ensemble EoS de nina2025 (0.947 LB)
- Notebook: `nina2025/birdclef-2026-ensemble-of-solutions`
- Utilise datasets publics Kaggle uniquement
- Runtime: 12m49s — tourne sur GPU Kaggle gratuit
- Si les datasets privés sont inaccessibles → Pilier 2

### Pilier 2 ⭐ — Perch + SSM Léger (autonome, reproductible)
- **Étape 1**: Extraire embeddings Perch pour tous les segments train
- **Étape 2**: Entraîner ProtoSSM 5-fold (3-5h sur T4)
- **Étape 3**: Pseudo-labeling itératif + Power Transform
- **Étape 4**: Exporter classifier_ssm.onnx (~20 MB)
- **Étape 5**: Inférence: Perch ONNX → embedding → classifier ONNX → submission
- Notebook: `notebooks/train_perch_ssm.py`
- Soumission: `notebooks/submission_perch.py`

### Pilier 3 — Ensemble + Modèles Publics
- Fusionner: Perch direct logits + SSM classifier + modèles publics (tonylica)
- Optimiser poids via scipy.optimize
- Calibration Platt optionnelle
- Notebook: `notebooks/ensemble_blend.py`

---

## 📋 Plan d'Action J-4 à J-1

### Jour 1 (30 mai) — Setup Kaggle
- [ ] Uploader `rishikeshjani/perch-onnx-for-birdclef-2026` comme dataset Kaggle
- [ ] Forker et tester `nina2025/birdclef-2026-ensemble-of-solutions`
- [ ] Si datasets privés bloquants → passer au Pilier 2
- [ ] Lancer `train_perch_ssm.py` sur K1 (extraction embeddings + training SSM)

### Jour 2 (31 mai) — Entraînement + Pseudo-labels
- [ ] Finaliser 5-fold SSM training
- [ ] Lancer pseudo-labeling iteration 1
- [ ] Ré-entraîner avec 50% réel + 50% pseudo
- [ ] Exporter classifier_ssm.onnx

### Jour 3 (1 juin) — Ensemble & Soumission Test
- [ ] Fusionner les prédictions (Perch + SSM + publics)
- [ ] Optimiser poids d'ensemble
- [ ] Exécuter `submission_perch.py` en mode test
- [ ] Vérifier runtime < 90 min CPU
- [ ] Soumettre et vérifier score public

### Jour 4 (2 juin) — Derniers Ajustements
- [ ] Si score < 0.92: ajuster poids, ajouter calibration
- [ ] Si score >= 0.93: soumission finale
- [ ] Option: intégrer vos specs pour un CNN complémentaire si temps

---

## 📂 Fichiers Clés (Nouveaux)

| Fichier | Rôle |
|---|---|
| `notebooks/submission_perch.py` | Notebook soumission CPU (Perch ONNX + classifieur) |
| `notebooks/train_perch_ssm.py` | Entraînement SSM sur embeddings Perch (GPU Kaggle) |
| `notebooks/ensemble_blend.py` | Fusion et optimisation des poids d'ensemble |
| `notebooks/submission_notebook.py` | Ancien notebook CNN (conservé comme fallback) |

### Dataset Kaggle à Préparer
```
birdclef-perch-models/
├── perch_v2.onnx              # Depuis rishikeshjani/perch-onnx-for-birdclef-2026
├── labels.csv                 # Depuis le même dataset
├── classifier_ssm.onnx        # Votre classifieur entraîné (~20 MB)
├── config.pkl                 # Paramètres du classifieur
├── weights.json               # Poids d'ensemble
└── calibrators.pkl            # Optionnel
```

---

## ⚠️ Points de Vigilance

1. **Reset quota K1**: Vérifier que le quota GPU est bien reset (~29-30 mai)
2. **Datasets privés nina2025**: Le notebook EoS référence des "Private Dataset" → peuvent être inaccessibles
3. **Perch ONNX performance**: ~427 MB, inférence CPU — à benchmarker
4. **130 Go de specs**: Les garder sur GDrive comme sécurité, ne pas uploader sur Kaggle
5. **Fallback**: Si tout échoue, l'ancien `submission_notebook.py` peut être adapté avec les modèles publics (tonylica)

---

## 🔗 Références Kaggle

| Ressource | URL |
|---|---|
| Perch ONNX | `rishikeshjani/perch-onnx-for-birdclef-2026` |
| Modèles publics | `tonylica/birdclef-2026-model` |
| EoS Ensemble (0.947) | `nina2025/birdclef-2026-ensemble-of-solutions` |
| Iter-Pseudo Perch+SED (0.934) | `needless090/birdclef-2026-iter-pseudo-perch-sed-lb-0.934-s` |
| Perch+ProtoSSM (0.925) | `imaadmahmood/birdclef-2026-perch-v2-protossm-0-925` |
| Reproduce Perch+SSM | `hideyukizushi/bird26-reproduce-perch-protossm-resssm-inf-train` |

# 1. Telecharger les derniers checkpoints
rclone copy "$REMOTE/models/" models/ --progress

# 2. Verifier quels folds sont termines
python scripts/check_progress.py

# 3. Relancer les folds manquants
python scripts/launch_missing_folds.py
```

### Script de verification de progression

```python
# scripts/check_progress.py
"""Scanne models/ pour determiner quels jobs sont termines,
   lesquels sont en cours (fichier .lock) et lesquels restent a faire.
   Genere un rapport et le fichier de reprise.
"""
```

---

## Fichiers a fournir pour chaque GPU

### A6000 (VM)

```bash
git clone <repo> && cd __birdclef-v3
bash scripts/setup_a6000.sh        # install dependencies
bash scripts/sync_checkpoints.sh   # start checkpoint streaming cron
bash scripts/launch_a6000_jobs.sh  # lance SE+NF en // + coordination
```

### Kaggle K1

```python
# Notebook K1_baseline.ipynb
# 1. git clone + pip install
# 2. Telecharger norm_stats.pkl + folds.pkl depuis dataset Kaggle
# 3. Entrainer les folds assignes
# 4. Sauvegarder .pth dans /kaggle/working/
# 5. Uploader vers dataset Kaggle prive
```

### Colab C1 (et C2)

```python
# Utiliser notebooks/colab_train.ipynb.py
# 1. Ouvrir Colab → coller le contenu du fichier
# 2. Modifier BACKBONE et FOLD dans Cell 2
# 3. Run All
# 4. Checkpoint auto-sauvegarde sur Google Drive
# 5. Si crash: relancer avec le fold suivant (le script skip les checkpoints existants)
```

---

## Recapitulatif cout & temps

| Ressource | Jobs | Temps mur | Cout |
|---|---|---|---|
| A6000 VM | SE+NF profs + PL + logits | ~40h | **23.60€** |
| Kaggle K1 | B0+B3 + PL + eleves | ~20h | Gratuit |
| Colab C1 | B0+B3 + PL + eleves | ~15h | Gratuit |
| Colab C2 | Overflow + EffVit | ~8h | Gratuit |
| **TOTAL** | **Tous les jobs** | **~45h mur max** | **23.60€** |

> **Sans A6000 (Kaggle+Colab seuls):** ~55h mur, faisable si C1+C2 tiennent.  
> **Sans parallelisme (A6000 seul, 1 run):** ~80h mur, 47€.  
> **Avec distribution:** 23.60€, redondance, delai ~2 jours.

---

## Actions immediate

- [ ] Louer la VM A6000 (coquille vide Linux)
- [ ] Creer dossier `BirdCLEF2026/` sur Google Drive
- [ ] Placer `kaggle.json` a la racine du repo
- [ ] Uploader `norm_stats.pkl` + `folds.pkl` sur Google Drive (depuis ton PC local)
- [ ] Lancer `bash scripts/setup_a6000.sh` sur la VM
- [ ] Lancer `bash scripts/launch_a6000_jobs.sh` sur la VM
- [ ] Ouvrir Colab C1 → coller `notebooks/colab_train.ipynb.py` → Run All
- [ ] Ouvrir Colab C2 (si 2e compte) → idem, choisir autres folds
- [ ] Kaggle K1 → notebook GPU avec les folds restants
- [ ] Monitorer via `python scripts/check_progress.py`

**Colab te demandera :** ouvrir le notebook, changer `BACKBONE` et `FOLD` dans la Cell 2, puis Run All. Checkpoint auto-sauvegarde sur Drive. Si le runtime crash, relance avec le fold suivant. Le script skip les checkpoints deja sur Drive.
