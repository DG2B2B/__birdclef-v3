# TODO BIRDCLEF 2026 — Plan d'Exécution Complet

> **Objectif** : 1ère place Kaggle BirdCLEF 2026 (macro ROC-AUC maximal)  
> **Deadline** : 3 juin 2026  
> **Contrainte clé** : inférence CPU < 90 min, ~16 Go RAM, **234 espèces** (dont 28 zero-shot), format WIDE (235 colonnes)  

Ce document fusionne `1.thnkings.md`, `2.globalPlan.md` (corrigé), `3.multiAgentOrders.md` et `4.guideKaggle.md`.  
Chaque tâche est ordonnée et actionnable.

---

## Phase 0 — Environnement & Données

- [ ] **0.1** Configurer le notebook Kaggle d'entraînement (GPU P100 ou T4x2, Docker `nvidia/pytorch:24.02-py3`)
- [ ] **0.2** Installer les dépendances : `torch`, `torchaudio`, `torchvision`, `timm`, `librosa`, `scikit-learn`, `onnx`, `onnxruntime`, `pandas`, `numpy`, `iterative-stratification`
- [ ] **0.3** Télécharger les données officielles (`train_audio/`, `train_soundscapes/`, `test_soundscapes/`, `train_metadata.csv`, `taxonomy.csv`, `sample_submission.csv`, `perch_meta`)
- [ ] **0.4** Télécharger les données Xeno-Canto additionnelles partagées sur le forum (format `.mp3` 32 kHz)
- [ ] **0.5** Analyser `sample_submission.csv` → confirmé : **235 colonnes** (row_id + **234 espèces**), format WIDE, row_id = `{soundscape}_{end_time}`
- [ ] **0.6** Explorer les données : distribution des espèces (1 à 499 clips/classe), ratio de silence, bruit de fond
- [ ] **0.7** ⚠️ **Identifier et documenter les 28 espèces zero-shot** (3 Amphibia + 25 Insect sonotypes 47158son01-25) — elles n'ont aucun fichier dans `train_audio/` mais doivent être prédites. Plan d'action : données Xeno-Canto + Perch embeddings.

---

## Phase 1 — Prétraitement Audio & Feature Engineering

- [ ] **1.1** Script `src/preprocessing.py` :
  - `resample_and_segment(audio_path, sr=32000, duration=5)` → liste de tableaux numpy
  - `resample_and_segment(audio_path, sr=32000, duration=20)` → pour chunks longs (option 1ère place 2025)
  - Filtre de silence : suppression segments avec amplitude max < -40 dB
  - Nettoyage artefacts : détection voix humaines, silences anormaux
- [ ] **1.2** Extraction de caractéristiques `src/features.py` :
  - `log_mel_spectrogram(y, sr, n_mels=224, hop_length=320, win_length=800)` → array (224, T)
  - Tronquer / padder à 224 trames → (224, 224)
  - Duplication 3 canaux → (3, 224, 224)
  - **PCEN** (Per-Channel Energy Normalization) comme alternative au Log-Mel
  - Calcul mean/std sur 5000 segments aléatoires, sauvegarde `data/norm_stats.pkl`
- [ ] **1.3** Dataset PyTorch `src/dataset.py` :
  - Lecture CSV (chemins + labels multi-hot 234 espèces)
  - Augmentations : `SpecAugment` (time_mask=20, freq_mask=10) + **Background Mix** (pluie/insectes/vent) + MixUp additif (blending fixe 0.5)
  - Retourne (spectrogramme 3×224×224, label binaire 234)
- [ ] **1.4** Dataset pour chunks 20s (pour SED) :
  - Input: (3, 224, 512) pour 20 secondes (selon specs 1er 2025)
  - Mêmes augmentations
- [ ] **1.5** Notebook de validation Phase 1 : charger un batch, afficher specs, vérifier shapes

---

## Phase 2 — Validation Croisée GroupKFold

- [ ] **2.1** Implémenter `src/validation.py` :
  - **GroupKFold** par `site_id` / `location` / `recordist` (PAS stratified — évite le leakage de bruit de fond)
  - 5 folds, seed 42, splits fixes sauvegardés dans `data/folds.pkl`
  - "Replay protocol" : 1 fold fantôme pour tests rapides d'idées
- [ ] **2.2** Métrique `macro_auc()` : calcul ROC-AUC par espèce, moyenne macro
- [ ] **2.3** TTA validation : 5 crops aléatoires + flip horizontal

---

## Phase 3 — Modèles Professeurs (Niveau 1) avec SED Head

- [ ] **3.1** Architecture SED `src/models_sed.py` :
  - Classe `BirdSEDClassifier(model_name, n_classes=234, pretrained=True)`
  - Backbones : `tf_efficientnet_b3.ns_jft_in1k`, `se_resnext50_32x4d`, `nfnet_f0`, `tf_efficientnet_b0.ns_jft_in1k`
  - **SED Head** : GeM frequency pooling (p=3) → classification framewise (pas global pooling classique)
  - Sortie : (batch, T, 234) → agrégation temporelle pour éval
- [ ] **3.2** Script d'entraînement `src/train_teachers.py` :
  - 5-fold GroupKFold
  - Loss : `FocalLoss(alpha=0.25, gamma=2.0)`
  - Optimiseur : `AdamW(lr=1e-3, weight_decay=1e-4)`
  - Scheduler : `CosineAnnealingLR` + warmup linéaire 5%
  - Early stopping : patience 10 epochs sur val loss
  - Epoch max : 80
  - Sauvegarde : `models/teachers/{model_name}_fold{i}.pth`
- [ ] **3.3** Config YAML `configs/train_teachers.yaml`
- [ ] **3.4** Entraîner EfficientNet-B3 (5 folds) → score cible > 0.87
- [ ] **3.5** Entraîner SE-ResNeXt50 (5 folds)
- [ ] **3.6** Entraîner NFNet-F0 (5 folds)
- [ ] **3.7** Entraîner EfficientNet-B0 (5 folds)
- [ ] **3.8** Notebook de suivi : courbes loss/AUC par fold, tableau de bord `PROGRESS.md`

---

## Phase 4 — Modèle Conformer (Temporel)

- [ ] **4.1** Implémenter `src/conformer.py` :
  - Conformer léger : 6 blocs, dim 256, 4 têtes d'attention, kernel conv 31
  - Entrée : (batch, T=100, mel=224) pour séquences de ~10s
  - Sortie : attention pooling multi-tête → tête linéaire 234
  - Gestion du padding/masking
- [ ] **4.2** Adapter le dataset pour générer des séquences de 100 trames
- [ ] **4.3** Script `src/train_conformer.py` : même config optim (FocalLoss, AdamW, scheduler)
- [ ] **4.4** Entraîner Conformer (5 folds) → score cible > 0.88
- [ ] **4.5** Sauvegarde : `models/teachers/conformer_fold{i}.pth`

---

## Phase 5 — Modèle Perch (Embeddings Pré-entraînés, Optionnel)

- [ ] **5.1** Charger le modèle Perch v2 ONNX (`perch_v2_no_dft.onnx`)
- [ ] **5.2** Extraire les embeddings pour tous les segments d'entraînement
- [ ] **5.3** Ajouter une tête de classification fine-tunée (MLP ou Linear)
- [ ] **5.4** Entraîner et évaluer → intégrer au pool professeurs si bénéfique

---

## Phase 6 — Pseudo-Étiquetage Itératif avec Power Transform

- [ ] **6.1** Script `src/pseudo_label.py` :
  - Prédiction sur les soundscapes de test avec le meilleur fold de chaque architecture professeur
  - Moyenne des probabilités → seuillage → pseudo-labels binaires
  - **Power Transform** pour nettoyer les labels bruités (technique 1er 2025)
  - Sauvegarde `data/pl_train.csv`
- [ ] **6.2** Itération 1 :
  - Dataset = 50% données originales + 50% pseudo-labelisées
  - MixUp blending fixe 0.5 (ratio 1:1 dans les batches)
  - Stochastic Depth (`drop_path_rate=0.15`)
  - Ré-entraînement de TOUS les professeurs (FocalLoss)
  - Score cible : gain ≥ +0.02 vs baseline
- [ ] **6.3** Itération 2 :
  - Générer de nouveaux pseudo-labels avec les modèles de l'itération 1
  - Power Transform avec valeur ajustée (ex: 0.65)
  - Ré-entraînement
- [ ] **6.4** Itération 3 :
  - Power Transform (ex: 0.55)
  - Ajouter les modèles plus lourds (NFNet-F0, SE-ResNeXt50) à ce stade
- [ ] **6.5** Itération 4 (si gain) :
  - Power Transform (ex: 0.6)
  - Ensemble des professeurs final

---

## Phase 7 — Modèle Dédié Espèces Rares / Zero-Shot / Amphibia-Insecta

- [ ] **7.1** Lister les 28 espèces zero-shot : 3 Amphibia (1491113, 25073, 517063) + 25 Insect sonotypes (47158son01-25)
- [ ] **7.2** Télécharger données Xeno-Canto ciblées pour ces 28 espèces (priorité absolue) + espèces avec < 10 clips
- [ ] **7.3** Entraîner un EfficientNet-B0 dédié sur ce sous-ensemble (zero-shot + espèces rares)
- [ ] **7.4** Alternative/complément : utiliser les embeddings Perch pour les 28 espèces (le modèle Perch couvre probablement ces taxons)
- [ ] **7.5** Évaluer l'apport : cible +0.003 à +0.01 LB

---

## Phase 8 — Distillation Professeurs → Élèves

- [ ] **8.1** Sélection architectures élèves : EfficientNet-B0, EfficientVit-b0, MnasNet-100
- [ ] **8.2** Script `src/distillation.py` :
  - `DistillationLoss(alpha=0.7, T=2.0)` = 0.7×BCEWithLogitsLoss + 0.3×KLDivLoss(log_softmax(pred/T), softmax(teacher_logits/T)) × T²
  - `generate_teacher_logits(dataset)` : logits moyens sur folds et architectures professeurs
  - Sauvegarde : `models/students/{name}_fold{i}.pth`
- [ ] **8.3** Entraîner EfficientNet-B0 élève (5 folds)
- [ ] **8.4** Entraîner EfficientVit-b0 élève (5 folds)
- [ ] **8.5** Entraîner MnasNet-100 élève (5 folds)

---

## Phase 9 — Fusion, Stacking & Calibration

- [ ] **9.1** Script `src/fusion.py` :
  - Prédictions OOF (out-of-fold) de chaque élève sur la validation
  - Optimiser les poids `w_i` (Σ w_i = 1, w_i ≥ 0) via `scipy.optimize.minimize` ou Optuna
  - Maximiser macro-AUC sur validation
  - Sauvegarde `models/weights.json`
- [ ] **9.2** Stacking (optionnel) :
  - Méta-modèle léger (régression logistique) entraîné sur prédictions OOF des élèves
  - Validation stricte : pas de leakage entre folds
- [ ] **9.3** Calibration Platt par espèce :
  - `sklearn.calibration.CalibratedClassifierCV(method='sigmoid')` sur scores OOF
  - Sauvegarde `models/calibrators.pkl`
  - Valider que ça n'abîme pas le score (omettre si dégradation)
- [ ] **9.4** Évaluer l'ensemble final sur validation → score cible > 0.92

---

## Phase 10 — Export ONNX & Optimisation CPU

- [ ] **10.1** Script `src/export_onnx.py` :
  - Convertir chaque élève `.pth` → `.onnx` (`opset_version=17`, dynamic batch size)
  - Vérifier la sortie avec `onnxruntime` (précision < 1e-5 d'écart vs PyTorch)
  - Sauvegarde : `models/onnx/effnet_b0.onnx`, `effvit_b0.onnx`, `mnasnet.onnx`
- [ ] **10.2** OpenVINO (alternative) :
  - Convertir les modèles ONNX → OpenVINO IR
  - Benchmark vs ONNX Runtime sur CPU → choisir le plus rapide
- [ ] **10.3** Benchmark CPU :
  - Mesurer temps d'inférence pour 1000 segments
  - Cible : < 15 ms par segment (soit < 60 min pour l'ensemble du test)
- [ ] **10.4** Quantification dynamique (si bénéfique et sans perte de précision)

---

## Phase 11 — Notebook de Soumission Final

- [ ] **11.1** Structurer `submission_notebook.ipynb` (inference-only) :
  ```
  ╔═ CELL 1: Imports + Config ═══════════════════════════╗
  ║ os.environ['KAGGLE_NO_INTERNET'] = '1'              ║
  ║ TEST_DIR = "/kaggle/input/birdclef-2026/test_soundscapes" ║
  ║ MODEL_DIR = "/kaggle/input/your-model-dataset"      ║
  ║ SAMPLE_SUB = "/kaggle/input/birdclef-2026/sample_submission.csv" ║
  ╚══════════════════════════════════════════════════════╝
  ```
- [ ] **11.2** Chargement modèles ONNX/OpenVINO avec `CPUExecutionProvider`
- [ ] **11.3** Fonction de prétraitement (log-mel + normalisation, identique à l'entraînement)
- [ ] **11.4** Boucle d'inférence par soundscape :
  - Découpage en segments de 5s
  - **Sliding window avec overlap** + lissage temporel `[0.1, 0.2, 0.4, 0.2, 0.1]`
  - Batch inference (batch_size=64)
  - Fusion pondérée + calibration
  - `gc.collect()` tous les 10 fichiers
- [ ] **11.5** Génération du CSV **format WIDE** :
  - 1 ligne par segment : `{soundscape_id}_{end_time_seconds}` + 234 colonnes de probabilités
  - Vérification : même nombre de lignes que `sample_submission.csv`, mêmes colonnes
  - Pas de NaN, pas de valeurs négatives
  - Sauvegarde : `/kaggle/working/submission.csv`
- [ ] **11.6** Tests hors-ligne :
  - Exécuter avec `test_soundscapes/` vide/réduit → ne doit pas planter
  - Mesurer le temps total → doit être < 90 min (viser < 60 min)
  - Vérifier mémoire < 16 Go

---

## Phase 12 — Dataset Modèle Kaggle (Upload)

- [ ] **12.1** Préparer le dossier à uploader :
  ```
  birdclef-2026-model/
  ├── effnet_b0.onnx
  ├── effvit_b0.onnx
  ├── mnasnet.onnx
  ├── scaler.pkl          (mean/std normalisation)
  ├── weights.json        (poids de fusion)
  └── calibrators.pkl     (optionnel)
  ```
- [ ] **12.2** Taille totale < 1 Go (utiliser quantification si nécessaire)
- [ ] **12.3** Upload via Kaggle UI ou API : `kaggle datasets create -p ./birdclef-2026-model`

---

## Phase 13 — Soumission & Itérations Finales

- [ ] **13.1** Commit du notebook → "Save and Run All" → "Submit to Competition"
- [ ] **13.2** Vérifier le score public après ~15 minutes
- [ ] **13.3** Analyser les erreurs : quelles espèces performent mal ?
- [ ] **13.4** Itérations d'amélioration :
  - Ajuster les poids de fusion
  - Ajouter/retirer la calibration
  - Tester PCEN vs Log-Mel (entraîner un modèle PCEN dédié à ajouter à l'ensemble)
  - Tester chunks 20s vs 5s
  - Ajouter des données Xeno-Canto supplémentaires
- [ ] **13.5** Dernière soumission avant le 3 juin 2026

---

## Phase 14 — Monitoring & Documentation Continue

- [ ] **14.1** Maintenir `PROGRESS.md` avec :
  - Scores macro-AUC par fold pour chaque modèle (professeurs et élèves)
  - Gain apporté par chaque composant (SED, pseudo-labels, Power Transform, Conformer, distillation, fusion)
  - Temps d'inférence mesuré
- [ ] **14.2** Tableau comparatif des itérations de pseudo-labeling (Power values, scores)

---

## Dépendances entre phases

```
Phase 0 ──► Phase 1 ──► Phase 2 ──► Phase 3 (CNN)
                                ├─► Phase 4 (Conformer) ──┐
                                ├─► Phase 5 (Perch) ──────┤
                                └──────────────────────────┼──► Phase 6 (Pseudo-labels)
                                                           │        │
                                                           │        ▼
                                                           │    Phase 7 (Rares)
                                                           │        │
                                                           └────────┼──► Phase 8 (Distillation)
                                                                    │        │
                                                                    │        ▼
                                                                    │    Phase 9 (Fusion)
                                                                    │        │
                                                                    │        ▼
                                                                    │    Phase 10 (ONNX)
                                                                    │        │
                                                                    │        ▼
                                                                    │    Phase 11 (Notebook)
                                                                    │        │
                                                                    │        ▼
                                                                    │    Phase 12 (Upload)
                                                                    │        │
                                                                    │        ▼
                                                                    │    Phase 13 (Soumission)
                                                                    │        │
                                                                    └────────┴──► Phase 14 (Monitoring)
```

---

## Checklist Finale Avant Soumission

- [ ] Les modèles ONNX se chargent sans erreur
- [ ] Shapes entrée/sortie correspondent (`input`: `(batch, 3, 224, 224)`)
- [ ] `providers = ['CPUExecutionProvider']`
- [ ] Normalisation (mean/std) identique à l'entraînement
- [ ] CSV format WIDE : `row_id` + 234 colonnes espèces, même shape que `sample_submission.csv`
- [ ] Pas de NaN, pas de valeurs négatives
- [ ] Pas de `pip install` dans le notebook de soumission
- [ ] Fonctionne sans Internet (`os.environ['KAGGLE_NO_INTERNET'] = '1'`)
- [ ] Mémoire < 16 Go
- [ ] Temps d'exécution < 90 min (mesuré)
