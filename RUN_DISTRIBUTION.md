# RUN DISTRIBUTION — BirdCLEF 2026 Multi-GPU Execution Plan

> **Statut**: Plan d'execution — pret a etre deploye des que la VM A6000 est louee.  
> **Date**: 27 mai 2026  
> **Deadline competition**: 3 juin 2026 (J-7)

---

## Vue d'ensemble

```
GPU disponibles:
  [A6000]  VM louee 48 GB VRAM — 0.59€/h — illimite en temps
  [K1]     Kaggle account #1 T4 16 GB — 30h/semaine gratuites
  [K2]     Kaggle account #2 T4 16 GB — 30h/semaine gratuites
  [K3]     Kaggle account #3 T4 16 GB — 30h/semaine gratuites (optionnel)
  [Colab]  Google Colab T4 16 GB — ~25h/semaine gratuites (reco manuelle)

Total gratuit: ~85-115h/semaine de T4
Total payant:  39h A6000 = 23€
```

---

## Graphe de dependances

```
                    ┌─────────────────────────────────────────────┐
                    │         Phase 1: Preprocessing (CPU)        │
                    │  segments + norm_stats + folds              │
                    │  [DEJA FAIT localement]                     │
                    └─────────────────────┬───────────────────────┘
                                          │
            ┌─────────────────────────────┼─────────────────────────────┐
            │                             │                             │
            ▼                             ▼                             ▼
┌───────────────────────┐   ┌───────────────────────┐   ┌───────────────────────┐
│ Phase 3: Profs baseline│   │ Phase 4: Conformer     │   │ Phase 5: Perch         │
│ 20 jobs independants   │   │ (skip initial)         │   │ (skip initial)         │
│ B0×5 B3×5 SE×5 NF×5   │   │                        │   │                        │
└───────────┬───────────┘   └───────────────────────┘   └───────────────────────┘
            │
            │ DES qu'UN modele a tous ses folds termines
            ▼
┌──────────────────────────────────────────────────────────────┐
│ Phase 6 ITER 1: Pseudo-labels                                │
│  Step A: Generer PL sur soundscapes (CPU, 30 min) [A6000]    │
│  Step B: Creer dataset enrichi (CPU, instant)       [A6000]  │
│  Step C: Retrain TOUS les profs avec PL (GPU)                │
│    5 folds × 4 archis = 20 jobs independants                 │
│    B0×5 B3×5 SE×5 NF×5 — mais 40 epochs, pas 80              │
└───────────┬──────────────────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────────────────────┐
│ Phase 6 ITER 2: Power=0.65, meme pattern                      │
│  Retrain depuis checkpoints iter 1                            │
└───────────┬──────────────────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────────────────────┐
│ Phase 6 ITER 3: Power=0.55 (modeles lourds: B3+SE+NF)        │
│ Phase 6 ITER 4: Power=0.60 (final)                            │
└───────────┬──────────────────────────────────────────────────┘
            │──────────┐
            │          │ Phase 7: Rare species (1 modele × 5 folds)
            │          │ INDEPENDANT — peut tourner n'importe quand
            │◄─────────┘
            ▼
┌──────────────────────────────────────────────────────────────┐
│ Phase 8: Distillation                                         │
│  Step A: Generer teacher logits (GPU, 2h)           [A6000]  │
│  Step B: Entrainer 3 eleves × 5 folds = 15 jobs independants │
│     EffNet-B0 ×5, EffVit-b0 ×5, MnasNet ×5                   │
└───────────┬──────────────────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────────────────────┐
│ Phase 9-10-11: Fusion + ONNX + Submission (CPU)               │
│  Independants du GPU — peuvent tourner en parallele           │
│  Fusion: 10 min CPU                                           │
│  ONNX export: 5 min CPU                                       │
│  Submission notebook: test ~30 min CPU                        │
└──────────────────────────────────────────────────────────────┘
```

---

## Inventaire des jobs GPU

### Phase 3 — Profs baseline (20 jobs independants)

| # | Job | Param | Temps 4090 | Temps T4 AMP | Route | Priorite |
|---|---|---|---|---|---|---|
| B0-0 | EffNet-B0 fold 0 | 5.3M | 35 min | 45 min | K1 | ⭐⭐⭐ |
| B0-1 | EffNet-B0 fold 1 | 5.3M | 35 min | 45 min | K2 | ⭐⭐⭐ |
| B0-2 | EffNet-B0 fold 2 | 5.3M | 35 min | 45 min | Colab | ⭐⭐⭐ |
| B0-3 | EffNet-B0 fold 3 | 5.3M | 35 min | 45 min | K1 | ⭐⭐⭐ |
| B0-4 | EffNet-B0 fold 4 | 5.3M | 35 min | 45 min | K2 | ⭐⭐⭐ |
| B3-0 | EffNet-B3 fold 0 | 12.2M | 65 min | 85 min | K3 | ⭐⭐⭐ |
| B3-1 | EffNet-B3 fold 1 | 12.2M | 65 min | 85 min | Colab | ⭐⭐⭐ |
| B3-2 | EffNet-B3 fold 2 | 12.2M | 65 min | 85 min | K1 | ⭐⭐⭐ |
| B3-3 | EffNet-B3 fold 3 | 12.2M | 65 min | 85 min | K2 | ⭐⭐⭐ |
| B3-4 | EffNet-B3 fold 4 | 12.2M | 65 min | 85 min | K3 | ⭐⭐⭐ |
| SE-0 | SE-ResNeXt50 fold 0 | 27.5M | 75 min | 100 min | A6000 | ⭐⭐ |
| SE-1 | SE-ResNeXt50 fold 1 | 27.5M | 75 min | 100 min | A6000 | ⭐⭐ |
| SE-2 | SE-ResNeXt50 fold 2 | 27.5M | 75 min | 100 min | A6000 | ⭐⭐ |
| SE-3 | SE-ResNeXt50 fold 3 | 27.5M | 75 min | 100 min | A6000 | ⭐⭐ |
| SE-4 | SE-ResNeXt50 fold 4 | 27.5M | 75 min | 100 min | A6000 | ⭐⭐ |
| NF-0 | NFNet-F0 fold 0 | 71.5M | 100 min | 130 min | A6000 | ⭐⭐ |
| NF-1 | NFNet-F0 fold 1 | 71.5M | 100 min | 130 min | A6000 | ⭐⭐ |
| NF-2 | NFNet-F0 fold 2 | 71.5M | 100 min | 130 min | A6000 | ⭐⭐ |
| NF-3 | NFNet-F0 fold 3 | 71.5M | 100 min | 130 min | A6000 | ⭐⭐ |
| NF-4 | NFNet-F0 fold 4 | 71.5M | 100 min | 130 min | A6000 | ⭐⭐ |

> **Strategie P3**: B0+B3 sur Kaggle/Colab (priorite max — debute Phase 6 au plus tot).  
> SE+NF sur A6000 en parallele 2-a-2. Des que B0 a ses 5 folds → lancer Phase 6 iter 1.

### Phase 6 iter 1 — Retrain avec pseudo-labels (20 jobs)

| Job | Temps A6000 | Notes |
|---|---|---|
| B0-PL1 × 5 folds | 4h (2 en //) | Plus rapide: 40 epochs, fine-tuning |
| B3-PL1 × 5 folds | 6h (2 en //) | 40 epochs |
| SE-PL1 × 5 folds | 7h (2 en //) | 40 epochs |
| NF-PL1 × 5 folds | 9h (2 en //) | 40 epochs |

### Phase 6 iter 2-4 (20 jobs chacun, progressivement plus rapide)

> Meme pattern. Temps reduit car fine-tuning depuis iteration precedente.

### Phase 7 — Rare species (5 jobs independants)

| Job | Temps T4/Kaggle | Route |
|---|---|---|
| Rare-B0 × 5 folds | 20 min/fold | K1/K2 |

### Phase 8 — Distillation (15 jobs independants)

| Job | Temps | Route |
|---|---|---|
| Teacher logits | 2h | A6000 |
| EffNet-B0 student × 5 | 25 min/fold | K1/K2 |
| EffVit-b0 student × 5 | 20 min/fold | K3/Colab |
| MnasNet student × 5 | 20 min/fold | K1/K2 |

---

## Plan d'execution timeline

```
J-7 (27 mai) ──────────────────────────────────────────── J-1 ── J-0 (3 juin)
│                                                              │
│  HEURE 0: Lancer VM A6000 + setup                            │
│  HEURE 1: Lancer B0×5 + B3×5 sur Kaggle/Colab (parallele)   │
│           Lancer SE×2 + NF×2 en // sur A6000 (2 paires)      │
│                                                              │
│  HEURE 7: B0 5 folds termines → LANCER Phase 6 iter 1       │
│           Step A+B sur A6000 (CPU, 30 min)                   │
│           Step C: B0-PL1×5 sur Kaggle/Colab                  │
│                                                              │
│  HEURE 12: B3 5 folds termines → B3-PL1×5 sur Kaggle        │
│  HEURE 15: SE-ResNeXt50 5 folds termines sur A6000           │
│            → SE-PL1×5 sur A6000                              │
│                                                              │
│  HEURE 18: B0-PL1 termine → iter 2 si AUC > iter 1           │
│  HEURE 24: NFNet-F0 termine sur A6000 → NF-PL1               │
│                                                              │
│  HEURE 25-35: Iters PL 2-3-4 en cascade                      │
│               Phase 7 (rare) en parallele sur slot libre      │
│                                                              │
│  HEURE 36: Teacher logits (A6000, 2h)                        │
│  HEURE 38: Distillation 15 jobs sur Kaggle/Colab (3h)        │
│                                                              │
│  HEURE 41: Fusion + ONNX (CPU, 1h)                           │
│  HEURE 42: Upload modele dataset Kaggle                      │
│  HEURE 43: Test submission notebook → soumettre              │
│                                                              │
│  REPITER: ameliorations, ajustements, re-soumissions         │
│           jusqu'a J-0 23h59                                  │
└──────────────────────────────────────────────────────────────┘
```

---

## Routage des jobs par GPU

### A6000 (VM louee — 48 GB — jobs lourds + coordination)

```
Phase 3:  SE-ResNeXt50 × 5 folds  (// par paires, ~15h mur)
          NFNet-F0 × 5 folds       (// par paires, ~22h mur)
          TOTAL P3 A6000: ~23h mur (SE+NF en //)

Phase 6:  PL iter 1-4 pour SE+NF   (~13h mur par iteration)
          TOTAL P6 A6000: ~15-20h mur

Phase 8:  Teacher logits            (2h dedie)

TOTAL A6000: ~40h mur × 0.59€ = 23.60€
```

### Kaggle K1 (T4 30h/sem)

```
Phase 3:  EffNet-B0 f0, f3 | EffNet-B3 f2     (~3.5h)
Phase 6:  B0-PL f0-f1 | B3-PL f0-f1            (~3h/iter)
Phase 7:  Rare-B0 f0-f2                         (~1h)
Phase 8:  Student B0 f0-f2 | MnasNet f0-f1     (~2h)
TOTAL K1: ~15h (sur 30h quota)
```

### Kaggle K2 (T4 30h/sem)

```
Phase 3:  EffNet-B0 f1, f4 | EffNet-B3 f3     (~3.5h)
Phase 6:  B0-PL f2-f4 | B3-PL f2-f3           (~3h/iter)
Phase 7:  Rare-B0 f3-f4                         (~40min)
Phase 8:  Student B0 f3-f4 | MnasNet f2-f4     (~2h)
TOTAL K2: ~15h
```

### Kaggle K3 (optionnel — T4 30h/sem)

```
Phase 3:  EffNet-B3 f0, f4                      (~3h)
Phase 6:  B3-PL f4 | overflow                   (~2h)
Phase 8:  EffVit-b0 × 5 folds                   (~2h)
TOTAL K3: ~10h
```

### Colab (T4 ~25h/sem — reco manuelle necessaire)

```
Phase 3:  EffNet-B0 f2 | EffNet-B3 f1           (~2.5h)
Phase 6:  Overflow folds (si K1/K2 satures)     (variable)
Phase 8:  EffVit-b0 × 2 folds | overflow         (~1h)
TOTAL Colab: ~8h
```

---

## Checkpointing & Crash Recovery

### Structure des checkpoints

```
models/
├── teachers/
│   ├── efficientnet_b0/
│   │   ├── efficientnet_b0_fold0.pth
│   │   └── ...
│   ├── efficientnet_b3/
│   ├── se_resnext50/
│   └── nfnet_f0/
├── teachers_pl/
│   ├── pl_iter1/
│   │   ├── efficientnet_b0/
│   │   └── ...
│   └── pl_iter2/
├── students/
│   ├── efficientnet_b0/
│   ├── efficientvit_b0/
│   └── mnasnet_100/
└── fusion/
    ├── weights.json
    └── calibrators.pkl

data/
├── folds.pkl
├── norm_stats.pkl
└── pseudo_labels/
    ├── pseudo_labels_iter1.csv
    ├── train_enriched_iter1.csv
    └── ...
```

### Streaming automatise (A6000 → stockage permanent)

```bash
# scripts/sync_checkpoints.sh — lance en crontab toutes les 5 min
#!/bin/bash
REMOTE="gdrive:/BirdCLEF2026/checkpoints/"   # ou S3, Dropbox, Kaggle dataset...

# Upload tous les .pth modifies dans les 10 dernieres minutes
find models/ -name "*.pth" -mmin -10 | while read f; do
    rclone copy "$f" "$REMOTE/$(dirname $f)/" --progress
done

# Upload les CSVs pseudo-labels
find data/pseudo_labels/ -name "*.csv" -mmin -10 | while read f; do
    rclone copy "$f" "$REMOTE/data/pseudo_labels/"
done

# Upload weights.json
rclone copy models/fusion/weights.json "$REMOTE/models/fusion/"
```

### Reprise apres crash/expiration VM

```bash
# scripts/resume.sh
#!/bin/bash
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

### Kaggle (chaque compte)

```python
# Notebook K1/k2/k3_baseline.ipynb
# 1. git clone + pip install
# 2. Telecharger norm_stats.pkl + folds.pkl depuis dataset Kaggle
# 3. Entrainer les folds assignes
# 4. Sauvegarder .pth dans /kaggle/working/
# 5. Uploader vers dataset Kaggle partage
```

### Colab

```python
# Notebook Colab_baseline.ipynb
# 1. Monter Google Drive
# 2. git clone + pip install
# 3. Charger norm_stats/folds depuis Drive
# 4. Entrainer fold assigne
# 5. Sauvegarder .pth → Drive
```

---

## Recapitulatif cout & temps

| Ressource | Jobs | Temps mur | Cout |
|---|---|---|---|
| A6000 VM | SE+NF profs + PL + logits | ~40h | **23.60€** |
| Kaggle K1 | B0+B3 profs + PL + eleves | ~15h | Gratuit |
| Kaggle K2 | B0+B3 profs + PL + eleves | ~15h | Gratuit |
| Kaggle K3 | Overflow + EffVit | ~10h | Gratuit |
| Colab | Overflow | ~8h | Gratuit |
| **TOTAL** | **Tous les jobs** | **~45h mur max** | **23.60€** |

> **Sans A6000 (Kaggle seul):** ~55h mur, impossible en 7 jours avec quota 30h/sem/compte.  
> **Sans parallelisme (A6000 seul, 1 run):** ~80h mur, 47€, pas de redondance.  
> **Avec distribution:** 23.60€, redondance (si une ressource tombe, les autres continuent), delai ~2 jours.

---

## Actions immediate

- [ ] Creer les comptes Kaggle K2, K3
- [ ] Louer la VM A6000
- [ ] Setup rclone Google Drive ou S3 pour streaming checkpoints
- [ ] Upload norm_stats.pkl + folds.pkl + data/ vers chaque plateforme
- [ ] Preparer notebooks Kaggle pour K1, K2, K3
- [ ] Preparer notebook Colab
- [ ] Lancer `scripts/launch_a6000_jobs.sh` sur la VM
- [ ] Lancer les notebooks Kaggle/Colab
- [ ] Monitorer via `scripts/check_progress.py`
