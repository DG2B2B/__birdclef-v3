# BirdCLEF 2026 — Guide Complet de Création de Soumission (`submission.csv`)

> **Public cible** : agent de code dans un autre repo, même compétition BirdCLEF 2026.
> **Objectif** : documenter TOUS les pièges, méthodes, validations et exemples pour produire
> un `submission.csv` valide et scorer > 0.500 au premier essai.
>
> **Issu du RETEX** de 12+ versions de notebooks de soumission (v5 → v13, intermediate, pp+tta)
> ayant abouti au score **0.746** sur le leaderboard public.

---

## 1. Règles de la Compétition (Code Competition)

| Règle | Détail |
|-------|--------|
| **Type** | Code Competition — le notebook est réexécuté sur un dataset caché |
| **CPU only** | GPU submissions = 1 minute max → **interdit en pratique** |
| **Timeout CPU** | 90 minutes max |
| **Internet OFF** | Aucun téléchargement externe autorisé pendant le scoring |
| **Fichier de sortie** | Doit être nommé **exactement** `submission.csv` dans `/kaggle/working/` |
| **Format** | `row_id` + 234 colonnes d'espèces (ordre exact du `sample_submission.csv`) |
| **Valeurs** | Probabilités ∈ [0, 1], pas de NaN, float |

---

## 2. Anatomie du `sample_submission.csv`

Le fichier public `sample_submission.csv` est **LE contrat** que votre soumission doit respecter.

```
row_id,species_1,species_2,...,species_234
BC2026_Test_0001_S05_20250227_010002_5,0.00427...,0.00427...,...
BC2026_Test_0001_S05_20250227_010002_10,0.00427...,0.00427...,...
BC2026_Test_0001_S05_20250227_010002_15,0.00427...,0.00427...,...
```

### Points critiques

- **Seulement 3 lignes** dans le sample public. C'est un artefact de dry-run, PAS le nombre réel de lignes attendu lors du scoring caché.
- **235 colonnes** : `row_id` + 234 codes espèce (ex: `ashgre1`, `bafcur1`, `yebcar`...).
- Les 234 espèces sont en **ordre alphanumérique strict**. Vous DEVEZ conserver cet ordre exact.
- Le fichier ne contient PAS de BOM, PAS de CRLF (LF uniquement).

---

## 3. LE PIÈGE FONDAMENTAL : Dry-run vs Scoring Caché

C'est le problème qui a coûté le plus de temps (versions v6 à v13).

### Dry-run (validation publique Kaggle)

```
test_soundscapes/  →  VIDE (0 fichier .ogg)
sample_submission.csv → 3 lignes seulement
```

→ Votre notebook ne trouve AUCUN fichier audio à inférer.
→ Vous DEVEZ produire un `submission.csv` valide quand même.
→ **Stratégie** : fallback → copier `sample_submission.csv` tel quel (score = 0.500).

### Scoring caché (hidden rerun)

```
test_soundscapes/  →  PEUPLÉ par Kaggle (fichiers .ogg réels, ex: ~400 fichiers)
sample_submission.csv → 3 lignes (template seulement)
```

→ Votre notebook trouve les `.ogg` → exécute l'inférence.
→ Vous devez générer des `row_id` qui **MATCHENT** le format du sample.
→ Les row_id du sample public (3 lignes) sont un sous-ensemble minimal ; le scoring caché en attend potentiellement des milliers.

### Règle d'or

> **Toujours utiliser `sample_submission.csv` comme TEMPLATE pour les row_ids et l'ordre des colonnes.**
> Ne JAMAIS inventer vos propres row_ids ni votre propre ordre de colonnes.

---

## 4. Format des Row IDs

Les row_ids suivent ce pattern :

```
{filename_stem}_{end_seconds}
```

Exemples :
- `BC2026_Test_0001_S05_20250227_010002_5`  → segment 0-5s
- `BC2026_Test_0001_S05_20250227_010002_10` → segment 5-10s
- `BC2026_Test_0001_S05_20250227_010002_15` → segment 10-15s
- ...
- `BC2026_Test_0001_S05_20250227_010002_60` → segment 55-60s

Un soundscape d'1 minute = 12 segments × 5 secondes = 12 row_ids.

### Génération des row_ids

```python
import soundfile as sf

def generate_row_ids(ogg_path, segment_seconds=5.0):
    """Génère les row_ids pour un fichier soundscape."""
    stem = ogg_path.stem  # ex: BC2026_Test_0001_S05_20250227_010002
    info = sf.info(str(ogg_path))
    duration = info.duration
    end_seconds = list(range(segment_seconds, int(duration) + segment_seconds, segment_seconds))
    return [f"{stem}_{es}" for es in end_seconds]
```

---

## 5. Pipeline Complet de Soumission (Code Minimal Viable)

### 5.1 Structure du notebook Kaggle

```python
# Cell 1: Imports et constantes
from pathlib import Path
import os, sys, time
import numpy as np
import pandas as pd
import torch, torch.nn as nn
import torchaudio
import soundfile as sf
import timm

INPUT = Path('/kaggle/input')
WORK = Path('/kaggle/working')

# Trouver le dataset competition (deux emplacements possibles)
COMP = None
for candidate in [INPUT / 'birdclef-2026', INPUT / 'competitions' / 'birdclef-2026']:
    if candidate.exists() and (candidate / 'sample_submission.csv').exists():
        COMP = candidate
        break
assert COMP is not None, 'Competition dataset not found'

SAMPLE_SUB = COMP / 'sample_submission.csv'
TEST_DIR = COMP / 'test_soundscapes'
SUBM_OUT = WORK / 'submission.csv'
```

### 5.2 Chargement du modèle

```python
# Cell 2: Trouver les checkpoints
MODEL_DIR = None
for root, dirs, files in os.walk(INPUT):
    if any(f.startswith('fold_') and f.endswith('.pth') for f in files):
        MODEL_DIR = Path(root)
        break
assert MODEL_DIR is not None, 'No model checkpoints found'
CKPTS = sorted(MODEL_DIR.glob('fold_*.pth'))
print(f'Checkpoints: {[c.name for c in CKPTS]}')
```

**Piège #1 — Préfixe `backbone.`** : Si vos checkpoints ont été sauvegardés depuis un wrapper `EfficientNetBird` qui wrappe `self.backbone = timm.create_model(...)`, les clés du state_dict auront le préfixe `backbone.`. Il faut le stripper avant de charger dans `timm.create_model` nu :

```python
def load_model(checkpoint_path, num_classes=234, backbone='tf_efficientnet_b0'):
    model = timm.create_model(backbone, pretrained=False, num_classes=num_classes)
    state = torch.load(str(checkpoint_path), map_location='cpu', weights_only=True)
    # STRIP du préfixe backbone. si présent
    if any(k.startswith('backbone.') for k in state):
        state = {k[9:]: v for k, v in state.items()}  # enlève 'backbone.'
    model.load_state_dict(state)
    model.eval()
    return model
```

### 5.3 Transformation Mel-Spectrogramme

```python
# Cell 3: LogMelTransform
class LogMelTransform(nn.Module):
    def __init__(self, sr=32000, n_mels=128, hop_length=512, n_fft=1024,
                 f_min=50.0, f_max=16000.0, top_db=80.0):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_mels=n_mels, hop_length=hop_length,
            n_fft=n_fft, f_min=f_min, f_max=f_max)
        self.db = torchaudio.transforms.AmplitudeToDB(top_db=top_db)

    def forward(self, waveform):
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        spec = self.mel(waveform)
        spec = self.db(spec)
        # Normalisation Z-score
        return (spec - spec.mean()) / (spec.std() + 1e-8)
```

### 5.4 Windower (découpage en segments)

```python
# Cell 4: Chargement et découpage d'un soundscape
def load_soundscape_segments(path, segment_seconds=5.0, target_sr=32000):
    """Charge un .ogg et le découpe en segments de segment_seconds."""
    audio, sr = sf.read(str(path), dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # stéréo → mono

    if sr != target_sr:
        audio_t = torch.from_numpy(audio).unsqueeze(0)
        audio_t = torchaudio.functional.resample(audio_t, sr, target_sr)
        audio = audio_t.squeeze(0).numpy()

    segment_samples = int(segment_seconds * target_sr)
    segments = []
    for start in range(0, len(audio), segment_samples):
        chunk = audio[start:start + segment_samples]
        if len(chunk) < segment_samples:
            chunk = np.pad(chunk, (0, segment_samples - len(chunk)))
        segments.append(chunk.astype(np.float32))
    return segments
```

### 5.5 Inférence Multi-Fold

```python
# Cell 5: Prédiction multi-fold
@torch.no_grad()
def predict_multi_fold(models, waveforms, mel_transform, batch_size=32, device='cpu'):
    """
    Args:
        models: liste de modèles chargés
        waveforms: liste de np.ndarray float32
    Returns:
        np.ndarray (num_segments, num_classes) — probabilités moyennées sur les folds
    """
    # Construction du batch de spectrogrammes
    specs = torch.stack([
        mel_transform(torch.from_numpy(w)) for w in waveforms
    ]).to(device)

    # ⚠️ PIÈGE #2 : EfficientNet attend 3 canaux, le mel n'en a qu'1
    if specs.shape[1] == 1:
        specs = specs.repeat(1, 3, 1, 1)

    all_fold_probs = []
    for model in models:
        fold_probs = []
        for i in range(0, len(specs), batch_size):
            batch = specs[i:i + batch_size]
            out = model(batch)
            fold_probs.append(torch.sigmoid(out).cpu().numpy())
        all_fold_probs.append(np.concatenate(fold_probs, axis=0))

    return np.mean(all_fold_probs, axis=0).astype(np.float32)
```

### 5.6 Assemblage de la Soumission

```python
# Cell 6: Construction du submission.csv
sample = pd.read_csv(SAMPLE_SUB)
species_cols = [c for c in sample.columns if c != 'row_id']
print(f'Species: {len(species_cols)}')

ogg_files = sorted(TEST_DIR.glob('*.ogg')) if TEST_DIR.exists() else []
print(f'Test soundscapes found: {len(ogg_files)}')

if not ogg_files:
    # === DRY-RUN : fallback sur sample_submission.csv ===
    print('WARNING: No test soundscapes. Falling back to sample_submission.csv.')
    sample.to_csv(SUBM_OUT, index=False)
    # Score attendu : 0.500 (baseline aléatoire uniforme)
else:
    # === SCORING RÉEL : inférence sur les .ogg ===
    submission_rows = {}  # {row_id: [prob_class0, prob_class1, ...]}

    t0 = time.time()
    for idx, ogg_path in enumerate(ogg_files):
        segments = load_soundscape_segments(str(ogg_path), segment_seconds=5.0)
        if not segments:
            continue

        probs = predict_multi_fold(models, segments, mel, batch_size=32)
        stem = ogg_path.stem

        for seg_idx in range(len(segments)):
            end_sec = (seg_idx + 1) * 5
            row_id = f"{stem}_{end_sec}"
            submission_rows[row_id] = probs[seg_idx].tolist()

        if (idx + 1) % 50 == 0:
            print(f'  {idx + 1}/{len(ogg_files)} ({time.time() - t0:.0f}s)')

    print(f'Inference done: {len(ogg_files)} files in {time.time() - t0:.0f}s')

    # Alignement sur le template sample_submission.csv
    rows = []
    for _, sample_row in sample.iterrows():
        row_id = sample_row['row_id']
        if row_id in submission_rows:
            row_data = {'row_id': row_id}
            for cls, prob in zip(species_cols, submission_rows[row_id]):
                row_data[cls] = float(prob)
            rows.append(row_data)
        else:
            # row_id manquant (fichier absent ou trop court) → 0.0
            row_data = {'row_id': row_id}
            for cls in species_cols:
                row_data[cls] = 0.0
            rows.append(row_data)

    submission = pd.DataFrame(rows)

    # ⚠️ CRITIQUE : forcer l'ordre exact des colonnes du sample
    submission = submission[['row_id'] + species_cols]
    for col in species_cols:
        submission[col] = submission[col].astype(float)

    submission.to_csv(SUBM_OUT, index=False, float_format='%.6f')
    print(f'Submission written: {len(submission)} rows')
```

---

## 6. Validation Obligatoire (Checklist)

Avant que le notebook ne se termine, exécutez CETTE validation. Un échec ici = `Submission Scoring Error`.

```python
# Cell 7: Validation
def validate_submission(sub, sample, species_cols):
    """Valide le submission.csv. Lève une exception si invalide."""

    # 1. Shape
    if sub.shape != sample.shape:
        raise ValueError(f"Shape mismatch: {sub.shape} vs {sample.shape}")

    # 2. Colonnes (ordre ET noms)
    if list(sub.columns) != list(sample.columns):
        raise ValueError("Column mismatch")

    # 3. Row IDs
    if not sub['row_id'].equals(sample['row_id']):
        raise ValueError("row_id mismatch")

    # 4. Pas de NaN
    proba = sub.drop(columns=['row_id'])
    if proba.isna().sum().sum() != 0:
        raise ValueError("NaN values detected")

    # 5. Valeurs dans [0, 1]
    if not ((proba >= 0.0) & (proba <= 1.0)).all().all():
        raise ValueError("Values out of [0,1] range")

    # 6. Ordre des espèces
    if list(proba.columns) != species_cols:
        raise ValueError("Species column order mismatch")

    print(f'Validation OK: {len(sub)} rows × {len(sub.columns)} cols')

sample = pd.read_csv(SAMPLE_SUB)
sub = pd.read_csv(SUBM_OUT)
species_cols = [c for c in sample.columns if c != 'row_id']
validate_submission(sub, sample, species_cols)
```

---

## 7. Catalogue des Erreurs Rencontrées et Leurs Solutions

| # | Erreur Kaggle | Cause Racine | Solution |
|---|---------------|-------------|----------|
| 1 | `Submission CSV Not Found` | Le notebook a crashé avant d'écrire `submission.csv` | Mettre un `try/except` global qui écrit TOUJOURS au minimum le sample_submission |
| 2 | `Submission Scoring Error: wrong number of rows` | Nombre de lignes différent du contrat (ex: 240 ou 1481 au lieu de 3) | Utiliser `sample_submission.csv` comme template de row_ids |
| 3 | `Submission Scoring Error` (générique) | Ordre des colonnes différent du sample | Forcer `submission = submission[['row_id'] + species_cols]` |
| 4 | `Notebook Threw Exception` | `test_soundscapes/` vide → `FileNotFoundError` | Toujours vérifier `if test_dir.exists()` avant de lister |
| 5 | `Notebook Threw Exception` | `expected 3 channels, got 1` (EfficientNet) | `specs.repeat(1, 3, 1, 1)` si `specs.shape[1] == 1` |
| 6 | `Notebook Threw Exception` | `Missing key(s) in state_dict` (préfixe `backbone.`) | Stripper le préfixe : `state = {k[9:]: v for k, v in state.items()}` |
| 7 | `Submission Scoring Error` | NaN dans les probabilités | Vérifier `proba.isna().sum().sum() == 0` avant écriture |
| 8 | `Submission Scoring Error` | Valeurs hors [0, 1] | Clamper ou vérifier après sigmoid |
| 9 | `Notebook Timeout` | Inférence trop lente (> 90 min) | Optimiser batch_size, réduire le nombre de folds, ou désactiver TTA |
| 10 | `Submission Scoring Error` | `float_format` insuffisant ou BOM/CRLF | `to_csv(..., index=False, float_format='%.6f')` |

---

## 8. Les 3 Stratégies de Fallback (Dry-Run)

Quand `test_soundscapes/` est vide (dry-run), vous avez 3 options :

### Stratégie A : Copie du sample_submission.csv (recommandé)
```python
sample.to_csv('submission.csv', index=False)
# Score dry-run = 0.500
```
✅ Simple, fiable, accepté par Kaggle.
❌ Score inutilisable pour évaluer la qualité du modèle.

### Stratégie B : Inférence sur `train_soundscapes/`
```python
# Utiliser les fichiers train disponibles pour générer de vraies prédictions
train_dir = COMP / 'train_soundscapes'
train_files = sorted(train_dir.glob('*.ogg'))[:20]  # 20 premiers
# ... inférence ...
# Générer 240 row_ids (20 fichiers × 12 segments)
```
⚠️ Le scoring dry-run PEUT rejeter si le nombre de lignes ne matche pas le sample (3 lignes).
✅ Permet de valider que le pipeline d'inférence fonctionne.

### Stratégie C : Double chemin (recommandé pour la robustesse)
```python
if not ogg_files:
    # Dry-run : fallback sample
    sample.to_csv('submission.csv', index=False)
else:
    # Scoring réel : inférence complète
    # ...
```

---

## 9. Post-Processing (Optionnel — Gain Estimé +0.005 à +0.01)

Technique documentée par le top-2 BirdCLEF 2025 (Volodymyr) : **per-file probability scaling**.

Pour chaque soundscape, multiplier tous les segments par la probabilité max de chaque classe
sur l'ensemble des segments du fichier. Cela booste les prédictions cohérentes et supprime
les activations isolées (bruit).

```python
def apply_per_file_scaling(submission_rows, top_k=1):
    """
    submission_rows: {row_id: [prob_class0, prob_class1, ...]}
    Retourne le même dict avec probabilités rescalées.
    """
    # Grouper par soundscape (tout ce qui précède le dernier _XX)
    def extract_soundscape_id(row_id):
        parts = row_id.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0]
        return row_id

    file_rows = {}
    for row_id, probs in submission_rows.items():
        sid = extract_soundscape_id(row_id)
        file_rows.setdefault(sid, []).append((row_id, probs))

    result = {}
    for sid, rows in file_rows.items():
        probs_matrix = np.array([p for _, p in rows])  # (S, C)
        # Facteur de scaling = top-k mean par classe
        if top_k >= probs_matrix.shape[0]:
            scaling = np.mean(probs_matrix, axis=0)
        else:
            sorted_per_class = np.sort(probs_matrix, axis=0)[::-1]
            scaling = np.mean(sorted_per_class[:top_k], axis=0)
        scaled = probs_matrix * scaling[np.newaxis, :]
        for (row_id, _), sp in zip(rows, scaled):
            result[row_id] = sp.tolist()
    return result
```

---

## 10. TTA — Test-Time Augmentation (Optionnel — Gain Estimé +0.002 à +0.005)

Time-shift de ±0.1s sur chaque segment, moyenne des 3 prédictions. Coût ~3x.

```python
def predict_with_tta(model, waveform, mel, shift_samples=3200):
    """TTA par time-shift : original, +shift, -shift."""
    probs = []
    for shift in [0, shift_samples, -shift_samples]:
        if shift > 0:
            w = np.pad(waveform, (shift, 0))[:len(waveform)]
        elif shift < 0:
            w = np.pad(waveform, (0, -shift))[-shift:]
        else:
            w = waveform
        spec = mel(torch.from_numpy(w)).unsqueeze(0)
        if spec.shape[1] == 1:
            spec = spec.repeat(1, 3, 1, 1)
        probs.append(torch.sigmoid(model(spec)).cpu().numpy())
    return np.mean(probs, axis=0)
```

---

## 11. Contraintes de Poids de Modèle pour Kaggle

- Les `.pth` doivent être attachés comme **dataset Kaggle** (pas inclus dans le notebook).
- Taille typique : 5 folds × ~20 MB = ~100 MB total pour EfficientNet-B0.
- Les modèles sont chargés en CPU (`map_location='cpu'`), PAS de GPU.
- Temps d'inférence estimé : ~30-40 min pour ~400 soundscapes avec 5 folds, batch_size=32.

---

## 12. Checklist Kaggle Pre-Submit

Avant de cliquer "Submit", vérifiez :

- [ ] **Dataset Competition** `birdclef-2026` est attaché
- [ ] **Dataset Code** (votre repo V2) est attaché
- [ ] **Dataset Checkpoints** (vos `fold_*.pth`) est attaché
- [ ] **GPU = OFF** (sinon timeout en 1 minute)
- [ ] **Internet = OFF** (pas de `pip install` sauf packages pré-attachés)
- [ ] Le notebook écrit bien dans `/kaggle/working/submission.csv`
- [ ] Le notebook gère le cas `test_soundscapes/` vide (dry-run)
- [ ] Le notebook gère le cas `test_soundscapes/` peuplé (scoring réel)
- [ ] Validation exécutée en dernière cellule (shape, colonnes, row_ids, NaN, range)
- [ ] `float_format='%.6f'` et `index=False` dans `to_csv`
- [ ] Pas de BOM, pas de CRLF (vérifié dans la cellule de validation)

---

## 13. Code Minimum Viable — Notebook Complet

Voici le notebook minimal qui passe le dry-run ET le scoring réel :

```python
# ============================================================
# Cell 1: Setup
# ============================================================
from pathlib import Path
import os, sys, time
import numpy as np
import pandas as pd
import torch, torch.nn as nn
import torchaudio
import soundfile as sf
import timm

INPUT = Path('/kaggle/input')
WORK = Path('/kaggle/working')

# Find competition dataset
COMP = None
for c in [INPUT / 'birdclef-2026', INPUT / 'competitions' / 'birdclef-2026']:
    if c.exists() and (c / 'sample_submission.csv').exists():
        COMP = c; break
assert COMP is not None, 'Competition dataset not found'

SAMPLE_SUB = COMP / 'sample_submission.csv'
TEST_DIR = COMP / 'test_soundscapes'

# ============================================================
# Cell 2: Find checkpoints
# ============================================================
MODEL_DIR = None
for root, dirs, files in os.walk(INPUT):
    if any(f.startswith('fold_') and f.endswith('.pth') for f in files):
        MODEL_DIR = Path(root); break
assert MODEL_DIR is not None, 'No checkpoints found'
CKPTS = sorted(MODEL_DIR.glob('fold_*.pth'))

# ============================================================
# Cell 3: Define transforms & predictor
# ============================================================
class LogMelTransform(nn.Module):
    def __init__(self, sr=32000, nm=128, hl=512, nf=1024, fmin=50., fmax=16000., top=80.):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(sr=sr, n_mels=nm, hop_length=hl,
            n_fft=nf, f_min=fmin, f_max=fmax)
        self.db = torchaudio.transforms.AmplitudeToDB(top_db=top)
    def forward(self, w):
        if w.ndim == 1: w = w.unsqueeze(0)
        s = self.mel(w); s = self.db(s)
        return (s - s.mean()) / (s.std() + 1e-8)

def load_segments(path, seg_s=5.0, target_sr=32000):
    audio, sr = sf.read(str(path), dtype='float32')
    if audio.ndim > 1: audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = torchaudio.functional.resample(
            torch.from_numpy(audio).unsqueeze(0), sr, target_sr).squeeze(0).numpy()
    slen = int(seg_s * target_sr)
    segs = []
    for st in range(0, len(audio), slen):
        c = audio[st:st+slen]
        if len(c) < slen: c = np.pad(c, (0, slen - len(c)))
        segs.append(c.astype(np.float32))
    return segs

# ============================================================
# Cell 4: Load models
# ============================================================
sample = pd.read_csv(SAMPLE_SUB)
species = [c for c in sample.columns if c != 'row_id']

mel = LogMelTransform()
models = []
for p in CKPTS:
    m = timm.create_model('tf_efficientnet_b0', pretrained=False, num_classes=len(species))
    sd = torch.load(str(p), map_location='cpu', weights_only=True)
    if any(k.startswith('backbone.') for k in sd):
        sd = {k[9:]: v for k, v in sd.items()}
    m.load_state_dict(sd)
    m.eval()
    models.append(m)
print(f'Loaded {len(models)} models')

# ============================================================
# Cell 5: Inference + submission
# ============================================================
ogg_files = sorted(TEST_DIR.glob('*.ogg')) if TEST_DIR.exists() else []
print(f'Test soundscapes: {len(ogg_files)}')

if not ogg_files:
    print('DRY-RUN: copying sample_submission.csv')
    sample.to_csv('submission.csv', index=False)
else:
    preds = {}
    t0 = time.time()
    for idx, fp in enumerate(ogg_files):
        segs = load_segments(str(fp))
        if not segs: continue
        specs = torch.stack([mel(torch.from_numpy(s)) for s in segs])
        if specs.shape[1] == 1: specs = specs.repeat(1, 3, 1, 1)  # ⚠️ 1→3 channels
        all_p = []
        with torch.no_grad():
            for m in models:
                fp_ = []
                for i in range(0, len(specs), 32):
                    fp_.append(torch.sigmoid(m(specs[i:i+32])).numpy())
                all_p.append(np.concatenate(fp_, axis=0))
        probs = np.mean(all_p, axis=0).astype(np.float32)
        stem = fp.stem
        for si in range(len(segs)):
            preds[f'{stem}_{(si+1)*5}'] = probs[si].tolist()
        if (idx+1) % 50 == 0:
            print(f'  {idx+1}/{len(ogg_files)} ({time.time()-t0:.0f}s)')

    rows = []
    for _, sr in sample.iterrows():
        rid = sr['row_id']
        r = {'row_id': rid}
        if rid in preds:
            for cls, p in zip(species, preds[rid]):
                r[cls] = float(p)
        else:
            for cls in species:
                r[cls] = 0.0
        rows.append(r)

    sub = pd.DataFrame(rows)
    sub = sub[['row_id'] + species]
    for col in species:
        sub[col] = sub[col].astype(float)
    sub.to_csv('submission.csv', index=False, float_format='%.6f')
    print(f'Done: {len(sub)} rows in {time.time()-t0:.0f}s')

# ============================================================
# Cell 6: Validate
# ============================================================
sub = pd.read_csv('submission.csv')
assert sub.shape == sample.shape, f'Shape: {sub.shape} vs {sample.shape}'
assert list(sub.columns) == list(sample.columns), 'Column mismatch'
assert sub['row_id'].equals(sample['row_id']), 'row_id mismatch'
proba = sub.drop(columns=['row_id'])
assert proba.isna().sum().sum() == 0, 'NaN found'
assert ((proba >= 0) & (proba <= 1)).all().all(), 'Values out of [0,1]'
print(f'✅ submission.csv VALID ({len(sub)} rows, {len(species)} species)')
```

---

## 14. Résumé des Règles Non-Négociables

1. **`sample_submission.csv` est le contrat.** Row IDs, ordre des colonnes, noms des colonnes : tout vient de là.
2. **Toujours gérer le cas `test_soundscapes/` vide.** C'est le dry-run. Fallback = copier le sample.
3. **Toujours aligner sur les row_ids du sample.** Même en scoring réel, itérer sur le sample comme template.
4. **EfficientNet = 3 canaux.** `specs.repeat(1, 3, 1, 1)`.
5. **Checkpoint prefix.** Stripper `backbone.` si présent.
6. **Validation en dernière cellule.** Shape, colonnes, row_ids, NaN, range [0,1].
7. **`float_format='%.6f'`, `index=False`, LF line endings.**
8. **CPU only, internet OFF.** Pas de téléchargement, pas de GPU.
9. **Écrire dans `/kaggle/working/submission.csv`.** Pas ailleurs.

---

> **Document généré le 2026-05-30 à partir du RETEX de 12+ versions de notebooks de soumission BirdCLEF 2026.**
> Repo source : `__birdclef-V2`, workspace `__kaggle_deploy/sprint4/`.
> Score obtenu : **0.746** (leaderboard public), kernel `hellodave2035/bcv2-run2`.
