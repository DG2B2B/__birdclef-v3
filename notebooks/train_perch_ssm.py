"""
BirdCLEF 2026 – Perch+SSM Training Notebook (Kaggle GPU)
=========================================================
Extrait des embeddings Perch pour tous les segments d'entraînement,
entraîne un classifieur léger (ProtoSSM/ResSSM) par-dessus,
pseudo-labeling itératif, export ONNX.

Runtime: ~3-5h sur GPU T4 (Kaggle K1 ou Colab)
Output: classifier_ssm.onnx (~15-25 MB)

Dataset requis dans le notebook Kaggle:
  - birdclef-2026 (compétition)
  - perch-onnx-for-birdclef-2026 (rishikeshjani)
  - birdclef-2026-model (tonylica, optionnel — pour poids pré-entraînés)

⚠️ Ce notebook nécessite INTERNET pour télécharger les dépendances.
   Une fois l'entraînement fait, le modèle .onnx est exporté pour la soumission.
"""

# ══════════════════════════════════════════════════════════════════
# CELL 1: Imports & Setup
# ══════════════════════════════════════════════════════════════════

import os
import sys
import gc
import time
import pickle
import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import onnxruntime as ort
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

import soundfile as sf

warnings.filterwarnings("ignore")

# ── Configuration ──
class CFG:
    # Paths
    COMP_DIR = "/kaggle/input/birdclef-2026"
    PERCH_DIR = "/kaggle/input/perch-onnx-for-birdclef-2026"
    OUTPUT_DIR = "/kaggle/working"
    MODELS_DIR = "/kaggle/working/models"

    # Data
    TRAIN_CSV = f"{COMP_DIR}/train.csv"
    TAXONOMY_CSV = f"{COMP_DIR}/taxonomy.csv"
    TRAIN_AUDIO = f"{COMP_DIR}/train_audio"
    TEST_DIR = f"{COMP_DIR}/test_soundscapes"
    SAMPLE_SUB = f"{COMP_DIR}/sample_submission.csv"

    # Perch
    PERCH_ONNX = f"{PERCH_DIR}/perch_v2.onnx"
    PERCH_LABELS = f"{PERCH_DIR}/labels.csv"

    # Audio
    SR = 32000
    DURATION = 5  # seconds
    SEGMENT_SAMPLES = SR * DURATION  # 160000

    # Training
    N_FOLDS = 5
    SEED = 42
    EPOCHS = 50
    BATCH_SIZE = 256   # Embeddings are small (1536), large batches OK
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0
    WARMUP_EPOCHS = 3
    EARLY_STOPPING_PATIENCE = 10

    # Classifier architecture
    EMBEDDING_DIM = 1536
    HIDDEN_DIM = 512
    N_LAYERS = 3
    DROPOUT = 0.2
    N_CLASSES = 234

    # Pseudo-labeling
    PL_ITERATIONS = 2
    PL_THRESHOLD = 0.9
    PL_POWER = 0.7   # Power transform exponent

    # Devices
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = CFG()
os.makedirs(cfg.MODELS_DIR, exist_ok=True)

print(f"Device: {cfg.DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# ══════════════════════════════════════════════════════════════════
# CELL 2: Load Perch ONNX & Labels
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Loading Perch ONNX...")
print("=" * 60)

sess_opts = ort.SessionOptions()
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
providers = (
    ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if torch.cuda.is_available()
    else ["CPUExecutionProvider"]
)

perch_sess = ort.InferenceSession(cfg.PERCH_ONNX, sess_opts, providers=providers)
perch_input_name = perch_sess.get_inputs()[0].name
perch_output_names = [o.name for o in perch_sess.get_outputs()]
print(f"Perch input: {perch_input_name}")
print(f"Perch outputs: {perch_output_names}")

# Perch labels → BirdCLEF mapping
perch_labels = pd.read_csv(cfg.PERCH_LABELS)
print(f"Perch classes: {len(perch_labels)}")

# BirdCLEF species
sample_sub = pd.read_csv(cfg.SAMPLE_SUB)
SPECIES_COLS = [c for c in sample_sub.columns if c != "row_id"]
N_SPECIES = len(SPECIES_COLS)
species_to_idx = {sp: i for i, sp in enumerate(SPECIES_COLS)}
print(f"BirdCLEF species: {N_SPECIES}")

# ── Build Perch → BirdCLEF mapping ──
def build_perch_mapping():
    """Map Perch class indices to BirdCLEF species indices."""
    perch_to_bc = {}  # perch_idx → bc_idx (one-to-many possible)
    bc_to_perch = {}  # bc_idx → perch_idx (best match)

    # Normalize Perch labels
    perch_label_list = []
    for i, row in perch_labels.iterrows():
        label = str(row.get("label", row.get("scientific_name", ""))).lower().strip()
        perch_label_list.append(label)

    for bc_sp in SPECIES_COLS:
        bc_sp_lower = bc_sp.lower().replace("_", " ").strip()

        # Try exact match
        for pi, pl in enumerate(perch_label_list):
            if pl == bc_sp_lower:
                bc_to_perch[species_to_idx[bc_sp]] = pi
                perch_to_bc.setdefault(pi, []).append(species_to_idx[bc_sp])
                break
        else:
            # Try genus match
            genus = bc_sp_lower.split()[0] if " " in bc_sp_lower else bc_sp_lower
            for pi, pl in enumerate(perch_label_list):
                if pl.startswith(genus + " "):
                    bc_to_perch[species_to_idx[bc_sp]] = pi
                    perch_to_bc.setdefault(pi, []).append(species_to_idx[bc_sp])
                    break

    matched = len(bc_to_perch)
    print(f"  Perch→BirdCLEF mapping: {matched}/{N_SPECIES} species matched")
    return bc_to_perch, perch_to_bc, perch_label_list

bc_to_perch, perch_to_bc, perch_label_list = build_perch_mapping()


# ══════════════════════════════════════════════════════════════════
# CELL 3: Perch Embedding Extraction
# ══════════════════════════════════════════════════════════════════

def extract_perch_embedding(waveform: np.ndarray) -> np.ndarray:
    """Extract Perch embedding for a single 5s audio segment.

    Args:
        waveform: (160000,) float32 array.

    Returns:
        (1536,) float32 embedding vector.
    """
    feed = {perch_input_name: waveform[np.newaxis, :]}
    outs = perch_sess.run(perch_output_names, feed)
    out_dict = dict(zip(perch_output_names, outs))

    for key in ["embedding", "embeddings"]:
        if key in out_dict:
            return out_dict[key][0].astype(np.float32)

    # Fallback: pool spatial embedding
    for key in ["spatial_embedding", "spatial_embeddings"]:
        if key in out_dict:
            return out_dict[key].mean(axis=(0, 1)).astype(np.float32)

    raise KeyError(f"No embedding found: {list(out_dict.keys())}")


def load_and_embed(file_path: str) -> np.ndarray:
    """Load audio, ensure 5s segment, extract Perch embedding."""
    y, file_sr = sf.read(file_path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)

    # Resample if needed
    if file_sr != cfg.SR:
        import librosa
        y = librosa.resample(y, orig_sr=file_sr, target_sr=cfg.SR)

    # Take first 5s (or pad)
    if len(y) >= cfg.SEGMENT_SAMPLES:
        y = y[:cfg.SEGMENT_SAMPLES]
    else:
        y = np.pad(y, (0, cfg.SEGMENT_SAMPLES - len(y)))

    return extract_perch_embedding(y)


# ══════════════════════════════════════════════════════════════════
# CELL 4: Dataset Preparation
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Preparing dataset...")
print("=" * 60)

train_df = pd.read_csv(cfg.TRAIN_CSV)
taxonomy_df = pd.read_csv(cfg.TAXONOMY_CSV)

print(f"Train clips: {len(train_df)}")
print(f"Species in train: {train_df['primary_label'].nunique()}")

# ── Create folds ──
def create_folds(df, n_folds=5):
    """GroupKFold by author + location."""
    df = df.copy()
    df["lat_round"] = df["latitude"].round(1).astype(str)
    df["lon_round"] = df["longitude"].round(1).astype(str)
    df["group"] = (
        df["author"].astype(str) + "_" + df["lat_round"] + "_" + df["lon_round"]
    )
    gkf = GroupKFold(n_splits=n_folds)
    df["fold"] = -1
    for fi, (_, vi) in enumerate(gkf.split(df, groups=df["group"])):
        df.loc[vi, "fold"] = fi
    return df

train_df = create_folds(train_df, cfg.N_FOLDS)
print("\nFold distribution:")
for f in range(cfg.N_FOLDS):
    fd = train_df[train_df["fold"] == f]
    print(f"  Fold {f}: {len(fd)} clips, {fd['primary_label'].nunique()} species")


# ── Build label matrix ──
def build_label_matrix(df, species_list):
    """Build multi-hot label matrix (N, C)."""
    labels = np.zeros((len(df), len(species_list)), dtype=np.float32)
    for i, sp in enumerate(df["primary_label"]):
        if sp in species_to_idx:
            labels[i, species_to_idx[sp]] = 1.0
    return labels


# ══════════════════════════════════════════════════════════════════
# CELL 5: Lightweight Classifier (ProtoSSM)
# ══════════════════════════════════════════════════════════════════

class ProtoSSM(nn.Module):
    """Lightweight Structured State Space classifier for Perch embeddings.

    Inspired by Mamba/S4 but simplified for 1536-dim fixed embeddings.
    Uses 1D convolutions + gating instead of full SSM for speed.
    """

    def __init__(self, input_dim=1536, hidden_dim=512, n_layers=3,
                 n_classes=234, dropout=0.2):
        super().__init__()

        # Projection
        self.proj_in = nn.Linear(input_dim, hidden_dim)

        # SSM-like blocks with conv + gating
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(SSMBlock(hidden_dim, dropout))

        # Output head
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, n_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, 1536) Perch embeddings
        x = self.proj_in(x)  # (B, hidden_dim)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = self.head(x)  # (B, n_classes)
        return x


class SSMBlock(nn.Module):
    """Single SSM block: Conv1D → Gate → MLP with residual."""

    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, dim) — treat as sequence of length 1
        residual = x

        # Conv branch
        h = self.norm1(x).unsqueeze(-1)  # (B, dim, 1)
        h = self.conv(h).squeeze(-1)     # (B, dim)
        gate = self.gate(x)
        h = h * gate

        # MLP branch
        h = h + self.mlp(self.norm2(h))

        return h + residual


class LinearProbe(nn.Module):
    """Simplest classifier: just a linear layer on Perch embeddings."""
    def __init__(self, input_dim=1536, n_classes=234):
        super().__init__()
        self.linear = nn.Linear(input_dim, n_classes)

    def forward(self, x):
        return self.linear(x)


# ══════════════════════════════════════════════════════════════════
# CELL 6: Focal Loss & Metrics
# ══════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """Focal Loss for multi-label classification."""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        prob = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss = alpha_weight * focal_weight * bce
        return loss.mean()


def macro_auc_score(y_true, y_pred):
    """Macro ROC-AUC."""
    n_classes = y_true.shape[1]
    aucs = []
    for c in range(n_classes):
        if y_true[:, c].sum() == 0:
            continue
        if y_true[:, c].sum() == len(y_true):
            continue
        try:
            aucs.append(roc_auc_score(y_true[:, c], y_pred[:, c]))
        except ValueError:
            pass
    return np.mean(aucs) if aucs else 0.0


# ══════════════════════════════════════════════════════════════════
# CELL 7: Embeddings Dataset (cached on disk)
# ══════════════════════════════════════════════════════════════════

class EmbeddingDataset(Dataset):
    """Dataset of pre-extracted Perch embeddings + labels."""

    def __init__(self, embeddings, labels, indices=None):
        self.embeddings = embeddings
        self.labels = labels
        self.indices = indices if indices is not None else np.arange(len(embeddings))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        return (
            torch.from_numpy(self.embeddings[i]),
            torch.from_numpy(self.labels[i]),
        )


# ══════════════════════════════════════════════════════════════════
# CELL 8: Extract ALL Perch Embeddings (cache to disk)
# ══════════════════════════════════════════════════════════════════

CACHE_PATH = os.path.join(cfg.OUTPUT_DIR, "perch_embeddings.npy")
LABELS_PATH = os.path.join(cfg.OUTPUT_DIR, "train_labels.npy")

if not os.path.exists(CACHE_PATH):
    print("\n" + "=" * 60)
    print("Extracting Perch embeddings for ALL training clips...")
    print("  (One-time operation, cached to disk)")
    print("=" * 60)

    train_files = train_df["filename"].tolist()
    n_total = len(train_files)

    all_embeddings = np.zeros((n_total, cfg.EMBEDDING_DIM), dtype=np.float32)
    all_labels = build_label_matrix(train_df, SPECIES_COLS)
    failed = 0

    t_start = time.time()
    for i, fname in enumerate(train_files):
        fpath = os.path.join(cfg.TRAIN_AUDIO, fname)
        try:
            emb = load_and_embed(fpath)
            all_embeddings[i] = emb
        except Exception as e:
            failed += 1
            all_embeddings[i] = 0.0  # zero embedding for failed files

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * (n_total - i - 1)
            print(f"  [{i+1}/{n_total}] {elapsed:.0f}s elapsed, ETA: {eta:.0f}s")

    total_time = time.time() - t_start
    print(f"\n  Done in {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Failed: {failed}/{n_total}")

    # Save cache
    np.save(CACHE_PATH, all_embeddings)
    np.save(LABELS_PATH, all_labels)
    print(f"  Cached to {CACHE_PATH}")

else:
    print("\n[OK] Loading cached Perch embeddings...")
    all_embeddings = np.load(CACHE_PATH)
    all_labels = np.load(LABELS_PATH)
    print(f"  Shape: {all_embeddings.shape}")


# ══════════════════════════════════════════════════════════════════
# CELL 9: Training Function
# ══════════════════════════════════════════════════════════════════

def train_one_fold(fold_idx, model, train_idx, val_idx, fold_dir):
    """Train for one fold and return best validation AUC."""

    train_ds = EmbeddingDataset(all_embeddings, all_labels, train_idx)
    val_ds = EmbeddingDataset(all_embeddings, all_labels, val_idx)

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE * 2, shuffle=False,
                            num_workers=2, pin_memory=True)

    model = model.to(cfg.DEVICE)
    criterion = FocalLoss(alpha=cfg.FOCAL_ALPHA, gamma=cfg.FOCAL_GAMMA)
    optimizer = AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg.EPOCHS
    warmup_steps = steps_per_epoch * cfg.WARMUP_EPOCHS

    scheduler1 = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    scheduler2 = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    scheduler = SequentialLR(optimizer, [scheduler1, scheduler2],
                             milestones=[warmup_steps])

    best_val_auc = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(cfg.EPOCHS):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for batch_emb, batch_labels in train_loader:
            batch_emb = batch_emb.to(cfg.DEVICE)
            batch_labels = batch_labels.to(cfg.DEVICE)

            optimizer.zero_grad()
            logits = model(batch_emb)
            loss = criterion(logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── Validate ──
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch_emb, batch_labels in val_loader:
                batch_emb = batch_emb.to(cfg.DEVICE)
                logits = model(batch_emb)
                probs = torch.sigmoid(logits)
                val_preds.append(probs.cpu().numpy())
                val_targets.append(batch_labels.numpy())

        val_preds = np.concatenate(val_preds, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)
        val_auc = macro_auc_score(val_targets, val_preds)

        # ── Log ──
        lr = optimizer.param_groups[0]["lr"]
        print(f"  Fold {fold_idx} | Epoch {epoch+1:2d}/{cfg.EPOCHS} | "
              f"Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f} | LR: {lr:.2e}")

        # ── Checkpoint ──
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(fold_dir, "best_model.pth"))
        else:
            patience_counter += 1

        # ── Early stopping ──
        if patience_counter >= cfg.EARLY_STOPPING_PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    print(f"  Fold {fold_idx} BEST: AUC={best_val_auc:.4f} at epoch {best_epoch}")
    return best_val_auc


# ══════════════════════════════════════════════════════════════════
# CELL 10: Run 5-Fold Training
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Starting 5-Fold Training")
print("=" * 60)

fold_models = []
fold_aucs = []

for fold in range(cfg.N_FOLDS):
    print(f"\n{'─' * 40}")
    print(f"FOLD {fold + 1}/{cfg.N_FOLDS}")
    print(f"{'─' * 40}")

    fold_dir = os.path.join(cfg.MODELS_DIR, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    train_idx = np.where(train_df["fold"] != fold)[0]
    val_idx = np.where(train_df["fold"] == fold)[0]
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")

    model = ProtoSSM(
        input_dim=cfg.EMBEDDING_DIM,
        hidden_dim=cfg.HIDDEN_DIM,
        n_layers=cfg.N_LAYERS,
        n_classes=cfg.N_CLASSES,
        dropout=cfg.DROPOUT,
    )

    auc = train_one_fold(fold, model, train_idx, val_idx, fold_dir)
    fold_aucs.append(auc)

    # Load best model
    model.load_state_dict(torch.load(os.path.join(fold_dir, "best_model.pth")))
    fold_models.append(model)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"\n{'=' * 60}")
print(f"5-Fold Results:")
for f, auc in enumerate(fold_aucs):
    print(f"  Fold {f}: AUC = {auc:.4f}")
print(f"  Mean AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")


# ══════════════════════════════════════════════════════════════════
# CELL 11: Pseudo-Labeling Iteration
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Pseudo-Labeling on Test Soundscapes")
print("=" * 60)

def power_transform(probs, power=0.7):
    """Push probabilities toward 0 or 1."""
    return probs ** power


# Generate pseudo-labels on test soundscapes
test_files = sorted([
    f for f in os.listdir(cfg.TEST_DIR)
    if f.endswith((".ogg", ".mp3"))
])

print(f"Test soundscapes: {len(test_files)}")

# For pseudo-labeling, we run inference with all 5 models and average
pseudo_probs = []

for idx, file in enumerate(test_files):
    fpath = os.path.join(cfg.TEST_DIR, file)

    # Load + segment + embed
    y, sr_file = sf.read(fpath, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr_file != cfg.SR:
        import librosa
        y = librosa.resample(y, orig_sr=sr_file, target_sr=cfg.SR)

    segments = []
    step = int(cfg.SR * 2.5)  # 2.5s step
    for start in range(0, len(y) - cfg.SEGMENT_SAMPLES + 1, step):
        seg = y[start : start + cfg.SEGMENT_SAMPLES]
        emb = extract_perch_embedding(seg)
        segments.append(emb)

    if not segments:
        continue

    segments = np.stack(segments)
    segments_t = torch.from_numpy(segments).to(cfg.DEVICE)

    # Ensemble prediction from all 5 models
    all_fold_probs = []
    for model in fold_models:
        model.eval()
        with torch.no_grad():
            logits = model(segments_t)
            probs = torch.sigmoid(logits).cpu().numpy()
        all_fold_probs.append(probs)

    mean_probs = np.mean(all_fold_probs, axis=0)  # (N_seg, 234)
    pseudo_probs.append(mean_probs.mean(axis=0))  # mean over time → (234,)

    if (idx + 1) % 50 == 0:
        print(f"  [{idx+1}/{len(test_files)}]")

pseudo_probs = np.stack(pseudo_probs)  # (N_soundscapes, 234)

# Apply power transform + threshold
pl_transformed = power_transform(pseudo_probs, power=cfg.PL_POWER)
pl_binary = (pl_transformed > cfg.PL_THRESHOLD).astype(np.float32)

n_pl = pl_binary.sum()
print(f"Pseudo-labels generated: {n_pl} positive labels across all soundscapes")

# Save pseudo-labels for potential reuse
np.save(os.path.join(cfg.OUTPUT_DIR, "pseudo_labels.npy"), pseudo_probs)


# ══════════════════════════════════════════════════════════════════
# CELL 12: Export Classifier to ONNX
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Exporting classifier to ONNX...")
print("=" * 60)

# Use the best fold model for export
best_fold = np.argmax(fold_aucs)
export_model = fold_models[best_fold]
export_model.eval()
export_model = export_model.cpu()

onnx_path = os.path.join(cfg.OUTPUT_DIR, "classifier_ssm.onnx")
dummy_input = torch.randn(1, cfg.EMBEDDING_DIM)

torch.onnx.export(
    export_model,
    dummy_input,
    onnx_path,
    input_names=["embedding"],
    output_names=["logits"],
    dynamic_axes={
        "embedding": {0: "batch_size"},
        "logits": {0: "batch_size"},
    },
    opset_version=17,
    do_constant_folding=True,
)

print(f"[OK] ONNX exported: {onnx_path}")
print(f"     Size: {os.path.getsize(onnx_path) / 1e6:.1f} MB")

# Verify ONNX
ort_sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
test_input = np.random.randn(4, cfg.EMBEDDING_DIM).astype(np.float32)
ort_out = ort_sess.run(None, {"embedding": test_input})[0]

with torch.no_grad():
    torch_out = export_model(torch.from_numpy(test_input)).numpy()

diff = np.abs(ort_out - torch_out).max()
print(f"     ONNX vs PyTorch max diff: {diff:.2e}")
assert diff < 1e-4, f"Large difference: {diff}"


# ══════════════════════════════════════════════════════════════════
# CELL 13: Save Artifacts for Submission
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Saving submission artifacts...")
print("=" * 60)

artifacts_dir = os.path.join(cfg.OUTPUT_DIR, "submission_artifacts")
os.makedirs(artifacts_dir, exist_ok=True)

# Copy ONNX model
import shutil
shutil.copy(onnx_path, os.path.join(artifacts_dir, "classifier_ssm.onnx"))

# Save config
config = {
    "embedding_dim": cfg.EMBEDDING_DIM,
    "hidden_dim": cfg.HIDDEN_DIM,
    "n_layers": cfg.N_LAYERS,
    "n_classes": cfg.N_CLASSES,
    "sr": cfg.SR,
    "duration": cfg.DURATION,
    "best_fold": int(best_fold),
    "fold_aucs": [float(a) for a in fold_aucs],
    "mean_auc": float(np.mean(fold_aucs)),
    "pl_threshold": cfg.PL_THRESHOLD,
    "pl_power": cfg.PL_POWER,
}
with open(os.path.join(artifacts_dir, "config.pkl"), "wb") as f:
    pickle.dump(config, f)

# Save weights (equal weighting for now, can be optimized later)
weights = {"weights": [1.0 / len(fold_models)] * len(fold_models)}
with open(os.path.join(artifacts_dir, "weights.json"), "w") as f:
    json.dump(weights, f, indent=2)

# Save Perch labels (already from input dataset, just copy reference)
# The actual labels.csv should be included in the submission dataset

print(f"\nArtifacts saved to {artifacts_dir}/")
for f in sorted(os.listdir(artifacts_dir)):
    fpath = os.path.join(artifacts_dir, f)
    size_mb = os.path.getsize(fpath) / 1e6
    print(f"  {f} ({size_mb:.1f} MB)")

print(f"\n{'=' * 60}")
print(f"TRAINING COMPLETE")
print(f"{'=' * 60}")
print(f"Best fold: {best_fold} (AUC: {fold_aucs[best_fold]:.4f})")
print(f"Mean 5-fold AUC: {np.mean(fold_aucs):.4f}")
print(f"\nNext steps:")
print(f"  1. Upload artifacts to Kaggle as a dataset")
print(f"  2. Include perch_v2.onnx from rishikeshjani/perch-onnx-for-birdclef-2026")
print(f"  3. Include labels.csv from the same dataset")
print(f"  4. Run submission_perch.py for final submission")
