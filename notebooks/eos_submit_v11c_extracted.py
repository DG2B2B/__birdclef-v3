# =================================================================
# CELL 0 - SCRIPT DE DIAGNOSTIC - VERIFICATION BIRDNET 2.4
# =================================================================
from pathlib import Path
import os

print("--- DÉBUT DE LA DÉTECTION BIRDNET ---")
found_model = None
found_labels = None

# Scan récursif pour trouver le fichier du modèle
for p in Path("/kaggle/input").rglob("*.tflite"):
    if "birdnet" in p.name.lower():
        print(f"✅ Modèle TFLite détecté : {p}")
        found_model = p

# Scan récursif pour trouver le fichier de labels
for p in Path("/kaggle/input").rglob("*.txt"):
    if "birdnet" in p.name.lower() or "label" in p.name.lower():
        # S'assurer qu'il s'agit des labels officiels de BirdNET
        if "global" in p.name.lower() or "6k" in p.name.lower():
            print(f"✅ Fichier de labels détecté : {p}")
            found_labels = p

if found_model and found_labels:
    print("\n[OK] BirdNET est correctement attaché à votre environnement.")
    try:
        # Tenter d'instancier l'interpréteur TensorFlow Lite pour valider la compatibilité
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite.python.interpreter import Interpreter
            
        interpreter = Interpreter(model_path=str(found_model))
        interpreter.allocate_tensors()
        print("✅ L'interpréteur TFLite a alloué les tenseurs avec succès.")
        print("-> Votre Model_4 basculera automatiquement sur le blend à 3 voies.")
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation de l'interpréteur : {e}")
else:
    print("\n❌ ÉCHEC : BirdNET n'est pas détecté. Vérifiez que vous l'avez bien ajouté depuis l'onglet 'Models'.")


# === CELL 1 ===

solutions = {
 'type_add' : 'TAX_SMOOTHING',  # Conservation du lissage taxonomique direct
 'task'     : 'run Model_2_SED once',
 'Models'   : [
  {'Model':'Model_2','subm':'subm_2.csv','weight':0.020,'xSED':[],'LB':'0.928'},
  {'Model':'Model_3','subm':'subm_3.csv','weight':0.013,'xSED':[],'LB':'0.947'},
  {'Model':'Model_4','subm':'subm_4.csv','weight':0.967,'xSED':[0.60,0.40],'LB':'0.949'}
 ]
}

# === CELL 2 ===

_ensemble_models = [model['Model' ] for model in solutions['Models']]
_files_subm      = [model['subm'  ] for model in solutions['Models']]
_weights         = [model['weight'] for model in solutions['Models']]
_xsed            = [model['xSED'  ] for model in solutions['Models']]
_lbs             = [model['LB'    ] for model in solutions['Models']]



# === ## Model_2

# === CELL 3 ===

if 'Model_2' in _ensemble_models or \
   'task' in solutions and 'run Model_2_SED once' in solutions['task']:

    _file_name_submission = "subm_2.csv"

    # Install from bundled wheel (programmatic execution for pure Python compatibility)
    import subprocess
    import sys
    import os
    ort_whl = "/kaggle/input/datasets/hellodave2035/birdclef-perch-models/onnxruntime-1.24.4-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
    if os.path.exists(ort_whl):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", ort_whl])
        
    # =================================================================
    # S1 -- IMPORTS + CONFIG
    # =================================================================
    import os, sys, time, json, pickle, gc, random, math
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from collections import defaultdict
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, ConcatDataset
    from torch.cuda.amp import GradScaler, autocast
    import torchaudio
    import timm
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from scipy.special import expit as sigmoid_np
    import warnings
    warnings.filterwarnings("ignore")
    
    SEED = 42
    random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name()}")
    
    # ------------------------------------------------------------------
    # Notebook Mode
    # -----------------------------------------------------------------
    MODE = "infer"  # "train" or "infer"
    
    if MODE == "train":
        import subprocess
        import sys
        subprocess.check_call([
            sys.executable, "-m", "pip", 
            "install", "-q", 
            "timm", "torchaudio", "onnxscript", "onnx"
        ]) 
        DEBUG = True   
    else:
        DEBUG = False
    
    # ------------------------------------------------------------------
    # Paths -- Kaggle dataset layout
    # -----------------------------------------------------------------
    COMP_DIR = Path("/kaggle/input/competitions/birdclef-2026")
    WAVEFORM_CACHE_DIR = Path("/kaggle/input/datasets/tuckerarrants/birdclef-2026-waveform-cache/waveform_cache")
    PERCH_ONNX_PATH = Path("/kaggle/input/datasets/hellodave2035/birdclef-perch-models/perch_v2.onnx")
    
    LABELS_PATH     = COMP_DIR / "train_soundscapes_labels.csv"
    TAXONOMY_PATH   = COMP_DIR / "taxonomy.csv"
    SAMPLE_SUB_PATH = COMP_DIR / "sample_submission.csv"
    TEST_DIR        = COMP_DIR / "test_soundscapes"
    
    OUT_DIR = Path("/kaggle/working")
    
    NUM_CLASSES = 234
    SR = 32000
    
    # --- Duration ---
    TRAIN_DURATION = 5    # seconds
    VAL_DURATION   = 5    # always 5s for competition eval
    TRAIN_SAMPLES  = SR * TRAIN_DURATION
    VAL_SAMPLES    = SR * VAL_DURATION
    
    N_FOLDS = 5
    
    # --- Mel spectrogram ---
    N_FFT      = 2048
    HOP_LENGTH = 512
    N_MELS     = 256
    FMIN       = 20
    FMAX       = 16000
    
    # --- Model ---
    BACKBONE_NAME = "tf_efficientnet_b0.ns_jft_in1k"  
    
    # --- Perch distillation ---
    USE_PERCH_DISTILL = True
    PERCH_EMBED_DIM   = 1536
    ALPHA_DISTILL     = 1.0   # MSE loss weight
    
    # --- Training ---
    FOLDS  = [0, 1, 2, 3, 4]
    EPOCHS = 25
    BATCH  = 16 if DEBUG else 64        # used 64 locally, 16 for Kaggle
    LR     = 5e-4
    MIN_LR = 1e-5
    WD     = 1e-4
    WARMUP_EPOCHS = 2
    
    # --- Upsampling ---
    MIN_SAMPLE = 20
    
    # --- Augmentation ---
    AUG_PROB = 0.5
    AUG_GAIN_DB_RANGE      = (-6.0, 6.0)
    AUG_NOISE_SNR_DB_RANGE = (10.0, 30.0)
    
    # --- MixUp ---
    USE_FOCAL_MIXUP    = True
    MIXUP_PROB         = 0.5
    MIXUP_ALPHA        = 0.4
    MIXUP_HARD         = True    # union labels (hard) vs weighted blend (soft)
    
    USE_FOCAL_SC_MIXUP     = True
    FOCAL_SC_MIXUP_PROB    = 0.5
    FOCAL_SC_MIXUP_ALPHA   = 0.4
    
    # --- FreqMixStyle (disabled by default) ---
    FREQ_MIXSTYLE_PROB  = 0.0
    FREQ_MIXSTYLE_ALPHA = 0.1
    
    # --- SpecAugment ---
    FREQ_MASK_PARAM = 10
    TIME_MASK_PARAM = 10
    NUM_FREQ_MASKS  = 1
    NUM_TIME_MASKS  = 2
    
    # --- Source weights ---
    USE_FOCAL           = True
    USE_FOCAL_SECONDARY = True
    USE_LABELED_SC      = True
    
    ACTIVE_SOURCES = ["focal", "sc"]
    SHARES = {"focal": 0.9, "sc": 0.1}
    SOURCE_WEIGHTS = {
        "focal":         1.0,
        "focal_missing": 0.0,
        "sc":            1.0,
    }
    
    print(f"Backbone: {BACKBONE_NAME}")
    print(f"Train duration: {TRAIN_DURATION}s | Mel: {N_MELS} mels, n_fft={N_FFT}, hop={HOP_LENGTH}")
    print(f"Distillation: {'ON' if USE_PERCH_DISTILL else 'OFF'} (alpha={ALPHA_DISTILL})")
    print(f"Batch: {BATCH} | Epochs: {EPOCHS} | Folds: {FOLDS}")
    
    
    if MODE == "train":
        pass  # Déjà installé programmatiquement au début de la cellule
    
    # =================================================================
    # S2 -- LOAD DATA
    # =================================================================
    
    # --- Label ordering from sample_submission (defines column order) ---
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
    PRIMARY_LABELS = sample_sub.columns[1:].tolist()
    LABEL2IDX = {label: idx for idx, label in enumerate(PRIMARY_LABELS)}
    taxonomy = pd.read_csv(TAXONOMY_PATH)
    label_to_taxon = dict(zip(taxonomy["primary_label"].astype(str),
                              taxonomy["class_name"].astype(str)))
    TAXON_MASKS = {t: np.array([i for i, l in enumerate(PRIMARY_LABELS)
                                if label_to_taxon.get(l, "") == t])
                   for t in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]}
    
    # --- Focal recording metadata ---
    audio_cache_meta = pd.read_csv(WAVEFORM_CACHE_DIR / "audio_cache_meta.csv")
    train_df = pd.read_csv(COMP_DIR / "train.csv")
    audio_cache_meta = audio_cache_meta.merge(
        train_df[["filename", "secondary_labels"]], on="filename", how="left"
    )
    audio_cache_meta = audio_cache_meta[
        audio_cache_meta["primary_label"].isin(LABEL2IDX)
    ].reset_index(drop=True)
    print(f"Focal audio cache: {len(audio_cache_meta)} entries")
    
    # --- Soundscape window metadata ---
    sc_cache_meta = pd.read_csv(WAVEFORM_CACHE_DIR / "soundscape_cache_meta.csv")
    sc_cache_meta["label_list"] = sc_cache_meta["label_list"].apply(
        lambda x: x.split(";") if isinstance(x, str) else []
    )
    print(f"Soundscape cache: {len(sc_cache_meta)} windows")
    
    # --- Build soundscape label matrix from ground truth ---
    sc_labels_raw = pd.read_csv(LABELS_PATH).drop_duplicates()
    sc_labels_raw["start_sec"] = pd.to_timedelta(sc_labels_raw["start"]).dt.total_seconds().astype(int)
    
    Y_SC = np.zeros((len(sc_cache_meta), NUM_CLASSES), dtype=np.float32)
    for i, row in sc_cache_meta.iterrows():
        matches = sc_labels_raw[
            (sc_labels_raw["filename"] == row["filename"]) &
            (sc_labels_raw["start_sec"] == row["start_sec"])
        ]
        for _, m in matches.iterrows():
            for lbl in str(m["primary_label"]).split(";"):
                lbl = lbl.strip()
                if lbl in LABEL2IDX:
                    Y_SC[i, LABEL2IDX[lbl]] = 1.0
    
    labeled_sc_mask = Y_SC.sum(axis=1) > 0
    print(f"Soundscape labels: {labeled_sc_mask.sum()}/{len(Y_SC)} windows labeled, "
          f"{int(Y_SC.sum())} positives, {int((Y_SC.sum(axis=0) > 0).sum())} species")
    
    # =================================================================
    # FOLD ASSIGNMENT
    # =================================================================
    
    # --- Focal: StratifiedKFold by species ---
    audio_for_split = audio_cache_meta.drop_duplicates("original_idx").reset_index(drop=True)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    audio_for_split["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(audio_for_split, audio_for_split["primary_label"])):
        audio_for_split.loc[val_idx, "fold"] = fold
    audio_cache_meta = audio_cache_meta.merge(
        audio_for_split[["original_idx", "fold"]], on="original_idx", how="left"
    )
    print(f"\nFocal fold distribution:\n{audio_cache_meta['fold'].value_counts().sort_index()}")
    
    # --- Soundscape: file-level folds, all 66 files distributed ---
    # All files including S22 participate in CV for maximum species coverage
    # (46 multi-fold species vs 32 with S22 holdout). The non_s22_mask_sc
    # filter in evaluation still excludes S22 windows from the primary metric.
    from sklearn.model_selection import GroupKFold
    
    sc_files = sc_cache_meta[["filename", "site"]].drop_duplicates().reset_index(drop=True)
    gkf = GroupKFold(n_splits=N_FOLDS)
    sc_files["fold"] = -1
    for fold, (_, val_idx) in enumerate(gkf.split(sc_files, groups=sc_files["filename"])):
        sc_files.loc[sc_files.index[val_idx], "fold"] = fold
    
    file_to_fold = dict(zip(sc_files["filename"], sc_files["fold"]))
    sc_cache_meta["fold"] = sc_cache_meta["filename"].map(file_to_fold).fillna(-1).astype(int)
    print(f"\nSoundscape fold distribution:")
    print(sc_cache_meta["fold"].value_counts().sort_index())
    # =================================================================
    # UPSAMPLE RARE SPECIES
    # =================================================================
    counts = audio_cache_meta["primary_label"].value_counts()
    rare_species = counts[counts < MIN_SAMPLE].index
    extra_rows = []
    for sp in rare_species:
        sp_rows = audio_cache_meta[audio_cache_meta["primary_label"] == sp]
        n_copies = int(np.ceil(MIN_SAMPLE / len(sp_rows))) - 1
        for _ in range(n_copies):
            extra_rows.append(sp_rows)
    
    n_before = len(audio_cache_meta)
    if extra_rows:
        audio_cache_meta = pd.concat([audio_cache_meta] + extra_rows, ignore_index=True)
    print(f"\nUpsampled {len(rare_species)} rare species (min={MIN_SAMPLE}): "
          f"{n_before} -> {len(audio_cache_meta)} samples")
    
    # Non-S22 mask for evaluation (S22 is a site with known label noise)
    sc_sites = sc_cache_meta["site"].values
    non_s22_mask_sc = sc_sites != "S22"
    print(f"S22: {(~non_s22_mask_sc).sum()}, non-S22: {non_s22_mask_sc.sum()}")
    print("OK Data loaded")
    
    
    if DEBUG:
        EPOCHS = 1
        FOLDS = [0]
        audio_cache_meta = audio_cache_meta.groupby("primary_label").head(3).reset_index(drop=True)
        sc_cache_meta = sc_cache_meta.head(50)
        Y_SC = Y_SC[:50]
        non_s22_mask_sc = non_s22_mask_sc[:50]
        print(f"DEBUG MODE: {len(audio_cache_meta)} focal, {len(sc_cache_meta)} sc, "
              f"{EPOCHS} epoch, folds={FOLDS}")
    
    # =================================================================
    # S3 -- EVAL UTILITIES + MEL TRANSFORM + SED MODELS
    # =================================================================
    
    def compute_macro_auc(y_true, y_pred, mask=None, class_mask=None):
        """Macro-averaged AUC across evaluable species."""
        if mask is not None:
            y_true, y_pred = y_true[mask], y_pred[mask]
        if class_mask is not None:
            y_true, y_pred = y_true[:, class_mask], y_pred[:, class_mask]
        aucs = []
        for c in range(y_true.shape[1]):
            col = y_true[:, c]
            if col.sum() == 0 or col.sum() == len(col):
                continue
            try:
                aucs.append(roc_auc_score(col, y_pred[:, c]))
            except ValueError:
                continue
        return (np.mean(aucs) if aucs else float("nan")), len(aucs)
    
    def full_eval(y_true, y_pred, ns22, tm):
        r = {}
        a, n = compute_macro_auc(y_true, y_pred)
        r["macro_auc_all"], r["n_all"] = round(a, 4), n
        a, n = compute_macro_auc(y_true, y_pred, mask=ns22)
        r["non_s22_macro"], r["n_ns22"] = round(a, 4), n
        for t, cm in tm.items():
            a, n = compute_macro_auc(y_true, y_pred, mask=ns22, class_mask=cm)
            r[f"non_s22_{t}"] = round(a, 4)
        return r
    
    # ------------------------------------------------------------------
    # GPU Mel Spectrogram
    # ------------------------------------------------------------------
    class MelSpecTransform(nn.Module):
        def __init__(self):
            super().__init__()
            self.mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
                n_mels=N_MELS, f_min=FMIN, f_max=FMAX, power=2.0,
            )
            self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)
    
        def forward(self, waveform):
            return self.db_transform(self.mel_spec(waveform))
    
    # ------------------------------------------------------------------
    # GPU SpecAugment
    # ------------------------------------------------------------------
    class SpecAugment(nn.Module):
        def __init__(self):
            super().__init__()
            self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=FREQ_MASK_PARAM)
            self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=TIME_MASK_PARAM)
    
        def forward(self, mel):
            for _ in range(NUM_FREQ_MASKS):
                mel = self.freq_mask(mel)
            for _ in range(NUM_TIME_MASKS):
                mel = self.time_mask(mel)
            return mel
    
    # ------------------------------------------------------------------
    # Frozen Perch teacher -- ONNX inference, no gradients
    # ------------------------------------------------------------------
    import onnxruntime as ort
    
    class PerchTeacher:
        """Frozen Perch v2 via ONNX. Takes 5s waveforms, returns 1536-d embeddings.
        The teacher is never updated -- it provides a stable distillation target."""
    
        def __init__(self, onnx_path, device_str="cuda"):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
                if device_str == "cuda" else ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(str(onnx_path), providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self._out_names = [o.name for o in self.session.get_outputs()]
            self._embed_idx = None
            for i, o in enumerate(self.session.get_outputs()):
                if o.shape and o.shape[-1] == PERCH_EMBED_DIM:
                    self._embed_idx = i
                    break
            if self._embed_idx is None:
                self._embed_idx = 1
            print(f"Perch ONNX loaded: embed_idx={self._embed_idx}")
    
        @torch.no_grad()
        def embed(self, waveforms_5s):
            """waveforms_5s: (B, 160000) float32, returns (B, 1536) embeddings."""
            wav_np = waveforms_5s.cpu().numpy()
            results = self.session.run(None, {self.input_name: wav_np})
            return torch.from_numpy(results[self._embed_idx]).float()
    
    # ------------------------------------------------------------------
    # Distillation head: GAP + Linear to Perch embedding space
    # ------------------------------------------------------------------
    class DistillHead(nn.Module):
        """Projects backbone features to Perch's 1536-d space via GAP + Linear."""
        def __init__(self, backbone_dim, embed_dim=1536):
            super().__init__()
            self.proj = nn.Linear(backbone_dim, embed_dim)
    
        def forward(self, feature_map):
            gap = feature_map.mean(dim=[2, 3])   # (B, C, F, T) -> (B, C)
            return self.proj(gap)                 # (B, embed_dim)
    
    # ------------------------------------------------------------------
    # SED Model V2: GeMFreq + bottleneck + AttBlock (recommended)
    # ------------------------------------------------------------------
    class GeMFreqPool(nn.Module):
        """Generalized Mean pooling over frequency. Learnable p starts at 3.0
        (sharper than mean, softer than max). Lets the model emphasize
        frequency bands where species vocalize."""
        def __init__(self, p_init=3.0, eps=1e-6):
            super().__init__()
            self.p = nn.Parameter(torch.tensor(float(p_init)))
            self.eps = eps
    
        def forward(self, x):
            p = self.p.clamp(min=1.0)
            x = x.clamp(min=self.eps).pow(p)
            x = x.mean(dim=2)
            return x.pow(1.0 / p)
    
    class BirdSEDModel(nn.Module):
        """SED model with 1st-place-inspired design: https://www.kaggle.com/code/nikitababich/birdclef2025-1st-place-inference
        - GeMFreq pooling (learnable, sharper than mean)
        - 512-dim bottleneck with dropout
        - Attention-weighted clip logits from frame logits
        - Distillation: GAP+Linear branch for MSE to Perch
        - Stop gradient: backbone trains from distillation only
        """
        def __init__(self, backbone_name=BACKBONE_NAME, num_classes=NUM_CLASSES,
                     drop_path_rate=0.1, hidden_dim=512):
            super().__init__()
            self.backbone = timm.create_model(
                backbone_name, pretrained=True, in_chans=1,
                num_classes=0, global_pool="", drop_path_rate=drop_path_rate,
            )
            with torch.no_grad():
                n_tf = TRAIN_SAMPLES // HOP_LENGTH + 1
                dummy = torch.randn(1, 1, N_MELS, n_tf)
                feat = self.backbone(dummy)
                self.backbone_dim = feat.shape[1]
                print(f"V2 backbone: {tuple(feat.shape)}  (C={self.backbone_dim})")
    
            self.gem_freq = GeMFreqPool(p_init=3.0)
            self.dense = nn.Sequential(
                nn.Dropout(0.25),
                nn.Linear(self.backbone_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
            )
            self.att = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
            self.cla = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
            nn.init.xavier_uniform_(self.att.weight)
            nn.init.xavier_uniform_(self.cla.weight)
            self.att.bias.data.fill_(0.)
            self.cla.bias.data.fill_(0.)
            if USE_PERCH_DISTILL:
                self.distill_head = DistillHead(self.backbone_dim, PERCH_EMBED_DIM)
    
        def forward(self, x, return_framewise=False, return_distill=False):
            h = self.backbone(x)
            distill_emb = None
            if return_distill and hasattr(self, 'distill_head'):
                distill_emb = self.distill_head(h)
    
            # Stop gradient: SED head doesn't update the backbone
            h_cls = h.detach() if USE_PERCH_DISTILL else h
    
            h_cls = self.gem_freq(h_cls)            # (B, C, T)
            h_cls = h_cls.permute(0, 2, 1)          # (B, T, C)
            h_cls = self.dense(h_cls)               # (B, T, 512)
            h_cls = h_cls.permute(0, 2, 1)          # (B, 512, T)
    
            norm_att = torch.softmax(torch.tanh(self.att(h_cls)), dim=-1)
            framewise_logits = self.cla(h_cls)
            clip_logits = torch.sum(norm_att * framewise_logits, dim=2)
    
            fw = framewise_logits.permute(0, 2, 1) if return_framewise else None
            if return_framewise and return_distill: return clip_logits, fw, distill_emb
            elif return_framewise: return clip_logits, fw
            elif return_distill: return clip_logits, distill_emb
            return clip_logits
    
    def make_model():
            return BirdSEDModel(BACKBONE_NAME).to(device)
    
    print("OK Model definitions ready")
    
    
    # =================================================================
    # S4 -- DATA PIPELINE
    # =================================================================
    
    def load_int16(path):
        """Load int16 waveform tensor to float32 in [-1, 1]."""
        waveform_int16 = torch.load(path, map_location="cpu")
        return waveform_int16.float() / 32767.0
    
    _FC = {}
    def load_focal(p):
        """Load focal waveform with simple LRU cache."""
        if p in _FC: return _FC[p]
        pp = WAVEFORM_CACHE_DIR / p
        if not pp.exists(): return None
        a = load_int16(pp).numpy()
        if len(_FC) >= 2000:
            _FC.pop(next(iter(_FC)))
        _FC[p] = a
        return a
    
    _SC_CACHE = {}
    def load_sc_waveform_from(cache_dir, cache_file):
        """Load a soundscape waveform with LRU cache."""
        key = str(cache_dir / cache_file)
        if key in _SC_CACHE: return _SC_CACHE[key]
        pp = cache_dir / cache_file
        if not pp.exists(): return None
        a = load_int16(pp).numpy()
        if len(_SC_CACHE) >= 200:
            _SC_CACHE.pop(next(iter(_SC_CACHE)))
        _SC_CACHE[key] = a
        return a
    
    def extract_chunk_np(waveform, start_sample, n_samples):
        """Extract a chunk, left-padding if the recording is too short."""
        total = len(waveform)
        if total <= n_samples:
            return np.pad(waveform, (n_samples - total, 0))
        end = start_sample + n_samples
        if end > total:
            start_sample = max(0, total - n_samples)
        return waveform[start_sample:start_sample + n_samples]
    
    def apply_aug(w):
        """Simple waveform augmentation: gain jitter + noise + shift."""
        if np.random.random() < AUG_PROB:
            w = w * (10 ** (np.random.uniform(*AUG_GAIN_DB_RANGE) / 20))
        if np.random.random() < AUG_PROB:
            sp = (w ** 2).mean()
            if sp > 1e-10:
                w = w + np.random.randn(*w.shape).astype(w.dtype) * np.sqrt(
                    sp / (10 ** (np.random.uniform(*AUG_NOISE_SNR_DB_RANGE) / 10)))
        return w
    
    # ------------------------------------------------------------------
    # Build soundscape MixUp pool (labeled windows only)
    # ------------------------------------------------------------------
    sc_mixup_sources = []
    _sc_file_meta = pd.read_csv(WAVEFORM_CACHE_DIR / "soundscape_file_meta.csv")
    _sc_file_dict = dict(zip(_sc_file_meta["filename"], _sc_file_meta["cache_file"]))
    _labeled_rows = []
    for i in range(len(sc_cache_meta)):
        row = sc_cache_meta.iloc[i]
        if Y_SC[i].sum() > 0:
            cf = _sc_file_dict.get(row["filename"])
            if cf is not None:
                _labeled_rows.append({
                    "filename": row["filename"], "start_sec": int(row["start_sec"]),
                    "cache_file": cf, "label_idx": i, "fold": int(row.get("fold", -1)),
                })
    if _labeled_rows:
        _labeled_meta = pd.DataFrame(_labeled_rows)
        sc_mixup_sources.append((WAVEFORM_CACHE_DIR, _labeled_meta, Y_SC))
        print(f"SC MixUp pool: {len(_labeled_meta)} labeled windows")
    
    # ------------------------------------------------------------------
    # FocalDS -- with Focal-Focal AND Focal-Soundscape MixUp
    # ------------------------------------------------------------------
    class FocalDS(Dataset):
        """Focal recording dataset. Returns (waveform, label, weight, mask, source_tag)."""
        def __init__(self, df, l2i, secondary_lookup=None,
                     sc_mixup_sources=None, fold_k=None, aug=False):
            self.df, self.l2i, self.aug = df.reset_index(drop=True), l2i, aug
            self.secondary_lookup = secondary_lookup
            self.sc_mixup_sources = sc_mixup_sources
            self.fold_k = fold_k
    
        def __len__(self): return len(self.df)
    
        def _load_chunk(self, r):
            w = load_focal(r["cache_file"])
            if w is None: return None, None
            if self.aug:
                start = np.random.randint(0, max(1, len(w) - TRAIN_SAMPLES + 1)) if len(w) > TRAIN_SAMPLES else 0
            else:
                start = int(r.get("start_sec", 0)) * SR
            ch = extract_chunk_np(w, start, TRAIN_SAMPLES)
            lb = np.zeros(NUM_CLASSES, dtype=np.float32)
            if str(r["primary_label"]) in self.l2i:
                lb[self.l2i[str(r["primary_label"])]] = 1.0
            if self.secondary_lookup is not None and "original_idx" in self.df.columns:
                for s in self.secondary_lookup.get(int(r["original_idx"]), []):
                    if s in self.l2i: lb[self.l2i[s]] = 1.0
            return ch, lb
    
        def __getitem__(self, i):
            r1 = self.df.iloc[i]
            ch1, lb1 = self._load_chunk(r1)
            if ch1 is None:
                return (torch.zeros(1, TRAIN_SAMPLES), torch.zeros(NUM_CLASSES),
                        torch.ones(NUM_CLASSES), torch.ones(NUM_CLASSES), "focal_missing")
    
            # Focal-Focal MixUp
            if USE_FOCAL_MIXUP and self.aug and np.random.random() < MIXUP_PROB:
                ch2 = None
                for _ in range(3):
                    j = np.random.randint(len(self.df))
                    ch2, lb2 = self._load_chunk(self.df.iloc[j])
                    if ch2 is not None: break
                if ch2 is not None:
                    lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
                    ch_mix = (lam * ch1 + (1 - lam) * ch2).astype(np.float32)
                    if self.aug: ch_mix = apply_aug(ch_mix)
                    lb = np.maximum(lb1, lb2) if MIXUP_HARD else (lam * lb1 + (1 - lam) * lb2)
                    return (torch.from_numpy(ch_mix).unsqueeze(0), torch.from_numpy(lb),
                            torch.ones(NUM_CLASSES), torch.ones(NUM_CLASSES), "focal")
    
            # Focal-Soundscape MixUp
            if (USE_FOCAL_SC_MIXUP and self.aug and self.sc_mixup_sources
                    and np.random.random() < FOCAL_SC_MIXUP_PROB):
                src_idx = np.random.randint(len(self.sc_mixup_sources))
                cache_dir, meta_df_sc, labels = self.sc_mixup_sources[src_idx]
                eligible = meta_df_sc[meta_df_sc["fold"] != self.fold_k] if self.fold_k is not None else meta_df_sc
                if len(eligible) > 0:
                    sc_row = eligible.iloc[np.random.randint(len(eligible))]
                    sc_wav = load_sc_waveform_from(cache_dir, sc_row["cache_file"])
                    if sc_wav is not None and len(sc_wav) >= TRAIN_SAMPLES:
                        sc_chunk = extract_chunk_np(sc_wav, int(sc_row["start_sec"]) * SR, TRAIN_SAMPLES)
                        lam = np.random.beta(FOCAL_SC_MIXUP_ALPHA, FOCAL_SC_MIXUP_ALPHA)
                        ch_mix = (lam * ch1 + (1 - lam) * sc_chunk).astype(np.float32)
                        if self.aug: ch_mix = apply_aug(ch_mix)
                        lb_sc = labels[int(sc_row["label_idx"])].astype(np.float32)
                        lb = np.maximum(lb1, lb_sc) if MIXUP_HARD else lam * lb1 + (1 - lam) * lb_sc
                        return (torch.from_numpy(ch_mix).unsqueeze(0), torch.from_numpy(lb),
                                torch.ones(NUM_CLASSES), torch.ones(NUM_CLASSES), "focal")
    
            # No MixUp
            if self.aug: ch1 = apply_aug(ch1)
            return (torch.from_numpy(ch1.astype(np.float32)).unsqueeze(0),
                    torch.from_numpy(lb1),
                    torch.ones(NUM_CLASSES), torch.ones(NUM_CLASSES), "focal")
    
    # ------------------------------------------------------------------
    # ScDS -- Labeled soundscape windows
    # ------------------------------------------------------------------
    class ScDS(Dataset):
        def __init__(self, Y, sc_df, aug=False):
            self.Y, self.df, self.aug = Y, sc_df.reset_index(drop=True), aug
        def __len__(self): return len(self.Y)
        def __getitem__(self, i):
            row = self.df.iloc[i]
            wav_full = load_sc_waveform_from(WAVEFORM_CACHE_DIR, row.get("cache_file")) \
                       if row.get("cache_file") else None
            if wav_full is None:
                wav_t = torch.zeros(1, TRAIN_SAMPLES)
            else:
                chunk = extract_chunk_np(wav_full, int(row["start_sec"]) * SR, TRAIN_SAMPLES)
                if self.aug: chunk = apply_aug(chunk)
                wav_t = torch.from_numpy(chunk.astype(np.float32)).unsqueeze(0)
            return (wav_t, torch.from_numpy(self.Y[i].astype(np.float32)),
                    torch.ones(NUM_CLASSES), torch.ones(NUM_CLASSES), "sc")
    
    # ------------------------------------------------------------------
    # Load focal secondary labels
    # ------------------------------------------------------------------
    focal_secondary_labels = None
    if USE_FOCAL_SECONDARY:
        focal_secondary_labels = {}
        for idx, row in train_df.iterrows():
            sec = row.get("secondary_labels", "")
            if pd.isna(sec) or sec in ("", "[]"): continue
            try:
                sec_list = eval(sec) if isinstance(sec, str) else []
            except: continue
            valid = [s for s in sec_list if s in LABEL2IDX]
            if valid: focal_secondary_labels[idx] = valid
        print(f"Focal secondary labels: {len(focal_secondary_labels)} files")
    
    # ------------------------------------------------------------------
    # Multi-source batch sampler
    # ------------------------------------------------------------------
    class MixSamp(torch.utils.data.Sampler):
        """Controls batch composition via per-source shares."""
        def __init__(self, sizes, names, shares, bs, nst, seed=0):
            self.sizes, self.names, self.bs, self.nst = sizes, names, bs, nst
            self.rng = np.random.default_rng(seed)
            per_src = [max(1, int(round(bs * shares.get(n, 0.0)))) for n in names]
            total = sum(per_src)
            if total != bs:
                per_src[int(np.argmax(per_src))] += (bs - total)
            self.per_src = per_src
            self.offsets = [0]
            for s in sizes[:-1]:
                self.offsets.append(self.offsets[-1] + s)
        def __len__(self): return self.nst
        def __iter__(self):
            for _ in range(self.nst):
                batch = []
                for off, size, n in zip(self.offsets, self.sizes, self.per_src):
                    if n <= 0 or size <= 0: continue
                    idxs = self.rng.integers(0, size, size=n)
                    batch.extend([off + int(i) for i in idxs])
                self.rng.shuffle(batch)
                yield batch
    
    def collate_m(batch):
        return (torch.stack([b[0] for b in batch]),
                torch.stack([b[1] for b in batch]),
                torch.stack([b[2] for b in batch]),
                torch.stack([b[3] for b in batch]),
                [b[4] for b in batch])
    
    def mk_sw(sr):
        """Per-sample source weight tensor."""
        return torch.tensor([SOURCE_WEIGHTS.get(s, 0.0) for s in sr], dtype=torch.float32)
    
    print("OK Data pipeline ready")
    
    
    # =================================================================
    # S5 -- TRAINING
    # =================================================================
    
    def _load_val_waveforms(val_sc_df):
        """Load validation waveforms (always 5s)."""
        sc_file_meta = pd.read_csv(WAVEFORM_CACHE_DIR / "soundscape_file_meta.csv")
        sc_file_dict = dict(zip(sc_file_meta["filename"], sc_file_meta["cache_file"]))
        wavs = []
        for _, row in val_sc_df.iterrows():
            cf = sc_file_dict.get(row["filename"])
            if cf is not None:
                w = load_sc_waveform_from(WAVEFORM_CACHE_DIR, cf)
                if w is not None:
                    chunk = extract_chunk_np(w, int(row["start_sec"]) * SR, VAL_SAMPLES)
                    wavs.append(torch.from_numpy(chunk.astype(np.float32)).unsqueeze(0))
                else: wavs.append(torch.zeros(1, VAL_SAMPLES))
            else: wavs.append(torch.zeros(1, VAL_SAMPLES))
        return wavs
    
    def _predict_from_waveforms(model, mel_transform, wav_list, batch_size=64):
        """Inference: mel -> model -> sigmoid. Distillation head is NOT used."""
        model.eval()
        preds_clip, preds_fmax, preds_blend = [], [], []
        with torch.no_grad():
            for s in range(0, len(wav_list), batch_size):
                batch = torch.stack(wav_list[s:s+batch_size]).to(device)
                mel = mel_transform(batch)
                B = mel.size(0)
                for i in range(B):
                    mel[i] = (mel[i] - mel[i].mean()) / (mel[i].std() + 1e-6)
                with autocast():
                    clip_logits, framewise = model(mel, return_framewise=True)
                    frame_max = framewise.max(dim=1).values
                    p_clip = torch.sigmoid(clip_logits).cpu().numpy()
                    p_fmax = torch.sigmoid(frame_max).cpu().numpy()
                    p_blend = 0.5 * p_clip + 0.5 * p_fmax
                preds_clip.append(p_clip); preds_fmax.append(p_fmax); preds_blend.append(p_blend)
        return {"clip": np.concatenate(preds_clip), "fmax": np.concatenate(preds_fmax),
                "blend": np.concatenate(preds_blend)}
    
    def build_active_datasets(fold_k):
        items = []
        if USE_FOCAL:
            fds = FocalDS(audio_cache_meta[audio_cache_meta["fold"] != fold_k],
                          LABEL2IDX, secondary_lookup=focal_secondary_labels,
                          sc_mixup_sources=sc_mixup_sources if USE_FOCAL_SC_MIXUP else None,
                          fold_k=fold_k, aug=True)
            items.append(("focal", fds, len(fds)))
        if USE_LABELED_SC:
            vm = sc_cache_meta["fold"].values == fold_k
            sc_train_df = sc_cache_meta[~vm].reset_index(drop=True)
            Y_tr = Y_SC[~vm]
            sds = ScDS(Y_tr, sc_train_df, aug=True)
            items.append(("sc", sds, len(sds)))
        return items
    
    def train_fold(fold_k):
        vm = sc_cache_meta["fold"].values == fold_k
        Y_val = Y_SC[vm]
        ns22_val = non_s22_mask_sc[vm]
        val_sc_df = sc_cache_meta[vm].reset_index(drop=True)
    
        active = build_active_datasets(fold_k)
        names, datasets, sizes = zip(*active)
        mds = ConcatDataset(list(datasets))
        nst = max(100, int(sum(sizes) / BATCH))
    
        print(f"  Streams: {dict(zip(names, sizes))}  steps/ep: {nst}")
    
        m = make_model()
        mel_transform = MelSpecTransform().to(device)
        spec_augment = SpecAugment().to(device)
        perch_teacher = PerchTeacher(PERCH_ONNX_PATH,
                                      "cuda" if torch.cuda.is_available() else "cpu") \
                        if USE_PERCH_DISTILL else None
    
        opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
        scaler = GradScaler()
        warmup_steps = nst * WARMUP_EPOCHS
        total_steps  = nst * EPOCHS
        warmup_sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1/25, end_factor=1.0,
                                                          total_iters=warmup_steps)
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps - warmup_steps,
                                                                   eta_min=1e-6)
        sch = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warmup_sched, cosine_sched],
                                                     milestones=[warmup_steps])
    
        history = {"ep": [], "train_loss": [], "cls_loss": [], "dist_loss": [],
                   "macro": [], "ns22_macro": [],
                   "ns22_Aves": [], "ns22_Amphibia": [], "ns22_Insecta": [], "ns22_Mammalia": [],
                   "val_preds": []}
        best_ns22, best_state_ns22 = -1.0, None
        best_macro, best_state_macro = -1.0, None
        val_wavs = _load_val_waveforms(val_sc_df)
    
        for ep in range(EPOCHS):
            m.train()
            smp = MixSamp(list(sizes), list(names), SHARES, BATCH, nst, seed=42 + ep)
            tl = DataLoader(mds, batch_sampler=smp, collate_fn=collate_m,
                            num_workers=0, pin_memory=True)
            el, el_cls, el_dist, nb_count = 0.0, 0.0, 0.0, 0
            t0 = time.time()
    
            for wav, lb, wt, mk, sr in tl:
                wav, lb, wt, mk = wav.to(device), lb.to(device), wt.to(device), mk.to(device)
                sw = mk_sw(sr).to(device)
    
                with torch.no_grad():
                    mel = mel_transform(wav)
                    B = mel.size(0)
                    for i in range(B):
                        mel[i] = (mel[i] - mel[i].mean()) / (mel[i].std() + 1e-6)
                    mel = spec_augment(mel)
    
                with autocast():
                    if USE_PERCH_DISTILL:
                        clip_logits, framewise, distill_emb = m(mel, return_framewise=True,
                                                                return_distill=True)
                    else:
                        clip_logits, framewise = m(mel, return_framewise=True)
    
                    frame_max_logits = framewise.max(dim=1).values
    
                    # Classification loss
                    bce_clip = F.binary_cross_entropy_with_logits(clip_logits, lb, reduction="none")
                    bce_frame = F.binary_cross_entropy_with_logits(frame_max_logits, lb, reduction="none")
                    bce = 0.5 * bce_clip + 0.5 * bce_frame
                    ps = (bce * wt * mk).sum(1) / (mk.sum(1) + 1e-8)
                    cls_loss = (ps * sw).mean()
    
                    # Distillation loss
                    if USE_PERCH_DISTILL and perch_teacher is not None:
                        with torch.no_grad():
                            wav_5s = wav.squeeze(1)
                            N = wav_5s.shape[1]
                            if N > 160000:
                                start = (N - 160000) // 2
                                wav_5s = wav_5s[:, start:start + 160000]
                            elif N < 160000:
                                wav_5s = F.pad(wav_5s, (0, 160000 - N))
                            perch_emb = perch_teacher.embed(wav_5s).to(device)
                        distill_loss = F.mse_loss(distill_emb, perch_emb)
                        loss = cls_loss + ALPHA_DISTILL * distill_loss
                    else:
                        distill_loss = torch.tensor(0.0)
                        loss = cls_loss
    
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sch.step()
                el += loss.item(); el_cls += cls_loss.item()
                el_dist += distill_loss.item(); nb_count += 1
    
            # Validation
            val_preds_dict = _predict_from_waveforms(m, mel_transform, val_wavs)
            val_preds = val_preds_dict["blend"]
            r = full_eval(Y_val, val_preds, ns22_val, TAXON_MASKS)
            for mode in ["clip", "fmax", "blend"]:
                r_mode = full_eval(Y_val, val_preds_dict[mode], ns22_val, TAXON_MASKS)
                r[f"ns22_{mode}"] = r_mode["non_s22_macro"]
    
            history["ep"].append(ep)
            history["train_loss"].append(round(el / nb_count, 5))
            history["cls_loss"].append(round(el_cls / nb_count, 5))
            history["dist_loss"].append(round(el_dist / nb_count, 5))
            history["macro"].append(r["macro_auc_all"])
            history["ns22_macro"].append(r["non_s22_macro"])
            for t in ["Aves", "Amphibia", "Insecta", "Mammalia"]:
                history[f"ns22_{t}"].append(r[f"non_s22_{t}"])
            history["val_preds"].append(val_preds.astype(np.float32))
    
            tag_ns22 = ""; tag_macro = ""
            if r["non_s22_macro"] > best_ns22:
                best_ns22 = r["non_s22_macro"]
                best_state_ns22 = {k: v.cpu().clone() for k, v in m.state_dict().items()}
                tag_ns22 = " *ns22"
            if r["macro_auc_all"] > best_macro:
                best_macro = r["macro_auc_all"]
                best_state_macro = {k: v.cpu().clone() for k, v in m.state_dict().items()}
                tag_macro = " *macro"
    
            dist_str = f" dist={el_dist/nb_count:.4f}" if USE_PERCH_DISTILL else ""
            print(f"    Ep{ep:02d}: loss={el/nb_count:.4f} cls={el_cls/nb_count:.4f}{dist_str} "
                  f"lr={opt.param_groups[0]['lr']:.1e} | "
                  f"ns22: {r['ns22_blend']:.4f} | "
                  f"Av={r['non_s22_Aves']:.4f} Am={r['non_s22_Amphibia']:.4f} "
                  f"In={r['non_s22_Insecta']:.4f} Ma={r['non_s22_Mammalia']:.4f} "
                  f"[{time.time()-t0:.0f}s]{tag_ns22}{tag_macro}")
    
        del perch_teacher, m, mel_transform, spec_augment
        torch.cuda.empty_cache(); gc.collect()
        return best_state_ns22, best_state_macro, history
    
    print("OK Training function ready")
    
    
    # =================================================================
    # S6 -- FOLD LOOP + ONNX EXPORT
    # =================================================================
    
    if MODE != "train":
        print("Skipping training (MODE='infer')")
        oof_ns22 = None
        all_hist = {}
    else:
    
        oof_ns22 = np.full((len(sc_cache_meta), NUM_CLASSES), np.nan, dtype=np.float32)
        all_hist = {}
        for fold_k in FOLDS:
            print(f"\n{'='*60}")
            print(f"FOLD {fold_k}")
            print(f"{'='*60}")
            vm = sc_cache_meta["fold"].values == fold_k
            val_sc_df_k = sc_cache_meta[vm].reset_index(drop=True)
        
            best_ns22_state, best_macro_state, hist = train_fold(fold_k)
            all_hist[fold_k] = hist
        
            mel_tf = MelSpecTransform().to(device)
            val_wavs_k = _load_val_waveforms(val_sc_df_k)
        
            if best_macro_state is not None:
                # Save PyTorch checkpoint
                torch.save(best_macro_state, OUT_DIR / f"fold{fold_k}_best_ns22.pt")
                m = make_model()
                m.load_state_dict(best_macro_state, strict=False)
                oof_ns22[vm] = _predict_from_waveforms(m, mel_tf, val_wavs_k)["blend"]
        
                # --- ONNX Export (Conv1d remap for stable tracing) ---
                m.eval()
                INF_N_MELS = 128
                INF_N_FRAMES = VAL_SAMPLES // HOP_LENGTH + 1
        
                class SEDExportWrapper(nn.Module):
                    def __init__(self, backbone_name, num_classes, backbone_dim, hidden_dim=512):
                        super().__init__()
                        self.backbone = timm.create_model(
                            backbone_name, pretrained=False, in_chans=1,
                            num_classes=0, global_pool="", drop_path_rate=0.1,
                        )
                        self.gem_freq = GeMFreqPool(p_init=3.0)
                        self.dense_drop1 = nn.Dropout(0.25)
                        self.dense_conv = nn.Conv1d(backbone_dim, hidden_dim, 1)
                        self.dense_relu = nn.ReLU(inplace=True)
                        self.dense_drop2 = nn.Dropout(0.5)
                        self.att = nn.Conv1d(hidden_dim, num_classes, 1)
                        self.cla = nn.Conv1d(hidden_dim, num_classes, 1)
        
                    def forward(self, mel):
                        h = self.backbone(mel)
                        h = self.gem_freq(h)
                        h = self.dense_drop1(h)
                        h = self.dense_conv(h)
                        h = self.dense_relu(h)
                        h = self.dense_drop2(h)
                        norm_att = torch.softmax(torch.tanh(self.att(h)), dim=-1)
                        framewise = self.cla(h)
                        clip = torch.sum(norm_att * framewise, dim=2)
                        return clip, framewise.permute(0, 2, 1)
        
                def load_and_remap_state(export_model, trained_state):
                    remap = {}
                    for k, v in trained_state.items():
                        if k.startswith("distill_head."):
                            continue
                        if k == "dense.1.weight":
                            remap["dense_conv.weight"] = v.unsqueeze(-1)
                        elif k == "dense.1.bias":
                            remap["dense_conv.bias"] = v
                        else:
                            remap[k] = v
                    export_model.load_state_dict(remap, strict=False)
        
                export_model = SEDExportWrapper(
                    BACKBONE_NAME, NUM_CLASSES, m.backbone_dim
                ).to(device)
                load_and_remap_state(export_model, best_macro_state)
                export_model.eval()
        
                dummy_mel = torch.randn(1, 1, INF_N_MELS, INF_N_FRAMES).to(device)
                onnx_path = OUT_DIR / f"sed_distill_fold{fold_k}.onnx"
                torch.onnx.export(
                    export_model, dummy_mel, str(onnx_path),
                    input_names=["mel"],
                    output_names=["clip_logits", "framewise_logits"],
                    dynamic_axes={"mel": {0: "batch"},
                                  "clip_logits": {0: "batch"},
                                  "framewise_logits": {0: "batch"}},
                    opset_version=17,
                )
        
                # Verify
                _sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
                _onnx_out = _sess.run(None, {'mel': dummy_mel.cpu().numpy()})
                with torch.no_grad():
                    _ref_clip, _ref_frame = export_model(dummy_mel)
                _diff = np.abs(_ref_clip.cpu().numpy() - _onnx_out[0]).max()
                print(f"  ONNX verify: max|diff|={_diff:.3e}")
                assert _diff < 1e-3, f"ONNX export diverged: {_diff}"
                del _sess
        
                size_mb = onnx_path.stat().st_size / 1e6
                print(f"  Exported {onnx_path.name} ({size_mb:.1f} MB)")
                del m, export_model
    
    # =================================================================
    # S7 -- EVALUATION SUMMARY
    # =================================================================
    
    if MODE != "train":
        print("Skipping evaluation (MODE='infer')")
    else:
    
        has = ~np.isnan(oof_ns22[:, 0])
        if has.sum() > 0:
            r_all = full_eval(Y_SC[has], oof_ns22[has], non_s22_mask_sc[has], TAXON_MASKS)
            print("=" * 60)
            print("OOF RESULTS (best-ns22 checkpoints)")
            print("=" * 60)
            print(f"  macro AUC (all):        {r_all['macro_auc_all']:.4f}")
            print(f"  macro AUC (non-S22):    {r_all['non_s22_macro']:.4f}")
            for t in ["Aves", "Amphibia", "Insecta", "Mammalia"]:
                print(f"    {t:<12}: {r_all.get(f'non_s22_{t}', float('nan')):.4f}")
        
        # Per-epoch progression
        print("\nPer-epoch pooled non-S22 AUC:")
        fold_true, fold_ns22_m = {}, {}
        for fk in range(N_FOLDS):
            vm = sc_cache_meta["fold"].values == fk
            fold_true[fk] = Y_SC[vm]
            fold_ns22_m[fk] = non_s22_mask_sc[vm]
        
        n_eps = [len(all_hist[k]["val_preds"]) for k in range(N_FOLDS) if k in all_hist]
        max_ep = min(n_eps) if n_eps else 0
        for ep in range(max_ep):
            pp = np.concatenate([all_hist[k]["val_preds"][ep] for k in range(N_FOLDS) if k in all_hist])
            pt = np.concatenate([fold_true[k] for k in range(N_FOLDS) if k in all_hist])
            pm = np.concatenate([fold_ns22_m[k] for k in range(N_FOLDS) if k in all_hist])
            ns, _ = compute_macro_auc(pt, pp, mask=pm)
            print(f"  Ep{ep:02d}: {ns:.4f}")
    
    
    # =================================================================
    # INFERENCE -- Mel spectrogram (librosa)
    # =================================================================
    if MODE != "infer":
        print("Skipping inference setup (MODE='train')")
    else:
        import librosa
    
        INF_N_MELS   = 256
        INF_N_FFT    = 2048
        INF_HOP      = 512
        INF_FMIN     = 20
        INF_FMAX     = 16000
        INF_TOP_DB   = 80
        INF_SR       = 32000
        INF_CHUNK_S  = 5
        INF_CHUNK_N  = INF_SR * INF_CHUNK_S   # 160,000
        INF_N_FRAMES = INF_CHUNK_N // INF_HOP + 1  # 313
        
        def audio_to_mel(chunks):
            """Raw audio chunks (N, 160000) -> normalized mel dB (N, 1, 128, 313)."""
            mels = []
            for i in range(chunks.shape[0]):
                S = librosa.feature.melspectrogram(
                    y=chunks[i], sr=INF_SR, n_fft=INF_N_FFT, hop_length=INF_HOP,
                    n_mels=INF_N_MELS, fmin=INF_FMIN, fmax=INF_FMAX, power=2.0,
                )
                S_dB = librosa.power_to_db(S, top_db=INF_TOP_DB)
                S_dB = (S_dB - S_dB.mean()) / (S_dB.std() + 1e-6)
                mels.append(S_dB)
            return np.stack(mels)[:, np.newaxis, :, :].astype(np.float32)
        
        print(f"Inference mel: {INF_N_MELS} mels, {INF_N_FRAMES} frames/chunk")
    
    
    # =================================================================
    # INFERENCE -- Load ONNX sessions
    # =================================================================
    if MODE != "infer":
        print("Skipping ONNX loading (MODE='train')")
    else:
        import re, glob
    
        def discover_folds(sed_dir):
            pat = re.compile(r'sed_fold(\d+)\.onnx$')
            folds = []
            for fname in os.listdir(sed_dir):
                m = pat.match(fname)
                if m: folds.append(int(m.group(1)))
            return sorted(folds)
        
        def make_session(onnx_path):
            so = ort.SessionOptions()
            so.intra_op_num_threads = 4
            so.inter_op_num_threads = 1
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            return ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
        
        SED_DIR_CANDIDATES = [
            str(OUT_DIR),
            '/kaggle/input/datasets/tuckerarrants/bc2026-distilled-sed-public',
        ]
        SED_DIR = next((p for p in SED_DIR_CANDIDATES if os.path.isdir(p) and
                        any(f.endswith('.onnx') and 'sed' in f for f in os.listdir(p))), None)
        assert SED_DIR, f'No SED ONNX files found in {SED_DIR_CANDIDATES}'
        
        INF_FOLDS = discover_folds(SED_DIR)
        assert INF_FOLDS, f'No sed_distill_fold*.onnx in {SED_DIR}'
        print(f'Found {len(INF_FOLDS)} fold(s) in {SED_DIR}: {INF_FOLDS}')
        
        fold_sessions = []
        for fold in INF_FOLDS:
            p = f'{SED_DIR}/sed_fold{fold}.onnx'
            sess = make_session(p)
            fold_sessions.append(sess)
            size_mb = os.path.getsize(p) / 1e6
            print(f'  fold {fold}: {size_mb:5.1f}MB  providers={sess.get_providers()}')
    
    
    # =================================================================
    # INFERENCE -- Audio loading + main loop
    # =================================================================
    from scipy.ndimage import convolve1d
    
    GAUSSIAN_KERNEL = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    N_WINDOWS = 12  # 60s / 5s
    
    if MODE != "infer":
        print("Skipping inference (MODE='train')")
    else:
        try:
            import soundfile as sf
            DECODER = 'soundfile'
        except ImportError:
            DECODER = 'librosa'
        print(f'Audio decoder: {DECODER}')
    
        def load_audio_32k_mono(path):
            if DECODER == 'soundfile':
                wav, sr = sf.read(path, dtype='float32', always_2d=False)
                if wav.ndim > 1: wav = wav.mean(axis=1)
                if sr != INF_SR:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=INF_SR)
                return wav.astype(np.float32)
            else:
                wav, _ = librosa.load(path, sr=INF_SR, mono=True)
                return wav.astype(np.float32)
    
        def file_to_chunks(path):
            wav = load_audio_32k_mono(path)
            target_len = 60 * INF_SR
            if len(wav) < target_len:
                wav = np.pad(wav, (0, target_len - len(wav)))
            elif len(wav) > target_len:
                wav = wav[:target_len]
            n_chunks = target_len // INF_CHUNK_N
            chunks = wav[:n_chunks * INF_CHUNK_N].reshape(n_chunks, INF_CHUNK_N)
            end_times = np.arange(1, n_chunks + 1) * INF_CHUNK_S
            return chunks.astype(np.float32), end_times
    
        def sigmoid_inf(x):
            return np.where(
                x >= 0,
                1.0 / (1.0 + np.exp(-np.clip(x, -50, 50))),
                np.exp(np.clip(x, -50, 50)) / (1.0 + np.exp(np.clip(x, -50, 50))),
            ).astype(np.float32)
    
        def gauss_smooth_final(scores, weights=GAUSSIAN_KERNEL):
            """Gaussian smooth predictions across the 12 windows within each file.
            Operates on the leading axis of the per-file slab."""
            smoothed = scores.reshape(-1, N_WINDOWS, scores.shape[1]).copy()
            for i in range(smoothed.shape[0]):
                smoothed[i] = convolve1d(smoothed[i], weights, axis=0, mode='nearest')
            return smoothed.reshape(-1, scores.shape[1])
    
        # Discover test files (with train_soundscapes fallback for debugging)
        test_files = sorted(glob.glob(f'{TEST_DIR}/*.ogg')) if TEST_DIR.is_dir() else []
        if len(test_files) == 0:
            fallback = COMP_DIR / "train_soundscapes"
            if fallback.is_dir():
                test_files = sorted(glob.glob(f'{fallback}/*.ogg'))[:5]
                print(f'No test_soundscapes -- using {len(test_files)} train files for debug')
        print(f'Test files: {len(test_files)}')
    
        # --- Main inference loop ---
        t0 = time.time()
        all_rows, all_preds = [], []
    
        for file_idx, file_path in enumerate(test_files):
            basename = os.path.basename(file_path).replace('.ogg', '')
            chunks, end_times = file_to_chunks(file_path)
            mel = audio_to_mel(chunks)
    
            # Accumulate logits in logit space across folds (matches the
            # smoothing-in-logit-space convention from the other inference notebook).
            # Per fold: blend clip + frame_max in LOGIT space.
            logits_sum = np.zeros((chunks.shape[0], NUM_CLASSES), dtype=np.float32)
            for sess in fold_sessions:
                outs = sess.run(None, {'mel': mel})
                clip_logits = outs[0]
                frame_max = outs[1].max(axis=1)
                logits_sum += 0.5 * clip_logits + 0.5 * frame_max
            logits_mean = logits_sum / len(INF_FOLDS)
    
            # Smooth across windows in logit space, then sigmoid once.
            logits_smoothed = gauss_smooth_final(logits_mean)
            probs = sigmoid_inf(logits_smoothed)
    
            all_rows.extend([f'{basename}_{int(t)}' for t in end_times])
            all_preds.append(probs)
    
            if (file_idx + 1) % 50 == 0 or file_idx == 0 or file_idx == len(test_files) - 1:
                elapsed = time.time() - t0
                rate = (file_idx + 1) / elapsed
                print(f'  [{file_idx+1:4d}/{len(test_files)}] {elapsed:.1f}s  {rate:.2f} files/s')
    
        all_preds_arr = np.concatenate(all_preds) if all_preds else np.zeros((0, NUM_CLASSES), np.float32)
        print(f'\nInference: {len(all_rows)} rows, {time.time()-t0:.1f}s total')
    
    # =================================================================
    # WRITE SUBMISSION
    # =================================================================
    if MODE != "infer":
        print("Skipping submission (MODE='train')")
    else:
        submission = pd.DataFrame(all_preds_arr, columns=PRIMARY_LABELS)
        submission.insert(0, 'row_id', all_rows)
        
        assert submission.shape[1] == NUM_CLASSES + 1
        assert submission['row_id'].is_unique
        assert not submission.iloc[:, 1:].isna().any().any()
        submission.iloc[:, 1:] = submission.iloc[:, 1:].clip(0.0, 1.0)
        
        submission.to_csv(_file_name_submission, index=False)
        print(f'Wrote submission.csv: {len(submission)} rows x {submission.shape[1]} cols')
        print(submission.head(3).iloc[:, :6])



# === ## Model_3

# === CELL 4 ===

if 'Model_3' in _ensemble_models:

    _file_name_submission = "subm_3.csv"
    
    # Cell 0 ? Install ONNX Runtime & TF 2.20
    import subprocess, sys, os
    
    # ONNX Runtime installation
    ort_whl = "/kaggle/input/datasets/hellodave2035/birdclef-perch-models/onnxruntime-1.24.4-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
    if os.path.exists(ort_whl):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", ort_whl])
    
    # TensorFlow installation
    tf_dir = "/kaggle/input/notebooks/ashok205/tf-wheels/tf_wheels"
    if os.path.exists(tf_dir):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", f"{tf_dir}/tensorboard-2.20.0-py3-none-any.whl"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", f"{tf_dir}/tensorflow-2.20.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"])
    
    import random
    import os
    import tensorflow as tf
    import torch
    import numpy as np
    
    def seed_everything(seed=42):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        tf.random.set_seed(seed)
    
    seed_everything(1891)
    
    try:
        import onnxruntime as ort
        _ONNX_AVAILABLE = True
    except ImportError:
        _ONNX_AVAILABLE = False
    
    _ONNX_AVAILABLE
    
    # Cell 1 ? Mode switch
    MODE = "submit" 
    
    assert MODE in {"train", "submit"}
    
    print("MODE =", MODE)
    
    # Cell 2 ? Imports and run config
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    import gc
    import json
    import re
    import time
    import warnings
    from collections import defaultdict
    from pathlib import Path
    
    import numpy as np
    import pandas as pd
    import soundfile as sf
    import tensorflow as tf
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import roc_auc_score
    try:
        from lightgbm import LGBMClassifier
        _LGBM_AVAILABLE = True
    except ImportError:
        _LGBM_AVAILABLE = False
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    
    from tqdm.auto import tqdm
    
    warnings.filterwarnings("ignore")
    tf.experimental.numpy.experimental_enable_numpy_behavior()
    
    _WALL_START = time.time()
    
    BASE = Path("/kaggle/input/competitions/birdclef-2026")
    MODEL_DIR = Path("/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1")
    
    SR = 32000
    WINDOW_SEC = 5
    WINDOW_SAMPLES = SR * WINDOW_SEC
    FILE_SAMPLES = 60 * SR
    N_WINDOWS = 12
    
    DEVICE = torch.device("cpu")  # Competition constraint
    
    LOGS = {}  # Comprehensive logging dict
    
    CFG = {
        "mode": MODE,
        "verbose": MODE == "train",
    
        # expensive research blocks
        "run_oof_baseline": MODE == "train",
        "run_probe_check": False,
        "run_probe_grid": False,
    
        # inference
        "batch_files": 16,
        "proxy_reduce_grid": ["max", "mean"],
        "proxy_reduce": "max",
        "run_proxy_reduce_grid": False,
        "dryrun_n_files": 50 if MODE == "train" else 20,
    
        # cache behavior
        "require_full_cache_in_submit": False,
        "full_cache_input_dir": Path("/kaggle/input/perch-meta"),
        "full_cache_work_dir": Path("/kaggle/working/perch_cache"),
    
        # frozen baseline fusion params
        "best_fusion": {
            "lambda_event": 0.4,
            "lambda_texture": 1.0,
            "lambda_proxy_texture": 0.8,
            "smooth_texture": 0.35,
            "smooth_event": 0.15,
        },
    
        # V17: ProtoSSM v5 ? LARGER model
        "proto_ssm": {
            "d_model": 256,               # V17: increased from 128?256
            "d_state": 16,
            "n_ssm_layers": 3,            # V17: increased from 2?3
            "dropout": 0.15,
            "n_prototypes": 1,
            "n_sites": 20,
            "meta_dim": 16,
            "use_cross_attn": True,
            "cross_attn_heads": 4,
        },
    
        # ProtoSSM v5 training
        "proto_ssm_train": {
            "n_epochs": 60 if MODE == "train" else 40,   # ? was always 60,
            "lr": 1e-3,
            "weight_decay": 2e-3,
            "val_ratio": 0.15,
            "patience": 15  if MODE == "train" else 8,    # ? was always 15
            "pos_weight_cap": 30.0,
            "distill_weight": 0.1,
            "proto_margin": 0.1,
            "label_smoothing": 0.02,
            "oof_n_splits": 3,
            "mixup_alpha": 0.3,
            "focal_gamma": 2.0,
            "swa_start_frac": 0.7,
            "swa_lr": 5e-4,
        },
    
        # frozen probe params
        "frozen_best_probe": {
            "pca_dim": 64,
            "min_pos": 8,
            "C": 0.50,
            "alpha": 0.40,
        },
    
        # Residual SSM
        "residual_ssm": {
            "d_model": 64,
            "d_state": 8,
            "n_ssm_layers": 1,
            "dropout": 0.1,
            "correction_weight": 0.3,
            "n_epochs": 30,
            "lr": 1e-3,
            "patience": 8,
        },
    
        # Per-taxon temperature
        "temperature": {
            "aves": 1.10,
            "texture": 0.95,
        },
    
        # V17: Post-processing parameters
        "file_level_top_k": 2,
        "tta_shifts": [0, 1, -1],
        
        # V17 NEW: Rank-aware post-processing
        "rank_aware_scale": True,
        "rank_aware_power": 0.5,  # Power transform on file max
        
        # V17 NEW: Delta shift smoothing
        "delta_shift_alpha": 0.15,
        
        # V17 NEW: Per-class thresholds (grid search range)
        "threshold_grid": [0.3, 0.4, 0.5, 0.6, 0.7],
    
        "probe_backend": "mlp",
        "mlp_params": {
            "hidden_layer_sizes": (128,),
            "activation": "relu",
            "max_iter": 300,
            "early_stopping": True,
            "validation_fraction": 0.15,
            "n_iter_no_change": 15,
            "random_state": 42,
            "learning_rate_init": 0.001,
            "alpha": 0.01,
        },
    }
    
    CFG["full_cache_work_dir"].mkdir(parents=True, exist_ok=True)
    
    print("TensorFlow:", tf.__version__)
    print("PyTorch:", torch.__version__)
    print("Competition dir exists:", BASE.exists())
    print("Model dir exists:", MODEL_DIR.exists())
    print("V17 CFG: d_model=256, n_ssm_layers=3")
    print(json.dumps(
        {k: (str(v) if isinstance(v, Path) else v) for k, v in CFG.items()},
        indent=2
    ))
    
    # ?? V18 CFG UPGRADES ??????????????????????
    CFG["proto_ssm"] = {
        "d_model": 320, "d_state": 32, "n_ssm_layers": 4,
        "dropout": 0.12, "n_prototypes": 2, "n_sites": 20,
        "meta_dim": 24, "use_cross_attn": True, "cross_attn_heads": 8,
    }
    CFG["proto_ssm_train"] = {
        "n_epochs": 80, "lr": 8e-4, "weight_decay": 1e-3,
        "val_ratio": 0.15, "patience": 20, "pos_weight_cap": 25.0,
        "distill_weight": 0.15, "proto_margin": 0.15,
        "label_smoothing": 0.03, "oof_n_splits": 5,
        "mixup_alpha": 0.4, "focal_gamma": 2.5,
        "swa_start_frac": 0.65, "swa_lr": 4e-4,
        "use_cosine_restart": True, "restart_period": 20,
    }
    CFG["residual_ssm"] = {
        "d_model": 128, "d_state": 16, "n_ssm_layers": 2,
        "dropout": 0.1, "correction_weight": 0.35,
        "n_epochs": 40, "lr": 8e-4, "patience": 12,
    }
    CFG["best_fusion"]["lambda_event"]         = 0.45
    CFG["best_fusion"]["lambda_texture"]       = 1.1
    CFG["best_fusion"]["lambda_proxy_texture"] = 0.9
    CFG["threshold_grid"] = [0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70]
    CFG["tta_shifts"]        = [0, 1, -1, 2, -2]
    CFG["rank_aware_power"]  = 0.4
    CFG["delta_shift_alpha"] = 0.20
    CFG["mlp_params"] = {
        "hidden_layer_sizes": (256, 128), "activation": "relu",
        "max_iter": 500, "early_stopping": True,
        "validation_fraction": 0.15, "n_iter_no_change": 20,
        "random_state": 42, "learning_rate_init": 5e-4, "alpha": 0.005,
    }
    CFG["frozen_best_probe"] = {
        "pca_dim": 128, "min_pos": 5, "C": 0.75, "alpha": 0.45
    }
    print("? V18 CFG loaded")
    
    
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    
    def get_cosine_restart_scheduler(optimizer, restart_period=20):
        return CosineAnnealingWarmRestarts(
            optimizer, T_0=restart_period, T_mult=1, eta_min=1e-5
        )
    
    print("? Cosine Restart Scheduler defined")
    
    # ?? STEP 3: Mixup + CutMix Hybrid ?
    def mixup_cutmix(emb, logits, labels, alpha=0.4, cutmix_prob=0.3):
        B, T, D = emb.shape
        lam = np.random.beta(alpha, alpha)
        idx = torch.randperm(B)
    
        if np.random.rand() < cutmix_prob:
            # CutMix on time dimension
            cut_len = max(1, int(T * (1 - lam)))
            cut_start = np.random.randint(0, T - cut_len + 1)
            new_emb = emb.clone()
            new_emb[:, cut_start:cut_start+cut_len, :] = emb[idx, cut_start:cut_start+cut_len, :]
            new_logits = logits.clone()
            new_logits[:, cut_start:cut_start+cut_len, :] = logits[idx, cut_start:cut_start+cut_len, :]
            lam_actual = 1.0 - cut_len / T
            new_labels = lam_actual * labels + (1-lam_actual) * labels[idx]
        else:
            # Standard Mixup
            new_emb    = lam * emb    + (1-lam) * emb[idx]
            new_logits = lam * logits + (1-lam) * logits[idx]
            new_labels = lam * labels + (1-lam) * labels[idx]
    
        return new_emb, new_logits, new_labels
    
    print("? Mixup+CutMix defined")
    
    # ?? STEP 4: Species-Frequency Aware Focal Loss ??
    def build_class_freq_weights(Y_FULL, cap=10.0):
        pos_count = Y_FULL.sum(axis=0).astype(np.float32) + 1.0
        total     = Y_FULL.shape[0]
        freq      = pos_count / total
        weights   = 1.0 / (freq ** 0.5)
        weights   = np.clip(weights, 1.0, cap)
        weights   = weights / weights.mean()
        return torch.tensor(weights, dtype=torch.float32)
    
    def species_focal_loss(logits, targets, class_weights, 
                           gamma=2.5, label_smoothing=0.03):
        targets_smooth = targets * (1 - label_smoothing) + label_smoothing / 2.0
        bce    = F.binary_cross_entropy_with_logits(
                     logits, targets_smooth, reduction="none")
        pt     = torch.exp(-bce)
        focal  = ((1 - pt) ** gamma) * bce
        w      = class_weights.to(logits.device).unsqueeze(0)
        return (focal * w).mean()
    
    print("? Species Focal Loss defined")
    
    taxonomy = pd.read_csv(BASE / "taxonomy.csv")
    sample_sub = pd.read_csv(BASE / "sample_submission.csv")
    soundscape_labels = pd.read_csv(BASE / "train_soundscapes_labels.csv")
    
    PRIMARY_LABELS = sample_sub.columns[1:].tolist()
    N_CLASSES = len(PRIMARY_LABELS)
    
    taxonomy["primary_label"] = taxonomy["primary_label"].astype(str)
    soundscape_labels["primary_label"] = soundscape_labels["primary_label"].astype(str)
    
    def parse_soundscape_labels(x):
        if pd.isna(x):
            return []
        return [t.strip() for t in str(x).split(";") if t.strip()]
    
    FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")
    
    def parse_soundscape_filename(name):
        m = FNAME_RE.match(name)
        if not m:
            return {
                "file_id": None,
                "site": None,
                "date": pd.NaT,
                "time_utc": None,
                "hour_utc": -1,
                "month": -1,
            }
        file_id, site, ymd, hms = m.groups()
        dt = pd.to_datetime(ymd, format="%Y%m%d", errors="coerce")
        return {
            "file_id": file_id,
            "site": site,
            "date": dt,
            "time_utc": hms,
            "hour_utc": int(hms[:2]),
            "month": int(dt.month) if pd.notna(dt) else -1,
        }
    
    def union_labels(series):
        return sorted(set(lbl for x in series for lbl in parse_soundscape_labels(x)))
    
    # Deduplicate duplicated rows and aggregate labels per 5s window
    sc_clean = (
        soundscape_labels
        .groupby(["filename", "start", "end"])["primary_label"]
        .apply(union_labels)
        .reset_index(name="label_list")
    )
    
    sc_clean["start_sec"] = pd.to_timedelta(sc_clean["start"]).dt.total_seconds().astype(int)
    sc_clean["end_sec"] = pd.to_timedelta(sc_clean["end"]).dt.total_seconds().astype(int)
    sc_clean["row_id"] = sc_clean["filename"].str.replace(".ogg", "", regex=False) + "_" + sc_clean["end_sec"].astype(str)
    
    meta = sc_clean["filename"].apply(parse_soundscape_filename).apply(pd.Series)
    sc_clean = pd.concat([sc_clean, meta], axis=1)
    
    # Fully-labeled files
    windows_per_file = sc_clean.groupby("filename").size()
    full_files = sorted(windows_per_file[windows_per_file == N_WINDOWS].index.tolist())
    sc_clean["file_fully_labeled"] = sc_clean["filename"].isin(full_files)
    
    # Multi-hot label matrix aligned with sc_clean row order
    label_to_idx = {c: i for i, c in enumerate(PRIMARY_LABELS)}
    Y_SC = np.zeros((len(sc_clean), N_CLASSES), dtype=np.uint8)
    
    for i, labels in enumerate(sc_clean["label_list"]):
        idxs = [label_to_idx[lbl] for lbl in labels if lbl in label_to_idx]
        if idxs:
            Y_SC[i, idxs] = 1
    
    full_truth = (
        sc_clean[sc_clean["file_fully_labeled"]]
        .sort_values(["filename", "end_sec"])
        .reset_index(drop=False)
    )
    
    Y_FULL_TRUTH = Y_SC[full_truth["index"].to_numpy()]
    
    print("sc_clean:", sc_clean.shape)
    print("Y_SC:", Y_SC.shape, Y_SC.dtype)
    print("Full files:", len(full_files))
    print("Trusted full windows:", len(full_truth))
    print("Active classes in full windows:", int((Y_FULL_TRUTH.sum(axis=0) > 0).sum()))
    
    CLASS_WEIGHTS = build_class_freq_weights(Y_FULL_TRUTH)
    print("? Class weights built")
    
    # ?? STEP 5: Isotonic Calibration + Threshold Optimization ??
    from sklearn.isotonic import IsotonicRegression
    
    def calibrate_and_optimize_thresholds(oof_probs, Y_FULL, 
                                           threshold_grid, n_windows=12):
        n_samples, n_cls = oof_probs.shape
        thresholds = np.full(n_cls, 0.5, dtype=np.float32)
        n_files  = n_samples // n_windows
        file_oof = oof_probs.reshape(n_files, n_windows, n_cls).max(axis=1)
        file_y   = Y_FULL.reshape(n_files, n_windows, n_cls).max(axis=1)
    
        for c in range(n_cls):
            y_true, y_prob = file_y[:, c], file_oof[:, c]
            if y_true.sum() < 3:
                continue
            try:
                ir = IsotonicRegression(out_of_bounds="clip")
                ir.fit(y_prob, y_true)
                y_cal = ir.transform(y_prob)
            except:
                y_cal = y_prob
    
            best_f1, best_t = 0.0, 0.5
            for t in threshold_grid:
                pred = (y_cal >= t).astype(int)
                tp = ((pred==1)&(y_true==1)).sum()
                fp = ((pred==1)&(y_true==0)).sum()
                fn = ((pred==0)&(y_true==1)).sum()
                prec = tp/(tp+fp+1e-8)
                rec  = tp/(tp+fn+1e-8)
                f1   = 2*prec*rec/(prec+rec+1e-8)
                if f1 > best_f1:
                    best_f1, best_t = f1, t
            thresholds[c] = best_t
    
        print(f"Mean threshold: {thresholds.mean():.3f}")
        print(f"Range: [{thresholds.min():.2f}, {thresholds.max():.2f}]")
        return thresholds
    
    print("? Calibration + Threshold function defined")
    
    # ?? STEP 6: Ensemble Weight Sweep ??
    def sweep_ensemble_weight(oof_proto, oof_mlp, Y_FULL, 
                              n_windows=12,
                              candidates=np.arange(0.3, 0.8, 0.05)):
        n_files = oof_proto.shape[0] // n_windows
        file_y  = Y_FULL.reshape(n_files, n_windows, -1).max(axis=1)
        best_auc, best_w = 0.0, 0.6
    
        for w in candidates:
            blended   = w * oof_proto + (1-w) * oof_mlp
            file_pred = blended.reshape(n_files, n_windows, -1).max(axis=1)
            try:
                auc = macro_auc_skip_empty(file_y, file_pred)
            except:
                continue
            if auc > best_auc:
                best_auc, best_w = auc, w
    
        print(f"Best ensemble weight (proto): {best_w:.2f}")
        print(f"Best AUC: {best_auc:.5f}")
        return best_w
    
    print("? Ensemble Weight Sweep defined")
    
    # Cell 3 ? Load Perch, mapping, and selective frog proxies
    BEST = CFG["best_fusion"]
    
    # ? ONNX Load
    ONNX_PERCH_PATH = Path("/kaggle/input/datasets/hellodave2035/birdclef-perch-models/perch_v2.onnx")
    USE_ONNX_PERCH = _ONNX_AVAILABLE and ONNX_PERCH_PATH.exists()
    
    if USE_ONNX_PERCH:
        print(f"Using ONNX Perch (150x faster)")
        _so = ort.SessionOptions()
        _so.intra_op_num_threads = 4
        ONNX_SESSION = ort.InferenceSession(str(ONNX_PERCH_PATH), sess_options=_so, providers=["CPUExecutionProvider"])
        ONNX_INPUT_NAME = ONNX_SESSION.get_inputs()[0].name
        ONNX_OUTPUT_MAP = {o.name: i for i, o in enumerate(ONNX_SESSION.get_outputs())}
    
    birdclassifier = tf.saved_model.load(str(MODEL_DIR))
    infer_fn = birdclassifier.signatures["serving_default"]
    
    bc_labels = (
        pd.read_csv(MODEL_DIR / "assets" / "labels.csv")
        .reset_index()
        .rename(columns={"index": "bc_index", "inat2024_fsd50k": "scientific_name"})
    )
    
    NO_LABEL_INDEX = len(bc_labels)
    
    MANUAL_SCIENTIFIC_NAME_MAP = {
        # Optional future synonym fixes (add manual name corrections here)
    }
    
    taxonomy = taxonomy.copy()
    taxonomy["scientific_name_lookup"] = taxonomy["scientific_name"].replace(MANUAL_SCIENTIFIC_NAME_MAP)
    
    bc_lookup = bc_labels.rename(columns={"scientific_name": "scientific_name_lookup"})
    
    mapping = taxonomy.merge(
        bc_lookup[["scientific_name_lookup", "bc_index"]],
        on="scientific_name_lookup",
        how="left"
    )
    
    mapping["bc_index"] = mapping["bc_index"].fillna(NO_LABEL_INDEX).astype(int)
    
    label_to_bc_index = mapping.set_index("primary_label")["bc_index"]
    BC_INDICES = np.array([int(label_to_bc_index.loc[c]) for c in PRIMARY_LABELS], dtype=np.int32)
    
    MAPPED_MASK = BC_INDICES != NO_LABEL_INDEX
    MAPPED_POS = np.where(MAPPED_MASK)[0].astype(np.int32)
    UNMAPPED_POS = np.where(~MAPPED_MASK)[0].astype(np.int32)
    MAPPED_BC_INDICES = BC_INDICES[MAPPED_MASK].astype(np.int32)
    
    CLASS_NAME_MAP = taxonomy.set_index("primary_label")["class_name"].to_dict()
    TEXTURE_TAXA = {"Amphibia", "Insecta"}
    
    ACTIVE_CLASSES = [PRIMARY_LABELS[i] for i in np.where(Y_SC.sum(axis=0) > 0)[0]]
    
    idx_active_texture = np.array(
        [label_to_idx[c] for c in ACTIVE_CLASSES if CLASS_NAME_MAP.get(c) in TEXTURE_TAXA],
        dtype=np.int32
    )
    idx_active_event = np.array(
        [label_to_idx[c] for c in ACTIVE_CLASSES if CLASS_NAME_MAP.get(c) not in TEXTURE_TAXA],
        dtype=np.int32
    )
    
    idx_mapped_active_texture = idx_active_texture[MAPPED_MASK[idx_active_texture]]
    idx_mapped_active_event = idx_active_event[MAPPED_MASK[idx_active_event]]
    
    idx_unmapped_active_texture = idx_active_texture[~MAPPED_MASK[idx_active_texture]]
    idx_unmapped_active_event = idx_active_event[~MAPPED_MASK[idx_active_event]]
    
    idx_unmapped_inactive = np.array(
        [i for i in UNMAPPED_POS if PRIMARY_LABELS[i] not in ACTIVE_CLASSES],
        dtype=np.int32
    )
    
    # Build automatic genus proxies for unmapped non-sonotypes
    unmapped_df = mapping[mapping["bc_index"] == NO_LABEL_INDEX].copy()
    unmapped_non_sonotype = unmapped_df[
        ~unmapped_df["primary_label"].astype(str).str.contains("son", na=False)
    ].copy()
    
    def get_genus_hits(scientific_name):
        genus = str(scientific_name).split()[0]
        hits = bc_labels[
            bc_labels["scientific_name"].astype(str).str.match(rf"^{re.escape(genus)}\s", na=False)
        ].copy()
        return genus, hits
    
    proxy_map = {}
    for _, row in unmapped_non_sonotype.iterrows():
        target = row["primary_label"]
        sci = row["scientific_name"]
        genus, hits = get_genus_hits(sci)
        if len(hits) > 0:
            proxy_map[target] = {
                "target_scientific_name": sci,
                "genus": genus,
                "bc_indices": hits["bc_index"].astype(int).tolist(),
                "proxy_scientific_names": hits["scientific_name"].tolist(),
            }
    
    # Enable genus proxies for Amphibia, Insecta, and Aves (unmapped species)
    PROXY_TAXA = {"Amphibia", "Insecta", "Aves"}
    SELECTED_PROXY_TARGETS = sorted([
        t for t in proxy_map.keys()
        if CLASS_NAME_MAP.get(t) in PROXY_TAXA
    ])
    print(f"Proxy targets by class: { {cls: sum(1 for t in SELECTED_PROXY_TARGETS if CLASS_NAME_MAP.get(t)==cls) for cls in PROXY_TAXA} }")
    
    selected_proxy_pos = np.array([label_to_idx[c] for c in SELECTED_PROXY_TARGETS], dtype=np.int32)
    
    selected_proxy_pos_to_bc = {
        label_to_idx[target]: np.array(proxy_map[target]["bc_indices"], dtype=np.int32)
        for target in SELECTED_PROXY_TARGETS
    }
    
    idx_selected_proxy_active_texture = np.intersect1d(selected_proxy_pos, idx_active_texture)
    idx_selected_prioronly_active_texture = np.setdiff1d(idx_unmapped_active_texture, selected_proxy_pos)
    idx_selected_prioronly_active_event = np.setdiff1d(idx_unmapped_active_event, selected_proxy_pos)
    
    print(f"Mapped classes: {MAPPED_MASK.sum()} / {N_CLASSES}")
    print(f"Unmapped classes: {(~MAPPED_MASK).sum()}")
    print("Selected frog proxy targets:", SELECTED_PROXY_TARGETS)
    print("Active texture classes:", len(idx_active_texture))
    print("Selected proxy active texture:", len(idx_selected_proxy_active_texture))
    print("Prior-only active texture:", len(idx_selected_prioronly_active_texture))
    print("Prior-only active event:", len(idx_selected_prioronly_active_event))
    
    # Cell 4 ? Metrics and helper utilities
    def macro_auc_skip_empty(y_true, y_score):
        keep = y_true.sum(axis=0) > 0
        return roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro")
    
    def smooth_cols_fixed12(scores, cols, alpha=0.35):
        if alpha <= 0 or len(cols) == 0:
            return scores.copy()
    
        s = scores.copy()
        assert len(s) % N_WINDOWS == 0, "Expected full-file blocks of 12 windows"
        view = s.reshape(-1, N_WINDOWS, s.shape[1])
    
        x = view[:, :, cols]
        prev_x = np.concatenate([x[:, :1, :], x[:, :-1, :]], axis=1)
        next_x = np.concatenate([x[:, 1:, :], x[:, -1:, :]], axis=1)
    
        view[:, :, cols] = (1.0 - alpha) * x + 0.5 * alpha * (prev_x + next_x)
        return s
    
    def smooth_events_fixed12(scores, cols, alpha=0.15):
        """Soft max-pool context for event birds (Aves).
        Uses local_max instead of average neighbor, preserving transient call detection."""
        if alpha <= 0 or len(cols) == 0:
            return scores.copy()
        s = scores.copy()
        assert len(s) % N_WINDOWS == 0
        view = s.reshape(-1, N_WINDOWS, s.shape[1])
        x = view[:, :, cols]
        prev_x = np.concatenate([x[:, :1, :], x[:, :-1, :]], axis=1)
        next_x = np.concatenate([x[:, 1:, :], x[:, -1:, :]], axis=1)
        local_max = np.maximum(x, np.maximum(prev_x, next_x))
        view[:, :, cols] = (1.0 - alpha) * x + alpha * local_max
        return s
    
    def seq_features_1d(v):
        """
        v: shape (n_rows,), ordered as full-file blocks of 12 windows
        Extended: tambah std_v untuk capture variance temporal dalam file
        """
        assert len(v) % N_WINDOWS == 0, "Expected full-file blocks of 12 windows"
        x = v.reshape(-1, N_WINDOWS)
    
        prev_v = np.concatenate([x[:, :1], x[:, :-1]], axis=1).reshape(-1)
        next_v = np.concatenate([x[:, 1:], x[:, -1:]], axis=1).reshape(-1)
        mean_v = np.repeat(x.mean(axis=1), N_WINDOWS)
        max_v  = np.repeat(x.max(axis=1),  N_WINDOWS)
        std_v  = np.repeat(x.std(axis=1),  N_WINDOWS)
    
        return prev_v, next_v, mean_v, max_v, std_v
    
    # V16/V17 NEW: Focal loss, file-level scaling, TTA, rank-aware, delta shift, per-class thresholds
    
    def focal_bce_with_logits(logits, targets, gamma=2.0, pos_weight=None, reduction="mean"):
        """Focal loss for multi-label classification.
        Reduces contribution of easy examples, focuses on hard ones."""
        if pos_weight is not None:
            bce = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pos_weight, reduction="none"
            )
        else:
            bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        
        p = torch.sigmoid(logits)
        pt = targets * p + (1 - targets) * (1 - p)
        focal_weight = (1 - pt) ** gamma
        loss = focal_weight * bce
        
        if reduction == "mean":
            return loss.mean()
        return loss
    
    
    def file_level_confidence_scale(preds, n_windows=12, top_k=2):
        """Rank 1/2 technique: scale each window's predictions by the file's top-K mean confidence."""
        N, C = preds.shape
        assert N % n_windows == 0
        view = preds.reshape(-1, n_windows, C)
        sorted_view = np.sort(view, axis=1)
        top_k_mean = sorted_view[:, -top_k:, :].mean(axis=1, keepdims=True)
        scaled = view * top_k_mean
        return scaled.reshape(N, C)
    
    
    # def temporal_shift_tta(emb_files, logits_files, model, site_ids, hours, shifts=[0, 1, -1]):
    #     """TTA by circular-shifting the 12-window embedding sequence."""
    #     all_preds = []
    #     model.eval()
        
    #     for shift in shifts:
    #         if shift == 0:
    #             e = emb_files
    #             l = logits_files
    #         else:
    #             e = np.roll(emb_files, shift, axis=1)
    #             l = np.roll(logits_files, shift, axis=1)
            
    #         with torch.no_grad():
    #             out, _, _ = model(
    #                 torch.tensor(e, dtype=torch.float32),
    #                 torch.tensor(l, dtype=torch.float32),
    #                 site_ids=torch.tensor(site_ids, dtype=torch.long),
    #                 hours=torch.tensor(hours, dtype=torch.long),
    #             )
    #             pred = out.numpy()
            
    #         if shift != 0:
    #             pred = np.roll(pred, -shift, axis=1)
            
    #         all_preds.append(pred)
        
    #     return np.mean(all_preds, axis=0)
    def temporal_shift_tta(emb_files, logits_files, model, site_ids, hours, shifts=[0, 1, -1], max_batch_size=512):
        """
        TTA by circular-shifting the 12-window embedding sequence.
        Batched and optimized for faster single-pass inference.
        """
        n_files = emb_files.shape[0]
        n_shifts = len(shifts)
        
        if n_shifts == 0:
            return np.zeros((n_files, emb_files.shape[1], logits_files.shape[2]), dtype=np.float32)
    
        e_list, l_list = [], []
        for shift in shifts:
            if shift == 0:
                e_list.append(emb_files)
                l_list.append(logits_files)
            else:
                e_list.append(np.roll(emb_files, shift, axis=1))
                l_list.append(np.roll(logits_files, shift, axis=1))
                
        e_batch = np.concatenate(e_list, axis=0)
        l_batch = np.concatenate(l_list, axis=0)
        
        site_batch = np.tile(site_ids, n_shifts)
        hour_batch = np.tile(hours, n_shifts)
        
        model.eval()
        pred_batch_list = []
        
        with torch.no_grad():
            total_samples = e_batch.shape[0]
            for start_idx in range(0, total_samples, max_batch_size):
                end_idx = min(start_idx + max_batch_size, total_samples)
                
                out, _, _ = model(
                    torch.tensor(e_batch[start_idx:end_idx], dtype=torch.float32),
                    torch.tensor(l_batch[start_idx:end_idx], dtype=torch.float32),
                    site_ids=torch.tensor(site_batch[start_idx:end_idx], dtype=torch.long),
                    hours=torch.tensor(hour_batch[start_idx:end_idx], dtype=torch.long),
                )
                pred_batch_list.append(out.numpy())
                
        pred_batch = np.concatenate(pred_batch_list, axis=0)
        pred_batch = pred_batch.reshape(n_shifts, n_files, pred_batch.shape[1], pred_batch.shape[2])
        
        all_preds = []
        for i, shift in enumerate(shifts):
            pred_i = pred_batch[i]
            if shift != 0:
                pred_i = np.roll(pred_i, -shift, axis=1)
            all_preds.append(pred_i)
        return np.mean(all_preds, axis=0)
    
    
    
    # V17: Post-processing utilities
    
    def rank_aware_scaling(scores, n_windows=12, power=0.5):
        """V17: 2025 Rank 3 technique. Scale each window by (file_max)^power.
        Suppresses predictions in uncertain files, boosts confident files."""
        N, C = scores.shape
        assert N % n_windows == 0
        n_files = N // n_windows
        
        view = scores.reshape(n_files, n_windows, C)
        file_max = view.max(axis=1, keepdims=True)  # (F, 1, C)
        
        # Apply power transform to file max
        scale = np.power(file_max, power)
        
        # Scale each window
        scaled = view * scale
        return scaled.reshape(N, C)
    
    
    def delta_shift_smooth(scores, n_windows=12, alpha=0.15):
        """V17: 2025 Rank 1 technique. Temporal smoothing across windows.
        new[t] = (1-alpha)*old[t] + 0.5*alpha*(old[t-1] + old[t+1])"""
        N, C = scores.shape
        assert N % n_windows == 0
        n_files = N // n_windows
        
        view = scores.reshape(n_files, n_windows, C)
        
        # Create shifted versions
        prev_view = np.concatenate([view[:, :1, :], view[:, :-1, :]], axis=1)
        next_view = np.concatenate([view[:, 1:, :], view[:, -1:, :]], axis=1)
        
        # Delta shift smoothing
        smoothed = (1 - alpha) * view + 0.5 * alpha * (prev_view + next_view)
        
        return smoothed.reshape(N, C)
    
    
    def optimize_per_class_thresholds(oof_scores, y_true, n_windows=12, thresholds=[0.3, 0.4, 0.5, 0.6, 0.7]):
        """V17: Find optimal decision threshold per class from OOF predictions.
        Optimizes F1-like metric (precision-recall balance) for each species."""
        n_classes = oof_scores.shape[1]
        best_thresholds = np.zeros(n_classes)
        best_scores = np.zeros(n_classes)
        
        for c in range(n_classes):
            y_c = y_true[:, c]
            scores_c = oof_scores[:, c]
            
            # Skip classes with no positive samples
            if y_c.sum() == 0:
                best_thresholds[c] = 0.5
                continue
                
            # Find best threshold
            best_f1 = 0
            best_t = 0.5
            
            for t in thresholds:
                pred_c = (scores_c > t).astype(int)
                tp = ((pred_c == 1) & (y_c == 1)).sum()
                fp = ((pred_c == 1) & (y_c == 0)).sum()
                fn = ((pred_c == 0) & (y_c == 1)).sum()
                
                if tp + fp == 0 or tp + fn == 0:
                    continue
                    
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = t
            
            best_thresholds[c] = best_t
            best_scores[c] = best_f1
        
        return best_thresholds, best_scores
    
    
    def apply_per_class_thresholds(scores, thresholds, n_windows=12):
        """V17: Apply per-class thresholds to convert scores to binary predictions."""
        N, C = scores.shape
        assert C == len(thresholds)
        
        # For competition, we submit probabilities but threshold for metrics
        # Apply threshold as a scaling factor that sharpens confident predictions
        scaled = np.copy(scores)
        
        for c in range(C):
            t = thresholds[c]
            # Sharpen: push above-threshold scores higher, below-threshold lower
            mask_above = scores[:, c] > t
            scaled[mask_above, c] = 0.5 + 0.5 * (scores[mask_above, c] - t) / (1 - t + 1e-8)
            scaled[~mask_above, c] = 0.5 * scores[~mask_above, c] / (t + 1e-8)
        
        return np.clip(scaled, 0, 1)
    
    
    print("V17 utilities defined: focal_bce_with_logits, file_level_confidence_scale, temporal_shift_tta,")
    print("  rank_aware_scaling, delta_shift_smooth, optimize_per_class_thresholds, apply_per_class_thresholds")
    
    # Cell 5 ? Perch inference with embeddings + selective proxies
    def read_soundscape_60s(path):
        y, sr = sf.read(path, dtype="float32", always_2d=False)
        if y.ndim == 2:
            y = y.mean(axis=1)
        if sr != SR:
            raise ValueError(f"Unexpected sample rate {sr} in {path}; expected {SR}")
        if len(y) < FILE_SAMPLES:
            y = np.pad(y, (0, FILE_SAMPLES - len(y)))
        elif len(y) > FILE_SAMPLES:
            y = y[:FILE_SAMPLES]
        return y
    
    # def infer_perch_with_embeddings(paths, batch_files=16, verbose=True, proxy_reduce="max"):
    #     paths = [Path(p) for p in paths]
    #     n_files = len(paths)
    #     n_rows = n_files * N_WINDOWS
    
    #     row_ids = np.empty(n_rows, dtype=object)
    #     filenames = np.empty(n_rows, dtype=object)
    #     sites = np.empty(n_rows, dtype=object)
    #     hours = np.empty(n_rows, dtype=np.int16)
    
    #     scores = np.zeros((n_rows, N_CLASSES), dtype=np.float32)
    #     embeddings = np.zeros((n_rows, 1536), dtype=np.float32)
    
    #     write_row = 0
    #     iterator = range(0, n_files, batch_files)
    #     if verbose:
    #         iterator = tqdm(iterator, total=(n_files + batch_files - 1) // batch_files, desc="Perch batches")
    
    #     for start in iterator:
    #         batch_paths = paths[start:start + batch_files]
    #         batch_n = len(batch_paths)
    
    #         x = np.empty((batch_n * N_WINDOWS, WINDOW_SAMPLES), dtype=np.float32)
    #         batch_row_start = write_row
    #         x_pos = 0
    
    #         for path in batch_paths:
    #             y = read_soundscape_60s(path)
    #             x[x_pos:x_pos + N_WINDOWS] = y.reshape(N_WINDOWS, WINDOW_SAMPLES)
    
    #             meta = parse_soundscape_filename(path.name)
    #             stem = path.stem
    
    #             row_ids[write_row:write_row + N_WINDOWS] = [f"{stem}_{t}" for t in range(5, 65, 5)]
    #             filenames[write_row:write_row + N_WINDOWS] = path.name
    #             sites[write_row:write_row + N_WINDOWS] = meta["site"]
    #             hours[write_row:write_row + N_WINDOWS] = int(meta["hour_utc"])
    
    #             x_pos += N_WINDOWS
    #             write_row += N_WINDOWS
    
    #         outputs = infer_fn(inputs=tf.convert_to_tensor(x))
    #         logits = outputs["label"].numpy().astype(np.float32, copy=False)
    #         emb = outputs["embedding"].numpy().astype(np.float32, copy=False)
    
    #         scores[batch_row_start:write_row, MAPPED_POS] = logits[:, MAPPED_BC_INDICES]
    #         embeddings[batch_row_start:write_row] = emb
    
    #         # Selected frog proxies
    #         for pos, bc_idx_arr in selected_proxy_pos_to_bc.items():
    #             sub = logits[:, bc_idx_arr]
    #             if proxy_reduce == "max":
    #                 proxy_score = sub.max(axis=1)
    #             elif proxy_reduce == "mean":
    #                 proxy_score = sub.mean(axis=1)
    #             else:
    #                 raise ValueError("proxy_reduce must be 'max' or 'mean'")
    #             scores[batch_row_start:write_row, pos] = proxy_score.astype(np.float32)
    
    #         del x, outputs, logits, emb
    #         gc.collect()
    
    #     meta_df = pd.DataFrame({
    #         "row_id": row_ids,
    #         "filename": filenames,
    #         "site": sites,
    #         "hour_utc": hours,
    #     })
    
    #     return meta_df, scores, embeddings
    
    
    # ---------------------------------------- #
    # 2026/04/02 Update Process 
    # ---------------------------------------- #
    import concurrent.futures
    def infer_perch_with_embeddings(paths, batch_files=16, verbose=True, proxy_reduce="max"):
        paths = [Path(p) for p in paths]
        n_files = len(paths)
        n_rows = n_files * N_WINDOWS
    
        row_ids = np.empty(n_rows, dtype=object)
        filenames = np.empty(n_rows, dtype=object)
        sites = np.empty(n_rows, dtype=object)
        hours = np.empty(n_rows, dtype=np.int16)
    
        scores = np.zeros((n_rows, N_CLASSES), dtype=np.float32)
        embeddings = np.zeros((n_rows, 1536), dtype=np.float32)
    
        write_row = 0
        iterator = range(0, n_files, batch_files)
        if verbose:
            iterator = tqdm(iterator, total=(n_files + batch_files - 1) // batch_files, desc="Perch batches")
    
        # ?????ThreadPoolExecutor??
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as io_executor:
            
            # 1. Reserve the loading of the first batch in the background (start prefetching).
            next_paths = paths[0:batch_files]
            future_audio = [io_executor.submit(read_soundscape_60s, p) for p in next_paths]
    
            for start in iterator:
                batch_paths = next_paths
                batch_n = len(batch_paths)
    
                # --- Phase A: Receiving audio data ---
                batch_audio = [f.result() for f in future_audio]
    
                # --- Phase B: Immediately begin loading the "next batch". ---
                next_start = start + batch_files
                if next_start < n_files:
                    next_paths = paths[next_start:next_start + batch_files]
                    future_audio = [io_executor.submit(read_soundscape_60s, p) for p in next_paths]
    
                # --- Phase C: Data Formatting ---
                x = np.empty((batch_n * N_WINDOWS, WINDOW_SAMPLES), dtype=np.float32)
                batch_row_start = write_row
                x_pos = 0
    
                for i, path in enumerate(batch_paths):
                    y = batch_audio[i]
                    x[x_pos:x_pos + N_WINDOWS] = y.reshape(N_WINDOWS, WINDOW_SAMPLES)
    
                    meta = parse_soundscape_filename(path.name)
                    stem = path.stem
    
                    row_ids[write_row:write_row + N_WINDOWS] = [f"{stem}_{t}" for t in range(5, 65, 5)]
                    filenames[write_row:write_row + N_WINDOWS] = path.name
                    sites[write_row:write_row + N_WINDOWS] = meta["site"]
                    hours[write_row:write_row + N_WINDOWS] = int(meta["hour_utc"])
    
                    x_pos += N_WINDOWS
                    write_row += N_WINDOWS
    
                # --- Phase D: Heavy Inference (CPU Compute Bound) ---
                if USE_ONNX_PERCH:
                    onnx_outs = ONNX_SESSION.run(None, {ONNX_INPUT_NAME: x})
                    logits = onnx_outs[ONNX_OUTPUT_MAP["label"]].astype(np.float32, copy=False)
                    emb = onnx_outs[ONNX_OUTPUT_MAP["embedding"]].astype(np.float32, copy=False)
                else:
                    outputs = infer_fn(inputs=tf.convert_to_tensor(x))
                    logits = outputs["label"].numpy().astype(np.float32, copy=False)
                    emb = outputs["embedding"].numpy().astype(np.float32, copy=False)
    
                scores[batch_row_start:write_row, MAPPED_POS] = logits[:, MAPPED_BC_INDICES]
                embeddings[batch_row_start:write_row] = emb
    
                # Selected frog proxies
                for pos, bc_idx_arr in selected_proxy_pos_to_bc.items():
                    sub = logits[:, bc_idx_arr]
                    if proxy_reduce == "max":
                        proxy_score = sub.max(axis=1)
                    elif proxy_reduce == "mean":
                        proxy_score = sub.mean(axis=1)
                    else:
                        raise ValueError("proxy_reduce must be 'max' or 'mean'")
                    scores[batch_row_start:write_row, pos] = proxy_score.astype(np.float32)
    
                # Memory leak (OOM) prevention
                if USE_ONNX_PERCH:
                    del x, onnx_outs, logits, emb, batch_audio
                else:
                    del x, outputs, logits, emb, batch_audio
                gc.collect()
    
        meta_df = pd.DataFrame({
            "row_id": row_ids,
            "filename": filenames,
            "site": sites,
            "hour_utc": hours,
        })
    
        return meta_df, scores, embeddings
    
    # Cell 6 ? Load or compute full-file Perch cache
    def resolve_full_cache_paths():
        candidates = []
    
        # Working dir cache
        candidates.append((
            CFG["full_cache_work_dir"] / "full_perch_meta.parquet",
            CFG["full_cache_work_dir"] / "full_perch_arrays.npz"
        ))
    
        # Legacy working paths
        candidates.append((
            Path("/kaggle/working/full_perch_meta.parquet"),
            Path("/kaggle/working/full_perch_arrays.npz")
        ))
    
        # Attached input dataset
        if CFG["full_cache_input_dir"].exists():
            candidates.append((
                CFG["full_cache_input_dir"] / "full_perch_meta.parquet",
                CFG["full_cache_input_dir"] / "full_perch_arrays.npz"
            ))
    
        for meta_path, npz_path in candidates:
            if meta_path.exists() and npz_path.exists():
                return meta_path, npz_path
    
        return None, None
    
    cache_meta, cache_npz = resolve_full_cache_paths()
    
    if cache_meta is not None and cache_npz is not None:
        print("Loading cached full-file Perch outputs from:")
        print("  ", cache_meta)
        print("  ", cache_npz)
    
        meta_full = pd.read_parquet(cache_meta)
        arr = np.load(cache_npz)
        scores_full_raw = arr["scores_full_raw"].astype(np.float32)
        emb_full = arr["emb_full"].astype(np.float32)
    
    else:
        if CFG["mode"] == "submit" and CFG["require_full_cache_in_submit"]:
            raise FileNotFoundError(
                "Submit mode requires cached full-file Perch outputs. "
                "Attach the cache dataset or place full_perch_meta.parquet/full_perch_arrays.npz in working dir."
            )
    
        print("No cache found. Running Perch on trusted full files...")
        full_paths = [BASE / "train_soundscapes" / fn for fn in full_files]
    
        # Use CFG["proxy_reduce"] for consistency with grid search
        meta_full, scores_full_raw, emb_full = infer_perch_with_embeddings(
            full_paths,
            batch_files=CFG["batch_files"],
            verbose=CFG["verbose"],
            proxy_reduce=CFG["proxy_reduce"],
        )
    
        out_meta = CFG["full_cache_work_dir"] / "full_perch_meta.parquet"
        out_npz = CFG["full_cache_work_dir"] / "full_perch_arrays.npz"
    
        meta_full.to_parquet(out_meta, index=False)
        np.savez_compressed(
            out_npz,
            scores_full_raw=scores_full_raw,
            emb_full=emb_full,
        )
    
        print("Saved cache to:")
        print("  ", out_meta)
        print("  ", out_npz)
    
    # Align truth to cached order
    full_truth_aligned = full_truth.set_index("row_id").loc[meta_full["row_id"]].reset_index()
    Y_FULL = Y_SC[full_truth_aligned["index"].to_numpy()]
    
    assert np.all(full_truth_aligned["filename"].values == meta_full["filename"].values)
    assert np.all(full_truth_aligned["row_id"].values == meta_full["row_id"].values)
    
    print("meta_full:", meta_full.shape)
    print("scores_full_raw:", scores_full_raw.shape, scores_full_raw.dtype)
    print("emb_full:", emb_full.shape, emb_full.dtype)
    print("Y_FULL:", Y_FULL.shape, Y_FULL.dtype)
    
    # [MODIFIED - Opsi 3] Grid search proxy_reduce: evaluasi "max" vs "mean" via OOF AUC
    # Dilakukan hanya saat train mode; hasilnya di-freeze ke CFG["proxy_reduce"] untuk submit
    PROXY_REDUCE_CACHE = CFG["full_cache_work_dir"] / "proxy_reduce_grid.json"
    
    if CFG.get("run_proxy_reduce_grid", False):
        print("\n[Opsi 3] Running proxy_reduce grid search: max vs mean...")
        proxy_reduce_results = {}
    
        for pr in CFG["proxy_reduce_grid"]:
            full_paths = [BASE / "train_soundscapes" / fn for fn in full_files]
            _meta, _scores, _emb = infer_perch_with_embeddings(
                full_paths,
                batch_files=CFG["batch_files"],
                verbose=False,
                proxy_reduce=pr,
            )
    
            # OOF baseline AUC untuk proxy_reduce ini (tanpa probe)
            _oof_b, _oof_p, _ = build_oof_base_prior(
                scores_full_raw=_scores,
                meta_full=_meta,
                sc_clean=sc_clean,
                Y_SC=Y_SC,
                n_splits=5,
                verbose=False,
            )
            auc = macro_auc_skip_empty(Y_FULL, _oof_b)
            proxy_reduce_results[pr] = float(auc)
            print(f"  proxy_reduce={pr!r:6s} ? OOF baseline AUC = {auc:.6f}")
    
        best_pr = max(proxy_reduce_results, key=proxy_reduce_results.get)
        CFG["proxy_reduce"] = best_pr
        print(f"\n  Best proxy_reduce = {best_pr!r} (AUC={proxy_reduce_results[best_pr]:.6f})")
    
        PROXY_REDUCE_CACHE.write_text(json.dumps({
            "results": proxy_reduce_results,
            "best_proxy_reduce": best_pr,
        }, indent=2))
        print("  Saved to:", PROXY_REDUCE_CACHE)
    
    elif PROXY_REDUCE_CACHE.exists():
        _pr_data = json.loads(PROXY_REDUCE_CACHE.read_text())
        CFG["proxy_reduce"] = _pr_data["best_proxy_reduce"]
        print(f"[Opsi 3] Loaded proxy_reduce from cache: {CFG['proxy_reduce']!r}")
        print("  Grid results:", _pr_data["results"])
    
    else:
        print(f"[Opsi 3] Using default proxy_reduce={CFG['proxy_reduce']!r} (submit mode or no cache)")
    
    # Cell 7 ? Fold-safe metadata prior tables
    def fit_prior_tables(prior_df, Y_prior):
        prior_df = prior_df.reset_index(drop=True)
    
        global_p = Y_prior.mean(axis=0).astype(np.float32)
    
        # Site
        site_keys = sorted(prior_df["site"].dropna().astype(str).unique().tolist())
        site_to_i = {k: i for i, k in enumerate(site_keys)}
        site_n = np.zeros(len(site_keys), dtype=np.float32)
        site_p = np.zeros((len(site_keys), Y_prior.shape[1]), dtype=np.float32)
    
        for s in site_keys:
            i = site_to_i[s]
            mask = prior_df["site"].astype(str).values == s
            site_n[i] = mask.sum()
            site_p[i] = Y_prior[mask].mean(axis=0)
    
        # Hour
        hour_keys = sorted(prior_df["hour_utc"].dropna().astype(int).unique().tolist())
        hour_to_i = {h: i for i, h in enumerate(hour_keys)}
        hour_n = np.zeros(len(hour_keys), dtype=np.float32)
        hour_p = np.zeros((len(hour_keys), Y_prior.shape[1]), dtype=np.float32)
    
        for h in hour_keys:
            i = hour_to_i[h]
            mask = prior_df["hour_utc"].astype(int).values == h
            hour_n[i] = mask.sum()
            hour_p[i] = Y_prior[mask].mean(axis=0)
    
        # Site-hour
        sh_to_i = {}
        sh_n_list = []
        sh_p_list = []
    
        for (s, h), idx in prior_df.groupby(["site", "hour_utc"]).groups.items():
            sh_to_i[(str(s), int(h))] = len(sh_n_list)
            idx = np.array(list(idx))
            sh_n_list.append(len(idx))
            sh_p_list.append(Y_prior[idx].mean(axis=0))
    
        sh_n = np.array(sh_n_list, dtype=np.float32)
        sh_p = np.stack(sh_p_list).astype(np.float32) if len(sh_p_list) else np.zeros((0, Y_prior.shape[1]), dtype=np.float32)
    
        return {
            "global_p": global_p,
            "site_to_i": site_to_i,
            "site_n": site_n,
            "site_p": site_p,
            "hour_to_i": hour_to_i,
            "hour_n": hour_n,
            "hour_p": hour_p,
            "sh_to_i": sh_to_i,
            "sh_n": sh_n,
            "sh_p": sh_p,
        }
    
    def prior_logits_from_tables(sites, hours, tables, eps=1e-4):
        n = len(sites)
        p = np.repeat(tables["global_p"][None, :], n, axis=0).astype(np.float32, copy=True)
    
        site_idx = np.fromiter(
            (tables["site_to_i"].get(str(s), -1) for s in sites),
            dtype=np.int32,
            count=n
        )
        hour_idx = np.fromiter(
            (tables["hour_to_i"].get(int(h), -1) if int(h) >= 0 else -1 for h in hours),
            dtype=np.int32,
            count=n
        )
        sh_idx = np.fromiter(
            (tables["sh_to_i"].get((str(s), int(h)), -1) if int(h) >= 0 else -1 for s, h in zip(sites, hours)),
            dtype=np.int32,
            count=n
        )
    
        valid = hour_idx >= 0
        if valid.any():
            nh = tables["hour_n"][hour_idx[valid]][:, None]
            wh = nh / (nh + 8.0)
            p[valid] = wh * tables["hour_p"][hour_idx[valid]] + (1.0 - wh) * p[valid]
    
        valid = site_idx >= 0
        if valid.any():
            ns = tables["site_n"][site_idx[valid]][:, None]
            ws = ns / (ns + 8.0)
            p[valid] = ws * tables["site_p"][site_idx[valid]] + (1.0 - ws) * p[valid]
    
        valid = sh_idx >= 0
        if valid.any():
            nsh = tables["sh_n"][sh_idx[valid]][:, None]
            wsh = nsh / (nsh + 4.0)
            p[valid] = wsh * tables["sh_p"][sh_idx[valid]] + (1.0 - wsh) * p[valid]
    
        np.clip(p, eps, 1.0 - eps, out=p)
        return (np.log(p) - np.log1p(-p)).astype(np.float32, copy=False)
    
    def fuse_scores_with_tables(base_scores, sites, hours, tables,
                                lambda_event=BEST["lambda_event"],
                                lambda_texture=BEST["lambda_texture"],
                                lambda_proxy_texture=BEST["lambda_proxy_texture"],
                                smooth_texture=BEST["smooth_texture"],
                                smooth_event=BEST["smooth_event"]):
        scores = base_scores.copy()
        prior = prior_logits_from_tables(sites, hours, tables)
    
        # mapped active
        if len(idx_mapped_active_event):
            scores[:, idx_mapped_active_event] += lambda_event * prior[:, idx_mapped_active_event]
    
        if len(idx_mapped_active_texture):
            scores[:, idx_mapped_active_texture] += lambda_texture * prior[:, idx_mapped_active_texture]
    
        # selected frog proxies
        if len(idx_selected_proxy_active_texture):
            scores[:, idx_selected_proxy_active_texture] += lambda_proxy_texture * prior[:, idx_selected_proxy_active_texture]
    
        # prior-only active unmapped
        if len(idx_selected_prioronly_active_event):
            scores[:, idx_selected_prioronly_active_event] = lambda_event * prior[:, idx_selected_prioronly_active_event]
    
        if len(idx_selected_prioronly_active_texture):
            scores[:, idx_selected_prioronly_active_texture] = lambda_texture * prior[:, idx_selected_prioronly_active_texture]
    
        # inactive unmapped
        if len(idx_unmapped_inactive):
            scores[:, idx_unmapped_inactive] = -8.0
    
        scores = smooth_cols_fixed12(scores, idx_active_texture, alpha=smooth_texture)
        scores = smooth_events_fixed12(scores, idx_active_event, alpha=smooth_event)
        return scores.astype(np.float32, copy=False), prior
    
    # Cell 8 ? Honest OOF base/prior meta-features (required for final stacker fit)
    # def build_oof_base_prior(scores_full_raw, meta_full, sc_clean, Y_SC, n_splits=5, verbose=True):
    #     groups_full = meta_full["filename"].to_numpy()
    #     gkf = GroupKFold(n_splits=n_splits)
    
    #     oof_base = np.zeros_like(scores_full_raw, dtype=np.float32)
    #     oof_prior = np.zeros_like(scores_full_raw, dtype=np.float32)
    #     fold_id = np.full(len(meta_full), -1, dtype=np.int16)
    
    #     splits = list(gkf.split(scores_full_raw, groups=groups_full))
    #     iterator = tqdm(splits, desc="OOF base/prior folds", disable=not verbose)
    
    #     for fold, (tr_idx, va_idx) in enumerate(iterator, 1):
    #         tr_idx = np.sort(tr_idx)
    #         va_idx = np.sort(va_idx)
    
    #         val_files = set(meta_full.iloc[va_idx]["filename"].tolist())
    
    #         # Fold-safe prior tables: exclude all validation files
    #         prior_mask = ~sc_clean["filename"].isin(val_files).values
    #         prior_df_fold = sc_clean.loc[prior_mask].reset_index(drop=True)
    #         Y_prior_fold = Y_SC[prior_mask]
    
    #         tables = fit_prior_tables(prior_df_fold, Y_prior_fold)
    
    #         va_base, va_prior = fuse_scores_with_tables(
    #             scores_full_raw[va_idx],
    #             sites=meta_full.iloc[va_idx]["site"].to_numpy(),
    #             hours=meta_full.iloc[va_idx]["hour_utc"].to_numpy(),
    #             tables=tables,
    #         )
    
    #         oof_base[va_idx] = va_base
    #         oof_prior[va_idx] = va_prior
    #         fold_id[va_idx] = fold
    
    #     assert (fold_id >= 0).all()
    #     return oof_base, oof_prior, fold_id
    from sklearn.model_selection import StratifiedGroupKFold
    def build_oof_base_prior(scores_full_raw, meta_full, sc_clean, Y_SC, n_splits=5, verbose=True):
        groups_full = meta_full["filename"].to_numpy()
        
        row_id_to_idx = {r: i for i, r in enumerate(sc_clean["row_id"])}
        aligned_indices = [row_id_to_idx[r] for r in meta_full["row_id"]]
        Y_ALIGNED = Y_SC[aligned_indices]  # ?????? 708 ?????
        
        y_strat = np.argmax(Y_ALIGNED, axis=1)
        unique_classes, counts = np.unique(y_strat, return_counts=True)
        rare_classes = unique_classes[counts < n_splits]
        y_strat[np.isin(y_strat, rare_classes)] = -1
        
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=91)
    
        oof_base = np.zeros_like(scores_full_raw, dtype=np.float32)
        oof_prior = np.zeros_like(scores_full_raw, dtype=np.float32)
        fold_id = np.full(len(meta_full), -1, dtype=np.int16)
    
        splits = list(sgkf.split(scores_full_raw, y_strat, groups=groups_full))
        iterator = tqdm(splits, desc="OOF base/prior folds", disable=not verbose)
    
        for fold, (tr_idx, va_idx) in enumerate(iterator, 1):
            tr_idx = np.sort(tr_idx)
            va_idx = np.sort(va_idx)
    
            val_files = set(meta_full.iloc[va_idx]["filename"].tolist())
    
            # Fold-safe prior tables: exclude all validation files
            prior_mask = ~sc_clean["filename"].isin(val_files).values
            prior_df_fold = sc_clean.loc[prior_mask].reset_index(drop=True)
            Y_prior_fold = Y_SC[prior_mask]
    
            tables = fit_prior_tables(prior_df_fold, Y_prior_fold)
    
            va_base, va_prior = fuse_scores_with_tables(
                scores_full_raw[va_idx],
                sites=meta_full.iloc[va_idx]["site"].to_numpy(),
                hours=meta_full.iloc[va_idx]["hour_utc"].to_numpy(),
                tables=tables,
            )
    
            oof_base[va_idx] = va_base
            oof_prior[va_idx] = va_prior
            fold_id[va_idx] = fold
    
        assert (fold_id >= 0).all()
        return oof_base, oof_prior, fold_id
    
    
    OOF_META_CACHE = CFG["full_cache_work_dir"] / "full_oof_meta_features.npz"
    
    if OOF_META_CACHE.exists():
        print("Loading cached OOF meta-features from:", OOF_META_CACHE)
        arr = np.load(OOF_META_CACHE)
        oof_base = arr["oof_base"].astype(np.float32)
        oof_prior = arr["oof_prior"].astype(np.float32)
        oof_fold_id = arr["fold_id"].astype(np.int16)
    else:
        print("Building OOF meta-features...")
        oof_base, oof_prior, oof_fold_id = build_oof_base_prior(
            scores_full_raw=scores_full_raw,
            meta_full=meta_full,
            sc_clean=sc_clean,
            Y_SC=Y_SC,
            n_splits=5,
            verbose=CFG["verbose"],
        )
    
        np.savez_compressed(
            OOF_META_CACHE,
            oof_base=oof_base,
            oof_prior=oof_prior,
            fold_id=oof_fold_id,
        )
        print("Saved OOF meta-features to:", OOF_META_CACHE)
    
    baseline_oof_auc = macro_auc_skip_empty(Y_FULL, oof_base)
    
    if MODE == "train":
        raw_local_auc = macro_auc_skip_empty(Y_FULL, scores_full_raw)
        print(f"Raw local AUC (not OOF-dependent): {raw_local_auc:.6f}")
        print(f"Honest OOF baseline AUC: {baseline_oof_auc:.6f}")
    
    import torch
    import torch.nn as nn
    import numpy as np
    
    def build_all_class_features_vectorized(Z, raw_scores, prior_scores, base_scores, valid_classes, n_windows=12):
        """
        A function that constructs all 14 types of features for all classes in one go, without using a for loop.
        Output tensor shape: (V: number of effective classes, N: number of samples, D+14)
        """
        N, D = Z.shape
        V = len(valid_classes)
        
        # (V, N)
        raw = raw_scores[:, valid_classes].T
        prior = prior_scores[:, valid_classes].T
        base = base_scores[:, valid_classes].T
        
        n_files = N // n_windows
        base_view = base.reshape(V, n_files, n_windows)
        
        # Batch calculation of time series features
        prev_base = np.concatenate([base_view[:, :, :1], base_view[:, :, :-1]], axis=2).reshape(V, N)
        next_base = np.concatenate([base_view[:, :, 1:], base_view[:, :, -1:]], axis=2).reshape(V, N)
        mean_base = np.repeat(base_view.mean(axis=2), n_windows, axis=1)
        max_base = np.repeat(base_view.max(axis=2), n_windows, axis=1)
        std_base = np.repeat(base_view.std(axis=2), n_windows, axis=1)
        
        diff_mean = base - mean_base
        diff_prev = base - prev_base
        diff_next = base - next_base
        
        interact_rp = raw * prior
        interact_rb = raw * base
        interact_pb = prior * base
        
        # Stack 14 scalar features in the last dimension -> (V, N, 14)
        scalar_feats = np.stack([
            raw, prior, base, prev_base, next_base, 
            mean_base, max_base, std_base, 
            diff_mean, diff_prev, diff_next, 
            interact_rp, interact_rb, interact_pb
        ], axis=-1)
        
        # Z (N, D) -> (V, N, D) 
        Z_expanded = np.broadcast_to(Z, (V, N, D))
        
        # features -> (V, N, D+14)
        X_all = np.concatenate([Z_expanded, scalar_feats], axis=-1)
        return X_all.astype(np.float32)
    
    class VectorizedMLPProbes(nn.Module):
        """
        A class that combines multiple scikit-learn MLPClassifier classes into a single PyTorch model.
        """
        def __init__(self, probe_models, device="cpu"):
            super().__init__()
            self.valid_classes = sorted(list(probe_models.keys()))
            self.V = len(self.valid_classes)
            
            if self.V == 0:
                return
                
            sample_clf = probe_models[self.valid_classes[0]]
            self.n_layers = len(sample_clf.coefs_)
            
            self.weights = nn.ParameterList()
            self.biases = nn.ParameterList()
            
            # (V, in_dim, out_dim)
            for layer_idx in range(self.n_layers):
                W = np.stack([probe_models[c].coefs_[layer_idx] for c in self.valid_classes], axis=0)
                b = np.stack([probe_models[c].intercepts_[layer_idx] for c in self.valid_classes], axis=0)
                
                self.weights.append(nn.Parameter(torch.tensor(W, dtype=torch.float32), requires_grad=False))
                self.biases.append(nn.Parameter(torch.tensor(b, dtype=torch.float32), requires_grad=False))
                
            self.to(device)
    
        def forward(self, x):
            # x shape: (V, N, in_dim)
            h = x
            for i in range(self.n_layers):
                h = torch.bmm(h, self.weights[i]) + self.biases[i].unsqueeze(1)
                if i < self.n_layers - 1:
                    h = torch.relu(h)
            
            return h.squeeze(-1) # (V, N)
    
    def get_vectorized_mlp_scores(Z, raw, prior, base, probe_models, alpha_p, n_windows=12, device="cpu"):
        """
        A wrapper function that wraps all of the above vectorization processes
        """
        mlp_scores = base.copy()
        if len(probe_models) == 0:
            return mlp_scores
            
        valid_classes = sorted(list(probe_models.keys()))
        
        # 1. Building a tensor
        X_all = build_all_class_features_vectorized(Z, raw, prior, base, valid_classes, n_windows)
        
        # 2. Batch inference using PyTorch
        vec_probe = VectorizedMLPProbes(probe_models, device=device)
        vec_probe.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_all, dtype=torch.float32, device=device)
            preds = vec_probe(X_tensor).cpu().numpy() # (V, N)
            
        # 3. Blending
        preds_t = preds.T # (N, V)
        base_valid = base[:, valid_classes]
        
        mlp_scores[:, valid_classes] = (1.0 - alpha_p) * base_valid + alpha_p * preds_t
        return mlp_scores
    
    # Cell 9 ? Classwise embedding-probe helpers
    def build_class_features(emb_proj, raw_col, prior_col, base_col):
        """
        emb_proj: (n, d)
        raw_col, prior_col, base_col: (n,)
        returns: (n, d + 13)
    
        Fitur: embedding + 7 sequential + 3 interaction + std + 3 diff
        """
        prev_base, next_base, mean_base, max_base, std_base = seq_features_1d(base_col)
    
        # Diff features: posisi window relatif terhadap konteks file
        diff_mean = base_col - mean_base   # apakah window ini lebih tinggi dari rata2 file?
        diff_prev = base_col - prev_base   # onset: naik dari window sebelumnya?
        diff_next = base_col - next_base   # offset: turun ke window berikutnya?
    
        feats = np.concatenate([
            emb_proj,
            raw_col[:, None],
            prior_col[:, None],
            base_col[:, None],
            prev_base[:, None],
            next_base[:, None],
            mean_base[:, None],
            max_base[:, None],
            std_base[:, None],             # variance temporal dalam file
            diff_mean[:, None],            # deviasi dari mean file
            diff_prev[:, None],            # deteksi onset
            diff_next[:, None],            # deteksi offset
            # interaction terms
            (raw_col * prior_col)[:, None],
            (raw_col * base_col)[:, None],
            (prior_col * base_col)[:, None],
        ], axis=1)
    
        return feats.astype(np.float32, copy=False)
    
    from sklearn.model_selection import StratifiedGroupKFold
    
    def run_oof_embedding_probe(
        scores_raw,
        emb,
        meta_df,
        y_true,
        pca_dim=64,
        min_pos=8,
        C=0.25,
        alpha=0.5,
    ):
        groups = meta_df["filename"].to_numpy()
        
        y_strat = np.argmax(Y_SC, axis=1) 
        
        unique_classes, counts = np.unique(y_strat, return_counts=True)
        rare_classes = unique_classes[counts < n_splits]
        y_strat[np.isin(y_strat, rare_classes)] = -1
        
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=91)
    
        oof_base_local = np.zeros_like(scores_raw, dtype=np.float32)
        oof_final = np.zeros_like(scores_raw, dtype=np.float32)
        modeled_counts = np.zeros(scores_raw.shape[1], dtype=np.int32)
        oof_models = {}
    
        split_list = list(sgkf.split(scores_raw, y_strat, groups=groups))
    
        for fold, (tr_idx, va_idx) in enumerate(tqdm(split_list, desc="Embedding-probe folds", disable=not CFG.get("verbose", True)), 1):
            tr_idx = np.sort(tr_idx)
            va_idx = np.sort(va_idx)
    
            val_files = set(meta_df.iloc[va_idx]["filename"].tolist())
    
            # Fold-safe priors
            prior_mask = ~sc_clean["filename"].isin(val_files).values
            prior_df_fold = sc_clean.loc[prior_mask].reset_index(drop=True)
            Y_prior_fold = Y_SC[prior_mask]
            tables = fit_prior_tables(prior_df_fold, Y_prior_fold)
    
            base_tr, prior_tr = fuse_scores_with_tables(
                scores_raw[tr_idx],
                sites=meta_df.iloc[tr_idx]["site"].to_numpy(),
                hours=meta_df.iloc[tr_idx]["hour_utc"].to_numpy(),
                tables=tables,
            )
            base_va, prior_va = fuse_scores_with_tables(
                scores_raw[va_idx],
                sites=meta_df.iloc[va_idx]["site"].to_numpy(),
                hours=meta_df.iloc[va_idx]["hour_utc"].to_numpy(),
                tables=tables,
            )
    
            oof_base_local[va_idx] = base_va
            oof_final[va_idx] = base_va
    
            # Embedding preprocessing on train fold only
            scaler = StandardScaler()
            emb_tr_s = scaler.fit_transform(emb[tr_idx])
            emb_va_s = scaler.transform(emb[va_idx])
    
            n_comp = min(pca_dim, emb_tr_s.shape[0] - 1, emb_tr_s.shape[1])
            pca = PCA(n_components=n_comp)
            Z_tr = pca.fit_transform(emb_tr_s).astype(np.float32)
            Z_va = pca.transform(emb_va_s).astype(np.float32)
    
            class_iterator = np.where(y_true[tr_idx].sum(axis=0) >= min_pos)[0].tolist()
    
            for cls_idx in tqdm(class_iterator, desc=f"Fold {fold} classes", leave=False, disable=not CFG["verbose"]):
            # for cls_idx in tqdm(class_iterator, desc=f"Fold {fold} classes", leave=False):
                y_tr = y_true[tr_idx, cls_idx]
    
                if y_tr.sum() == 0 or y_tr.sum() == len(y_tr):
                    continue
    
                X_tr_cls = build_class_features(
                    Z_tr,
                    raw_col=scores_raw[tr_idx, cls_idx],
                    prior_col=prior_tr[:, cls_idx],
                    base_col=base_tr[:, cls_idx],
                )
                X_va_cls = build_class_features(
                    Z_va,
                    raw_col=scores_raw[va_idx, cls_idx],
                    prior_col=prior_va[:, cls_idx],
                    base_col=base_va[:, cls_idx],
                )
    
                # Pilih backend probe: mlp | lgbm | logreg
                backend = CFG.get("probe_backend", "mlp")
                n_pos = int(y_tr.sum())
                n_neg = len(y_tr) - n_pos
    
                if backend == "mlp":
                    # MLPClassifier tidak support sample_weight
                    # Gunakan oversampling: duplikasi positif agar balance
                    if n_pos > 0 and n_neg > n_pos:
                        repeat = max(1, n_neg // n_pos)
                        pos_idx = np.where(y_tr == 1)[0]
                        X_bal = np.vstack([X_tr_cls, np.tile(X_tr_cls[pos_idx], (repeat, 1))])
                        y_bal = np.concatenate([y_tr, np.ones(len(pos_idx) * repeat, dtype=y_tr.dtype)])
                    else:
                        X_bal, y_bal = X_tr_cls, y_tr
                    clf = MLPClassifier(**CFG["mlp_params"])
                    clf.fit(X_bal, y_bal)
                    pred_va = clf.predict_proba(X_va_cls)[:, 1].astype(np.float32)
                    pred_va = np.log(pred_va + 1e-7) - np.log(1 - pred_va + 1e-7)
                elif backend == "lgbm" and _LGBM_AVAILABLE:
                    scale_pos = max(1.0, n_neg / max(n_pos, 1))
                    clf = LGBMClassifier(
                        **CFG["lgbm_params"],
                        scale_pos_weight=scale_pos,
                    )
                    clf.fit(X_tr_cls, y_tr)
                    pred_va = clf.predict_proba(X_va_cls)[:, 1].astype(np.float32)
                    pred_va = np.log(pred_va + 1e-7) - np.log(1 - pred_va + 1e-7)
                else:
                    clf = LogisticRegression(
                        C=C, max_iter=400, solver="liblinear",
                        class_weight="balanced",
                    )
                    clf.fit(X_tr_cls, y_tr)
                    pred_va = clf.decision_function(X_va_cls).astype(np.float32)
    
                oof_final[va_idx, cls_idx] = (
                    (1.0 - alpha) * base_va[:, cls_idx] +
                    alpha * pred_va
                )
    
                modeled_counts[cls_idx] += 1
    
        score_base = macro_auc_skip_empty(y_true, oof_base_local)
        score_final = macro_auc_skip_empty(y_true, oof_final)
    
        return {
            "oof_base": oof_base_local,
            "oof_final": oof_final,
            "modeled_counts": modeled_counts,
            "score_base": score_base,
            "score_final": score_final,
        }
    
    # ProtoSSM v4 ? Enhanced with Cross-Attention Layer
    
    class SelectiveSSM(nn.Module):
        # Simplified Mamba-style selective state space model.
        # Input-dependent (selective) discretization of continuous-time SSM.
        # For T=12 bioacoustic windows, the sequential scan is efficient on CPU.
    
        def __init__(self, d_model, d_state=16, d_conv=4):
            super().__init__()
            self.d_model = d_model
            self.d_state = d_state
    
            self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
            self.conv1d = nn.Conv1d(
                d_model, d_model, d_conv,
                padding=d_conv - 1, groups=d_model
            )
            self.dt_proj = nn.Linear(d_model, d_model, bias=True)
    
            A = torch.arange(1, d_state + 1, dtype=torch.float32)
            A = A.unsqueeze(0).expand(d_model, -1)
            self.A_log = nn.Parameter(torch.log(A))
            self.D = nn.Parameter(torch.ones(d_model))
            self.B_proj = nn.Linear(d_model, d_state, bias=False)
            self.C_proj = nn.Linear(d_model, d_state, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)
    
        def forward(self, x):
            B_size, T, D = x.shape
            xz = self.in_proj(x)
            x_ssm, z = xz.chunk(2, dim=-1)
    
            x_conv = self.conv1d(x_ssm.transpose(1, 2))[:, :, :T].transpose(1, 2)
            x_conv = F.silu(x_conv)
    
            dt = F.softplus(self.dt_proj(x_conv))
            A = -torch.exp(self.A_log)
            B = self.B_proj(x_conv)
            C = self.C_proj(x_conv)
    
            h = torch.zeros(B_size, D, self.d_state, device=x.device)
            ys = []
            for t in range(T):
                dt_t = dt[:, t, :]
                dA = torch.exp(A[None, :, :] * dt_t[:, :, None])
                dB = dt_t[:, :, None] * B[:, t, None, :]
                h = h * dA + x[:, t, :, None] * dB
                y_t = (h * C[:, t, None, :]).sum(-1)
                ys.append(y_t)
    
            y = torch.stack(ys, dim=1)
            return y + x * self.D[None, None, :]
    
    
    class TemporalCrossAttention(nn.Module):
        """Multi-head cross-attention between temporal windows.
        Captures non-local patterns (e.g., dawn chorus onset, counter-singing)
        that sequential SSM may miss."""
        
        def __init__(self, d_model, n_heads=4, dropout=0.1):
            super().__init__()
            self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            self.norm = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
                nn.Dropout(dropout),
            )
            self.norm2 = nn.LayerNorm(d_model)
        
        def forward(self, x):
            # x: (B, T, D)
            residual = x
            x = self.norm(x)
            attn_out, _ = self.attn(x, x, x)
            x = residual + attn_out
            
            residual = x
            x = self.norm2(x)
            x = residual + self.ffn(x)
            return x
    
    
    class ProtoSSMv2(nn.Module):
        # Prototypical State Space Model v4 with cross-attention and metadata awareness.
        #
        # V16 additions:
        # - Cross-attention layer after SSM for non-local temporal patterns
        # - All other v2 features preserved (metadata, prototypes, gated fusion)
        
        def __init__(self, d_input=1536, d_model=192, d_state=16,
                     n_ssm_layers=2, n_classes=234, n_windows=12,
                     dropout=0.2, n_sites=20, meta_dim=16,
                     use_cross_attn=True, cross_attn_heads=4):
            super().__init__()
            self.d_model = d_model
            self.n_classes = n_classes
            self.n_windows = n_windows
    
            # 1. Feature projection
            self.input_proj = nn.Sequential(
                nn.Linear(d_input, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
    
            # 2. Learnable positional encoding
            self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)
    
            # 3. Metadata embeddings
            self.site_emb = nn.Embedding(n_sites, meta_dim)
            self.hour_emb = nn.Embedding(24, meta_dim)
            self.meta_proj = nn.Linear(2 * meta_dim, d_model)
    
            # 4. Bidirectional SSM layers
            self.ssm_fwd = nn.ModuleList()
            self.ssm_bwd = nn.ModuleList()
            self.ssm_merge = nn.ModuleList()
            self.ssm_norm = nn.ModuleList()
            for _ in range(n_ssm_layers):
                self.ssm_fwd.append(SelectiveSSM(d_model, d_state))
                self.ssm_bwd.append(SelectiveSSM(d_model, d_state))
                self.ssm_merge.append(nn.Linear(2 * d_model, d_model))
                self.ssm_norm.append(nn.LayerNorm(d_model))
            self.ssm_drop = nn.Dropout(dropout)
    
            # 4b. NEW: Cross-attention after SSM
            self.use_cross_attn = use_cross_attn
            if use_cross_attn:
                self.cross_attn = TemporalCrossAttention(d_model, n_heads=cross_attn_heads, dropout=dropout)
    
            # 5. Learnable class prototypes
            self.prototypes = nn.Parameter(torch.randn(n_classes, d_model) * 0.02)
            self.proto_temp = nn.Parameter(torch.tensor(5.0))
    
            # 6. Per-class calibration bias
            self.class_bias = nn.Parameter(torch.zeros(n_classes))
    
            # 7. Per-class gated fusion with Perch logits
            self.fusion_alpha = nn.Parameter(torch.zeros(n_classes))
    
            # 8. Taxonomic auxiliary head
            self.n_families = 0
            self.family_head = None
    
        def init_prototypes_from_data(self, embeddings, labels):
            with torch.no_grad():
                h = self.input_proj(embeddings)
                for c in range(self.n_classes):
                    mask = labels[:, c] > 0.5
                    if mask.sum() > 0:
                        self.prototypes.data[c] = F.normalize(h[mask].mean(0), dim=0)
    
        def init_family_head(self, n_families, class_to_family):
            self.n_families = n_families
            self.family_head = nn.Linear(self.d_model, n_families)
            self.register_buffer('class_to_family', torch.tensor(class_to_family, dtype=torch.long))
    
        def forward(self, emb, perch_logits=None, site_ids=None, hours=None):
            B, T, _ = emb.shape
    
            # Project embeddings
            h = self.input_proj(emb)
            h = h + self.pos_enc[:, :T, :]
    
            # Add metadata embeddings
            if site_ids is not None and hours is not None:
                s_emb = self.site_emb(site_ids)
                h_emb = self.hour_emb(hours)
                meta = self.meta_proj(torch.cat([s_emb, h_emb], dim=-1))
                h = h + meta[:, None, :]
    
            # Bidirectional SSM
            for fwd, bwd, merge, norm in zip(
                self.ssm_fwd, self.ssm_bwd, self.ssm_merge, self.ssm_norm
            ):
                residual = h
                h_f = fwd(h)
                h_b = bwd(h.flip(1)).flip(1)
                h = merge(torch.cat([h_f, h_b], dim=-1))
                h = self.ssm_drop(h)
                h = norm(h + residual)
    
            # NEW: Cross-attention for non-local temporal patterns
            if self.use_cross_attn:
                h = self.cross_attn(h)
    
            h_temporal = h
    
            # Prototypical cosine similarity + class bias
            h_norm = F.normalize(h, dim=-1)
            p_norm = F.normalize(self.prototypes, dim=-1)
            temp = F.softplus(self.proto_temp)
            sim = torch.matmul(h_norm, p_norm.T) * temp + self.class_bias[None, None, :]
    
            # Gated fusion with Perch logits
            if perch_logits is not None:
                alpha = torch.sigmoid(self.fusion_alpha)[None, None, :]
                species_logits = alpha * sim + (1 - alpha) * perch_logits
            else:
                species_logits = sim
    
            # Taxonomic auxiliary prediction
            family_logits = None
            if self.family_head is not None:
                h_pool = h.mean(dim=1)
                family_logits = self.family_head(h_pool)
    
            return species_logits, family_logits, h_temporal
    
        def count_parameters(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    ssm_cfg = CFG["proto_ssm"]
    print("ProtoSSMv4 architecture defined (with cross-attention).")
    test_model = ProtoSSMv2(
        d_model=ssm_cfg["d_model"], n_ssm_layers=2,
        n_sites=ssm_cfg["n_sites"], meta_dim=ssm_cfg["meta_dim"],
        use_cross_attn=ssm_cfg.get("use_cross_attn", True),
        cross_attn_heads=ssm_cfg.get("cross_attn_heads", 4),
    )
    print(f"Parameter count: {test_model.count_parameters():,}")
    del test_model
    
    # ProtoSSM v4 Training Loop ? with Mixup, Focal Loss, SWA
    
    def build_taxonomy_groups(taxonomy_df, primary_labels):
        for col in ["family", "order", "class_name"]:
            if col in taxonomy_df.columns:
                group_map = taxonomy_df.set_index("primary_label")[col].to_dict()
                break
        else:
            group_map = {label: "Unknown" for label in primary_labels}
    
        groups = sorted(set(group_map.values()))
        grp_to_idx = {g: i for i, g in enumerate(groups)}
        class_to_group = []
        for label in primary_labels:
            grp = group_map.get(label, "Unknown")
            class_to_group.append(grp_to_idx.get(grp, 0))
        return len(groups), class_to_group, grp_to_idx
    
    
    def build_site_mapping(meta_df):
        sites = meta_df["site"].unique().tolist()
        site_to_idx = {s: i + 1 for i, s in enumerate(sites)}
        n_sites = len(sites) + 1
        return site_to_idx, n_sites
    
    
    def reshape_to_files(flat_array, meta_df, n_windows=N_WINDOWS):
        filenames = meta_df["filename"].to_numpy()
        unique_files = []
        seen = set()
        for f in filenames:
            if f not in seen:
                unique_files.append(f)
                seen.add(f)
    
        n_files = len(unique_files)
        assert len(flat_array) == n_files * n_windows, \
            f"Expected {n_files * n_windows} rows, got {len(flat_array)}"
    
        new_shape = (n_files, n_windows) + flat_array.shape[1:]
        return flat_array.reshape(new_shape), unique_files
    
    
    def get_file_metadata(meta_df, file_list, site_to_idx, n_sites_max):
        file_to_row = {}
        filenames = meta_df["filename"].to_numpy()
        sites = meta_df["site"].to_numpy()
        hours = meta_df["hour_utc"].to_numpy()
        for i, f in enumerate(filenames):
            if f not in file_to_row:
                file_to_row[f] = i
    
        site_ids = np.zeros(len(file_list), dtype=np.int64)
        hour_ids = np.zeros(len(file_list), dtype=np.int64)
        for fi, fname in enumerate(file_list):
            row = file_to_row.get(fname)
            if row is not None:
                sid = site_to_idx.get(sites[row], 0)
                site_ids[fi] = min(sid, n_sites_max - 1)
                hour_ids[fi] = int(hours[row]) % 24
        return site_ids, hour_ids
    
    
    def mixup_files(emb, logits, labels, site_ids, hours, families, alpha=0.3):
        """File-level mixup augmentation for ProtoSSM training.
        Mixes pairs of files with random lambda from Beta(alpha, alpha).
        Returns augmented versions of all inputs."""
        n = len(emb)
        if alpha <= 0 or n < 2:
            return emb, logits, labels, site_ids, hours, families
        
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1.0 - lam)  # Ensure lam >= 0.5 (dominant sample stays dominant)
        
        perm = np.random.permutation(n)
        
        emb_mix = lam * emb + (1 - lam) * emb[perm]
        logits_mix = lam * logits + (1 - lam) * logits[perm]
        labels_mix = lam * labels + (1 - lam) * labels[perm]
        
        # For discrete features (site, hour), keep the dominant sample's values
        families_mix = lam * families + (1 - lam) * families[perm] if families is not None else None
        
        return emb_mix, logits_mix, labels_mix, site_ids, hours, families_mix
    
    # ?????????????????????????????????????????????????????????????????????????????
    # ?[IMPORTANT]ProtoSSM exists, skip training.???????????????????????????
    # * If you want to perform training, please set it to None.              
    # ?????????????????????????????????????????????????????????????????????????????
    # ProtoSSM_PATH = "train_proto_ssm_single/models/proto_ssm_best.pt"
    # ProtoSSM_JSON = "train_proto_ssm_single/models/proto_ssm_history.json"
    ProtoSSM_PATH = "/kaggle/input/datasets/hideyukizushi/sgkfk-202604041716/train_proto_ssm_single/models/proto_ssm_best.pt"
    ProtoSSM_JSON = "/kaggle/input/datasets/hideyukizushi/sgkfk-202604041716/train_proto_ssm_single/models/proto_ssm_history.json"
    
    
    def train_proto_ssm_single(model, emb_train, logits_train, labels_train,
                               site_ids_train=None, hours_train=None,
                               emb_val=None, logits_val=None, labels_val=None,
                               site_ids_val=None, hours_val=None,
                               file_families_train=None, file_families_val=None,
                               cfg=None, verbose=True):
        """Train a single ProtoSSM v4 model with mixup, focal loss, and SWA."""
        print("????????????????????????????????????????????????????????")
        print("?????ProtoSSM Train...:")
        print("????????????????????????????????????????????????????????")
        if ProtoSSM_PATH is not None and ProtoSSM_JSON is not None:
            print("????????????????????????????????????????????????????????")
            print("?????ProtoSSM Load Model(TrainSkip)...:")
            print("????????????????????????????????????????????????????????")
            load_model_path = CFG.get("pretrained_proto_path", ProtoSSM_PATH)
            load_hist_path = CFG.get("pretrained_hist_path", ProtoSSM_JSON)
            
            # Model Load
            if os.path.exists(load_model_path):
                model.load_state_dict(torch.load(load_model_path, map_location=DEVICE))
                model.eval()
                if verbose:
                    print(f"? [Load] Loaded pre-trained ProtoSSM from {load_model_path}")
            else:
                print(f"?? WARNING: Pre-trained model not found at {load_model_path}!")
                
            # History Load
            history = {"train_loss": [], "val_loss": [], "val_auc": []}
            if os.path.exists(load_hist_path):
                import json
                with open(load_hist_path, "r") as f:
                    history = json.load(f)
                    
            return model, history
        
    
        if cfg is None:
            cfg = CFG["proto_ssm_train"]
    
        label_smoothing = cfg.get("label_smoothing", 0.0)
        mixup_alpha = cfg.get("mixup_alpha", 0.0)
        focal_gamma = cfg.get("focal_gamma", 0.0)
        swa_start_frac = cfg.get("swa_start_frac", 1.0)  # 1.0 = disabled
        n_epochs = cfg["n_epochs"]
        swa_start_epoch = int(n_epochs * swa_start_frac)
    
        # Convert to tensors (base ? unmixed)
        labels_np = labels_train.copy()
        
        # Apply label smoothing
        if label_smoothing > 0:
            labels_np = labels_np * (1.0 - label_smoothing) + label_smoothing / 2.0
    
        has_val = emb_val is not None
        if has_val:
            emb_v = torch.tensor(emb_val, dtype=torch.float32)
            logits_v = torch.tensor(logits_val, dtype=torch.float32)
            labels_v = torch.tensor(labels_val, dtype=torch.float32)
            site_v = torch.tensor(site_ids_val, dtype=torch.long) if site_ids_val is not None else None
            hour_v = torch.tensor(hours_val, dtype=torch.long) if hours_val is not None else None
    
        fam_v = torch.tensor(file_families_val, dtype=torch.float32) if (has_val and file_families_val is not None) else None
    
        # Class weights for imbalanced data
        labels_tr_t = torch.tensor(labels_np, dtype=torch.float32)
        pos_counts = labels_tr_t.sum(dim=(0, 1))
        total = labels_tr_t.shape[0] * labels_tr_t.shape[1]
        pos_weight = ((total - pos_counts) / (pos_counts + 1)).clamp(max=cfg["pos_weight_cap"])
    
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=cfg["lr"],
            epochs=n_epochs, steps_per_epoch=1,
            pct_start=0.1, anneal_strategy='cos'
        )
    
        best_val_loss = float('inf')
        best_state = None
        wait = 0
        history = {"train_loss": [], "val_loss": [], "val_auc": []}
    
        # SWA state accumulator
        swa_state = None
        swa_count = 0
    
        for epoch in range(n_epochs):
            # === Mixup augmentation (per-epoch re-sampling) ===
            if mixup_alpha > 0 and epoch > 5:  # Skip mixup for first 5 epochs (warmup)
                emb_mix, logits_mix, labels_mix, _, _, fam_mix = mixup_files(
                    emb_train, logits_train, labels_np,
                    site_ids_train, hours_train, file_families_train,
                    alpha=mixup_alpha,
                )
            else:
                emb_mix, logits_mix, labels_mix = emb_train, logits_train, labels_np
                fam_mix = file_families_train
    
            emb_tr = torch.tensor(emb_mix, dtype=torch.float32)
            logits_tr = torch.tensor(logits_mix, dtype=torch.float32)
            labels_tr = torch.tensor(labels_mix, dtype=torch.float32)
            site_tr = torch.tensor(site_ids_train, dtype=torch.long) if site_ids_train is not None else None
            hour_tr = torch.tensor(hours_train, dtype=torch.long) if hours_train is not None else None
            fam_tr = torch.tensor(fam_mix, dtype=torch.float32) if fam_mix is not None else None
    
            # === Train ===
            model.train()
            species_out, family_out, _ = model(emb_tr, logits_tr, site_ids=site_tr, hours=hour_tr)
    
            # Primary loss: focal BCE or weighted BCE
            if focal_gamma > 0:
                loss_main = focal_bce_with_logits(
                    species_out, labels_tr,
                    gamma=focal_gamma,
                    pos_weight=pos_weight[None, None, :],
                )
            else:
                loss_main = F.binary_cross_entropy_with_logits(
                    species_out, labels_tr,
                    pos_weight=pos_weight[None, None, :]
                )
    
            # Knowledge distillation loss
            loss_distill = F.mse_loss(species_out, logits_tr)
    
            # Total loss
            loss = loss_main + cfg["distill_weight"] * loss_distill
    
            # Taxonomic auxiliary loss
            if family_out is not None and fam_tr is not None:
                loss_family = F.binary_cross_entropy_with_logits(family_out, fam_tr)
                loss = loss + 0.1 * loss_family
    
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
    
            # === SWA accumulation ===
            if epoch >= swa_start_epoch:
                if swa_state is None:
                    swa_state = {k: v.clone() for k, v in model.state_dict().items()}
                    swa_count = 1
                else:
                    for k in swa_state:
                        swa_state[k] += model.state_dict()[k]
                    swa_count += 1
    
            # === Validate ===
            model.eval()
            with torch.no_grad():
                if has_val:
                    val_out, val_fam, _ = model(emb_v, logits_v, site_ids=site_v, hours=hour_v)
                    val_loss = F.binary_cross_entropy_with_logits(
                        val_out, labels_v,
                        pos_weight=pos_weight[None, None, :]
                    )
    
                    val_pred = val_out.reshape(-1, val_out.shape[-1]).numpy()
                    val_true = labels_v.reshape(-1, labels_v.shape[-1]).numpy()
                    try:
                        val_auc = macro_auc_skip_empty(val_true, val_pred)
                    except Exception:
                        val_auc = 0.0
                else:
                    val_loss = loss
                    val_auc = 0.0
    
            history["train_loss"].append(loss.item())
            history["val_loss"].append(val_loss.item())
            history["val_auc"].append(val_auc)
    
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
    
            if verbose and (epoch + 1) % 20 == 0:
                lr_now = optimizer.param_groups[0]['lr']
                swa_info = f" swa={swa_count}" if swa_count > 0 else ""
                print(f"  Epoch {epoch+1:3d}: train={loss.item():.4f} val={val_loss.item():.4f} "
                      f"auc={val_auc:.4f} lr={lr_now:.6f} wait={wait}{swa_info}")
    
            if wait >= cfg["patience"]:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1} (best val_loss={best_val_loss:.4f})")
                break
    
        # Apply SWA if we accumulated enough checkpoints
        if swa_state is not None and swa_count >= 3:
            if verbose:
                print(f"  Applying SWA (averaged {swa_count} checkpoints)")
            avg_state = {k: v / swa_count for k, v in swa_state.items()}
            model.load_state_dict(avg_state)
        elif best_state is not None:
            model.load_state_dict(best_state)
    
        if verbose:
            print(f"  Training complete. Best val_loss={best_val_loss:.4f}")
            with torch.no_grad():
                alphas = torch.sigmoid(model.fusion_alpha).numpy()
                print(f"  Fusion alpha: mean={alphas.mean():.3f} min={alphas.min():.3f} max={alphas.max():.3f}")
                print(f"  Proto temperature: {F.softplus(model.proto_temp).item():.3f}")
        
        # ??????? Fix 2: Save Model & History???????
        PROC_MODE = "DoTrain"
        if PROC_MODE == "DoTrain":
            save_model_path = CFG.get("proto_model_path", "train_proto_ssm_single/models/proto_ssm_best.pt")
            save_hist_path = CFG.get("proto_hist_path", "train_proto_ssm_single/models/proto_ssm_history.json")
            
            os.makedirs(os.path.dirname(save_model_path) or ".", exist_ok=True)
            
            torch.save(model.state_dict(), save_model_path)
            
            import json
            with open(save_hist_path, "w") as f:
                json.dump(history, f, indent=4)
                
            if verbose:
                print(f"? [Save] Model successfully saved to {save_model_path}")
                print(f"? [Save] History successfully saved to {save_hist_path}")
        # ??????????????????????????????????????????????????????????????????????
        
        return model, history
    
    from sklearn.model_selection import StratifiedGroupKFold
    
    def run_proto_ssm_oof(emb_files, logits_files, labels_files,
                          site_ids_all, hours_all,
                          file_families, file_groups,
                          n_families, class_to_family,
                          cfg=None, verbose=True):
        """Run StratifiedGroupKFold OOF cross-validation for ProtoSSM v4."""
        if cfg is None:
            cfg = CFG["proto_ssm_train"]
    
        n_splits = cfg.get("oof_n_splits", 5)
        n_files = len(emb_files)
        ssm_cfg = CFG["proto_ssm"]
    
        oof_preds = np.zeros((n_files, N_WINDOWS, N_CLASSES), dtype=np.float32)
        fold_histories = []
        fold_alphas = []
    
        n_unique_groups = len(set(file_groups))
        if n_unique_groups < n_splits:
            print(f"  WARNING: Only {n_unique_groups} groups, reducing n_splits from {n_splits} to {n_unique_groups}")
            n_splits = n_unique_groups
    
        
        file_level_labels = labels_files.max(axis=1) # (n_files, N_CLASSES)
        
        y_strat = np.argmax(Y_SC, axis=1) 
        
        
        unique_classes, counts = np.unique(y_strat, return_counts=True)
        rare_classes = unique_classes[counts < n_splits]
        y_strat[np.isin(y_strat, rare_classes)] = -1
        
        
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=91)
        for fold_i, (train_idx, val_idx) in enumerate(sgkf.split(emb_files, y_strat, groups=file_groups)):
            if verbose:
                print(f"\n--- Fold {fold_i+1}/{n_splits} (train={len(train_idx)}, val={len(val_idx)}) ---")
    
            fold_model = ProtoSSMv2(
                d_input=emb_files.shape[2],
                d_model=ssm_cfg["d_model"],
                d_state=ssm_cfg["d_state"],
                n_ssm_layers=ssm_cfg["n_ssm_layers"],
                n_classes=N_CLASSES,
                n_windows=N_WINDOWS,
                dropout=ssm_cfg["dropout"],
                n_sites=ssm_cfg["n_sites"],
                meta_dim=ssm_cfg["meta_dim"],
                use_cross_attn=ssm_cfg.get("use_cross_attn", True),
                cross_attn_heads=ssm_cfg.get("cross_attn_heads", 4),
            ).to(DEVICE)
    
            # Initialize prototypes
            emb_flat_fold = emb_files[train_idx].reshape(-1, emb_files.shape[2])
            labels_flat_fold = labels_files[train_idx].reshape(-1, N_CLASSES)
            fold_model.init_prototypes_from_data(
                torch.tensor(emb_flat_fold, dtype=torch.float32),
                torch.tensor(labels_flat_fold, dtype=torch.float32)
            )
            fold_model.init_family_head(n_families, class_to_family)
    
            # Train on fold
            fold_model, fold_hist = train_proto_ssm_single(
                fold_model,
                emb_files[train_idx], logits_files[train_idx], labels_files[train_idx].astype(np.float32),
                site_ids_train=site_ids_all[train_idx], hours_train=hours_all[train_idx],
                emb_val=emb_files[val_idx], logits_val=logits_files[val_idx],
                labels_val=labels_files[val_idx].astype(np.float32),
                site_ids_val=site_ids_all[val_idx], hours_val=hours_all[val_idx],
                file_families_train=file_families[train_idx],
                file_families_val=file_families[val_idx],
                cfg=cfg, verbose=verbose,
            )
    
            # OOF predictions with TTA
            fold_model.eval()
            tta_shifts = CFG.get("tta_shifts", [0])
            if len(tta_shifts) > 1:
                oof_preds[val_idx] = temporal_shift_tta(
                    emb_files[val_idx], logits_files[val_idx], fold_model,
                    site_ids_all[val_idx], hours_all[val_idx], shifts=tta_shifts
                )
            else:
                with torch.no_grad():
                    val_emb = torch.tensor(emb_files[val_idx], dtype=torch.float32)
                    val_logits = torch.tensor(logits_files[val_idx], dtype=torch.float32)
                    val_sites = torch.tensor(site_ids_all[val_idx], dtype=torch.long)
                    val_hours = torch.tensor(hours_all[val_idx], dtype=torch.long)
                    val_out, _, _ = fold_model(val_emb, val_logits, site_ids=val_sites, hours=val_hours)
                    oof_preds[val_idx] = val_out.numpy()
    
            fold_alphas.append(torch.sigmoid(fold_model.fusion_alpha).detach().numpy().copy())
            fold_histories.append(fold_hist)
    
        return oof_preds, fold_histories, fold_alphas
    
    
    def optimize_ensemble_weight(oof_proto_flat, oof_mlp_flat, y_true_flat):
        """Grid search over blend weights to find optimal ProtoSSM ensemble weight."""
        weights = np.arange(0.0, 1.05, 0.05)
        results = []
    
        for w in weights:
            blended = w * oof_proto_flat + (1.0 - w) * oof_mlp_flat
            try:
                auc = macro_auc_skip_empty(y_true_flat, blended)
            except Exception:
                auc = 0.0
            results.append((w, auc))
    
        best_w, best_auc = max(results, key=lambda x: x[1])
        return best_w, best_auc, results
    
    
    print("ProtoSSM v4 training functions defined (with mixup, focal loss, SWA, TTA).")
    
    # Cell 10 ? Probe tuning (train mode only)
    grid_results = None
    BEST_PROBE = None
    
    if CFG["run_probe_check"]:
        probe_result = run_oof_embedding_probe(
            scores_raw=scores_full_raw,
            emb=emb_full,
            meta_df=meta_full,
            y_true=Y_FULL,
            pca_dim=64,
            min_pos=8,
            C=0.25,
            alpha=0.5,
        )
    
        print(f"Honest OOF baseline AUC: {probe_result['score_base']:.6f}")
        print(f"Honest OOF embedding-probe AUC: {probe_result['score_final']:.6f}")
        print(f"Delta: {probe_result['score_final'] - probe_result['score_base']:.6f}")
    
        modeled_classes = np.where(probe_result["modeled_counts"] > 0)[0]
        print("Modeled classes:", len(modeled_classes))
        print([PRIMARY_LABELS[i] for i in modeled_classes[:20]])
    
    if CFG["run_probe_grid"]:
        param_grid = [
            {"pca_dim": 32, "min_pos": 8,  "C": 0.25, "alpha": 0.4},
            {"pca_dim": 64, "min_pos": 8,  "C": 0.25, "alpha": 0.4},
            {"pca_dim": 64, "min_pos": 8,  "C": 0.25, "alpha": 0.5},
            {"pca_dim": 64, "min_pos": 12, "C": 0.25, "alpha": 0.4},
            {"pca_dim": 96, "min_pos": 8,  "C": 0.25, "alpha": 0.4},
            {"pca_dim": 64, "min_pos": 8,  "C": 0.50, "alpha": 0.4},
        ]
    
        results = []
        for params in tqdm(param_grid, desc="Probe grid", disable=not CFG["verbose"]):
            out = run_oof_embedding_probe(
                scores_raw=scores_full_raw,
                emb=emb_full,
                meta_df=meta_full,
                y_true=Y_FULL,
                pca_dim=params["pca_dim"],
                min_pos=params["min_pos"],
                C=params["C"],
                alpha=params["alpha"],
            )
            results.append({
                **params,
                "baseline_oof_auc": out["score_base"],
                "probe_oof_auc": out["score_final"],
                "delta": out["score_final"] - out["score_base"],
                "n_modeled_classes": int((out["modeled_counts"] > 0).sum()),
            })
    
        grid_results = pd.DataFrame(results).sort_values("probe_oof_auc", ascending=False).reset_index(drop=True)
        display(grid_results)
    
        BEST_PROBE = {
            "pca_dim": int(grid_results.iloc[0]["pca_dim"]),
            "min_pos": int(grid_results.iloc[0]["min_pos"]),
            "C": float(grid_results.iloc[0]["C"]),
            "alpha": float(grid_results.iloc[0]["alpha"]),
        }
    
        # Save best params for future freezing
        best_probe_path = CFG["full_cache_work_dir"] / "best_probe_params.json"
        best_probe_path.write_text(json.dumps(BEST_PROBE, indent=2))
        print("Saved best probe params to:", best_probe_path)
    
    else:
        BEST_PROBE = CFG["frozen_best_probe"]
        print("Using frozen BEST_PROBE in submit mode:")
        print(BEST_PROBE)
    
    if grid_results is not None:
        grid_results.to_csv(CFG["full_cache_work_dir"] / "probe_grid_results.csv", index=False)
    
    # Cell 11 ? Freeze final probe params
    if BEST_PROBE is None:
        BEST_PROBE = CFG["frozen_best_probe"]
    
    print("Final BEST_PROBE =", BEST_PROBE)
    
    # Optional ? rerun best OOF probe once for diagnostics / caching
    BEST_OOF_RESULT = None
    
    if MODE == "train":
        BEST_OOF_RESULT = run_oof_embedding_probe(
            scores_raw=scores_full_raw,
            emb=emb_full,
            meta_df=meta_full,
            y_true=Y_FULL,
            pca_dim=int(BEST_PROBE["pca_dim"]),
            min_pos=int(BEST_PROBE["min_pos"]),
            C=float(BEST_PROBE["C"]),
            alpha=float(BEST_PROBE["alpha"]),
        )
    
        print(f"Honest OOF baseline AUC (BEST_PROBE rerun): {BEST_OOF_RESULT['score_base']:.6f}")
        print(f"Honest OOF probe AUC   (BEST_PROBE rerun): {BEST_OOF_RESULT['score_final']:.6f}")
    
    # Cell 12 ? Fit final prior tables on all labeled soundscapes
    final_prior_tables = fit_prior_tables(sc_clean.reset_index(drop=True), Y_SC)
    
    print("Built final prior tables for inference.")
    print("OOF baseline AUC used for stacker training:", baseline_oof_auc)
    
    # Cell 13 ? Fit embedding scaler + PCA on all trusted full windows
    emb_scaler = StandardScaler()
    emb_full_scaled = emb_scaler.fit_transform(emb_full)
    
    n_comp = min(
        int(BEST_PROBE["pca_dim"]),
        emb_full_scaled.shape[0] - 1,
        emb_full_scaled.shape[1]
    )
    
    emb_pca = PCA(n_components=n_comp)
    Z_FULL = emb_pca.fit_transform(emb_full_scaled).astype(np.float32)
    
    print("emb_full:", emb_full.shape)
    print("Z_FULL:", Z_FULL.shape)
    print("Explained variance ratio sum:", emb_pca.explained_variance_ratio_.sum())
    
    # Instantiate and train ProtoSSM v4
    
    # --- Step 1: Reshape to file-level ---
    emb_files, file_list = reshape_to_files(emb_full, meta_full)
    logits_files, _ = reshape_to_files(scores_full_raw, meta_full)
    labels_files, _ = reshape_to_files(Y_FULL, meta_full)
    
    print(f"Reshaped to file-level: emb={emb_files.shape}, logits={logits_files.shape}, labels={labels_files.shape}")
    print(f"Files: {len(file_list)}")
    
    # --- Step 2: Build taxonomy groups, site mapping, file metadata ---
    n_families, class_to_family, fam_to_idx = build_taxonomy_groups(taxonomy, PRIMARY_LABELS)
    print(f"Taxonomic groups: {n_families}")
    
    site_to_idx, n_sites_mapped = build_site_mapping(meta_full)
    n_sites_cfg = CFG["proto_ssm"]["n_sites"]
    print(f"Sites mapped: {n_sites_mapped} (capped to {n_sites_cfg})")
    
    site_ids_all, hours_all = get_file_metadata(meta_full, file_list, site_to_idx, n_sites_cfg)
    
    # Build per-file family labels (multi-hot)
    file_families = np.zeros((len(file_list), n_families), dtype=np.float32)
    for fi in range(len(file_list)):
        active_classes = np.where(labels_files[fi].sum(axis=0) > 0)[0]
        for ci in active_classes:
            file_families[fi, class_to_family[ci]] = 1.0
    
    # --- OOF Cross-Validation (TRAIN MODE ONLY) ---
    ENSEMBLE_WEIGHT_PROTO = 0.5  # default, overridden by OOF in train mode
    oof_proto_flat = None
    fold_alphas = []
    
    if MODE == "train":
        file_groups = np.array([f.split("_")[3] if len(f.split("_")) > 3 else f for f in file_list])
        print(f"File groups for OOF: {len(set(file_groups))} unique groups: {sorted(set(file_groups))}")
    
        t0_oof = time.time()
        oof_proto_preds, fold_histories, fold_alphas = run_proto_ssm_oof(
            emb_files, logits_files, labels_files,
            site_ids_all, hours_all,
            file_families, file_groups,
            n_families, class_to_family,
            cfg=CFG["proto_ssm_train"],
            verbose=CFG["verbose"],
        )
        oof_time = time.time() - t0_oof
        print(f"\nOOF cross-validation time: {oof_time:.1f}s")
    
        oof_proto_flat = oof_proto_preds.reshape(-1, N_CLASSES)
        y_flat = labels_files.reshape(-1, N_CLASSES).astype(np.float32)
    
        per_class_auc_proto = {}
        for ci in range(N_CLASSES):
            if y_flat[:, ci].sum() > 0 and y_flat[:, ci].sum() < len(y_flat):
                try:
                    per_class_auc_proto[ci] = roc_auc_score(y_flat[:, ci], oof_proto_flat[:, ci])
                except Exception:
                    pass
    
        overall_oof_auc_proto = macro_auc_skip_empty(y_flat, oof_proto_flat)
        print(f"ProtoSSM OOF macro AUC: {overall_oof_auc_proto:.4f}")
    
        LOGS["oof_auc_proto"] = overall_oof_auc_proto
        LOGS["per_class_auc_proto"] = {PRIMARY_LABELS[k]: v for k, v in per_class_auc_proto.items()}
        LOGS["oof_time"] = oof_time
    else:
        print("Submit mode: skipping OOF cross-validation")
    
    # --- Train final model on ALL data ---
    ssm_cfg = CFG["proto_ssm"]
    model = ProtoSSMv2(
        d_input=emb_full.shape[1],
        d_model=ssm_cfg["d_model"],
        d_state=ssm_cfg["d_state"],
        n_ssm_layers=ssm_cfg["n_ssm_layers"],
        n_classes=N_CLASSES,
        n_windows=N_WINDOWS,
        dropout=ssm_cfg["dropout"],
        n_sites=ssm_cfg["n_sites"],
        meta_dim=ssm_cfg["meta_dim"],
        use_cross_attn=ssm_cfg.get("use_cross_attn", True),
        cross_attn_heads=ssm_cfg.get("cross_attn_heads", 4),
    ).to(DEVICE)
    
    emb_flat_tensor = torch.tensor(emb_full, dtype=torch.float32)
    labels_flat_tensor = torch.tensor(Y_FULL, dtype=torch.float32)
    model.init_prototypes_from_data(emb_flat_tensor, labels_flat_tensor)
    model.init_family_head(n_families, class_to_family)
    
    print(f"\nProtoSSM v4 parameters: {model.count_parameters():,}")
    
    t0_final = time.time()
    model, train_history = train_proto_ssm_single(
        model,
        emb_files, logits_files, labels_files.astype(np.float32),
        site_ids_train=site_ids_all, hours_train=hours_all,
        cfg=CFG["proto_ssm_train"],
        verbose=True,
    )
    train_time = time.time() - t0_final
    print(f"Final model training time: {train_time:.1f}s")
    
    with torch.no_grad():
        final_alphas = torch.sigmoid(model.fusion_alpha).numpy()
        print(f"Fusion alpha: mean={final_alphas.mean():.4f} min={final_alphas.min():.4f} max={final_alphas.max():.4f}")
    
    # --- Train MLP probes ---
    PROBE_CLASS_IDX = np.where(Y_FULL.sum(axis=0) >= int(CFG["frozen_best_probe"]["min_pos"]))[0].astype(np.int32)
    
    probe_models = {}
    for cls_idx in tqdm(PROBE_CLASS_IDX, desc="Training MLP probes", disable=not CFG["verbose"]):
        y = Y_FULL[:, cls_idx]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        X_cls = build_class_features(
            Z_FULL,
            raw_col=scores_full_raw[:, cls_idx],
            prior_col=oof_prior[:, cls_idx],
            base_col=oof_base[:, cls_idx],
        )
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        if n_pos > 0 and n_neg > n_pos:
            repeat = max(1, n_neg // n_pos)
            pos_idx = np.where(y == 1)[0]
            X_bal = np.vstack([X_cls, np.tile(X_cls[pos_idx], (repeat, 1))])
            y_bal = np.concatenate([y, np.ones(len(pos_idx) * repeat, dtype=y.dtype)])
        else:
            X_bal, y_bal = X_cls, y
        clf = MLPClassifier(**CFG["mlp_params"])
        clf.fit(X_bal, y_bal)
        probe_models[cls_idx] = clf
    
    print(f"MLP probes trained: {len(probe_models)}")
    
    # --- Optimize ensemble weight (TRAIN MODE ONLY) ---
    if MODE == "train" and oof_proto_flat is not None:
        oof_mlp_flat = oof_base.copy()
        for cls_idx, clf in probe_models.items():
            X_cls = build_class_features(
                Z_FULL,
                raw_col=scores_full_raw[:, cls_idx],
                prior_col=oof_prior[:, cls_idx],
                base_col=oof_base[:, cls_idx],
            )
            if hasattr(clf, "predict_proba"):
                prob = clf.predict_proba(X_cls)[:, 1].astype(np.float32)
                pred = np.log(prob + 1e-7) - np.log(1 - prob + 1e-7)
            else:
                pred = clf.decision_function(X_cls).astype(np.float32)
            alpha_probe = float(CFG["frozen_best_probe"]["alpha"])
            oof_mlp_flat[:, cls_idx] = (1.0 - alpha_probe) * oof_base[:, cls_idx] + alpha_probe * pred
    
        y_flat = labels_files.reshape(-1, N_CLASSES).astype(np.float32)
        best_w, best_auc, weight_results = optimize_ensemble_weight(oof_proto_flat, oof_mlp_flat, y_flat)
        ENSEMBLE_WEIGHT_PROTO = best_w
    
        mlp_only_auc = macro_auc_skip_empty(y_flat, oof_mlp_flat)
        print(f"\n=== Ensemble Optimization ===")
        print(f"Best ProtoSSM weight: {ENSEMBLE_WEIGHT_PROTO:.2f}")
        print(f"Best ensemble OOF AUC: {best_auc:.4f}")
        print(f"MLP-only OOF AUC: {mlp_only_auc:.4f}")
    
        for w, auc in weight_results:
            marker = " <-- best" if abs(w - best_w) < 0.01 else ""
            print(f"  w={w:.2f}: AUC={auc:.4f}{marker}")
    
        LOGS["ensemble_weight"] = ENSEMBLE_WEIGHT_PROTO
        LOGS["ensemble_auc"] = best_auc
        LOGS["mlp_only_auc"] = mlp_only_auc
    else:
        print(f"\nUsing default ensemble weight: ProtoSSM={ENSEMBLE_WEIGHT_PROTO:.2f}")
    
    LOGS["train_time_final"] = train_time
    LOGS["n_probe_models"] = len(probe_models)
    
    if fold_alphas:
        mean_alphas = np.stack(fold_alphas).mean(axis=0)
        print(f"\nFusion alpha (mean across folds):")
        print(f"  ProtoSSM-dominant (alpha>0.5): {(mean_alphas > 0.5).sum()} classes")
        print(f"  Perch-dominant (alpha<=0.5): {(mean_alphas <= 0.5).sum()} classes")
    
    # ?????????????????????????????????????????????????????????????????????????????
    # ?[IMPORTANT]ResidualSSM exists, skip training.???????????????????????????
    # * If you want to perform training, please set it to None.              
    # ?????????????????????????????????????????????????????????????????????????????
    # ResidualSSM_PATH = "ResidualSSM/models/residual_ssm_best.pt"
    ResidualSSM_PATH = "/kaggle/input/datasets/hideyukizushi/sgkfk-202604041716/ResidualSSM/models/residual_ssm_best.pt"
    
    # Residual SSM: second-pass boosting on first-pass errors
    # Wall-time safety: skip if > 4 min elapsed (leave max time for test inference)
    _wall_min = (time.time() - _WALL_START) / 60.0
    print(f"Wall time: {_wall_min:.1f} min")
    
    res_model = None
    CORRECTION_WEIGHT = 0.0
    
    # ??????? Fix 1: ???????????Submit????????????????? ???????
    class ResidualSSM(nn.Module):
        # Lightweight SSM that takes first-pass scores + embeddings and predicts corrections.
        # Architecture: project(concat(emb, first_pass)) -> 1-layer BiSSM -> linear head
    
        def __init__(self, d_input=1536, d_scores=234, d_model=64, d_state=8,
                     n_classes=234, n_windows=12, dropout=0.1, n_sites=20, meta_dim=8):
            super().__init__()
            self.d_model = d_model
            self.n_classes = n_classes
    
            # Project embeddings + first-pass scores
            self.input_proj = nn.Sequential(
                nn.Linear(d_input + d_scores, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
    
            # Metadata
            self.site_emb = nn.Embedding(n_sites, meta_dim)
            self.hour_emb = nn.Embedding(24, meta_dim)
            self.meta_proj = nn.Linear(2 * meta_dim, d_model)
    
            # Positional encoding
            self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)
    
            # Single bidirectional SSM layer (lightweight)
            self.ssm_fwd = SelectiveSSM(d_model, d_state)
            self.ssm_bwd = SelectiveSSM(d_model, d_state)
            self.ssm_merge = nn.Linear(2 * d_model, d_model)
            self.ssm_norm = nn.LayerNorm(d_model)
            self.ssm_drop = nn.Dropout(dropout)
    
            # Output: per-class correction (additive)
            self.output_head = nn.Linear(d_model, n_classes)
    
            # Initialize output near zero (corrections start small)
            nn.init.zeros_(self.output_head.weight)
            nn.init.zeros_(self.output_head.bias)
    
        def forward(self, emb, first_pass_scores, site_ids=None, hours=None):
            # emb: (B, T, d_input), first_pass_scores: (B, T, n_classes)
            B, T, _ = emb.shape
    
            # Concatenate embeddings with first-pass scores
            x = torch.cat([emb, first_pass_scores], dim=-1)  # (B, T, d_input + d_scores)
            h = self.input_proj(x)
    
            # Add metadata
            if site_ids is not None and hours is not None:
                site_e = self.site_emb(site_ids.clamp(0, self.site_emb.num_embeddings - 1))
                hour_e = self.hour_emb(hours.clamp(0, 23))
                meta = self.meta_proj(torch.cat([site_e, hour_e], dim=-1))
                h = h + meta.unsqueeze(1)
    
            h = h + self.pos_enc[:, :T, :]
    
            # Bidirectional SSM
            residual = h
            h_f = self.ssm_fwd(h)
            h_b = self.ssm_bwd(h.flip(1)).flip(1)
            h = self.ssm_merge(torch.cat([h_f, h_b], dim=-1))
            h = self.ssm_drop(h)
            h = self.ssm_norm(h + residual)
    
            # Output correction
            correction = self.output_head(h)  # (B, T, n_classes)
            return correction
    
        def count_parameters(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
    # ????????????????????????????????????????????????????????????????????????
    
    if ResidualSSM_PATH is not None:
        print("Loading pretrained ResidualSSM...")
        load_res_path = CFG.get("pretrained_residual_path", ResidualSSM_PATH)
        
        if os.path.exists(load_res_path):
            res_cfg = CFG["residual_ssm"]
            res_model = ResidualSSM(
                d_input=emb_full.shape[1],
                d_scores=N_CLASSES,
                d_model=res_cfg["d_model"],
                d_state=res_cfg["d_state"],
                n_classes=N_CLASSES,
                n_windows=N_WINDOWS,
                dropout=res_cfg["dropout"],
                n_sites=CFG["proto_ssm"]["n_sites"],
                meta_dim=8,
            ).to(DEVICE)
            
            res_model.load_state_dict(torch.load(load_res_path, map_location=DEVICE))
            res_model.eval()
            CORRECTION_WEIGHT = res_cfg["correction_weight"]
            print(f"? [Load] Loaded ResidualSSM from {load_res_path}")
            LOGS["residual_ssm"] = {"skipped": False, "mode": "submit", "loaded_from": load_res_path}
        else:
            print(f"?? WARNING: Pre-trained ResidualSSM not found at {load_res_path}. Skipping correction.")
            LOGS["residual_ssm"] = {"skipped": True, "mode": "submit", "reason": "weights_not_found"}
        # ????????????????????????????????????????????????????????????????????????
    
    elif _wall_min < 120.0:
        print("???????????????????????????????????")
        print("???????Training ResidualSSM...")
        print("???????????????????????????????????")
        
        # --- Train ResidualSSM on first-pass errors ---
        
        # Step 1: Compute first-pass scores on training data
        model.eval()
        with torch.no_grad():
            emb_train_t = torch.tensor(emb_files, dtype=torch.float32)
            logits_train_t = torch.tensor(logits_files, dtype=torch.float32)
            site_train_t = torch.tensor(site_ids_all, dtype=torch.long)
            hour_train_t = torch.tensor(hours_all, dtype=torch.long)
        
            proto_train_out, _, _ = model(emb_train_t, logits_train_t,
                                           site_ids=site_train_t, hours=hour_train_t)
            proto_train_scores = proto_train_out.numpy()  # (n_files, 12, 234)
        
        # MLP probe scores on training data (flat)
        mlp_train_scores_flat = np.zeros_like(scores_full_raw, dtype=np.float32)
        
        # Get prior-fused base for MLP
        train_base_scores, train_prior_scores = fuse_scores_with_tables(
            scores_full_raw,
            sites=meta_full["site"].to_numpy(),
            hours=meta_full["hour_utc"].to_numpy(),
            tables=final_prior_tables,
        )
        mlp_train_scores_flat = train_base_scores.copy()
        
        # for cls_idx, clf in probe_models.items():
        #     X_cls = build_class_features(
        #         Z_FULL,
        #         raw_col=scores_full_raw[:, cls_idx],
        #         prior_col=train_prior_scores[:, cls_idx],
        #         base_col=train_base_scores[:, cls_idx],
        #     )
        #     if hasattr(clf, "predict_proba"):
        #         prob = clf.predict_proba(X_cls)[:, 1].astype(np.float32)
        #         pred = np.log(prob + 1e-7) - np.log(1 - prob + 1e-7)
        #     else:
        #         pred = clf.decision_function(X_cls).astype(np.float32)
        #     alpha_p = float(CFG["frozen_best_probe"]["alpha"])
        #     mlp_train_scores_flat[:, cls_idx] = (1 - alpha_p) * train_base_scores[:, cls_idx] + alpha_p * pred
    
        # === [Update]Processing in one line using a tensorization function ===
        alpha_p = float(CFG["frozen_best_probe"]["alpha"])
        mlp_train_scores_flat = get_vectorized_mlp_scores(
            Z_FULL, scores_full_raw, train_prior_scores, train_base_scores, 
            probe_models, alpha_p, n_windows=N_WINDOWS, device=DEVICE
        )
        
        # Reshape MLP scores to file-level
        mlp_train_scores_files, _ = reshape_to_files(mlp_train_scores_flat, meta_full)
        
        # First-pass ensemble (same formula as test-time)
        first_pass_files = (
            ENSEMBLE_WEIGHT_PROTO * proto_train_scores +
            (1 - ENSEMBLE_WEIGHT_PROTO) * mlp_train_scores_files
        ).astype(np.float32)
        
        # Step 2: Compute residuals (what the first pass got wrong)
        # Target: Y_FULL reshaped to files. Residual = target - sigmoid(first_pass)
        labels_float = labels_files.astype(np.float32)
        first_pass_probs = 1.0 / (1.0 + np.exp(-first_pass_files))
        residuals = labels_float - first_pass_probs  # in [-1, 1]
        
        print(f"First-pass training scores: {first_pass_files.shape}")
        print(f"Residuals: mean={residuals.mean():.4f}, std={residuals.std():.4f}, "
              f"abs_mean={np.abs(residuals).mean():.4f}")
        
        # Step 3: Train ResidualSSM
        res_cfg = CFG["residual_ssm"]
        res_model = ResidualSSM(
            d_input=emb_full.shape[1],
            d_scores=N_CLASSES,
            d_model=res_cfg["d_model"],
            d_state=res_cfg["d_state"],
            n_classes=N_CLASSES,
            n_windows=N_WINDOWS,
            dropout=res_cfg["dropout"],
            n_sites=CFG["proto_ssm"]["n_sites"],
            meta_dim=8,
        ).to(DEVICE)
        
        print(f"ResidualSSM parameters: {res_model.count_parameters():,}")
        
        # Train with MSE loss on residuals
        n_files = len(file_list)
        n_val = max(1, int(n_files * 0.15))
        perm = torch.randperm(n_files, generator=torch.Generator().manual_seed(123))
        val_i = perm[:n_val].numpy()
        train_i = perm[n_val:].numpy()
        
        emb_tr = torch.tensor(emb_files[train_i], dtype=torch.float32)
        fp_tr = torch.tensor(first_pass_files[train_i], dtype=torch.float32)
        res_tr = torch.tensor(residuals[train_i], dtype=torch.float32)
        site_tr = torch.tensor(site_ids_all[train_i], dtype=torch.long)
        hour_tr = torch.tensor(hours_all[train_i], dtype=torch.long)
        
        emb_va = torch.tensor(emb_files[val_i], dtype=torch.float32)
        fp_va = torch.tensor(first_pass_files[val_i], dtype=torch.float32)
        res_va = torch.tensor(residuals[val_i], dtype=torch.float32)
        site_va = torch.tensor(site_ids_all[val_i], dtype=torch.long)
        hour_va = torch.tensor(hours_all[val_i], dtype=torch.long)
        
        optimizer = torch.optim.AdamW(res_model.parameters(), lr=res_cfg["lr"], weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=res_cfg["lr"],
            epochs=res_cfg["n_epochs"], steps_per_epoch=1,
            pct_start=0.1, anneal_strategy='cos'
        )
        
        best_val_loss = float('inf')
        best_state = None
        wait = 0
        
        t0_res = time.time()
        for epoch in range(res_cfg["n_epochs"]):
            res_model.train()
            correction = res_model(emb_tr, fp_tr, site_ids=site_tr, hours=hour_tr)
            loss = F.mse_loss(correction, res_tr)
        
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(res_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
            res_model.eval()
            with torch.no_grad():
                val_corr = res_model(emb_va, fp_va, site_ids=site_va, hours=hour_va)
                val_loss = F.mse_loss(val_corr, res_va)
        
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                best_state = {k: v.clone() for k, v in res_model.state_dict().items()}
                wait = 0
            else:
                wait += 1
        
            if (epoch + 1) % 20 == 0:
                print(f"  ResidualSSM epoch {epoch+1}: train={loss.item():.6f} val={val_loss.item():.6f} wait={wait}")
        
            if wait >= res_cfg["patience"]:
                print(f"  ResidualSSM early stop at epoch {epoch+1}")
                break
        
        if best_state is not None:
            res_model.load_state_dict(best_state)
        
        res_time = time.time() - t0_res
        print(f"ResidualSSM training time: {res_time:.1f}s")
        print(f"Best val MSE: {best_val_loss:.6f}")
        
        # ??????? Fix 3: Save?? (?????????????????) ???????
        save_res_path = CFG.get("residual_model_path", "ResidualSSM/models/residual_ssm_best.pt")
        os.makedirs(os.path.dirname(save_res_path) or ".", exist_ok=True)
        torch.save(res_model.state_dict(), save_res_path)
        print(f"? [Save] Saved best ResidualSSM model to {save_res_path}")
        # ????????????????????????????????????????????????????????????????????
        
        # Verify correction magnitude
        res_model.eval()
        with torch.no_grad():
            all_corr = res_model(emb_train_t, torch.tensor(first_pass_files, dtype=torch.float32),
                                 site_ids=site_train_t, hours=hour_train_t)
            corr_np = all_corr.numpy()
            print(f"Correction magnitude: mean_abs={np.abs(corr_np).mean():.4f}, max={np.abs(corr_np).max():.4f}")
        
        CORRECTION_WEIGHT = res_cfg["correction_weight"]
        print(f"Correction weight: {CORRECTION_WEIGHT}")
        LOGS["residual_ssm"] = {
            "params": res_model.count_parameters(),
            "train_time": res_time,
            "best_val_mse": best_val_loss,
            "correction_mean_abs": float(np.abs(corr_np).mean()),
            "correction_weight": CORRECTION_WEIGHT,
        }
        
    else:
        print("SKIPPED ResidualSSM (wall time safety)")
        LOGS["residual_ssm"] = {"skipped": True, "wall_min": _wall_min}
    
    # Cell 15 ? Diagnostics
    if MODE == "train":
        if grid_results is not None:
            best_row = grid_results.iloc[0]
            print(f"Best honest OOF probe AUC: {best_row['probe_oof_auc']:.6f}")
            print(f"Delta over honest OOF baseline: {best_row['delta']:.6f}")
    else:
        print("Skipping train diagnostics in submit mode.")
    
    # Cell 16 ? Infer Perch on hidden test with embeddings
    test_paths = sorted((BASE / "test_soundscapes").glob("*.ogg"))
    
    if len(test_paths) == 0:
        print(f"Hidden test not mounted. Dry-run on first {CFG['dryrun_n_files']} train soundscapes.")
        test_paths = sorted((BASE / "train_soundscapes").glob("*.ogg"))[:CFG["dryrun_n_files"]]
    else:
        print(f"Hidden test files: {len(test_paths)}")
    
    # [MODIFIED - Opsi 3] Gunakan proxy_reduce terbaik dari grid search (bukan hardcode "max")
    meta_test, scores_test_raw, emb_test = infer_perch_with_embeddings(
        test_paths,
        batch_files=CFG["batch_files"],
        verbose=CFG["verbose"],
        proxy_reduce=CFG["proxy_reduce"],  # hasil grid search, default "max"
    )
    print(f"proxy_reduce used for test inference: {CFG['proxy_reduce']!r}")
    
    print("meta_test:", meta_test.shape)
    print("scores_test_raw:", scores_test_raw.shape)
    print("emb_test:", emb_test.shape)
    
    
    # Score Fusion: ProtoSSM v4 + MLP Probes + Priors + TTA (OOF-optimized weight)
    
    # --- Step 1: ProtoSSM v4 inference on test with TTA ---
    emb_test_files, test_file_list = reshape_to_files(emb_test, meta_test)
    logits_test_files, _ = reshape_to_files(scores_test_raw, meta_test)
    
    # Build test metadata
    test_site_ids, test_hours = get_file_metadata(meta_test, test_file_list, site_to_idx, CFG["proto_ssm"]["n_sites"])
    
    emb_test_tensor = torch.tensor(emb_test_files, dtype=torch.float32)
    logits_test_tensor = torch.tensor(logits_test_files, dtype=torch.float32)
    test_site_tensor = torch.tensor(test_site_ids, dtype=torch.long)
    test_hour_tensor = torch.tensor(test_hours, dtype=torch.long)
    
    # V16: TTA ? average predictions from shifted temporal sequences
    model.eval()
    tta_shifts = CFG.get("tta_shifts", [0])
    if len(tta_shifts) > 1:
        print(f"Running TTA with shifts: {tta_shifts}")
        proto_scores = temporal_shift_tta(
            emb_test_files, logits_test_files, model,
            test_site_ids, test_hours, shifts=tta_shifts
        )
    else:
        with torch.no_grad():
            proto_out, _, h_test = model(emb_test_tensor, logits_test_tensor,
                                          site_ids=test_site_tensor, hours=test_hour_tensor)
            proto_scores = proto_out.numpy()
    
    # Flatten back to (n_rows, n_classes)
    proto_scores_flat = proto_scores.reshape(-1, N_CLASSES).astype(np.float32)
    
    print(f"ProtoSSM v4 test scores: {proto_scores_flat.shape}")
    print(f"Score range: {proto_scores_flat.min():.3f} to {proto_scores_flat.max():.3f}")
    
    # --- Step 2: Prior-fused base scores ---
    test_base_scores, test_prior_scores = fuse_scores_with_tables(
        scores_test_raw,
        sites=meta_test["site"].to_numpy(),
        hours=meta_test["hour_utc"].to_numpy(),
        tables=final_prior_tables,
    )
    
    # --- Step 3: MLP probe scores ---
    emb_test_scaled = emb_scaler.transform(emb_test)
    Z_TEST = emb_pca.transform(emb_test_scaled).astype(np.float32)
    
    mlp_scores = test_base_scores.copy()
    
    # for cls_idx, clf in probe_models.items():
    #     X_cls_test = build_class_features(
    #         Z_TEST,
    #         raw_col=scores_test_raw[:, cls_idx],
    #         prior_col=test_prior_scores[:, cls_idx],
    #         base_col=test_base_scores[:, cls_idx],
    #     )
    
    #     if hasattr(clf, "predict_proba"):
    #         prob = clf.predict_proba(X_cls_test)[:, 1].astype(np.float32)
    #         pred = np.log(prob + 1e-7) - np.log(1 - prob + 1e-7)
    #     else:
    #         pred = clf.decision_function(X_cls_test).astype(np.float32)
    
    #     alpha = float(CFG["frozen_best_probe"]["alpha"])
    #     mlp_scores[:, cls_idx] = (1.0 - alpha) * test_base_scores[:, cls_idx] + alpha * pred
    
    # === Processing in one line using a tensorization function ===
    alpha_p = float(CFG["frozen_best_probe"]["alpha"])
    mlp_scores = get_vectorized_mlp_scores(
        Z_TEST, scores_test_raw, test_prior_scores, test_base_scores, 
        probe_models, alpha_p, n_windows=N_WINDOWS, device=DEVICE
    )
    
    # --- Step 4: Ensemble fusion with OOF-optimized weight ---
    print(f"\nUsing OOF-optimized ensemble weight: {ENSEMBLE_WEIGHT_PROTO:.2f}")
    
    final_test_scores = (
        ENSEMBLE_WEIGHT_PROTO * proto_scores_flat +
        (1.0 - ENSEMBLE_WEIGHT_PROTO) * mlp_scores
    ).astype(np.float32)
    
    # --- Step 5: Residual SSM correction (second pass) ---
    if res_model is not None and CORRECTION_WEIGHT > 0:
        first_pass_test_files, _ = reshape_to_files(final_test_scores, meta_test)
        first_pass_test_t = torch.tensor(first_pass_test_files, dtype=torch.float32)
    
        res_model.eval()
        with torch.no_grad():
            test_correction = res_model(
                emb_test_tensor, first_pass_test_t,
                site_ids=test_site_tensor, hours=test_hour_tensor
            ).numpy()
    
        test_correction_flat = test_correction.reshape(-1, N_CLASSES).astype(np.float32)
    
        print(f"\nResidual correction: mean_abs={np.abs(test_correction_flat).mean():.4f}, "
              f"max={np.abs(test_correction_flat).max():.4f}")
    
        final_test_scores = final_test_scores + CORRECTION_WEIGHT * test_correction_flat
        print(f"Final scores (after residual): range [{final_test_scores.min():.3f}, {final_test_scores.max():.3f}]")
    else:
        print("\nResidual correction: SKIPPED")
    
    print(f"Final scores: {final_test_scores.shape}")
    
    # --- Logging ---
    test_logs = {}
    window_scores = proto_scores.reshape(-1, N_WINDOWS, N_CLASSES).mean(axis=(0, 2))
    test_logs["window_position_scores"] = window_scores.tolist()
    print(f"\nWindow position mean scores: {[f'{s:.3f}' for s in window_scores]}")
    
    if hasattr(model, 'class_to_family'):
        taxon_scores = defaultdict(list)
        idx_to_fam = {v: k for k, v in fam_to_idx.items()}
        for ci in range(N_CLASSES):
            fam_idx = class_to_family[ci]
            fam_name = idx_to_fam.get(fam_idx, f"group_{fam_idx}")
            taxon_scores[fam_name].append(float(proto_scores_flat[:, ci].mean()))
    
        test_logs["taxon_mean_scores"] = {k: float(np.mean(v)) for k, v in taxon_scores.items()}
        for k, v in sorted(taxon_scores.items(), key=lambda x: -np.mean(x[1]))[:5]:
            print(f"  {k}: mean_score={np.mean(v):.4f} (n_classes={len(v)})")
    
    with torch.no_grad():
        p_norm = F.normalize(model.prototypes, dim=-1)
        cos_sim = torch.matmul(p_norm, p_norm.T)
        cos_sim.fill_diagonal_(0)
        top_sims = cos_sim.max(dim=1)[0].numpy()
        test_logs["prototype_max_similarity"] = {
            "mean": float(top_sims.mean()),
            "max": float(top_sims.max()),
            "min": float(top_sims.min()),
        }
        print(f"\nPrototype nearest-neighbor similarity: mean={top_sims.mean():.3f}, max={top_sims.max():.3f}")
    
    
    LOGS["test_inference"] = test_logs
    
    
    
    ## Submission
    ## Temperature scaling and CSV generation.
    
    
    
    # Cell 18 ? V17: Full post-processing pipeline
    
    # V17: Optimize per-class thresholds from OOF (train mode only)
    PER_CLASS_THRESHOLDS = np.full(N_CLASSES, 0.5, dtype=np.float32)
    if MODE == "train" and oof_proto_flat is not None:
        print("Optimizing per-class thresholds from OOF...")
        best_thresholds, best_scores = optimize_per_class_thresholds(
            oof_proto_flat, Y_FULL, n_windows=N_WINDOWS, thresholds=CFG["threshold_grid"]
        )
        PER_CLASS_THRESHOLDS = best_thresholds.astype(np.float32)
        print(f"  Mean threshold: {best_thresholds.mean():.3f}")
        print(f"  Threshold range: [{best_thresholds.min():.2f}, {best_thresholds.max():.2f}]")
        print(f"  Mean F1 (proxy): {best_scores.mean():.3f}")
        
        # Show classes with extreme thresholds
        high_t = np.where(best_thresholds > 0.6)[0]
        low_t = np.where(best_thresholds < 0.4)[0]
        if len(high_t) > 0:
            print(f"  High threshold classes (>0.6): {len(high_t)}")
        if len(low_t) > 0:
            print(f"  Low threshold classes (<0.4): {len(low_t)}")
    else:
        # Submit mode: use default 0.5 thresholds for all classes
        print("Using default per-class thresholds (0.5) for submit mode")
    
    
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
    
    # --- Step 1: Per-taxon temperature scaling ---
    temp_cfg = CFG["temperature"]
    T_AVES = temp_cfg["aves"]
    T_TEXTURE = temp_cfg["texture"]
    
    class_temperatures = np.ones(N_CLASSES, dtype=np.float32) * T_AVES
    for ci, label in enumerate(PRIMARY_LABELS):
        cn = CLASS_NAME_MAP.get(label, "Aves")
        if cn in TEXTURE_TAXA:
            class_temperatures[ci] = T_TEXTURE
    
    print(f"\nPer-taxon temperature: Aves={T_AVES}, Texture={T_TEXTURE}")
    
    scaled_scores = final_test_scores / class_temperatures[None, :]
    probs = sigmoid(scaled_scores)
    
    # --- Step 2: File-level confidence scaling ---
    top_k = CFG.get("file_level_top_k", 0)
    if top_k > 0:
        print(f"Applying file-level confidence scaling (top_k={top_k})")
        probs = file_level_confidence_scale(probs, n_windows=N_WINDOWS, top_k=top_k)
        probs = np.clip(probs, 0.0, 1.0)
    
    # --- Step 3: V17 Rank-aware post-processing ---
    if CFG.get("rank_aware_scale", False):
        power = CFG.get("rank_aware_power", 0.5)
        print(f"Applying rank-aware scaling (power={power})")
        probs = rank_aware_scaling(probs, n_windows=N_WINDOWS, power=power)
        probs = np.clip(probs, 0.0, 1.0)
    
    # --- Step 4: V17 Delta shift smoothing ---
    def adaptive_delta_smooth(probs, n_windows, base_alpha=0.20):
        n_files = probs.shape[0] // n_windows
        result = probs.copy()
        view = result.reshape(n_files, n_windows, -1)
        p_view = probs.reshape(n_files, n_windows, -1)
        for i in range(1, n_windows - 1):
            conf = p_view[:, i, :].max(axis=-1, keepdims=True)
            a = base_alpha * (1.0 - conf)
            neighbor_avg = (p_view[:, i-1, :] + p_view[:, i+1, :]) / 2.0
            view[:, i, :] = (1.0 - a) * p_view[:, i, :] + a * neighbor_avg
        return result.reshape(probs.shape)
    
    alpha = CFG.get("delta_shift_alpha", 0.0)
    if alpha > 0:
        print(f"Applying delta shift smoothing (alpha={alpha})")
        probs = adaptive_delta_smooth(probs, n_windows=N_WINDOWS, base_alpha=alpha)
        probs = np.clip(probs, 0.0, 1.0)
    # --- Step 5: V17 Per-class threshold sharpening ---
    print(f"Applying per-class threshold sharpening...")
    probs = apply_per_class_thresholds(probs, PER_CLASS_THRESHOLDS, n_windows=N_WINDOWS)
    
    # --- Build submission ---
    submission = pd.DataFrame(probs, columns=PRIMARY_LABELS)
    submission.insert(0, "row_id", meta_test["row_id"].values)
    submission[PRIMARY_LABELS] = submission[PRIMARY_LABELS].astype(np.float32)
    
    expected_rows = len(test_paths) * N_WINDOWS
    assert len(submission) == expected_rows, f"Expected {expected_rows}, got {len(submission)}"
    assert submission.columns.tolist() == ["row_id"] + PRIMARY_LABELS
    assert not submission.isna().any().any()
    
    submission.to_csv(_file_name_submission, index=False)
    
    print("\nSaved submission.csv")
    print("Submission shape:", submission.shape)
    print(f"Final score range: {probs.min():.6f} to {probs.max():.6f}")
    print(f"Final mean: {probs.mean():.4f}")
    print(submission.iloc[:3, :8])
    
    # Cell 19 ? Final Diagnostics and Logging
    
    # Save comprehensive logs
    wall_time = time.time() - _WALL_START
    LOGS["wall_time_seconds"] = wall_time
    LOGS["temperature"] = CFG["temperature"]
    LOGS["ensemble_weight_proto"] = ENSEMBLE_WEIGHT_PROTO
    LOGS["n_classes"] = N_CLASSES
    LOGS["n_windows"] = N_WINDOWS
    LOGS["cfg_proto_ssm"] = CFG["proto_ssm"]
    LOGS["cfg_proto_ssm_train"] = {k: v for k, v in CFG["proto_ssm_train"].items() if not isinstance(v, (np.ndarray,))}
    LOGS["v17_improvements"] = [
        "d_model_256", "n_ssm_layers_3", "cross_attention", "mixup", "focal_loss", "swa",
        "per_taxon_temperature", "file_level_scaling", "tta", "rank_aware_scaling",
        "delta_shift_smooth", "per_class_thresholds"
    ]
    LOGS["per_class_thresholds"] = PER_CLASS_THRESHOLDS.tolist()
    
    try:
        with open("/kaggle/working/v17_logs.json", "w") as f:
            json.dump(LOGS, f, indent=2, default=str)
        print("Saved /kaggle/working/v17_logs.json")
    except Exception as e:
        print(f"Warning: could not save logs: {e}")
    
    if MODE == "train":
        print("=== ProtoSSM v5 Training Summary ===")
        print(f"Parameters: {model.count_parameters():,}")
        print(f"d_model: {CFG['proto_ssm']['d_model']}, n_ssm_layers: {CFG['proto_ssm']['n_ssm_layers']}")
        print(f"Wall time: {wall_time:.1f}s")
        print(f"OOF CV time: {LOGS.get('oof_time', 0):.1f}s")
        print(f"Final model training time: {LOGS.get('train_time_final', 0):.1f}s")
        print(f"Final train loss: {train_history['train_loss'][-1]:.4f}")
        print(f"Best val loss: {min(train_history['val_loss']):.4f}")
        print(f"Best val AUC: {max(train_history['val_auc']):.4f}")
    
        print(f"\n=== OOF Results ===")
        print(f"ProtoSSM OOF AUC: {LOGS.get('oof_auc_proto', 0):.4f}")
        print(f"MLP-only OOF AUC: {LOGS.get('mlp_only_auc', 0):.4f}")
        print(f"Ensemble OOF AUC: {LOGS.get('ensemble_auc', 0):.4f}")
        print(f"Optimized ProtoSSM weight: {ENSEMBLE_WEIGHT_PROTO:.2f}")
    
        with torch.no_grad():
            alphas = torch.sigmoid(model.fusion_alpha).numpy()
            high_proto = (alphas > 0.5).sum()
            high_perch = (alphas <= 0.5).sum()
            print(f"\nFusion alpha distribution (final model):")
            print(f"  ProtoSSM-dominant (alpha>0.5): {high_proto} classes")
            print(f"  Perch-dominant (alpha<=0.5): {high_perch} classes")
    
        print(f"\nPer-class calibration bias stats:")
        with torch.no_grad():
            cb = model.class_bias.numpy()
            print(f"  mean={cb.mean():.4f} std={cb.std():.4f} min={cb.min():.4f} max={cb.max():.4f}")
    
        print(f"\nMLP probes: {len(probe_models)} classes")
    
        if "per_class_auc_proto" in LOGS and LOGS["per_class_auc_proto"]:
            sorted_aucs = sorted(LOGS["per_class_auc_proto"].items(), key=lambda x: x[1], reverse=True)
            print(f"\nTop 10 classes by ProtoSSM OOF AUC:")
            for label, auc in sorted_aucs[:10]:
                print(f"  {label}: {auc:.4f}")
            print(f"\nBottom 10 classes by ProtoSSM OOF AUC:")
            for label, auc in sorted_aucs[-10:]:
                print(f"  {label}: {auc:.4f}")
    
        print("\nSubmission probability stats:")
        print(submission.iloc[:, 1:].stack().describe())
    else:
        print("Submit mode completed.")
        print(f"ProtoSSM v5 parameters: {model.count_parameters():,}")
        print(f"Ensemble weight: {ENSEMBLE_WEIGHT_PROTO:.2f}")
        print(f"Wall time: {wall_time:.1f}s")
        print(f"V17 improvements: {LOGS['v17_improvements']}")



# === ## Model_4

# === CELL 5 ===

if 'Model_4' in _ensemble_models:
    
    _file_name_submission = "subm_4.csv"

    import subprocess, sys, os
    from pathlib import Path
    import random
    import numpy as np
    import torch
    
    INPUT_ROOT = Path("/kaggle/input")
    
    def find_wheel(pattern):
        for p in INPUT_ROOT.rglob(pattern):
            return p
        print(f"Wheel not found: {pattern}"); return None  # non-fatal
    ONNX_WHL = Path("/kaggle/input/datasets/hellodave2035/birdclef-perch-models/onnxruntime-1.24.4-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl")
    if ONNX_WHL.exists():
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", str(ONNX_WHL)], check=True)
        print("ONNX Runtime installed")
    
    _w = find_wheel("tensorboard-2.20.0-*.whl")
    if _w:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", str(_w)], check=True)
    _w = find_wheel("tensorflow-2.20.0-*.whl")
    if _w:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", str(_w)], check=True)
    print("TF 2.20 installed")
    
    try:
        import onnxruntime as ort
        _ONNX_AVAILABLE = True
        print("ONNX Runtime available")
    except ImportError:
        _ONNX_AVAILABLE = False
        print("ONNX not available, falling back to TF")
    
    def seed_everything(seed=42):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    seed_everything(4)
    print("Global random seed set to 4")
    
    MODE = "submit"
    assert MODE in {"train", "submit"}
    print("MODE =", MODE)
    
    import os, re, gc, time, warnings
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    warnings.filterwarnings("ignore")
    
    import pandas as pd
    import soundfile as sf
    import tensorflow as tf
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from tqdm.auto import tqdm
    
    tf.experimental.numpy.experimental_enable_numpy_behavior()
    try: tf.config.set_visible_devices([], "GPU")
    except: pass
    
    _WALL_START = time.time()
    
    BASE      = Path("/kaggle/input/competitions/birdclef-2026")
    MODEL_DIR = Path("/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1")
    WORK_DIR  = Path("/kaggle/working/cache")
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    
    SR             = 32_000
    WINDOW_SEC     = 5
    WINDOW_SAMPLES = SR * WINDOW_SEC
    FILE_SAMPLES   = 60 * SR
    N_WINDOWS      = 12
    
    CFG = {
        "batch_files": 16,
        "oof_n_splits": 5   if MODE == "train" else 3,
        "dryrun_n_files": 20 if MODE == "train" else 0,
        "run_oof": MODE == "train",
        "verbose": MODE == "train",
        "proto_ssm_train": {
            "n_epochs":        80  if MODE == "train" else 40,
            "lr":              8e-4,
            "weight_decay":    1e-3,
            "val_ratio":       0.15,
            "patience":        20  if MODE == "train" else 8,
            "pos_weight_cap":  25.0,
            "distill_weight":  0.15,
            "proto_margin":    0.15,
            "label_smoothing": 0.03,
            "oof_n_splits":    5   if MODE == "train" else 3,
            "mixup_alpha":     0.4,
            "focal_gamma":     2.5,
            "swa_start_frac":  0.65,
            "swa_lr":          4e-4,
            "use_cosine_restart": True,
            "restart_period":  20,
        },
        "residual_ssm": {
            "d_model": 128, "d_state": 16, "n_ssm_layers": 2,
            "dropout": 0.1, "correction_weight": 0.35,
            "n_epochs": 40  if MODE == "train" else 20,
            "lr": 8e-4,
            "patience": 12  if MODE == "train" else 6,
        },
        "mlp_params": {
            "hidden_layer_sizes": (256, 128), "activation": "relu",
            "max_iter": 500  if MODE == "train" else 200,
            "early_stopping": True,
            "validation_fraction": 0.15,
            "n_iter_no_change": 20  if MODE == "train" else 10,
            "random_state": 42,
            "learning_rate_init": 5e-4,
            "alpha": 0.005,
        },
    }
    print("CFG loaded")
    
    
    # ?? Data ??????????????????????????????????????????????????????????????????????
    taxonomy          = pd.read_csv(BASE / "taxonomy.csv")
    sample_sub        = pd.read_csv(BASE / "sample_submission.csv")
    soundscape_labels = pd.read_csv(BASE / "train_soundscapes_labels.csv")
    
    PRIMARY_LABELS = sample_sub.columns[1:].tolist()
    N_CLASSES      = len(PRIMARY_LABELS)
    label_to_idx   = {c: i for i, c in enumerate(PRIMARY_LABELS)}
    
    FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")
    
    def parse_fname(name):
        m = FNAME_RE.match(name)
        if not m: return {"site": "unknown", "hour_utc": -1}
        _, site, _, hms = m.groups()
        return {"site": site, "hour_utc": int(hms[:2])}
    
    def union_labels(series):
        out = set()
        for x in series:
            if pd.notna(x):
                for t in str(x).split(";"):
                    t = t.strip()
                    if t: out.add(t)
        return sorted(out)
    
    sc = (soundscape_labels
          .groupby(["filename", "start", "end"])["primary_label"]
          .apply(union_labels)
          .reset_index(name="label_list"))
    
    sc["end_sec"] = pd.to_timedelta(sc["end"]).dt.total_seconds().astype(int)
    sc["row_id"]  = sc["filename"].str.replace(".ogg", "", regex=False) + "_" + sc["end_sec"].astype(str)
    
    _meta = sc["filename"].apply(parse_fname).apply(pd.Series)
    sc = pd.concat([sc, _meta], axis=1)
    
    Y_SC = np.zeros((len(sc), N_CLASSES), dtype=np.uint8)
    for i, lbls in enumerate(sc["label_list"]):
        for lbl in lbls:
            if lbl in label_to_idx:
                Y_SC[i, label_to_idx[lbl]] = 1
    
    windows_per_file = sc.groupby("filename").size()
    full_files = sorted(windows_per_file[windows_per_file == N_WINDOWS].index.tolist())
    sc["fully_labeled"] = sc["filename"].isin(full_files)
    
    full_rows = (sc[sc["fully_labeled"]]
                 .sort_values(["filename", "end_sec"])
                 .reset_index(drop=False))
    Y_FULL = Y_SC[full_rows["index"].to_numpy()]
    
    print(f"Classes: {N_CLASSES} | Fully-labeled files: {len(full_files)}")
    print(f"Full-file windows: {len(full_rows)} | Active classes: {int((Y_FULL.sum(0) > 0).sum())}")
    
    
    # ?? Perch backbone ????????????????????????????????????????????????????????????
    birdclassifier = tf.saved_model.load(str(MODEL_DIR))
    infer_fn       = birdclassifier.signatures["serving_default"]
    
    # Prefer no-DFT variant, fallback to standard
    ONNX_PERCH_PATH = next(INPUT_ROOT.glob("**/perch_v2_no_dft*.onnx"),
                       next(INPUT_ROOT.glob("**/perch_v2*.onnx"), Path("")))
    USE_ONNX = _ONNX_AVAILABLE and ONNX_PERCH_PATH.exists()
    
    if USE_ONNX:
        _so = ort.SessionOptions()
        _so.intra_op_num_threads = 4
        ONNX_SESSION    = ort.InferenceSession(str(ONNX_PERCH_PATH), sess_options=_so,
                                                providers=["CPUExecutionProvider"])
        ONNX_INPUT_NAME = ONNX_SESSION.get_inputs()[0].name
        ONNX_OUT_MAP    = {o.name: i for i, o in enumerate(ONNX_SESSION.get_outputs())}
        print(f"Using ONNX Perch: {ONNX_PERCH_PATH.name}")
    else:
        print("Using TF SavedModel Perch")
    
    bc_labels = (pd.read_csv(MODEL_DIR / "assets" / "labels.csv")
                 .reset_index()
                 .rename(columns={"index": "bc_index", "inat2024_fsd50k": "scientific_name"}))
    NO_LABEL = len(bc_labels)
    
    mapping = (taxonomy
               .merge(bc_labels.rename(columns={"scientific_name": "scientific_name"}),
                      on="scientific_name", how="left"))
    mapping["bc_index"] = mapping["bc_index"].fillna(NO_LABEL).astype(int)
    lbl2bc = mapping.set_index("primary_label")["bc_index"]
    
    BC_INDICES    = np.array([int(lbl2bc.loc[c]) for c in PRIMARY_LABELS], dtype=np.int32)
    MAPPED_MASK   = BC_INDICES != NO_LABEL
    MAPPED_POS    = np.where(MAPPED_MASK)[0].astype(np.int32)
    MAPPED_BC_IDX = BC_INDICES[MAPPED_MASK].astype(np.int32)
    
    print(f"Mapped: {MAPPED_MASK.sum()} / {N_CLASSES} species have a Perch logit")
    
    import re as _re
    UNMAPPED_POS  = np.where(~MAPPED_MASK)[0].astype(np.int32)
    CLASS_NAME_MAP = taxonomy.set_index("primary_label")["class_name"].to_dict()
    TEXTURE_TAXA   = {"Amphibia", "Insecta"}
    
    proxy_map = {}
    unmapped_df = (taxonomy[taxonomy["primary_label"]
                   .isin([PRIMARY_LABELS[i] for i in UNMAPPED_POS])].copy())
    
    for _, row in unmapped_df.iterrows():
        target = row["primary_label"]
        sci    = str(row["scientific_name"])
        genus  = sci.split()[0]
        hits = bc_labels[
            bc_labels["scientific_name"]
            .astype(str)
            .str.match(rf"^{_re.escape(genus)}\s", na=False)
        ]
        if len(hits) > 0:
            proxy_map[label_to_idx[target]] = hits["bc_index"].astype(int).tolist()
    
    PROXY_TAXA = {"Amphibia", "Insecta", "Aves"}
    proxy_map  = {
        idx: bc_idxs
        for idx, bc_idxs in proxy_map.items()
        if CLASS_NAME_MAP.get(PRIMARY_LABELS[idx]) in PROXY_TAXA
    }
    
    print(f"Unmapped: {len(UNMAPPED_POS)} | Proxy: {len(proxy_map)} | No signal: {len(UNMAPPED_POS)-len(proxy_map)}")
    
    
    # ?? Per-taxon temperatures ????????????????????????????????????????????????????
    temperatures = np.ones(N_CLASSES, dtype=np.float32)
    for ci, label in enumerate(PRIMARY_LABELS):
        cls = CLASS_NAME_MAP.get(label, "Aves")
        temperatures[ci] = 0.95 if cls in TEXTURE_TAXA else 1.10
    
    
    # ?? Perch inference engine ????????????????????????????????????????????????????
    import concurrent.futures
    
    def read_60s(path):
        y, sr = sf.read(path, dtype="float32", always_2d=False)
        if y.ndim == 2: y = y.mean(axis=1)
        if len(y) < FILE_SAMPLES: y = np.pad(y, (0, FILE_SAMPLES - len(y)))
        else:                      y = y[:FILE_SAMPLES]
        return y
    
    def run_perch(paths, batch_files=16, verbose=True):
        paths  = [Path(p) for p in paths]
        n_rows = len(paths) * N_WINDOWS
        row_ids   = np.empty(n_rows, dtype=object)
        filenames = np.empty(n_rows, dtype=object)
        sites     = np.empty(n_rows, dtype=object)
        hours     = np.zeros(n_rows, dtype=np.int16)
        scores    = np.zeros((n_rows, N_CLASSES), dtype=np.float32)
        embs      = np.zeros((n_rows, 1536),      dtype=np.float32)
        wr  = 0
        itr = tqdm(range(0, len(paths), batch_files), desc="Perch") if verbose else range(0, len(paths), batch_files)
    
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as io_executor:
            next_paths   = paths[0:batch_files]
            future_audio = [io_executor.submit(read_60s, p) for p in next_paths]
            for start in itr:
                batch_paths  = next_paths
                batch_n      = len(batch_paths)
                batch_audio  = [f.result() for f in future_audio]
                next_start = start + batch_files
                if next_start < len(paths):
                    next_paths   = paths[next_start:next_start + batch_files]
                    future_audio = [io_executor.submit(read_60s, p) for p in next_paths]
                x  = np.empty((batch_n * N_WINDOWS, WINDOW_SAMPLES), dtype=np.float32)
                br = wr
                for bi, path in enumerate(batch_paths):
                    y    = batch_audio[bi]
                    meta = parse_fname(path.name)
                    stem = path.stem
                    x[bi * N_WINDOWS:(bi + 1) * N_WINDOWS] = y.reshape(N_WINDOWS, WINDOW_SAMPLES)
                    row_ids  [wr:wr + N_WINDOWS] = [f"{stem}_{t}" for t in range(5, 65, 5)]
                    filenames[wr:wr + N_WINDOWS] = path.name
                    sites    [wr:wr + N_WINDOWS] = meta["site"]
                    hours    [wr:wr + N_WINDOWS] = meta["hour_utc"]
                    wr += N_WINDOWS
                if USE_ONNX:
                    outs   = ONNX_SESSION.run(None, {ONNX_INPUT_NAME: x})
                    logits = outs[ONNX_OUT_MAP["label"]].astype(np.float32)
                    emb    = outs[ONNX_OUT_MAP["embedding"]].astype(np.float32)
                else:
                    out    = infer_fn(inputs=tf.convert_to_tensor(x))
                    logits = out["label"].numpy().astype(np.float32)
                    emb    = out["embedding"].numpy().astype(np.float32)
                scores[br:wr, MAPPED_POS] = logits[:, MAPPED_BC_IDX]
                embs  [br:wr]             = emb
                for pos_idx, bc_idxs in proxy_map.items():
                    bc_arr = np.array(bc_idxs, dtype=np.int32)
                    scores[br:wr, pos_idx] = logits[:, bc_arr].max(axis=1)
                del x, logits, emb, batch_audio
                gc.collect()
        meta_df = pd.DataFrame({"row_id": row_ids, "filename": filenames,
                                 "site": sites, "hour_utc": hours})
        return meta_df, scores, embs
    
    print("Perch inference engine defined")
    
    
    # ?? Cache ?????????????????????????????????????????????????????????????????????
    print(f"USE_ONNX = {USE_ONNX}")
    
    EXTERNAL_CACHE_DIRS = [
        Path("/kaggle/input/notebooks/vyankteshdwivedi/notebook1b25083f0d"),
        Path("/kaggle/input/datasets/jaejohn/perch-meta"),
    ]
    CACHE_META_LOCAL = WORK_DIR / "perch_meta.parquet"
    CACHE_NPZ_LOCAL  = WORK_DIR / "perch_arrays.npz"
    
    def _find_external_cache():
        for d in EXTERNAL_CACHE_DIRS:
            meta = d / "perch_meta.parquet"
            npz  = d / "perch_arrays.npz"
            if meta.exists() and npz.exists():
                return meta, npz
        return None, None
    
    SCORE_KEYS = ["scores", "sc", "logits", "perch_scores", "preds", "arr_0"]
    EMB_KEYS   = ["embs", "emb", "embeddings", "features", "perch_embs", "arr_1"]
    
    def _pick_array(arr, candidates, shape_hint_cols):
        for k in candidates:
            if k in arr.files:
                return arr[k], k
        for k in arr.files:
            v = arr[k]
            if v.ndim == 2 and v.shape[1] == shape_hint_cols:
                return v, k
        raise KeyError(f"None of {candidates} found in npz. Available keys: {arr.files}")
    
    def _build_cache():
        print(f"Building Perch cache from {len(full_files)} training files?")
        train_paths = [BASE / "train_soundscapes" / fn for fn in full_files]
        train_paths = [p for p in train_paths if p.exists()]
        t0 = time.time()
        meta_built, sc_built, emb_built = run_perch(train_paths, batch_files=CFG["batch_files"], verbose=True)
        print(f"  Perch pass done in {time.time()-t0:.1f}s  scores={sc_built.shape} embs={emb_built.shape}")
        meta_built.to_parquet(CACHE_META_LOCAL)
        np.savez(CACHE_NPZ_LOCAL, scores=sc_built.astype(np.float32),
                 embs=emb_built.astype(np.float32), primary_labels=np.array(PRIMARY_LABELS))
        print(f"  Cache saved to {WORK_DIR}")
        return CACHE_META_LOCAL, CACHE_NPZ_LOCAL
    
    ext_meta, ext_npz = _find_external_cache()
    if ext_meta is not None:
        CACHE_META, CACHE_NPZ = ext_meta, ext_npz
        print(f"Using external cache: {CACHE_META.parent}")
    elif CACHE_META_LOCAL.exists() and CACHE_NPZ_LOCAL.exists():
        CACHE_META, CACHE_NPZ = CACHE_META_LOCAL, CACHE_NPZ_LOCAL
        print(f"Using local cache: {WORK_DIR}")
    else:
        print("No cache found ? building from scratch")
        CACHE_META, CACHE_NPZ = _build_cache()
    
    meta_tr = pd.read_parquet(CACHE_META)
    _arr    = np.load(CACHE_NPZ)
    sc_tr_raw,  sk = _pick_array(_arr, SCORE_KEYS, N_CLASSES)
    emb_tr_raw, ek = _pick_array(_arr, EMB_KEYS,   1536)
    sc_tr  = sc_tr_raw.astype(np.float32)
    emb_tr = emb_tr_raw.astype(np.float32)
    
    if "primary_labels" in _arr.files:
        if _arr["primary_labels"].tolist() != PRIMARY_LABELS:
            print("  WARNING: cached primary_labels differ ? scores columns may not align!")
        else:
            print("  primary_labels schema OK")
    
    if "row_id" not in meta_tr.columns:
        if "end_sec" in meta_tr.columns:
            end_sec = meta_tr["end_sec"].astype(int)
        elif "window_idx" in meta_tr.columns:
            end_sec = (meta_tr["window_idx"].astype(int) + 1) * 5
        else:
            end_sec = np.tile(np.arange(5, 65, 5), len(meta_tr) // N_WINDOWS)
        meta_tr["row_id"] = (meta_tr["filename"].str.replace(".ogg", "", regex=False)
                             + "_" + end_sec.astype(str))
    
    row_id_to_index = full_rows.set_index("row_id")["index"]
    missing_rows = set(meta_tr["row_id"]) - set(row_id_to_index.index)
    if missing_rows:
        raise RuntimeError(f"Cache has {len(missing_rows)} row_ids not in labeled set.")
    
    Y_FULL_aligned = Y_SC[row_id_to_index.loc[meta_tr["row_id"]].to_numpy()]
    print(f"sc_tr: {sc_tr.shape}  emb_tr: {emb_tr.shape}  Y_FULL_aligned: {Y_FULL_aligned.shape}")
    
    
    # ?? Post-processing helpers ???????????????????????????????????????????????????
    def macro_auc(y_true, y_score):
        keep = y_true.sum(axis=0) > 0
        return roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro")
    
    def smooth_predictions(probs, n_windows=12, alpha=0.3):
        N, C = probs.shape
        assert N % n_windows == 0
        view = probs.reshape(-1, n_windows, C).copy()
        prev_w = np.concatenate([view[:, :1, :],  view[:, :-1, :]], axis=1)
        next_w = np.concatenate([view[:, 1:,  :], view[:, -1:, :]], axis=1)
        return ((1 - alpha) * view + 0.5 * alpha * (prev_w + next_w)).reshape(N, C)
    
    
    # ?? UPGRADED prior tables ? joint site-hour bucket ????????????????????????????
    def build_prior_tables(sc_df, Y_labels):
        sc_df = sc_df.reset_index(drop=True)
        global_p = Y_labels.mean(axis=0).astype(np.float32)
    
        site_keys = sorted(sc_df["site"].dropna().astype(str).unique())
        site_to_i = {k: i for i, k in enumerate(site_keys)}
        site_p = np.zeros((len(site_keys), Y_labels.shape[1]), dtype=np.float32)
        site_n = np.zeros(len(site_keys), dtype=np.float32)
        for s in site_keys:
            i = site_to_i[s]
            mask = sc_df["site"].astype(str).values == s
            site_n[i] = mask.sum()
            site_p[i] = Y_labels[mask].mean(axis=0)
    
        hour_keys = sorted(sc_df["hour_utc"].dropna().astype(int).unique())
        hour_to_i = {h: i for i, h in enumerate(hour_keys)}
        hour_p = np.zeros((len(hour_keys), Y_labels.shape[1]), dtype=np.float32)
        hour_n = np.zeros(len(hour_keys), dtype=np.float32)
        for h in hour_keys:
            i = hour_to_i[h]
            mask = sc_df["hour_utc"].astype(int).values == h
            hour_n[i] = mask.sum()
            hour_p[i] = Y_labels[mask].mean(axis=0)
    
        # Joint site-hour bucket (new ? tighter shrinkage factor 4)
        sh_keys = sorted({(str(s), int(h)) for s, h in zip(sc_df["site"].dropna(), sc_df["hour_utc"].dropna())
                          if not pd.isna(s) and not pd.isna(h)})
        sh_to_i = {k: i for i, k in enumerate(sh_keys)}
        sh_p = np.zeros((len(sh_keys), Y_labels.shape[1]), dtype=np.float32)
        sh_n = np.zeros(len(sh_keys), dtype=np.float32)
        for (s, h) in sh_keys:
            i = sh_to_i[(s, h)]
            mask = (sc_df["site"].astype(str).values == s) & (sc_df["hour_utc"].astype(int).values == h)
            sh_n[i] = mask.sum()
            sh_p[i] = Y_labels[mask].mean(axis=0)
    
        return {
            "global_p": global_p,
            "site_to_i": site_to_i, "site_p": site_p, "site_n": site_n,
            "hour_to_i": hour_to_i, "hour_p": hour_p, "hour_n": hour_n,
            "sh_to_i": sh_to_i,    "sh_p": sh_p,    "sh_n": sh_n,
        }
    
    def apply_prior(scores, sites, hours, tables, lambda_prior=0.4):
        eps = 1e-4; n = len(scores); out = scores.copy()
        p = np.tile(tables["global_p"], (n, 1))
        for i, h in enumerate(hours):
            h = int(h)
            if h in tables["hour_to_i"]:
                j = tables["hour_to_i"][h]; nh = tables["hour_n"][j]; w = nh / (nh + 8.0)
                p[i] = w * tables["hour_p"][j] + (1 - w) * tables["global_p"]
        for i, s in enumerate(sites):
            s = str(s)
            if s in tables["site_to_i"]:
                j = tables["site_to_i"][s]; ns = tables["site_n"][j]; w = ns / (ns + 8.0)
                p[i] = w * tables["site_p"][j] + (1 - w) * p[i]
        if "sh_to_i" in tables:
            for i, (s, h) in enumerate(zip(sites, hours)):
                key = (str(s), int(h))
                if key in tables["sh_to_i"]:
                    j = tables["sh_to_i"][key]; nsh = tables["sh_n"][j]; w = nsh / (nsh + 4.0)
                    p[i] = w * tables["sh_p"][j] + (1 - w) * p[i]
        p = np.clip(p, eps, 1 - eps)
        out += lambda_prior * (np.log(p) - np.log1p(-p))
        return out.astype(np.float32)
    
    def file_confidence_scale(probs, n_windows=12, top_k=2, power=0.4):
        N, C = probs.shape
        view      = probs.reshape(-1, n_windows, C)
        sorted_v  = np.sort(view, axis=1)
        top_k_mean = sorted_v[:, -top_k:, :].mean(axis=1, keepdims=True)
        return (view * np.power(top_k_mean, power)).reshape(N, C)
    
    def rank_aware_scaling(probs, n_windows=12, power=0.4):
        N, C = probs.shape
        view     = probs.reshape(-1, n_windows, C)
        file_max = view.max(axis=1, keepdims=True)
        return (view * np.power(file_max, power)).reshape(N, C)
    
    def adaptive_delta_smooth(probs, n_windows=12, base_alpha=0.20):
        N, C = probs.shape
        result = probs.copy(); view = probs.reshape(-1, n_windows, C); out = result.reshape(-1, n_windows, C)
        for t in range(n_windows):
            conf = view[:, t, :].max(axis=-1, keepdims=True); alpha = base_alpha * (1.0 - conf)
            if t == 0:           neighbor_avg = (view[:, t, :] + view[:, t+1, :]) / 2.0
            elif t == n_windows-1: neighbor_avg = (view[:, t-1, :] + view[:, t, :]) / 2.0
            else:                  neighbor_avg = (view[:, t-1, :] + view[:, t+1, :]) / 2.0
            out[:, t, :] = (1.0 - alpha) * view[:, t, :] + alpha * neighbor_avg
        return result
    
    
    # ?? MLP probes ????????????????????????????????????????????????????????????????
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier
    from sklearn.isotonic import IsotonicRegression
    import torch.nn as nn
    import torch.nn.functional as F
    
    def build_class_freq_weights(Y, cap=10.0):
        pos_count = Y.sum(axis=0).astype(np.float32) + 1.0
        freq = pos_count / Y.shape[0]
        weights = np.clip(1.0 / (freq ** 0.5), 1.0, cap)
        return (weights / weights.mean()).astype(np.float32)
    
    def build_sequential_features(scores_col, n_windows=12):
        x     = scores_col.reshape(-1, n_windows)
        prev  = np.concatenate([x[:, :1], x[:, :-1]], axis=1)
        next_ = np.concatenate([x[:, 1:], x[:, -1:]], axis=1)
        mean  = np.repeat(x.mean(axis=1), n_windows)
        max_  = np.repeat(x.max(axis=1),  n_windows)
        std   = np.repeat(x.std(axis=1),  n_windows)
        return prev.reshape(-1), next_.reshape(-1), mean, max_, std
    
    def train_mlp_probes(emb, scores_raw, Y, min_pos=5, pca_dim=64, alpha_blend=0.4):
        scaler = StandardScaler(); emb_s = scaler.fit_transform(emb)
        pca    = PCA(n_components=min(pca_dim, emb_s.shape[1] - 1))
        Z      = pca.fit_transform(emb_s).astype(np.float32)
        print(f"Embedding: {emb.shape} ? PCA: {Z.shape}  (variance retained: {pca.explained_variance_ratio_.sum():.2%})")
        class_weights = build_class_freq_weights(Y, cap=10.0)
        probe_models = {}; active = np.where(Y.sum(axis=0) >= min_pos)[0]; MAX_ROWS = 3000
        for ci in tqdm(active, desc="MLP probes"):
            y = Y[:, ci]
            if y.sum() == 0 or y.sum() == len(y): continue
            prev, next_, mean, max_, std = build_sequential_features(scores_raw[:, ci])
            X = np.hstack([Z, scores_raw[:, ci:ci+1], prev[:, None], next_[:, None], mean[:, None], max_[:, None], std[:, None]])
            n_pos = int(y.sum()); n_neg = len(y) - n_pos; pos_idx = np.where(y == 1)[0]
            w = float(class_weights[ci]); repeat = max(1, min(int(round(w * n_neg / max(n_pos, 1))), 8))
            if n_pos * repeat + len(y) > MAX_ROWS: repeat = max(1, (MAX_ROWS - len(y)) // max(n_pos, 1))
            X_bal = np.vstack([X, np.tile(X[pos_idx], (repeat, 1))])
            y_bal = np.concatenate([y, np.ones(n_pos * repeat, dtype=y.dtype)])
            clf = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", max_iter=300,
                                early_stopping=True, validation_fraction=0.15, n_iter_no_change=15,
                                random_state=42, learning_rate_init=5e-4, alpha=0.005)
            clf.fit(X_bal, y_bal); probe_models[ci] = clf
        print(f"Trained {len(probe_models)} MLP probes")
        return probe_models, scaler, pca, alpha_blend
    
    class VectorizedMLPProbes(nn.Module):
        def __init__(self, probe_models):
            super().__init__(); self.valid_classes = sorted(probe_models.keys()); V = len(self.valid_classes)
            if V == 0: self.weights = nn.ParameterList(); self.biases = nn.ParameterList(); self.n_layers = 0; return
            sample = probe_models[self.valid_classes[0]]; self.n_layers = len(sample.coefs_)
            self.weights = nn.ParameterList(); self.biases = nn.ParameterList()
            for li in range(self.n_layers):
                W = np.stack([probe_models[c].coefs_[li] for c in self.valid_classes], axis=0)
                b = np.stack([probe_models[c].intercepts_[li] for c in self.valid_classes], axis=0)
                self.weights.append(nn.Parameter(torch.tensor(W, dtype=torch.float32), requires_grad=False))
                self.biases.append(nn.Parameter(torch.tensor(b, dtype=torch.float32), requires_grad=False))
        def forward(self, x):
            h = x
            for i in range(self.n_layers):
                h = torch.bmm(h, self.weights[i]) + self.biases[i].unsqueeze(1)
                if i < self.n_layers - 1: h = torch.relu(h)
            return h.squeeze(-1)
    
    def apply_mlp_probes_vectorized(emb_test, scores_test, probe_models, scaler, pca, alpha_blend=0.4):
        if len(probe_models) == 0: return scores_test.copy()
        Z_test = pca.transform(scaler.transform(emb_test)).astype(np.float32)
        valid_classes = sorted(probe_models.keys()); V = len(valid_classes); N = len(scores_test)
        raw = scores_test[:, valid_classes].T; n_files = N // N_WINDOWS; raw_view = raw.reshape(V, n_files, N_WINDOWS)
        prev = np.concatenate([raw_view[:, :, :1], raw_view[:, :, :-1]], axis=2).reshape(V, N)
        nxt  = np.concatenate([raw_view[:, :, 1:], raw_view[:, :, -1:]], axis=2).reshape(V, N)
        mean = np.repeat(raw_view.mean(axis=2), N_WINDOWS, axis=1); mx = np.repeat(raw_view.max(axis=2), N_WINDOWS, axis=1)
        std  = np.repeat(raw_view.std(axis=2),  N_WINDOWS, axis=1)
        scalar_feats = np.stack([raw, prev, nxt, mean, mx, std], axis=-1).astype(np.float32)
        Z_expanded = np.broadcast_to(Z_test, (V, N, Z_test.shape[1]))
        X_all = np.concatenate([Z_expanded.astype(np.float32), scalar_feats], axis=-1)
        vec_probe = VectorizedMLPProbes(probe_models).eval()
        with torch.no_grad(): preds = vec_probe(torch.tensor(X_all)).numpy()
        result = scores_test.copy()
        result[:, valid_classes] = (1.0 - alpha_blend) * scores_test[:, valid_classes] + alpha_blend * preds.T
        return result
    
    def calibrate_and_optimize_thresholds(oof_probs, Y_FULL, threshold_grid=None, n_windows=12):
        if threshold_grid is None: threshold_grid = [0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70]
        n_samples, n_cls = oof_probs.shape; thresholds = np.full(n_cls, 0.5, dtype=np.float32)
        n_files = n_samples // n_windows
        file_oof = oof_probs.reshape(n_files, n_windows, n_cls).max(axis=1)
        file_y   = Y_FULL.reshape(n_files, n_windows, n_cls).max(axis=1)
        n_calibrated = 0
        for c in range(n_cls):
            y_true = file_y[:, c]; y_prob = file_oof[:, c]
            if y_true.sum() < 3: continue
            try:
                ir = IsotonicRegression(out_of_bounds="clip"); ir.fit(y_prob, y_true); y_cal = ir.transform(y_prob)
            except: y_cal = y_prob
            best_f1, best_t = 0.0, 0.5
            for t in threshold_grid:
                pred = (y_cal >= t).astype(int)
                tp=((pred==1)&(y_true==1)).sum(); fp=((pred==1)&(y_true==0)).sum(); fn=((pred==0)&(y_true==1)).sum()
                prec=tp/(tp+fp+1e-8); rec=tp/(tp+fn+1e-8); f1=2*prec*rec/(prec+rec+1e-8)
                if f1 > best_f1: best_f1,best_t = f1,t
            thresholds[c] = best_t; n_calibrated += 1
        print(f"Calibrated {n_calibrated} classes | Mean threshold: {thresholds.mean():.3f} | Range: [{thresholds.min():.2f}, {thresholds.max():.2f}]")
        return thresholds
    
    def apply_per_class_thresholds(scores, thresholds):
        C = scores.shape[1]; scaled = np.copy(scores)
        for c in range(C):
            t = thresholds[c]; above = scores[:, c] > t
            scaled[above, c]  = 0.5 + 0.5 * (scores[above, c]  - t) / (1 - t + 1e-8)
            scaled[~above, c] = 0.5 * scores[~above, c] / (t + 1e-8)
        return np.clip(scaled, 0.0, 1.0)
    
    
    # ?? SSM Architecture ?????????????????????????????????????????????????????????
    class SelectiveSSM(nn.Module):
        def __init__(self, d_model, d_state=16, d_conv=4):
            super().__init__(); self.d_model=d_model; self.d_state=d_state
            self.in_proj=nn.Linear(d_model,2*d_model,bias=False)
            self.conv1d=nn.Conv1d(d_model,d_model,d_conv,padding=d_conv-1,groups=d_model)
            self.dt_proj=nn.Linear(d_model,d_model,bias=True)
            A=torch.arange(1,d_state+1,dtype=torch.float32).unsqueeze(0).expand(d_model,-1)
            self.A_log=nn.Parameter(torch.log(A)); self.D=nn.Parameter(torch.ones(d_model))
            self.B_proj=nn.Linear(d_model,d_state,bias=False); self.C_proj=nn.Linear(d_model,d_state,bias=False)
            self.out_proj=nn.Linear(d_model,d_model,bias=False)
        def forward(self,x):
            B_sz,T,D=x.shape; xz=self.in_proj(x); x_ssm,z=xz.chunk(2,dim=-1)
            x_conv=F.silu(self.conv1d(x_ssm.transpose(1,2))[:,:,:T].transpose(1,2))
            dt=F.softplus(self.dt_proj(x_conv)); A=-torch.exp(self.A_log)
            B=self.B_proj(x_conv); C=self.C_proj(x_conv)
            h=torch.zeros(B_sz,D,self.d_state,device=x.device); ys=[]
            for t in range(T):
                dA=torch.exp(A[None]*dt[:,t,:,None]); dB=dt[:,t,:,None]*B[:,t,None,:]
                h=h*dA+x[:,t,:,None]*dB; ys.append((h*C[:,t,None,:]).sum(-1))
            return torch.stack(ys,dim=1)+x*self.D[None,None,:]
    
    class LightProtoSSM(nn.Module):
        def __init__(self,d_input=1536,d_model=128,d_state=16,n_classes=234,n_windows=12,
                     dropout=0.15,n_sites=20,meta_dim=16,use_cross_attn=True,cross_attn_heads=2):
            super().__init__(); self.n_classes=n_classes; self.n_windows=n_windows; self.use_cross_attn=use_cross_attn
            self.input_proj=nn.Sequential(nn.Linear(d_input,d_model),nn.LayerNorm(d_model),nn.GELU(),nn.Dropout(dropout))
            self.pos_enc=nn.Parameter(torch.randn(1,n_windows,d_model)*0.02)
            self.site_emb=nn.Embedding(n_sites,meta_dim); self.hour_emb=nn.Embedding(24,meta_dim)
            self.meta_proj=nn.Linear(2*meta_dim,d_model)
            self.ssm_fwd=nn.ModuleList([SelectiveSSM(d_model,d_state) for _ in range(2)])
            self.ssm_bwd=nn.ModuleList([SelectiveSSM(d_model,d_state) for _ in range(2)])
            self.ssm_merge=nn.ModuleList([nn.Linear(2*d_model,d_model) for _ in range(2)])
            self.ssm_norm=nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])
            self.drop=nn.Dropout(dropout)
            if use_cross_attn:
                self.cross_attn=nn.ModuleList([nn.MultiheadAttention(d_model,cross_attn_heads,dropout=dropout,batch_first=True) for _ in range(2)])
                self.cross_norm=nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])
            self.prototypes=nn.Parameter(torch.randn(n_classes,d_model)*0.02)
            self.proto_temp=nn.Parameter(torch.tensor(5.0))
            self.class_bias=nn.Parameter(torch.zeros(n_classes))
            self.fusion_alpha=nn.Parameter(torch.zeros(n_classes))
        def init_prototypes(self,emb_tensor,labels_tensor):
            with torch.no_grad():
                h=self.input_proj(emb_tensor)
                for c in range(self.n_classes):
                    mask=labels_tensor[:,c]>0.5
                    if mask.sum()>0: self.prototypes.data[c]=F.normalize(h[mask].mean(0),dim=0)
        def forward(self,emb,perch_logits=None,site_ids=None,hours=None):
            B,T,_=emb.shape; h=self.input_proj(emb)+self.pos_enc[:,:T,:]
            if site_ids is not None and hours is not None:
                meta=self.meta_proj(torch.cat([self.site_emb(site_ids),self.hour_emb(hours)],dim=-1))
                h=h+meta[:,None,:]
            for i,(fwd,bwd,merge,norm) in enumerate(zip(self.ssm_fwd,self.ssm_bwd,self.ssm_merge,self.ssm_norm)):
                res=h; hf=fwd(h); hb=bwd(h.flip(1)).flip(1)
                h=self.drop(merge(torch.cat([hf,hb],dim=-1))); h=norm(h+res)
                if self.use_cross_attn:
                    attn_out,_=self.cross_attn[i](h,h,h); h=self.cross_norm[i](h+attn_out)
            h_n=F.normalize(h,dim=-1); p_n=F.normalize(self.prototypes,dim=-1)
            sim=torch.matmul(h_n,p_n.T)*F.softplus(self.proto_temp)+self.class_bias[None,None,:]
            if perch_logits is not None:
                alpha=torch.sigmoid(self.fusion_alpha)[None,None,:]
                out=alpha*sim+(1-alpha)*perch_logits
            else: out=sim
            return out
    
    class ResidualSSM(nn.Module):
        def __init__(self,d_input=1536,d_scores=234,d_model=64,d_state=8,n_classes=234,
                     n_windows=12,dropout=0.1,n_sites=20,meta_dim=8):
            super().__init__(); self.n_classes=n_classes
            self.input_proj=nn.Sequential(nn.Linear(d_input+d_scores,d_model),nn.LayerNorm(d_model),nn.GELU(),nn.Dropout(dropout))
            self.site_emb=nn.Embedding(n_sites,meta_dim); self.hour_emb=nn.Embedding(24,meta_dim)
            self.meta_proj=nn.Linear(2*meta_dim,d_model)
            self.pos_enc=nn.Parameter(torch.randn(1,n_windows,d_model)*0.02)
            self.ssm_fwd=SelectiveSSM(d_model,d_state); self.ssm_bwd=SelectiveSSM(d_model,d_state)
            self.ssm_merge=nn.Linear(2*d_model,d_model); self.ssm_norm=nn.LayerNorm(d_model); self.ssm_drop=nn.Dropout(dropout)
            self.output_head=nn.Linear(d_model,n_classes)
            nn.init.zeros_(self.output_head.weight); nn.init.zeros_(self.output_head.bias)
        def forward(self,emb,first_pass,site_ids=None,hours=None):
            B,T,_=emb.shape; x=torch.cat([emb,first_pass],dim=-1)
            h=self.input_proj(x)+self.pos_enc[:,:T,:]
            if site_ids is not None and hours is not None:
                meta=self.meta_proj(torch.cat([self.site_emb(site_ids.clamp(0,self.site_emb.num_embeddings-1)),
                                                self.hour_emb(hours.clamp(0,23))],dim=-1))
                h=h+meta.unsqueeze(1)
            res=h; hf=self.ssm_fwd(h); hb=self.ssm_bwd(h.flip(1)).flip(1)
            h=self.ssm_drop(self.ssm_merge(torch.cat([hf,hb],dim=-1))); h=self.ssm_norm(h+res)
            return self.output_head(h)
    
    def train_light_proto_ssm(emb_full, scores_full, Y_full, meta_full, n_epochs=40, patience=8, lr=1e-3, n_sites=20, verbose=False):
        n_files=len(emb_full)//N_WINDOWS; emb_f=emb_full.reshape(n_files,N_WINDOWS,-1)
        log_f=scores_full.reshape(n_files,N_WINDOWS,-1); lab_f=Y_full.reshape(n_files,N_WINDOWS,-1).astype(np.float32)
        fnames=meta_full["filename"].unique(); sites_u=sorted(meta_full["site"].unique())
        site2i={s:i+1 for i,s in enumerate(sites_u)}
        site_ids=np.array([min(site2i.get(meta_full.loc[meta_full["filename"]==fn,"site"].iloc[0],0),n_sites-1) for fn in fnames],dtype=np.int64)
        hour_ids=np.array([int(meta_full.loc[meta_full["filename"]==fn,"hour_utc"].iloc[0])%24 for fn in fnames],dtype=np.int64)
        model=LightProtoSSM(n_classes=N_CLASSES,n_sites=n_sites,use_cross_attn=True,cross_attn_heads=2)
        model.init_prototypes(torch.tensor(emb_full,dtype=torch.float32),torch.tensor(Y_full,dtype=torch.float32))
        emb_t=torch.tensor(emb_f,dtype=torch.float32); log_t=torch.tensor(log_f,dtype=torch.float32)
        lab_t=torch.tensor(lab_f,dtype=torch.float32); site_t=torch.tensor(site_ids,dtype=torch.long)
        hour_t=torch.tensor(hour_ids,dtype=torch.long)
        pos_cnt=lab_t.sum(dim=(0,1)); total=lab_t.shape[0]*lab_t.shape[1]
        pos_weight=((total-pos_cnt)/(pos_cnt+1)).clamp(max=25.0)
        opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-3)
        sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=lr,epochs=n_epochs,steps_per_epoch=1,pct_start=0.1,anneal_strategy="cos")
        best_loss,best_state,wait=float("inf"),None,0
        swa_model=torch.optim.swa_utils.AveragedModel(model); swa_start=int(n_epochs*0.65)
        swa_sched=torch.optim.swa_utils.SWALR(opt,swa_lr=4e-4)
        for ep in range(n_epochs):
            model.train()
            out=model(emb_t,log_t,site_ids=site_t,hours=hour_t)
            loss=F.binary_cross_entropy_with_logits(out,lab_t,pos_weight=pos_weight[None,None,:])+0.15*F.mse_loss(out,log_t)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            if ep>=swa_start: swa_model.update_parameters(model); swa_sched.step()
            else: sched.step()
            if loss.item()<best_loss:
                best_loss=loss.item(); best_state={k:v.clone() for k,v in model.state_dict().items()}; wait=0
            else:
                wait+=1
                if wait>=patience: break
        if ep>=swa_start:
            torch.optim.swa_utils.update_bn(emb_t.unsqueeze(0),swa_model); model=swa_model
        else: model.load_state_dict(best_state)
        model.eval(); return model,site2i
    
    def run_tta_proto(proto_model, emb_files, sc_files, site_t, hour_t, shifts=[0,1,-1,2,-2]):
        proto_model.eval(); all_preds=[]
        emb_t=torch.tensor(emb_files,dtype=torch.float32); sc_t=torch.tensor(sc_files,dtype=torch.float32)
        for shift in shifts:
            e=torch.roll(emb_t,shift,dims=1) if shift else emb_t
            s=torch.roll(sc_t,shift,dims=1) if shift else sc_t
            with torch.no_grad():
                out=proto_model(e,s,site_ids=site_t,hours=hour_t).numpy()
            if shift: out=np.roll(out,-shift,axis=1)
            all_preds.append(out)
        return np.mean(all_preds,axis=0)
    
    def train_residual_ssm(emb_full, first_pass_flat, Y_full, site_ids, hour_ids,
                            n_epochs=30, patience=8, lr=1e-3, correction_weight=0.30, verbose=False):
        n_files=len(emb_full)//N_WINDOWS; emb_f=emb_full.reshape(n_files,N_WINDOWS,-1)
        fp_f=first_pass_flat.reshape(n_files,N_WINDOWS,-1); lab_f=Y_full.reshape(n_files,N_WINDOWS,-1).astype(np.float32)
        fp_prob=1.0/(1.0+np.exp(-np.clip(fp_f,-30,30))); residuals=lab_f-fp_prob
        n_val=max(1,int(n_files*0.15)); rng=torch.Generator(); rng.manual_seed(42)
        perm=torch.randperm(n_files,generator=rng).numpy(); val_i=perm[:n_val]; train_i=perm[n_val:]
        emb_t=torch.tensor(emb_f,dtype=torch.float32); fp_t=torch.tensor(fp_f,dtype=torch.float32)
        res_t=torch.tensor(residuals,dtype=torch.float32)
        site_t=torch.tensor(site_ids,dtype=torch.long); hour_t=torch.tensor(hour_ids,dtype=torch.long)
        model=ResidualSSM(n_classes=N_CLASSES)
        opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-3)
        sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=lr,epochs=n_epochs,steps_per_epoch=1,pct_start=0.1,anneal_strategy="cos")
        best_loss,best_state,wait=float("inf"),None,0
        for ep in range(n_epochs):
            model.train()
            corr=model(emb_t[train_i],fp_t[train_i],site_ids=site_t[train_i],hours=hour_t[train_i])
            loss=F.mse_loss(corr,res_t[train_i])
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sched.step()
            model.eval()
            with torch.no_grad():
                val_corr=model(emb_t[val_i],fp_t[val_i],site_ids=site_t[val_i],hours=hour_t[val_i])
                val_loss=F.mse_loss(val_corr,res_t[val_i])
            if val_loss.item()<best_loss:
                best_loss=val_loss.item(); best_state={k:v.clone() for k,v in model.state_dict().items()}; wait=0
            else:
                wait+=1
                if wait>=patience: break
        model.load_state_dict(best_state); return model,correction_weight
    
    print("Sequence Models defined")
    
    
    # ?? Test inference ????????????????????????????????????????????????????????????
    test_paths = sorted((BASE / "test_soundscapes").glob("*.ogg"))
    IS_DRY_RUN = len(test_paths) == 0
    if IS_DRY_RUN:
        n = CFG["dryrun_n_files"] or 20
        print(f"No hidden test ? dry-run on {n} train files")
        test_paths = sorted((BASE / "train_soundscapes").glob("*.ogg"))[:n]
    else:
        print(f"Hidden test files: {len(test_paths)}")
    
    meta_te, sc_te, emb_te = run_perch(test_paths, CFG["batch_files"], verbose=CFG["verbose"])
    print(f"Test scores: {sc_te.shape}")
    
    
    # ?? Full ProtoSSM pipeline ????????????????????????????????????????????????????
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
    
    t0 = time.time()
    proto_model, site2i_tr = train_light_proto_ssm(
        emb_tr, sc_tr, Y_FULL_aligned, meta_tr,
        n_epochs=40, patience=8, lr=1e-3, verbose=False)
    print(f"ProtoSSM training: {time.time()-t0:.1f}s")
    
    n_test_files  = len(sc_te) // N_WINDOWS
    emb_te_f      = emb_te.reshape(n_test_files, N_WINDOWS, -1)
    sc_te_f       = sc_te.reshape(n_test_files, N_WINDOWS, -1)
    
    test_fnames   = meta_te.drop_duplicates("filename")["filename"].tolist()
    n_sites_cap   = 20
    test_site_ids = np.array([min(site2i_tr.get(meta_te.loc[meta_te["filename"]==fn,"site"].iloc[0],0),n_sites_cap-1)
                               for fn in test_fnames], dtype=np.int64)
    test_hour_ids = np.array([int(meta_te.loc[meta_te["filename"]==fn,"hour_utc"].iloc[0])%24
                               for fn in test_fnames], dtype=np.int64)
    
    proto_model.eval()
    with torch.no_grad():
        proto_out = proto_model(
            torch.tensor(emb_te_f, dtype=torch.float32),
            torch.tensor(sc_te_f,  dtype=torch.float32),
            site_ids=torch.tensor(test_site_ids, dtype=torch.long),
            hours   =torch.tensor(test_hour_ids, dtype=torch.long),
        ).numpy()
    proto_scores_flat = proto_out.reshape(-1, N_CLASSES).astype(np.float32)
    
    prior_tables   = build_prior_tables(sc, Y_SC)
    sc_te_adjusted = apply_prior(sc_te, sites=meta_te["site"].to_numpy(),
                                  hours=meta_te["hour_utc"].to_numpy(), tables=prior_tables, lambda_prior=0.4)
    
    probe_models, emb_scaler, emb_pca, alpha_blend = train_mlp_probes(
        emb=emb_tr, scores_raw=sc_tr, Y=Y_FULL_aligned, min_pos=5, pca_dim=64, alpha_blend=0.4)
    sc_te_adjusted = apply_mlp_probes_vectorized(emb_te, sc_te_adjusted, probe_models, emb_scaler, emb_pca, alpha_blend)
    
    ENSEMBLE_W      = 0.5
    first_pass_flat = (ENSEMBLE_W * proto_scores_flat + (1.0 - ENSEMBLE_W) * sc_te_adjusted)
    
    n_tr_files    = len(sc_tr) // N_WINDOWS
    emb_tr_f      = emb_tr.reshape(n_tr_files, N_WINDOWS, -1)
    sc_tr_f       = sc_tr.reshape(n_tr_files, N_WINDOWS, -1)
    
    tr_fnames     = meta_tr.drop_duplicates("filename")["filename"].tolist()
    tr_site_ids   = np.array([min(site2i_tr.get(meta_tr.loc[meta_tr["filename"]==fn,"site"].iloc[0],0),n_sites_cap-1)
                               for fn in tr_fnames], dtype=np.int64)
    tr_hour_ids   = np.array([int(meta_tr.loc[meta_tr["filename"]==fn,"hour_utc"].iloc[0])%24
                               for fn in tr_fnames], dtype=np.int64)
    
    proto_tr_out = run_tta_proto(proto_model, emb_tr_f, sc_tr_f,
        site_t=torch.tensor(tr_site_ids, dtype=torch.long),
        hour_t=torch.tensor(tr_hour_ids, dtype=torch.long),
        shifts=[0, 1, -1, 2, -2])
    proto_tr_flat = proto_tr_out.reshape(-1, N_CLASSES).astype(np.float32)
    
    sc_tr_prior = apply_prior(sc_tr, sites=meta_tr["site"].to_numpy(),
                               hours=meta_tr["hour_utc"].to_numpy(), tables=prior_tables, lambda_prior=0.4)
    sc_tr_mlp = apply_mlp_probes_vectorized(emb_tr, sc_tr_prior, probe_models, emb_scaler, emb_pca, alpha_blend)
    first_pass_tr = (ENSEMBLE_W * proto_tr_flat + (1.0 - ENSEMBLE_W) * sc_tr_mlp)
    
    train_probs_for_calib = sigmoid(first_pass_tr)
    PER_CLASS_THRESHOLDS = calibrate_and_optimize_thresholds(
        oof_probs=train_probs_for_calib, Y_FULL=Y_FULL_aligned,
        threshold_grid=[0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70], n_windows=N_WINDOWS)
    
    t0 = time.time()
    res_model, correction_weight = train_residual_ssm(
        emb_full=emb_tr, first_pass_flat=first_pass_tr, Y_full=Y_FULL_aligned,
        site_ids=tr_site_ids, hour_ids=tr_hour_ids, n_epochs=30, patience=8, lr=1e-3,
        correction_weight=0.30, verbose=False)
    print(f"ResidualSSM training: {time.time()-t0:.1f}s")
    
    first_pass_te_f = first_pass_flat.reshape(n_test_files, N_WINDOWS, -1)
    res_model.eval()
    with torch.no_grad():
        test_correction = res_model(
            torch.tensor(emb_te_f,         dtype=torch.float32),
            torch.tensor(first_pass_te_f,  dtype=torch.float32),
            site_ids=torch.tensor(test_site_ids, dtype=torch.long),
            hours   =torch.tensor(test_hour_ids, dtype=torch.long),
        ).numpy()
    correction_flat = test_correction.reshape(-1, N_CLASSES).astype(np.float32)
    final_scores    = first_pass_flat + correction_weight * correction_flat
    final_scores    = final_scores / temperatures[None, :]
    probs = sigmoid(final_scores)
    probs = file_confidence_scale(probs, n_windows=N_WINDOWS, top_k=2, power=0.4)
    probs = rank_aware_scaling(probs,    n_windows=N_WINDOWS, power=0.4)
    probs = adaptive_delta_smooth(probs, n_windows=N_WINDOWS, base_alpha=0.20)
    probs = np.clip(probs, 0.0, 1.0)
    probs = apply_per_class_thresholds(probs, PER_CLASS_THRESHOLDS)   # ? now applied
    
    sub = pd.DataFrame(probs.astype(np.float32), columns=PRIMARY_LABELS)
    sub.insert(0, "row_id", meta_te["row_id"].values)
    sub.to_csv("submission_protossm.csv", index=False)
    print("ProtoSSM execution complete")
    print(f"Total wall time so far: {(time.time() - _WALL_START)/60:.1f} min")
    del emb_tr_f, sc_tr_f, proto_model, res_model
    gc.collect()
    print("Memory freed. Ready for SED cell.")
    
    
    import librosa
    from scipy.ndimage import gaussian_filter1d
    
    N_MELS_SED = 256
    N_FFT_SED  = 2048
    HOP_SED    = 512
    FMIN_SED   = 20
    FMAX_SED   = 16000
    TOP_DB_SED = 80
    
    def find_sed_dir():
        hits = sorted(Path("/kaggle/input").rglob("sed_fold0.onnx"))
        if not hits:
            raise FileNotFoundError("sed_fold0.onnx not found. Attach tuckerarrants/bc2026-distilled-sed-public.")
        return hits[0].parent
    
    def make_sed_session(path):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
    
    def audio_to_mel(chunks):
        mels = []
        for x in chunks:
            s = librosa.feature.melspectrogram(y=x, sr=SR, n_fft=N_FFT_SED, hop_length=HOP_SED,
                                                n_mels=N_MELS_SED, fmin=FMIN_SED, fmax=FMAX_SED, power=2.0)
            s = librosa.power_to_db(s, top_db=TOP_DB_SED)
            s = (s - s.mean()) / (s.std() + 1e-6)
            mels.append(s)
        return np.stack(mels)[:, None].astype(np.float32)
    
    def file_to_sed_chunks(path):
        y, sr0 = sf.read(str(path), dtype="float32", always_2d=False)
        if y.ndim == 2: y = y.mean(axis=1)
        if sr0 != SR: y = librosa.resample(y, orig_sr=sr0, target_sr=SR)
        n = 60 * SR
        if len(y) < n: y = np.pad(y, (0, n - len(y)))
        else:          y = y[:n]
        chunks = y.reshape(N_WINDOWS, WINDOW_SAMPLES)
        ends   = np.arange(1, N_WINDOWS + 1) * WINDOW_SEC
        return chunks, ends
    
    def sigmoid_sed(x):
        return (1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))).astype(np.float32)
    
    # Use the same test files as Cell 1
    test_paths = sorted((BASE / "test_soundscapes").glob("*.ogg"))
    IS_DRY_RUN = len(test_paths) == 0
    if IS_DRY_RUN:
        dry_n = CFG["dryrun_n_files"] if "CFG" in dir() else 20
        test_paths = sorted((BASE / "train_soundscapes").glob("*.ogg"))[:(dry_n or 20)]
    
    sed_dir = find_sed_dir()
    sed_fold_paths = sorted(sed_dir.glob("sed_fold*.onnx"),
                             key=lambda p: int(re.search(r"sed_fold(\d+)", p.name).group(1)))
    sed_sessions = [make_sed_session(p) for p in sed_fold_paths]
    
    print(f"SED dir: {sed_dir}")
    print(f"SED folds loaded: {[p.name for p in sed_fold_paths]}")
    
    sed_rows, sed_preds = [], []
    
    for i, path in enumerate(test_paths, 1):
        chunks, ends = file_to_sed_chunks(path)
        mel = audio_to_mel(chunks)
        p_sum = np.zeros((len(chunks), N_CLASSES), dtype=np.float32)
    
        for sess in sed_sessions:
            outs = sess.run(None, {sess.get_inputs()[0].name: mel})
            clip_logits = outs[0]             # (12, 234)
            frame_max   = outs[1].max(axis=1) # (12, 234)
            p_sum += 0.5 * sigmoid_sed(clip_logits) + 0.5 * sigmoid_sed(frame_max)
    
        p_mean = p_sum / len(sed_sessions)
    
        if len(p_mean) > 1:
            p_mean = gaussian_filter1d(p_mean, sigma=0.65, axis=0, mode="nearest").astype(np.float32)
    
        stem = path.stem
        sed_rows.extend([f"{stem}_{int(t)}" for t in ends])
        sed_preds.append(p_mean)
    
        if i == 1 or i % 50 == 0 or i == len(test_paths):
            print(f"SED: {i}/{len(test_paths)}")
    
    sed_preds_arr = np.concatenate(sed_preds, axis=0)
    sed_sub = pd.DataFrame(np.clip(sed_preds_arr, 0.0, 1.0), columns=PRIMARY_LABELS)
    sed_sub.insert(0, "row_id", sed_rows)
    sed_sub.to_csv("submission_sed.csv", index=False)
    print(f"Distilled SED Processing Complete. Shape: {sed_sub.shape}")
    
    
    # ??????????????????????????????????????????????????????????????????????????????
    # BirdNET v2.4 ? Third model branch (NEW)
    # Uses: shadiakiki1/birdnet-analyzer/liteRT/birdnet_global_6k_v2.4_model_fp32-1
    # This cell is safe to run even if BirdNET model is not attached ?
    # it will gracefully set USE_BIRDNET=False and skip.
    # ??????????????????????????????????????????????????????????????????????????????
    import librosa as _librosa
    from scipy.ndimage import gaussian_filter1d as _gf1d
    
    BIRDNET_SR = 48_000
    BIRDNET_CHUNK_SEC = 3
    BIRDNET_CHUNK_SAMPLES = BIRDNET_SR * BIRDNET_CHUNK_SEC  # 144000
    
    def _find_birdnet_model():
        for pat in ["**/birdnet_global_6k_v2.4_model_fp32*.tflite",
                    "**/BirdNET_GLOBAL_6K_V2.4_Model_FP32*.tflite",
                    "**/*birdnet*fp32*.tflite",
                    "**/*birdnet*.tflite"]:
            hits = sorted(Path("/kaggle/input").rglob(pat))
            if hits:
                print(f"  Found BirdNET model: {hits[0].name}")
                return hits[0]
        return None
    
    def _find_birdnet_labels():
        for pat in ["**/BirdNET_GLOBAL_6K_V2.4_Labels.txt",
                    "**/birdnet*labels*.txt",
                    "**/*birdnet*label*.txt"]:
            hits = sorted(Path("/kaggle/input").rglob(pat))
            if hits:
                print(f"  Found BirdNET labels: {hits[0].name}")
                return hits[0]
        return None
    
    _bn_model_path  = _find_birdnet_model()
    _bn_labels_path = _find_birdnet_labels()
    
    if _bn_model_path is None:
        USE_BIRDNET = False
        print("BirdNET model not found ! will use original 60/40 blend")
        BN_TO_COMP = {}
        BN_PROXY   = {}
    else:
        USE_BIRDNET = True
        try:
            from tflite_runtime.interpreter import Interpreter as _TFLiteInterp
        except ImportError:
            from tensorflow.lite.python.interpreter import Interpreter as _TFLiteInterp
    
        _bn_interp = _TFLiteInterp(model_path=str(_bn_model_path), num_threads=4)
        _bn_interp.allocate_tensors()
        _bn_in  = _bn_interp.get_input_details()[0]
        _bn_out = _bn_interp.get_output_details()
        _bn_logit_idx = _bn_out[-1]["index"]
        print(f"  BirdNET input:  {_bn_in['shape']}")
        print(f"  BirdNET output: {_bn_out[-1]['shape']}")
    
        # Load labels
        if _bn_labels_path:
            _bn_labels_raw = [l.strip() for l in _bn_labels_path.read_text().splitlines() if l.strip()]
        else:
            _bn_labels_raw = []
            print("  WARNING: BirdNET labels not found ? species mapping will be empty")
    
        # Parse scientific names (before first underscore)
        _bn_sci = [lbl.split("_", 1)[0].strip() for lbl in _bn_labels_raw]
    
        # Direct mapping: BirdNET index ? competition class index via scientific name
        _tax_sci = taxonomy.set_index("scientific_name")["primary_label"].to_dict()
        BN_TO_COMP = {}
        for bn_i, sci in enumerate(_bn_sci):
            if sci in _tax_sci and _tax_sci[sci] in label_to_idx:
                BN_TO_COMP[bn_i] = label_to_idx[_tax_sci[sci]]
    
        # Genus-level proxy for unmapped competition classes
        _mapped_comp = set(BN_TO_COMP.values())
        BN_PROXY = {}  # comp_class_idx ? list of bn_indices
        for ci, primary in enumerate(PRIMARY_LABELS):
            if ci in _mapped_comp:
                continue
            row = taxonomy[taxonomy["primary_label"] == primary]
            if row.empty:
                continue
            genus = str(row.iloc[0]["scientific_name"]).split()[0]
            idxs = [i for i, s in enumerate(_bn_sci) if s.startswith(genus + " ")]
            if idxs:
                BN_PROXY[ci] = idxs
    
        n_mapped = len(set(BN_TO_COMP.values()) | set(BN_PROXY.keys()))
        print(f"  BirdNET?competition: {len(BN_TO_COMP)} direct + {len(BN_PROXY)} genus-proxy = {n_mapped}/{N_CLASSES} classes")
    
    # Window overlap: each 5s competition window overlaps certain 3s BirdNET chunks
    _N_BN_CHUNKS = 20  # 60s / 3s
    _win_to_chunks = []
    for w in range(N_WINDOWS):
        ws, we = w * 5, (w + 1) * 5
        _win_to_chunks.append([j for j in range(_N_BN_CHUNKS)
                                if 3*j < we and 3*(j+1) > ws])
    
    def run_birdnet(paths, verbose=True):
        if not USE_BIRDNET or not _bn_labels_raw:
            return None, None
    
        paths = [Path(p) for p in paths]
        n_rows = len(paths) * N_WINDOWS
        row_ids   = np.empty(n_rows, dtype=object)
        filenames = np.empty(n_rows, dtype=object)
        scores    = np.zeros((n_rows, N_CLASSES), dtype=np.float32)
        wr = 0
        itr = tqdm(paths, desc="BirdNET") if verbose else paths
    
        for path in itr:
            y, sr0 = sf.read(str(path), dtype="float32", always_2d=False)
            if y.ndim == 2: y = y.mean(axis=1)
            if sr0 != BIRDNET_SR:
                y = _librosa.resample(y, orig_sr=sr0, target_sr=BIRDNET_SR)
            tgt = 60 * BIRDNET_SR
            if len(y) < tgt: y = np.pad(y, (0, tgt - len(y)))
            else:             y = y[:tgt]
    
            chunks = y.reshape(_N_BN_CHUNKS, BIRDNET_CHUNK_SAMPLES)
            chunk_probs = np.zeros((_N_BN_CHUNKS, len(_bn_labels_raw)), dtype=np.float32)
            for j, chunk in enumerate(chunks):
                _bn_interp.set_tensor(_bn_in["index"], chunk[None, :].astype(np.float32))
                _bn_interp.invoke()
                logits = _bn_interp.get_tensor(_bn_logit_idx)[0]
                chunk_probs[j] = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    
            # Aggregate 3s chunks ? 5s competition windows via max pooling
            stem = path.stem
            for w, clist in enumerate(_win_to_chunks):
                wp = chunk_probs[clist].max(axis=0)
                r = wr + w
                row_ids[r]   = f"{stem}_{(w+1)*5}"
                filenames[r] = path.name
                for bn_i, ci in BN_TO_COMP.items():
                    if wp[bn_i] > scores[r, ci]:
                        scores[r, ci] = wp[bn_i]
                for ci, bn_idxs in BN_PROXY.items():
                    v = wp[bn_idxs].max()
                    if v > scores[r, ci]:
                        scores[r, ci] = v
            wr += N_WINDOWS
    
        meta_df = pd.DataFrame({"row_id": row_ids[:wr], "filename": filenames[:wr]})
        return meta_df, scores[:wr]
    
    print("BirdNET inference engine defined")
    
    # Run BirdNET on test files
    _test_paths_bn = sorted((BASE / "test_soundscapes").glob("*.ogg"))
    if len(_test_paths_bn) == 0:
        _dry_n = 20
        _test_paths_bn = sorted((BASE / "train_soundscapes").glob("*.ogg"))[:_dry_n]
    
    if USE_BIRDNET:
        t0 = time.time()
        _meta_bn, _scores_bn = run_birdnet(_test_paths_bn, verbose=True)
        print(f"BirdNET inference: {time.time()-t0:.1f}s  shape={_scores_bn.shape}")
    
        # Smooth BirdNET predictions (same sigma as SED)
        _scores_bn_v = _scores_bn.reshape(len(_scores_bn)//N_WINDOWS, N_WINDOWS, N_CLASSES)
        for fi in range(len(_scores_bn_v)):
            _scores_bn_v[fi] = _gf1d(_scores_bn_v[fi], sigma=0.65, axis=0, mode="nearest")
        _scores_bn = _scores_bn_v.reshape(-1, N_CLASSES)
    
        _bn_sub = pd.DataFrame(np.clip(_scores_bn, 0.0, 1.0), columns=PRIMARY_LABELS)
        _bn_sub.insert(0, "row_id", _meta_bn["row_id"].values)
        _bn_sub.to_csv("submission_birdnet.csv", index=False)
        covered = (_scores_bn > 0.01).any(axis=0).sum()
        print(f"BirdNET saved. Coverage: {covered}/{N_CLASSES} classes")
    else:
        # Save zero-filled CSV so blend cell always has the file
        _dummy = pd.read_csv("submission_protossm.csv")
        for c in PRIMARY_LABELS: _dummy[c] = 0.0
        _dummy.to_csv("submission_birdnet.csv", index=False)
        print("BirdNET unavailable ? zero submission saved")
    
    
    import os
    import numpy as np
    import pandas as pd
    from pathlib import Path
    
    PROTOSSM_CSV = "submission_protossm.csv"
    SED_CSV      = "submission_sed.csv"
    OUT_CSV      = "submission.csv"
    EPS = 1e-5
    
    df_proto = pd.read_csv(PROTOSSM_CSV)
    df_sed   = pd.read_csv(SED_CSV)
    
    cols = [c for c in df_proto.columns if c != "row_id"]
    
    # Align row order
    df_sed = df_sed.set_index("row_id").loc[df_proto["row_id"]].reset_index()
    p_proto = np.clip(df_proto[cols].to_numpy(np.float32), EPS, 1.0 - EPS)
    p_sed   = np.clip(df_sed[cols].to_numpy(np.float32),   EPS, 1.0 - EPS)
    
    rank_proto = pd.DataFrame(p_proto).rank(axis=0, pct=True).to_numpy(np.float32)
    rank_sed   = pd.DataFrame(p_sed).rank(axis=0, pct=True).to_numpy(np.float32)
    
    # ?? Standard 60/40 rank blend ?????????????????????????????????????????????????
    # ?? Load BirdNET submission ????????????????????????????????????????????????????
    try:
        df_birdnet = pd.read_csv("submission_birdnet.csv")
        df_birdnet = df_birdnet.set_index("row_id").loc[df_proto["row_id"]].reset_index()
        p_birdnet  = np.clip(df_birdnet[cols].to_numpy(np.float32), EPS, 1.0 - EPS)
        _bn_active = (p_birdnet > 0.01).any()
        if _bn_active:
            rank_birdnet = pd.DataFrame(p_birdnet).rank(axis=0, pct=True).to_numpy(np.float32)
            print("Executing 3-way rank blend (50% Proto / 30% SED / 20% BirdNET)...")
        else:
            rank_birdnet = None
            print("BirdNET scores all zero ? using original 60/40 blend")
    except Exception as _e:
        rank_birdnet = None
        print(f"BirdNET load failed ({_e}) ? using original 60/40 blend")
    
    
    print("Executing standard 2-way rank blend (60% Proto / 40% SED)...")
    if rank_birdnet is not None:
        pred = (rank_proto * 0.50) + (rank_sed * 0.30) + (rank_birdnet * 0.20)
    else:
        pred = (rank_proto * 0.595) + (rank_sed * 0.405)
        
    
    row_ids  = df_proto["row_id"].astype(str).to_numpy()
    file_ids = np.array(["_".join(r.split("_")[:-1]) for r in row_ids])
    
    # ?? Gate 1: Noise suppression ?????????????????????????????????????????????????
    # If ProtoSSM is confident but SED strongly disagrees ? trust ProtoSSM more
    fake_only = (p_proto > 0.50) & (p_sed < 0.05)
    pred = np.where(fake_only, (1.0 - 0.08) * pred + 0.08 * rank_proto, pred)
    
    # ?? Gate 2: Temporal continuity (fat-tailed t-distribution kernel) ?????????????
    # 35-second context window to protect continuous calls across windows
    offs = np.arange(-3, 4, dtype=np.float32)
    proto_kernel = (1.0 + (offs / 1.20) ** 2 / 2.0) ** (-1.5)
    proto_kernel = (proto_kernel / proto_kernel.sum()).astype(np.float32)
    
    pa_ctx = p_proto.copy()
    for fid in pd.unique(file_ids):
        m  = file_ids == fid
        x  = p_proto[m]
        if len(x) > 1:
            xp = np.pad(x, ((3, 3), (0, 0)), mode="edge")
            pa_ctx[m] = sum(proto_kernel[i] * xp[i:i + len(x)] for i in range(7))
    
    xctx = pd.DataFrame(pa_ctx).rank(axis=0, pct=True).to_numpy(np.float32)
    proto_cont = (xctx > 0.88) & (rank_proto > 0.75) & (p_sed < 0.12) & (~fake_only)
    pred = np.where(proto_cont,
                    (1.0 - 0.15) * pred + 0.15 * np.maximum(rank_proto, xctx),
                    pred)
    
    # ?? Gate 3: SED spike preservation ????????????????????????????????????????????
    # Brief high-confidence SED detections that ProtoSSM missed
    sed_only = (rank_sed > 0.95) & (rank_proto < 0.80) & (~fake_only) & (~proto_cont)
    pred = np.where(sed_only, (1.0 - 0.12) * pred + 0.12 * rank_sed, pred)
    # ?? Gate 3b: BirdNET spike preservation (NEW) ?????????????????????????????????
    # Brief high-confidence BirdNET detections that both ProtoSSM and SED missed.
    # Particularly helps unmapped species where Perch has no logit.
    if rank_birdnet is not None:
        bn_only = (rank_birdnet > 0.95) & (rank_proto < 0.75) & (rank_sed < 0.80) & (~fake_only) & (~proto_cont) & (~sed_only)
        pred = np.where(bn_only, (1.0 - 0.10) * pred + 0.10 * rank_birdnet, pred)
    
    
    sub = df_proto.copy()
    sub[cols] = pred.astype(np.float32)
    
    # ?? Gate 4: Sonotype mirroring ????????????????????????????????????????????????
    # Max-pool across visually identical species groups
    MIRROR_PAIRS = (
        ("47158son15", "47158son16"),
        ("47158son09", "47158son12"),
        ("47158son02", "47158son14"),
        ("47158son13", "47158son21", "47158son22", "47158son23")
    )
    col_to_idx = {l: i for i, l in enumerate(cols)}
    
    mirror_count = 0
    for group in MIRROR_PAIRS:
        valid_idx = [col_to_idx[s] for s in group if s in col_to_idx]
        if len(valid_idx) >= 2:
            group_max = sub[cols].iloc[:, valid_idx].max(axis=1).to_numpy(np.float32)
            for idx in valid_idx:
                sub.iloc[:, idx + 1] = group_max
            mirror_count += len(valid_idx)
    print(f"Sonotype mirroring applied to {mirror_count} columns.")
    
    # ?? Gate 5: Adaptive rare-class thresholding ??????????????????????????????????
    BASE_PATH = Path("/kaggle/input/competitions/birdclef-2026")
    try:
        tax_df = pd.read_csv(BASE_PATH / "taxonomy.csv").set_index("primary_label")
        rare_classes = {"Amphibia", "Mammalia", "Reptilia"}
        rare_count = 0
        for ci, species in enumerate(cols):
            if species in tax_df.index and tax_df.loc[species, "class_name"] in rare_classes:
                col_idx = ci + 1
                vals = sub.iloc[:, col_idx].to_numpy(np.float32)
                thr = vals.mean() + 0.05
                sub.iloc[:, col_idx] = np.where(vals < thr, vals * 0.9, vals)
                rare_count += 1
        print(f"Adaptive thresholding applied to {rare_count} rare species.")
    except Exception as e:
        print(f"Adaptive thresholding skipped: {e}")
    
    # ?? Dry-run alignment ??????????????????????????????????????????????????????????
    test_paths = list(BASE_PATH.glob("test_soundscapes/*.ogg"))
    IS_DRY_RUN = len(test_paths) == 0
    if IS_DRY_RUN:
        print("Dry-run detected: Aligning rows with sample_submission.csv")
        sample_public = pd.read_csv(BASE_PATH / "sample_submission.csv")
        template = sub[cols].mean(axis=0).astype(np.float32)
        sub = sample_public.copy()
        for label in cols:
            sub[label] = template[label]
    
    sub.to_csv("subm_4.csv", index=False)
    print(f"Blend and post-processing complete. Saved {OUT_CSV} shape={sub.shape}")
    print("Ready for submission!")
    
    
    # Final submission diagnostics: does not alter submission.csv
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from IPython.display import display, Markdown
    
    submission_path = Path("subm_4.csv")
    assert submission_path.exists(), "submission.csv was not created. Run the blend cell first."
    
    sub_check = pd.read_csv(submission_path)
    prob_cols = [c for c in sub_check.columns if c != "row_id"]
    
    summary = pd.DataFrame({
        "check": [
            "rows",
            "columns",
            "class columns",
            "missing values",
            "min probability",
            "max probability",
            "duplicated row_id",
        ],
        "value": [
            len(sub_check),
            sub_check.shape[1],
            len(prob_cols),
            int(sub_check.isna().sum().sum()),
            float(sub_check[prob_cols].min().min()) if prob_cols else np.nan,
            float(sub_check[prob_cols].max().max()) if prob_cols else np.nan,
            int(sub_check["row_id"].duplicated().sum()) if "row_id" in sub_check.columns else "row_id missing",
        ]
    })
    
    display(Markdown("### ? Submission diagnostic summary"))
    display(summary)
    
    assert "row_id" in sub_check.columns, "row_id column is missing."
    assert len(prob_cols) > 0, "No class probability columns found."
    assert np.isfinite(sub_check[prob_cols].to_numpy()).all(), "Non-finite values found in probability columns."
    assert sub_check[prob_cols].min().min() >= 0.0, "Probability columns contain values below 0."
    assert sub_check[prob_cols].max().max() <= 1.0, "Probability columns contain values above 1."
    
    print("submission.csv passed basic diagnostics.")



# === ## blend

# === CELL 6 ===

import pandas as pd, os, time, sys
from pathlib import Path
from warnings import filterwarnings; filterwarnings("ignore")


def direct_add2():
    try:
        print(f'Ensemble: {_ensemble_models},   LB: {_lbs},   weights: {_weights}')
        df0 = pd.read_csv(_files_subm[0], index_col = "row_id")              ;display(df0)
        df1 = pd.read_csv(_files_subm[1], index_col = "row_id")[df0.columns] ;display(df1)
        dfs = df0 * _weights[0] + _weights[1] * df1
        return dfs
    except Exception as e:
        print(f"Direct blend 2 failed with error: {e}")
        print("Dry-run detected or shape mismatch. Gracefully falling back to sample_submission.csv template.")
        COMP = Path('/kaggle/input/competitions/birdclef-2026')
        if not COMP.exists(): COMP = Path('/kaggle/input/birdclef-2026')
        return pd.read_csv(COMP / 'sample_submission.csv', index_col="row_id")


def direct_add3():
    try:
        print(f'Ensemble: {_ensemble_models},   LB: {_lbs},   weights: {_weights}')
        df0 = pd.read_csv(_files_subm[0], index_col = "row_id")              ;display(df0)
        df1 = pd.read_csv(_files_subm[1], index_col = "row_id")[df0.columns] ;display(df1)
        df2 = pd.read_csv(_files_subm[2], index_col = "row_id")[df0.columns] ;display(df2)
        dfs = \
            df0 * _weights[0] + \
            df1 * _weights[1] + \
            df2 * _weights[2]
        return dfs
    except Exception as e:
        print(f"Direct blend 3 failed with error: {e}")
        print("Dry-run detected or shape mismatch. Gracefully falling back to sample_submission.csv template.")
        COMP = Path('/kaggle/input/competitions/birdclef-2026')
        if not COMP.exists(): COMP = Path('/kaggle/input/birdclef-2026')
        return pd.read_csv(COMP / 'sample_submission.csv', index_col="row_id")


def rank_1_add2():
    try:
        PROTOSSM_W = _weights[0]
        SED_W      = _weights[1]
        EPS        = 1e-5
        PROTO      = pd.read_csv(_files_subm[0])
        SED        = pd.read_csv(_files_subm[1]) 
        cols       = [c for c in SED.columns if c != "row_id"]
        missing    = sorted(set(SED["row_id"]) - set(PROTO["row_id"]))
        if missing:
            raise ValueError(f"row_id mismatch: {len(missing)} rows in ProtoSSM not in SED; first={missing[:3]}")
        PROTO      = PROTO.set_index("row_id").loc[SED["row_id"]].reset_index()
        pSED       = np.clip(SED  [cols].to_numpy("float32"), EPS, 1.0 - EPS)
        pPROTO     = np.clip(PROTO[cols].to_numpy("float32"), EPS, 1.0 - EPS)
        wSED       = PROTOSSM_W / (PROTOSSM_W + SED_W)
        wPROTO     = SED_W      / (PROTOSSM_W + SED_W)
        rSED       = pd.DataFrame(pSED)  .rank(axis=0, pct=True).to_numpy("float32")
        rPROTO     = pd.DataFrame(pPROTO).rank(axis=0, pct=True).to_numpy("float32")
        pred       = wSED * rSED  +  rPROTO * wPROTO 
        subm       = SED.copy()
        subm[cols] = pred.astype("float32")
        return subm
    except Exception as e:
        print(f"Rank blend 2 failed with error: {e}")
        print("Dry-run detected or shape mismatch. Gracefully falling back to sample_submission.csv template.")
        COMP = Path('/kaggle/input/competitions/birdclef-2026')
        if not COMP.exists(): COMP = Path('/kaggle/input/birdclef-2026')
        return pd.read_csv(COMP / 'sample_submission.csv')


def rank_1_add3():
    try:
        PROTO_w1    = _weights[0]
        PROTO_w2    = _weights[1]
        SED_w       = _weights[2]
        EPS         = 1e-5
        PROTO_1     = pd.read_csv(_files_subm[0])
        PROTO_2     = pd.read_csv(_files_subm[1])
        SED         = pd.read_csv(_files_subm[2]) 
        cols        = [c for c in SED.columns if c != "row_id"]
        missing1    = sorted(set(SED["row_id"]) - set(PROTO_1["row_id"]))
        missing2    = sorted(set(SED["row_id"]) - set(PROTO_2["row_id"]))
        if missing1 :
            raise ValueError(f"row_id mismatch: {len(missing1)} rows in ProtoSSM not in SED; first={missing1[:3]}")
        if missing2 :
            raise ValueError(f"row_id mismatch: {len(missing2)} rows in ProtoSSM not in SED; first={missing2[:3]}")
        PROTO_1     = PROTO_1.set_index("row_id").loc[SED["row_id"]].reset_index()
        PROTO_2     = PROTO_2.set_index("row_id").loc[SED["row_id"]].reset_index()
        pSED        = np.clip(SED    [cols].to_numpy("float32"), EPS, 1.0 - EPS)
        pPROTO_1    = np.clip(PROTO_1[cols].to_numpy("float32"), EPS, 1.0 - EPS)
        pPROTO_2    = np.clip(PROTO_2[cols].to_numpy("float32"), EPS, 1.0 - EPS)
        
        # Robust and correct weight normalization
        total_w     = PROTO_w1 + PROTO_w2 + SED_w
        wPROTO_1    = PROTO_w1 / total_w
        wPROTO_2    = PROTO_w2 / total_w
        wSED        = SED_w / total_w
        
        rSED        = pd.DataFrame(pSED)    .rank(axis=0, pct=True).to_numpy("float32")
        rPROTO_1    = pd.DataFrame(pPROTO_1).rank(axis=0, pct=True).to_numpy("float32")
        rPROTO_2    = pd.DataFrame(pPROTO_2).rank(axis=0, pct=True).to_numpy("float32")
        pred        = wSED * rSED   +   rPROTO_1 * wPROTO_1   +   rPROTO_2 * wPROTO_2 
        subm        = SED.copy()
        subm[cols]  = pred.astype("float32")
        return subm
    except Exception as e:
        print(f"Rank blend 3 failed with error: {e}")
        print("Dry-run detected or shape mismatch. Gracefully falling back to sample_submission.csv template.")
        COMP = Path('/kaggle/input/competitions/birdclef-2026')
        if not COMP.exists(): COMP = Path('/kaggle/input/birdclef-2026')
        return pd.read_csv(COMP / 'sample_submission.csv')


def direct():
    if len(_ensemble_models) == 2:  return direct_add2()
    if len(_ensemble_models) == 3:  return direct_add3()


def rank_1():
    if len(_ensemble_models) == 2:  return rank_1_add2()
    if len(_ensemble_models) == 3:  return rank_1_add3()

# === CELL 7 ===

if solutions['type_add'] == 'rank.1' : f_add = rank_1
if solutions['type_add'] in ['direct', 'TAX_SMOOTHING'] : f_add = direct

# === ## Submit

# === CELL 8 ===

submission = f_add()



# === CELL 9 ===

# V11: Taxonomic smoothing (genus + class level)
# Adapted from Nina EoS v9 TAX_SMOOTHING_POSTPROC
import pandas as pd, numpy as np, os

# Load taxonomy
tax_paths = [
    "/kaggle/input/competitions/birdclef-2026/taxonomy.csv",
    "/kaggle/input/birdclef-2026/taxonomy.csv",
]
tax = None
for p in tax_paths:
    if os.path.exists(p):
        tax = pd.read_csv(p)
        break

if tax is not None:
    species_to_genus = {}
    species_to_class = {}
    for _, r in tax.iterrows():
        label = str(r["primary_label"])
        sci = str(r.get("scientific_name", ""))
        cls = str(r.get("class_name", ""))
        genus = sci.split(" ")[0] if " " in sci else sci
        species_to_genus[label] = genus
        species_to_class[label] = cls

    cols = list(submission.columns)
    genus_groups = {}
    class_groups = {}
    for c in cols:
        g = species_to_genus.get(c, c)
        cl = species_to_class.get(c, "")
        genus_groups.setdefault(g, []).append(c)
        if cl:
            class_groups.setdefault(cl, []).append(c)

    multi_genus = {g: m for g, m in genus_groups.items() if len(m) > 1}
    multi_class = {cl: m for cl, m in class_groups.items() if len(m) > 1}
    print(f"Tax smoothing: {len(multi_genus)} multi-genus groups, {len(multi_class)} multi-class groups")

    alpha_genus = 0.15
    alpha_class = 0.05
    probs = submission[cols].values.copy()

    # Genus-level smoothing
    for members in multi_genus.values():
        idxs = [cols.index(m) for m in members if m in cols]
        if len(idxs) < 2:
            continue
        mean_probs = probs[:, idxs].mean(axis=1, keepdims=True)
        probs[:, idxs] = (1 - alpha_genus) * probs[:, idxs] + alpha_genus * mean_probs

    # Class-level smoothing  
    for members in multi_class.values():
        idxs = [cols.index(m) for m in members if m in cols]
        if len(idxs) < 2:
            continue
        mean_probs = probs[:, idxs].mean(axis=1, keepdims=True)
        probs[:, idxs] = (1 - alpha_class) * probs[:, idxs] + alpha_class * mean_probs

    submission[cols] = np.clip(probs, 0.0, 1.0)
    print("Taxonomic smoothing applied")
else:
    print("taxonomy.csv not found - skipping tax smoothing")



# === CELL 10 ===

submission.to_csv(f"submission.csv", index = True)
submission



# === CELL 11 ===

# [EoS V2] Dry-run override
import pandas as pd
from pathlib import Path
COMP = Path('/kaggle/input/competitions/birdclef-2026')
if not COMP.exists(): COMP = Path('/kaggle/input/birdclef-2026')
sample_sub = pd.read_csv(COMP / 'sample_submission.csv')
sub = pd.read_csv('/kaggle/working/submission.csv')
print(f'Sample: {len(sample_sub)} | Submission: {len(sub)}')
if len(sub) != len(sample_sub):
    print('DRY-RUN: overriding')
    sample_sub.to_csv('/kaggle/working/submission.csv', index=False)
sub2 = pd.read_csv('/kaggle/working/submission.csv')
assert sub2.shape == sample_sub.shape
print('VALID: submission.csv ready')


