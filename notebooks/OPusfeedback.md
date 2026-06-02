Analyse complète du notebook BirdCLEF 2026 — Pistes d'amélioration
Score actuel : 0.948. Voici un inventaire exhaustif des leviers d'amélioration, classés par catégorie.

1. ARCHITECTURE & BACKBONE DES MODÈLES
Model_2 (SED distillé — LB 0.917, poids 0.03)
Backbone trop petit : tf_efficientnet_b0 est le plus léger de la famille. Passer à EfficientNet-B3/B4 ou ConvNeXt-V2-Tiny/Small donnerait un meilleur feature map avec un coût raisonnable.
EfficientNet-V2 ou EAT (Efficient Audio Transformer) : des backbones audio-spécifiques (BEATs, HTS-AT, AST) sont plus performants que des backbones ImageNet adaptés.
Stop gradient trop agressif : h_cls = h.detach() coupe totalement le gradient du backbone côté SED. Essayer un partial detach (gradient scaling factor 0.1–0.3) ou un scheduling progressif (détacher les premières epochs puis laisser passer).
Nombre de mels : 256 mels mais l'export ONNX mentionne INF_N_MELS = 128 — incohérence potentielle entre train et inférence.
GeMFreq p_init=3.0 : explorer des valeurs différentes (p=4 ou p=2) ou un attention pooling sur la fréquence plutôt qu'un GeM.
Hidden dim : le bottleneck est fixé à 512. Tester 1024 ou un MLP plus profond (2 couches).
Pas de multi-scale : ajouter un Feature Pyramid ou multi-resolution mel (2 hop lengths différents, concaténés).
Model_3 (ProtoSSM v5 — LB 0.928, poids 0.10)
d_model=320, n_ssm_layers=4 : déjà assez gros mais le modèle reste CPU-only. Forcer l'inférence GPU si disponible.
Selective SSM est séquentiel : la boucle for t in range(T) est lente. Implémenter un parallel scan (comme dans Mamba officiel) ou utiliser directement mamba_ssm de Tri Dao.
Cross-attention limitée : 2 têtes seulement dans certains configs. Augmenter à 8 têtes, et ajouter du relative positional bias (Alibi ou RoPE adapté 1D).
Prototypes figés après init : init_prototypes_from_data est appelé une seule fois. Faire un momentum update des prototypes pendant l'entraînement (comme dans MoCo/DINO).
Pas de data augmentation temporelle : TTA par shifts circulaires uniquement. Ajouter du time stretching, pitch shifting, ou du random window cropping pendant le train.
MLP probes : les MLPClassifier de sklearn sont sous-optimaux. Remplacer par un petit réseau PyTorch avec batch norm, entraîné end-to-end ou avec des embeddings gelés + fine-tuning.
Perch embeddings non fine-tunés : les 1536-d de Perch sont utilisés tels quels. Un adaptation layer fine-tuné (LoRA-style) sur les données de train pourrait aider.
Model_4 (Pipeline ProtoSSM+SED+BirdNET — LB 0.947, poids 0.87)
LightProtoSSM est plus petit que le ProtoSSM de Model_3 (128 vs 320 d_model, 2 vs 4 layers). Unifier vers la version la plus grosse.
BirdNET v2.4 en tflite : la résolution 3s chunks est sous-optimale pour le format 5s de la compétition. Envisager un overlap-and-average plus fin.
SED folds = 5 avec des exports ONNX : vérifier que tous les 5 folds sont utilisés et pas juste un sous-ensemble.
2. ENTRAÎNEMENT
Régime d'entraînement SED (Model_2)
25 epochs seulement : pour un EfficientNet-B0 avec distillation, c'est court. Passer à 50-80 epochs avec cosine annealing.
Batch size 64 localement mais 16 sur Kaggle : ça change la dynamique du batch norm et du LR. Utiliser du gradient accumulation pour simuler batch 64.
Pas de SAM/ASAM : l'optimiseur Sharpness-Aware Minimization améliore la généralisation en audio. Coût ~2x mais gain significatif.
Pas de label smoothing sur le SED (contrairement au ProtoSSM qui en a).
Perch distillation loss = MSE : essayer une cosine similarity loss ou contrastive loss (InfoNCE) qui aligne mieux les espaces d'embedding.
ALPHA_DISTILL=1.0 : le poids de la distillation est très élevé par rapport au BCE. Sweeper entre 0.1 et 2.0.
MixUp seulement focal-focal et focal-soundscape : ajouter du CutMix sur spectrogrammes (découpage de patches mel) et du SpecMix (mélange de bandes de fréquence entre échantillons).
Régime ProtoSSM (Models 3 & 4)
Focal gamma=2.5 : très agressif. Tester 1.0 et 2.0 aussi.
SWA commence à 65% : peut-être trop tard avec early stopping à patience=20. Réduire swa_start_frac à 0.5.
Pas de curriculum learning : commencer avec des espèces faciles (Aves communes) puis ajouter progressivement les rares.
OOF CV : utiliser StratifiedGroupKFold (déjà fait) mais avec repeated CV (3 repeats × 5 folds = 15 modèles) pour réduire la variance.
3. AUGMENTATIONS
SpecAugment est minimal : FREQ_MASK_PARAM=10, TIME_MASK_PARAM=10 avec 1 freq mask et 2 time masks. Passer à FREQ_MASK_PARAM=30, TIME_MASK_PARAM=50, NUM_FREQ_MASKS=2, NUM_TIME_MASKS=3.
Pas de Mixup sur spectrogrammes : le mixup est fait au niveau waveform. Faire aussi du Mel-Mixup (mélange directement sur les spectrogrammes).
Pas de background noise injection : ajouter des bruits de fond réalistes (pluie, vent, routes) provenant de datasets comme ESC-50 ou FSD50K.
Pas de random EQ : appliquer des filtres passe-bande aléatoires pour simuler différentes qualités de microphone.
Pas de polarity inversion : augmentation gratuite (flip le signe du waveform).
Pas de random crop/padding asymétrique : varier les positions de début au-delà du random start actuel.
Freq MixStyle désactivé : FREQ_MIXSTYLE_PROB=0.0. Activer à 0.3-0.5 pour de la domain generalization.
4. DONNÉES & LABELS
Secondary labels sous-utilisés : ils sont ajoutés en binaire (1.0) au même niveau que le primary. Pondérer à 0.5-0.7 pour refléter l'incertitude.
S22 exclu de l'évaluation mais inclus dans le training : les labels bruités de S22 peuvent dégrader le modèle. Tester l'exclusion complète de S22 du training ou une pondération réduite (0.3x).
Upsampling naïf (copie) : pour les espèces rares, au lieu de copier, utiliser de l'oversampling augmenté (chaque copie est augmentée différemment).
MIN_SAMPLE=20 : très bas. Monter à 50-100 avec des augmentations variées.
Pas de pseudo-labeling : utiliser les prédictions du modèle actuel sur les données non labellisées ou faiblement labellisées pour bootstrapper.
Pas d'external data : Xeno-Canto contient des enregistrements pour la majorité des 234 espèces. Les focal recordings du train sont potentiellement insuffisantes pour certaines espèces.
Sonotypes (47158son)* : ces espèces non identifiées n'ont que du mirroring basique. Construire des prototypes acoustiques spécifiques via clustering des embeddings.
5. ENSEMBLE & POST-PROCESSING
Poids d'ensemble
Model_2 à 0.03 est quasi-ignoré (LB 0.917). Il tire probablement l'ensemble vers le bas. Le retirer entièrement ou l'améliorer significativement avant de le garder.
Poids non optimisés : les poids [0.03, 0.10, 0.87] semblent choisis manuellement. Faire un Nelder-Mead ou grid search sur les poids en utilisant un OOF score fiable.
Type d'ensemble "direct" : le weighted average simple est sous-optimal. Tester :
Rank averaging (déjà implémenté mais non utilisé ! solutions['type_add']='direct' au lieu de 'rank.1')
Stacking avec un meta-learner (LogisticRegression ou LightGBM)
Power mean : (w1*p1^β + w2*p2^β + w3*p3^β)^(1/β) avec β optimisé
Geometric mean (convient mieux aux probabilités)

Blend interne de Model_4
Le blend ProtoSSM 59.5% + SED 40.5% utilise du rank blending avec des gates sophistiqués. Mais le 3-way blend avec BirdNET (50/30/20) n'est activé que si BirdNET est disponible. S'assurer que BirdNET est bien attaché.
Les 5 gates (noise suppression, temporal continuity, SED spike, BirdNET spike, sonotype mirroring) ont des seuils hardcodés. Ces seuils devraient être optimisés par CV.
Temporal smoothing
Alpha_texture=0.25 et alpha_event=0.08 : les espèces à texture (insectes, amphibiens) sont plus lissées, ce qui est correct. Mais optimiser ces alphas par espèce serait mieux (certains insectes ont des appels pulsés, pas continus).
Le smoothing gaussien dans Model_2 (sigma=0.65) et le delta-shift dans Model_3/4 sont appliqués en cascade : les prédictions sont lissées 3-4 fois au total. Risque d'over-smoothing qui noie les détections brèves.
Platt calibration
C=100.0 : très permissif, quasi-pas de régularisation. La calibration peut overfitter.
MIN_POS=50 : seulement les espèces fréquentes sont calibrées. Les rares (les plus critiques) ne le sont pas.
Calibration sur 100% OOF sans split : risque de leakage si les folds OOF ne sont pas parfaitement alignés avec les données de calibration.
Remplacer par Temperature Scaling (un seul paramètre global) qui est plus robuste, ou Beta Calibration.
Taxonomic smoothing
Alpha_genus=0.15 et alpha_class=0.05 : ces valeurs mélangent les prédictions d'espèces différentes. C'est dangereux si des espèces proches ont des patterns acoustiques très différents. Conditionner le smoothing à la similarité des embeddings Perch entre espèces du même genre.
6. INFÉRENCE
Pas de TTA audio sur Model_2 : ajouter du gain TTA (±3dB) et du time-shift TTA (±0.5s).
SED blend clip+frame_max 50/50 : optimiser ce ratio. Frame_max capture mieux les vocalisations courtes, clip capture le contexte global.
Gaussian smoothing post-SED (sigma=0.65) appliqué dans les logits ET en probas selon les modèles — uniformiser.
Pas d'adaptive thresholding per-file : si un fichier a une confidence globale faible (environnement bruyant), baisser le seuil de détection.
7. ASPECTS TECHNIQUES & ENGINEERING
Cache LRU trop petit : _FC (focal cache) = 2000 entries, _SC_CACHE = 200 entries. Augmenter si la RAM le permet pour éviter les re-chargements.
Num_workers=0 dans les DataLoaders : mettre au moins 4 pour le prefetching parallèle.
ONNX opset 17 : vérifier que l'opset supporte bien toutes les ops (GeM, Conv1d remap).
Mémoire : pas de torch.cuda.empty_cache() entre Model_2 et Model_3. Risque d'OOM.
Reproducibilité : seeds différentes entre modèles (42, 1891, 4) — c'est bien pour la diversité de l'ensemble.
8. STRATÉGIE GLOBALE
Diversité de l'ensemble
Les 3 modèles partagent Perch v2 comme backbone/teacher. Ajouter un modèle basé sur un backbone différent :
BEATs (Microsoft, pre-trained audio)
Audio Spectrogram Transformer (AST)
HTS-AT
CLAP (contrastive language-audio pretraining)

Pas de modèle CNN pur (sans SSM ni attention) : un simple ResNet-50 sur mel avec SED peut être un bon diversificateur.
Espèces non mappées
41 espèces sans logit Perch direct. Le genus proxy (max des espèces du même genre) est un pis-aller. Entraîner un classifieur dédié aux espèces non-mappées sur les embeddings Perch + focal recordings.
Score cible
0.948 → 0.96+ requiert probablement :
Un 4ème modèle diversifié (backbone non-Perch)
Un ensemble rank-based correctement optimisé
Des augmentations audio plus agressives
Du pseudo-labeling itératif