
# Let me analyze the current notebook structure and identify key improvement areas
# The notebook is at 0.947 LB, targeting >0.95

analysis = """
=== CURRENT NOTEBOOK ANALYSIS ===

Models in ensemble:
- Model_2 (SED): 0.917 LB - Distilled SED with EfficientNet-B0 + Perch distillation
- Model_3 (ProtoSSM): 0.928 LB - Perch + ProtoSSM + ResSSM + MLP probes
- Model_4 (tedBirdCLEF): 0.947 LB - Perch + LightProtoSSM + ResidualSSM + SED + BirdNET

Current blend: Model_3 (0.0 weight) + Model_4 (1.0 weight) = effectively just Model_4

=== KEY GAPS IDENTIFIED ===

1. Model_3 has weight 0.0 - completely unused! This is a 0.928 model being wasted.
2. No pseudo-labeling loop (BirdCLEF 2024 2nd place used 3 iterative cycles)
3. No checkpoint soup / model soup
4. No test-time augmentation (TTA) for SED model
5. No iterative pseudo-label refinement
6. The blend is rank-based but could be optimized with OOF calibration
7. No multi-resolution mel spectrogram ensemble
8. Missing context window aggregation (neighboring window smoothing)
9. No species-specific calibration/temperature scaling per taxon
10. BirdNET branch is optional and may not be attached

=== WINNING STRATEGIES FROM BIRDCLEF 2024/2025 ===

From BirdCLEF 2024 2nd place (yuto_mo article):
- Pseudo-labeling with iterative ensemble improvement (3 cycles)
- Neighborhood smoothing: window[t] += 0.5*(window[t-1] + window[t+1])
- Checkpoint soup instead of early stopping
- Different mel params ensemble
- GEM pooling

From BirdCLEF 2026 strategy playbook:
- Context/sequence models over adjacent segments
- Hybrid grouped+stratified CV
- Subgroup robustness tracking
- Pseudo labels from test domain

=== IMPROVEMENT PRIORITIES ===

1. FIX: Give Model_3 non-zero weight (it's 0.928!)
2. ADD: Pseudo-labeling loop on unlabeled test data
3. ADD: Neighborhood temporal smoothing (0.5 factor)
4. ADD: Checkpoint soup for ProtoSSM training
5. ADD: Multi-resolution mel ensemble in SED
6. ADD: Better calibration with per-class thresholds
7. ADD: Test-time augmentation for all models
8. ADD: Model soup / SWA for SED
9. OPTIMIZE: Blend weights via OOF optimization
10. ADD: Context window features (±2 windows)
"""
print(analysis)

=== CURRENT NOTEBOOK ANALYSIS ===

Models in ensemble:
- Model_2 (SED): 0.917 LB - Distilled SED with EfficientNet-B0 + Perch distillation
- Model_3 (ProtoSSM): 0.928 LB - Perch + ProtoSSM + ResSSM + MLP probes
- Model_4 (tedBirdCLEF): 0.947 LB - Perch + LightProtoSSM + ResidualSSM + SED + BirdNET

Current blend: Model_3 (0.0 weight) + Model_4 (1.0 weight) = effectively just Model_4

=== KEY GAPS IDENTIFIED ===

1. Model_3 has weight 0.0 - completely unused! This is a 0.928 model being wasted.
2. No pseudo-labeling loop (BirdCLEF 2024 2nd place used 3 iterative cycles)
3. No checkpoint soup / model soup
4. No test-time augmentation (TTA) for SED model
5. No iterative pseudo-label refinement
6. The blend is rank-based but could be optimized with OOF calibration
7. No multi-resolution mel spectrogram ensemble
8. Missing context window aggregation (neighboring window smoothing)
9. No species-specific calibration/temperature scaling per taxon
10. BirdNET branch is optional and may not be attached

=== WINNING STRATEGIES FROM BIRDCLEF 2024/2025 ===

From BirdCLEF 2024 2nd place (yuto_mo article):
- Pseudo-labeling with iterative ensemble improvement (3 cycles)
- Neighborhood smoothing: window[t] += 0.5*(window[t-1] + window[t+1])
- Checkpoint soup instead of early stopping
- Different mel params ensemble
- GEM pooling

From BirdCLEF 2026 strategy playbook:
- Context/sequence models over adjacent segments
- Hybrid grouped+stratified CV
- Subgroup robustness tracking
- Pseudo labels from test domain

=== IMPROVEMENT PRIORITIES ===

1. FIX: Give Model_3 non-zero weight (it's 0.928!)
2. ADD: Pseudo-labeling loop on unlabeled test data
3. ADD: Neighborhood temporal smoothing (0.5 factor)
4. ADD: Checkpoint soup for ProtoSSM training
5. ADD: Multi-resolution mel ensemble in SED
6. ADD: Better calibration with per-class thresholds
7. ADD: Test-time augmentation for all models
8. ADD: Model soup / SWA for SED
9. OPTIMIZE: Blend weights via OOF optimization
10. ADD: Context window features (±2 windows)





# Now let me create the comprehensive improvement plan and generate the enhanced notebook
# I'll write out a detailed analysis document first

improvement_plan = """
================================================================================
BIRDCLEF+ 2026 NOTEBOOK IMPROVEMENT PLAN: 0.947 → >0.95
================================================================================

=== CRITICAL FIXES (Immediate +0.5-1.0 points) ===

1. ACTIVATE MODEL_3 IN BLEND
   Current: Model_3 weight = 0.0 (completely unused!)
   Fix: Set weight to 0.15-0.20. Model_3 is 0.928 LB - it's complementary diversity.
   
2. PSEUDO-LABELING ON UNLABELED SOUNDSCAPES
   BirdCLEF 2024 2nd place: +4-6% AUC from 3 iterative pseudo-label cycles
   BirdCLEF 2025 1st place: "Multi-Iterative Noisy Student Is All You Need"
   Implementation:
   - Use current ensemble (Model_4) to predict on unlabeled train_soundscapes
   - Filter: keep only predictions > 0.3 confidence
   - Mix pseudo-labeled soundscapes with focal training data at 30-40% ratio
   - Retrain SED (Model_2) with pseudo labels
   - Generate new pseudo labels with updated ensemble
   - Repeat 2-3 cycles

3. NEIGHBORHOOD TEMPORAL SMOOTHING (BirdCLEF 2024 2nd place technique)
   Post-process: scores[t] += 0.5 * (scores[t-1] + scores[t+1]) for each file
   This is the single highest-impact post-processing trick in the competition.

=== HIGH-IMPACT IMPROVEMENTS (+0.3-0.5 points) ===

4. CHECKPOINT SOUP FOR PROTOSSM
   Instead of single best checkpoint, average last N checkpoints (soup)
   BirdCLEF 2024 2nd place: "Creating checkpoint soups instead of using early stopping"
   Implementation: Save checkpoints every 5 epochs, average weights of last 3-5

5. MULTI-RESOLUTION MEL ENSEMBLE IN SED
   Current: single mel config (256 mels, hop=512)
   Add: second SED with different mel params (128 mels, hop=256 or 384 mels, hop=640)
   Ensemble the two SED outputs
   BirdCLEF 2024 2nd place: "ensemble with different mel params"

6. TEST-TIME AUGMENTATION (TTA) FOR SED
   Current: no TTA for SED inference
   Add: horizontal flip + time shift ±1s + gain jitter during inference
   Average predictions across augmentations

7. IMPROVED BLEND OPTIMIZATION
   Current: fixed weights [0.0, 1.0] for [Model_3, Model_4]
   Better: Use OOF predictions to optimize weights via grid search or logistic regression
   Target: Model_2 (0.917) + Model_3 (0.928) + Model_4 (0.947) blended optimally
   
=== MEDIUM-IMPACT IMPROVEMENTS (+0.2-0.3 points) ===

8. PER-CLASS TEMPERATURE SCALING
   Current: single temperature per taxon
   Better: Learn per-class temperature from OOF calibration
   Use isotonic regression per class for better probability calibration

9. CONTEXT WINDOW FEATURES
   For each 5s window, add features from ±1 and ±2 neighboring windows
   This captures temporal continuity of bird calls

10. ENHANCED SONOTYPE MIRRORING
    Current: only 4 mirror groups
    Add: all visually similar sonotype pairs from taxonomy

11. SPECIES-SPECIFIC POST-PROCESSING
    Different smoothing alphas for different taxa
    Amphibia/Insecta (texture): stronger temporal smoothing
    Aves (event): lighter smoothing, preserve transients

=== IMPLEMENTATION PRIORITY ===

Phase 1 (Must do): 1, 2, 3, 7  → Expected gain: +0.8-1.2 points
Phase 2 (Should do): 4, 5, 6    → Expected gain: +0.3-0.5 points  
Phase 3 (Nice to have): 8, 9, 10, 11 → Expected gain: +0.2-0.4 points

Total expected: 0.947 + 1.3-2.1 = 0.960-0.968
"""
print(improvement_plan)



================================================================================
BIRDCLEF+ 2026 NOTEBOOK IMPROVEMENT PLAN: 0.947 → >0.95
================================================================================

=== CRITICAL FIXES (Immediate +0.5-1.0 points) ===

1. ACTIVATE MODEL_3 IN BLEND
   Current: Model_3 weight = 0.0 (completely unused!)
   Fix: Set weight to 0.15-0.20. Model_3 is 0.928 LB - it's complementary diversity.

2. PSEUDO-LABELING ON UNLABELED SOUNDSCAPES
   BirdCLEF 2024 2nd place: +4-6% AUC from 3 iterative pseudo-label cycles
   BirdCLEF 2025 1st place: "Multi-Iterative Noisy Student Is All You Need"
   Implementation:
   - Use current ensemble (Model_4) to predict on unlabeled train_soundscapes
   - Filter: keep only predictions > 0.3 confidence
   - Mix pseudo-labeled soundscapes with focal training data at 30-40% ratio
   - Retrain SED (Model_2) with pseudo labels
   - Generate new pseudo labels with updated ensemble
   - Repeat 2-3 cycles

3. NEIGHBORHOOD TEMPORAL SMOOTHING (BirdCLEF 2024 2nd place technique)
   Post-process: scores[t] += 0.5 * (scores[t-1] + scores[t+1]) for each file
   This is the single highest-impact post-processing trick in the competition.

=== HIGH-IMPACT IMPROVEMENTS (+0.3-0.5 points) ===

4. CHECKPOINT SOUP FOR PROTOSSM
   Instead of single best checkpoint, average last N checkpoints (soup)
   BirdCLEF 2024 2nd place: "Creating checkpoint soups instead of using early stopping"
   Implementation: Save checkpoints every 5 epochs, average weights of last 3-5

5. MULTI-RESOLUTION MEL ENSEMBLE IN SED
   Current: single mel config (256 mels, hop=512)
   Add: second SED with different mel params (128 mels, hop=256 or 384 mels, hop=640)
   Ensemble the two SED outputs
   BirdCLEF 2024 2nd place: "ensemble with different mel params"

6. TEST-TIME AUGMENTATION (TTA) FOR SED
   Current: no TTA for SED inference
   Add: horizontal flip + time shift ±1s + gain jitter during inference
   Average predictions across augmentations

7. IMPROVED BLEND OPTIMIZATION
   Current: fixed weights [0.0, 1.0] for [Model_3, Model_4]
   Better: Use OOF predictions to optimize weights via grid search or logistic regression
   Target: Model_2 (0.917) + Model_3 (0.928) + Model_4 (0.947) blended optimally

=== MEDIUM-IMPACT IMPROVEMENTS (+0.2-0.3 points) ===

8. PER-CLASS TEMPERATURE SCALING
   Current: single temperature per taxon
   Better: Learn per-class temperature from OOF calibration
   Use isotonic regression per class for better probability calibration

9. CONTEXT WINDOW FEATURES
   For each 5s window, add features from ±1 and ±2 neighboring windows
   This captures temporal continuity of bird calls

10. ENHANCED SONOTYPE MIRRORING
    Current: only 4 mirror groups
    Add: all visually similar sonotype pairs from taxonomy

11. SPECIES-SPECIFIC POST-PROCESSING
    Different smoothing alphas for different taxa
    Amphibia/Insecta (texture): stronger temporal smoothing
    Aves (event): lighter smoothing, preserve transients

=== IMPLEMENTATION PRIORITY ===

Phase 1 (Must do): 1, 2, 3, 7  → Expected gain: +0.8-1.2 points
Phase 2 (Should do): 4, 5, 6    → Expected gain: +0.3-0.5 points  
Phase 3 (Nice to have): 8, 9, 10, 11 → Expected gain: +0.2-0.4 points

Total expected: 0.947 + 1.3-2.1 = 0.960-0.968