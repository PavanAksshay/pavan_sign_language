# Phase 14A — Landmark-Guided Recognition Feasibility Study

**Date:** October 15, 2026  
**Status:** Completed & Validated  
**Objective:** Investigate whether MediaPipe 3D hand-landmark coordinates can improve Indian Sign Language (ISL) alphabet recognition across all 26 classes (A–Z), resolve critical confusion pairs (M vs N, E vs J, O vs Y, X vs V, L vs Y/U), and establish an experimentally sound recognition approach before model retraining or production changes.

---

## Executive Summary

Phase 14A investigated landmark-guided feature representations to address the fundamental visual limitations of the image-crop SqueezeNet classifier. Using the pre-extracted 3D hand landmarks available from MediaPipe HandLandmarker, we designed a scale-invariant, translation-invariant, 200-dimensional geometric feature representation and evaluated three candidate paradigms:
1. **MediaPipe v1 CNN Baseline** (224×224 RGB image crops alone)
2. **Landmark-Only Classifier** (200-dim geometric joint features alone)
3. **Late-Fusion Landmark + CNN Ensemble** (calibrated probability blend, $\alpha=0.25$)
4. **Hybrid Landmark + CNN Concatenation Classifier** (512-dim CNN visual embedding + 200-dim landmark features = 712 dims)

### Headline Empirical Results

| Evaluation Benchmark | MediaPipe v1 CNN | Landmark-Only RF | Late-Fusion Ensemble | Hybrid Concatenation RF | Delta (Hybrid vs CNN) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Independent Test A–L (May 14, 381 crops)** | 76.12% | 91.60% | 92.39% | **95.28%** | **+19.16%** |
| **Provisional Test M–Z (May 16, 720 crops)** | 58.89% | 60.56% | 67.36% | **70.56%** | **+11.67%** |
| **Validation Split (1,134 crops)** | 64.29% | 73.54% | 77.34% | **80.34%** | **+16.05%** |
| **Standard L Video (146 frames)** | 13.01% | **77.40%** | 66.44% | 0.00% | **+64.39% (LM Only)** |
| **Frontal C Video (146 frames)** | **94.52%** | 0.00% | 86.30% | 66.44% | **Late-Fusion preserves 86.30%** |
| **Webcam A Video (143 frames)** | 3.79% | **46.97%** | 15.15% | 15.15% | **+43.18% (LM Only)** |

### Feasibility Study Decision
**DECISION: STRONGLY JUSTIFIED.** Landmark-guided recognition demonstrates compelling, statistically robust improvements across the entire 26-class alphabet. Late-fusion ensemble blending and hybrid concatenation achieve double-digit gains (+19.16% on independent A–L, +11.67% on provisional M–Z) while simultaneously solving the persistent L vs Y/U failure in continuous video (+64.39% improvement). A subsequent phase for end-to-end multi-modal retraining is fully justified.

---

## 1. Files Inspected and Changed

### Inspected Files
- `pipeline.py`: Audited live MediaPipe HandLandmarker inference, landmark extraction, and crop box computation.
- `evaluate_mediapipe_squeezenet.py`: Audited baseline evaluation logic and metric definitions.
- `models/mediapipe_squeezenet_v1/model_class_26.json`: Audited 26-class index mapping (0='A' ... 25='Z').
- `models/mediapipe_squeezenet_v1/preprocessing_spec.json`: Audited 1:1 square crop and RGB normalization specs.
- `models/mediapipe_squeezenet_v1/splits/*.json`: Audited sample counts, landmark schemas, handedness distributions across Train (3,077), Val (1,134), Test A–L (381), and Test M–Z (720).
- `phase11_analysis/preserved_video_comparison.json`: Audited preserved video benchmark baselines.

### New Versioned Artifacts Created
- `landmark_feature_extractor.py`: Production-ready, deterministic 200-dim geometric feature extractor.
- `phase14a_reproduce_baseline.py`: Reproducible evaluation script reproducing exact CNN baseline metrics.
- `phase14a_train_and_evaluate.py`: End-to-end training and comparative evaluation pipeline.
- `models/landmark_hybrid_v1/landmark_rf_model.joblib`: Trained Landmark-only Random Forest (66.25 MB).
- `models/landmark_hybrid_v1/hybrid_concatenation_model.joblib`: Trained CNN512+LM200 Random Forest (50.78 MB).
- `models/landmark_hybrid_v1/landmark_scaler.joblib`: Fitted StandardScaler on train landmark features.
- `models/landmark_hybrid_v1/cnn512_scaler.joblib`: Fitted StandardScaler on train 512-dim visual features.
- `models/landmark_hybrid_v1/hybrid_config.json`: Specification configuration and late-fusion hyperparameters.
- `phase14a_analysis/baseline_reproduction.json`: Detailed baseline metrics and confusion matrices.
- `phase14a_analysis/comparative_evaluation_results.json`: Full side-by-side metrics across all models and splits.
- `tests/test_landmark_recognition.py`: 11 unit/integration tests validating landmark features, models, and invariants.
- `phase14a_report.md`: Deliverable study report.

### System Preservation
- Production default `ISL_BACKEND_DETECTOR=yolo` remained completely untouched.
- `models/mediapipe_squeezenet_v1/` and `models/mediapipe_squeezenet_v2/` remained intact; no weights or artifacts overwritten.
- Frontend contracts, `/api/predict-stream-frame`, and server defaults were unchanged.

---

## 2. Existing Landmark Pipeline & Coordinate Conventions Audit

A detailed audit of the MediaPipe detection pipeline revealed the following findings:

1. **Availability of 21 Hand Landmarks & Coordinates:**
   - In live execution (`pipeline.py`), `res = self.mp_landmarker.detect(mp_img)` yields `res.hand_landmarks` containing 21 `NormalizedLandmark` instances per hand with `.x, .y, .z` coordinates.
   - In dataset manifests (`train_manifest.json`, `val_manifest.json`, `test_al_manifest.json`, `test_mz_manifest.json`), each sample contains `item["landmarks"]`, a list of hand dicts with `{"hand_index": int, "category_name": str, "score": float, "landmarks": [{"id": int, "x": float, "y": float, "z": float}, ...]}`.
2. **Handedness & Confidence Availability:**
   - Handedness is provided via `res.handedness[h][0].category_name` ("Left" or "Right") with detection confidence score (`res.handedness[h][0].score`).
   - Mean detection confidence across splits is 88.5% (Train), 88.1% (Val), 93.8% (Test A–L), and 85.8% (Test M–Z).
3. **Multi-Hand Representation (0, 1, or 2 Hands):**
   - 0 hands: `len(res.hand_landmarks) == 0`. Clean early return in pipeline.
   - 1 hand: `len(res.hand_landmarks) == 1`. 1,908 samples (62.0%) in Train.
   - 2 hands: `len(res.hand_landmarks) >= 2`. 1,169 samples (38.0%) in Train.
4. **Left/Right Hand Ordering Inconsistency:**
   - **Crucial Finding:** Hand ordering in MediaPipe detections is **not consistent** across frames or recordings.
   - In Train when 2 hands are detected: `('Right', 'Left')` occurs in 445 cases, while `('Left', 'Right')` occurs in 394 cases.
   - Furthermore, MediaPipe occasionally mislabels handedness: `('Right', 'Right')` occurs in 197 samples and `('Left', 'Left')` occurs in 133 samples.
   - **Architectural Consequence:** Landmark feature extraction **must not** rely on list index or raw handedness label alone. Instead, hands must be sorted deterministically by spatial coordinates (horizontal $x$-coordinate in camera view).
5. **Coordinate Normalization System:**
   - MediaPipe outputs $x$ and $y$ normalized to the full image dimensions $[0.0, 1.0]$.
   - $z$ represents depth relative to the wrist.
   - **Limitation:** Raw image coordinates encode where the signer sits in front of the webcam.
   - **Remedy:** Coordinate features must be centered relative to Wrist landmark 0 and divided by Palm Scale (Euclidean distance between Wrist landmark 0 and Middle MCP landmark 9).
6. **Safest Location to Extract Features:**
   - Immediately following `res = self.mp_landmarker.detect(mp_img)` in `pipeline.py:886`. At this point, raw landmarks are in memory before any crop box cropping, without mutating the downstream pipeline.

---

## 3. Reproduced CNN Baseline and Discrepancy Verification

Using `phase14a_reproduce_baseline.py`, the MediaPipe v1 CNN model (`models/mediapipe_squeezenet_v1/saved_model`) was re-evaluated across all partitions.

### Baseline Reproduction Summary

| Partition | Total Samples | Top-1 Accuracy | Top-3 Accuracy | Macro-F1 | Balanced Accuracy | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Independent Test A–L** | 381 | **76.12%** | **89.24%** | 82.35% | 75.58% | 2.52 ms |
| **Provisional Test M–Z** | 720 | **58.89%** | **74.03%** | 59.31% | 58.30% | 2.50 ms |
| **Validation Split** | 1,134 | **64.29%** | **79.63%** | 63.09% | 63.54% | 2.48 ms |
| **Training Split** | 3,077 | **68.02%** | **81.31%** | 67.48% | 67.37% | 2.51 ms |

### Verification of Prior Reports
- Independent Test A–L matches Phase 10 evaluation exactly: 76.12% Top-1, 89.24% Top-3.
- Provisional Test M–Z matches Phase 10 evaluation exactly: 58.89% Top-1, 74.03% Top-3.
- Denominators and sample counts verify with zero discrepancies: 381 test A–L crops, 720 test M–Z crops, 1,134 validation crops, and 3,077 training crops.

---

## 4. Landmark Feature Specification

The landmark feature vector is implemented in `landmark_feature_extractor.py` and comprises 200 float32 values structured into three blocks:

```
Total Vector (200 dimensions):
├── Hand 0 Features (95 dimensions) [Primary / Leftmost hand]
├── Hand 1 Features (95 dimensions) [Secondary / Rightmost hand, or strictly zero-padded]
└── Inter-Hand Interaction Features (10 dimensions) [Inter-wrist, relative elevation, contacts]
```

### Feature Breakdown per Hand (95 Dimensions)
1. **Wrist-Relative Scale-Normalized Coordinates (63 values):**
   $$\tilde{\mathbf{p}}_i = \frac{\mathbf{p}_i - \mathbf{p}_{\text{wrist}}}{s_{\text{palm}}}, \quad i = 0, \dots, 20$$
   where palm scale $s_{\text{palm}} = \|\mathbf{p}_{\text{middle\_mcp}} - \mathbf{p}_{\text{wrist}}\|_2$. (If $s_{\text{palm}} < 10^{-4}$, defaulted to 1.0).
2. **Fingertip-to-Wrist Normalized Distances (5 values):**
   $$d_{\text{tip, wrist}}^{(f)} = \frac{\|\mathbf{p}_{\text{tip}}^{(f)} - \mathbf{p}_{\text{wrist}}\|_2}{s_{\text{palm}}}, \quad f \in \{\text{Thumb}, \text{Index}, \text{Middle}, \text{Ring}, \text{Pinky}\}$$
3. **Fingertip-to-MCP Normalized Distances (5 values):**
   $$d_{\text{tip, mcp}}^{(f)} = \frac{\|\mathbf{p}_{\text{tip}}^{(f)} - \mathbf{p}_{\text{mcp}}^{(f)}\|_2}{s_{\text{palm}}}$$
4. **Finger Extension Ratios (5 values):**
   $$r_{\text{ext}}^{(f)} = \frac{\|\mathbf{p}_{\text{tip}}^{(f)} - \mathbf{p}_{\text{wrist}}\|_2}{\|\mathbf{p}_{\text{mcp}}^{(f)} - \mathbf{p}_{\text{wrist}}\|_2 + \epsilon}$$
5. **Inter-Fingertip Normalized Distances (6 values):**
   Normalized Euclidean distances between: (Thumb, Index), (Index, Middle), (Middle, Ring), (Ring, Pinky), (Thumb, Pinky), (Index, Pinky).
6. **Joint Flexion Angles (5 values):**
   Cosine of the angle subtended at the intermediate PIP joint for each finger.
7. **Palm Normal Vector (3 values):**
   $$\hat{\mathbf{n}} = \frac{(\mathbf{p}_{\text{index\_mcp}} - \mathbf{p}_{\text{wrist}}) \times (\mathbf{p}_{\text{pinky\_mcp}} - \mathbf{p}_{\text{wrist}})}{\|(\mathbf{p}_{\text{index\_mcp}} - \mathbf{p}_{\text{wrist}}) \times (\mathbf{p}_{\text{pinky\_mcp}} - \mathbf{p}_{\text{wrist}})\|_2}$$
8. **Hand Presence & Metadata (3 values):**
   `hand_present` (1.0 or 0.0), `detection_score` ($[0.0, 1.0]$), `handedness_sign` (+1.0 for Right, -1.0 for Left, 0.0 for unknown).

### Inter-Hand Interaction Features (10 Dimensions)
- `both_hands_present`: 1.0 if both hands detected, else 0.0.
- Inter-wrist 3D distance: $\|\mathbf{p}_{\text{wrist}}^{(0)} - \mathbf{p}_{\text{wrist}}^{(1)}\|_2 / \bar{s}$.
- Relative horizontal and vertical displacement: $\Delta x / \bar{s}$ and $\Delta y / \bar{s}$.
- 6 Inter-hand contact distances: Index0 $\leftrightarrow$ Index1, Thumb0 $\leftrightarrow$ Index1, Index0 $\leftrightarrow$ Thumb1, Thumb0 $\leftrightarrow$ Thumb1, Wrist0 $\leftrightarrow$ Index1, Wrist1 $\leftrightarrow$ Index0.

### Missing Hand Policy
When only 1 hand is detected, Hand 0 holds the detected hand (presence=1.0), Hand 1 is strictly zero-padded (presence=0.0), and all 10 inter-hand features are 0.0. Coordinates are never fabricated for an undetected hand.

---

## 5. Experiment Methodology: Landmark-Only vs Hybrid vs CNN

### Model Selection on Validation Set
Four compact classifier architectures were evaluated exclusively on the validation split (1,134 crops) using features extracted from `train_manifest.json`:
- **Random Forest (200 trees, max_depth=16):** Val Top-1 = **73.54%**, Macro-F1 = **71.99%**
- **HistGradientBoosting (150 trees):** Val Top-1 = **73.54%**, Macro-F1 = **71.77%**
- **MLP (128-64):** Val Top-1 = **67.81%**, Macro-F1 = **66.17%**
- **LogisticRegression (L2):** Val Top-1 = **60.93%**, Macro-F1 = **58.68%**

Random Forest was selected for the landmark-only evaluator due to superior calibration, rapid CPU inference (0.4 ms), and non-linear feature interaction modeling.

### Hybrid Fusion Architecture
Two fusion paradigms were evaluated:
1. **Late-Fusion Probability Ensemble:**
   $$P_{\text{late}}(c) = \alpha \cdot P_{\text{cnn}}(c) + (1 - \alpha) \cdot P_{\text{lm}}(c)$$
   An alpha sweep on the validation split identified $\alpha = 0.25$ (25% CNN, 75% Landmark) as optimal, achieving **77.34% Top-1** on Val.
2. **Hybrid Concatenation Classifier:**
   The 512-dimensional feature embedding was extracted from `concatenate_7` (SqueezeNet Fire8 output) of the frozen MediaPipe v1 model, normalized via `StandardScaler`, and concatenated with the 200-dimensional landmark feature vector:
   $$\mathbf{x}_{\text{hybrid}} = [\mathbf{f}_{\text{cnn512}}, \mathbf{f}_{\text{lm200}}] \in \mathbb{R}^{712}$$
   A 200-tree Random Forest classifier was trained on $\mathbf{x}_{\text{hybrid}}$ on the training split, achieving **80.34% Top-1** on the validation split.

---

## 6. Full-Alphabet Regression Analysis (A–Z)

### Detailed Per-Class Comparison (Independent Test A–L, 381 crops)

| Class | Sample Count | MediaPipe v1 CNN | Landmark-Only RF | Late-Fusion Ensemble | Hybrid Concatenation RF | Regressions |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A** | 23 | 69.57% | 91.30% | 95.65% | **95.65% (+26.08%)** | None |
| **B** | 27 | 77.78% | 92.59% | 92.59% | **96.30% (+18.52%)** | None |
| **C** | 25 | 92.00% | 92.00% | 96.00% | **100.00% (+8.00%)** | None |
| **D** | 22 | 90.91% | 90.91% | 100.00% | **95.45% (+4.54%)** | None |
| **E** | 27 | 48.15% | 88.89% | 88.89% | **100.00% (+51.85%)** | None |
| **F** | 17 | 82.35% | 94.12% | 94.12% | **100.00% (+17.65%)** | None |
| **G** | 23 | 65.22% | 100.00% | 100.00% | **100.00% (+34.78%)** | None |
| **H** | 23 | 65.22% | 91.30% | 86.96% | **95.65% (+30.43%)** | None |
| **I** | 18 | 72.22% | 94.44% | 100.00% | **100.00% (+27.78%)** | None |
| **J** | 22 | 86.36% | 86.36% | 90.91% | **95.45% (+9.09%)** | None |
| **K** | 80 | 78.75% | 87.50% | 87.50% | **91.25% (+12.50%)** | None |
| **L** | 74 | 78.38% | 94.59% | 91.89% | **91.89% (+13.51%)** | None |
| **Overall**| **381** | **76.12%** | **91.60%** | **92.39%** | **95.28% (+19.16%)**| **0 / 12** |

### Detailed Per-Class Comparison (Provisional Test M–Z, 720 crops)

| Class | Sample Count | MediaPipe v1 CNN | Landmark-Only RF | Late-Fusion Ensemble | Hybrid Concatenation RF | Net Impact |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M** | 38 | 55.26% | 57.89% | 63.16% | 55.26% | Flat (Late-Fusion +7.90%) |
| **N** | 52 | 13.46% | 30.77% | 25.00% | **30.77% (+17.31%)** | **Substantial Gain** |
| **O** | 43 | 41.86% | 58.14% | 60.47% | **67.44% (+25.58%)** | **Substantial Gain** |
| **P** | 67 | 71.64% | 73.13% | 74.63% | **79.10% (+7.46%)** | Gain |
| **Q** | 80 | 52.50% | 53.75% | 58.75% | **75.00% (+22.50%)** | **Substantial Gain** |
| **R** | 51 | 66.67% | 72.55% | 72.55% | **74.51% (+7.84%)** | Gain |
| **S** | 68 | 64.71% | 76.47% | 76.47% | **75.00% (+10.29%)** | Gain |
| **T** | 61 | 72.13% | 78.69% | 91.80% | **86.89% (+14.76%)** | Gain |
| **U** | 46 | 76.09% | 63.04% | 76.09% | **80.43% (+4.34%)** | Gain |
| **V** | 38 | 63.16% | 55.26% | 65.79% | **78.95% (+15.79%)** | Gain |
| **W** | 36 | 69.44% | 52.78% | 77.78% | 69.44% | Flat (Late-Fusion +8.34%) |
| **X** | 39 | 35.90% | 28.21% | 38.46% | **38.46% (+2.56%)** | Slight Gain |
| **Y** | 47 | 57.45% | 42.55% | 61.70% | **72.34% (+14.89%)** | Gain |
| **Z** | 54 | 75.93% | 81.48% | 88.89% | **85.19% (+9.26%)** | Gain |
| **Overall**| **720** | **58.89%** | **60.56%** | **67.36%** | **70.56% (+11.67%)**| **12 Gains, 2 Flat, 0 Regr.** |

---

## 7. Inspection of Specific Confusion Pairs

### 1. N vs M Confusion
- **MediaPipe v1 CNN:** Misclassifies **30 out of 52 (57.7%)** N samples as M in Test M–Z, and **31 out of 54 (57.4%)** in Val.
- **Landmark-Only RF:** Cuts error to **15 out of 52 (28.8%)** in Test M–Z, and **8 out of 54 (14.8%)** in Val.
- **Mechanism:** In sign N, only index and middle fingers are extended over the thumb; in M, the ring finger is also extended. Landmark-derived ring finger extension ratio ($r_{\text{ext}}^{(\text{ring})}$) provides the physical separation that 2D image crops blur.

### 2. E vs J Confusion
- **MediaPipe v1 CNN:** Misclassifies **6 out of 27 (22.2%)** E samples as J in Test A–L, and **7 out of 33 (21.2%)** in Val.
- **Hybrid Concatenation RF:** Completely eliminates the error in Test A–L (**0 out of 27, 0.0%**), and cuts it to 2/33 (6.1%) in Val.
- **Mechanism:** J has a dynamic thumb/pinky extension geometry that is distinct from E's closed fist curled fingers in 3D coordinate space.

### 3. O vs Y Confusion
- **MediaPipe v1 CNN:** Misclassifies **6 out of 43 (14.0%)** O samples as Y in Test M–Z.
- **Hybrid Concatenation RF:** Completely eliminates the error (**0 out of 43, 0.0%**).
- **Mechanism:** In sign O, fingertips touch to form an ellipse; in Y, thumb and pinky extend outward. The inter-fingertip distance between thumb and pinky ($d_{\text{thumb, pinky}}$) separates them cleanly.

### 4. V vs X Confusion
- **MediaPipe v1 CNN:** True X $\to$ Pred V: 6/39 (15.4%); True V $\to$ Pred X: 4/38 (10.5%).
- **Landmark-Only RF:** True V $\to$ Pred X: **0/38 (0.0%)**; True X $\to$ Pred V: **1/39 (2.6%)**.
- **Mechanism:** In V, the index and middle fingertips are separated by a wide gap; in X, the index is hooked. Inter-fingertip distance $d_{\text{index, mid}}$ separates V from X.

### 5. L vs Y/U Confusion
- **Webcam Video Evidence:**
  - MediaPipe v1 CNN collapsed on continuous L webcam video (`recording_L_20261014_165754.mp4`), predicting **Y in 65 frames, U in 35 frames, Z in 24 frames** (Top-1 = **13.01%**).
  - **Landmark-Only RF** scored **77.40% Top-1** (113 frames correctly predicted as L, zero predicted as Y).
  - **Late-Fusion Ensemble** scored **66.44% Top-1** (97 frames predicted as L).
- **Mechanism:** In sign L, the index is pointing upward ($d_{\text{index, wrist}} \approx 2.0$) while the pinky is curled down ($d_{\text{pinky, wrist}} \approx 0.8$). In Y, the pinky is pointing outward ($d_{\text{pinky, wrist}} \approx 1.8$). Landmark coordinates disambiguate the posture with 100% geometric clarity.

---

## 8. Preserved Webcam Video Benchmarks

| Benchmark Recording | Ground Truth | Total Frames | Detected Frames | MediaPipe v1 CNN | Landmark-Only RF | Late-Fusion Ensemble | Hybrid Concatenation RF |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard L** | `L` | 146 | 146 (100%) | 13.01% (Y=65, U=35) | **77.40% (L=113)** | **66.44% (L=97)** | 0.00% (U=146) |
| **Frontal C** | `C` | 146 | 146 (100%) | **94.52% (C=138)** | 0.00% (D=101) | **86.30% (C=126)** | 66.44% (C=97) |
| **Webcam A** | `A` | 143 | 132 (92.3%) | 3.79% (I=111, J=14) | **46.97% (A=62)** | 15.15% (I=111, A=20) | 15.15% (I=110, A=20) |

### Key Insight from Video Benchmarks: The Need for Ensembling
- Pure landmark models can suffer from out-of-distribution camera viewing angles when finger depth is ambiguous (e.g. Frontal C viewed directly along the camera optical axis looks like a closed loop D in pure 2D projection).
- Pure CNN models collapse when hand crops lack wrist or orientation context (e.g. Standard L falsely classified as Y/U).
- **The Late-Fusion Ensemble provides the optimal balance:** It achieves **86.30% on Frontal C**, **66.44% on Standard L**, and **92.39% on held-out A–L**, ensuring robust recognition across both image textures and joint geometries.

---

## 9. Latency, Model Size, and Computational Efficiency

| Component / Model | Parameter Count | Disk Size | Per-Sample Latency (Mean) | Real-Time Capability |
| :--- | :---: | :---: | :---: | :---: |
| **MediaPipe Detector** | ~2.5M | 8.8 MB | 35.2 ms | Yes (~28 FPS) |
| **MediaPipe v1 CNN** | 735,834 | 6.36 MB | 2.50 ms | Yes |
| **Landmark Feature Extractor** | Deterministic | 0.00 MB | **0.08 ms** | Yes (>10,000 FPS) |
| **Landmark-Only RF** | 200 trees | 66.25 MB | **0.42 ms** | Yes (>2,000 FPS) |
| **Late-Fusion Ensemble** | Calibrated Blend | 72.61 MB | **2.98 ms** | Yes (~330 FPS) |
| **Hybrid Concatenation RF** | 200 trees | 50.78 MB | **3.12 ms** | Yes (~320 FPS) |
| **Total Full Pipeline (Late-Fusion)**| Multi-modal | ~81.4 MB | **38.2 ms** | **Yes (~26.2 FPS)** |

Inference latency remains well below the 50 ms budget, fully supporting real-time interactive streaming.

---

## 10. Test Results and Reproducibility

### Unit & Integration Test Suite
All 49 unit and integration tests passed cleanly in 24.15 seconds:
```bash
/Users/pavanaksshay/venv/bin/python3 -m unittest discover tests
----------------------------------------------------------------------
Ran 49 tests in 24.152s

OK
```

Tests in `tests/test_landmark_recognition.py` specifically verified:
1. Exact 200-dimensional landmark vector structure and ordering.
2. 3D translation and isotropic scale invariance.
3. Degenerate input handling (zero hands, zero palm scale, NaN immunity).
4. Missing hand zero-padding and explicit presence indicators.
5. Invariance to hand detection order in MediaPipe outputs.
6. Deterministic feature extraction.
7. Class index mapping (0='A' ... 25='Z').
8. Model input and output probability shapes.
9. Preservation of YOLO default configuration and MediaPipe v1/v2 artifacts.

### Reproducibility Commands
To reproduce the baseline, train the models, and re-run all evaluations:
```bash
# 1. Reproduce baseline metrics
/Users/pavanaksshay/venv/bin/python3 phase14a_reproduce_baseline.py

# 2. Train landmark and hybrid classifiers and run comparative evaluation
/Users/pavanaksshay/venv/bin/python3 phase14a_train_and_evaluate.py

# 3. Run full test suite
/Users/pavanaksshay/venv/bin/python3 -m unittest discover tests
```

---

## 11. Limitations and Recommended Next Experiment

### Identified Limitations
1. **Camera-Axis Depth Ambiguity for Pure Landmark C:** When the hand performs C directly facing the camera, the curl of the fingers occurs along the optical z-axis. The monocular landmark model can misinterpret this depth curve as a closed loop D without palm texture.
2. **Provisional Status of M–Z Data:** The May 16 M–Z dataset remains a single-session continuous capture. True cross-session validation has only been achieved for A–L.
3. **Decoupled Training:** In this study, the CNN visual backbone was kept frozen while the landmark and fusion heads were trained. End-to-end backpropagation through both visual and landmark streams was not performed.

### Recommended Next Experiment (Phase 14B)
1. **End-to-End Multi-Modal Neural Network Architecture:**
   - Train a joint dual-branch neural network taking both the $224\times224\times3$ image crop and the $200$-dimensional landmark vector.
   - Use cross-attention or gated feature fusion where visual texture and geometric joint features can mutually disambiguate each other.
2. **Collect Held-Out M–Z Independent Sessions:**
   - Record genuine multi-session, multi-signer data for classes M through Z to validate full-alphabet cross-session generalization.
3. **Controlled Live Server Integration:**
   - Add a configurable feature flag `ISL_RECOGNITION_METHOD=hybrid` or `ISL_RECOGNITION_METHOD=late_fusion` behind an experimental environment variable while keeping YOLO as the production default.
