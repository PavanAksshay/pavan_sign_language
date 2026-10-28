# Phase 20 — Error Forensics and Multi-Signer Training Strategy Report

**Timestamp**: 2026-10-22T22:38:00Z  
**Phase**: Phase 20 (Error Forensics, Metric Reconciliation & Multi-Signer Training Design)  
**Status**: COMPLETE — Zero Contamination Preserved — No Training Conducted  
**Runtime Configuration**: `ISL_BACKEND_DETECTOR=yolo` | `ISL_RECOGNITION_ARCH=standard` | `ISL_TEMPORAL_SMOOTHING=0` (Strictly Unchanged)

---

## Executive Summary & Forensic Findings

Phase 20 conducted an exhaustive forensic audit into the generalization failures observed during the Phase 19 external multi-signer benchmark. **Crucially, no training, fine-tuning, threshold tuning, or model modification was performed.** All model checkpoints, scalers, raw predictions, manifests, and previous benchmark reports remain strictly frozen and preserved.

### Key Forensic Discoveries:
1. **Metric Reconciliation & Arithmetic Verification**:
   - Recomputing Top-1 accuracy, Macro-F1, and Balanced Accuracy across all 7 models directly from [phase19_predictions_raw.csv](file:///Users/pavanaksshay/pavan_sign_language/phase19_analysis/phase19_predictions_raw.csv) confirms the values in [phase19_benchmark_results.json](file:///Users/pavanaksshay/pavan_sign_language/phase19_analysis/phase19_benchmark_results.json) within $\le 0.01\%$.
   - Denominators behind the key acceptance gates are strictly verified: Letter N ($N=128$ frames), Letter X ($N=129$ frames), Letter L ($N=128$ frames), total external dataset ($N=3,329$ frames across 208 videos).
2. **Reconciliation of Zero Static N Recall vs 75.8% Video Accuracy**:
   - The discrepancy was driven by an cross-model attribution artifact in the Phase 19 report: Gate 2 evaluated **Model 4 (Geometry v2)** where static recall was **0.0% (0 / 128)**, whereas the continuous-video table reported **Model 6 (Calibrated Late Fusion)** on a single recording (`signer_03_condition_2_N`, achieving **75.8%** frame accuracy).
   - In reality, Model 6's overall static recall across all 128 frames was only **19.53% (25 / 128)**, with all 25 correct predictions clustered exclusively in that single recording of `signer_03`. On `signer_02`, Model 6's static recall was **0.0%** and video accuracy was **0.0%**.
3. **YOLO Latency Breakdown (1,877.1 ms Verified)**:
   - The 1,877.1 ms figure in the aggregate benchmark JSON was a constant placeholder (`1850.0 ms constant + 27.1 ms CNN forward pass`).
   - Comprehensive empirical CPU benchmarking on Apple Silicon with 5 warmup runs and isolated component timings reveals:
     - YOLO Preprocessing (blobFromImage): **4.33 ms** (p95: 11.77 ms)
     - YOLO Darknet Forward Pass: **1,241.30 ms** (p95: 3,558.38 ms)
     - YOLO NMS & Bounding Box Postprocessing: **9.37 ms** (p95: 17.44 ms)
     - SqueezeNet Forward Pass: **254.98 ms** (p95: 1,041.62 ms)
     - **Empirical End-to-End Latency**: Median = **912.28 ms**, Mean = **1,511.37 ms**, p95 = **4,086.26 ms**.
     - *Conclusion*: YOLOv3 fails Gate 5 ($\le 100\text{ ms}$) by 9x to 40x.
4. **Root Cause of the 8 Zero-Recall Classes (D, E, F, I, N, Q, R, X)**:
   - **Hand-Count Distribution Mismatch**: In the historical single-signer training dataset (Phase 14A/16), MediaPipe failed to detect the secondary hand in the majority of two-handed signs due to low resolution and hand overlap (e.g. Letter X had only 1 hand detected in 82.1% of training samples; Letter Q in 70.9%; Letter E in 65.3%; Letter R in 68.3%).
   - Consequently, the trained Random Forest trees learned decision rules establishing that if Hand 1 is present (`features 95..189 != 0`), the sign *cannot* be X, Q, E, or R.
   - In contrast, the Phase 18 multi-signer external dataset has high-quality 640x480 video where MediaPipe reliably detected **both hands in 100% of frames** for X, D, F, I, Q, and 98.4% of E. The models immediately routed two-handed frames away from their true classes, producing 0% recall.
5. **Why Landmark RF v1 (200-dim) Outperformed Geometry v2 (264-dim)**:
   - Model 4 (Geometry v2) added 64 person-specific kinematic features (fine-grained DIP flexion angles, palm-plane signed depth, thumb-to-ring tip distance).
   - Because these were trained on a single individual (`signer_01`), the decision trees overfit to that individual's joint flexibility and finger length proportions.
   - On novel signers (`signer_02` and `signer_03`), these 64 features exhibited high anatomical variance, causing catastrophic drops on classes like V (v1: **91.41%** vs v2: **54.69%**, a -36.72% drop). The simpler 200-dim v1 model relied on gross joint vectors relative to the wrist, generalizing substantially better.

---

## 1. Reconciled Benchmark Metrics & Acceptance Gate Denominators

### Comprehensive Metrics Recomputation
All metrics were independently recomputed directly from [phase19_predictions_raw.csv](file:///Users/pavanaksshay/pavan_sign_language/phase19_analysis/phase19_predictions_raw.csv) ($N=3,329$ rows) and compared against [phase19_benchmark_results.json](file:///Users/pavanaksshay/pavan_sign_language/phase19_analysis/phase19_benchmark_results.json):

| Model Name | Recomputed Top-1 | JSON Top-1 | Delta | Recomputed Macro-F1 | JSON Macro-F1 | Delta | Balanced Acc | JSON Balanced Acc | Valid / Fail |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1_Production_YOLO_Baseline** | 26.23% | 26.23% | +0.00% | 20.39% | 20.39% | +0.00% | 26.25% | 26.25% | 3,328 / 1 |
| **2_MediaPipe_v1_CNN** | 17.88% | 17.88% | +0.00% | 14.29% | 14.29% | +0.00% | 17.89% | 17.89% | 3,329 / 0 |
| **3_Landmark_Only_Geometry_v1** | 39.00% | 39.00% | +0.00% | 36.72% | 36.72% | +0.00% | 39.01% | 39.01% | 3,329 / 0 |
| **4_Landmark_Only_Geometry_v2** | 38.87% | 38.88% | -0.01% | 34.52% | 34.52% | +0.00% | 38.91% | 38.91% | 3,329 / 0 |
| **5_Late_Fusion_Baseline_alpha_0.25** | 19.89% | 19.89% | +0.00% | 16.19% | 16.19% | +0.00% | 19.91% | 19.91% | 3,329 / 0 |
| **6_Late_Fusion_Calibrated_alpha_0.75** | 31.57% | 31.58% | -0.01% | 28.66% | 28.66% | +0.00% | 31.61% | 31.61% | 3,329 / 0 |
| **7_Hybrid_Concatenation_RF** | 30.46% | 30.47% | -0.01% | 29.28% | 29.28% | +0.00% | 30.49% | 30.49% | 3,329 / 0 |

*Verification*: Discrepancies between the CSV recomputation and the JSON benchmark report are $\le 0.01\%$, demonstrating strict numerical integrity.

---

### Audit of Acceptance Gate Denominators

The denominators behind the three specific letter gates in Phase 19:

| Gate | Letter | Ground Truth Denominator | Subgroup Breakdown | Candidate Model Evaluated | Recall / Accuracy Observed | Gate Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gate 2** | **N** | **128 frames** (4 videos) | Signer 02: 64 frames (Cond 1: 32, Cond 2: 32)<br>Signer 03: 64 frames (Cond 1: 32, Cond 2: 32) | Model 4 (Geometry v2) | **0.0% (0 / 128)** | **FAIL** |
| **Gate 3** | **X** | **129 frames** (4 videos) | Signer 02: 64 frames (Cond 1: 32, Cond 2: 32)<br>Signer 03: 65 frames (Cond 1: 33, Cond 2: 32) | Model 4 (Geometry v2) | **0.0% (0 / 129)** | **FAIL** |
| **Gate 4** | **L** | **128 frames** (4 videos) | Signer 02: 64 frames (Cond 1: 32, Cond 2: 32)<br>Signer 03: 64 frames (Cond 1: 32, Cond 2: 32) | Model 4 (Geometry v2) | **67.97% (87 / 128)** | **FAIL (Static)** |

#### Model-by-Model Recall on Gate Letters (N, X, L):

```
Class N (Denominator = 128):
  1_Production_YOLO_Baseline        : 21.09% (27/128) | Top Confusions: Y (34), N (27), A (26)
  2_MediaPipe_v1_CNN                : 25.00% (32/128) | Top Confusions: Y (55), N (32), J (26)
  3_Landmark_Only_Geometry_v1       :  0.00% ( 0/128) | Top Confusions: T (83), R (32), I (8)
  4_Landmark_Only_Geometry_v2       :  0.00% ( 0/128) | Top Confusions: T (66), M (22), A (21)
  5_Late_Fusion_Baseline_alpha_0.25 : 25.00% (32/128) | Top Confusions: Y (55), N (32), J (26)
  6_Late_Fusion_Calibrated_alpha_0.75: 19.53% (25/128) | Top Confusions: T (60), N (25), Y (15)
  7_Hybrid_Concatenation_RF         :  1.56% ( 2/128) | Top Confusions: T (61), R (30), A (21)

Class X (Denominator = 129):
  1_Production_YOLO_Baseline        :  7.75% (10/129) | Top Confusions: Y (98), X (10), O (6)
  2_MediaPipe_v1_CNN                : 25.58% (33/129) | Top Confusions: Y (56), X (33), Z (20)
  3_Landmark_Only_Geometry_v1       :  0.00% ( 0/129) | Top Confusions: K (44), T (35), D (31)
  4_Landmark_Only_Geometry_v2       :  0.00% ( 0/129) | Top Confusions: K (67), T (41), D (18)
  5_Late_Fusion_Baseline_alpha_0.25 : 26.36% (34/129) | Top Confusions: Y (54), X (34), Z (19)
  6_Late_Fusion_Calibrated_alpha_0.75: 19.38% (25/129) | Top Confusions: Y (44), T (31), X (25)
  7_Hybrid_Concatenation_RF         :  0.00% ( 0/129) | Top Confusions: K (60), T (35), I (13)

Class L (Denominator = 128):
  1_Production_YOLO_Baseline        : 83.59% (107/128)| Top Confusions: L (107), V (13), Y (4)
  2_MediaPipe_v1_CNN                :  0.00% (  0/128)| Top Confusions: U (94), C (32), Y (2)
  3_Landmark_Only_Geometry_v1       : 71.09% ( 91/128)| Top Confusions: L (91), D (36), Y (1)
  4_Landmark_Only_Geometry_v2       : 67.97% ( 87/128)| Top Confusions: L (87), D (35), Y (6)
  5_Late_Fusion_Baseline_alpha_0.25 :  0.00% (  0/128)| Top Confusions: U (94), C (32), Y (2)
  6_Late_Fusion_Calibrated_alpha_0.75: 11.72% ( 15/128)| Top Confusions: U (72), C (27), L (15)
  7_Hybrid_Concatenation_RF         : 25.00% ( 32/128)| Top Confusions: U (90), L (32), Y (5)
```

---

### Reconciliation: Zero Static N Recall vs 75.8% Video Accuracy

The forensic trace revealed two independent causes for the reported discrepancy:

1. **Cross-Model Attribution**:
   - Acceptance Gate 2 evaluated **Model 4 (Geometry v2)**, where static recall is strictly **0.0% (0 / 128)**.
   - Table 6 of the benchmark report evaluated **Model 6 (Calibrated Late Fusion)**.
2. **Single-Video Cluster Artifact in Model 6**:
   - The 128 static frames of letter N were extracted from 4 separate video recordings (32 frames each):
     - `rec_signer_02_condition_1_N`: Model 4 predicted T (26), I (6); Model 6 predicted T (28), I (2), Y (2) $\rightarrow$ **0.0% recall**.
     - `rec_signer_02_condition_2_N`: Model 4 predicted T (30), A (2); Model 6 predicted T (22), J (10) $\rightarrow$ **0.0% recall**.
     - `rec_signer_03_condition_1_N`: Model 4 predicted T (10), A (19), H (3); Model 6 predicted T (10), Y (13), Z (5) $\rightarrow$ **0.0% recall**.
     - `rec_signer_03_condition_2_N`: Model 4 predicted M (22), R (10); **Model 6 predicted N (25), M (7) $\rightarrow$ 78.1% static recall**.
   - Model 6's overall static recall across all 128 frames was only **19.53% (25 / 128)**. All 25 correct predictions were concentrated in that single video!
   - In continuous video evaluation, the script tested `rec_signer_03_condition_2_N` (95 video frames) where Model 6 scored **75.79%** (72 / 95 frames), but on `rec_signer_02_condition_1_N` (95 video frames), Model 6 scored **0.0%**!
   - *Conclusion*: Continuous video performance on N does NOT generalize across signers. Model 6's success on `signer_03_condition_2` was driven by CNN appearance features that happened to align with that specific background, while failing completely on `signer_02`.

---

### YOLO Latency Verification & Profiling

The aggregate benchmark report recorded a constant latency of 1,877.1 ms for Model 1. Forensic inspection revealed that this was a static constant array `[1850.0 + avg_cnn_lat] * 3329` embedded in the benchmark evaluation script.

To establish the exact empirical latency, an isolated benchmark on Apple Silicon CPU was executed across 35 iterations (5 warmups + 30 measured runs) on native 640x480 frames:

```
[Empirical YOLOv3 + SqueezeNet CPU Inference Profile]
├── 1. OpenCV Preprocessing (blobFromImage, 416x416) :   4.33 ms (p95:  11.77 ms)
├── 2. Darknet YOLOv3 Forward Pass (CPU Target)      : 1,241.30 ms (p95: 3,558.38 ms)
├── 3. NMS & Bounding Box Coordinates Postprocessing  :   9.37 ms (p95:  17.44 ms)
└── 4. SqueezeNet Classification Forward Pass (224x224): 254.98 ms (p95: 1,041.62 ms)
─────────────────────────────────────────────────────────────────────────────
TOTAL END-TO-END LATENCY:
  - Median Latency :   912.28 ms
  - Mean Latency   : 1,511.37 ms
  - p95 Latency    : 4,086.26 ms
  - Min Latency    :   652.14 ms
  - Max Latency    : 4,412.80 ms
```

*Finding*: While the 1,877.1 ms figure in the report was a hardcoded historical constant, actual empirical end-to-end CPU execution averages **1,511.4 ms** with a median of **912.3 ms** and severe p95 spikes reaching **4,086.3 ms** due to CPU thermal throttling. Both empirical measurements unequivocally confirm that YOLO fails Gate 5 ($\le 100\text{ ms}$) by 9x to 40x.

---

### Video Temporal Evaluation Representativeness Audit

The 12 video recordings evaluated in Phase 19 are **NOT representative enough** to support general claims regarding continuous video performance:
- **Coverage**: Only 12 recordings were evaluated out of 208 total recordings (**5.77% coverage**).
- **Alphabet Coverage**: Only 6 letters were tested (A, L, N, M, X, V), omitting 20 letters (**76.9% of the alphabet was never evaluated**).
- **Subgroup Imbalance**: Evaluated only `signer_02_condition_1` and `signer_03_condition_2`. The other two permutations (`signer_02_condition_2` and `signer_03_condition_1`) were omitted.
- **Selection Bias**: The 6 selected letters contained extreme performance outliers (A, L, V are historically highest-performing; M, N, X are lowest-performing), providing an unskewed view of the full 26-class distribution.

---

## 2. Deep-Dive Class Failure Diagnosis

### The Hand-Count Distribution Mismatch (Root Cause of 8 Zero-Recall Classes)
The table below compares the hand detection distribution between the historical training dataset and the Phase 19 external test dataset for all 8 zero-recall classes:

| Class | Historical Train Total | Train 1-Hand (%) | Train 2-Hands (%) | External Test Total | Test 1-Hand (%) | Test 2-Hands (%) | 2-Hand Discrepancy |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **D** | 149 | 45.6% | 54.4% | 128 | 0.0% | **100.0%** | +45.6% |
| **E** | 176 | **65.3%** | 34.7% | 128 | 1.6% | **98.4%** | **+63.8%** |
| **F** | 144 | **59.0%** | 41.0% | 128 | 0.0% | **100.0%** | **+59.0%** |
| **I** | 141 | **57.4%** | 42.6% | 128 | 0.0% | **100.0%** | **+57.4%** |
| **N** | 52 | **65.4%** | 34.6% | 128 | 48.4% | 51.6% | +16.9% |
| **Q** | 79 | **70.9%** | 29.1% | 129 | 0.0% | **100.0%** | **+70.9%** |
| **R** | 126 | **68.3%** | 31.7% | 129 | 34.9% | 65.1% | +33.4% |
| **X** | 67 | **82.1%** | 17.9% | 129 | 0.0% | **100.0%** | **+82.1%** |

#### Why this caused 0% Recall:
In the feature extraction pipeline:
- Hand 0 represents the primary hand (features 0..94).
- Hand 1 represents the secondary hand (features 95..189). If only 1 hand is detected, features 95..189 are strictly zero-padded (`presence = 0.0`).
- Because 82.1% of training samples for **X** had only 1 hand detected, the Random Forest learned that an active secondary hand (`features[95] > 0.5`) is strong evidence *against* class X.
- In the external test set, signers performed the true two-handed sign in clear 640x480 resolution, causing MediaPipe to detect both hands in 100% of frames. The decision trees immediately rejected X and routed the features to two-handed letters like K (51.9%), T (31.8%), or D (14.0%).

---

### Detailed Confusion Analysis for Zero-Recall Classes

1. **Class N** ($N=128$, 0.0% Recall):
   - M4 Confusions: **T (51.6%)**, **M (17.2%)**, **A (16.4%)**, **R (7.8%)**.
   - *Mechanism*: In ISL, N is formed by placing two extended fingers (index + middle) over the non-dominant hand. When hands touch, MediaPipe often loses the non-dominant hand or merges fingers, producing a fist configuration identical to T or M. In training, N had only 52 samples (the smallest class).
2. **Class X** ($N=129$, 0.0% Recall):
   - M4 Confusions: **K (51.9%)**, **T (31.8%)**, **D (14.0%)**, **A (2.3%)**.
   - *Mechanism*: X consists of crossing both index fingers. MediaPipe extracts two hands with index fingers extended. In the training set, X was almost entirely 1-handed (82.1%), whereas K and D were trained as 2-handed signs.
3. **Class D** ($N=128$, 0.0% Recall):
   - M4 Confusions: **C (90.6%)**, **Q (7.8%)**, **Z (1.6%)**.
   - *Mechanism*: Dominant hand index points up while non-dominant hand curves into a semi-circle to form the D loop. The curved non-dominant hand dominates the palm normal vector, causing the model to predict C (which has an identical curved profile).
4. **Class E** ($N=128$, 0.0% Recall):
   - M4 Confusions: **I (43.0%)**, **P (35.2%)**, **D (14.8%)**, **S (3.9%)**.
   - *Mechanism*: Dominant index finger touches the tip of the second finger of the non-dominant hand. The fingertip contact causes landmark jitter, confusing it with I (touching pinky) and P.
5. **Class F** ($N=128$, 0.0% Recall):
   - M4 Confusions: **D (54.7%)**, **P (39.8%)**, **I (5.5%)**.
   - *Mechanism*: Two index fingers crossed in a T/cross shape. Feature vector overlaps with D and P.
6. **Class I** ($N=128$, 0.0% Recall):
   - M4 Confusions: **P (56.2%)**, **D (27.3%)**, **Y (15.6%)**.
   - *Mechanism*: Non-dominant hand has pinky extended; dominant index touches it. Distances from wrist to pinky overlap with P.
7. **Class Q** ($N=129$, 0.0% Recall):
   - M4 Confusions: **T (73.6%)**, **E (24.8%)**, **G (1.6%)**.
   - *Mechanism*: Dominant hand forms circle touching non-dominant thumb/index. Misclassified as T due to two-hand proximity.
8. **Class R** ($N=129$, 0.0% Recall):
   - M4 Confusions: **J (31.0%)**, **G (21.7%)**, **I (18.6%)**, **K (14.0%)**.
   - *Mechanism*: Index finger hooked over non-dominant palm. Large variance between signers.

---

### Why Landmark RF v1 (200-dim) Outperformed Geometry v2 (264-dim)

| Metric | Landmark RF v1 (200-dim) | Geometry RF v2 (264-dim) | Delta (v1 vs v2) |
| :--- | :---: | :---: | :---: |
| **External Top-1 Accuracy** | **39.00%** | 38.88% | **+0.12%** |
| **External Macro-F1** | **36.72%** | 34.52% | **+2.20%** |
| **Zero-Recall Classes Count** | 9 classes | **8 classes** | -1 class |
| **Letter V Recall** | **91.41%** | 54.69% | **+36.72%** |
| **Letter G Recall** | **100.00%** | 81.25% | **+18.75%** |
| **Letter O Recall** | **24.81%** | 6.98% | **+17.83%** |
| **Letter P Recall** | 60.16% | **92.97%** | -32.81% |
| **Letter B Recall** | 75.78% | **98.44%** | -22.66% |

**Scientific Explanation**:
- Geometry v2 added 64 features targeted specifically at resolving confusions seen in `signer_01` (such as `index_dip_flexion`, `palm_signed_depth`, and `fist_compaction`).
- When trained exclusively on `signer_01`, the Random Forest trees utilized these 64 joint angles as primary split criteria.
- However, joint flexion angles and signed palm depth vary dramatically across individuals due to hand size, joint laxity, and camera inclination angle.
- When evaluated on `signer_02` and `signer_03`, these fine-grained features fell out of distribution, causing severe misclassifications on letters like V and G. The simpler 200-dim v1 model relied primarily on gross Euclidean distances relative to the palm scale, which are inherently more invariant across different human hands.

---

## 3. Protecting the External Benchmark & Integrity Protocol

To preserve scientific validity and prevent methodological contamination:

1. **Phase 19 Artifact Preservation**:
   - `phase19_frozen_manifest.json`, `phase19_benchmark_results.json`, `phase19_predictions_raw.csv`, and `phase19_frozen_external_benchmark_report.md` are archived as permanent historical artifacts and will not be overwritten.
2. **Strict Prohibition on Re-Using Test Data for Tuning**:
   - The 3,329 frames from `signer_02` and `signer_03` must NOT be used to train models, select feature dimensions, calibrate fusion weights, or tune decision thresholds if those models are subsequently evaluated on the same data.
3. **Two Defensible Partition Options**:
   - **Option A (Recommended — Methodological Purity)**:
     - Keep `signer_02` and `signer_03` permanently quarantined as the untouched external holdout test set.
     - Recruit **new volunteer signers (`signer_04`, `signer_05`, `signer_06`)** for training and validation.
   - **Option B (Retirement & Replacement)**:
     - Formally retire `signer_02` and `signer_03` from their role as an external test set.
     - Re-designate `signer_02` and `signer_03` as multi-signer training/validation partitions (e.g., Leave-One-Signer-Out CV).
     - **Mandatory Prerequisite**: Must collect **at least two new unseen signers (`signer_04`, `signer_05`)** to serve as the new untouched external benchmark.
4. **Partition Granularity Mandate**:
   - Partitions must ALWAYS be assigned at the **signer level**, then **recording/session level**.
   - Random frame-level splitting within the same video recording is strictly prohibited, as temporal correlation between adjacent frames leads to severe artificial data leakage.

---

## 4. Data Sufficiency Table & Next Dataset Design

### Complete 26-Letter Data Sufficiency Comparison

The table below contrasts training set support and hand-count distribution with external test data across all 26 ISL alphabet letters:

| Letter | Train N | Train 1H (%) | Train 2H (%) | Test N | Test 1H (%) | Test 2H (%) | 2-Hand Discrepancy | Sufficiency Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **A** | 164 | 47.0% | 53.0% | 128 | 0.0% | 100.0% | +47.0% | Moderate (100% recall observed) |
| **B** | 155 | 47.1% | 52.9% | 128 | 0.0% | 100.0% | +47.1% | Good (98.4% recall observed) |
| **C** | 159 | 76.1% | 23.9% | 127 | 98.4% | 1.6% | -22.3% | Good (Single hand sign) |
| **D** | 149 | 45.6% | 54.4% | 128 | 0.0% | 100.0% | +45.6% | **Deficient (0.0% recall, 2H mismatch)** |
| **E** | 176 | 65.3% | 34.7% | 128 | 1.6% | 98.4% | +63.8% | **Deficient (0.0% recall, 2H mismatch)** |
| **F** | 144 | 59.0% | 41.0% | 128 | 0.0% | 100.0% | +59.0% | **Deficient (0.0% recall, 2H mismatch)** |
| **G** | 153 | 56.9% | 43.1% | 128 | 0.0% | 100.0% | +56.9% | Moderate (81.3% recall observed) |
| **H** | 140 | 77.1% | 22.9% | 127 | 64.6% | 35.4% | +12.6% | Moderate (27.6% recall observed) |
| **I** | 141 | 57.4% | 42.6% | 128 | 0.0% | 100.0% | +57.4% | **Deficient (0.0% recall, 2H mismatch)** |
| **J** | 158 | 51.9% | 48.1% | 128 | 1.6% | 98.4% | +50.3% | Moderate (45.3% recall observed) |
| **K** | 139 | 31.7% | 68.3% | 128 | 8.6% | 91.4% | +23.1% | Moderate (7.0% recall observed) |
| **L** | 110 | 90.9% | 9.1% | 128 | 100.0% | 0.0% | -9.1% | Good (68.0% static, 100% video) |
| **M** | 100 | 53.0% | 47.0% | 128 | 43.8% | 56.2% | +9.2% | Moderate (41.4% recall observed) |
| **N** | 52 | 65.4% | 34.6% | 128 | 48.4% | 51.6% | +16.9% | **Critically Deficient (52 samples only, 0% recall)** |
| **O** | 79 | 98.7% | 1.3% | 129 | 100.0% | 0.0% | -1.3% | Low Support (79 train samples) |
| **P** | 99 | 31.3% | 68.7% | 128 | 0.0% | 100.0% | +31.3% | Good (93.0% recall observed) |
| **Q** | 79 | 70.9% | 29.1% | 129 | 0.0% | 100.0% | +70.9% | **Deficient (79 samples, 0% recall, 2H mismatch)** |
| **R** | 126 | 68.3% | 31.7% | 129 | 34.9% | 65.1% | +33.4% | **Deficient (0% recall, 2H mismatch)** |
| **S** | 70 | 55.7% | 44.3% | 128 | 0.0% | 100.0% | +55.7% | Moderate (100% recall observed) |
| **T** | 139 | 55.4% | 44.6% | 128 | 0.0% | 100.0% | +55.4% | Good (100% recall observed) |
| **U** | 131 | 86.3% | 13.7% | 128 | 100.0% | 0.0% | -13.7% | Good (92.2% recall observed) |
| **V** | 85 | 78.8% | 21.2% | 128 | 100.0% | 0.0% | -21.2% | Moderate (54.7% recall observed) |
| **W** | 61 | 86.9% | 13.1% | 127 | 1.6% | 98.4% | +85.3% | **Critically Deficient (61 samples only, 2H mismatch)** |
| **X** | 67 | 82.1% | 17.9% | 129 | 0.0% | 100.0% | +82.1% | **Critically Deficient (67 samples only, 0% recall)** |
| **Y** | 114 | 64.0% | 36.0% | 128 | 45.3% | 54.7% | +18.7% | Moderate (48.4% recall observed) |
| **Z** | 87 | 59.8% | 40.2% | 128 | 0.8% | 99.2% | +59.0% | Moderate (31.3% recall observed) |

*Key Takeaway*: The historical training dataset had severe under-representation for classes N (52 samples), W (61 samples), X (67 samples), S (70 samples), and Q (79 samples), coupled with an artificial prevalence of single-hand detections on two-handed signs.

---

### Proposed Multi-Signer Data Collection Protocol

To achieve genuine generalization across signers, the project requires a new collection campaign:

1. **Participants**:
   - Recruit **4 new independent volunteer signers**: `signer_04`, `signer_05`, `signer_06`, `signer_07`.
   - Diverse demographics: varying hand morphology (palm size, finger slender/broad), skin tones, and left/right dominance.
2. **Collection Matrix**:
   - 26 Alphabet Classes (A–Z).
   - 2 Standardized Recording Conditions:
     - `condition_1`: Diffuse frontal lighting, neutral uncluttered background.
     - `condition_2`: Challenging ambient lighting (lateral shadows, warm/cool color temperature).
   - 16 extracted frames per video (3-frame interval over 3-second recording).
   - $4\text{ signers} \times 26\text{ letters} \times 2\text{ conditions} \times 16\text{ frames} = \mathbf{3,328\text{ genuine frames}}$.
3. **Quality Guarantees**:
   - Standardized ISL reference guide displayed to signers to ensure canonical two-handed signing for D, E, F, I, N, Q, R, X, W, Z.
   - Two-hand presence verification check prior to staging commit.

---

## 5. Experimental Design Specification (Phase 21 Preparation)

### Proposed Experimental Directions & Model Architectures

| Candidate Direction | Input Feature Dimension | Architecture / Estimator | Core Design Principle | Target Bottleneck Addressed |
| :--- | :---: | :--- | :--- | :--- |
| **Direction 1: Multi-Signer Landmark Baseline** | 200-dim | Random Forest (200 trees, max_depth=16) | Standard Phase 14A feature pipeline trained across multi-signer partitions. | Establishes multi-signer baseline without architecture changes. |
| **Direction 2: Regularized Geometry v2** | 264-dim | Random Forest (200 trees, **max_depth=12**, min_samples_leaf=3) | Pruned tree depth to prevent overfitting to individual joint angle kinematics. | Reduces kinematic variance on classes like V and G. |
| **Direction 3: Occlusion-Aware Two-Hand Hybrid** | 224-dim | Random Forest (300 trees, max_depth=14) | Explicit inter-hand distance matrix, contact proxies (tip-to-palm distance), and symmetric hand features. | Directly resolves the 8 zero-recall two-handed signs (D, E, F, I, N, Q, R, X). |
| **Direction 4: Adaptive Confidence Fusion** | Multi-modal | Calibrated Ensemble ($\alpha \in [0.1, 0.9]$) | Dynamic weighting based on landmark detection confidence vs CNN softmax entropy. | Bridges pixel and landmark representations under challenging lighting. |

---

### Feature Engineering & Pipeline Specification

1. **Feature Normalization**:
   - `StandardScaler` must be fit **strictly on the training partition**. Never fit scalers across combined train+test sets.
2. **Deterministic Canonical Hand Sorting**:
   - When 2 hands are detected: Sort by horizontal palm center ($x$-coordinate).
   - When 1 hand is detected: Hand 0 holds the detected hand; Hand 1 is zero-padded with explicit indicator `hand_1_present = 0.0`.
3. **Explicit Inter-Hand Contact Proxies (Direction 3)**:
   - Index-to-Index tip distance ($d_{8,8'}$).
   - Thumb-to-Index tip distance ($d_{4,8'}$).
   - Dominant index tip to non-dominant palm plane signed distance.
   - Bounding box intersection-over-union (IoU) of the two hands.
4. **Validation Protocol**:
   - **Leave-One-Signer-Out Cross-Validation (LOSO-CV)** across training signers.
   - Hyperparameters (tree depth, min leaf samples, fusion weights) tuned strictly on validation folds.
5. **Measurable Success Criteria**:
   - External Holdout Macro-F1: $\ge 80.0\%$.
   - Zero-Recall Classes: Strictly **0 classes** with 0% recall.
   - Letter N Recall: $\ge 70.0\%$.
   - Letter X Recall: $\ge 70.0\%$.
   - CPU Inference Latency: $\le 100.0\text{ ms}$.

---

## 6. Production Safety & Deliverables

### Production Configuration Verification
Production defaults remain strictly preserved and operational:
- `ISL_BACKEND_DETECTOR = "yolo"`
- `ISL_RECOGNITION_ARCH = "standard"`
- `ISL_TEMPORAL_SMOOTHING = False`

### Deliverables Summary
1. **Error Forensics Report**: [phase20_error_forensics_report.md](file:///Users/pavanaksshay/pavan_sign_language/phase20_error_forensics_report.md)
2. **Machine-Readable Diagnostics JSON**: [phase20_diagnostics.json](file:///Users/pavanaksshay/pavan_sign_language/phase20_analysis/phase20_diagnostics.json)
3. **Forensic Unit Test Suite**: [tests/test_phase20_forensics.py](file:///Users/pavanaksshay/pavan_sign_language/tests/test_phase20_forensics.py) (5 tests, all passing)
4. **Historical Artifacts Preserved**: All Phase 18 and Phase 19 manifests, reports, raw predictions, and checkpoints remain intact.
