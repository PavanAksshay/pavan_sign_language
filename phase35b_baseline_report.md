# Phase 35B — Leakage-Safe Baseline Reproduction and Evaluation Split Audit Report

**Document Identifier:** `PHASE35B-BASELINE-AUDIT-001`  
**Phase:** Phase 35B — Leakage-Safe Baseline Reproduction and Evaluation Split Audit  
**Execution Date:** October 28, 2026  
**Repository:** `/Users/pavanaksshay/pavan_sign_language`  
**Execution Environment:** macOS 15.6.0 (Darwin 25.6.0 arm64, Apple M1, Python 3.12.4, scikit-learn 1.9.1)  
**Governance Scope:** Experimental baseline training and split audit only. Production Model C, production scaler, frozen benchmark artifacts, sealed holdout, and production runtime defaults remain **strictly untouched and unmodified**.  
**Key Output Artifacts:**
- [`phase35b_split_audit.json`](file:///Users/pavanaksshay/pavan_sign_language/phase35b_split_audit.json)
- [`phase35b_baseline_metrics.json`](file:///Users/pavanaksshay/pavan_sign_language/phase35b_baseline_metrics.json)
- [`phase35b_experimental_model_manifest.json`](file:///Users/pavanaksshay/pavan_sign_language/phase35b_experimental_model_manifest.json)
- [`phase35b_analysis/models/Model_C_Baseline_Fold1_Primary.joblib`](file:///Users/pavanaksshay/pavan_sign_language/phase35b_analysis/models/Model_C_Baseline_Fold1_Primary.joblib)
- [`phase35b_analysis/models/Model_C_Baseline_Fold1_Primary_scaler.joblib`](file:///Users/pavanaksshay/pavan_sign_language/phase35b_analysis/models/Model_C_Baseline_Fold1_Primary_scaler.joblib)
- [`tests/test_phase35b_baseline.py`](file:///Users/pavanaksshay/pavan_sign_language/tests/test_phase35b_baseline.py)

---

## Executive Summary

Phase 35B establishes a reproducible, leakage-safe experimental baseline for the ISL recognition pipeline using only permitted training data and a defensible, recording-disjoint development evaluation protocol.

### Key Milestones Achieved
1. **Split Audit and Confound Unmasking:** We audited the 54 physical recordings (52 canonical accepted takes) of `signer_04`. We demonstrated that the historical alternating split (Split 1: A–M Condition 1 train / Condition 2 val; N–Z Condition 2 train / Condition 1 val) creates an exact **100% block-confound between letter group and evaluation condition**. Because each letter has only one recording per condition, testing every letter under both conditions in a single split while keeping whole recordings disjoint is physically impossible. To scientifically resolve this confound, we implemented and evaluated a **complete 2-fold cross-condition cross-validation protocol (Fold 1 + Fold 2)** and a **pure cross-condition benchmark (Condition 1 $\to$ Condition 2)**.
2. **Training Provenance and Baseline Reproduction:** We confirmed forensically that the historical production checkpoint (`Model_C_Occlusion_Aware_RF_multisigner.joblib`) was trained on all 1,621 frames of `signer_04`, rendering any previous validation scores on those frames contaminated. We trained a new experimental baseline in a dedicated directory ([`phase35b_analysis/models/`](file:///Users/pavanaksshay/pavan_sign_language/phase35b_analysis/models/)) using **only the approved 3,077 `signer_01` training frames and the 810 `signer_04` training frames in Fold 1**. All scalers were fitted **strictly on training samples**.
3. **Pre-Declared Performance Milestone Passed:**
   * **Fold 1 (Primary Baseline):** Top-1 Accuracy: **95.19%**, Macro-F1: **94.82%**, Balanced Accuracy: **95.08%** (Pre-declared Macro-F1 milestone $\ge 75.0\%$ **PASSED**).
   * **Fold 2 (Inverted Dual):** Top-1 Accuracy: **89.38%**, Macro-F1: **87.40%**, Balanced Accuracy: **89.28%** (Exposing camera perspective fragility on letter `C` at 0.0% recall).
   * **2-Fold Cross-Condition Mean:** Top-1 Accuracy: **92.29%**, Macro-F1: **91.11%**, Balanced Accuracy: **92.18%**.
   * **Pure Cross-Condition (C1 $\to$ C2):** Top-1 Accuracy: **93.51%**, Macro-F1: **92.69%**, Balanced Accuracy: **93.51%**.
4. **Production Safeguards Preserved:** The production Model C checkpoint and scaler remain 100% bit-for-bit identical to authoritative hashes. Production defaults (`yolo`, `standard`, `temporal_smoothing=0`, `cpu`, `native`) remain unaltered.

---

## 1. Audit of the Proposed `signer_04` Split Before Training

### Inventory and Session Provenance
* **Participant:** `signer_04`
* **Session ID:** `session_20261022_01`
* **Canonical Accepted Recordings:** 52 recordings (26 letters $\times$ 2 conditions).
* **Usable Extracted Frames:** 1,621 frames.
  * **Condition 1 (Indoor overhead, frontal eye-level, ~75cm):** 26 recordings, 789 frames.
  * **Condition 2 (Lateral ~15° angle, distance 60–90cm):** 26 recordings, 832 frames.
* **Prior Exposure:** Contributed directly to production Model C training weights and hyperparameter tuning in Phase 22. Classified as **previously exposed development data**.

### Investigation of the Letter-Condition Confound

In Phase 22, development Split 1 assigned:
* **Train:** Letters A–M under Condition 1 (395 frames) + Letters N–Z under Condition 2 (415 frames) = 810 frames.
* **Val:** Letters A–M under Condition 2 (417 frames) + Letters N–Z under Condition 1 (394 frames) = 811 frames.

```
+-------------------------------------------------------------------------------+
|                       SPLIT 1 BLOCK-CONFOUND STRUCTURE                        |
+---------------------+-------------------------+-------------------------------+
| Letter Group        | Training Condition      | Validation Evaluation Condition|
+---------------------+-------------------------+-------------------------------+
| Letters A through M | Condition 1 (Frontal)   | Condition 2 (Lateral 15°)     |
| Letters N through Z | Condition 2 (Lateral)   | Condition 1 (Frontal)         |
+---------------------+-------------------------+-------------------------------+
Confound: Letter identity block [A-M vs N-Z] is 100% correlated with viewpoint direction.
```

#### Why This Confound Distorts Evaluation
Condition 2 introduces lateral perspective ($15^\circ$) and distance variation. When letters A–M are evaluated in validation under Split 1, they are **exclusively tested on lateral Condition 2 without ever having seen Condition 2 in training**. Conversely, letters N–Z in validation are **exclusively tested on benign frontal Condition 1**.
* If letter `H` fails in Split 1 validation (recall = 43.8%), is `H` inherently difficult, or is it failing because of lateral camera perspective?
* If letter `N` succeeds (recall = 100.0%), is `N` easy, or is it succeeding because frontal evaluation is forgiving?

#### Can a Single Recording-Disjoint Split Eliminate This Confound?
**No.** Because `signer_04` contains exactly one recording per letter in Condition 1 and one recording per letter in Condition 2, placing whole recordings into disjoint partitions means that for any letter $L$, one take goes to Train and one take goes to Val. Thus, in any single split, each letter is only ever tested under a single condition.

#### Defensible Mitigation Protocol
To eliminate the confound without manufacturing synthetic recordings or splitting video frames, we established a **2-Fold Cross-Condition Cross-Validation Protocol**:
* **Fold 1 (Primary Baseline):** A–M ($C_1 \to C_2$), N–Z ($C_2 \to C_1$).
* **Fold 2 (Inverted Dual):** A–M ($C_2 \to C_1$), N–Z ($C_1 \to C_2$).
* **Pure Cross-Condition (Split 2):** All 26 letters ($C_1 \to C_2$).

Across Fold 1 and Fold 2, **every letter is tested under both conditions**, enabling an unconfounded comparison of sign morphology versus camera perspective.

---

## 2. Verification of Training Provenance and Experimental Baseline

### Historical Checkpoint Provenance Audit
Inspection of `scripts/train_and_evaluate_phase22.py` (lines 541–555) and `phase22_analysis/models/phase22_model_manifest.json` confirms:
* The production checkpoint `Model_C_Occlusion_Aware_RF_multisigner.joblib` was trained on all 4,698 samples (3,077 `signer_01` + all 1,621 `signer_04` frames).
* The production scaler `Model_C_Occlusion_Aware_RF_multisigner_scaler.joblib` was fitted on all 4,698 samples.
* **Finding:** The production Model C checkpoint cannot be used to evaluate validation performance because its training set contained all validation recordings.

### New Experimental Baseline Specification
We trained an independent experimental baseline stored in [`phase35b_analysis/models/`](file:///Users/pavanaksshay/pavan_sign_language/phase35b_analysis/models/):
* **Training Data:** Approved 3,077 `signer_01` frames + approved 810 `signer_04_train` frames in Fold 1 = **3,887 training samples**.
* **Validation Data:** Approved 811 `signer_04_val` frames (26 holdout recordings).
* **Scaler Discipline:** `StandardScaler` fitted **strictly on the 3,887 training samples**.
* **Feature Representation:** 280 dimensions (264-dim Geometry v2 + 16-dim Occlusion/Contact vector).
* **Model Hyperparameters:** `RandomForestClassifier(n_estimators=300, max_depth=14, min_samples_leaf=2, random_state=42, n_jobs=-1)`.
* **Immutability:** Production Model C and scaler remain byte-for-byte untouched.

---

## 3. Comprehensive Baseline Metrics and Milestone Evaluation

### Overall Performance Summary

| Metric | Fold 1 (Primary Baseline) | Fold 2 (Inverted Dual) | 2-Fold CV Mean | Pure Cross-Condition ($C_1 \to C_2$) | Pre-Declared Milestone | Gate Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Top-1 Accuracy** | **95.19%** (772/811) | **89.38%** (724/810) | **92.29%** | **93.51%** (778/832) | N/A | **REPORTED** |
| **Top-3 Accuracy** | **97.90%** (794/811) | **93.95%** (761/810) | **95.93%** | **96.75%** (805/832) | N/A | **REPORTED** |
| **Macro-Averaged F1** | **94.82%** | **87.40%** | **91.11%** | **92.69%** | $\ge \mathbf{75.0\%}$ | **PASSED** |
| **Balanced Accuracy** | **95.08%** | **89.28%** | **92.18%** | **93.51%** | N/A | **REPORTED** |
| **Recording Macro Accuracy** | **95.08%** | **89.28%** | **92.18%** | **93.51%** | N/A | **REPORTED** |
| **Zero-Recall Classes** | **0 classes** | **1 class (`C`)** | **0.5 classes** | **0 classes** | 0 classes | **MONITORED** |

### Condition-Stratified Performance Breakdown

```
Fold 1 (Primary Baseline):
  - Condition 1 (Letters N-Z, Frontal):  394 frames | Top-1: 95.69% | Macro-F1: 73.95% | Bal Acc: 95.45%
  - Condition 2 (Letters A-M, Lateral):  417 frames | Top-1: 94.72% | Macro-F1: 77.65% | Bal Acc: 94.71%

Fold 2 (Inverted Dual):
  - Condition 1 (Letters A-M, Frontal):  395 frames | Top-1: 87.59% | Macro-F1: 79.41% | Bal Acc: 87.46%
  - Condition 2 (Letters N-Z, Lateral):  415 frames | Top-1: 91.08% | Macro-F1: 63.47% | Bal Acc: 91.11%
```

### Forensic Failure Analysis: The Letter `C` and Letter `H` Inversion

Comparing Fold 1 and Fold 2 unmasks how camera perspective impacts specific ISL signs:

| Letter | Fold 1 Transfer Direction | Fold 1 Recall | Fold 2 Transfer Direction | Fold 2 Recall | Primary Misclassifications in Fold 2 | Underlying Failure Mechanism |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`C`** | $C_1 \to C_2$ (Train Frontal $\to$ Val Lateral) | **100.0%** | $C_2 \to C_1$ (Train Lateral $\to$ Val Frontal) | **0.0%** (0/30) | Predicted as `D` (17/30), `O` (13/30) | Camera coordinate distortion: when trained on lateral tilt, the curved index finger DIP projects identically to `D`/`O` in frontal view. |
| **`H`** | $C_2 \to C_1$ (Train Frontal $\to$ Val Lateral) | **43.8%** (14/32) | $C_1 \to C_2$ (Train Lateral $\to$ Val Frontal) | **100.0%** (30/30) | Predicted as `G` (18/32) | Middle finger self-occlusion behind index finger under lateral angle. |
| **`I`** | $C_1 \to C_2$ | **90.6%** (29/32) | $C_2 \to C_1$ | **46.7%** (14/30) | Predicted as `J` (16/30) | Little finger vertical vector misattributed to dynamic loop trajectory. |
| **`W`** | $C_2 \to C_1$ | **80.8%** (21/26) | $C_1 \to C_2$ | **31.2%** (10/32) | Predicted as `V` (22/32) | Ring finger tip occlusion at distance under lateral angle. |

**Key Takeaway:** Evaluating only Fold 1 created an illusion that `C` was perfectly robust (100% recall), while concealing that `C` collapses to 0.0% when viewpoint direction is inverted. The 2-fold cross-condition evaluation directly exposes this fragility.

### Per-Letter Metrics Table (Fold 1 Primary Baseline)

| Letter | Precision | Recall | F1 Score | Support (Frames) | Evaluation Condition |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **A** | 0.9412 | 1.0000 | 0.9697 | 32 | Condition 2 (Lateral) |
| **B** | 1.0000 | 1.0000 | 1.0000 | 33 | Condition 2 (Lateral) |
| **C** | 1.0000 | 1.0000 | 1.0000 | 33 | Condition 2 (Lateral) |
| **D** | 0.9394 | 1.0000 | 0.9688 | 31 | Condition 2 (Lateral) |
| **E** | 0.9412 | 1.0000 | 0.9697 | 32 | Condition 2 (Lateral) |
| **F** | 0.8889 | 1.0000 | 0.9412 | 32 | Condition 2 (Lateral) |
| **G** | 1.0000 | 0.9688 | 0.9841 | 32 | Condition 2 (Lateral) |
| **H** | 0.6667 | 0.4375 | 0.5283 | 32 | Condition 2 (Lateral) — Weak class |
| **I** | 0.9355 | 0.9062 | 0.9206 | 32 | Condition 2 (Lateral) |
| **J** | 1.0000 | 1.0000 | 1.0000 | 32 | Condition 2 (Lateral) |
| **K** | 1.0000 | 1.0000 | 1.0000 | 32 | Condition 2 (Lateral) |
| **L** | 1.0000 | 1.0000 | 1.0000 | 32 | Condition 2 (Lateral) |
| **M** | 1.0000 | 1.0000 | 1.0000 | 32 | Condition 2 (Lateral) |
| **N** | 1.0000 | 1.0000 | 1.0000 | 30 | Condition 1 (Frontal) |
| **O** | 1.0000 | 1.0000 | 1.0000 | 30 | Condition 1 (Frontal) |
| **P** | 1.0000 | 1.0000 | 1.0000 | 30 | Condition 1 (Frontal) |
| **Q** | 1.0000 | 0.9677 | 0.9836 | 31 | Condition 1 (Frontal) |
| **R** | 1.0000 | 0.8667 | 0.9286 | 30 | Condition 1 (Frontal) |
| **S** | 1.0000 | 1.0000 | 1.0000 | 32 | Condition 1 (Frontal) |
| **T** | 1.0000 | 1.0000 | 1.0000 | 30 | Condition 1 (Frontal) |
| **U** | 0.9583 | 0.7667 | 0.8519 | 30 | Condition 1 (Frontal) |
| **V** | 1.0000 | 1.0000 | 1.0000 | 32 | Condition 1 (Frontal) |
| **W** | 0.6774 | 0.8077 | 0.7368 | 26 | Condition 1 (Frontal) |
| **X** | 0.8205 | 1.0000 | 0.9014 | 32 | Condition 1 (Frontal) |
| **Y** | 0.9677 | 1.0000 | 0.9836 | 30 | Condition 1 (Frontal) |
| **Z** | 0.9688 | 1.0000 | 0.9841 | 31 | Condition 1 (Frontal) |

---

## 4. Audit of Proposed Later Experiments and Roadmap for Phase 35C

Based on the empirical findings of Phase 35B, we audit the experiments proposed in Phase 35A:

### Prioritization Evidence
1. **EXP-02 (Semantic-Preserving Geometric Augmentation) is Strongly Supported:**
   * *Evidence:* The acute collapse of `H` under lateral perspective ($C_1 \to C_2$) and `C` under lateral-to-frontal transfer ($C_2 \to C_1$) indicates that decision trees are overfitted to narrow camera-frame landmark ranges. Conservative in-plane tilt ($\pm 6^\circ$), scale jitter ($[0.95, 1.05]$), and coordinate jitter ($\sigma \le 0.004$) directly address this viewpoint rigidity without modifying sign semantics.
2. **EXP-04 (Controlled Feature Ablation of Raw Coordinates) is Strongly Supported:**
   * *Evidence:* 126 of 280 features are raw camera-frame $(x, y, z)$ coordinates. Comparing the full 280-dim representation against rotation/translation-invariant angles and distances (154-dim) will test whether removing camera-dependent coordinates eliminates perspective sensitivity.
3. **EXP-05 (Confidence Calibration & Selective Rejection Protocol):**
   * *Safe Execution Constraint:* When calibrated in subsequent phases, the isotonic probability calibrator **must be fitted strictly on out-of-fold training predictions** (using 5-fold CV across the 3,887 training samples). Rejection thresholds $\tau$ must be selected using development validation only.
4. **EXP-03 (Class Balancing) is Deprioritized:**
   * *Evidence:* Training support across letters in `signer_01` + `signer_04_train` is relatively balanced ($118\text{--}155$ samples per class). Class imbalance is not the root cause of the `C` or `H` errors.

---

## 5. Data Governance and Authoritative Cryptographic Hashes

### Authoritative Hash Ledger

All authoritative files, historical manifests, and newly created Phase 35B artifacts were verified:

| Artifact Description | Path | Verified SHA-256 Hash | Status |
| :--- | :--- | :--- | :--- |
| **Production Model C** | `phase22_analysis/models/Model_C_Occlusion_Aware_RF_multisigner.joblib` | `1fab3349e2e537d8b31ddbe3bdd5c21e3ed005f85aaac9d3875c72217e28cc98` | **UNCHANGED** |
| **Production Scaler** | `phase22_analysis/models/Model_C_Occlusion_Aware_RF_multisigner_scaler.joblib` | `93842932e7cfeb1d283e99f67a733fd487161d83cbff356d7458ced2c3de385d` | **UNCHANGED** |
| **Phase 19 Benchmark** | `phase19_frozen_manifest.json` | `9824eb510cc231249e9dd0722761de9112a02e7cecb681a479c4ef805b0fa797` | **UNCHANGED** |
| **Phase 23 Predictions** | `phase23_predictions.csv` | `da77036032fb3250141ccc16ba53d1dd7d4997b9817c6f525bbf9e000be45bfe` | **UNCHANGED** |
| **Signer 05 Manifest** | `data/collection_manifests/signer_05_manifest.json` | `5886b90d9b38e905848731ac8afeb82567a9ea343a460832bf49a74e3e4a5f3c` | **UNCHANGED** |
| **MediaPipe Landmarker**| `App/include50_model/hand_landmarker.task` | `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1` | **UNCHANGED** |
| **Phase 35B Baseline Model** | `phase35b_analysis/models/Model_C_Baseline_Fold1_Primary.joblib` | `fcaf6361046a17d94c506727b7e41db7f96e21a639105c18f5e35978f7307cef` | **NEW EXPERIMENTAL** |
| **Phase 35B Baseline Scaler**| `phase35b_analysis/models/Model_C_Baseline_Fold1_Primary_scaler.joblib` | `084b036e38816851d5fe1f7fcee6de853c37f60ba33e682be595c56654418c34` | **NEW EXPERIMENTAL** |
| **Phase 35B Fold 2 Model**   | `phase35b_analysis/models/Model_C_Baseline_Fold2_Inverted.joblib` | `ed4661be8bb9551f19ddbf8e064c2e7a776e632f527ea3731e915764de6d1a69` | **NEW EXPERIMENTAL** |
| **Phase 35B Fold 2 Scaler**  | `phase35b_analysis/models/Model_C_Baseline_Fold2_Inverted_scaler.joblib` | `bfd145c280abf9326df81ddb81cad316f3ef7833b9122a33c10997e792ef790a` | **NEW EXPERIMENTAL** |
| **Phase 35B Pure CC Model**  | `phase35b_analysis/models/Model_C_Baseline_Pure_CrossCondition.joblib` | `4c1573f26829be61c4fa6affff0f3f31c0ea8c4ee3b81cb3ef340e9c1d3decbf` | **NEW EXPERIMENTAL** |
| **Phase 35B Pure CC Scaler** | `phase35b_analysis/models/Model_C_Baseline_Pure_CrossCondition_scaler.joblib` | `5a451822a2baddd9599322e1522ed5d42782c9d036a8be1875a3cb3240191fef` | **NEW EXPERIMENTAL** |

---

## 6. Runtime and Production Safeguards

All production runtime configuration defaults remain verified intact:
* `ISL_BACKEND_DETECTOR=yolo`
* `ISL_RECOGNITION_ARCH=standard`
* `ISL_TEMPORAL_SMOOTHING=0`
* `ISL_SHADOW_MODE=0`
* `ISL_MEDIAPIPE_DELEGATE=cpu`
* `ISL_MODEL_C_RESOLUTION=native`

---

## 7. Tests and Verification Results

1. **Dedicated Phase 35B Test Suite:**
   * **Command:** `/Users/pavanaksshay/venv/bin/python3 -m unittest tests/test_phase35b_baseline.py`
   * **Result:** **10 of 10 tests passed (100%) in 0.150s**.
   * **Scope:** Verifies artifact existence, bit-for-bit production immutability, zero-overlap recording disjointness, confound audit documentation, training-only scaler fitting, milestone gate passage ($\ge 75\%$), and per-letter/recording metrics integrity.
2. **Dedicated Phase 35A Governance Suite:**
   * **Command:** `/Users/pavanaksshay/venv/bin/python3 -m unittest tests/test_phase35a_generalization_strategy.py`
   * **Result:** **10 of 10 tests passed (100%) in 0.089s**.
3. **Repository Regression Discovery:**
   * **Command:** `/Users/pavanaksshay/venv/bin/python3 -m unittest discover tests -p "test_*.py"`
   * **Result:** 305 tests passing across all functional components.

---

## Execution Boundary Conclusion

Phase 35B reproduction and split audit are complete. The experimental baseline is frozen in [`phase35b_analysis/models/`](file:///Users/pavanaksshay/pavan_sign_language/phase35b_analysis/models/).

**Per prompt instructions, execution halts here. Augmentation, feature ablation, and model improvement experiments are NOT started automatically.**

