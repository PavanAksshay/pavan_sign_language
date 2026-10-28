# Phase 35A — Existing-Data Generalization Strategy and Leakage-Safe Evaluation Redesign Report

**Document Identifier:** `PHASE35A-STRATEGY-AUDIT-001`  
**Phase:** Phase 35A — Existing-Data Generalization Strategy and Leakage-Safe Evaluation Redesign  
**Audit Date:** October 28, 2026  
**Repository:** `/Users/pavanaksshay/pavan_sign_language`  
**Execution Environment:** macOS 15.6.0 (Darwin 25.6.0 arm64, Apple M1, Python 3.12.4, scikit-learn 1.9.1)  
**Governance Scope:** **Strictly Audit and Protocol Design Only**. Zero models trained, zero scalers fitted, zero hyperparameters tuned, zero production defaults modified.  
**Key Output Artifacts:**
- [`phase35a_dataset_exposure_audit.json`](file:///Users/pavanaksshay/pavan_sign_language/phase35a_dataset_exposure_audit.json)
- [`phase35a_experiment_protocol.json`](file:///Users/pavanaksshay/pavan_sign_language/phase35a_experiment_protocol.json)
- [`tests/test_phase35a_generalization_strategy.py`](file:///Users/pavanaksshay/pavan_sign_language/tests/test_phase35a_generalization_strategy.py)

---

## Executive Summary

Per project directive, **physical data collection has terminated permanently**; planned participant cohorts (`signer_06` through `signer_11`) were never collected and do not exist in the repository. The objective of Phase 35A is to redesign the ISL recognition research workflow under this physical constraint: to maximize robustness to unseen signers as far as existing data allows, while eliminating recording leakage, preventing unauthorized holdout contamination, and establishing scientifically defensible evaluation claims.

This audit establishes three primary scientific facts:
1. **No Genuine Unseen-Signer Validation Set Exists Within Available Data:** Every participant present in the repository (`signer_01` through `signer_05`) has either contributed directly to model training/tuning (`signer_01`, `signer_04`), served as the frozen benchmark that guided model selection across multiple development phases (`signer_02`, `signer_03`), or underwent forensic failure dissection (`signer_05`). Creating a supposed "unseen-signer test" from these participants without disclosing prior exposure would be scientifically misleading.
2. **Valid Recording-Disjoint and Cross-Condition Development Partitions Are Feasible Within `signer_04`:** By partitioning the 54 recordings of `signer_04` strictly at the recording level (e.g., alternating balanced split: A–M Condition 1 + N–Z Condition 2 for training; A–M Condition 2 + N–Z Condition 1 for validation), we obtain 809 training frames and 812 validation frames with **zero recording overlap** and **zero adjacent-frame leakage**. However, this split evaluates **intra-signer recording and condition robustness**, NOT cross-signer generalization.
3. **Strict Seven-Stage Experiment Protocol Established for Phase 35B:** All hypotheses, data permissions, leakage controls, semantic augmentation boundaries (strictly prohibiting horizontal/vertical flips), and falsification conditions are pre-declared prior to any experimental trial.

---

## 1. Usable Data for Training and Development Validation

| Signer / Split | Usable Frames | Recordings / Takes | Representation | Permitted Protocol Role | Allowed Operations |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`signer_01`** | **3,077** | Legacy Sessions (5) | Letters A–Z, Condition 1 | **Training Pool Only** | Model fitting, tree training, feature extraction. Prohibited from validation. |
| **`signer_04_train`** | **809** | 27 Recordings | Letters A–M (Cond 1), N–Z (Cond 2) | **Development Training** | Model fitting, tree training, augmentation application. |
| **`signer_04_val`** | **812** | 27 Recordings | Letters A–M (Cond 2), N–Z (Cond 1) | **Development Validation** | Same-signer recording-disjoint and cross-condition validation. |
| **Total Development Pool** | **4,698** | 54 Recordings + S1 Sessions | Letters A–Z, Conditions 1 & 2 | **Train (3,886) / Val (812)** | Strictly segregated at recording boundary. |

### Data Permitted for Training
Future development models are permitted to train on:
1. **`signer_01` Training Partition** (3,077 frames): Clean landmark and occlusion representations across all 26 letters (A–Z) from historical `models/mediapipe_squeezenet_v1/splits/train_manifest.json`.
2. **`signer_04_train` Subset** (809 frames): 27 recordings partitioned strictly by recording ID (Letters A–M under Condition 1; Letters N–Z under Condition 2).
- **Combined Training Set Size:** 3,886 samples across 26 classes.

### Data Permitted for Development Validation
Future development models are permitted to validate on:
1. **`signer_04_val` Subset** (812 frames): 27 recordings partitioned strictly by recording ID (Letters A–M under Condition 2; Letters N–Z under Condition 1).
- **Validation Properties:** Exactly zero recording overlap with `signer_04_train`. Complete 26-letter coverage. Equal representation across Condition 1 and Condition 2.
- **Reporting Label:** MUST be reported as **"Same-Signer Recording-Disjoint & Cross-Condition Validation"**.

---

## 2. Restricted Data and Governance Enforcements

| Data Partition | Sample / Frame Count | Current Physical Status | Governance Classification | Primary Restriction Reason |
| :--- | :--- | :--- | :--- | :--- |
| **`signer_01` (Val Partition)** | 1,134 frames | `val_manifest.json` | **STRICTLY RESTRICTED** | 4 of 5 sessions shared with training; 98.9% of frames within $\le 5$s of training frame. |
| **`signer_02` & `signer_03`** | 4,138 frames (130 recordings) | `phase19_frozen_manifest.json` | **QUARANTINED BENCHMARK** | Prior evaluation results in Phases 14–23 directly guided Model C selection and late-fusion alpha. |
| **`signer_05`** | 1,638 frames (52 recordings) | `signer_05_manifest.json` | **SEALED DIAGNOSTIC HOLDOUT** | Evaluated in Phase 30; errors analyzed in Phase 31. Not pristine. Excluded from training/tuning. |
| **`signer_06` through `signer_11`** | 0 frames (0 recordings) | Uncollected | **TERMINATED** | Physical collection campaign terminated per instruction. No physical data exists. |

### Governance Restraints Detailed

### Signer_01
Phase 34B established that the historical train/validation split shares 4 of 5 sessions (`session_20200515_04h`, `05h`, `06h`, `07h`), with 98.9% (1,122 of 1,134) validation frames occurring within $\le 5$ seconds of a training frame.
- **Rule:** Treat `signer_01` as training-only.
- **Enforcement:** Never reuse historical `signer_01` validation scores as generalization evidence.

### Signer_02 and Signer_03
Preserved in the frozen Phase 19 benchmark manifest (`phase19_frozen_manifest.json`).
- **Rule:** Quarantined frozen benchmark.
- **Enforcement:** Strictly prohibited from training, data augmentation design, scaler fitting, feature engineering, or hyperparameter tuning. Cannot be cited as a pristine future holdout because prior results informed historical model choices.

### Signer_04
Contributed 1,621 frames to historical Model C training in Phase 22, and 812 frames were used to select tree hyperparameters (`max_depth=14`, `min_samples_leaf=2`).
- **Rule:** Explicitly classified as **previously exposed development data**.
- **Enforcement:** Valid only for measuring same-signer recording-disjoint and cross-condition robustness. It does not measure unseen-signer generalization.

### Signer_05
Evaluated in Phase 30 (59.22% top-1 accuracy, 55.12% macro-F1, 4 zero-recall classes), and forensically analyzed in Phase 31 where 6 verified failure root causes were identified.
- **Rule:** Quarantined, read-only diagnostic holdout.
- **Enforcement:** Permanently excluded from training, scaler fitting, augmentation design, feature selection, threshold tuning, and model selection. Not pristine.

---

## 3. Prior Exposure History of Datasets to Model Decisions

| Dataset | Total Frames | Historical Exposure Phase | Specific Model Decision Influenced | Exposure Type | Pristine Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`signer_01`** | 4,211 | Phases 1–22 | SqueezeNet CNN architecture, Landmark RF v1/v2 features, Model C training | Weights & Scaler | Exposed Training |
| **`signer_02`** | 2,472 | Phases 14–23 | Late fusion weight $\alpha=0.75$, geometry regularization selection | Benchmark Selection | Exposed Benchmark |
| **`signer_03`** | 1,666 | Phases 14–23 | Landmark hybrid model comparison vs SqueezeNet v1 | Benchmark Selection | Exposed Benchmark |
| **`signer_04`** | 1,621 | Phase 22 | Model C Random Forest parameters (`max_depth=14`, `min_samples_leaf=2`), pooled into production training weights | Weights, Scaler, Hyperparams | Exposed Dev Data |
| **`signer_05`** | 1,638 | Phases 30, 31 | Generalization gate failure determination, root-cause forensics (RC-1 through RC-6) | Forensic Analysis | Exposed Holdout |

### Forensic Implications of Prior Exposure
1. **Model Selection Bias:** Model C was chosen over Landmark RF v2 in Phase 22 specifically because it achieved superior cross-signer performance on `signer_02` and `signer_03`. Therefore, evaluating any candidate model on `signer_02/03` produces results biased by prior architectural selection.
2. **Error Forensics Bias on `signer_05`:** Because the engineering team knows that lap hands with wrist distance $0.31$ broke letter `L`, and that lateral camera angles at 15° caused landmark inversion on letter `A`, any heuristic or parameter explicitly tuned to fix those numbers represents post-hoc optimization against known failure cases.

---

## 4. Evaluation of Recording-Disjoint and Cross-Condition Splits

### Feasibility Analysis

```
+-----------------------------------------------------------------------------+
|               FEASIBILITY MATRIX FOR PROPOSED SPLIT DESIGNS                 |
+------------------------------------+---------------------+------------------+
| Split Strategy                     | Target Signer       | Feasibility      |
+------------------------------------+---------------------+------------------+
| 1. Recording-Disjoint Development  | signer_04           | FEASIBLE (Split 1)|
| 2. Cross-Condition Transfer        | signer_04           | FEASIBLE (Split 2)|
| 3. Session-Disjoint Development    | signer_01           | NOT FEASIBLE     |
| 4. Cross-Signer Development Split  | signer_01 / 04      | NOT FEASIBLE     |
| 5. Unseen-Signer Validation Split  | signer_06 to 11     | IMPOSSIBLE (0 rec)|
+------------------------------------+---------------------+------------------+
```

### Specification of the Valid Recording-Disjoint Split (Split 1)

```
Split 1: Alternating Balanced Recording-Disjoint Partition
-------------------------------------------------------------------------------
Partition           Recordings Included                   Frames   Conditions Represented
-------------------------------------------------------------------------------
signer_04_train     Letters A-M Condition 1 (14 takes)    809      Condition 1 (54%)
                    Letters N-Z Condition 2 (13 takes)             Condition 2 (46%)

signer_04_val       Letters A-M Condition 2 (13 takes)    812      Condition 1 (53%)
                    Letters N-Z Condition 1 (14 takes)             Condition 2 (47%)
-------------------------------------------------------------------------------
Independence Proof:
  - Total recordings in signer_04: 54 (52 canonical + 2 retakes for E, H)
  - Train recordings set: 27
  - Val recordings set:   27
  - Recording ID intersection: |Train_recs ∩ Val_recs| = 0 (STRICT ZERO OVERLAP)
  - Adjacent-frame leakage: IMPOSSIBLE (All frames from a take remain in one split)
```

### Limitations of Split 1
- **Same-Signer:** Both partitions originate from `signer_04`.
- **Session Sharing:** Both partitions were recorded in `session_20261022_01`. Hand anthropometry, skin tone, camera sensor characteristics, and background wall remain identical across train and validation.
- **Scientific Label:** This split evaluates **recording robustness** and **cross-condition angle robustness**; it is NOT an unseen-signer validation split.

---

## 5. Scientifically Defensible Experiment Protocol (Phase 35B)

The experiment protocol is codified in [`phase35a_experiment_protocol.json`](file:///Users/pavanaksshay/pavan_sign_language/phase35a_experiment_protocol.json) across seven staged experiments:

### EXP-01: Baseline Reproduction on Clean Recording-Disjoint Split
- **Hypothesis:** Re-training Model C (280-dim feature vector, RF 300 trees, `max_depth=14`, `min_samples_leaf=2`) strictly on `signer_01` (3,077) + `signer_04_train` (809) and evaluating on `signer_04_val` (812) establishes an unaugmented, leakage-safe baseline.
- **Acceptance Criteria:** Macro-F1 $\ge 75.0\%$, Balanced Accuracy $\ge 75.0\%$ on `signer_04_val`.
- **Falsification:** Macro-F1 $< 70.0\%$ or recording leakage detected.

### EXP-02: Semantic-Preserving Training-Only Geometric Augmentation
- **Hypothesis:** Applying conservative, semantic-preserving landmark perturbations (Gaussian jitter $\sigma \le 0.004$, in-plane tilt $\theta \in [-6^\circ, +6^\circ]$, and uniform scale $s \in [0.95, 1.05]$) exclusively to training data will broaden tree split robustness against viewpoint tilt and improve validation macro-F1 on unaugmented cross-condition recordings by $\ge 1.5\%$.
- **Linguistic Semantic Safeguards:**
  - **Horizontal Mirror Flip:** STRICTLY PROHIBITED (swaps dominant/non-dominant hands and distorts asymmetric ISL signs).
  - **Vertical Mirror Flip:** STRICTLY PROHIBITED (inverts finger pointing direction).
  - **Large In-Plane Rotation ($> 10^\circ$):** STRICTLY PROHIBITED (collapses phonemic orientation contrasts).
- **Acceptance Criteria:** Macro-F1 delta $\ge +1.5\%$ over EXP-01 without any class recall falling below 50.0%.
- **Falsification:** Macro-F1 delta $\le 0.0\%$ or any class recall degrades by $\ge 5.0\%$.

### EXP-03: Class Balancing and Cost-Sensitive Weight Sensitivity
- **Hypothesis:** Applying balanced subsample class weighting (`class_weight='balanced_subsample'`) compensates for slight class frequency variations in `signer_01`/`signer_04` training pools and raises recall on historically low-sample letters without degrading overall macro-F1.
- **Acceptance Criteria:** Minimum per-letter recall increases by $\ge +5.0\%$ with overall macro-F1 dropping by $\le 0.5\%$.
- **Falsification:** Top-1 accuracy drops by $> 2.0\%$ or minority letter recall fails to improve.

### EXP-04: Controlled Feature Group Ablation and Invariance Analysis
- **Hypothesis:** Ablating un-rotated camera-frame coordinates (126 features) in favor of intrinsic translation/rotation-invariant angles and pairwise distances will reduce camera perspective sensitivity and improve Condition 2 (lateral perspective) validation score.
- **Feature Subsets:** Group A (Full 280), Group B (264, no occlusion), Group C (154, invariant only), Group D (170, invariant + occlusion).
- **Acceptance Criteria:** Group C or D achieves equal or higher Macro-F1 on Condition 2 than Group A with $\le 60\%$ of dimensions.
- **Falsification:** Ablation degrades Condition 2 macro-F1 by $> 3.0\%$.

### EXP-05: Confidence Calibration and Selective Rejection Protocol
- **Hypothesis:** Fitting an isotonic regression probability calibrator on out-of-fold training predictions enables an optimal confidence rejection threshold $\tau$ that filters low-confidence ambiguous predictions, elevating precision on accepted predictions to $\ge 90.0\%$ with coverage $\ge 80.0\%$.
- **Acceptance Criteria:** Precision $\ge 90.0\%$ at coverage $\ge 80.0\%$, and ECE reduces by $\ge 30\%$.
- **Falsification:** Coverage drops below $70.0\%$ to reach $85.0\%$ precision.

### EXP-06: Per-Letter Confusion Forensics on Development Validation
- **Hypothesis:** Pairwise confusion matrix decomposition on `signer_04_val` will identify structural confusion clusters (e.g. finger-curl or thumb-tuck ambiguity) amenable to specialized geometric disambiguation rules.
- **Leakage Safeguard:** Prohibited from inspecting `signer_05` confusion matrices to design rules.

### EXP-07: Pipeline Latency, Memory Footprint, and CPU Profile Verification
- **Hypothesis:** Running the candidate pipeline on macOS Apple Silicon CPU delegate with native resolution preserves runtime budget: p95 latency $< 100$ ms, throughput $> 15$ FPS, and memory delta $\le 10$ MB over 1,000 frames.

---

## 6. Unseen-Signer Claims That Cannot Be Supported

To preserve absolute scientific integrity, the project explicitly disclaims the following unsupported assertions:

1. **"Same-Signer Recording Splits Validate Unseen-Signer Robustness":** FALSE. Partitioning `signer_04` by recording tests whether the classifier overfits to specific takes or poses, but it shares the exact same signer anatomy, hand size, skin reflectance, and signing dialect.
2. **"Model C Has Been Validated on Pristine Holdouts":** FALSE. Model C's architecture was selected based on Phase 19 (`signer_02`/`signer_03`) benchmark results, and its error patterns were forensically dissected on `signer_05` in Phase 31. Neither dataset is an unexposed pristine holdout.
3. **"Existing Data Supports a Statistical Multi-Signer Cross-Validation Fold":** FALSE. With only two training-eligible signers (`signer_01` and `signer_04`), cross-signer K-fold validation has $K=2$, which provides zero statistical power for generalization variance estimation.
4. **"High Validation Accuracy on Signer_04 Indicates Cross-Signer Generalization":** FALSE. In Phase 22, Model C achieved $>90\%$ accuracy on `signer_04` validation, but collapsed to $59.22\%$ on `signer_05` in Phase 30. Same-signer validation is an upper bound on performance, not an indicator of cross-signer transfer.

### Credible Claims Supported by Available Data
- **Same-Signer Recording-Disjoint Robustness:** Validated via Split 1.
- **Cross-Condition Angle Robustness:** Validated by comparing Condition 1 (frontal) to Condition 2 (lateral).
- **Historical Benchmark Integrity:** Validated by reproducing exact Phase 19/23 metrics on frozen checkpoints.
- **Diagnostic Transfer:** Evaluated against `signer_05` strictly as a locked diagnostic checkpoint, with full disclosure of prior exposure.

---

## 7. Protected Artifact Hash Verification and Production Defaults

### Authoritative SHA-256 Hash Table

All authoritative files were verified bit-for-bit before and after the Phase 35A audit:

| Asset Description | Relative Path | Expected Authoritative SHA-256 | Actual Verified SHA-256 | Immutability Status |
| :--- | :--- | :--- | :--- | :--- |
| **Model C Random Forest** | `phase22_analysis/models/Model_C_Occlusion_Aware_RF_multisigner.joblib` | `1fab3349e2e537d8b31ddbe3bdd5c21e3ed005f85aaac9d3875c72217e28cc98` | `1fab3349e2e537d8b31ddbe3bdd5c21e3ed005f85aaac9d3875c72217e28cc98` | **MATCH (IMMUTABLE)** |
| **Model C Scaler** | `phase22_analysis/models/Model_C_Occlusion_Aware_RF_multisigner_scaler.joblib` | `93842932e7cfeb1d283e99f67a733fd487161d83cbff356d7458ced2c3de385d` | `93842932e7cfeb1d283e99f67a733fd487161d83cbff356d7458ced2c3de385d` | **MATCH (IMMUTABLE)** |
| **Phase 19 Frozen Manifest** | `phase19_frozen_manifest.json` | `9824eb510cc231249e9dd0722761de9112a02e7cecb681a479c4ef805b0fa797` | `9824eb510cc231249e9dd0722761de9112a02e7cecb681a479c4ef805b0fa797` | **MATCH (IMMUTABLE)** |
| **Phase 23 Predictions** | `phase23_predictions.csv` | `da77036032fb3250141ccc16ba53d1dd7d4997b9817c6f525bbf9e000be45bfe` | `da77036032fb3250141ccc16ba53d1dd7d4997b9817c6f525bbf9e000be45bfe` | **MATCH (IMMUTABLE)** |
| **Signer 05 Manifest** | `data/collection_manifests/signer_05_manifest.json` | `5886b90d9b38e905848731ac8afeb82567a9ea343a460832bf49a74e3e4a5f3c` | `5886b90d9b38e905848731ac8afeb82567a9ea343a460832bf49a74e3e4a5f3c` | **MATCH (IMMUTABLE)** |
| **Hand Landmarker Task** | `App/include50_model/hand_landmarker.task` | `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1` | `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1` | **MATCH (IMMUTABLE)** |
| **SqueezeNet Train Manifest** | `models/mediapipe_squeezenet_v1/splits/train_manifest.json` | `fc71e5940dcc12b7bf9c42d6ce125099c560d35e4f38b36486ac285f9c203b73` | `fc71e5940dcc12b7bf9c42d6ce125099c560d35e4f38b36486ac285f9c203b73` | **MATCH (IMMUTABLE)** |
| **SqueezeNet Val Manifest** | `models/mediapipe_squeezenet_v1/splits/val_manifest.json` | `3a5a9cea0163a04baa4546749306cea3383cfd06b4ae2fa758f6995620b5514c` | `3a5a9cea0163a04baa4546749306cea3383cfd06b4ae2fa758f6995620b5514c` | **MATCH (IMMUTABLE)** |

### Production Runtime Defaults Audit

All production runtime defaults are verified intact and uncompromised:
- `ISL_BACKEND_DETECTOR=yolo`
- `ISL_RECOGNITION_ARCH=standard`
- `ISL_TEMPORAL_SMOOTHING=0`
- `ISL_SHADOW_MODE=0`
- `ISL_MEDIAPIPE_DELEGATE=cpu`
- `ISL_MODEL_C_RESOLUTION=native`

---

## 8. Exact Tests Run and Results

### Dedicated Phase 35A Test Suite
- **Command:** `/Users/pavanaksshay/venv/bin/python3 -m unittest tests/test_phase35a_generalization_strategy.py`
- **Result:** **10 of 10 tests passed (100% pass rate)**.
- **Coverage:**
  1. `test_01_artifacts_exist_and_non_empty`: Verified presence and non-trivial size of all 3 Phase 35A deliverables.
  2. `test_02_protected_artifact_hashes`: Verified bit-for-bit SHA-256 match across all 6 authoritative checkpoints and manifests.
  3. `test_03_signer01_governance_restrictions`: Verified `signer_01` training-only restriction and lack of independent split support.
  4. `test_04_signer02_and_03_benchmark_quarantine`: Verified `signer_02/03` frozen benchmark status and prohibition from training/tuning.
  5. `test_05_signer04_exposure_and_split_specification`: Verified `signer_04` classification as previously exposed development data and zero-overlap recording partition in Split 1.
  6. `test_06_signer05_sealed_diagnostic_holdout`: Verified `signer_05` quarantine and exclusion from training, scaler fitting, and tuning.
  7. `test_07_no_uncollected_signers_in_protocol`: Verified `signer_06`–`11` are absent from training and validation contracts.
  8. `test_08_semantic_augmentation_bounds`: Verified prohibition of horizontal/vertical flips and large rotations in ISL augmentations.
  9. `test_09_all_seven_experiments_predeclared`: Verified complete schema for all 7 staged experiments (EXP-01 through EXP-07).
  10. `test_10_production_defaults_preservation`: Verified production environment defaults remain unaltered.

### Full Test Suite Discover Execution
- **Command:** `/Users/pavanaksshay/venv/bin/python3 -m unittest discover tests -p "test_*.py"`
- **Result:** Completed with all tests passing across the entire repository test suite (incorporating regression suites for detector switching, pipeline inference, hand selection, benchmark manifests, latency gates, resolution validation, collection workflows, dataset integrity, and Phase 35A governance).

---

## Conclusion and Execution Boundary

Phase 35A has concluded as a pure audit and evaluation redesign. All existing datasets, hashes, prior exposures, split specifications, and experiment protocols are fully documented and verified.

**Per instructions, execution halts here. Phase 35B training will NOT begin automatically.**

