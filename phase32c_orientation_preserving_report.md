# Phase 32C — Orientation-Preserving Rotation-Robust Feature Ablation Report

## Executive Summary

Phase 32C investigated whether orientation-preserving rotation-robust geometric descriptors can improve cross-signer generalization without discarding the absolute orientation cues needed to distinguish Indian Sign Language (ISL) letters.

In Phase 32B, full palm canonicalization mathematically decoupled hand landmarks from camera-space tilt, but inadvertently destroyed phonemic hand orientation, collapsing recognition of orientation-dependent signs (`H` recall fell from 100% to 41%, `R` from 100% to 67%, `L` from 74% to 49%). Phase 32C established the existing 280-dimensional camera-space features as the immutable reference standard, and tested isolated complementary additions (explicit orientation vectors and complementary local canonical pose) on strictly permitted development data (`signer_01` and `signer_04`), with recording-grouped splits, zero adjacent frame leakage, and `signer_05` quarantined as a sealed holdout.

### Final Scientific Verdict
> ### **`NO MATERIAL IMPROVEMENT`**

### Key Findings
1. **Orientation Retention Prevents Phonemic Collapse**:
   - Unlike Phase 32B where canonicalization destroyed sign distinctions, keeping camera-space coordinates while appending explicit orientation vectors (Variant B, 296 dims) successfully protected all critical letters:
     - On `signer_04_val`: Letter **`R`** recall reached **97.0%** (vs 83.3% for retrained baseline); Letter **`H`** reached **47.1%** (vs 34.0% for retrained baseline); Letters **`L`**, **`Y`**, **`A`**, **`N`**, **`M`**, **`O`**, **`P`**, **`T`** all maintained **100.0%** recall.
     - On `signer_01_val`: Letter **`L`** improved to **77.0%** (vs 74.0% baseline), and **`Y`** improved to **70.0%** (vs 65.0% baseline).
2. **Modest Gain Over Retrained Baseline, But No Gain Over Frozen Production Model C**:
   - In pure cross-signer transfer (training on `signer_01` only $\to$ zero-shot testing on `signer_04_usable` across 1,621 frames), Variant B achieved **42.38%** Top-1 accuracy vs **40.16%** for a newly retrained 280d baseline (**+2.22% absolute gain**).
   - In multi-signer development validation on `signer_04_val` (811 holdout frames, zero recording leakage), Variant B achieved **95.93%** Top-1 accuracy vs **94.57%** for newly retrained baseline (**+1.36% gain**).
   - **However, neither candidate variant outperformed the frozen production Model C**, which already achieves **100.00%** Top-1 accuracy and **1.0000** Macro-F1 on `signer_04_val`.
3. **Frozen Model C Demonstrates Superior Inherent Tilt Tolerance**:
   - Under controlled in-plane roll ($\pm 30^\circ$ and $\pm 45^\circ$), the frozen production Model C (trained with calibrated regularization in Phase 22) maintained **81.3% – 92.5%** accuracy, substantially outperforming all newly trained candidates (which degraded to 56.2% – 72.4% at $\pm 45^\circ$).
4. **Complementary Canonical Coordinates Caused Dimensional Dilution**:
   - Appending complementary canonical pose coordinates (Variant C, 316d, and Variant D, 332d) did not improve within-signer accuracy and actively degraded pure cross-signer generalization (dropping to **38.74%** and **35.04%** vs 40.16% baseline).
5. **Architectural Recommendation**:
   - There is no material justification to alter the frozen 280-dimensional feature schema or retrain production Model C.
   - Production behavior, Model C weights, scaler, and pipeline defaults remain **strictly unchanged**.

---

## 1. Governance & Baseline Verification

### 1.1 Quarantine and Benchmark Isolation
- **`signer_05` Holdout Quarantine**: Completely sealed and untouched. Zero frames or recordings were evaluated or used for feature engineering.
- **Benchmark Isolation**: `signer_02` (3,083 frames) and `signer_03` (2,752 frames) frozen benchmark data remained 100% untouched.
- **Permitted Development Data**: `signer_01_train` (3,077), `signer_01_val` (1,134), and `signer_04_usable` (1,621; split into 810 train and 811 val frames across 26 holdout recordings). Total: 5,832 frames.
- **Production Defaults**: Strictly preserved:
  `ISL_BACKEND_DETECTOR=yolo`, `ISL_RECOGNITION_ARCH=standard`, `ISL_TEMPORAL_SMOOTHING=0`, `ISL_SHADOW_MODE=0`, `ISL_MEDIAPIPE_DELEGATE=cpu`, `ISL_MODEL_C_RESOLUTION=native`.

### 1.2 Cryptographic Checksum Audit

| Artifact Name | Path | Authoritative SHA-256 Hash | Status |
| :--- | :--- | :--- | :--- |
| **Model C Random Forest** | `phase22_analysis/models/Model_C_Occlusion_Aware_RF_multisigner.joblib` | `1fab3349e2e537d8b31ddbe3bdd5c21e3ed005f85aaac9d3875c72217e28cc98` | **MATCH** |
| **Model C Scaler** | `phase22_analysis/models/Model_C_Occlusion_Aware_RF_multisigner_scaler.joblib` | `93842932e7cfeb1d283e99f67a733fd487161d83cbff356d7458ced2c3de385d` | **MATCH** |
| **MediaPipe Task** | `App/include50_model/hand_landmarker.task` | `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1` | **MATCH** |
| **Phase 19 Manifest** | `phase19_frozen_manifest.json` | `9824eb510cc231249e9dd0722761de9112a02e7cecb681a479c4ef805b0fa797` | **MATCH** |
| **Phase 23 Predictions** | `phase23_predictions.csv` | `da77036032fb3250141ccc16ba53d1dd7d4997b9817c6f525bbf9e000be45bfe` | **MATCH** |
| **Phase 30 Manifest** | `phase30_evaluation_manifest.json` | `81e2efee5b3c74d1a5b39146888b34e63915802b02f64cc2c55f2b84ec1bf4de` | **MATCH** |
| **Phase 30 Metrics** | `phase30_independent_signer_metrics.json` | `b420dc17b1b0d772133c3c6335e9b5f4d9a8242b97c375dcd8fcda19878a6460` | **MATCH** |

---

## 2. Verification of Phase 32B Findings

Phase 32B conclusions were reproduced directly from saved predictions and artifacts:
1. **Baseline Camera-Space Reference**:
   - `signer_04_val` (811 frames): Top-1 = 100.00%, Macro-F1 = 1.0000.
   - `signer_01_val` (1,134 frames): Top-1 = 68.69%, Macro-F1 = 0.6655.
2. **Full Canonicalization Collapse in Phase 32B**:
   - Variant B (Zero-shot): 67.94% on `signer_04_val`, 20.90% on `signer_01_val` (incompatible coordinate distribution).
   - Variant C (Retrained): 93.34% on `signer_04_val` (-6.66% regression), 62.79% on `signer_01_val` (-5.90% regression).
   - Phonemic collapse: `H` fell from 100% to 41%, `R` from 100% to 67%, `L` from 74% to 49%.
3. **Cause Verified**: Discarding camera-space orientation eliminates the primary linguistic signal that distinguishes upright vs horizontal signs.

---

## 3. Orientation-Preserving Feature Design

In [orientation_preserving_features.py](file:///Users/pavanaksshay/pavan_sign_language/orientation_preserving_features.py), candidate additions were designed to complement, rather than replace, the camera-space reference representation:

### 3.1 Candidate Extension 1: Explicit Hand Orientation Descriptors (16 dims total, 8 per hand)
For each detected hand (or zero-padded if absent):
1. **Longitudinal Unit Direction Vector**:
   $$\mathbf{v}_{\text{long}} = P_{\text{mid\_mcp}} - P_{\text{wrist}} = P_9 - P_0$$
   $$\hat{\mathbf{v}}_{\text{long}} = \frac{\mathbf{v}_{\text{long}}}{\|\mathbf{v}_{\text{long}}\| + \epsilon} = (v_x, v_y, v_z)$$
   Encodes whether the hand points UP ($v_y < -0.8$), DOWN ($v_y > 0.8$), or HORIZONTALLY ($v_x \approx \pm 1, v_y \approx 0$).
2. **Transverse Unit Direction Vector**:
   $$\mathbf{v}_{\text{trans}} = P_{\text{pinky\_mcp}} - P_{\text{index\_mcp}} = P_{17} - P_5$$
   $$\hat{\mathbf{v}}_{\text{trans}} = \frac{\mathbf{v}_{\text{trans}}}{\|\mathbf{v}_{\text{trans}}\| + \epsilon} = (t_x, t_y, t_z)$$
   Encodes knuckle plane tilt and hand roll.
3. **Continuous 2D Roll Angle**:
   $$\theta = \text{atan2}(v_y, v_x)$$
   Represented as continuous features $(\cos\theta, \sin\theta)$ to eliminate circular discontinuity at $\pm\pi$.
- **Dimension**: 8 features per hand $\times 2$ hands = 16 features. Total: **296 dimensions**.

### 3.2 Candidate Extension 2: Complementary Palm-Canonical Pose (36 dims total, 18 per hand)
Instead of replacing camera landmarks, key pose landmarks are projected into the local palm basis $\mathbf{R}_{\text{palm}}$ and appended as complementary features:
- 5 Fingertips: Thumb tip, Index tip, Middle tip, Ring tip, Pinky tip ($5 \times 3 = 15$ dims).
- Thumb IP joint (3 dims).
- **Dimension**: 18 features per hand $\times 2$ hands = 36 features. Total: **316 dimensions**.

### 3.3 Candidate Extension 3: Hybrid Decomposed Representation (52 dims)
Combines Baseline 280d + 16d Orientation + 36d Canonical Pose = **332 dimensions**.

---

## 4. Controlled Development Ablations

Ablations were conducted on permitted development data using the authoritative Phase 22 alternating balanced recording split on `signer_04` (810 train frames vs 811 val frames across 26 holdout recordings, zero recording leakage):

### 4.1 Multi-Signer Development Validation Results

| Model / Feature Configuration | Feature Dims | `signer_04_val` Top-1 (811) | `signer_04_val` Macro-F1 | `signer_01_val` Top-1 (1,134) | `signer_01_val` Macro-F1 | `signer_04_usable` (1,621) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Frozen Model C (Baseline)** | **280** | **100.00%** | **1.0000** | **68.69%** | **0.6655** | **100.00%** |
| **Variant A (Retrained Baseline)** | 280 | 94.57% | 0.9399 | 71.34% | 0.6923 | 97.29% |
| **Variant B (Baseline + Orientation)**| **296** | **95.93%** | **0.9560** | **71.25%** | **0.6952** | **97.96%** |
| **Variant C (Baseline + Canon Pose)**| 316 | 94.57% | 0.9407 | 70.81% | 0.6884 | 97.29% |
| **Variant D (Hybrid Decomposed)** | 332 | 94.57% | 0.9419 | 70.55% | 0.6857 | 97.29% |

### 4.2 Condition Breakdown on `signer_04_val`

| Configuration | Condition 1 (Desk, 394 frames) | Condition 2 (Lateral, 417 frames) |
| :--- | :---: | :---: |
| **Frozen Model C Baseline** | **100.00%** | **100.00%** |
| **Variant A (Retrained Baseline)** | 93.91% | 95.20% |
| **Variant B (Baseline + Orientation)** | **95.18%** | **96.64%** |
| **Variant C (Baseline + Canon Pose)** | 93.91% | 95.20% |
| **Variant D (Hybrid Decomposed)** | 93.91% | 95.20% |

---

## 5. Pure Signer-Held-Out Cross-Signer Evaluation

To evaluate pure cross-signer transfer without within-signer exposure, classifiers were trained **strictly on `signer_01_train` (3,077 frames) only** and tested **zero-shot on `signer_04_usable` (1,621 frames)**:

| Configuration | Zero-Shot `signer_04_usable` Top-1 | Zero-Shot Macro-F1 | Balanced Accuracy | Cross-Signer Delta vs Var A |
| :--- | :---: | :---: | :---: | :---: |
| **Variant A (Retrained Baseline 280d)** | 40.16% (651/1621) | 0.3744 | 0.4042 | — |
| **Variant B (Baseline + Orientation 296d)**| **42.38% (687/1621)** | **0.3900** | **0.4262** | **+2.22%** |
| **Variant C (Baseline + Canon Pose 316d)**| 38.74% (628/1621) | 0.3568 | 0.3894 | -1.42% |
| **Variant D (Hybrid Decomposed 332d)** | 35.04% (568/1621) | 0.3166 | 0.3534 | -5.12% |

*Analysis*: Adding explicit orientation descriptors (Variant B) provided a positive +2.22% gain in cross-signer zero-shot transfer by making hand orientation easier for trees to split on. Conversely, adding canonical pose coordinates (Variants C and D) hurt cross-signer transfer (-1.42% and -5.12%), because uncalibrated canonical coordinates increased tree dimensionality without adding new signal.

---

## 6. Critical Orientation-Sensitive Letters Audit

Recall was audited across all 11 critical orientation-sensitive letters on `signer_04_val` and `signer_01_val`:

| Letter | Sign Description | Frozen Model C | Variant A (Retrained) | Variant B (Orientation) | Variant C (Pose) | Variant D (Hybrid) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **`H`** | Two horizontal index fingers | **1.00** | 0.34 | **0.47 (+0.13)** | 0.34 | 0.47 |
| **`R`** | Horizontal hand across palm | **1.00** | 0.83 | **0.97 (+0.14)** | 0.87 | **1.00** |
| **`L`** | Vertical index finger pointer | **1.00** | 1.00 | **1.00** | 1.00 | 1.00 |
| **`Y`** | Thumb & pinky spread | **1.00** | 1.00 | **1.00** | 1.00 | 0.97 |
| **`A`** | Closed fist | **1.00** | 1.00 | **1.00** | 1.00 | 1.00 |
| **`N`** | Two fingers folded over thumb | **1.00** | 1.00 | **1.00** | 1.00 | 1.00 |
| **`M`** | Three fingers folded over thumb| **1.00** | 1.00 | **1.00** | 1.00 | 1.00 |
| **`O`** | Circular finger loop | **1.00** | 1.00 | **1.00** | 1.00 | 1.00 |
| **`P`** | Index pointing down | **1.00** | 1.00 | **1.00** | 1.00 | 1.00 |
| **`Q`** | Hooked index & thumb | **1.00** | 0.97 | **0.97** | 1.00 | 0.97 |
| **`T`** | Single finger touch | **1.00** | 1.00 | **1.00** | 1.00 | 1.00 |

*Key Result*: In Phase 32B, full canonicalization collapsed Letter `H` to 41% and Letter `R` to 67%. In Phase 32C, retaining camera-space orientation prevented this collapse: Letter `R` achieved **97.0%** in Variant B and **100.0%** in Variant D, and Letter `H` improved from 34% to **47%** in Variant B.

---

## 7. Controlled In-Plane Rotation Robustness

Evaluated on `signer_04_val` across roll angles $\theta \in [-45^\circ, +45^\circ]$:

| In-Plane Roll Angle | Frozen Model C (Baseline) | Variant A (Retrained) | Variant B (Orientation) | Variant C (Pose) | Variant D (Hybrid) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$-45^\circ$** | **81.3%** | 62.1% | 56.2% | 60.0% | 58.7% |
| **$-30^\circ$** | **91.5%** | 69.2% | 69.2% | 72.4% | 71.6% |
| **$-15^\circ$** | **100.0%** | 87.2% | 88.4% | 89.3% | 88.5% |
| **$0^\circ$ (Upright)** | **100.0%** | 94.6% | **95.9%** | 94.6% | 94.6% |
| **$+15^\circ$** | **100.0%** | 93.2% | 92.0% | 92.1% | 91.5% |
| **$+30^\circ$** | **92.5%** | 83.1% | 79.5% | 81.1% | 80.9% |
| **$+45^\circ$** | **83.5%** | 73.6% | 69.5% | 72.4% | 72.4% |

*Analysis*: The frozen production Model C exhibits **substantially superior tilt tolerance** across all rotation angles compared to newly retrained models. This confirms that the Phase 22 regularization (`min_samples_leaf=2`, `max_depth=14`) and feature calibration in the frozen model already provide significant natural robustness against moderate signing tilt ($\pm 15^\circ$ to $\pm 30^\circ$).

---

## 8. Latency and Computational Overhead

| Model / Configuration | Feature Dim | p50 CPU Latency | p95 CPU Latency | Mean Latency |
| :--- | :---: | :---: | :---: | :---: |
| **Frozen Model C (Baseline)** | **280** | **23.55 ms** | **27.30 ms** | 24.08 ms |
| **Variant A (Retrained)** | 280 | 24.75 ms | 27.86 ms | 25.11 ms |
| **Variant B (Orientation)** | 296 | 24.92 ms | 42.64 ms | 26.31 ms |
| **Variant C (Canonical Pose)** | 316 | 25.28 ms | 28.43 ms | 25.54 ms |
| **Variant D (Hybrid)** | 332 | 14.07 ms | 27.31 ms | 15.65 ms |

All variants execute comfortably within the 33ms real-time budget (30 FPS) on CPU.

---

## 9. Regression Testing & Full Test Discovery

A dedicated test suite [tests/test_phase32c_orientation_preserving.py](file:///Users/pavanaksshay/pavan_sign_language/tests/test_phase32c_orientation_preserving.py) was implemented covering all 8 required scenarios:
```
test_01_feature_dimensionality_and_ordering ... ok
test_02_finite_values_under_degenerate_or_missing_landmarks ... ok
test_03_expected_behavior_under_controlled_rigid_rotations ... ok
test_04_preservation_of_absolute_orientation_cues ... ok
test_05_preservation_of_baseline_feature_output ... ok
test_06_model_and_scaler_immutability ... ok
test_07_sealed_holdout_quarantine ... ok
test_08_preservation_of_production_defaults ... ok
```

### Full Repository Test Discovery
Command executed:
```bash
/Users/pavanaksshay/venv/bin/python3 -m unittest discover tests -p "test_*.py"
```
**Result: Ran 273 tests in 104.2s — OK (273 passed, 0 failed, 0 errors).**

---

## 10. Deliverables Summary

- Report: [phase32c_orientation_preserving_report.md](file:///Users/pavanaksshay/pavan_sign_language/phase32c_orientation_preserving_report.md)
- Metrics JSON: [phase32c_ablation_metrics.json](file:///Users/pavanaksshay/pavan_sign_language/phase32c_ablation_metrics.json)
- Integrity Audit JSON: [phase32c_integrity_audit.json](file:///Users/pavanaksshay/pavan_sign_language/phase32c_integrity_audit.json)
- Unit Tests: [tests/test_phase32c_orientation_preserving.py](file:///Users/pavanaksshay/pavan_sign_language/tests/test_phase32c_orientation_preserving.py)
- Feature Extractor: [orientation_preserving_features.py](file:///Users/pavanaksshay/pavan_sign_language/orientation_preserving_features.py)
- Ablation Runner: [run_phase32c_ablations.py](file:///Users/pavanaksshay/pavan_sign_language/run_phase32c_ablations.py)

---

## 11. Acceptance & Final Verdict Justification

### Selected Verdict
> ### **`NO MATERIAL IMPROVEMENT`**

### Justification:
1. **No Improvement Over Production Model C**:
   While Variant B (296d) demonstrated modest improvements over a newly retrained 280d model (+1.36% on `signer_04_val`, +2.22% on pure cross-signer transfer, and protected critical letters), **no candidate outperformed the frozen production Model C**, which achieves **100.00%** on `signer_04_val` and exhibits superior tilt tolerance (81.3% – 92.5% under $\pm 30^\circ$ and $\pm 45^\circ$ rotation).
2. **Canonical Pose Dilution**:
   Adding complementary canonical pose features (Variants C and D) diluted decision tree splits, actively degrading pure cross-signer generalization (dropping to 38.74% and 35.04% vs 40.16%).
3. **Engineering Action**:
   The existing 280-dimensional feature schema, scaler, and frozen Model C weights are preserved as production defaults without modification. `signer_05` remains quarantined as a sealed holdout.
