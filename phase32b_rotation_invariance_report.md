# Phase 32B — Palm-Canonical Rotation-Invariant Feature Ablation Report

## Executive Summary

Phase 32B investigated whether expressing hand landmarks in a local palm-canonical coordinate frame improves robustness to hand rotation, perspective, and signer-specific anatomical variations. 

Phase 31 previously identified camera-space coordinate sensitivity as a candidate cause of cross-signer generalization error. Phase 32A established a robust, deterministic active-hand selection and ordering baseline without retraining Model C. This phase tested the local coordinate canonicalization hypothesis in isolation using strictly permitted development data (`signer_01` and `signer_04`), with recording-grouped splits, zero frame-level leakage, and `signer_05` quarantined as a sealed holdout.

### Final Scientific Verdict
> **`FEATURE STABILITY OR SIGN DISTINCTION REGRESSED`**

### Key Findings
1. **Mathematical Invariance Achieved, But Semantic Sign Distinction Destroyed**:
   - An orthonormal local palm coordinate frame was engineered in [palm_canonical_transformer.py](file:///Users/pavanaksshay/pavan_sign_language/palm_canonical_transformer.py) using Gram-Schmidt orthogonalization along the palm's anatomical axes ($O = P_{\text{wrist}}$, longitudinal $\hat{\mathbf{y}}$, lateral $\hat{\mathbf{x}}$, palm normal $\hat{\mathbf{z}}$).
   - Under controlled in-plane rotation ($\pm 45^\circ$), canonical features demonstrated superior mathematical rotation resistance: at $-45^\circ$ roll, canonical coordinates achieved **91.49%** accuracy versus **83.23%** for camera-space baseline (+8.26% advantage).
   - **However, in sign language, global hand orientation is phonemic, not noise.**
   - By mathematically erasing global orientation relative to the camera/torso, palm-canonical features destroyed the semantic distinctions between signs whose identities rely on hand orientation:
     - Letter **`H`** (horizontal finger pointing) collapsed from **100.0% $\to$ 41.2%** recall on `signer_04_val` (a 58.8% absolute drop!).
     - Letter **`R`** (horizontal palm overlay) collapsed from **100.0% $\to$ 67.3%** recall.
     - Letter **`L`** (upright vertical index pointer) collapsed from **74.0% $\to$ 49.3%** recall on `signer_01_val`.
2. **Development Validation Accuracy Regressed**:
   - On the authoritative Phase 22 recording-grouped holdout partition of `signer_04_val` (811 frames, zero recording leakage):
     - **Variant A (Baseline Phase 32A Camera-Space)**: **100.00%** Top-1 Accuracy, **1.0000** Macro-F1.
     - **Variant B (Canonical Zero-Shot on Frozen Model C)**: **67.94%** Top-1 Accuracy (severe mismatch with trained tree splits).
     - **Variant C (Canonical Retrained on Permitted Data)**: **93.34%** Top-1 Accuracy, **0.9280** Macro-F1 (**-6.66% regression**).
     - **Variant D (Mirrored Canonical Retrained)**: **93.83%** Top-1 Accuracy, **0.9332** Macro-F1 (**-6.17% regression**).
   - On `signer_01_val` (1,134 frames):
     - **Variant A (Baseline)**: **68.69%** Top-1 Accuracy, **0.6655** Macro-F1.
     - **Variant C (Canonical Retrained)**: **62.79%** Top-1 Accuracy, **0.6062** Macro-F1 (**-5.90% regression**).
3. **Architectural Recommendation**:
   - Pure palm-canonical coordinates must **NOT** replace camera-space features. Discarding orientation harms sign recognition more than rotation invariance helps.
   - Production behavior and Model C weights remain **strictly unchanged**.

---

## 1. Governance & Baseline Verification

### 1.1 Quarantine and Artifact Immutability
- **`signer_05` Holdout Quarantine**: Completely sealed and untouched. Zero frames or recordings were inspected or evaluated.
- **Benchmark Isolation**: `signer_02` (3,083 frames) and `signer_03` (2,752 frames) frozen benchmark data remained untouched.
- **Permitted Development Data**: `signer_01_train` (3,077), `signer_01_val` (1,134), and `signer_04_usable` (1,621; split into 810 train and 811 val frames across 26 holdout recordings). Total: 5,832 frames.
- **Production Defaults**: Strictly preserved:
  `ISL_BACKEND_DETECTOR=yolo`, `ISL_RECOGNITION_ARCH=standard`, `ISL_TEMPORAL_SMOOTHING=0`, `ISL_SHADOW_MODE=0`, `ISL_MEDIAPIPE_DELEGATE=cpu`, `ISL_MODEL_C_RESOLUTION=native`.

### 1.2 Verification of Phase 32A Baseline & False Suppression Proof
Before initiating Phase 32B experiments, we re-verified the baseline metrics and resolved the question regarding false suppressions:
- **Proof of Baseline Preservation**:
  - `signer_01_val`: Exactly 12 false suppressions in legacy baseline, exactly 12 in Phase 32A baseline. Exact match index-for-index (indices `[293, 316, 504, 653, 658]`, etc.).
  - `signer_01_train`: Exactly 13 false suppressions in legacy baseline, exactly 13 in Phase 32A baseline. Exact match index-for-index.
  - `signer_04_usable`: Exactly 0 false suppressions in legacy baseline, exactly 0 in Phase 32A baseline.
- **Conclusion**: The 12 and 13 cases are **100% legacy baseline occurrences** that existed in the historical codebase prior to Phase 32A (originating from transitions where one hand dropped out of camera frame during synthetic capture). Exactly **zero new false suppressions** were introduced.

### 1.3 Immutable Artifact Cryptographic Checksum Verification

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

## 2. Audit of Existing Coordinate Representation

The 280-dimensional feature schema in Model C comprises:
- **Geometry v2**: 264 dimensions (Hand 0: 125 dims, Hand 1: 125 dims, Inter-hand: 14 dims).
- **Occlusion & Contact**: 16 dimensions.

### 2.1 Coordinate Breakdown: Camera-Space vs. Geometric Invariants

```
+----------------------------------------------------------------------------------------------------+
| 280-DIMENSIONAL FEATURE VECTOR                                                                     |
|                                                                                                    |
| [143 Dimensions: Camera-Space Orientation Dependent (54.2%)]                                       |
|  - Relative Normalized 3D Coordinates (63 dims/hand x 2 = 126 dims): (P_i - P_wrist) / palm_scale  |
|  - Palm Normal Vector (3 dims/hand x 2 = 6 dims): Cross product of index/pinky MCP in camera space|
|  - Axis-Aligned Bounding Span (4 dims/hand x 2 = 8 dims): bbox_w, bbox_h, aspect, coverage        |
|  - Inter-Hand Wrist Displacement (3 dims): (w0 - w1) / mean_scale in camera coordinates            |
|                                                                                                    |
| [121 Dimensions: Naturally Rotation Invariant (45.8%)]                                             |
|  - 5 Fingertip-to-Wrist Euclidean Distances per hand (10 dims)                                     |
|  - 5 Fingertip-to-MCP Euclidean Distances per hand (10 dims)                                       |
|  - 5 Finger Extension Ratios per hand (10 dims)                                                    |
|  - 6 Inter-Fingertip Pairwise Distances per hand (12 dims)                                         |
|  - 5 PIP Joint Flexion Cosine Angles per hand (10 dims)                                            |
|  - 5 Finger Curl Ratios per hand (10 dims)                                                         |
|  - 5 DIP Joint Flexion Cosine Angles per hand (10 dims)                                            |
|  - 5 MCP Base Flexion Cosine Angles per hand (10 dims)                                             |
|  - 5 Signed Palm-Plane Depths per hand (10 dims)                                                   |
|  - 2 Additional Inter-Tip Distances per hand (4 dims)                                              |
|  - 4 Adjacent Finger Directional Spread Angles per hand (8 dims)                                   |
|  - 10 Inter-Hand Distances, Contact Proxies, and Palm Normal Cosine Alignment (10 dims)            |
|  - 7 Presence & Confidence Flags (7 dims)                                                          |
+----------------------------------------------------------------------------------------------------+
```

### 2.2 Sensitivity Analysis
- **In-Plane Roll ($\theta$)**: Rotating the camera or the hand in the 2D image plane rotates $(x, y)$ coordinates by $\mathbf{R}_{2D}$. Because Random Forest trees evaluate axis-aligned thresholds ($x_i \le \tau$), in-plane tilt rotates coordinates into completely different decision paths.
- **Pitch and Yaw**: Foreshortens finger bones along the line of sight and alters estimated depth $z$.
- **Handedness**: The existing representation intentionally encodes handedness in metadata ($h_{\text{sign}} \in \{-1, 0, 1\}$) and Slot 0/1 assignment.

---

## 3. Palm-Canonical Coordinate System Specification

In [palm_canonical_transformer.py](file:///Users/pavanaksshay/pavan_sign_language/palm_canonical_transformer.py), an orthonormal local palm coordinate frame is defined:

### 3.1 Mathematical Definition
1. **Origin**:
   $$O = P_{\text{wrist}} = P_0$$
2. **Longitudinal Axis ($\hat{\mathbf{y}}$, Proximal $\to$ Distal)**:
   $$\mathbf{v}_{\text{long}} = P_{\text{mid\_mcp}} - P_{\text{wrist}} = P_9 - P_0$$
   $$\hat{\mathbf{y}} = \frac{\mathbf{v}_{\text{long}}}{\|\mathbf{v}_{\text{long}}\|} \quad (\text{fallback to } [0, -1, 0]^T \text{ if } \|\mathbf{v}_{\text{long}}\| < 10^{-4})$$
3. **Transverse Reference Vector ($\mathbf{v}_{\text{trans}}$, Radial $\to$ Ulnar)**:
   $$\mathbf{v}_{\text{trans}} = P_{\text{pinky\_mcp}} - P_{\text{index\_mcp}} = P_{17} - P_5$$
4. **Palm Normal Axis ($\hat{\mathbf{z}}$, Dorsal $\to$ Palmar)**:
   $$\mathbf{n} = \mathbf{v}_{\text{trans}} \times \hat{\mathbf{y}}$$
   $$\hat{\mathbf{z}} = \frac{\mathbf{n}}{\|\mathbf{n}\|} \quad (\text{fallback to } [0, 0, 1]^T \text{ if } \|\mathbf{n}\| < 10^{-4})$$
5. **Lateral Axis ($\hat{\mathbf{x}}$, Orthonormal to $\hat{\mathbf{y}}$ and $\hat{\mathbf{z}}$)**:
   $$\hat{\mathbf{x}} = \hat{\mathbf{y}} \times \hat{\mathbf{z}}$$
   $$\hat{\mathbf{z}} = \hat{\mathbf{x}} \times \hat{\mathbf{y}} \quad (\text{guarantees } \det(\mathbf{R}_{\text{palm}}) = +1.0)$$

6. **Coordinate Transformation**:
   The orthonormal rotation matrix from camera space to the palm frame is:
   $$\mathbf{R}_{\text{palm}} = \begin{bmatrix} \hat{\mathbf{x}}^T \\ \hat{\mathbf{y}}^T \\ \hat{\mathbf{z}}^T \end{bmatrix} \in \mathbb{R}^{3 \times 3}$$
   Normalized canonical landmark coordinates:
   $$P_i' = \frac{1}{s} \mathbf{R}_{\text{palm}} (P_i - P_{\text{wrist}}), \quad s = \|\mathbf{v}_{\text{long}}\|$$

### 3.2 Handedness Mirroring Policy (Variant D)
For a left hand, $\mathbf{v}_{\text{trans}}$ points in the opposite direction relative to the thumb. In Variant D, we evaluate anatomical mirroring:
$$\hat{\mathbf{x}}_{\text{mirrored}} = -\hat{\mathbf{x}}, \quad \hat{\mathbf{z}}_{\text{mirrored}} = -\hat{\mathbf{z}}$$
This maps both left and right hands into a single canonical right-hand anatomical frame ($\det(\mathbf{R}) = +1.0$).

---

## 4. Controlled Ablation Experiments

### 4.1 Experimental Variants
- **Variant A (Production Baseline)**: Existing 280d Phase 32A camera-space features + frozen Model C Random Forest.
- **Variant B (Zero-Shot Canonical)**: 280d Palm-canonical features evaluated directly on frozen Model C.
- **Variant C (Retrained Canonical)**: 280d Palm-canonical features with an experimental Random Forest and StandardScaler fit on permitted training data (`signer_01_train` + `signer_04_train`, exact Phase 22 hyperparams: 300 trees, depth=14, leaf=2).
- **Variant D (Retrained Mirrored Canonical)**: 280d Anatomically mirrored canonical features with an experimental Random Forest fit on permitted training data.

### 4.2 Overall Performance on Development Partitions

| Partition / Metric | Variant A (Baseline 32A) | Variant B (Zero-Shot) | Variant C (Canonical Retrained) | Variant D (Mirrored Retrained) | Best Candidate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`signer_04_val` (811 frames)** | | | | | |
| - Top-1 Accuracy | **100.00%** | 67.94% | 93.34% | 93.83% | **Variant A** |
| - Top-3 Accuracy | **100.00%** | 86.81% | 97.90% | 98.15% | **Variant A** |
| - Macro-F1 | **1.0000** | 0.6149 | 0.9280 | 0.9332 | **Variant A** |
| - Balanced Accuracy | **1.0000** | 0.6793 | 0.9338 | 0.9388 | **Variant A** |
| - Condition 1 Top-1 (Desk) | **100.00%** | 68.02% | 93.15% | 93.91% | **Variant A** |
| - Condition 2 Top-1 (Lateral)| **100.00%** | 67.87% | 93.53% | 93.76% | **Variant A** |
| **`signer_01_val` (1,134 frames)** | | | | | |
| - Top-1 Accuracy | **68.69%** | 20.90% | 62.79% | 63.58% | **Variant A** |
| - Top-3 Accuracy | **88.36%** | 35.71% | 85.89% | 85.80% | **Variant A** |
| - Macro-F1 | **0.6655** | 0.1556 | 0.6062 | 0.6107 | **Variant A** |
| - Balanced Accuracy | **0.6908** | 0.2091 | 0.6305 | 0.6385 | **Variant A** |
| **`signer_04_usable_all` (1,621 frames)** | | | | | |
| - Top-1 Accuracy | **100.00%** | 68.48% | 96.67% | 96.92% | **Variant A** |
| - Macro-F1 | **1.0000** | 0.6226 | 0.9655 | 0.9680 | **Variant A** |

---

## 5. Why Palm Canonicalization Fails: Phonemic Orientation Analysis

To understand why Variant C regressed by 6.66% on `signer_04_val` and 5.90% on `signer_01_val`, we audited the per-letter recall across all 26 classes:

### 5.1 Focus Letters Recall Audit

| Letter | Sign Description | Variant A Recall | Variant C Recall | Delta (C - A) | Root-Cause Mechanism |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **`H`** | Two index fingers extended **horizontally** pointing toward each other | **1.00** | **0.41** | **-0.59** | In canonical frame, the longitudinal axis is normalized to $+Y$. The critical distinction that index fingers point horizontally (not vertically like `I` or `G`) is completely erased! |
| **`R`** | One hand placed **horizontally across** the upright palm | **1.00** | **0.67** | **-0.33** | The horizontal cross orientation is normalized into the palm frame, causing severe confusion with vertical open-palm signs. |
| **`L`** | Index finger pointing **vertically UP**, thumb pointing horizontally | **0.74** | **0.49** | **-0.25** | Normalization removes the absolute vertical orientation of the index finger in camera space. |
| **`Y`** | Thumb and pinky extended, hand held upright | **0.59** | **0.41** | **-0.18** | Confused with tilted open hands. |
| **`A`** | Closed fist, thumb alongside index | **0.98** | **0.84** | **-0.14** | Fist orientation changes relative to wrist frame. |
| **`M`** | Three fingers folded over thumb | **0.57** | **0.46** | **-0.11** | Contact plane altered. |
| **`J`** | Index finger traces curve | **0.76** | **0.83** | **+0.07** | Slight benefit due to curvature invariance. |
| **`P`** | Two hands forming loop and stem | **0.74** | **0.60** | **-0.14** | Inter-hand relative displacement altered. |

### 5.2 Phonemic Orientation Principle
> [!IMPORTANT]
> **Phonemic Hand Orientation Principle**: Unlike 3D object recognition (where rotation is typically nuisance variation), in sign language **global hand orientation relative to the human body and camera is phonemic**. A hand held vertically versus horizontally communicates a completely different letter. Forcing every hand to point along $+Y$ in a local coordinate frame discards this primary linguistic signal.

---

## 6. Rotation Robustness Experiments

To verify the mathematical properties of canonicalization, we evaluated controlled geometric perturbations on `signer_04_val`:

### 6.1 In-Plane Roll Rotation ($\theta \in [-45^\circ, +45^\circ]$)

| In-Plane Roll Angle | Variant A (Baseline) | Variant C (Canonical) | Delta (C - A) | Analysis |
| :---: | :---: | :---: | :---: | :--- |
| **$-45^\circ$** | 83.23% | **91.49%** | **+8.26%** | Baseline degrades by 16.77%; Canonical is highly resilient. |
| **$-30^\circ$** | 88.90% | **93.34%** | **+4.44%** | Baseline degrades by 11.10%; Canonical maintains stability. |
| **$-15^\circ$** | **100.00%** | 93.22% | -6.78% | Baseline withstands moderate tilt. |
| **$0^\circ$ (Upright)** | **100.00%** | 93.34% | -6.66% | Baseline retains phonemic orientation. |
| **$+15^\circ$** | **99.75%** | 93.09% | -6.66% | Baseline withstands moderate tilt. |
| **$+30^\circ$** | 92.36% | **93.09%** | **+0.74%** | Canonical outperforms baseline. |
| **$+45^\circ$** | 82.12% | **86.31%** | **+4.19%** | Baseline degrades by 17.88%; Canonical maintains stability. |

### 6.2 Out-of-Plane Pitch Rotation ($\phi \in [-30^\circ, +30^\circ]$)

| Out-of-Plane Pitch Angle | Variant A (Baseline) | Variant C (Canonical) | Delta (C - A) |
| :---: | :---: | :---: | :---: |
| **$-30^\circ$** | **99.88%** | 92.48% | -7.40% |
| **$-15^\circ$** | **99.88%** | 93.22% | -6.66% |
| **$0^\circ$** | **100.00%** | 93.34% | -6.66% |
| **$+15^\circ$** | **99.88%** | 93.22% | -6.66% |
| **$+30^\circ$** | **99.75%** | 93.34% | -6.41% |

---

## 7. Latency and Computational Overhead

| Model / Pipeline Component | p50 CPU Latency | p95 CPU Latency | Feature Extraction Overhead |
| :--- | :---: | :---: | :---: |
| **Frozen Model C (Baseline)** | **24.09 ms** | **27.52 ms** | 0.358 ms |
| **Experimental RF C (Canonical)** | **25.50 ms** | **29.76 ms** | 0.442 ms (+0.084 ms for Gram-Schmidt basis) |

Basis calculation adds only ~84 microseconds per frame, demonstrating that numerical basis computation is extremely fast and feasible. The barrier to adoption is purely linguistic and accuracy-related, not computational.

---

## 8. Regression Tests and Test Discovery

A dedicated test suite [tests/test_phase32b_rotation_invariance.py](file:///Users/pavanaksshay/pavan_sign_language/tests/test_phase32b_rotation_invariance.py) was implemented covering all 8 required scenarios:
```
test_01_orthonormality_and_numerical_stability_of_palm_basis ... ok
test_02_invariance_to_controlled_in_plane_rotations ... ok
test_03_graceful_handling_of_degenerate_or_missing_landmarks ... ok
test_04_finite_output_and_exact_feature_dimensions ... ok
test_05_left_and_right_hand_handling ... ok
test_06_preservation_of_existing_extractor_and_frozen_artifacts ... ok
test_07_signer_05_holdout_quarantine_preserved ... ok
test_08_preservation_of_production_defaults ... ok
```

### Full Test Suite Discovery
Command executed:
```bash
/Users/pavanaksshay/venv/bin/python3 -m unittest discover tests -p "test_*.py"
```

**Result: Ran 265 tests in 94.1s — OK (265 passed, 0 failed, 0 errors).**

---

## 9. Deliverables Summary

- Report: [phase32b_rotation_invariance_report.md](file:///Users/pavanaksshay/pavan_sign_language/phase32b_rotation_invariance_report.md)
- Metrics: [phase32b_ablation_metrics.json](file:///Users/pavanaksshay/pavan_sign_language/phase32b_ablation_metrics.json)
- Integrity Audit: [phase32b_integrity_audit.json](file:///Users/pavanaksshay/pavan_sign_language/phase32b_integrity_audit.json)
- Unit Tests: [tests/test_phase32b_rotation_invariance.py](file:///Users/pavanaksshay/pavan_sign_language/tests/test_phase32b_rotation_invariance.py)
- Transformer Module: [palm_canonical_transformer.py](file:///Users/pavanaksshay/pavan_sign_language/palm_canonical_transformer.py)
- Ablation Script: [run_phase32b_ablations.py](file:///Users/pavanaksshay/pavan_sign_language/run_phase32b_ablations.py)

---

## 10. Conclusion and Next Steps

The empirical evidence is definitive:
- **Verdict**: `FEATURE STABILITY OR SIGN DISTINCTION REGRESSED`
- While palm canonicalization eliminates sensitivity to camera-plane tilt, it inadvertently discards phonemic hand orientation, regressing overall development accuracy by 6.66% on `signer_04_val` and 5.90% on `signer_01_val`, with severe degradation on orientation-dependent signs (`H` recall drops by 59%, `R` by 33%, `L` by 25%).
- **Governance Affirmation**: Production configuration, Model C weights, scaler, and pipeline defaults remain 100% intact. `signer_05` remains sealed.
- **Future Architectural Recommendation**: If rotation tolerance is desired in future phases, global orientation must **not** be discarded. Instead, an explicit decomposed representation (local finger pose angles + explicit camera-space orientation angles) or data augmentation during training should be explored rather than coordinate canonicalization.
