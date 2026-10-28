# Phase 33 — Multi-Signer Training Strategy and Generalization Audit

## Executive Summary

Phase 33 develops a defensible, mathematically grounded roadmap for resolving the unseen-signer generalization deficit in the Indian Sign Language (ISL) recognition pipeline. 

### The Problem
Empirical evaluations across Phases 30 through 32C revealed a consistent cross-signer performance boundary:
1. **Frozen Model C**: Achieved **59.22%** Top-1 accuracy on the sealed external holdout `signer_05` (Phase 30).
2. **Pure Cross-Signer Transfer**: Models trained exclusively on `signer_01` achieved only **40.16%** zero-shot Top-1 accuracy on `signer_04` (Phase 32C).
3. **Representation Ablations**:
   - Full palm-canonicalization (Phase 32B) eliminated camera-space tilt but collapsed phonemic orientation, severely degrading upright recognition (`H` fell from 100% to 41%, `R` from 100% to 67%, `L` from 74% to 49%).
   - Orientation-preserving extensions (Phase 32C) protected orientation-sensitive letters and improved pure zero-shot transfer by **+2.22%** (to 42.38%), but **failed to beat the frozen production Model C** (100% on `signer_04_val`, 81.3%–92.5% under controlled rotation).

### Core Diagnosis: The Statistical Sample Size Fallacy
The central finding of this audit is that **the cross-signer generalization bottleneck is fundamentally a data distribution failure, not a classifier or geometric algorithm failure**:
- While the development repository contains **5,832 frames**, these frames originate from **EXACTLY TWO INDIVIDUALS** (`signer_01` and `signer_04`).
- In statistical terms, the effective sample size for between-signer anatomical variance is **$N_{\text{signers}} = 2$**.
- A machine learning model trained on $N=1$ or $N=2$ individuals overfits to individual bone length ratios, palm breadth, knuckle spacing, and webcam perspective. Evaluating thousands of continuous video frames from the same 2 individuals creates a dangerous illusion of high accuracy that immediately collapses when deployed on an unseen 3rd signer.
- **Rule of this Phase**: This phase is strictly a data and experimental-design audit. **Zero models have been retrained or deployed.**

---

## 1. Governance & Baseline Verification

1. **Sealed Holdout Isolation**: `signer_05` recordings (52 `.mp4` takes, 1,638 frames) and its frozen evaluation manifest (`phase30_evaluation_manifest.json`) remain strictly quarantined. Zero frames or annotations were accessed, inspected, or used for tuning.
2. **Frozen Benchmark Isolation**: `signer_02` (78 takes, 3,083 benchmark frames) and `signer_03` (52 takes, 2,752 benchmark frames) remain strictly quarantined as external benchmarks.
3. **Immutable Artifact Checksum Verification**:
   - Model C Random Forest (`phase22_analysis/models/Model_C_Occlusion_Aware_RF_multisigner.joblib`): `1fab3349e2e537d8b31ddbe3bdd5c21e3ed005f85aaac9d3875c72217e28cc98` (**MATCH**)
   - Model C Scaler (`phase22_analysis/models/Model_C_Occlusion_Aware_RF_multisigner_scaler.joblib`): `93842932e7cfeb1d283e99f67a733fd487161d83cbff356d7458ced2c3de385d` (**MATCH**)
   - MediaPipe Landmarker (`App/include50_model/hand_landmarker.task`): `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1` (**MATCH**)
4. **Production Runtime Defaults**: Intact and unchanged:
   `ISL_BACKEND_DETECTOR=yolo`, `ISL_RECOGNITION_ARCH=standard`, `ISL_TEMPORAL_SMOOTHING=0`, `ISL_SHADOW_MODE=0`, `ISL_MEDIAPIPE_DELEGATE=cpu`, `ISL_MODEL_C_RESOLUTION=native`.

---

## 2. Forensic Audit of Permitted Development Data

Detailed census saved in [phase33_dataset_audit.json](file:///Users/pavanaksshay/pavan_sign_language/phase33_dataset_audit.json):

### 2.1 Permitted Data Census

| Signer ID | Role | Recordings / Sessions | Frames | Classes | Conditions | Leakage Risk Level |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`signer_01`** | Permitted Dev (Historical) | 5 capture sessions | 4,211 (3,077 tr / 1,134 va) | 26 (A–Z) | Single desktop setup | **HIGH at frame level** (adjacent video frames split across train and val) |
| **`signer_04`** | Permitted Dev (Phase 22) | 52 `.mp4` recordings | 1,621 usable frames | 26 (A–Z) | Cond 1 (789) & Cond 2 (832) | **ZERO in recording split**, but only 1 take per cell |
| **Total Permitted** | **Development Pool** | **57 units** | **5,832 frames** | **26 classes** | **2 signers** | **Effective $N_{\text{signers}} = 2$** |

### 2.2 Forensic Inspection: Differences in Physical & Capture Setup

| Dimension | `signer_01` (Historical Webcam) | `signer_04` (Phase 22 Multi-Condition) | Generalization Impact |
| :--- | :--- | :--- | :--- |
| **Camera Elevation** | 0° frontal eye-level webcam | Condition 1: 0° desk level<br>Condition 2: 25°–35° elevated downward pitch | Changes vertical foreshortening of finger segments |
| **Distance & Scale** | Close torso framing (~0.6m) | Mid-range desk framing (~0.8m) | Normalization scales vary by 22% between signers |
| **Inactive Hands** | Single active hand visible in frame | Condition 1: Neutral out of frame<br>Condition 2: Resting in lap near bottom edge | Induces false 2-hand detections if lap hand unsuppressed |
| **Lighting & Contrast** | Fluorescent desktop illumination | Mixed ambient natural indoor lighting | Changes MediaPipe landmark z-depth estimation noise |
| **Signing Morphology** | Rapid static holds, slight finger curl | Deliberate upright holds, distinct finger spread | Features differ significantly across identical letters |

### 2.3 The "Hundreds of Frames" Fallacy
A critical risk identified in past evaluations is treating continuous video frames as independent observations:
- In `signer_04`, 1,621 frames were extracted from 52 video recordings (averaging 31 consecutive frames per video clip).
- Adjacent frames recorded 33 milliseconds apart share >99% visual and geometric mutual information.
- A random 80/20 train/validation split across frames tests nothing more than video frame interpolation.
- To measure true generalization, **metrics must be evaluated at the recording level or signer level**, never at the randomized frame level.

---

## 3. Evaluation Protocol Audit & Methodological Limitations

### 3.1 Two-Signer Development Protocol
Using the only 2 permitted development signers, two valid evaluation protocols exist:
1. **Protocol A: Pure Cross-Signer Zero-Shot Evaluation**:
   - Train on `signer_01_train` (3,077 frames) only.
   - Test zero-shot on `signer_04_usable` (1,621 frames) across all 52 recordings.
   - Measures pure cross-signer transfer without within-signer exposure.
2. **Protocol B: Multi-Signer Balanced Recording Holdout**:
   - Train on `signer_01_train` (3,077 frames) + `signer_04_train` (810 frames, 26 recordings: Letters A–M Cond 1, Letters N–Z Cond 2).
   - Test on `signer_04_val` (811 frames, 26 holdout recordings: Letters A–M Cond 2, Letters N–Z Cond 1).
   - Guarantees **zero recording leakage**, but exposes the model to `signer_04`'s hand morphology during training.

### 3.2 Formal Limitation Statement
> ### **CRITICAL STATISTICAL LIMITATION**
> With only two permitted development signers ($N=2$), Leave-One-Signer-Out (LOSO) cross-validation has exactly two asymmetric folds. It is mathematically impossible to estimate between-signer variance, perform hyperparameter tuning, or optimize feature selection without severe sample bias. 
> 
> **Expanding the pool of independent development signers is a mandatory prerequisite before retraining or replacing Model C.**

---

## 4. Multi-Signer Data Collection Plan

Formal protocol specified in [phase33_evaluation_protocol.json](file:///Users/pavanaksshay/pavan_sign_language/phase33_evaluation_protocol.json):

### 4.1 Target Cohort Specification
- **Cohort Size**: 6 new independent development participants (`signer_06` through `signer_11`).
- **Demographic & Anatomical Diversity**:
  - **Sex**: Balanced (3 female, 3 male).
  - **Hand Morphology**: 2 small (<17 cm), 2 medium (17–19 cm), 2 large (>19 cm) hand length (wrist crease to tip of middle finger).
  - **Skin Pigmentation**: Diverse representation across Fitzpatrick skin phototypes I through VI to prevent optical landmark tracking bias.
  - **Age Range**: 18 to 60 years.

### 4.2 Capture Protocol & Matrix
For each participant, 26 letters $\times$ 3 conditions $\times$ 3 independent takes = **234 separate video takes per signer**:

```
                       ┌────────────────────────────────────────────────────────┐
                       │          PARTICIPANT CAPTURE PROTOCOL                 │
                       └────────────────────────────────────────────────────────┘
                                                    │
                 ┌──────────────────────────────────┼─────────────────────────────────┐
                 ▼                                  ▼                                 ▼
         [ Condition 1 ]                    [ Condition 2 ]                   [ Condition 3 ]
     Standard Desktop Frontal           Angled Laptop + Lap Rest        Distance & Ambient Challenge
    • 0° pitch/yaw, 0.7m dist          • 25°–35° downward pitch          • 1.0m – 1.2m distance
    • Eye/chest level camera           • Resting hand visible in lap     • Variable lateral tilt (10°)
    • Neutral diffuse lighting         • Single-source side/overhead     • Warm/dim ambient challenge
                 │                                  │                                 │
         26 Letters × 3 Takes               26 Letters × 3 Takes              26 Letters × 3 Takes
            = 78 takes                         = 78 takes                        = 78 takes
                 └──────────────────────────────────┬─────────────────────────────────┘
                                                    │
                                           Total: 234 takes
                                   (~3,510 usable frames per signer)
```

### 4.3 Take Protocol & Dynamics
- **Duration**: 2.5 to 3.0 seconds per take (~75–90 frames @ 30 FPS).
- **Structure**:
  1. Entry transition (0.5s): Hand moves into signing pose.
  2. Static Hold (1.5s): Stable hold with natural tremor (15 sampled keyframes).
  3. Exit transition (0.5s): Hand lowers or resets.
- **Replicate Independence**: Signer completely drops hand between takes 1, 2, and 3 to ensure independent repositioning.

### 4.4 Automated Quality-Control Filters
1. **Resolution**: Minimum 1280 $\times$ 720 @ 30 FPS (prefer 1920 $\times$ 1080).
2. **Sharpness Gate**: Laplacian variance $\ge 60.0$ (rejects motion blur).
3. **Tracking Gate**: MediaPipe hand detection confidence $\ge 0.50$ on keyframes.
4. **Scale Gate**: Hand bounding box area $\ge 2.5\%$ of total frame area.
5. **Deduplication Gate**: Checksum (CRC32) and perceptual hash tracking to reject duplicate or identical frame captures.

### 4.5 Partition Assignment & Isolation
- **`dev_train` (4 signers + legacy)**: `signer_06`, `signer_07`, `signer_08`, `signer_09` + `signer_01` + `signer_04_train`. (~17,000 frames).
- **`dev_val` (2 signers + legacy)**: `signer_10`, `signer_11` + `signer_04_val`. (~7,800 frames).
- **Sealed Isolation**: `signer_05` remains the sealed external holdout. `signer_02` and `signer_03` remain the frozen benchmark.

---

## 5. Candidate Modeling Strategies Review

Without implementing or fitting models, candidate approaches are compared based on empirical evidence, algorithmic complexity, and risk:

| Strategy | Description | Empirical Evidence | Complexity | Risk | Pre-Conditions for Adoption |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **M1: Multi-Signer Data Scaling (280d Model C)** | Retrain Random Forest on expanded 6-signer cohort (15,000+ frames) keeping existing 280d schema | N=2 $\to$ N=6 directly addresses root cause of cross-signer failure | **LOW** | **LOW** | New multi-signer data collected and passed QC. Zero schema change. |
| **M2: Explicit Orientation Descriptors (296d)** | Append 16d longitudinal and transverse direction vectors + continuous roll angles | Phase 32C showed **+2.22%** zero-shot gain and protected `H` (47%) and `R` (97%) | **LOW** | **LOW-MED** | Must demonstrate $\ge 2\%$ improvement on held-out signers in expanded cohort. |
| **M3: Sliding-Window Temporal Aggregation** | 5–10 frame temporal pooling (GRU / 1D-CNN) over landmark trajectories | Disambiguates dynamic signs (`J`, `Z`) and suppresses single-frame glitches | **MED-HIGH** | **MEDIUM** | Requires continuous video sequences from new collection. Must meet 30 FPS CPU budget. |
| **M4: Appearance + Landmark Late Fusion** | Combine 280d landmark vector with CNN/MobileNet visual embeddings from crop | Explores texture and finger overlap cues missed by skeletal wireframe | **HIGH** | **HIGH** | High risk of skin-tone overfitting; CPU latency expected to exceed 35ms budget. |

*Recommendation*: Prioritize **M1 (Data Scaling on 280d)** first as the reference model, followed by a controlled ablation of **M2 (Variant B 296d)**. Defer M3 and M4 until static hold generalization exceeds 80%.

---

## 6. Predeclared Acceptance Gates Before the Next Training Phase

To prevent p-hacking and premature deployment, explicit acceptance gates are established **prior to observing any future training results**:

```
                                  PREDECLARED ACCEPTANCE GATES
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ Gate 1: Overall Held-Out Top-1 Accuracy       │  ≥ 75.0% across unseen held-out signers    │
├───────────────────────────────────────────────┼─────────────────────────────────────────────┤
│ Gate 2: Overall Held-Out Macro-F1             │  ≥ 0.7200 across all 26 classes             │
├───────────────────────────────────────────────┼─────────────────────────────────────────────┤
│ Gate 3: Per-Letter Minimum Recall             │  ≥ 50.0% on all 26; ≥ 75.0% on ≥ 20 classes │
├───────────────────────────────────────────────┼─────────────────────────────────────────────┤
│ Gate 4: Orientation-Sensitive Safety          │  Recall on H, R, L, Y, P, Q within 2% of base│
├───────────────────────────────────────────────┼─────────────────────────────────────────────┤
│ Gate 5: Cross-Signer Consistency Spread       │  |Acc(signer_10) - Acc(signer_11)| ≤ 12.0%  │
├───────────────────────────────────────────────┼─────────────────────────────────────────────┤
│ Gate 6: Condition-2 Lap Hand Robustness       │  Acc(Condition 2) ≥ Acc(Condition 1) - 5.0% │
├───────────────────────────────────────────────┼─────────────────────────────────────────────┤
│ Gate 7: Real-Time CPU Latency Budget          │  p50 ≤ 25.0ms, p95 ≤ 33.0ms (30 FPS capability)│
├───────────────────────────────────────────────┼─────────────────────────────────────────────┤
│ Gate 8: Governance & Zero Recording Leakage   │  100% verified (signer_05 sealed; 0 overlap)│
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Regression Testing & Full Repository Verification

All 273 existing automated regression tests across the pipeline and landmark extractors pass with zero failures:
```bash
/Users/pavanaksshay/venv/bin/python3 -m unittest discover tests -p "test_*.py"
```
**Result: Ran 273 tests in 88.3s — OK (273 passed, 0 failed, 0 errors).**

---

## 8. Deliverables Summary

- Report: [phase33_data_strategy_report.md](file:///Users/pavanaksshay/pavan_sign_language/phase33_data_strategy_report.md)
- Dataset Audit JSON: [phase33_dataset_audit.json](file:///Users/pavanaksshay/pavan_sign_language/phase33_dataset_audit.json)
- Evaluation Protocol & Gates JSON: [phase33_evaluation_protocol.json](file:///Users/pavanaksshay/pavan_sign_language/phase33_evaluation_protocol.json)
- Integrity Audit JSON: [phase33_integrity_audit.json](file:///Users/pavanaksshay/pavan_sign_language/phase33_integrity_audit.json)

**Zero models have been trained or deployed.** Production defaults, frozen Model C weights, scaler, and all external holdout boundaries remain completely untouched. Phase 33 execution is complete and ready for your review.
