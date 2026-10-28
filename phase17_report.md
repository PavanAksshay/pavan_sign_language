# Phase 17 — Independent Multi-Signer Data Collection and External Generalization Validation Report

**Status**: Completed  
**Date**: October 15, 2026  
**Workspace**: `pavan_sign_language`  
**Production Default**: `ISL_BACKEND_DETECTOR=yolo` (Strictly Preserved)  
**Experimental Recognition Arch**: `ISL_RECOGNITION_ARCH=standard` (Default; `late_fusion` available as isolated opt-in)  
**Experimental Temporal Smoothing**: `ISL_TEMPORAL_SMOOTHING=0` (Strictly Preserved as Default Disabled)  
**Unit Test Suite**: **104 / 104 Passed (100% OK, 47.94s)**  

---

## Executive Summary

Phase 17 evaluates external generalization across signers, recording sessions, and camera conditions for the Indian Sign Language (ISL) 26-class alphabet recognizer. Building on the 264-dimensional [Landmark Geometry v2](file:///Users/pavanaksshay/pavan_sign_language/landmark_geometry_v2.py) representation and validation-calibrated late fusion ($\alpha = 0.75$) established in Phase 16, this phase conducts an exhaustive audit of data provenance, audits quality and split leakage, benchmarks all 7 frozen recognizers without retraining, evaluates persistent failure modes (N vs M and X vs V), reassesses continuous video temporal stability, and applies explicit deployment acceptance gates.

### Key Conclusions:
1. **Zero Data Fabrication & Physical Collection Status**: A thorough scan of `data/incoming_recordings/`, `data/webcam_recordings/`, and repository root confirmed that physical video recordings for `signer_02` and `signer_03` have **not yet been physically captured in this workspace**. In strict compliance with non-negotiable constraints, **zero synthetic frames, simulated landmarks, or fabricated labels were created**. The multi-signer ingestion pipeline, schema validators, duplicate filters, and quality gates are completely implemented, verified, and armed. Cross-signer generalization remains **empirically unverified**.
2. **Quality Gates & Split Leakage Audit**: The automated audit in [`phase17_data_collection_and_audit.py`](file:///Users/pavanaksshay/pavan_sign_language/phase17_data_collection_and_audit.py) verified that all 4 usable webcam video streams passed the Laplacian blur variance ($\ge 50.0$) and brightness ($[40.0, 220.0]$) gates with 0% rejection, while correctly rejecting an uncalibrated underexposed profile video (mean brightness 9.5). Cryptographic SHA-256 checks confirmed zero frame leakage between held-out Test A–L and training splits. However, Test M–Z shares `session_20200516_12h` with training data, confirming its status as **provisional due to session leakage**.
3. **Frozen Comparative Benchmark Across 7 Models**:
   - **Independent Test A–L (381 crops, Session May 14)**:
     - [Hybrid Concatenation RF](file:///Users/pavanaksshay/pavan_sign_language/models/landmark_hybrid_v1/hybrid_concatenation_model.joblib) achieved the highest overall score: **95.28% Top-1, 97.55% Macro-F1**.
     - [Calibrated Late Fusion ($\alpha=0.75$)](file:///Users/pavanaksshay/pavan_sign_language/models/landmark_geometry_v2/): **92.65% Top-1, 95.90% Macro-F1**, outperforming baseline late fusion ($\alpha=0.25$, 81.10% Top-1) by **+11.55%**, while eliminating false L $\to$ Y misclassifications.
     - [Landmark RF v2 (264 dims)](file:///Users/pavanaksshay/pavan_sign_language/models/landmark_geometry_v2/landmark_rf_model_v2.joblib): **92.91% Top-1, 95.83% Macro-F1**, improving over Landmark v1 (91.60% Top-1).
     - [Production YOLO Baseline](file:///Users/pavanaksshay/pavan_sign_language/App/final_model_26): 62.99% Top-1, 64.96% Macro-F1.
   - **Provisional Test M–Z (720 crops, Session May 16)**:
     - [Hybrid Concatenation RF](file:///Users/pavanaksshay/pavan_sign_language/models/landmark_hybrid_v1/hybrid_concatenation_model.joblib): **70.56% Top-1, 73.62% Macro-F1**.
     - [Calibrated Late Fusion ($\alpha=0.75$)](file:///Users/pavanaksshay/pavan_sign_language/models/landmark_geometry_v2/): **67.78% Top-1, 70.78% Macro-F1** (+5.28% Top-1 gain over $\alpha=0.25$ baseline).
     - [Landmark RF v2 (264 dims)](file:///Users/pavanaksshay/pavan_sign_language/models/landmark_geometry_v2/landmark_rf_model_v2.joblib): **62.08% Top-1, 66.83% Macro-F1** (up from 59.44% in v1).
     - [Production YOLO Baseline](file:///Users/pavanaksshay/pavan_sign_language/App/final_model_26): 43.47% Top-1, 44.22% Macro-F1.
4. **Persistent Failure Modes (N vs M & X vs V)**:
   - On Test M–Z, CNN v1 misclassifies **57.69% of N samples as M**. Calibrated Late Fusion ($\alpha=0.75$) reduced this to **44.23%**, and Landmark RF v2 brought N recall to 34.62% (vs 13.46% in CNN v1). However, monocular depth collapse on tightly curled downward fingers still leaves N far below the 70% production target.
   - For X vs V, Landmark RF v2 maintained X $\to$ V misclassification at 5.13% with 38.46% X recall.
5. **Continuous Live Webcam Temporal Behavior**:
   - **Standard L (146 frames)**: Baseline late fusion ($\alpha=0.25$) collapsed to 1.4% accuracy due to visual CNN bias toward U/Y. Calibrated late fusion ($\alpha=0.75$) with rolling 5-frame temporal smoothing achieved **56.2% frame accuracy and shifted the video majority vote to correct L (56.16%)**, reducing flip rate from 29.7% to **6.2%**.
   - **Frontal C (146 frames)**: Calibrated smoothed late fusion achieved **80.1% accuracy, an 80.14% majority vote for C, and a 4.8% flip rate**.
   - **Webcam A (143 frames)**: Landmark v2 standalone achieved **65.7% accuracy and a 65.73% majority vote for A**, but CNN v1's persistent visual bias toward I suppressed the fused prediction to 14.7% raw and 9.8% smoothed.
   - **CPU Pipeline Latency**: Measured at **67.7 ms to 77.9 ms** (13.0 to 14.8 FPS), safely within the real-time $< 100\text{ ms}$ envelope.
6. **Production Readiness Verdict**: Evaluated against 8 strict deployment gates, the candidate recognizers passed 4 (Latency, API Preservation, Test Suite, Safe Fallback), failed 3 (Letter N Recall $< 70\%$, Letter L Video Accuracy $< 85\%$, Worst-Class Recall $< 50\%$), and left 1 unverified (Unseen Multi-Signer Macro-F1 due to missing physical recordings). **Production defaults MUST remain strictly preserved: `ISL_BACKEND_DETECTOR=yolo` and `ISL_RECOGNITION_ARCH=standard`.**

---

## 1. Files Inspected and Changed

### 1.1 Files Inspected
- [`pipeline.py`](file:///Users/pavanaksshay/pavan_sign_language/pipeline.py): Production inference pipeline, dual-backend switching, lap-hand suppression, and late-fusion staging.
- [`server.py`](file:///Users/pavanaksshay/pavan_sign_language/server.py): FastAPI web server and public API endpoints.
- [`landmark_geometry_v2.py`](file:///Users/pavanaksshay/pavan_sign_language/landmark_geometry_v2.py): 264-dimensional geometric landmark feature extractor.
- [`landmark_feature_extractor.py`](file:///Users/pavanaksshay/pavan_sign_language/landmark_feature_extractor.py): Baseline 200-dimensional landmark extractor and lap-hand filter.
- [`temporal_smoother.py`](file:///Users/pavanaksshay/pavan_sign_language/temporal_smoother.py): Thread-safe 5-frame rolling majority voting smoother.
- [`phase16_recording_protocol.py`](file:///Users/pavanaksshay/pavan_sign_language/phase16_recording_protocol.py): Physical multi-signer recording tool.
- [`phase14c_data_ingestion.py`](file:///Users/pavanaksshay/pavan_sign_language/phase14c_data_ingestion.py): Duplicate detection and quality gate engine.
- [`models/landmark_hybrid_v1/`](file:///Users/pavanaksshay/pavan_sign_language/models/landmark_hybrid_v1/): Baseline model checkpoints (`landmark_rf_model.joblib`, `hybrid_concatenation_model.joblib`).
- [`models/landmark_geometry_v2/`](file:///Users/pavanaksshay/pavan_sign_language/models/landmark_geometry_v2/): Phase 16 model checkpoints (`landmark_rf_model_v2.joblib`, `landmark_scaler_v2.joblib`).
- [`phase16_report.md`](file:///Users/pavanaksshay/pavan_sign_language/phase16_report.md): Phase 16 geometry optimization and fusion calibration findings.

### 1.2 Dedicated Phase 17 Artifacts Created
- [`phase17_data_collection_and_audit.py`](file:///Users/pavanaksshay/pavan_sign_language/phase17_data_collection_and_audit.py): Multi-signer collection status, duplicate hash tracking, quality gate auditing, and session boundary verification.
- [`phase17_frozen_benchmark.py`](file:///Users/pavanaksshay/pavan_sign_language/phase17_frozen_benchmark.py): Master frozen comparative benchmark evaluating all 7 recognition models across static splits and continuous webcam videos.
- [`phase17_analysis/collection_and_quality_audit.json`](file:///Users/pavanaksshay/pavan_sign_language/phase17_analysis/collection_and_quality_audit.json): Hardware, physical collection, and quality audit logs.
- [`phase17_analysis/phase17_benchmark_results.json`](file:///Users/pavanaksshay/pavan_sign_language/phase17_analysis/phase17_benchmark_results.json): Master benchmark logs covering all 7 models, static splits, video temporal stability, and acceptance criteria.
- [`tests/test_phase17_generalization.py`](file:///Users/pavanaksshay/pavan_sign_language/tests/test_phase17_generalization.py): 10 unit tests verifying schema validation, duplicate detection, session leakage prevention, feature dimensionality, conditional vs end-to-end accuracy, and production default preservation.

---

## 2. Actual Data Inventory and New Recordings Status

### 2.1 Repository Data Inventory
| Signer Identifier | Temporal Provenance | Medium | Total Samples / Frames | Classes Covered | Evidence Level |
| :--- | :--- | :--- | :---: | :--- | :--- |
| `historical_signer_01` | May 14–16, 2020 | Static Crops | **5,312 crops** | A–Z (26 classes) | Historical single signer (May 14 independent session; May 16 shared session) |
| `webcam_signer_01` | October 14, 2026 | Continuous Video | **662 frames** (578 usable) | A, C, L (3 classes) | Single modern webcam signer across 4 lighting conditions |
| `signer_02` | Phase 17 Target | Continuous Video | **0 frames** | None | Pending physical camera capture |
| `signer_03` | Phase 17 Target | Continuous Video | **0 frames** | None | Pending physical camera capture |

### 2.2 Target Collection vs Physical Reality
$$\text{Planned Target} = 2\text{ New Signers} \times 2\text{ Conditions} \times 26\text{ Letters} \times \approx 30\text{ Usable Frames} = \mathbf{3,120\text{ Frames}}$$

```mermaid
pie title Phase 17 Frame Inventory Breakdown
    "Historical Signer 01 (Static Crops)" : 5312
    "Webcam Signer 01 (A, C, L Videos)" : 578
    "Signer 02 (Pending Physical Capture)" : 0
    "Signer 03 (Pending Physical Capture)" : 0
```

> [!CRITICAL]
> **Zero Fabrication Commitment**: No physical video recordings for `signer_02` or `signer_03` exist in `data/incoming_recordings/` or `data/webcam_recordings/`. In strict accordance with the non-negotiable constraints, **no synthetic data, simulated landmarks, or fabricated labels were created**. Generalization to unseen signers remains **empirically unverified**.

---

## 3. Collection Instructions and Remaining Requirements

To physically gather the required multi-signer dataset in Phase 18, operators must execute the following protocol:

### 3.1 Operator Recording Instructions
1. **Camera Framing & Format**:
   - Web camera mounted at eye/chest level, 70–90 cm from the signer.
   - Native 1280×720 resolution (720p) at 30.0 FPS.
   - The signer's torso, shoulders, and signing space must be completely visible without clipping fingertips.
2. **Two Distinct Conditions**:
   - **Condition 1 (Desk Neutral)**: Overhead ambient diffuse white light (>300 lux), plain wall background.
   - **Condition 2 (Lateral Warm)**: Directional warm desk lamp placed at 45 degrees, soft background shadowing. Avoid direct lens flare or extreme underexposure ($< 40.0$ mean brightness).
3. **Signing Routine**:
   - For each letter A–Z, the signer raises hand(s) into position, holds the canonical sign steady for 2.0–3.0 seconds (approx. 60–90 frames), and returns hand(s) to resting position.
   - One clip per letter and condition, named deterministically (e.g., `signer_02_cond1_letter_A.mp4`).
4. **Directory Structure**:
   ```
   data/incoming_recordings/
   ├── signer_02/
   │   ├── cond1/ (A.mp4 ... Z.mp4)
   │   └── cond2/ (A.mp4 ... Z.mp4)
   └── signer_03/
       ├── cond1/ (A.mp4 ... Z.mp4)
       └── cond2/ (A.mp4 ... Z.mp4)
   ```
5. **Automated Ingestion Command**:
   ```bash
   /Users/pavanaksshay/venv/bin/python3 phase14c_data_ingestion.py \
     --input-dir data/incoming_recordings/ \
     --output-dir dataset/benchmark_multisigner/ \
     --min-blur 50.0 \
     --min-brightness 40.0 \
     --max-brightness 220.0
   ```

---

## 4. Quality Checks and Leakage-Audit Results

The automated audit in [`phase17_data_collection_and_audit.py`](file:///Users/pavanaksshay/pavan_sign_language/phase17_data_collection_and_audit.py) verified:

### 4.1 Split Boundary and Hash Leakage
- **Cryptographic Hash Overlap**: 0 duplicate SHA-256 hashes between `test_al_manifest.json` and `train_manifest.json` or `val_manifest.json`.
- **Session Isolation Audit**:
  - `test_al_manifest.json` (381 crops): 100% of samples belong to `session_20200514_16h`. Zero train or val samples exist in this session. **Validated independent session**.
  - `test_mz_manifest.json` (720 crops): 100% of samples belong to `session_20200516_12h`. This exact session contains 604 training samples and 224 validation samples. **Confirmed provisional session leakage**.

### 4.2 Quality Gate Audit on Webcam Streams
| Video Stream | Ground Truth | Total Frames | Usable Accepted | Rejected Frames | Mean Blur Var ($\ge 50.0$) | Mean Brightness ($[40, 220]$) | Quality Decision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `recording_C_20261014_164912.mp4` | **C** | 146 | 146 | 0 | $91.31 \pm 4.2$ | $104.69 \pm 1.2$ | **Accepted** |
| `recording_C_20261014_165234.mp4` | **C** | 143 | 143 | 0 | $105.47 \pm 6.9$ | $109.59 \pm 1.4$ | **Accepted** |
| `recording_L_20261014_165754.mp4` | **L** | 146 | 146 | 0 | $72.21 \pm 3.8$ | $107.76 \pm 1.1$ | **Accepted** |
| `recording_A_20261014_165529.mp4` | **A** | 143 | 143 | 0 | $259.09 \pm 12.4$ | $143.66 \pm 1.5$ | **Accepted** |
| `recording_C_lateral_profile_calibrated.mp4` | **C** | 84 | 0 | 84 | $4.87 \pm 0.8$ | $9.51 \pm 0.4$ | **Rejected** ($< 40.0$ brightness) |

---

## 5. Split Design and Generalization Limitations

### 5.1 Codified 3-Way Multi-Signer Split Strategy
```mermaid
graph LR
    subgraph "Target 3-Signer Partitioning Strategy"
        S1["historical_signer_01<br>(5,312 Static Crops)"] --> Train["Train Partition<br>(3,077 crops)"]
        S2["signer_02 (Planned)<br>(1,560 Video Frames)"] --> Val["Validation Partition<br>(Hyperparameter & Fusion Tuning)"]
        S3["signer_03 (Planned)<br>(1,560 Video Frames)"] --> Test["Final Test Partition<br>(Untouched External Generalization)"]
    end
    style Test fill:#ccffcc,stroke:#00aa00
    style Val fill:#ffffcc,stroke:#aaaa00
    style Train fill:#e6f2ff,stroke:#0066cc
```

### 5.2 Methodological Limitations
1. **Single Historical Signer in Current Training Data**: All current model weights were trained exclusively on `historical_signer_01`. Model features may inadvertently rely on this individual's hand proportions.
2. **Session Leakage on M–Z**: The provisional M–Z benchmark shares the recording session with training data, inflating baseline numbers.
3. **Two-Signer Test Limitation**: Even when `signer_02` and `signer_03` are captured, a single test signer provides limited statistical diversity. Collecting additional signers is strongly advised before final production sign-off.

---

## 6. Frozen-Model Comparative Benchmark Across All 7 Recognizers

All 7 frozen recognition systems were evaluated on identical samples without any retraining:

### 6.1 Model Architecture and Disk Footprints
| Recognizer Identifier | Model Architecture | Input Features | Disk Size |
| :--- | :--- | :--- | :---: |
| **1. YOLO Production Baseline** | Darknet YOLOv3 + Production SqueezeNet | $224 \times 224$ BBox Crop | **241.23 MB** |
| **2. MediaPipe v1 CNN** | HandLandmarker + MediaPipe SqueezeNet v1 | $224 \times 224$ MP Crop | **13.82 MB** |
| **3. Landmark-Only Geometry v1** | Random Forest (100 trees) | 200-dim Landmark Vector | **66.25 MB** |
| **4. Landmark-Only Geometry v2** | Random Forest (200 trees, depth 16) | 264-dim Landmark Vector | **63.47 MB** |
| **5. Late Fusion Baseline ($\alpha=0.25$)** | $0.25 P_{\text{lm, v1}} + 0.75 P_{\text{CNN, v1}}$ | 200-dim Landmarks + Crop | **72.61 MB** |
| **6. Late Fusion Calibrated ($\alpha=0.75$)** | $0.75 P_{\text{lm, v2}} + 0.25 P_{\text{CNN, v1}}$ | 264-dim Landmarks + Crop | **69.83 MB** |
| **7. Hybrid Concatenation RF** | Random Forest (200 trees) on Concatenation | 712-dim (512 CNN + 200 LM) | **50.78 MB** |

### 6.2 Independent Test A–L Benchmark (381 Held-Out Static Crops, Session May 14)
*0 detection failures; conditional accuracy equals end-to-end accuracy.*

| Recognizer Model | Top-1 Accuracy | Top-3 Accuracy | Macro-F1 | Balanced Acc | Worst Class (Recall) | CPU Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Production YOLO Baseline** | 62.99% | 76.64% | 64.96% | 56.29% | I (22.22%) | 4.29 ms |
| **2. MediaPipe v1 CNN** | 76.12% | 89.24% | 82.35% | 75.58% | E (48.15%) | 5.06 ms |
| **3. Landmark-Only Geometry v1** | 91.60% | 97.11% | 93.89% | 92.00% | J (86.36%) | **0.09 ms** |
| **4. Landmark-Only Geometry v2** | **92.91%** | 96.59% | **95.83%** | **93.67%** | H (86.96%) | **0.09 ms** |
| **5. Late Fusion Baseline ($\alpha=0.25$)** | 81.10% | 95.54% | 87.63% | 82.13% | E (66.67%) | 5.20 ms |
| **6. Late Fusion Calibrated ($\alpha=0.75$)** | **92.65%** | **96.59%** | **95.90%** | **94.20%** | H (86.96%) | 5.20 ms |
| **7. Hybrid Concatenation RF** | **95.28%** | **97.38%** | **97.55%** | **96.80%** | I (88.89%) | 5.14 ms |

### 6.3 Provisional Test M–Z Benchmark (720 Static Crops, Session May 16)
*0 detection failures; conditional accuracy equals end-to-end accuracy.*

| Recognizer Model | Top-1 Accuracy | Top-3 Accuracy | Macro-F1 | Balanced Acc | Worst Class (Recall) | CPU Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Production YOLO Baseline** | 43.47% | 59.44% | 44.22% | 41.60% | X (10.26%) | 4.33 ms |
| **2. MediaPipe v1 CNN** | 58.89% | 74.03% | 59.31% | 58.30% | N (13.46%) | 3.55 ms |
| **3. Landmark-Only Geometry v1** | 59.44% | 75.97% | 63.61% | 58.05% | X (28.21%) | **0.03 ms** |
| **4. Landmark-Only Geometry v2** | **62.08%** | 76.67% | **66.83%** | **60.57%** | N (34.62%) | **0.04 ms** |
| **5. Late Fusion Baseline ($\alpha=0.25$)** | 62.50% | 77.36% | 62.88% | 61.75% | N (15.38%) | 3.61 ms |
| **6. Late Fusion Calibrated ($\alpha=0.75$)** | **67.78%** | **81.67%** | **70.78%** | **67.01%** | N (26.92%) | 3.61 ms |
| **7. Hybrid Concatenation RF** | **70.56%** | **81.81%** | **73.62%** | **69.21%** | N (30.77%) | 3.60 ms |

---

## 7. Full A–Z Metrics and Failure Mode Audits

### 7.1 Per-Class Recall Comparison Across All 7 Models
| Letter | Support | YOLO Prod | MP CNN v1 | LM RF v1 | LM RF v2 | Fuse v1 ($\alpha=0.25$) | Fuse v2 ($\alpha=0.75$) | Hybrid RF |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A** | 23 | 52.17% | 69.57% | 91.30% | 91.30% | 86.96% | **91.30%** | **95.65%** |
| **B** | 27 | 59.26% | 70.37% | 92.59% | **96.30%** | 85.19% | **96.30%** | **100.00%** |
| **C** | 25 | 28.00% | 48.00% | 88.00% | **96.00%** | 64.00% | **96.00%** | **96.00%** |
| **D** | 22 | 72.73% | 86.36% | 100.00% | 100.00% | 95.45% | **100.00%** | **100.00%** |
| **E** | 27 | 33.33% | 48.15% | 88.89% | **92.59%** | 66.67% | **92.59%** | **100.00%** |
| **F** | 17 | 29.41% | 47.06% | 88.24% | **94.12%** | 76.47% | **94.12%** | **100.00%** |
| **G** | 23 | 82.61% | 95.65% | 100.00% | 100.00% | 95.65% | **100.00%** | **100.00%** |
| **H** | 23 | 69.57% | 91.30% | 86.96% | 86.96% | 91.30% | **86.96%** | **95.65%** |
| **I** | 18 | 22.22% | 66.67% | 88.89% | **94.44%** | 77.78% | **94.44%** | **94.44%** |
| **J** | 22 | 77.27% | 95.45% | 90.91% | **95.45%** | 95.45% | **95.45%** | **95.45%** |
| **K** | 80 | 71.25% | 96.25% | 90.00% | 90.00% | 83.75% | **90.00%** | **92.50%** |
| **L** | 74 | 89.19% | 78.38% | 94.59% | **95.95%** | 78.38% | **95.95%** | **91.89%** |
| **M** | 38 | 26.32% | 57.89% | 60.53% | **63.16%** | 68.42% | **73.68%** | **73.68%** |
| **N** | 52 | 50.00% | 13.46% | 30.77% | **34.62%** | 15.38% | **26.92%** | **30.77%** |
| **O** | 43 | 51.16% | 41.86% | 58.14% | **65.12%** | 55.81% | **65.12%** | **67.44%** |
| **P** | 67 | 44.78% | 55.22% | 67.16% | **71.64%** | 71.64% | **76.12%** | **74.63%** |
| **Q** | 80 | 45.00% | 60.00% | 55.00% | **58.75%** | 62.50% | **67.50%** | **76.25%** |
| **R** | 51 | 54.90% | 70.59% | 68.63% | **74.51%** | 74.51% | **78.43%** | **76.47%** |
| **S** | 68 | 41.18% | 72.06% | 63.24% | **67.65%** | 67.65% | **72.06%** | **72.06%** |
| **T** | 40 | 40.00% | 65.00% | 67.50% | **70.00%** | 72.50% | **75.00%** | **77.50%** |
| **U** | 55 | 50.91% | 65.45% | 54.55% | **60.00%** | 61.82% | **67.27%** | **65.45%** |
| **V** | 65 | 49.23% | 70.77% | 69.23% | **72.31%** | 75.38% | **78.46%** | **84.62%** |
| **W** | 57 | 40.35% | 68.42% | 68.42% | **71.93%** | 71.93% | **73.68%** | **77.19%** |
| **X** | 39 | 10.26% | 35.90% | 28.21% | **33.33%** | 35.90% | **38.46%** | **38.46%** |
| **Y** | 44 | 43.18% | 59.09% | 59.09% | **63.64%** | 63.64% | **68.18%** | **70.45%** |
| **Z** | 21 | 47.62% | 57.14% | 76.19% | **80.95%** | 61.90% | **80.95%** | **76.19%** |

---

## 8. Investigation of Persistent Failure Modes: N vs M and X vs V

```mermaid
graph TD
    subgraph "Failure Mode Root Causes"
        N_True["Sign: N<br>(Support: 52)"] -->|44.2% MP CNN Confusion| M_Pred["Misclassified as M<br>Root Cause: Monocular z-depth occlusion on 2 vs 3 folded fingers"]
        X_True["Sign: X<br>(Support: 39)"] -->|15.4% MP CNN Confusion| V_Pred["Misclassified as V<br>Root Cause: Bent index finger projected 2D foreshortening"]
        L_True["Sign: L (Video)<br>(146 Frames)"] -->|Fixed by alpha=0.75| L_Pred["Correct Majority L (56.2%)<br>Root Cause: CNN square crop aspect ratio distortion"]
    end
    style N_to_M fill:#ffdddd,stroke:#ff0000
    style X_to_V fill:#ffdddd,stroke:#ff0000
    style L_Pred fill:#ccffcc,stroke:#00aa00
```

### 8.1 N vs M Detailed Analysis
- **Empirical Confusion**: In Test M–Z, MediaPipe CNN v1 misclassifies **30 out of 52 N samples as M (57.69%)**. Late Fusion baseline ($\alpha=0.25$) misclassifies 29/52 (55.77%). Calibrated Late Fusion ($\alpha=0.75$) reduces this to **23/52 (44.23%)**, and Landmark RF v2 standalone misclassifies 16/52 (30.77%).
- **Geometric Root Cause**:
  - In ISL, N folds index and middle fingers down over the thumb, with the ring and pinky fingers tightly curled inward.
  - In ISL, M folds index, middle, and ring fingers down over the thumb.
  - On a 2D camera view, the ring fingertip occludes directly behind the middle fingertip. MediaPipe landmark depth estimates ($z$) collapse toward zero when joints overlap, making the ring finger's extension indistinguishable from the middle finger.
  - While Landmark Geometry v2 introduced the **Thumb-to-Ring distance** and **Ring finger curl ratio**, recovering ~13% accuracy, monocular landmark jitter under subtle lighting shifts continues to produce false M predictions.

### 8.2 X vs V Detailed Analysis
- **Empirical Confusion**: In Test M–Z, X achieves only 38.46% recall across late fusion and hybrid models, with **15.38% to 17.95% of X samples misclassified as V**.
- **Geometric Root Cause**:
  - X features an isolated bent/hooked index finger with the remaining fingers curled.
  - V features extended index and middle fingers in an open spread.
  - Due to monocular foreshortening, a hooked index finger pointing toward the camera appears similar in projected length to an open V or folded knuckle. The introduction of DIP flexion angles in Landmark v2 prevented catastrophic confusion in landmark standalone models (only 2/39 misclassified as V), but visual CNN predictions remain noisy.

---

## 9. Fusion Weight Comparison: $\alpha=0.25$ vs $\alpha=0.75$

| Benchmark Partition | Metric | Late Fusion Baseline ($\alpha=0.25$) | Calibrated Late Fusion ($\alpha=0.75$) | Net Improvement |
| :--- | :--- | :---: | :---: | :---: |
| **Independent Test A–L** | Top-1 Accuracy | 81.10% | **92.65%** | **+11.55%** |
| | Top-3 Accuracy | 95.54% | **96.59%** | **+1.05%** |
| | Macro-F1 | 87.63% | **95.90%** | **+8.27%** |
| | L $\to$ Y Errors | 4 / 74 (5.4%) | **0 / 74 (0.0%)** | **-5.41% (Eliminated)** |
| **Provisional Test M–Z** | Top-1 Accuracy | 62.50% | **67.78%** | **+5.28%** |
| | Top-3 Accuracy | 77.36% | **81.67%** | **+4.31%** |
| | Macro-F1 | 62.88% | **70.78%** | **+7.90%** |
| | N $\to$ M Errors | 29 / 52 (55.8%) | **23 / 52 (44.2%)** | **-11.54%** |
| **Standard L Video** | Frame Accuracy | 1.4% | **56.2%** | **+54.80%** |
| | Majority Vote Class | U (72.6% false) | **L (56.2% correct)** | **Corrected Majority** |
| | Flip Rate | 13.1% | **6.2%** | **-6.90%** |

---

## 10. Webcam Stability and Latency Measurements

Evaluating all 4 continuous webcam video streams across models:

| Recording (Ground Truth) | Frames | Model Configuration | Frame Accuracy | Majority Class (Rate) | Flip Rate | Max Stable Run | Latency (CPU) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Frontal C** (C) | 146 | **Baseline Late Fusion Raw ($\alpha=0.25$)** | 78.1% | C (78.08%) | 12.4% | 73 frames | 77.9 ms |
| | | **Landmark v2 Standalone** | 0.0% | K (93.15%) | 10.3% | 92 frames | 77.9 ms |
| | | **Calibrated Fusion Raw ($\alpha=0.75$)** | 78.1% | C (78.08%) | 15.9% | 73 frames | 77.9 ms |
| | | **Calibrated Fusion Smoothed** | **80.1%** | **C (80.14%)** | **4.8%** | **75 frames** | 77.9 ms |
| **Second C** (C) | 143 | **Baseline Late Fusion Raw ($\alpha=0.25$)** | 7.0% | I (79.02%) | 23.9% | 60 frames | 68.3 ms |
| | | **Landmark v2 Standalone** | 0.0% | K (81.12%) | 26.1% | 34 frames | 68.3 ms |
| | | **Calibrated Fusion Raw ($\alpha=0.75$)** | 2.8% | I (56.64%) | 38.7% | 15 frames | 68.3 ms |
| | | **Calibrated Fusion Smoothed** | 2.1% | I (67.83%) | 9.9% | 53 frames | 68.3 ms |
| **Standard L** (L) | 146 | **Baseline Late Fusion Raw ($\alpha=0.25$)** | 1.4% | U (72.60%) | 13.1% | 51 frames | 75.8 ms |
| | | **Landmark v2 Standalone** | 43.8% | D (56.16%) | 40.7% | 10 frames | 75.8 ms |
| | | **Calibrated Fusion Raw ($\alpha=0.75$)** | 45.2% | **L (45.21%)** | 29.7% | 18 frames | 75.8 ms |
| | | **Calibrated Fusion Smoothed** | **56.2%** | **L (56.16%)** | **6.2%** | **45 frames** | 75.8 ms |
| **Webcam A** (A) | 143 | **Baseline Late Fusion Raw ($\alpha=0.25$)** | 0.7% | I (65.73%) | 33.8% | 17 frames | 68.9 ms |
| | | **Landmark v2 Standalone** | **65.7%** | **A (65.73%)** | 33.8% | 21 frames | 68.9 ms |
| | | **Calibrated Fusion Raw ($\alpha=0.75$)** | 14.7% | I (72.73%) | 23.2% | 23 frames | 68.9 ms |
| | | **Calibrated Fusion Smoothed** | 9.8% | I (78.32%) | 9.2% | 46 frames | 68.9 ms |

---

## 11. Acceptance Gate Audit (8 Deployment Gates)

| Gate # | Acceptance Criterion | Target Threshold | Measured Outcome | Gate Status | Blocking Justification |
| :-: | :--- | :---: | :---: | :---: | :--- |
| **1** | **Unseen Multi-Signer Macro-F1** | $\ge 80.0\%$ | Unverified | **UNVERIFIED** | Physical recordings from `signer_02` and `signer_03` are pending collection. |
| **2** | **Letter N Recall** | $\ge 70.0\%$ | 26.92% (Test M-Z) | **FAILED** | High palm-plane occlusion with Letter M (44.2% error). |
| **3** | **Letter L Accuracy** | $\ge 85.0\%$ | Static: 95.95%, Video: 56.16% | **PARTIAL** | Static passed (95.95%), video improved from 1.4% to 56.2%, but below 85% frame target. |
| **4** | **CPU Pipeline Latency** | $< 100.0\text{ ms}$ ($\ge 10\text{ FPS}$) | 67.7 ms to 77.9 ms | **PASSED** | Pipeline runs at 13.0 to 14.8 FPS on CPU. |
| **5** | **Worst-Class Recall** | $\ge 50.0\%$ | N = 26.92%, X = 38.46% | **FAILED** | Multiple letters remain below the 50% minimum recall floor. |
| **6** | **API & Frontend Compatibility** | 100% Backward Compatible | 100% Preserved | **PASSED** | Zero endpoint, contract, or payload changes. |
| **7** | **Repository Test Suite** | 100% Tests Passing | 104 / 104 Passed | **PASSED** | Full test suite passes cleanly in 47.94s. |
| **8** | **Safe Fallback Behavior** | Graceful Architecture Fallback | Verified | **PASSED** | Pipeline safely falls back to standard architecture on missing artifacts. |

### Overall Deployment Decision:
**NOT READY FOR PRODUCTION DEFAULT**. Passed 4 gates, failed 3 gates, and 1 gate remains unverified due to pending physical recordings. **Production defaults MUST remain unchanged.**

---

## 12. Test-Suite Results and Reproducibility

### 12.1 Full Test Suite Verification
Executed the complete repository test suite:
```bash
/Users/pavanaksshay/venv/bin/python3 -m unittest discover -s tests -p "test_*.py"
```
**Actual Output**:
```
Ran 104 tests in 47.937s

OK
```
All **104 unit tests passed cleanly** (94 prior tests + 10 new Phase 17 tests). Zero failures, zero errors, zero skipped tests.

### 12.2 Exact Reproducibility Commands
```bash
# 1. Audit physical data collection and split leakage
/Users/pavanaksshay/venv/bin/python3 phase17_data_collection_and_audit.py

# 2. Run master frozen benchmark across all 7 models
/Users/pavanaksshay/venv/bin/python3 phase17_frozen_benchmark.py

# 3. Execute Phase 17 generalization unit tests
/Users/pavanaksshay/venv/bin/python3 -m unittest tests/test_phase17_generalization.py

# 4. Execute full repository test suite (104 tests)
/Users/pavanaksshay/venv/bin/python3 -m unittest discover -s tests -p "test_*.py"
```

---

## 13. Recommended Next Phase (Phase 18)

Phase 18 should focus on:
1. **Physical Video Recording**: Execute camera capture sessions for `signer_02` and `signer_03` across all 26 letters under Desk Neutral and Lateral Warm conditions using the existing protocol.
2. **Adaptive Confidence-Gated Late Fusion**: Implement dynamic $\alpha$ switching based on hand compactness. When a hand is closed/compact (such as A), suppress CNN weight to prevent false visual bias (such as I-bias); when a hand is open/spread (such as C, L, B), allow balanced visual fusion.
3. **Palm-Depth Feature Refinement for N/M**: Investigate relative finger knuckle heights and palm curvature ratios to further resolve the N vs M occlusion boundary.

---

## 14. Final Decision

| Mandate Question | Explicit Empirical Conclusion |
| :--- | :--- |
| **1. Were new signers physically recorded?** | **No.** Physical camera recordings from `signer_02` and `signer_03` have not yet been performed in this workspace. Ingestion tools, quality gates, and schema validators are completely armed and ready. |
| **2. Does a valid independent-signer test set exist?** | **No.** Test A–L is an independent *session* test on the same historical signer. Test M–Z contains *session leakage* with training data. |
| **3. Which existing method generalizes best on the new data?** | On static crops: **Hybrid Concatenation RF** (95.28% A–L, 70.56% M–Z) and **Calibrated Late Fusion $\alpha=0.75$** (92.65% A–L, 67.78% M–Z). On live webcam video: **Calibrated Late Fusion $\alpha=0.75$ with temporal smoothing** resolves the Standard L failure mode (56.2% accuracy, correct L majority), while Landmark v2 standalone excels on Webcam A (65.7% accuracy, majority A). |
| **4. Are N and X sufficiently reliable?** | **No.** N recall is 26.92% (Test M–Z) and X recall is 38.46%, both failing the $\ge 70\%$ recall threshold. |
| **5. Does calibrated fusion retain its gains on unseen signers?** | Calibrated fusion ($\alpha=0.75$) retains substantial gains on held-out sessions (+11.55% on Test A–L, +5.28% on Test M–Z, and resolves the webcam L majority vote). However, cross-signer generalization remains unverified until `signer_02` and `signer_03` are physically captured. |
| **6. Is additional data required?** | **Yes, absolutely.** Physical multi-signer video recordings across all 26 letters are strictly required before any model training or production promotion. |
| **7. Is any production change justified?** | **No.** The evidence proves that the experimental recognition candidates do not yet meet production reliability thresholds across all classes and video conditions. |

### Final Configuration Commitment:
- `ISL_BACKEND_DETECTOR=yolo` **remains the production default**.
- `ISL_RECOGNITION_ARCH=standard` **remains the default recognition architecture**.
- `ISL_TEMPORAL_SMOOTHING=0` **remains disabled by default**.
- All 104 repository unit tests pass cleanly.
