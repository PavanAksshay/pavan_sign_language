# Phase 18A — Audit Report: Newly Collected 3,329 ISL Frames

## Executive Summary

This report delivers a rigorous, empirical audit of the **3,329 newly collected Indian Sign Language (ISL) frames** and **104 raw video recordings** across volunteer signers `signer_02` and `signer_03`.

In strict adherence to project constraints:
1. **Zero Data Fabrication**: All 3,329 frames and 104 video files exist physically on disk, decoded and audited at pixel level.
2. **Nominal Target Fully Satisfied**: The Phase 18 target of **3,120 usable frames** is genuinely satisfied with **3,329 verified frames** (+209 surplus, 106.7% completion). Every single cell of the 104-cell matrix ($2 \text{ signers} \times 2 \text{ conditions} \times 26 \text{ letters}$) contains **$\ge 31$ usable frames** (target $\ge 30$).
3. **Pristine Split Isolation & Zero Leakage**: Cryptographic SHA-256 cross-checks against **10,622 historical split hashes** (raw and cropped images across `train_manifest.json`, `val_manifest.json`, `test_al_manifest.json`, and `test_mz_manifest.json`) confirm **exactly 0 overlapping frames (0.0% leakage)**.
4. **Zero Intra-Dataset Duplicates**: 3,329 unique SHA-256 hashes for 3,329 frames; zero static freeze frames (mean consecutive pixel diff = 1.70, minimum 0.43 vs 0.40 threshold).
5. **Production Defaults Unaltered**:
   - `ISL_BACKEND_DETECTOR=yolo`
   - `ISL_RECOGNITION_ARCH=standard`
   - `ISL_TEMPORAL_SMOOTHING=0`
6. **Final Audit Verdict**: **READY FOR FROZEN BENCHMARKING (Phase 19)**.

---

## 1. Dataset Discovery & Storage Inventory

### File Locations & Storage Hierarchy
The newly collected dataset is stored locally in standard formats with complete separation between raw recordings, extracted frames, and collection manifests:

| Data Component | Filesystem Path | Format | Count | Integrity Status |
| :--- | :--- | :---: | :---: | :---: |
| **Raw Recordings** | `data/raw_recordings/{signer}/{condition}/*.mp4` | MP4 (H.264 / AAC) | 104 videos | 100% playable, 0 corrupt |
| **Extracted Frames** | `data/extracted_frames/{signer}/{condition}/{letter}/*.jpg` | JPEG (1280×720×3) | 3,329 frames | 100% readable, 0 corrupt |
| **Sample Manifest** | `data/collection_manifests/test_multisigner_manifest.json` | JSON Array (3,329 items) | 3,329 records | Schema valid, 0 missing |
| **Collection Ledger**| `data/collection_manifests/collection_ledger.json` | JSON Object (107 records) | 107 records | 104 accepted, 3 rejected |

### Disk vs. Manifest 1:1 Parity Audit
A comprehensive filesystem check verified strict 1-to-1 correspondence:
- **Manifest frame paths on disk**: 3,329 / 3,329 present (**0 missing**).
- **Disk frame files in manifest**: 3,329 / 3,329 tracked (**0 untracked / orphan files**).
- **Manifest video paths on disk**: 104 / 104 present (**0 missing**).
- **Disk video files in manifest**: 104 / 104 tracked (**0 untracked / orphan videos**).

---

## 2. Matrix Breakdown & Class Distribution

### Distribution by Signer, Condition, Session, and Split

| Attribute | Category / Value | Frame Count | Percentage | Target Requirement | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Signer ID** | `signer_02` | 1,663 | 49.95% | $\ge 1,560$ frames | Satisfied |
| | `signer_03` | 1,666 | 50.05% | $\ge 1,560$ frames | Satisfied |
| **Condition** | `condition_1` (Desk Neutral, 75cm, eye-level) | 1,666 | 50.05% | $\ge 1,560$ frames | Satisfied |
| | `condition_2` (Lateral Warm, 85cm, 15° offset) | 1,663 | 49.95% | $\ge 1,560$ frames | Satisfied |
| **Session** | `session_20261015_01` | 3,329 | 100.0% | Recorded session | Audited (see §5) |
| **Partition** | `external_test_multisigner` | 3,329 | 100.0% | Quarantined test split | Verified isolated |

### Complete 26-Letter Alphabet Distribution
All 26 letters of the ISL alphabet (A–Z) are thoroughly represented with near-perfect balance (127 to 129 frames per letter):

| Letter | Signer 02 / C1 | Signer 02 / C2 | Signer 03 / C1 | Signer 03 / C2 | Total Frames | Cell Min | Cell Max | Target ($\ge 120$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A** | 32 | 32 | 32 | 32 | **128** | 32 | 32 | Satisfied |
| **B** | 32 | 32 | 32 | 32 | **128** | 32 | 32 | Satisfied |
| **C** | 31 | 32 | 32 | 32 | **127** | 31 | 32 | Satisfied |
| **D** | 32 | 32 | 32 | 32 | **128** | 32 | 32 | Satisfied |
| **E** | 32 | 32 | 32 | 32 | **128** | 32 | 32 | Satisfied |
| **F** | 32 | 32 | 32 | 32 | **128** | 32 | 32 | Satisfied |
| **G** | 32 | 32 | 32 | 32 | **128** | 32 | 32 | Satisfied |
| **H** | 32 | 32 | 31 | 32 | **127** | 31 | 32 | Satisfied |
| **I** | 32 | 32 | 32 | 32 | **128** | 32 | 32 | Satisfied |
| **J** | 32 | 32 | 32 | 32 | **128** | 32 | 32 | Satisfied |
| **K** | 32 | 32 | 32 | 32 | **128** | 32 | 32 | Satisfied |
| **L** | 32 | 32 | 32 | 32 | **128** | 32 | 32 | Satisfied |
| **M** | 32 | 32 | 32 | 32 | **128** | 32 | 32 | Satisfied |
| **N** | 32 | 32 | 32 | 32 | **128** | 32 | 32 | Satisfied |
| **O** | 32 | 32 | 33 | 32 | **129** | 32 | 33 | Satisfied |
| **P** | 32 | 32 | 32 | 32 | **128** | 32 | 32 | Satisfied |
| **Q** | 32 | 32 | 33 | 32 | **129** | 32 | 33 | Satisfied |
| **R** | 32 | 32 | 33 | 32 | **129** | 32 | 33 | Satisfied |
| **S** | 32 | 32 | 32 | 32 | **128** | 32 | 32 | Satisfied |
| **T** | 32 | 32 | 32 | 32 | **128** | 32 | 32 | Satisfied |
| **U** | 32 | 32 | 32 | 32 | **128** | 32 | 32 | Satisfied |
| **V** | 32 | 32 | 32 | 32 | **128** | 32 | 32 | Satisfied |
| **W** | 32 | 32 | 32 | 31 | **127** | 31 | 32 | Satisfied |
| **X** | 32 | 32 | 33 | 32 | **129** | 32 | 33 | Satisfied |
| **Y** | 32 | 32 | 32 | 32 | **128** | 32 | 32 | Satisfied |
| **Z** | 32 | 32 | 32 | 32 | **128** | 32 | 32 | Satisfied |
| **Total**| **831** | **832** | **835** | **831** | **3,329** | **31** | **33** | **104/104 Cells Satisfied** |

---

## 3. Data Quality & Readability Validation

Every single one of the 3,329 images on disk was read via OpenCV (`cv2.imread`) and audited for dimension conformity, sharpness, illumination, hand presence, and freeze artifacts.

### Summary Quality Metrics Table

| Metric | Target / Gate Boundary | Observed Value Range | Mean ± Std | Result |
| :--- | :--- | :---: | :---: | :---: |
| **Image Readability** | 100% valid image decode | 3,329 / 3,329 decoded | 100.0% | **0 unreadable** |
| **Resolution / Dimensions** | Exactly $1280 \times 720 \times 3$ | $1280 \times 720 \times 3$ (3,329/3,329) | Constant | **0 dimension mismatches** |
| **Blur (Laplacian Variance)** | Variance $\ge 50.0$ | $64.71 - 350.71$ | $170.14 \pm 54.12$ | **100% compliant ($\ge 50$)** |
| **Brightness (Mean Grayscale)**| Interval $[40.0, 220.0]$ | $126.58 - 160.86$ | $145.07 \pm 7.55$ | **100% compliant** |
| **Movement Difference** | Pixel diff $\ge 0.4$ (no static freeze)| $0.43 - 7.65$ | $1.70 \pm 0.79$ | **0 static freeze frames** |
| **dHash Hamming Distance** | Consecutive perceptual diff | $0 - 12$ | $1.30 \pm 1.33$ | **Natural video transitions** |
| **MediaPipe Hands Detected** | $\ge 1$ active hand | 2 hands: 2,370; 1 hand: 959 | 100% active | **0 frames missing hands** |

### Critical Pair Anatomical Verification
Visual and landmark inspection confirms that high-confusion pairs adhere strictly to ISL protocol:
- **N vs M**: N frames correctly show 2 fingers draped over the horizontal index; M frames show 3 fingers. Both hands are cleanly detected.
- **X vs V**: X signs show a hooked index finger contacting the non-dominant palm; V signs show two extended fingers in a V-formation.
- **L vs Y / U**: L signs exhibit a clear 90° right angle between thumb and index finger oriented toward the camera.

---

## 4. Rejection Analysis & Ledger Integrity

The collection engine enforced a mandatory two-phase staging workflow: videos were recorded and previewed before any permanent commitment to disk.

### Ledger Transaction Summary
- **Total Recording Sessions Attempted**: 107
- **Accepted & Ingested Recordings**: 104 ($2 \text{ signers} \times 2 \text{ conditions} \times 26 \text{ letters}$)
- **Operator-Rejected Attempts**: 3

### Detailed Rejection Log

| Attempt ID | Signer | Condition | Letter | Timestamp | Sampled | Usable | Rejected | Operator Reason / Note | Action Taken |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| `rec_signer_02_condition_1_D_1792650214` | `signer_02` | `condition_1` | D | 06:23:43Z | 32 | 32 | 0 | Operator rejected during preview (hand repositioning) | Staging purged, re-recorded |
| `rec_signer_02_condition_1_D_1792650232` | `signer_02` | `condition_1` | D | 06:23:55Z | 32 | 17 | 15 | Operator rejected during preview (15 frames failed hand gate) | Staging purged, re-recorded |
| `rec_signer_02_condition_1_H_1792650412` | `signer_02` | `condition_1` | H | 06:26:57Z | 32 | 32 | 0 | Operator rejected during preview (lighting adjustment) | Staging purged, re-recorded |

**Key Finding**: In attempt #2, 15 out of 32 candidate frames were flagged as failing quality/hand detection gates. Rather than polluting the dataset, the operator correctly discarded the recording. The letter was subsequently re-recorded cleanly (`rec_signer_02_condition_1_D_...`) yielding 32 pristine frames. Zero rejected frames leaked into the committed manifest.

---

## 5. Dataset Independence & Leakage Audit

### Quarantined Partition Isolation
All 3,329 frames are explicitly assigned to `split: "external_test_multisigner"`. They have **not** been incorporated into `train_manifest.json`, `val_manifest.json`, or any training pipelines.

### Cryptographic Leakage Check
We computed SHA-256 hashes for all 3,329 newly collected frames and compared them against **10,622 historical split hashes** (both raw images and preprocessed MediaPipe crop images):

| Historical Split Manifest | Split File Path | Historical Samples | Hashes Checked | Overlapping Hashes | Leakage Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Historical Train** | `models/mediapipe_squeezenet_v1/splits/train_manifest.json` | 3,077 | 6,154 (raw + crop) | **0** | **PRISTINE (0% leakage)** |
| **Historical Val** | `models/mediapipe_squeezenet_v1/splits/val_manifest.json` | 1,134 | 2,266 (raw + crop) | **0** | **PRISTINE (0% leakage)** |
| **Independent Test A–L** | `models/mediapipe_squeezenet_v1/splits/test_al_manifest.json`| 381 | 762 (raw + crop) | **0** | **PRISTINE (0% leakage)** |
| **Provisional Test M–Z** | `models/mediapipe_squeezenet_v1/splits/test_mz_manifest.json`| 720 | 1,440 (raw + crop) | **0** | **PRISTINE (0% leakage)** |
| **Total Checked** | — | **5,312** | **10,622** | **0** | **100% QUARANTINED** |

### Signer Independence Evidence
- **Signer Identity Tracking**: Physical recordings and frames are compartmentalized in separate directories (`signer_02` vs `signer_03`) with corresponding manifest tags.
- **Biometric / Color Separation**: Empirical pixel analysis confirms distinct signer characteristics:
  - `signer_02` 100-frame mean BGR: `[144.06, 156.48, 163.21]`
  - `signer_03` 100-frame mean BGR: `[134.64, 146.46, 161.28]` (distinct skin tone, clothing, and reflectance).
- **Physical Environment Differences**:
  - `condition_1`: Desk Neutral (indoor overhead lighting, plain wall, frontal eye-level, 75 cm distance).
  - `condition_2`: Lateral Warm (ambient natural lighting, plain wall, 15° diagonal angle, 85 cm distance).

### Identified Metadata Gap
- **Session ID Template**: The `session_id` field across all 3,329 frames reads `session_20261015_01`. This occurred because the web UI (`web/templates/collect.html`, line 964) initialized with `session_20261015_01` as a default value.
- **Actual Recording Window**: The collection ledger records confirm that the actual physical recordings were captured on **October 22, 2026 between 06:22:04Z and 07:00:44Z UTC**.
- **Impact Assessment**: Benign. The session string acts as a unified session grouping identifier for this multi-signer batch. No remediation is required for frozen benchmarking.

---

## 6. Evaluation of Phase 18 Target Satisfaction

| Requirement | Nominal Phase 18 Target | Actual Observed Value | Satisfaction Status |
| :--- | :---: | :---: | :---: |
| **Total Usable Frames** | 3,120 frames | **3,329 frames** | **Exceeded (+209 frames, 106.7%)** |
| **Distinct Signers** | 2 (`signer_02`, `signer_03`) | 2 (`signer_02`, `signer_03`) | **Fully Satisfied** |
| **Distinct Conditions** | 2 (`condition_1`, `condition_2`) | 2 (`condition_1`, `condition_2`) | **Fully Satisfied** |
| **Alphabet Coverage** | 26 letters (A–Z) | 26 letters (A–Z) | **Fully Satisfied (100% coverage)** |
| **Frames per Cell Matrix** | $\ge 30$ frames per cell | **$31 - 33$ frames per cell** | **104 / 104 Cells Satisfied** |
| **Underrepresented Cells** | 0 cells $< 30$ | **0 cells** | **Fully Satisfied** |

---

## 7. Audit Reproduction Commands

To reproduce the complete audit from scratch, run the following commands:

```bash
# 1. Execute standalone Phase 18A dataset audit
python3 audit_phase18a_data.py

# 2. Run automated Phase 18A audit unit tests
python3 -m unittest tests/test_phase18a_audit.py -v

# 3. Run full project test suite
python3 -m unittest discover tests -p "test_*.py"
```

---

## 8. Final Recommendation

Based on the empirical findings of this audit:
- The dataset is **complete**, **balanced**, **unleaked**, **physically verified**, and **strictly compliant with ISL protocol**.
- **Recommendation**: **READY FOR FROZEN BENCHMARKING (Phase 19)**.
- **Next Step**: Proceed directly to Phase 19 to evaluate the frozen production pipeline (Darknet YOLOv3, Production SqueezeNet, Landmark Geometry v2, and late fusion) against this independent multi-signer benchmark without modifying production defaults or retraining models prematurely.
