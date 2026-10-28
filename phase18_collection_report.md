# Phase 18 — ISL Alphabet Data Collection Interface & Benchmark Infrastructure Report

## Executive Summary

**Phase 18** delivers a complete, local, privacy-preserving webcam data collection studio for recording genuine Indian Sign Language (ISL) alphabet examples from two independent volunteer signers (`signer_02` and `signer_03`).

In strict compliance with project constraints:
1. **Zero Data Fabrication**: No synthetic frames, downloaded images, or fabricated labels have been generated. The actual accepted frame count is honestly reported as **0 / 3,120 frames**, reflecting that physical volunteer capture sessions remain pending.
2. **Production Defaults Preserved**:
   - `ISL_BACKEND_DETECTOR=yolo` (Darknet YOLOv3 default)
   - `ISL_RECOGNITION_ARCH=standard` (Production SqueezeNet default)
   - `ISL_TEMPORAL_SMOOTHING=0` (Disabled by default)
3. **Quarantined Split Isolation**: All newly collected data is strictly isolated into an external test holdout partition (`data/collection_manifests/test_multisigner_manifest.json`) and never leaks into `train_manifest.json` or `val_manifest.json`.
4. **Complete Local Privacy & Right to Deletion**: 100% of video ingestion, frame extraction, and landmark auditing occurs locally on the workstation. No data is sent to external services, and volunteers retain full right to deletion at any time.
5. **Comprehensive Test Suite**: All **115 repository unit tests** pass cleanly (including 11 new Phase 18 collection engine tests).

---

## 1. Files Inspected, Created, and Modified

### Inspected
- `phase14b_analysis/metadata_schema.json`: Standard benchmark sample metadata schema.
- `phase14c_data_ingestion.py`: Preprocessing, duplicate detection, and quality audit engine.
- `phase16_recording_protocol.py`: Physical recording specifications for signers 02 and 03.
- `phase17_report.md`: External generalization findings and acceptance criteria.
- `web/static/manual/`: 26 alphabet ground-truth reference images (`A.jpg` to `Z.jpg`).
- `pipeline.py`: Production inference pipeline and configuration parameters.

### Created
1. `phase18_collector.py`: Core backend engine implementing video ingestion, configurable sampling, duplicate detection, quality gating, two-phase staging, ledger management, and split-isolation audits.
2. `web/templates/collect.html`: Modern, responsive capture interface featuring live webcam stream, framing reticle, countdown overlay, active letter guidance, quality review modal, thumbnail gallery, and A–Z progress matrix.
3. `docs/phase18_recording_guide.md`: Comprehensive operator and volunteer manual detailing lighting conditions, camera positions, letter-by-letter ISL instructions (highlighting N vs M, X vs V, L), and consent procedures.
4. `tests/test_phase18_collection.py`: 11 automated unit and integration tests verifying schema validation, duplicate hash rejection, blur/brightness gates, consent enforcement, staging lifecycle, deletion, and split isolation.
5. `phase18_collection_report.md`: This technical report.

### Modified
1. `server.py`: Registered collection studio routes (`/collect`, `/api/collect/progress`, `/api/collect/stage`, `/api/collect/commit`, `/api/collect/recording/{id}`, `/api/collect/split-audit`) without altering existing inference endpoints.
2. `web/templates/index.html`: Added a navigation button linking to `/collect` in the header.

---

## 2. Actual Data Inventory & Collection Status

| Partition / Split | Signer ID | Sessions / Source | Frame Count | Class Coverage | Partition Role |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Historical Train** | `historical_signer_01` | May 2020 Sessions | 3,077 | A–Z (26 classes) | Model Training |
| **Historical Val** | `historical_signer_01` | May 2020 Sessions | 1,134 | A–Z (26 classes) | Validation / Tuning |
| **Independent Test A–L** | `historical_signer_01` | May 14, 2020 Held-Out | 381 | A–L (12 classes) | Held-Out Session Benchmark |
| **Provisional Test M–Z** | `historical_signer_01` | May 16, 2020 Shared Session | 720 | M–Z (14 classes) | Provisional Benchmark |
| **Diagnostic Webcam** | `webcam_signer_01` | October 2026 Recordings | 662 | A, C, L | Video Stability Benchmark |
| **Phase 18 Multi-Signer** | `signer_02`, `signer_03` | Pending Physical Capture | **0** | A–Z (Target: 3,120) | **Quarantined External Test** |

### Collection Target vs Current Actuals
* **Target Signers**: 2 (`signer_02`, `signer_03`)
* **Target Conditions**: 2 (`condition_1`: Desk Neutral, `condition_2`: Lateral Warm)
* **Target Letters**: 26 (A–Z)
* **Nominal Frames per Letter/Condition**: 30 frames
* **Total Nominal Target**: **3,120 frames**
* **Actual Accepted Frames**: **0 frames (0.0% complete)**
* **Actual Rejected Attempts**: **0**

*Integrity Confirmation: The capture tooling and storage directories are fully deployed and verified. In adherence to zero-fabrication constraints, 0 frames are claimed until human volunteers physically sit in front of the webcam and complete recording sessions.*

---

## 3. Capture Interface & Review Workflow

### 1. Volunteer Consent Enforcement
* Before recording can be initiated, the operator must enable the **"Volunteer Consent Confirmed"** toggle.
* Recording buttons are locked until consent is confirmed.
* Clear notice is displayed confirming that all processing is 100% local, no cloud transmission occurs, and volunteers retain right to deletion.

### 2. Live Capture & Guidance Stage
* Real-time webcam preview with mirror inversion for natural signing.
* Framing reticle overlay guiding the volunteer to position their hand at chest/face level.
* Active letter hero display featuring the ground-truth reference image (`/static/manual/{letter}.jpg`) and specific anatomical instructions.
* High-visibility distinction guidance for high-confusion pairs:
  - **N vs M**: 2 fingers over thumb for N; 3 fingers for M.
  - **X vs V**: Single hooked index finger for X; extended V-shape for V.
  - **L vs C / Y**: 90° right angle between index and thumb facing camera.

### 3. Sampling & Quality Auditing
* **Timed Recording**: 3.0s recording duration with a 3-2-1 visual countdown.
* **Configurable Sampling**: Default is every 3 frames (~10 fps at 30 fps), producing ~30 candidate frames per recording.
* **Static Freeze Rejection**: Measures grayscale pixel difference between consecutive frames ($\ge 0.4$ required) to prevent storing frozen webcam frames.
* **Cryptographic Hash Tracking**: SHA-256 hash computed for every frame to reject duplicates against previous sessions and existing datasets.
* **Laplacian Blur Audit**: Variance $\ge 50.0$ required.
* **Illumination Bounds**: Mean grayscale brightness $\in [40.0, 220.0]$ required.
* **Landmark Detection**: MediaPipe HandLandmarker with lap-hand suppression ($\ge 1$ active hand required).

### 4. Interactive Review Modal (Preview Before Ingest)
* Video playback element allows the operator to review the captured sign.
* Summary cards show total frames, usable frames, and rejected frames with explicit reasons (e.g. `motion_blur`, `underexposed`, `no_hand_detected`).
* Scrollable thumbnail gallery displays every candidate frame with pass/fail badges.
* Two explicit actions:
  - **[Accept Recording & Ingest Frames]**: Saves the raw video and usable frames to disk, appends metadata to the test manifest, logs the transaction in the ledger, and advances to the next letter.
  - **[Discard & Re-Record]**: Completely purges the staged video and candidate frames from disk, recording the rejection reason in the ledger.

---

## 4. Storage Architecture & Split Isolation

### Directory Structure
```
data/
├── raw_recordings/
│   ├── signer_02/
│   │   ├── condition_1/
│   │   └── condition_2/
│   └── signer_03/
├── extracted_frames/
│   ├── signer_02/
│   │   ├── condition_1/
│   │   │   ├── A/
│   │   │   └── ...
├── collection_manifests/
│   ├── collection_ledger.json           # Permanent transaction ledger
│   └── test_multisigner_manifest.json   # External test partition manifest
└── staging/                             # Ephemeral staging directories
```

### Partition Quarantine Guarantee
* The training and validation manifests (`models/mediapipe_squeezenet_v1/splits/train_manifest.json` and `val_manifest.json`) are **strictly read-only** during data collection.
* All accepted samples are assigned partition tag `"external_test_multisigner"`.
* The automated audit method `collection_engine.audit_split_isolation()` verifies:
  1. `signer_02` and `signer_03` NEVER appear in training or validation manifests.
  2. Zero hash overlap exists between `test_multisigner_manifest.json` and training/validation sets.
  3. Every recording is treated as an indivisible unit (adjacent frames are never split across partitions).

---

## 5. Voluntary Participant Right to Deletion

To comply with research ethics and participant privacy standards:
* Any recording can be permanently purged via the API:
  ```http
  DELETE /api/collect/recording/{recording_id}
  ```
* Calling this endpoint:
  1. Deletes the raw video file from `data/raw_recordings/`.
  2. Deletes all extracted frames from `data/extracted_frames/`.
  3. Removes all sample entries from `test_multisigner_manifest.json`.
  4. Marks the ledger transaction status as `"deleted"`.

---

## 6. Test Suite & Verification Results

All 115 tests in the repository pass cleanly:

```
Ran 115 tests in 24.031s
OK
```

### Test Suite Breakdown
* `test_phase18_collection.py`: **11 tests** (Schema validation, duplicate hashes, blur/brightness gates, consent enforcement, short video rejection, staging lifecycle, rejection lifecycle, split isolation, deletion, progress matrix).
* `test_phase17_generalization.py`: **10 tests** (Multi-signer metadata schema, split leakage prevention, frozen benchmark accuracy accounting, production defaults preservation).
* `test_phase16_geometry_and_fusion.py`: **11 tests** (Geometry v2 264 dims, RF v2, calibrated late fusion $\alpha=0.75$).
* `test_phase15_benchmarking.py`: **11 tests** (Inventory audit, quality thresholds, multi-signer splits, frozen model comparison).
* `test_multi_signer_ingestion.py`: **6 tests** (Quality gates, blur, brightness, duplicate rejection, split leakage).
* `test_multi_signer_protocol.py`: **4 tests** (Collection matrix, video format, metadata schema).
* `test_late_fusion_pipeline.py`: **4 tests** (Late fusion prediction, alpha weighting, confidence aggregation).
* `test_temporal_smoothing.py`: **5 tests** (Rolling 5-frame buffer, majority voting, tie-breaking).
* `test_landmark_hand_selection.py`: **6 tests** (Lap-hand spatial suppression, active hand selection).
* `test_landmark_recognition.py`: **10 tests** (Landmark feature extraction, 200 dims, Random Forest classification).
* `test_backend_switching.py`: **7 tests** (YOLO vs MediaPipe detector, runtime switching, defaults).
* `test_mediapipe_squeezenet.py`: **5 tests** (MediaPipe fine-tuned SqueezeNet inference, input preprocessing).
* `test_inference_fixes.py`: **25 tests** (YOLO Darknet pipeline, two-hand merging, face suppression, bounding boxes).

---

## 7. How to Run the Capture Studio on macOS

### 1. Launch FastAPI Backend
```bash
cd /Users/pavanaksshay/pavan_sign_language
/Users/pavanaksshay/venv/bin/python3 server.py
```
*Server starts on `http://0.0.0.0:8050`.*

### 2. Open Studio in Browser
Open Chrome or Safari and navigate to:
```
http://localhost:8050/collect
```

### 3. Verification & Diagnostic Commands
```bash
# Check collection progress matrix
/Users/pavanaksshay/venv/bin/python3 -c "from phase18_collector import collection_engine; import json; print(json.dumps(collection_engine.get_progress(), indent=2))"

# Verify split isolation (zero leakage into train/val)
/Users/pavanaksshay/venv/bin/python3 -c "from phase18_collector import collection_engine; print(collection_engine.audit_split_isolation())"

# Run collection engine test suite
/Users/pavanaksshay/venv/bin/python3 -m unittest tests/test_phase18_collection.py

# Run complete repository test suite (115 tests)
/Users/pavanaksshay/venv/bin/python3 -m unittest discover tests/
```

---

## 8. Summary of Physical Next Actions

1. **Schedule Volunteer Sessions**:
   - Recruit Volunteer 1 (`signer_02`) and Volunteer 2 (`signer_03`).
2. **Execute Condition 1 (Desk Neutral)**:
   - Position volunteer at 75 cm from camera with overhead white light.
   - Record letters A through Z (3 seconds per letter).
   - Review and accept ~30 usable frames per letter (780 frames per signer).
3. **Execute Condition 2 (Lateral Warm)**:
   - Position camera at 15° offset, 85 cm distance with warm desk lamp.
   - Record letters A through Z (3 seconds per letter).
   - Review and accept ~30 usable frames per letter (780 frames per signer).
4. **Run Benchmark**:
   - Once the 3,120-frame matrix is collected, run `phase17_frozen_benchmark.py` on the new data to obtain the first genuine, definitive external generalization validation of the ISL alphabet recognizer.
