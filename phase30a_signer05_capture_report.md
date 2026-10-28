# Phase 30A — Signer 05 Independent Data Collection in Website Report

## Executive Summary

**Phase 30A** implements a dedicated, isolated data collection workflow on the Indian Sign Language (ISL) website studio for participant `signer_05` (**Volunteer D — Evaluation Only**). This work establishes the physical capture infrastructure needed for an independent benchmark evaluation holdout across all **26 letters (A–Z)** and both **Condition 1 (Desk Neutral, Frontal 75cm)** and **Condition 2 (Lateral Warm, 15° Offset 85cm)** without contaminating existing training splits or altering production models.

Strict Phase 30A boundaries were adhered to:
1. **Implementation Only**: The capture pipeline, frontend interface, APIs, schema, retake tracking, and isolation safeguards were implemented and verified. **Phase 30 evaluation was NOT executed.**
2. **Zero Synthetic / Placeholder Data**: No fabricated recordings, placeholder evaluation samples, or mock evaluation metrics were ingested into permanent manifests. The dataset readiness gate correctly reports `is_dataset_ready: false` (0/52 cells completed) awaiting real volunteer collection sessions.
3. **Strict Evaluation Isolation**: `signer_05` is quarantined from all model training, fine-tuning, calibration, and feature normalization pipelines.
4. **Preserved Production Baselines**: Production defaults (`ISL_BACKEND_DETECTOR=yolo`, `ISL_RECOGNITION_ARCH=standard`, `ISL_TEMPORAL_SMOOTHING=0`, `ISL_SHADOW_MODE=0`, `ISL_MEDIAPIPE_DELEGATE=cpu`, `ISL_MODEL_C_RESOLUTION=native`) and authoritative SHA-256 hashes for Model C and scalers remain completely unaltered.
5. **Full Repository Validation**: The complete test suite discovered and passed all **234 unit and integration tests** (100% pass rate).

---

## 1. Architectural Audit & Root-Cause Analysis

Prior to Phase 30A, `signer_05` could not be recorded or staged through the website. A comprehensive audit of the frontend, backend, schemas, and collector modules identified five specific bottlenecks:

| Component | Pre-Phase 30A State | Limitation / Root Cause | Resolution in Phase 30A |
| :--- | :--- | :--- | :--- |
| **`phase14b_analysis/metadata_schema.json`** | `signer_id.enum: ["signer_01", "signer_02", "signer_03", "signer_04"]` | Schema validation rejected any sample containing `signer_05`. | Updated enum to include `"signer_05"`. |
| **`phase18_collector.py` Signer Enums** | `ALLOWED_SIGNERS = ["signer_02", "signer_03", "signer_04"]` | Staging parameter validation threw `ValueError: Invalid signer_id`. | Added `"signer_05"` to `ALLOWED_SIGNERS` and `ALL_VALID_SIGNERS`. Declared `EVALUATION_ONLY_SIGNERS = ["signer_05"]`. |
| **Collector Frame Persistence** | `f"{signer_id}_{session_id}_{letter}_{idx:04d}.jpg"` | Repeating a letter within the same session silently overwrote existing frame files on disk. | Added collision detection: if destination filename exists, appends timestamped take tags `_t{rec_tag}_` so all physical frames remain intact for audit history. |
| **`server.py` Route Authorization** | Routes only allowed `signer_02`, `signer_03`, `signer_04`. | API rejected `signer_05` staging and lacked summary endpoint. | Allowed `signer_05` across `/collect` and `/api/collect/*`. Added `@app.get("/api/collect/summary/{signer_id}")`. |
| **`web/templates/collect.html`** | Dropdown lacked `signer_05`; no retake intercept; no isolation banner; no summary modal. | Signer was not selectable; retakes lacked warning; no visibility into 52-cell readiness. | Added Volunteer D option, isolation banner, retake modal, session completion summary modal, and updated JavaScript. |

---

## 2. Signer 05 Collection Workflow Architecture

### 2.1 Operator Interface & Interaction Flow

The data collection interface at `/collect` now provides an end-to-end studio workflow tailored for `signer_05`:

1. **Signer & Condition Selection**:
   - The volunteer dropdown provides `signer_05 (Volunteer D - Evaluation Only)`.
   - Selection dynamically reveals a high-visibility amber/emerald **Independent Evaluation Quarantine Banner**:
     > *"Signer 05 data is reserved strictly for independent model evaluation in Phase 30. These recordings will NEVER be used for training, fine-tuning, feature normalization, calibration, or model selection."*
2. **Alphabet Navigation & Reference Guidance**:
   - Operator selects any letter from the 26-tile A–Z matrix.
   - Reference photo (`/static/manual/{letter}.jpg`), signing instructions, and single/two-hand posture badges are displayed.
3. **Capture & 3.0s Auto-Stop**:
   - Volunteer consent confirmation toggle unlocks the recording trigger.
   - 3-second animated countdown overlay prepares the signer.
   - Real-time recording overlay pulses with live elapsed timer (`0.0s / 3.0s`).
4. **Retake Intercept & Confirmation Modal**:
   - If an active letter already possesses accepted frames, the primary recording button dynamically shifts from `Start Recording Letter` to `Retake Letter {letter}`.
   - Clicking opens the `#retake-modal` with explicit warning:
     > *"A recording for this letter and condition already exists. Initiating a retake will preserve prior recordings in the audit ledger while superseding them for final evaluation."*
5. **Quality Review Modal**:
   - Shows looping video playback, usable frame counts, blur/brightness quality scores, and sampled frame thumbnail gallery before committing.
   - Operator can either `Accept & Commit` or `Discard / Reject`.
6. **Session Completion Summary Modal**:
   - Accessed via the `#btn-session-summary` button in the header.
   - Shows completed matrix cells (`0 / 52`), usable frames (`0 / 1560`), superseded retake counts, and rejected attempt counts.
   - Visual 26-letter pill grids for Condition 1 and Condition 2 (green = completed, amber = partial, dim = missing).
   - Live readiness evaluation badge (`INCOMPLETE (CAPTURE IN PROGRESS)` vs `READY FOR EVALUATION`).

---

## 3. Quality & Manifest Integrity Architecture

### 3.1 Non-Destructive Frame Persistence & Overwrite Prevention

When retakes occur in the same session, prior implementations would overwrite existing files because sample IDs were formed deterministically from `(signer, session, letter, index)`. 

In Phase 30A, `commit_recording()` in [phase18_collector.py](file:///Users/pavanaksshay/pavan_sign_language/phase18_collector.py) implements collision detection:
```python
dest_frame_name = f"{sample_id}.jpg"
dest_frame_path = os.path.join(target_frames_dir, dest_frame_name)
if os.path.exists(dest_frame_path):
    rec_tag = recording_id.split("_")[-1]
    sample_id = f"{sample_id}_t{rec_tag}"
    dest_frame_name = f"{sample_id}.jpg"
    dest_frame_path = os.path.join(target_frames_dir, dest_frame_name)
```
This ensures that all historical frames remain physically accessible on disk for audit review.

### 3.2 Audit Trail & Manifest Supersession

When a retake is committed:
- **Ledger Audit**: Prior accepted records in `collection_ledger.json` have their status updated to:
  ```json
  "status": "superseded",
  "superseded_by": "rec_signer_05_condition_1_A_1792936474_a1b2",
  "superseded_at": "2026-10-25T14:10:00Z"
  ```
- **Manifest Updates**: Samples from prior takes in `test_multisigner_manifest.json` and `signer_05_manifest.json` are preserved but marked:
  ```json
  "quality": { "is_usable": false, "rejection_reason": "superseded_by_retake_..." },
  "is_superseded": true,
  "superseded_by": "rec_signer_05_condition_1_A_1792936474_a1b2",
  "eligible_for_evaluation": false
  ```
- **Progress Calculation**: `get_progress()` and `get_signer_summary()` query only samples where `quality.is_usable is True` and `is_superseded is False`. Superseded takes are audited separately.

### 3.3 Dedicated Signer 05 Manifest

In addition to the aggregate test manifest, `signer_05` captures are atomically written to `data/manifests/signer_05_manifest.json`. This dedicated manifest will serve as the exact evaluation set for Phase 30.

---

## 4. Evaluation Isolation Architecture & Safeguards

To prevent accidental data leakage into model training:

1. **Partition Tagging**: Every `signer_05` sample is stamped with:
   - `"split": "external_test_signer_05"`
   - `"dataset_role": "independent_evaluation_holdout"`
   - `"allow_training": false`
   - `"eligible_for_evaluation": true`
2. **Automated Split Isolation Verification**:
   - `collection_engine.audit_split_isolation()` verifies zero overlap between test signers/hashes and training/validation sets.
   - `verify_training_isolation_safeguard(manifest_path)` scans any candidate training/validation set and raises a `ValueError` if any sample from `EVALUATION_ONLY_SIGNERS` is detected.
3. **Freeze Verification**:
   - `train_manifest.json` SHA-256: `fc71e5940dcc12b7bf9c42d6ce125099c560d35e4f38b36486ac285f9c203b73` (0 signer_05 samples).
   - `val_manifest.json` SHA-256: `3a5a9cea0163a04baa4546749306cea3383cfd06b4ae2fa758f6995620b5514c` (0 signer_05 samples).
   - Production Model C Random Forest SHA-256: `1fab3349e2e537d8b31ddbe3bdd5c21e3ed005f85aaac9d3875c72217e28cc98` (unchanged).
   - Production Model C Scaler SHA-256: `93842932e7cfeb1d283e99f67a733fd487161d83cbff356d7458ced2c3de385d` (unchanged).

---

## 5. Test Suite Verification

A dedicated test suite was implemented in [tests/test_phase30a_signer05_capture.py](file:///Users/pavanaksshay/pavan_sign_language/tests/test_phase30a_signer05_capture.py) covering 11 critical test cases:

```
test_01_signer_05_constants_and_schema_authorization ... ok
test_02_alphabet_and_both_conditions_coverage ... ok
test_03_metadata_generation_and_role_isolation ... ok
test_04_dedicated_signer05_manifest_maintenance ... ok
test_05_unique_recording_ids_and_retake_supersession ... ok
test_06_corrupt_empty_and_interrupted_recording_handling ... ok
test_07_duplicate_detection_via_sha256 ... ok
test_08_progress_and_completion_calculation ... ok
test_09_training_data_isolation_safeguard ... ok
test_10_preservation_of_signer_02_03_04_behavior ... ok
test_11_preservation_of_production_defaults_and_artifact_hashes ... ok
----------------------------------------------------------------------
Ran 11 tests in 2.861s — OK
```

### Full Repository Suite Execution

The entire test suite was executed using:
```bash
/Users/pavanaksshay/venv/bin/python3 -m unittest discover tests -p "test_*.py"
```

**Results**:
- **Total Tests Discovered & Executed**: 234
- **Passed**: 234 (100%)
- **Failures**: 0
- **Errors**: 0
- **Execution Time**: 83.660 seconds

---

## 6. Manual Browser Verification Protocol

Automated integration tests validated staging, persistence, ledger auditing, and isolation using synthetic MP4 payloads. When physical camera capture is conducted with volunteer participant `signer_05`, verify the following steps:

1. **Launch Server**:
   Ensure backend server is running (`/Users/pavanaksshay/venv/bin/python3 server.py` on port 8050).
2. **Access Collection Studio**:
   Navigate to `http://localhost:8050/collect` in Google Chrome or Safari.
3. **Verify Signer 05 Option & Banner**:
   - Select `signer_05 (Volunteer D - Evaluation Only)` from the volunteer dropdown.
   - Confirm the amber **Independent Evaluation Dataset — Quarantined** banner appears with the `EVALUATION ONLY` badge.
4. **Confirm Volunteer Consent**:
   - Check the informed consent checkbox. Verify telemetry displays `Consent: Acknowledged ✓` and the record button becomes enabled.
5. **Initiate First Letter (Condition 1, Letter A)**:
   - Ensure Condition 1 (Desk Neutral) is selected.
   - Click `Start Recording Letter A (3.0s Auto-Stop)`.
   - Observe 3-second countdown, frame the hand in the reticle, and hold sign 'A'.
   - Confirm video automatically stops at 3.0s and opens `#review-modal`.
   - Check thumbnail preview cards, verify blur/brightness scores, and click `Accept & Commit Staged Recording`.
   - Confirm Letter A tile turns green (`status-completed` or `status-partial`) and active letter advances to B.
6. **Verify Retake Intercept**:
   - Click tile `A` again.
   - Confirm the recording button changes to `Retake Letter A (3.0s Auto-Stop)` with `data-is-retake="true"`.
   - Click the button; confirm `#retake-modal` appears with supersession warning.
   - Click `Confirm Retake & Record`, record a second take, and accept it.
   - Check that Letter A reflects the new take without error.
7. **Verify Session Summary Modal**:
   - Click `📊 Session Summary` in the top header.
   - Verify modal shows completed cells, usable frames, superseded takes count incremented, condition pill grids, and the `INCOMPLETE` readiness badge.

---

## 7. Acceptance Criteria Checklist

| Criterion | Status | Evidence |
| :--- | :---: | :--- |
| 1. Website exposes signer_05 workflow | **MET** | Dropdown option, isolation banner, retake modal, and summary modal implemented in [collect.html](file:///Users/pavanaksshay/pavan_sign_language/web/templates/collect.html). |
| 2. Complete capture session can be initiated and saved using existing infrastructure | **MET** | Verified via `test_03_metadata_generation_and_role_isolation` and API endpoint verification. |
| 3. All 26 letters and both conditions represented in session workflow | **MET** | 52 matrix cells configured in presets, API summary, and frontend pill grids. |
| 4. Failed and superseded recordings are auditable | **MET** | `status='superseded'` and `status='rejected'` tracked in ledger; frame files not overwritten. |
| 5. Signer_05 data isolated from all training and tuning paths | **MET** | `allow_training=False`, `verify_training_isolation_safeguard()` enforced, 0 samples in train/val manifests. |
| 6. Existing workflows remain functional | **MET** | `signer_02`, `signer_03`, `signer_04` workflows validated via `test_10`. |
| 7. Tests pass and production defaults remain unchanged | **MET** | 234/234 tests passing; production defaults (`yolo`, `standard`, `smoothing=0`) and model hashes verified. |

---

## 8. Summary of Files Changed

- [phase14b_analysis/metadata_schema.json](file:///Users/pavanaksshay/pavan_sign_language/phase14b_analysis/metadata_schema.json): Added `"signer_05"` to `properties.signer_id.enum`.
- [phase18_collector.py](file:///Users/pavanaksshay/pavan_sign_language/phase18_collector.py):
  - Added `"signer_05"` to `ALLOWED_SIGNERS` and `ALL_VALID_SIGNERS`.
  - Added `EVALUATION_ONLY_SIGNERS = ["signer_05"]` and `SIGNER_05_MANIFEST_PATH`.
  - Added `load_signer05_manifest()` and `save_signer05_manifest()`.
  - Enhanced `commit_recording()` to detect prior takes, mark them `status="superseded"` in ledger, mark prior samples `quality.is_usable=False` in manifests, and generate unique frame filenames to prevent file overwrites.
  - Added `get_signer_summary()` for 52-cell matrix tracking and readiness calculation.
  - Added `verify_training_isolation_safeguard()` to quarantine signer 05 from training data.
- [server.py](file:///Users/pavanaksshay/pavan_sign_language/server.py):
  - Updated `/collect` template handler to pass all signers including `signer_05`.
  - Added `@app.get("/api/collect/summary/{signer_id}")`.
- [web/templates/collect.html](file:///Users/pavanaksshay/pavan_sign_language/web/templates/collect.html):
  - Added UI banner `#signer05-isolation-banner`.
  - Added `#signer-select` option for Volunteer D.
  - Added `#retake-modal` with confirmation dialog.
  - Added `#session-summary-modal` with 52-cell progress visualization and readiness badge.
  - Updated JavaScript to handle signer changes, retake button states, and summary retrieval.
- [tests/test_phase30a_signer05_capture.py](file:///Users/pavanaksshay/pavan_sign_language/tests/test_phase30a_signer05_capture.py): Created 11 comprehensive unit/integration tests for Phase 30A.
- [phase30a_capture_integrity_audit.json](file:///Users/pavanaksshay/pavan_sign_language/phase30a_capture_integrity_audit.json): Generated cryptographic and schema integrity audit report.
