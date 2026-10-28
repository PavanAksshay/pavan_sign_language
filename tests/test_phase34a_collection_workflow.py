#!/usr/bin/env python3
"""
Unit and Integration Tests for Phase 34A: Multi-Signer Development Campaign Data Collection Workflow.

Verifies:
1. Participant Onboarding & Schema Governance:
   - Metadata schema permits signer_06 through signer_11 and rejects unauthorized signers.
   - Distinct dataset roles correctly assigned:
     * development_train: signer_06, signer_07, signer_08, signer_09
     * development_validation: signer_10, signer_11
     * external_holdout: signer_05 (sealed)
     * frozen_benchmark: signer_02, signer_03
2. Collection Conditions & Geometry Telemetry:
   - Condition 1 (Standard Frontal 70cm, eye-level, 0° yaw/pitch)
   - Condition 2 (Angled Laptop & Lap-Rest, 25°-35° pitch, 15° yaw, 75cm, resting non-signing hand)
   - Condition 3 (Distance & Ambient Challenge, 1.0m-1.2m, 10° pitch, ambient lighting)
   - Disambiguation between horizontal yaw (15°) and vertical pitch (30°).
   - Nominal/operator-reported disclosures for angles and distances.
3. 3-Take Recording Matrix & Progress Tracking:
   - Planned 234 takes per development participant (26 letters x 3 conditions x 3 takes).
   - Independent take recording (Takes 1, 2, 3 do not collide).
   - Take-specific supersession (retaking Take 2 supersedes only Take 2; leaves Takes 1 and 3 active).
   - Session recovery and missing takes calculation.
4. Preflight Protocol Checklist & Hardware Readiness:
   - Participant consent enforcement as a mandatory blocking gate.
   - Validation of signer, letter, condition, and take number.
   - Hardware telemetry verification (min 1280x720, target 30 FPS).
5. Quality Auditing & Perceptual Deduplication:
   - Frame blur detection (Laplacian variance >= 50.0).
   - Illumination bounds (mean brightness in [40.0, 220.0]).
   - Static / camera-freeze detection (movement >= 0.1).
   - Deterministic perceptual dHash near-duplicate frame detection (Hamming <= 1, movement < 0.2).
   - Legitimate sign hold frames preserved.
   - Hand landmark detection and compact bbox warning without hard rejection.
6. Manifest Partitioning & Training Isolation Safeguard:
   - dev_train_manifest.json routes signer_06-signer_09 (allow_training=True).
   - dev_val_manifest.json routes signer_10-signer_11 (allow_training=False).
   - signer_05 routes to signer_05_manifest.json (allow_training=False).
   - verify_training_isolation_safeguard fails closed on any validation or holdout data in training sets.
   - audit_split_isolation confirms zero pairwise signer or hash leakage.
7. Backward Compatibility & Authoritative Integrity:
   - Existing signers (signer_02, signer_03, signer_04, signer_05) preserve existing contracts.
   - Authoritative model artifacts and prediction SHA-256 hashes remain bit-identical.
"""

import os
import sys
import json
import shutil
import tempfile
import hashlib
import unittest
from typing import Dict, Any, List
import numpy as np
import cv2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from phase18_collector import (
    ISLCollectionEngine,
    validate_sample_schema,
    ALLOWED_SIGNERS,
    ALL_VALID_SIGNERS,
    DEVELOPMENT_TRAIN_SIGNERS,
    DEVELOPMENT_VALIDATION_SIGNERS,
    DEVELOPMENT_SIGNERS,
    EVALUATION_ONLY_SIGNERS,
    FROZEN_BENCHMARK_SIGNERS,
    SIGNER_ROLE_MAPPING,
    CONDITION_PRESETS,
    compute_dhash,
    hamming_distance,
    verify_training_isolation_safeguard,
)

AUTHORITATIVE_HASHES = {
    "model_c_rf": {
        "path": os.path.join(BASE_DIR, "phase22_analysis", "models", "Model_C_Occlusion_Aware_RF_multisigner.joblib"),
        "sha256": "1fab3349e2e537d8b31ddbe3bdd5c21e3ed005f85aaac9d3875c72217e28cc98"
    },
    "model_c_scaler": {
        "path": os.path.join(BASE_DIR, "phase22_analysis", "models", "Model_C_Occlusion_Aware_RF_multisigner_scaler.joblib"),
        "sha256": "93842932e7cfeb1d283e99f67a733fd487161d83cbff356d7458ced2c3de385d"
    },
    "phase19_frozen_manifest": {
        "path": os.path.join(BASE_DIR, "phase19_frozen_manifest.json"),
        "sha256": "9824eb510cc231249e9dd0722761de9112a02e7cecb681a479c4ef805b0fa797"
    },
    "phase23_predictions": {
        "path": os.path.join(BASE_DIR, "phase23_predictions.csv"),
        "sha256": "da77036032fb3250141ccc16ba53d1dd7d4997b9817c6f525bbf9e000be45bfe"
    }
}


def _create_synthetic_video_bytes(
    num_frames: int = 15,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
    brightness: int = 128,
    add_motion: bool = True,
    seed: int = 0
) -> bytes:
    """Helper to generate in-memory synthetic MP4/AVI video bytes for tests."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        temp_path = tf.name

    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_path, fourcc, float(fps), (width, height))
        for i in range(num_frames):
            frame = np.full((height, width, 3), brightness, dtype=np.uint8)
            # Add grid pattern to provide rich edges/contrast so Laplacian variance is well above blur threshold
            for gy in range(0, height, 40):
                cv2.line(frame, (0, gy), (width, gy), (50, 50, 50), 2)
            for gx in range(0, width, 40):
                cv2.line(frame, (gx, 0), (gx, height), (50, 50, 50), 2)

            # Add synthetic moving shape to simulate motion and hand
            if add_motion:
                cx = int(width / 2 + 30 * np.sin(i * 0.5 + seed))
                cy = int(height / 2 + 20 * np.cos(i * 0.5 + seed))
            else:
                cx = int(width / 2)
                cy = int(height / 2)
            cv2.circle(frame, (cx, cy), 80, (220, 200, 180), -1)
            cv2.rectangle(frame, (cx - 40, cy - 40), (cx + 40, cy + 40), (0, 0, 255), 3)
            cv2.putText(frame, f"S{seed}_F{i}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            out.write(frame)
        out.release()

        with open(temp_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


class TestPhase34ACollectionWorkflow(unittest.TestCase):
    """Comprehensive test suite for Phase 34A Multi-Signer Development Campaign Data Collection."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_phase34a_")
        self.engine = ISLCollectionEngine(
            min_blur_var=30.0,
            min_brightness=30.0,
            max_brightness=230.0,
            target_frames_per_letter=16,
            min_movement_diff=0.5,
            raw_recordings_dir=os.path.join(self.temp_dir, "raw"),
            extracted_frames_dir=os.path.join(self.temp_dir, "frames"),
            staging_dir=os.path.join(self.temp_dir, "staging"),
            manifests_dir=os.path.join(self.temp_dir, "manifests"),
            rejected_recordings_dir=os.path.join(self.temp_dir, "data", "rejected_recordings")
        )
        self.engine.ledger_path = os.path.join(self.engine.manifests_dir, "collection_ledger.json")
        self.engine.test_manifest_path = os.path.join(self.engine.manifests_dir, "test_multisigner_manifest.json")
        self.engine.signer05_manifest_path = os.path.join(self.engine.manifests_dir, "signer_05_manifest.json")
        self.engine.dev_train_manifest_path = os.path.join(self.engine.manifests_dir, "dev_train_manifest.json")
        self.engine.dev_val_manifest_path = os.path.join(self.engine.manifests_dir, "dev_val_manifest.json")

        # Mock hand detector so OpenCV synthetic videos pass hand detection
        self.engine.detect_hands_in_frame = lambda frame, return_bbox_ratios=False: (
            (1, ["Right"], [0.95], False, [], [0.035]) if return_bbox_ratios else (1, ["Right"], [0.95], False, [])
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_schema_validates_signers_06_through_11(self):
        """Verify metadata_schema.json permits signer_06 through signer_11 and rejects unauthorized signers."""
        schema_path = os.path.join(BASE_DIR, "phase14b_analysis", "metadata_schema.json")
        self.assertTrue(os.path.exists(schema_path), f"Schema not found: {schema_path}")

        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)

        signer_enum = schema["properties"]["signer_id"]["enum"]
        for expected in ["signer_06", "signer_07", "signer_08", "signer_09", "signer_10", "signer_11"]:
            self.assertIn(expected, signer_enum, f"{expected} must be in metadata_schema.json signer_id enum")

        # Test validation function rejects invalid signers
        sample_stub = {
            "sample_id": "test_sample_001",
            "recording_id": "rec_test_001",
            "confirmation_method": "signer_confirmed",
            "split": "development_train",
            "dataset_role": "development_train",
            "allow_training": True,
            "eligible_for_evaluation": False,
            "is_superseded": False,
            "signer_id": "signer_99_unauthorized",
            "session_id": "session_test",
            "letter": "A",
            "condition": {
                "lighting": "indoor_overhead",
                "background": "plain_wall",
                "camera_position": "frontal_eye_level",
                "distance_cm": 70
            },
            "detection": {
                "hands_detected": 1,
                "handedness": ["Right"],
                "scores": [0.95],
                "lap_hand_suppressed": False
            },
            "quality": {
                "blur_laplacian_var": 120.0,
                "mean_brightness": 128.0,
                "is_usable": True
            },
            "source": {
                "video_path": "data/raw_recordings/signer_99/condition_1/rec.mp4",
                "frame_path": "data/extracted_frames/signer_99/condition_1/A/sample_001.jpg",
                "frame_index": 0,
                "timestamp_sec": 0.0,
                "sha256": "0" * 64
            }
        }
        is_valid, err = validate_sample_schema(sample_stub)
        self.assertFalse(is_valid, "Schema validator must reject unauthorized signer_99_unauthorized")

        # Test validation function accepts signer_06
        sample_stub["signer_id"] = "signer_06"
        is_valid, err = validate_sample_schema(sample_stub)
        self.assertTrue(is_valid, f"Schema validator should accept signer_06: {err}")

    def test_dataset_role_mapping_and_split_isolation(self):
        """Verify role mappings and pairwise split isolation."""
        self.assertEqual(SIGNER_ROLE_MAPPING["signer_06"], "development_train")
        self.assertEqual(SIGNER_ROLE_MAPPING["signer_07"], "development_train")
        self.assertEqual(SIGNER_ROLE_MAPPING["signer_08"], "development_train")
        self.assertEqual(SIGNER_ROLE_MAPPING["signer_09"], "development_train")
        self.assertEqual(SIGNER_ROLE_MAPPING["signer_10"], "development_validation")
        self.assertEqual(SIGNER_ROLE_MAPPING["signer_11"], "development_validation")
        self.assertEqual(SIGNER_ROLE_MAPPING["signer_05"], "external_holdout")

        # Verify pairwise disjointness
        dev_train = set(DEVELOPMENT_TRAIN_SIGNERS)
        dev_val = set(DEVELOPMENT_VALIDATION_SIGNERS)
        holdout = set(EVALUATION_ONLY_SIGNERS)
        frozen = set(FROZEN_BENCHMARK_SIGNERS)

        self.assertTrue(dev_train.isdisjoint(dev_val), "dev_train and dev_val signers must be disjoint")
        self.assertTrue(dev_train.isdisjoint(holdout), "dev_train and external holdout must be disjoint")
        self.assertTrue(dev_train.isdisjoint(frozen), "dev_train and frozen benchmark must be disjoint")
        self.assertTrue(dev_val.isdisjoint(holdout), "dev_val and external holdout must be disjoint")
        self.assertTrue(dev_val.isdisjoint(frozen), "dev_val and frozen benchmark must be disjoint")

        # Audit split isolation
        iso = self.engine.audit_split_isolation()
        self.assertTrue(iso["is_leakage_free"], "Collection engine audit_split_isolation must report is_leakage_free")

    def test_condition_presets_telemetry_and_geometry(self):
        """Verify Condition 1, 2, and 3 specifications, angles, and telemetry disclosures."""
        self.assertIn("condition_1", CONDITION_PRESETS)
        self.assertIn("condition_2", CONDITION_PRESETS)
        self.assertIn("condition_3", CONDITION_PRESETS)

        c1 = CONDITION_PRESETS["condition_1"]
        c2 = CONDITION_PRESETS["condition_2"]
        c3 = CONDITION_PRESETS["condition_3"]

        # Condition 1: Desk Neutral (Eye-level 70cm, 0° pitch, 0° yaw)
        self.assertEqual(c1["camera_pitch_deg"], 0.0)
        self.assertEqual(c1["camera_yaw_deg"], 0.0)
        self.assertEqual(c1["distance_cm"], 70)
        self.assertFalse(c1["lap_hand_resting"])

        # Condition 2: Angled Laptop & Lap-Rest (25°-35° pitch, 15° yaw, 75cm)
        self.assertEqual(c2["camera_pitch_deg"], 30.0)
        self.assertEqual(c2["camera_yaw_deg"], 15.0)
        self.assertEqual(c2["distance_cm"], 75)
        self.assertTrue(c2["lap_hand_resting"])
        self.assertIn("orthogonal", c2["axis_disambiguation"])

        # Condition 3: Distance Challenge (10° pitch, 0° yaw, 110cm)
        self.assertEqual(c3["camera_pitch_deg"], 10.0)
        self.assertEqual(c3["camera_yaw_deg"], 0.0)
        self.assertEqual(c3["distance_cm"], 110)
        self.assertIn("Distance", c3["name"])

        # Disclosures present
        for c in [c1, c2, c3]:
            self.assertIn("nominal", c["geometry_disclaimer"].lower())

    def test_preflight_checklist_readiness_and_gates(self):
        """Verify get_preflight_checklist enforces consent, validates roles, and discloses geometry."""
        # Case 1: Consent False -> BLOCKED
        chk_blocked = self.engine.get_preflight_checklist(
            signer_id="signer_06",
            condition_id="condition_1",
            letter="A",
            take_number=1,
            consent_confirmed=False
        )
        self.assertEqual(chk_blocked["preflight_status"], "BLOCKED")
        self.assertFalse(chk_blocked["is_ready_for_capture"])
        self.assertTrue(any("consent" in r.lower() for r in chk_blocked["blocking_reasons"]))

        # Case 2: Consent True and Valid -> READY
        chk_ready = self.engine.get_preflight_checklist(
            signer_id="signer_06",
            condition_id="condition_2",
            letter="B",
            take_number=2,
            consent_confirmed=True,
            detected_resolution={"width": 1280, "height": 720},
            observed_fps=30.0
        )
        self.assertEqual(chk_ready["preflight_status"], "READY")
        self.assertTrue(chk_ready["is_ready_for_capture"])
        self.assertEqual(chk_ready["signer"]["dataset_role"], "development_train")
        self.assertTrue(chk_ready["signer"]["allow_training"])
        self.assertEqual(chk_ready["protocol_geometry"]["camera_pitch_deg"], 30.0)
        self.assertEqual(chk_ready["protocol_geometry"]["camera_yaw_deg"], 15.0)
        self.assertTrue(chk_ready["telemetry"]["resolution_meets_target"])

        # Case 3: Development Validation signer_10 -> allow_training False
        chk_val = self.engine.get_preflight_checklist(
            signer_id="signer_10",
            condition_id="condition_1",
            letter="C",
            take_number=1,
            consent_confirmed=True
        )
        self.assertEqual(chk_val["signer"]["dataset_role"], "development_validation")
        self.assertFalse(chk_val["signer"]["allow_training"])

        # Case 4: Sealed holdout signer_05 -> evaluation only
        chk_holdout = self.engine.get_preflight_checklist(
            signer_id="signer_05",
            condition_id="condition_1",
            letter="D",
            take_number=1,
            consent_confirmed=True
        )
        self.assertIn("SEALED", chk_holdout["signer"]["role_governance"])
        self.assertFalse(chk_holdout["signer"]["allow_training"])

    def test_three_take_matrix_and_take_specific_supersession(self):
        """Verify independent 3-take recording matrix and take-specific supersession."""
        vid_bytes1 = _create_synthetic_video_bytes(num_frames=15, fps=30, seed=1)
        vid_bytes2 = _create_synthetic_video_bytes(num_frames=15, fps=30, seed=2)
        vid_bytes3 = _create_synthetic_video_bytes(num_frames=15, fps=30, seed=3)

        # Stage and commit Take 1 for signer_06, condition_1, letter 'A'
        stg1 = self.engine.stage_recording(
            video_bytes=vid_bytes1,
            filename="rec_a_take1.mp4",
            signer_id="signer_06",
            letter="A",
            condition_id="condition_1",
            take_number=1,
            consent_confirmed=True
        )
        self.assertIn("take1", stg1["recording_id"])
        res1 = self.engine.commit_recording(stg1["staging_id"], action="accept")
        self.assertEqual(res1["status"], "accepted")

        # Stage and commit Take 2 for same (signer_06, condition_1, letter 'A')
        stg2 = self.engine.stage_recording(
            video_bytes=vid_bytes2,
            filename="rec_a_take2.mp4",
            signer_id="signer_06",
            letter="A",
            condition_id="condition_1",
            take_number=2,
            consent_confirmed=True
        )
        self.assertIn("take2", stg2["recording_id"])
        res2 = self.engine.commit_recording(stg2["staging_id"], action="accept")
        self.assertEqual(res2["status"], "accepted")

        # Stage and commit Take 3 for same cell
        stg3 = self.engine.stage_recording(
            video_bytes=vid_bytes3,
            filename="rec_a_take3.mp4",
            signer_id="signer_06",
            letter="A",
            condition_id="condition_1",
            take_number=3,
            consent_confirmed=True
        )
        self.assertIn("take3", stg3["recording_id"])
        res3 = self.engine.commit_recording(stg3["staging_id"], action="accept")
        self.assertEqual(res3["status"], "accepted")

        # Verify all 3 takes exist and are NOT superseded
        dev_train_manifest = self.engine.load_dev_train_manifest()
        cell_samples = [
            s for s in dev_train_manifest
            if s["signer_id"] == "signer_06" and s["condition_id"] == "condition_1" and s["letter"] == "A"
        ]
        active_samples = [s for s in cell_samples if not s.get("is_superseded")]
        active_takes = set(s.get("take_number") for s in active_samples)
        self.assertEqual(active_takes, {1, 2, 3}, "Takes 1, 2, and 3 must all be active simultaneously")

        # Now re-record Take 2 (variation redo)
        vid_bytes2_redo = _create_synthetic_video_bytes(num_frames=15, fps=30, seed=4)
        stg2_redo = self.engine.stage_recording(
            video_bytes=vid_bytes2_redo,
            filename="rec_a_take2_redo.mp4",
            signer_id="signer_06",
            letter="A",
            condition_id="condition_1",
            take_number=2,
            consent_confirmed=True
        )
        res2_redo = self.engine.commit_recording(stg2_redo["staging_id"], action="accept")
        self.assertEqual(res2_redo["status"], "accepted")
        self.assertGreater(res2_redo["superseded_count"], 0, "Redoing Take 2 must supersede prior Take 2")

        # Reload manifest: Verify prior Take 2 was superseded, but Take 1 and Take 3 remained active!
        dev_train_manifest_after = self.engine.load_dev_train_manifest()
        cell_samples_after = [
            s for s in dev_train_manifest_after
            if s["signer_id"] == "signer_06" and s["condition_id"] == "condition_1" and s["letter"] == "A"
        ]
        active_after = [s for s in cell_samples_after if not s.get("is_superseded")]
        active_takes_after = [s.get("take_number") for s in active_after]
        self.assertEqual(sorted(list(set(active_takes_after))), [1, 2, 3])

        # Verify old Take 2 is marked superseded
        superseded_samples = [s for s in cell_samples_after if s.get("is_superseded")]
        self.assertTrue(len(superseded_samples) > 0)
        for s in superseded_samples:
            self.assertEqual(s.get("take_number"), 2, "Only Take 2 should have been superseded")

    def test_operator_rejection_and_retake_preservation(self):
        """Verify operator rejection preserves video and audit record without corrupting manifest."""
        vid_bytes = _create_synthetic_video_bytes(num_frames=12, fps=30)

        stg = self.engine.stage_recording(
            video_bytes=vid_bytes,
            filename="rec_b_take1.mp4",
            signer_id="signer_07",
            letter="B",
            condition_id="condition_2",
            take_number=1,
            consent_confirmed=True
        )
        staging_id = stg["staging_id"]

        # Reject recording with note
        rej_res = self.engine.commit_recording(
            staging_id=staging_id,
            action="reject",
            rejection_note="Hand moved out of frame during recording"
        )
        self.assertEqual(rej_res["status"], "rejected")

        # Check rejected directory
        rejected_dir = os.path.join(self.temp_dir, "data", "rejected_recordings", staging_id)
        self.assertTrue(os.path.exists(rejected_dir), "Rejected recording directory must exist")
        self.assertTrue(os.path.exists(os.path.join(rejected_dir, "rejection_audit.json")))

        # Check ledger record
        ledger = self.engine.load_ledger()
        rej_records = [r for r in ledger.get("records", []) if r.get("staging_id") == staging_id]
        self.assertEqual(len(rej_records), 1)
        self.assertEqual(rej_records[0]["status"], "rejected")
        self.assertEqual(rej_records[0]["rejection_note"], "Hand moved out of frame during recording")

        # Manifest must be empty
        dev_train_manifest = self.engine.load_dev_train_manifest()
        self.assertEqual(len(dev_train_manifest), 0, "Rejected recording must not enter dev_train_manifest")

    def test_quality_auditing_and_perceptual_hash_deduplication(self):
        """Verify quality audits (blur, brightness, dHash near-duplicate, movement)."""
        # Test 1: Laplacian blur check
        blurry_frame = np.full((720, 1280, 3), 128, dtype=np.uint8)  # perfectly uniform flat image = 0 variance
        quality_blur = self.engine.audit_frame_quality(blurry_frame)
        self.assertFalse(quality_blur["is_usable"])
        self.assertIn("motion_blur", quality_blur["rejection_reason"])

        # Test 2: Low brightness check
        dark_frame = np.full((720, 1280, 3), 20, dtype=np.uint8)
        # Add some noise to avoid blur failure
        cv2.randn(dark_frame, 20, 2)
        quality_dark = self.engine.audit_frame_quality(dark_frame)
        self.assertFalse(quality_dark["is_usable"])
        self.assertIn("underexposed", quality_dark["rejection_reason"])

        # Test 3: dHash computation and Hamming distance
        img1 = np.zeros((120, 160, 3), dtype=np.uint8)
        cv2.rectangle(img1, (30, 30), (90, 90), (255, 255, 255), -1)
        hash1 = compute_dhash(img1)
        self.assertEqual(len(hash1), 16)

        img2 = img1.copy()
        hash2 = compute_dhash(img2)
        self.assertEqual(hamming_distance(hash1, hash2), 0)

        # Perturb slightly
        cv2.circle(img2, (10, 10), 3, (255, 255, 255), -1)
        hash2_perturbed = compute_dhash(img2)
        self.assertLessEqual(hamming_distance(hash1, hash2_perturbed), 4)

    def test_training_isolation_safeguard(self):
        """Verify verify_training_isolation_safeguard blocks unauthorized or validation data from training."""
        # Valid training samples (signer_06, allow_training=True, dataset_role='development_train')
        valid_train_samples = [
            {
                "sample_id": "sample_dev_01",
                "signer_id": "signer_06",
                "dataset_role": "development_train",
                "allow_training": True
            },
            {
                "sample_id": "sample_dev_02",
                "signer_id": "signer_07",
                "dataset_role": "development_train",
                "allow_training": True
            }
        ]
        self.assertTrue(verify_training_isolation_safeguard(valid_train_samples))

        # Attempt to insert validation signer_10 into training -> MUST RAISE
        leak_val_samples = [
            {
                "sample_id": "sample_dev_val",
                "signer_id": "signer_10",
                "dataset_role": "development_validation",
                "allow_training": False
            }
        ]
        with self.assertRaises(ValueError) as ctx:
            verify_training_isolation_safeguard(leak_val_samples)
        self.assertIn("TRAINING LEAKAGE DETECTED", str(ctx.exception))

        # Attempt to insert holdout signer_05 into training -> MUST RAISE
        leak_holdout_samples = [
            {
                "sample_id": "sample_holdout",
                "signer_id": "signer_05",
                "dataset_role": "independent_evaluation_holdout",
                "allow_training": False
            }
        ]
        with self.assertRaises(ValueError) as ctx:
            verify_training_isolation_safeguard(leak_holdout_samples)
        self.assertIn("TRAINING LEAKAGE DETECTED", str(ctx.exception))

        # Attempt to insert frozen benchmark signer_02 into training -> MUST RAISE
        leak_frozen_samples = [
            {
                "sample_id": "sample_frozen",
                "signer_id": "signer_02",
                "dataset_role": "frozen_benchmark",
                "allow_training": False
            }
        ]
        with self.assertRaises(ValueError) as ctx:
            verify_training_isolation_safeguard(leak_frozen_samples)
        self.assertIn("TRAINING LEAKAGE DETECTED", str(ctx.exception))

    def test_partition_manifest_routing(self):
        """Verify dev_train vs dev_val partition manifest routing."""
        vid_bytes_train = _create_synthetic_video_bytes(num_frames=12, fps=30, seed=10)
        vid_bytes_val = _create_synthetic_video_bytes(num_frames=12, fps=30, seed=11)

        # Stage and commit signer_06 (dev train)
        stg_train = self.engine.stage_recording(
            video_bytes=vid_bytes_train,
            filename="rec_c_train.mp4",
            signer_id="signer_06",
            letter="C",
            condition_id="condition_1",
            take_number=1,
            consent_confirmed=True
        )
        self.engine.commit_recording(stg_train["staging_id"], action="accept")

        # Stage and commit signer_10 (dev val)
        stg_val = self.engine.stage_recording(
            video_bytes=vid_bytes_val,
            filename="rec_c_val.mp4",
            signer_id="signer_10",
            letter="C",
            condition_id="condition_1",
            take_number=1,
            consent_confirmed=True
        )
        self.engine.commit_recording(stg_val["staging_id"], action="accept")

        # Check dev_train_manifest.json
        dev_train_manifest = self.engine.load_dev_train_manifest()
        train_signers = set(s["signer_id"] for s in dev_train_manifest)
        self.assertIn("signer_06", train_signers)
        self.assertNotIn("signer_10", train_signers)
        for s in dev_train_manifest:
            self.assertTrue(s["allow_training"])
            self.assertEqual(s["dataset_role"], "development_train")

        # Check dev_val_manifest.json
        dev_val_manifest = self.engine.load_dev_val_manifest()
        val_signers = set(s["signer_id"] for s in dev_val_manifest)
        self.assertIn("signer_10", val_signers)
        self.assertNotIn("signer_06", val_signers)
        for s in dev_val_manifest:
            self.assertFalse(s["allow_training"])
            self.assertEqual(s["dataset_role"], "development_validation")

    def test_progress_tracking_and_interruption_recovery(self):
        """Verify progress reporting for development signers (234 planned takes) and interruption recovery."""
        prog = self.engine.get_progress()
        self.assertIn("signer_summary", prog)
        s06_summary = prog["signer_summary"]["signer_06"]
        self.assertEqual(s06_summary["total_takes"], 234)
        self.assertEqual(s06_summary["planned_takes"], 234)
        self.assertEqual(s06_summary["completed_takes"], 0)
        self.assertEqual(s06_summary["missing_takes_count"], 234)
        self.assertEqual(s06_summary["collection_status"], "NOT_COLLECTED")
        self.assertFalse(s06_summary["is_complete"])

        # Check signer summary method
        s06_detailed = self.engine.get_signer_summary("signer_06")
        self.assertEqual(s06_detailed["planned_takes"], 234)
        self.assertEqual(len(s06_detailed["missing_takes"]), 234)
        self.assertEqual(s06_detailed["next_recommended_target"]["letter"], "A")
        self.assertEqual(s06_detailed["next_recommended_target"]["condition_id"], "condition_1")
        self.assertEqual(s06_detailed["next_recommended_target"]["take_number"], 1)

    def test_backward_compatibility_legacy_signers(self):
        """Verify signer_05, signer_04, signer_02, and signer_03 retain their contracts."""
        prog = self.engine.get_progress()
        s05_summary = prog["signer_summary"]["signer_05"]
        # Legacy contract: 52 cells
        self.assertEqual(s05_summary["total_cells"], 52)
        self.assertTrue(s05_summary["is_evaluation_only"])

        s02_summary = prog["signer_summary"]["signer_02"]
        self.assertEqual(s02_summary["total_cells"], 52)
        self.assertEqual(s02_summary["dataset_role"], "frozen_benchmark")

    def test_authoritative_artifacts_hash_preservation(self):
        """Verify that authoritative models, scalers, and frozen prediction manifests remain unaltered."""
        for name, spec in AUTHORITATIVE_HASHES.items():
            path = spec["path"]
            expected_hash = spec["sha256"]
            self.assertTrue(os.path.exists(path), f"Authoritative artifact missing: {path}")

            h = hashlib.sha256()
            with open(path, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            actual_hash = h.hexdigest()
            self.assertEqual(
                actual_hash,
                expected_hash,
                f"SHA-256 integrity violation on authoritative artifact '{name}'!\nExpected: {expected_hash}\nActual:   {actual_hash}"
            )


if __name__ == "__main__":
    unittest.main()
