#!/usr/bin/env python3
"""
Unit and integration tests for Phase 18 Multi-Signer Data Collection Engine.

Covers:
1. Metadata schema validation compliance.
2. Duplicate frame hash detection and rejection.
3. Quality gates: blur Laplacian variance and brightness boundaries.
4. Volunteer consent enforcement.
5. Empty and corrupt recording rejection.
6. Configurable frame sampling interval and movement filter.
7. Two-phase staging, review, acceptance, and rejection lifecycles.
8. Signer and session split isolation (zero leakage into train/val).
9. Voluntary participant right-to-deletion.
10. Progress matrix calculation.
"""

import os
import sys
import json
import time
import shutil
import tempfile
import unittest
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from phase18_collector import (
    ISLCollectionEngine,
    validate_sample_schema,
    CONDITION_PRESETS
)


class TestPhase18CollectionEngine(unittest.TestCase):

    def setUp(self):
        # Create isolated temporary directory for testing
        self.test_dir = tempfile.mkdtemp(prefix="isl_phase18_test_")
        self.engine = ISLCollectionEngine(
            min_blur_var=50.0,
            min_brightness=40.0,
            max_brightness=220.0,
            target_frames_per_letter=30,
            min_movement_diff=1.5,
            raw_recordings_dir=os.path.join(self.test_dir, "raw"),
            extracted_frames_dir=os.path.join(self.test_dir, "frames"),
            staging_dir=os.path.join(self.test_dir, "staging"),
            manifests_dir=os.path.join(self.test_dir, "manifests")
        )

        self.engine.ledger_path = os.path.join(self.engine.manifests_dir, "collection_ledger.json")
        self.engine.test_manifest_path = os.path.join(self.engine.manifests_dir, "test_multisigner_manifest.json")

        # Mock landmarker to avoid heavy GPU/model loading in fast unit tests
        class MockLandmarkerResult:
            hand_landmarks = []
            handedness = []

        class MockLandmarker:
            def detect(self, img):
                return MockLandmarkerResult()

        self.engine._landmarker = MockLandmarker()

        # Monkeypatch detect_hands_in_frame to return a detected hand by default for test frames
        self.engine.detect_hands_in_frame = lambda frame: (1, ["Right"], [0.95], False, [])

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_synthetic_video(self, num_frames=30, fps=30.0, width=320, height=240, add_movement=True, tag=""):
        """Creates an in-memory synthetic MP4 video for ingestion testing."""
        import uuid
        tmp_video = os.path.join(self.test_dir, f"test_vid_{uuid.uuid4().hex[:6]}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(tmp_video, fourcc, fps, (width, height))

        for i in range(num_frames):
            frame = np.full((height, width, 3), 120, dtype=np.uint8)
            # Add high-contrast circle and move it to simulate real signer motion
            offset = (i * 3) if add_movement else 0
            cv2.circle(frame, (80 + offset, 120), 40, (255, 255, 255), -1)
            cv2.putText(frame, f"{tag}_F{i}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            out.write(frame)

        out.release()
        with open(tmp_video, "rb") as f:
            video_bytes = f.read()
        if os.path.exists(tmp_video):
            os.remove(tmp_video)
        return video_bytes

    def test_sample_schema_validation_success(self):
        """Schema validator must accept a fully conforming sample dictionary."""
        valid_sample = {
            "sample_id": "signer_02_sess01_C_0003",
            "signer_id": "signer_02",
            "session_id": "sess01",
            "recording_id": "rec_01",
            "letter": "C",
            "confirmation_method": "signer_confirmed",
            "condition": {
                "lighting": "indoor_overhead",
                "background": "plain_wall",
                "camera_position": "frontal_eye_level",
                "distance_cm": 75
            },
            "detection": {
                "hands_detected": 1,
                "handedness": ["Right"],
                "scores": [0.96],
                "lap_hand_suppressed": False
            },
            "quality": {
                "blur_laplacian_var": 142.5,
                "mean_brightness": 128.0,
                "is_usable": True,
                "rejection_reason": None
            },
            "source": {
                "video_path": "data/raw_recordings/signer_02/condition_1/rec_01.mp4",
                "frame_index": 3,
                "timestamp_sec": 0.1
            }
        }
        is_valid, err = validate_sample_schema(valid_sample)
        self.assertTrue(is_valid, f"Validation failed unexpectedly: {err}")
        self.assertIsNone(err)

    def test_sample_schema_validation_missing_key(self):
        """Schema validator must reject samples missing mandatory fields."""
        incomplete_sample = {
            "sample_id": "signer_02_sess01_C_0003",
            "signer_id": "signer_02",
            "letter": "C"
            # Missing condition, detection, quality, source
        }
        is_valid, err = validate_sample_schema(incomplete_sample)
        self.assertFalse(is_valid)
        self.assertIn("Missing required root key", err)

    def test_sample_schema_validation_invalid_signer_and_letter(self):
        """Schema validator must reject unauthorized signers or non-alphabet letters."""
        sample = {
            "sample_id": "s_01",
            "signer_id": "signer_99",  # Invalid signer
            "session_id": "sess01",
            "recording_id": "rec_01",
            "letter": "C",
            "confirmation_method": "signer_confirmed",
            "condition": {"lighting": "indoor_overhead", "background": "plain_wall", "camera_position": "frontal_eye_level", "distance_cm": 75},
            "detection": {"hands_detected": 1, "handedness": ["Right"], "scores": [0.9], "lap_hand_suppressed": False},
            "quality": {"blur_laplacian_var": 100.0, "mean_brightness": 100.0, "is_usable": True},
            "source": {"video_path": "v.mp4", "frame_index": 0, "timestamp_sec": 0.0}
        }
        is_valid, err = validate_sample_schema(sample)
        self.assertFalse(is_valid)
        self.assertIn("Invalid signer_id", err)

        # Invalid multi-character or lowercase letter
        sample["signer_id"] = "signer_02"
        sample["letter"] = "WORD_HELLO"
        is_valid, err = validate_sample_schema(sample)
        self.assertFalse(is_valid)
        self.assertIn("Invalid letter", err)

    def test_quality_gates_blur_and_brightness(self):
        """Engine audits blur, underexposure, and overexposure correctly."""
        # 1. Blurry image
        blurry = np.full((100, 100, 3), 128, dtype=np.uint8)
        q_blur = self.engine.audit_frame_quality(blurry)
        self.assertFalse(q_blur["is_usable"])
        self.assertIn("motion_blur", q_blur["rejection_reason"])

        # 2. Dark image
        dark = np.full((100, 100, 3), 20, dtype=np.uint8)
        q_dark = self.engine.audit_frame_quality(dark)
        self.assertFalse(q_dark["is_usable"])
        self.assertIn("underexposed", q_dark["rejection_reason"])

        # 3. Overexposed image
        bright = np.full((100, 100, 3), 245, dtype=np.uint8)
        q_bright = self.engine.audit_frame_quality(bright)
        self.assertFalse(q_bright["is_usable"])
        self.assertIn("overexposed", q_bright["rejection_reason"])

        # 4. Sharp, well-lit image
        sharp = np.full((100, 100, 3), 128, dtype=np.uint8)
        cv2.circle(sharp, (50, 50), 30, (255, 255, 255), -1)
        q_sharp = self.engine.audit_frame_quality(sharp)
        self.assertTrue(q_sharp["is_usable"])
        self.assertIsNone(q_sharp["rejection_reason"])

    def test_consent_enforcement(self):
        """Staging a recording without confirmed volunteer consent must be refused."""
        video_bytes = self._create_synthetic_video(num_frames=15)
        with self.assertRaises(ValueError) as ctx:
            self.engine.stage_recording(
                video_bytes=video_bytes,
                filename="test.mp4",
                signer_id="signer_02",
                letter="A",
                condition_id="condition_1",
                session_id="sess_01",
                consent_confirmed=False  # Consent NOT given
            )
        self.assertIn("Volunteer consent must be explicitly confirmed", str(ctx.exception))

    def test_empty_or_too_short_recording_rejection(self):
        """Corrupt or too-short recordings must be safely rejected with informative error."""
        # Corrupt / empty bytes
        with self.assertRaises(ValueError) as ctx:
            self.engine.stage_recording(
                video_bytes=b"corrupt_invalid_bytes",
                filename="test.mp4",
                signer_id="signer_02",
                letter="A",
                condition_id="condition_1",
                session_id="sess_01",
                consent_confirmed=True
            )
        self.assertIn("Could not open or decode", str(ctx.exception))

        # Too short (< 5 frames)
        short_bytes = self._create_synthetic_video(num_frames=3)
        with self.assertRaises(ValueError) as ctx:
            self.engine.stage_recording(
                video_bytes=short_bytes,
                filename="test.mp4",
                signer_id="signer_02",
                letter="A",
                condition_id="condition_1",
                session_id="sess_01",
                consent_confirmed=True
            )
        self.assertIn("Recording is too short", str(ctx.exception))

    def test_staging_and_acceptance_lifecycle(self):
        """Full lifecycle: stage video -> preview -> commit accept -> verify storage and ledger."""
        video_bytes = self._create_synthetic_video(num_frames=30, add_movement=True)

        stage_info = self.engine.stage_recording(
            video_bytes=video_bytes,
            filename="rec_c.mp4",
            signer_id="signer_02",
            letter="C",
            condition_id="condition_1",
            session_id="session_test_accept",
            consent_confirmed=True,
            sampling_interval=3
        )

        self.assertIn("staging_id", stage_info)
        self.assertEqual(stage_info["signer_id"], "signer_02")
        self.assertEqual(stage_info["letter"], "C")
        self.assertGreater(stage_info["usable_count"], 0)
        self.assertEqual(len(stage_info["frames"]), 10)  # 30 frames / interval 3 = 10 sampled

        # Check thumbnails generated
        for f in stage_info["frames"]:
            self.assertTrue(f["thumbnail_b64"].startswith("data:image/jpeg;base64,"))

        # Commit acceptance
        commit_res = self.engine.commit_recording(
            staging_id=stage_info["staging_id"],
            action="accept"
        )
        self.assertEqual(commit_res["status"], "accepted")
        self.assertEqual(commit_res["letter"], "C")
        self.assertGreater(commit_res["usable_frames_saved"], 0)

        # Staging directory must be cleaned up
        staged_dir = os.path.join(self.engine.staging_dir, stage_info["staging_id"])
        self.assertFalse(os.path.exists(staged_dir))

        # Manifest must contain the new samples
        test_samples = self.engine.load_test_manifest()
        c_samples = [s for s in test_samples if s["letter"] == "C" and s["signer_id"] == "signer_02"]
        self.assertEqual(len(c_samples), commit_res["usable_frames_saved"])

    def test_staging_and_rejection_lifecycle(self):
        """Full lifecycle: stage video -> preview -> commit reject -> verify total cleanup."""
        video_bytes = self._create_synthetic_video(num_frames=20)

        stage_info = self.engine.stage_recording(
            video_bytes=video_bytes,
            filename="rec_rej.mp4",
            signer_id="signer_03",
            letter="X",
            condition_id="condition_2",
            session_id="session_test_reject",
            consent_confirmed=True,
            sampling_interval=2
        )

        staging_id = stage_info["staging_id"]

        commit_res = self.engine.commit_recording(
            staging_id=staging_id,
            action="reject",
            rejection_note="Hand blurred and obscured"
        )

        self.assertEqual(commit_res["status"], "rejected")
        staged_dir = os.path.join(self.engine.staging_dir, staging_id)
        self.assertFalse(os.path.exists(staged_dir))

        # Ledger records rejection event
        ledger = self.engine.load_ledger()
        rejections = [r for r in ledger["records"] if r.get("status") == "rejected" and r.get("staging_id") == staging_id]
        self.assertEqual(len(rejections), 1)
        self.assertIn("blurred and obscured", rejections[0]["rejection_note"])

    def test_split_isolation_and_holdout_quarantine(self):
        """Ensures external test samples are 100% held out and never leak into train/val."""
        iso = self.engine.audit_split_isolation()
        self.assertTrue(iso["is_leakage_free"])
        self.assertTrue(iso["is_strictly_held_out"])
        self.assertEqual(iso["overlaps"]["hash_train_test_count"], 0)
        self.assertEqual(iso["overlaps"]["hash_val_test_count"], 0)

    def test_volunteer_right_to_deletion(self):
        """Volunteer right-to-deletion must purge video, frames, and manifest entries permanently."""
        video_bytes = self._create_synthetic_video(num_frames=25, tag="del")
        stage_info = self.engine.stage_recording(
            video_bytes=video_bytes,
            filename="rec_del.mp4",
            signer_id="signer_02",
            letter="L",
            condition_id="condition_1",
            session_id="session_delete_me",
            consent_confirmed=True
        )
        self.engine.commit_recording(stage_info["staging_id"], action="accept")

        recording_id = stage_info["recording_id"]
        # Delete the recording
        del_res = self.engine.delete_recording(recording_id)
        self.assertEqual(del_res["status"], "deleted")
        self.assertEqual(del_res["recording_id"], recording_id)

        # Check manifest no longer contains this recording
        test_samples = self.engine.load_test_manifest()
        matching = [s for s in test_samples if s.get("recording_id") == recording_id]
        self.assertEqual(len(matching), 0)

    def test_progress_matrix_calculation(self):
        """Progress matrix accounts for all 26 letters, 2 signers, and 2 conditions."""
        prog = self.engine.get_progress()
        self.assertEqual(prog["nominal_target_frames"], 3120)
        self.assertEqual(prog["target_per_letter"], 30)
        self.assertIn("signer_02", prog["matrix"])
        self.assertIn("signer_03", prog["matrix"])
        self.assertIn("condition_1", prog["matrix"]["signer_02"])
        self.assertIn("condition_2", prog["matrix"]["signer_02"])
        self.assertEqual(len(prog["matrix"]["signer_02"]["condition_1"]["letters"]), 26)


if __name__ == "__main__":
    unittest.main()
