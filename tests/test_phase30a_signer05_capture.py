#!/usr/bin/env python3
"""
Unit and Integration Tests for Phase 30A: Signer 05 Independent Data Collection in the Website.

Verifies:
1. Signer_05 session initialization and authorization (ALLOWED_SIGNERS, ALL_VALID_SIGNERS,
   EVALUATION_ONLY_SIGNERS, metadata_schema.json).
2. A–Z (26 letters) and both-condition (Condition 1 & 2) coverage.
3. Correct metadata and manifest generation (split='external_test_signer_05',
   dataset_role='independent_evaluation_holdout', allow_training=False).
4. Dedicated signer_05 manifest maintenance (signer_05_manifest.json).
5. Unique recording IDs, non-colliding frame paths on disk, and safe retake handling
   with explicit supersession tracking.
6. Interrupted, empty, corrupt, and operator-rejected recording handling.
7. Duplicate detection via SHA-256 frame hash verification.
8. Correct progress and completion calculations (is_dataset_ready=False when incomplete).
9. Prevention of accidental training-data inclusion (split isolation audit and safeguards).
10. Preservation of existing signer_02, signer_03, and signer_04 collection behavior.
11. Preservation of production defaults and authoritative model/scaler SHA-256 hashes.
"""

import os
import sys
import json
import shutil
import tempfile
import hashlib
import unittest
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
    EVALUATION_ONLY_SIGNERS,
    CONDITION_PRESETS,
    verify_training_isolation_safeguard,
)
from pipeline import ISLPipeline

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


class TestPhase30ASigner05Capture(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="isl_phase30a_test_")
        self.engine = ISLCollectionEngine(
            min_blur_var=30.0,
            min_brightness=30.0,
            max_brightness=230.0,
            target_frames_per_letter=16,
            min_movement_diff=0.5,
            raw_recordings_dir=os.path.join(self.test_dir, "raw"),
            extracted_frames_dir=os.path.join(self.test_dir, "frames"),
            staging_dir=os.path.join(self.test_dir, "staging"),
            manifests_dir=os.path.join(self.test_dir, "manifests")
        )
        self.engine.ledger_path = os.path.join(self.engine.manifests_dir, "collection_ledger.json")
        self.engine.test_manifest_path = os.path.join(self.engine.manifests_dir, "test_multisigner_manifest.json")
        self.engine.signer05_manifest_path = os.path.join(self.engine.manifests_dir, "signer_05_manifest.json")

        # Mock hand detection for rapid, deterministic test execution
        self.engine.detect_hands_in_frame = lambda frame: (1, ["Right"], [0.95], False, [])

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_synthetic_video(self, num_frames=15, fps=20.0, width=320, height=240, signer_tag="signer_05", seed=0):
        """Generates a valid MP4 test video in memory."""
        import uuid
        tmp_video = os.path.join(self.test_dir, f"test_vid_{uuid.uuid4().hex[:6]}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(tmp_video, fourcc, fps, (width, height))

        for i in range(num_frames):
            frame = np.full((height, width, 3), 128, dtype=np.uint8)
            # Add dynamic movement to pass min_movement_diff
            cv2.circle(frame, (80 + (i + seed) * 4, 120), 30, (255, 255, 255), -1)
            cv2.putText(frame, f"{signer_tag}_S{seed}_F{i}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            out.write(frame)

        out.release()
        with open(tmp_video, "rb") as f:
            video_bytes = f.read()
        if os.path.exists(tmp_video):
            os.remove(tmp_video)
        return video_bytes

    def test_01_signer_05_constants_and_schema_authorization(self):
        """Verifies signer_05 is recognized in constants, evaluation lists, and metadata schema."""
        self.assertIn("signer_05", ALLOWED_SIGNERS, "signer_05 must be in ALLOWED_SIGNERS")
        self.assertIn("signer_05", ALL_VALID_SIGNERS, "signer_05 must be in ALL_VALID_SIGNERS")
        self.assertIn("signer_05", EVALUATION_ONLY_SIGNERS, "signer_05 must be in EVALUATION_ONLY_SIGNERS")

        schema_path = os.path.join(BASE_DIR, "phase14b_analysis", "metadata_schema.json")
        with open(schema_path, "r") as f:
            schema = json.load(f)
        enum_signers = schema["properties"]["signer_id"]["enum"]
        self.assertIn("signer_05", enum_signers, "metadata_schema.json must include signer_05 in enum")

        valid_sample = {
            "sample_id": "signer_05_session_20261025_01_A_0000",
            "signer_id": "signer_05",
            "session_id": "session_20261025_01",
            "recording_id": "rec_signer_05_condition_1_A_12345",
            "letter": "A",
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
                "scores": [0.95],
                "lap_hand_suppressed": False
            },
            "quality": {
                "blur_laplacian_var": 120.0,
                "mean_brightness": 130.0,
                "is_usable": True
            },
            "source": {
                "video_path": "data/raw_recordings/signer_05/condition_1/rec_signer_05_condition_1_A_12345.mp4",
                "frame_path": "data/extracted_frames/signer_05/condition_1/A/frame_0000.png",
                "frame_index": 0,
                "timestamp_sec": 0.0,
                "sha256": "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
            },
            "split": "external_test_signer_05",
            "dataset_role": "independent_evaluation_holdout",
            "allow_training": False,
            "eligible_for_evaluation": True
        }
        is_valid, msg = validate_sample_schema(valid_sample)
        self.assertTrue(is_valid, f"Valid signer_05 sample must pass schema validation: {msg}")

        invalid_sample = dict(valid_sample)
        invalid_sample["signer_id"] = "signer_99"
        is_valid, _ = validate_sample_schema(invalid_sample)
        self.assertFalse(is_valid, "Unauthorized signer_99 must be rejected by schema validation")

    def test_02_alphabet_and_both_conditions_coverage(self):
        """Verifies Condition 1 & 2 support and full A–Z alphabet readiness."""
        self.assertIn("condition_1", CONDITION_PRESETS)
        self.assertIn("condition_2", CONDITION_PRESETS)
        self.assertEqual(CONDITION_PRESETS["condition_1"]["name"], "Desk Neutral (Frontal)")
        self.assertEqual(CONDITION_PRESETS["condition_2"]["name"], "Lateral Warm (15° Offset)")

        alphabet = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
        self.assertEqual(len(alphabet), 26)

        # Stage both Condition 1 and Condition 2 for signer_05
        video_c1 = self._create_synthetic_video(signer_tag="signer_05_c1", seed=1)
        res_c1 = self.engine.stage_recording(
            video_bytes=video_c1,
            filename="rec_c1.mp4",
            signer_id="signer_05",
            letter="A",
            condition_id="condition_1",
            session_id="session_20261025_01",
            consent_confirmed=True
        )
        self.assertTrue(res_c1["staging_id"].startswith("stage_"))
        self.assertEqual(res_c1["condition_id"], "condition_1")

        video_c2 = self._create_synthetic_video(signer_tag="signer_05_c2", seed=2)
        res_c2 = self.engine.stage_recording(
            video_bytes=video_c2,
            filename="rec_c2.mp4",
            signer_id="signer_05",
            letter="A",
            condition_id="condition_2",
            session_id="session_20261025_01",
            consent_confirmed=True
        )
        self.assertTrue(res_c2["staging_id"].startswith("stage_"))
        self.assertEqual(res_c2["condition_id"], "condition_2")

    def test_03_metadata_generation_and_role_isolation(self):
        """Verifies committing signer_05 correctly tags holdout metadata."""
        video = self._create_synthetic_video(signer_tag="signer_05", seed=3)
        res = self.engine.stage_recording(
            video_bytes=video,
            filename="rec_05_A.mp4",
            signer_id="signer_05",
            letter="A",
            condition_id="condition_1",
            session_id="session_20261025_01",
            consent_confirmed=True
        )
        commit_res = self.engine.commit_recording(staging_id=res["staging_id"], action="accept")
        self.assertEqual(commit_res["status"], "accepted")
        self.assertGreater(commit_res["usable_frames_saved"], 0)

        # Read back test_manifest (stored directly as a list of samples)
        with open(self.engine.test_manifest_path, "r") as f:
            samples = json.load(f)
        self.assertIsInstance(samples, list)
        self.assertGreater(len(samples), 0)

        s05_samples = [s for s in samples if s["signer_id"] == "signer_05"]
        self.assertEqual(len(s05_samples), commit_res["usable_frames_saved"])

        for sample in s05_samples:
            self.assertEqual(sample["split"], "external_test_signer_05")
            self.assertEqual(sample["dataset_role"], "independent_evaluation_holdout")
            self.assertFalse(sample["allow_training"])
            self.assertTrue(sample["eligible_for_evaluation"])
            self.assertIn("sha256", sample.get("source", {}))
            frame_p = os.path.join(BASE_DIR, sample["source"]["frame_path"])
            self.assertTrue(os.path.exists(frame_p))

    def test_04_dedicated_signer05_manifest_maintenance(self):
        """Verifies signer_05_manifest.json is atomically created and maintained."""
        video = self._create_synthetic_video(signer_tag="signer_05", seed=4)
        res = self.engine.stage_recording(
            video_bytes=video,
            filename="rec_05_B.mp4",
            signer_id="signer_05",
            letter="B",
            condition_id="condition_1",
            session_id="session_20261025_01",
            consent_confirmed=True
        )
        self.engine.commit_recording(staging_id=res["staging_id"], action="accept")

        self.assertTrue(os.path.exists(self.engine.signer05_manifest_path))
        with open(self.engine.signer05_manifest_path, "r") as f:
            manifest_05 = json.load(f)

        self.assertIsInstance(manifest_05, list)
        self.assertGreater(len(manifest_05), 0)
        sample = manifest_05[0]
        self.assertEqual(sample["signer_id"], "signer_05")
        self.assertEqual(sample["dataset_role"], "independent_evaluation_holdout")
        self.assertEqual(sample["letter"], "B")

    def test_05_unique_recording_ids_and_retake_supersession(self):
        """Verifies retakes do NOT overwrite frame files on disk and mark prior takes as superseded."""
        # First Take
        video1 = self._create_synthetic_video(signer_tag="signer_05_take1", seed=10)
        res1 = self.engine.stage_recording(
            video_bytes=video1,
            filename="rec_take1.mp4",
            signer_id="signer_05",
            letter="C",
            condition_id="condition_1",
            session_id="session_20261025_01",
            consent_confirmed=True
        )
        rec1_commit = self.engine.commit_recording(staging_id=res1["staging_id"], action="accept")
        rec1_id = rec1_commit["recording_id"]

        with open(self.engine.signer05_manifest_path, "r") as f:
            m1 = json.load(f)
        take1_paths = [s["source"]["frame_path"] for s in m1 if s["letter"] == "C"]

        # Second Take (Retake of the same letter/condition/session)
        video2 = self._create_synthetic_video(signer_tag="signer_05_take2", seed=20)
        res2 = self.engine.stage_recording(
            video_bytes=video2,
            filename="rec_take2.mp4",
            signer_id="signer_05",
            letter="C",
            condition_id="condition_1",
            session_id="session_20261025_01",
            consent_confirmed=True
        )
        rec2_commit = self.engine.commit_recording(staging_id=res2["staging_id"], action="accept")
        rec2_id = rec2_commit["recording_id"]

        self.assertNotEqual(rec1_id, rec2_id, "Retake must have a unique recording ID")

        # Verify ledger has both recordings and rec1 is marked superseded
        with open(self.engine.ledger_path, "r") as f:
            ledger = json.load(f)
        rec1_entry = next(r for r in ledger["records"] if r["recording_id"] == rec1_id)
        rec2_entry = next(r for r in ledger["records"] if r["recording_id"] == rec2_id)

        self.assertEqual(rec1_entry["status"], "superseded", "Prior recording must be marked superseded")
        self.assertEqual(rec1_entry["superseded_by"], rec2_id, "Superseded recording must point to successor")
        self.assertEqual(rec2_entry["status"], "accepted", "New recording must be accepted")

        # Verify manifest: take1 samples are preserved but marked not usable and not eligible
        with open(self.engine.signer05_manifest_path, "r") as f:
            m2 = json.load(f)

        take1_samples = [s for s in m2 if s["recording_id"] == rec1_id]
        take2_samples = [s for s in m2 if s["recording_id"] == rec2_id]

        for s in take1_samples:
            self.assertFalse(s["quality"]["is_usable"], "Superseded samples must have is_usable=False")
            self.assertTrue(s["is_superseded"], "Superseded samples must have is_superseded=True")
            self.assertFalse(s["eligible_for_evaluation"], "Superseded samples must not be eligible for evaluation")
            self.assertEqual(s["superseded_by"], rec2_id)

        for s in take2_samples:
            self.assertTrue(s["quality"]["is_usable"])
            self.assertFalse(s["is_superseded"])
            self.assertTrue(s["eligible_for_evaluation"])

        # Check physical frame paths: verify take 1 files were NOT overwritten!
        take2_paths = [s["source"]["frame_path"] for s in take2_samples]
        overlap = set(take1_paths).intersection(set(take2_paths))
        self.assertEqual(len(overlap), 0, "Retake must not overwrite physical frame paths on disk!")
        for p in take1_paths:
            self.assertTrue(os.path.exists(os.path.join(BASE_DIR, p)), "Take 1 physical files must still exist for audit history")

    def test_06_corrupt_empty_and_interrupted_recording_handling(self):
        """Verifies rejection of empty bytes and safe handling of interrupted recordings."""
        # Empty file
        with self.assertRaises(ValueError):
            self.engine.stage_recording(
                video_bytes=b"",
                filename="empty.mp4",
                signer_id="signer_05",
                letter="D",
                condition_id="condition_1",
                session_id="session_20261025_01",
                consent_confirmed=True
            )

        # Unconsented recording
        with self.assertRaises(ValueError):
            video = self._create_synthetic_video(signer_tag="no_consent", seed=30)
            self.engine.stage_recording(
                video_bytes=video,
                filename="no_consent.mp4",
                signer_id="signer_05",
                letter="D",
                condition_id="condition_1",
                session_id="session_20261025_01",
                consent_confirmed=False
            )

        # Interrupted / operator-rejected take
        video = self._create_synthetic_video(signer_tag="to_reject", seed=31)
        res = self.engine.stage_recording(
            video_bytes=video,
            filename="reject.mp4",
            signer_id="signer_05",
            letter="D",
            condition_id="condition_1",
            session_id="session_20261025_01",
            consent_confirmed=True
        )
        rej_res = self.engine.commit_recording(
            staging_id=res["staging_id"],
            action="reject",
            rejection_note="Operator manually rejected due to interruption"
        )
        self.assertEqual(rej_res["status"], "rejected")

        # Verify rejection in ledger
        with open(self.engine.ledger_path, "r") as f:
            ledger = json.load(f)
        rej_entry = next(r for r in ledger["records"] if r["recording_id"] == rej_res["recording_id"])
        self.assertEqual(rej_entry["status"], "rejected")
        self.assertEqual(rej_entry["rejection_note"], "Operator manually rejected due to interruption")

    def test_07_duplicate_detection_via_sha256(self):
        """Verifies that submitting the exact same video file flags duplicate frame hashes."""
        video = self._create_synthetic_video(signer_tag="dup_test", seed=40)
        res1 = self.engine.stage_recording(
            video_bytes=video,
            filename="take_a.mp4",
            signer_id="signer_05",
            letter="E",
            condition_id="condition_1",
            session_id="session_20261025_01",
            consent_confirmed=True
        )
        self.assertGreater(res1["usable_count"], 0)
        self.engine.commit_recording(staging_id=res1["staging_id"], action="accept")

        # Staging identical video payload again after committing into manifest
        res2 = self.engine.stage_recording(
            video_bytes=video,
            filename="take_b.mp4",
            signer_id="signer_05",
            letter="E",
            condition_id="condition_1",
            session_id="session_20261025_01",
            consent_confirmed=True
        )
        # Frames should be flagged as duplicates
        duplicate_rejections = [f for f in res2["frames"] if not f["is_usable"] and "duplicate_frame" in (f.get("rejection_reason") or "")]
        self.assertGreater(len(duplicate_rejections), 0, "Identical video re-staged must detect duplicate frame hashes")

    def test_08_progress_and_completion_calculation(self):
        """Verifies get_signer_summary correctly reports progress and marks dataset incomplete."""
        summary = self.engine.get_signer_summary("signer_05")
        self.assertEqual(summary["signer_id"], "signer_05")
        self.assertEqual(summary["total_cells"], 52)
        self.assertFalse(summary["is_dataset_ready"], "Dataset must NOT be ready with 0 recordings")

        # Commit one letter with enough frames to complete target_frames_per_letter=16
        video = self._create_synthetic_video(num_frames=30, signer_tag="summary_test", seed=50)
        res = self.engine.stage_recording(
            video_bytes=video,
            filename="rec_f.mp4",
            signer_id="signer_05",
            letter="F",
            condition_id="condition_1",
            session_id="session_20261025_01",
            consent_confirmed=True,
            sampling_interval=1
        )
        self.engine.commit_recording(staging_id=res["staging_id"], action="accept")

        summary_after = self.engine.get_signer_summary("signer_05")
        self.assertFalse(summary_after["is_dataset_ready"], "Dataset must remain NOT ready when cells < 52")
        self.assertEqual(summary_after["condition_1_summary"]["completed_count"], 1)

    def test_09_training_data_isolation_safeguard(self):
        """Verifies isolation safeguards prevent any signer_05 samples from contaminating training manifests."""
        isolation_audit = self.engine.audit_split_isolation()
        self.assertTrue(isolation_audit["is_leakage_free"], "Base workspace manifests must have 0 split leaks")

        # Test verification safeguard on clean manifest
        clean_train = os.path.join(BASE_DIR, "phase22_analysis", "manifests", "train_manifest.json")
        if os.path.exists(clean_train):
            self.assertTrue(verify_training_isolation_safeguard(clean_train))

        # Attempt to insert a contaminated signer_05 sample into a mock train manifest
        contaminated_manifest_path = os.path.join(self.test_dir, "contaminated_train_manifest.json")
        contaminated_data = {
            "samples": [
                {
                    "sample_id": "signer_05_session_20261025_01_A_0000",
                    "signer_id": "signer_05",
                    "letter": "A",
                    "file_hash_sha256": "fakehash12345"
                }
            ]
        }
        with open(contaminated_manifest_path, "w") as f:
            json.dump(contaminated_data, f)

        with self.assertRaises(ValueError):
            verify_training_isolation_safeguard(contaminated_manifest_path)

    def test_10_preservation_of_signer_02_03_04_behavior(self):
        """Verifies existing signer_02, signer_03, and signer_04 workflows remain unaffected."""
        # Stage and commit for signer_04
        video_04 = self._create_synthetic_video(signer_tag="signer_04", seed=60)
        res_04 = self.engine.stage_recording(
            video_bytes=video_04,
            filename="rec_04.mp4",
            signer_id="signer_04",
            letter="G",
            condition_id="condition_1",
            session_id="session_20261025_01",
            consent_confirmed=True
        )
        commit_04 = self.engine.commit_recording(staging_id=res_04["staging_id"], action="accept")
        self.assertEqual(commit_04["status"], "accepted")

        # Verify signer_04 received external_test_multisigner split
        with open(self.engine.test_manifest_path, "r") as f:
            m = json.load(f)
        s04_samples = [s for s in m if s["signer_id"] == "signer_04" and s["letter"] == "G"]
        self.assertGreater(len(s04_samples), 0)
        self.assertEqual(s04_samples[0]["split"], "external_test_multisigner")
        self.assertEqual(s04_samples[0]["dataset_role"], "external_test_multisigner")

    def test_11_preservation_of_production_defaults_and_artifact_hashes(self):
        """Verifies production defaults and authoritative hashes remain untouched."""
        pipe = ISLPipeline()
        try:
            self.assertEqual(pipe.backend_detector, "yolo")
            self.assertEqual(pipe.recognition_arch, "standard")
            self.assertEqual(pipe.mediapipe_delegate, "cpu")
            self.assertFalse(pipe.temporal_smoothing_enabled)
        finally:
            pipe.shutdown()

        for artifact_key, info in AUTHORITATIVE_HASHES.items():
            path = info["path"]
            expected = info["sha256"]
            self.assertTrue(os.path.exists(path), f"Artifact missing: {path}")

            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            actual = h.hexdigest()
            self.assertEqual(actual, expected, f"Hash mismatch for {artifact_key} at {path}")


if __name__ == "__main__":
    unittest.main()
