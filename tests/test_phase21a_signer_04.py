#!/usr/bin/env python3
"""
Unit and integration tests for Phase 21A: Adding Signer 04 to the Recording Interface.
Verifies:
1. signer_04 is recognized in ALLOWED_SIGNERS and metadata_schema.json.
2. The recording interface renders signer_04 as a selectable option.
3. End-to-end staging and committing with signer_04 persists the correct ID,
   stores raw video in data/raw_recordings/signer_04/ and extracted frames in
   data/extracted_frames/signer_04/, with ledger and manifest metadata verified.
4. Changing signer selections isolates directories without mixing.
5. Existing signer_02 and signer_03 options and data remain intact.
"""

import os
import sys
import json
import shutil
import tempfile
import unittest
import urllib.request
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from phase18_collector import (
    ISLCollectionEngine,
    validate_sample_schema,
    ALLOWED_SIGNERS,
    ALL_VALID_SIGNERS,
    CONDITION_PRESETS
)


class TestPhase21ASigner04(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="isl_phase21a_test_")
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

        # Mock hand detection for rapid test execution without loading neural network weights
        self.engine.detect_hands_in_frame = lambda frame: (1, ["Right"], [0.95], False, [])

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_synthetic_video(self, num_frames=15, fps=20.0, width=320, height=240, signer_tag="signer_04"):
        """Generates a valid MP4 test video in memory."""
        import uuid
        tmp_video = os.path.join(self.test_dir, f"test_vid_{uuid.uuid4().hex[:6]}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(tmp_video, fourcc, fps, (width, height))

        for i in range(num_frames):
            frame = np.full((height, width, 3), 128, dtype=np.uint8)
            cv2.circle(frame, (100 + i * 4, 120), 30, (255, 255, 255), -1)
            cv2.putText(frame, f"{signer_tag}_F{i}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            out.write(frame)

        out.release()
        with open(tmp_video, "rb") as f:
            video_bytes = f.read()
        if os.path.exists(tmp_video):
            os.remove(tmp_video)
        return video_bytes

    def test_01_signer_04_in_allowed_constants(self):
        """Verifies signer_04 is declared in ALLOWED_SIGNERS and ALL_VALID_SIGNERS."""
        self.assertIn("signer_04", ALLOWED_SIGNERS, "signer_04 must be in ALLOWED_SIGNERS")
        self.assertIn("signer_04", ALL_VALID_SIGNERS, "signer_04 must be in ALL_VALID_SIGNERS")
        self.assertIn("signer_02", ALLOWED_SIGNERS, "signer_02 must remain in ALLOWED_SIGNERS")
        self.assertIn("signer_03", ALLOWED_SIGNERS, "signer_03 must remain in ALLOWED_SIGNERS")

    def test_02_metadata_schema_accepts_signer_04(self):
        """Verifies schema validation succeeds for signer_04 and rejects invalid signers."""
        schema_path = os.path.join(BASE_DIR, "phase14b_analysis", "metadata_schema.json")
        with open(schema_path, "r") as f:
            schema = json.load(f)
        enum_signers = schema["properties"]["signer_id"]["enum"]
        self.assertIn("signer_04", enum_signers, "metadata_schema.json must include signer_04 in enum")

        valid_sample = {
            "sample_id": "signer_04_session_20261023_01_A_0000",
            "signer_id": "signer_04",
            "session_id": "session_20261023_01",
            "recording_id": "rec_signer_04_condition_1_A_12345",
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
                "video_path": "data/raw_recordings/signer_04/condition_1/rec_signer_04_condition_1_A_12345.mp4",
                "frame_path": "data/extracted_frames/signer_04/condition_1/A/frame_0000.png",
                "frame_index": 0,
                "timestamp_sec": 0.0,
                "sha256": "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
            }
        }
        is_valid, err = validate_sample_schema(valid_sample)
        self.assertTrue(is_valid, f"Sample with signer_04 failed schema validation: {err}")

        # Invalid signer must fail
        invalid_sample = dict(valid_sample)
        invalid_sample["signer_id"] = "signer_99"
        is_valid_inv, err_inv = validate_sample_schema(invalid_sample)
        self.assertFalse(is_valid_inv, "Sample with signer_99 must fail schema validation")

    def test_03_frontend_renders_signer_04_option(self):
        """Verifies template rendering and/or live GET /collect returns HTML with signer_04 selectable option."""
        from jinja2 import Environment, FileSystemLoader

        env = Environment(loader=FileSystemLoader(os.path.join(BASE_DIR, "web", "templates")))
        template = env.get_template("collect.html")
        rendered_html = template.render(
            signers=ALLOWED_SIGNERS,
            conditions=CONDITION_PRESETS,
            classes=[chr(65 + i) for i in range(26)]
        )

        self.assertIn('value="signer_04"', rendered_html, "Rendered HTML must contain value='signer_04'")
        self.assertIn("signer_04", rendered_html, "Rendered HTML must display signer_04")
        self.assertIn('value="signer_02"', rendered_html, "signer_02 must remain in HTML")
        self.assertIn('value="signer_03"', rendered_html, "signer_03 must remain in HTML")

        # Also test live server endpoint if reachable
        try:
            req = urllib.request.Request("http://localhost:8050/collect")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    live_html = resp.read().decode("utf-8")
                    self.assertIn('value="signer_04"', live_html)
        except Exception:
            pass  # Live server test is optional if process not bound

    def test_04_end_to_end_staging_and_commit_for_signer_04(self):
        """
        Tests end-to-end recording staging and committing for signer_04:
        - Staging saves correct metadata with signer_id=signer_04.
        - Committing stores raw video to raw/signer_04/<cond>/ and frames to frames/signer_04/<cond>/<letter>/.
        - Ledger record has signer_id=signer_04.
        - Test manifest has signer_id=signer_04.
        """
        video_bytes = self._create_synthetic_video(num_frames=12, signer_tag="signer_04")

        # 1. Stage recording
        stage_res = self.engine.stage_recording(
            video_bytes=video_bytes,
            filename="test_signer_04_rec.mp4",
            signer_id="signer_04",
            letter="B",
            condition_id="condition_1",
            consent_confirmed=True,
            sampling_interval=2,
            operator_notes="Phase 21A test recording"
        )

        self.assertEqual(stage_res["signer_id"], "signer_04")
        self.assertEqual(stage_res["letter"], "B")
        self.assertEqual(stage_res["condition_id"], "condition_1")
        self.assertGreater(stage_res["usable_count"], 0)
        staging_id = stage_res["staging_id"]

        # 2. Commit recording (accept)
        commit_res = self.engine.commit_recording(
            staging_id=staging_id,
            action="accept"
        )

        self.assertEqual(commit_res["status"], "accepted")
        self.assertEqual(commit_res["signer_id"], "signer_04")
        self.assertEqual(commit_res["letter"], "B")

        # 3. Verify filesystem storage locations
        raw_video_path = os.path.join(self.test_dir, "raw", "signer_04", "condition_1", f"{commit_res['recording_id']}.mp4")
        self.assertTrue(os.path.exists(raw_video_path), f"Raw video not found at: {raw_video_path}")
        self.assertIn(os.path.join("raw", "signer_04", "condition_1"), raw_video_path)

        frames_dir = os.path.join(self.test_dir, "frames", "signer_04", "condition_1", "B")
        self.assertTrue(os.path.isdir(frames_dir), f"Extracted frames dir not found: {frames_dir}")
        saved_frames = [f for f in os.listdir(frames_dir) if f.endswith(".jpg") or f.endswith(".png")]
        self.assertEqual(len(saved_frames), commit_res["usable_frames_saved"])
        self.assertGreater(len(saved_frames), 0)

        # 4. Verify collection ledger
        ledger = self.engine.load_ledger()
        self.assertEqual(len(ledger["records"]), 1)
        ledger_rec = ledger["records"][0]
        self.assertEqual(ledger_rec["signer_id"], "signer_04")
        self.assertEqual(ledger_rec["letter"], "B")
        self.assertEqual(ledger_rec["condition_id"], "condition_1")
        self.assertIn("signer_04", ledger_rec["raw_video_path"])

        # 5. Verify test manifest
        manifest = self.engine.load_test_manifest()
        self.assertEqual(len(manifest), commit_res["usable_frames_saved"])
        for item in manifest:
            self.assertEqual(item["signer_id"], "signer_04")
            self.assertEqual(item["letter"], "B")
            self.assertIn("signer_04", item["source"]["frame_path"])

    def test_05_signer_isolation_no_directory_mixing(self):
        """
        Verifies that recordings from signer_02, signer_03, and signer_04
        are strictly isolated in their respective directories without mixing.
        """
        signers = ["signer_02", "signer_03", "signer_04"]
        for s in signers:
            v_bytes = self._create_synthetic_video(num_frames=8, signer_tag=s)
            st_res = self.engine.stage_recording(
                video_bytes=v_bytes,
                filename=f"{s}_test.mp4",
                signer_id=s,
                letter="C",
                condition_id="condition_2",
                consent_confirmed=True,
                sampling_interval=2
            )
            self.engine.commit_recording(st_res["staging_id"], action="accept")

        # Verify raw directories
        for s in signers:
            s_raw_dir = os.path.join(self.test_dir, "raw", s, "condition_2")
            self.assertTrue(os.path.isdir(s_raw_dir), f"Missing raw dir for {s}")
            vids = os.listdir(s_raw_dir)
            self.assertEqual(len(vids), 1)
            self.assertTrue(vids[0].startswith(f"rec_{s}_condition_2_C_"))

            s_frames_dir = os.path.join(self.test_dir, "frames", s, "condition_2", "C")
            self.assertTrue(os.path.isdir(s_frames_dir), f"Missing frames dir for {s}")
            frames = os.listdir(s_frames_dir)
            self.assertGreater(len(frames), 0)
            for f in frames:
                self.assertTrue(f.startswith(f"{s}_"), f"Frame {f} does not start with {s}")

    def test_06_progress_matrix_includes_signer_04(self):
        """Verifies get_progress() contains matrix entries for signer_02, signer_03, and signer_04."""
        progress = self.engine.get_progress()
        matrix = progress["matrix"]
        self.assertIn("signer_02", matrix)
        self.assertIn("signer_03", matrix)
        self.assertIn("signer_04", matrix)
        for s in ["signer_02", "signer_03", "signer_04"]:
            self.assertIn("condition_1", matrix[s])
            self.assertIn("condition_2", matrix[s])
            self.assertEqual(len(matrix[s]["condition_1"]["letters"]), 26)

    def test_07_live_dataset_remains_unmodified(self):
        """Verifies that the frozen external test partition (first 3,329 frames) from signer_02 and signer_03 remains intact."""
        live_manifest_path = os.path.join(BASE_DIR, "data", "collection_manifests", "test_multisigner_manifest.json")
        with open(live_manifest_path, "r") as f:
            live_samples = json.load(f)
        self.assertGreaterEqual(len(live_samples), 3329, "Live test manifest must retain at least the 3,329 original frames")
        frozen_partition = live_samples[:3329]
        self.assertEqual(len(frozen_partition), 3329)
        signers = set(s["signer_id"] for s in frozen_partition)
        self.assertEqual(signers, {"signer_02", "signer_03"}, "Frozen test partition must only contain signer_02 and signer_03")


if __name__ == "__main__":
    unittest.main()
