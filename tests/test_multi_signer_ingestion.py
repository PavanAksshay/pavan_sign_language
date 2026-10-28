#!/usr/bin/env python3
"""
Unit tests for MultiSignerIngestionEngine in phase14c_data_ingestion.py.
"""

import unittest
import numpy as np
import cv2
from phase14c_data_ingestion import MultiSignerIngestionEngine


class TestMultiSignerIngestion(unittest.TestCase):

    def setUp(self):
        self.engine = MultiSignerIngestionEngine(min_blur_var=50.0, min_brightness=40.0, max_brightness=220.0)

    def test_duplicate_hash_rejection(self):
        """Identical image bytes must be rejected as duplicate on subsequent ingestion."""
        img = np.full((200, 200, 3), 128, dtype=np.uint8)
        # Add high-contrast circle so it's not blurry
        cv2.circle(img, (100, 100), 50, (255, 255, 255), -1)

        cond = {"lighting": "indoor_overhead", "background": "plain_wall", "camera_position": "frontal_eye_level", "distance_cm": 70}
        src1 = {"video_path": "vid1.mp4", "frame_index": 1, "timestamp_sec": 0.033}
        src2 = {"video_path": "vid1.mp4", "frame_index": 2, "timestamp_sec": 0.066}

        # Mock out _get_landmarker for unit test without mediapipe model loading
        class MockLandmarker:
            def detect(self, img):
                class MockRes:
                    hand_landmarks = []
                    handedness = []
                return MockRes()

        self.engine._landmarker = MockLandmarker()

        # First sample accepted or processed
        res1 = self.engine.process_sample(img, "signer_01", "sess_01", "rec_01", "A", cond, src1)
        # Note: res1 will be rejected due to no_hand_detected, but let's register its hash directly to test duplicate logic
        self.engine.seen_hashes[res1["sha256"]] = res1["sample_id"]

        # Ingesting same image again
        res2 = self.engine.process_sample(img, "signer_01", "sess_01", "rec_01", "A", cond, src2)
        self.assertEqual(res2["status"], "rejected")
        self.assertIn("duplicate_frame", res2["rejection_reason"])

    def test_blur_rejection(self):
        """Uniform / completely blurred image must be rejected for motion blur."""
        blurry = np.full((200, 200, 3), 128, dtype=np.uint8)
        cond = {"lighting": "indoor_overhead", "background": "plain_wall", "camera_position": "frontal_eye_level", "distance_cm": 70}
        src = {"video_path": "vid1.mp4", "frame_index": 1, "timestamp_sec": 0.033}

        res = self.engine.process_sample(blurry, "signer_01", "sess_01", "rec_01", "A", cond, src)
        self.assertEqual(res["status"], "rejected")
        self.assertIn("motion_blur", res["rejection_reason"])

    def test_brightness_boundaries(self):
        """Under-exposed or over-exposed images must be rejected."""
        dark = np.full((100, 100, 3), 10, dtype=np.uint8)
        bright = np.full((100, 100, 3), 250, dtype=np.uint8)
        cond = {"lighting": "indoor_overhead", "background": "plain_wall", "camera_position": "frontal_eye_level", "distance_cm": 70}
        src = {"video_path": "vid1.mp4", "frame_index": 1, "timestamp_sec": 0.033}

        res_dark = self.engine.process_sample(dark, "signer_01", "sess_01", "rec_01", "A", cond, src)
        res_bright = self.engine.process_sample(bright, "signer_01", "sess_01", "rec_01", "A", cond, src)

        self.assertEqual(res_dark["status"], "rejected")
        self.assertIn("underexposed", res_dark["rejection_reason"])
        self.assertEqual(res_bright["status"], "rejected")
        self.assertIn("overexposed", res_bright["rejection_reason"])

    def test_invalid_label_handling(self):
        """Invalid labels (e.g. non A-Z or multi-character) are marked as unlabeled."""
        img = np.full((200, 200, 3), 128, dtype=np.uint8)
        cv2.putText(img, "TEST", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cond = {"lighting": "indoor_overhead", "background": "plain_wall", "camera_position": "frontal_eye_level", "distance_cm": 70}
        src = {"video_path": "vid1.mp4", "frame_index": 1, "timestamp_sec": 0.033}

        res = self.engine.process_sample(img, "signer_01", "sess_01", "rec_01", "INVALID_123", cond, src)
        self.assertEqual(res["status"], "unlabeled")
        self.assertIn("invalid_label", res["rejection_reason"])

    def test_split_integrity_leakage_detection(self):
        """Split integrity audit detects signer and session overlaps."""
        # 1. Clean partitions
        train = [{"signer_id": "signer_01", "session_id": "sess_01", "source": {"sha256": "hash_a"}}]
        val = [{"signer_id": "signer_02", "session_id": "sess_02", "source": {"sha256": "hash_b"}}]
        test = [{"signer_id": "signer_03", "session_id": "sess_03", "source": {"sha256": "hash_c"}}]

        audit_clean = self.engine.audit_split_integrity(train, val, test)
        self.assertTrue(audit_clean["is_leakage_free"])
        self.assertTrue(audit_clean["is_strictly_signer_independent"])

        # 2. Injected signer overlap (signer_01 in train AND test)
        test_leaked = [{"signer_id": "signer_01", "session_id": "sess_04", "source": {"sha256": "hash_d"}}]
        audit_leaked = self.engine.audit_split_integrity(train, val, test_leaked)
        self.assertFalse(audit_leaked["is_strictly_signer_independent"])
        self.assertIn("signer_01", audit_leaked["overlaps"]["signer_train_test"])


if __name__ == "__main__":
    unittest.main()
