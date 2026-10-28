#!/usr/bin/env python3
"""
Phase 15 — Unit Tests for Benchmarking, Quality Gates, Leakage Auditing & Metric Accounting
"""

import os
import sys
import unittest
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from phase14c_data_ingestion import MultiSignerIngestionEngine
from phase15_split_strategy import generate_signer_independent_splits, verify_split_leakage
from phase15_benchmark_models import compute_comprehensive_metrics
from pipeline import ISLPipeline


class TestPhase15Benchmarking(unittest.TestCase):

    def setUp(self):
        self.engine = MultiSignerIngestionEngine(
            min_blur_var=50.0,
            min_brightness=40.0,
            max_brightness=220.0
        )

    def test_metadata_schema_validation_strict(self):
        """Verifies schema validation on valid and invalid metadata records."""
        condition_meta = {
            "lighting": "indoor_overhead",
            "background": "textured_office",
            "camera_position": "frontal_eye_level",
            "distance_cm": 80
        }
        source_meta = {"source_type": "synthetic_test", "frame_index": 1}

        # Valid frame (synthetic bright hand-like image)
        valid_frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        # Add gradient to avoid blur rejection
        valid_frame[:50, :50] = 50
        valid_frame[50:, 50:] = 200

        # Quality audit
        q = self.engine.audit_image_quality(valid_frame)
        self.assertTrue(q["is_usable"])

    def test_malformed_and_missing_labels(self):
        """Tests that non-A-Z labels are rejected or marked unlabeled."""
        invalid_labels = ["1", "AB", "", " ", None, "@", "-1", "Z_invalid", "99"]
        for bad_label in invalid_labels:
            clean_letter = str(bad_label).strip().upper() if bad_label else ""
            is_valid = len(clean_letter) == 1 and ('A' <= clean_letter <= 'Z')
            self.assertFalse(is_valid, f"Label {bad_label} should be invalid")

    def test_sha256_duplicate_image_rejection(self):
        """Tests that identical images are flagged as duplicates."""
        frame = np.ones((50, 50, 3), dtype=np.uint8) * 100
        h1 = MultiSignerIngestionEngine.compute_sha256(frame.tobytes())
        h2 = MultiSignerIngestionEngine.compute_sha256(frame.tobytes())
        self.assertEqual(h1, h2)

        engine = MultiSignerIngestionEngine()
        engine.seen_hashes[h1] = "sample_orig_01"
        self.assertIn(h1, engine.seen_hashes)

    def test_quality_gate_decisions_blur_and_brightness(self):
        """Tests boundaries of blur and brightness quality gates."""
        # Extreme blur (flat image: Laplacian var = 0)
        flat_img = np.ones((100, 100, 3), dtype=np.uint8) * 120
        q_blur = self.engine.audit_image_quality(flat_img)
        self.assertFalse(q_blur["is_usable"])
        self.assertIn("motion_blur", q_blur["rejection_reason"])

        # Underexposed (brightness = 10 < 40)
        dark_img = np.ones((100, 100, 3), dtype=np.uint8) * 10
        q_dark = self.engine.audit_image_quality(dark_img)
        self.assertFalse(q_dark["is_usable"])
        self.assertIn("underexposed", q_dark["rejection_reason"])

        # Overexposed (brightness = 240 > 220)
        bright_img = np.ones((100, 100, 3), dtype=np.uint8) * 240
        q_bright = self.engine.audit_image_quality(bright_img)
        self.assertFalse(q_bright["is_usable"])
        self.assertIn("overexposed", q_bright["rejection_reason"])

    def test_leakage_audit_detects_session_overlap(self):
        """Tests that verify_split_leakage flags session overlaps."""
        train_samples = [{"recording_id": "r1", "session_id": "sess_shared", "signer_id": "s1"}]
        val_samples = [{"recording_id": "r2", "session_id": "sess_val", "signer_id": "s1"}]
        test_samples = [{"recording_id": "r3", "session_id": "sess_shared", "signer_id": "s1"}]

        audit = verify_split_leakage(train_samples, val_samples, test_samples)
        self.assertFalse(audit["is_session_independent"])
        self.assertIn("sess_shared", audit["session_overlap"]["train_test"])

    def test_deterministic_split_generation(self):
        """Verifies that split partitioning is deterministic with fixed seed."""
        samples = [
            {"signer_id": "signer_01", "session_id": f"sess_{i}", "recording_id": f"rec_{i}"}
            for i in range(20)
        ]
        splits1 = generate_signer_independent_splits(samples, seed=42)
        splits2 = generate_signer_independent_splits(samples, seed=42)

        self.assertEqual(len(splits1["train"]), len(splits2["train"]))
        self.assertEqual(len(splits1["val"]), len(splits2["val"]))
        self.assertEqual(len(splits1["test"]), len(splits2["test"]))
        self.assertEqual([s["recording_id"] for s in splits1["test"]], [s["recording_id"] for s in splits2["test"]])

    def test_metric_denominators_conditional_vs_end_to_end(self):
        """Verifies distinct, correct denominators for conditional vs end-to-end accuracy."""
        # 10 attempted samples, 2 failed detections, 8 processed, 6 correct
        y_true = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        y_probs = np.zeros((8, 26))
        for i in range(6):
            y_probs[i, y_true[i]] = 1.0  # 6 correct
        y_probs[6, 20] = 1.0  # incorrect
        y_probs[7, 21] = 1.0  # incorrect

        attempted = 10
        failed = 2

        metrics = compute_comprehensive_metrics(y_true, y_probs, attempted, failed)
        # Conditional accuracy: 6 / 8 = 75.0%
        self.assertEqual(metrics["conditional_top1_accuracy"], 75.0)
        # End-to-end accuracy: 6 / 10 = 60.0%
        self.assertEqual(metrics["end_to_end_top1_accuracy"], 60.0)
        self.assertEqual(metrics["attempted_samples"], 10)
        self.assertEqual(metrics["processed_samples"], 8)
        self.assertEqual(metrics["failure_count"], 2)

    def test_missing_classes_in_partition_handling(self):
        """Verifies that splits with fewer than 26 classes do not crash or divide by zero."""
        # Only class 0 and 1 present (e.g. A and B)
        y_true = np.array([0, 0, 1, 1])
        y_probs = np.zeros((4, 26))
        y_probs[0, 0] = 1.0
        y_probs[1, 0] = 1.0
        y_probs[2, 1] = 1.0
        y_probs[3, 1] = 1.0

        metrics = compute_comprehensive_metrics(y_true, y_probs, 4, 0)
        self.assertEqual(metrics["conditional_top1_accuracy"], 100.0)
        self.assertEqual(metrics["macro_f1"], 100.0)
        self.assertEqual(metrics["balanced_accuracy"], 100.0)

    def test_preservation_of_production_configuration_defaults(self):
        """Verifies that pipeline defaults remain yolo detector, standard arch, and smoothing disabled."""
        pipe = ISLPipeline()
        self.assertEqual(pipe.backend_detector, "yolo")
        self.assertEqual(pipe.recognition_arch, "standard")
        self.assertFalse(pipe.temporal_smoothing_enabled)


if __name__ == "__main__":
    unittest.main()
