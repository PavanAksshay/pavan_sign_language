"""
Unit tests for Phase 23 — Frozen External Evaluation of Model C.
Validates:
1. Exact frozen benchmark evaluation denominator (3,329 frames).
2. Pre- and post-evaluation SHA256 integrity check (zero corruption of frozen files).
3. Evaluated Model C isolation (trained strictly on signer_01 + signer_04).
4. Machine-readable metrics and 80% macro-F1 gate verification.
5. Production default preservation (ISL_BACKEND_DETECTOR=yolo, ISL_RECOGNITION_ARCH=standard).
"""

import os
import sys
import json
import csv
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


class TestPhase23Evaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metrics_p = os.path.join(BASE_DIR, "phase23_external_metrics.json")
        cls.audit_p = os.path.join(BASE_DIR, "phase23_integrity_audit.json")
        cls.manifest_p = os.path.join(BASE_DIR, "phase23_model_manifest.json")
        cls.preds_csv_p = os.path.join(BASE_DIR, "phase23_predictions.csv")

    def test_01_external_denominator_and_predictions(self):
        """Verify exactly 3,329 predictions were generated across signer_02 and signer_03."""
        self.assertTrue(os.path.exists(self.preds_csv_p), "Missing phase23_predictions.csv")
        with open(self.preds_csv_p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        self.assertEqual(len(rows), 3329, f"Expected exactly 3,329 predictions, got {len(rows)}")
        signers = set(r["signer_id"] for r in rows)
        self.assertEqual(signers, {"signer_02", "signer_03"}, "Evaluated signers must be exclusively signer_02 and signer_03")

        s2_count = sum(1 for r in rows if r["signer_id"] == "signer_02")
        s3_count = sum(1 for r in rows if r["signer_id"] == "signer_03")
        self.assertEqual(s2_count, 1663, f"Expected 1,663 signer_02 frames, got {s2_count}")
        self.assertEqual(s3_count, 1666, f"Expected 1,666 signer_03 frames, got {s3_count}")

    def test_02_integrity_audit_and_zero_corruption(self):
        """Verify pre- and post-evaluation hashes are identical across all benchmark artifacts."""
        self.assertTrue(os.path.exists(self.audit_p), "Missing phase23_integrity_audit.json")
        with open(self.audit_p, "r", encoding="utf-8") as f:
            audit = json.load(f)

        self.assertTrue(audit.get("all_hashes_identical"), "Pre- and post-evaluation hashes must be identical")
        self.assertFalse(audit.get("corruption_detected"), "Zero corruption must be detected")
        self.assertEqual(audit.get("evaluation_sample_count"), 3329)

        pre = audit.get("pre_evaluation_sha256", {})
        post = audit.get("post_evaluation_sha256", {})
        for k in pre:
            self.assertEqual(pre[k], post[k], f"Hash mismatch for {k}")

    def test_03_model_provenance_and_isolation(self):
        """Verify Model C was trained strictly on signer_01 + signer_04 with 280 dimensions."""
        self.assertTrue(os.path.exists(self.manifest_p), "Missing phase23_model_manifest.json")
        with open(self.manifest_p, "r", encoding="utf-8") as f:
            m = json.load(f)

        model_info = m.get("evaluated_model", {})
        self.assertEqual(model_info.get("feature_dim"), 280)
        self.assertEqual(model_info.get("training_signers"), ["signer_01", "signer_04"])
        self.assertEqual(model_info.get("training_samples"), 4698)

    def test_04_macro_f1_target_gate(self):
        """Verify Model C exceeds the 80% macro-F1 benchmark threshold."""
        self.assertTrue(os.path.exists(self.metrics_p), "Missing phase23_external_metrics.json")
        with open(self.metrics_p, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        overall = metrics.get("overall_external_metrics", {})
        macro_f1 = overall.get("macro_f1", 0.0)
        top1_acc = overall.get("top1_accuracy", 0.0)

        self.assertGreaterEqual(macro_f1, 0.80, f"Macro-F1 {macro_f1} did not meet 80% target")
        self.assertGreaterEqual(top1_acc, 0.90, f"Top-1 accuracy {top1_acc} was below 90%")
        self.assertTrue(metrics.get("target_passed"))

    def test_05_production_defaults_intact(self):
        """Verify pipeline production defaults remain preserved."""
        from pipeline import ISLPipeline
        p = ISLPipeline()
        self.assertEqual(p.backend_detector, "yolo")
        self.assertEqual(p.recognition_arch, "standard")
        self.assertEqual(p.temporal_smoothing_enabled, False)


if __name__ == "__main__":
    unittest.main()
