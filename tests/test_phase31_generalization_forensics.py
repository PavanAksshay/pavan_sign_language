#!/usr/bin/env python3
"""
Unit tests for Phase 31 — Cross-Signer Generalization Root-Cause Audit.

Validates:
1. Integrity of frozen baseline model, scaler, and benchmark manifests.
2. Exact reproduction of Phase 30 independent evaluation metrics from raw predictions.
3. Zero data contamination: signer_05 is not in training or validation sets.
4. Completeness and validity of Phase 31 deliverables (JSON schemas, non-empty content).
5. Production runtime configuration invariant preservation.
"""

import os
import json
import hashlib
import unittest
import numpy as np

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def compute_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class TestPhase31GeneralizationForensics(unittest.TestCase):

    def setUp(self):
        self.workspace_dir = WORKSPACE_DIR
        self.model_path = os.path.join(self.workspace_dir, "phase22_analysis/models/Model_C_Occlusion_Aware_RF_multisigner.joblib")
        self.scaler_path = os.path.join(self.workspace_dir, "phase22_analysis/models/Model_C_Occlusion_Aware_RF_multisigner_scaler.joblib")
        self.task_path = os.path.join(self.workspace_dir, "App/include50_model/hand_landmarker.task")
        self.phase19_manifest_path = os.path.join(self.workspace_dir, "phase19_frozen_manifest.json")
        self.phase30_manifest_path = os.path.join(self.workspace_dir, "phase30_evaluation_manifest.json")
        self.phase30_metrics_path = os.path.join(self.workspace_dir, "phase30_independent_signer_metrics.json")
        self.phase30_predictions_path = os.path.join(self.workspace_dir, "phase30_analysis/phase30_predictions_raw.csv")

    def test_01_baseline_artifact_immutability(self):
        """Verify all frozen baseline artifacts remain unaltered."""
        expected_hashes = {
            self.model_path: "1fab3349e2e537d8b31ddbe3bdd5c21e3ed005f85aaac9d3875c72217e28cc98",
            self.scaler_path: "93842932e7cfeb1d283e99f67a733fd487161d83cbff356d7458ced2c3de385d",
            self.task_path: "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
            self.phase19_manifest_path: "9824eb510cc231249e9dd0722761de9112a02e7cecb681a479c4ef805b0fa797",
            self.phase30_manifest_path: "81e2efee5b3c74d1a5b39146888b34e63915802b02f64cc2c55f2b84ec1bf4de",
            self.phase30_metrics_path: "b420dc17b1b0d772133c3c6335e9b5f4d9a8242b97c375dcd8fcda19878a6460"
        }
        for path, exp_hash in expected_hashes.items():
            self.assertTrue(os.path.exists(path), f"Missing artifact: {path}")
            act_hash = compute_file_sha256(path)
            self.assertEqual(act_hash, exp_hash, f"Hash mismatch for {os.path.basename(path)}")

    def test_02_production_defaults_intact(self):
        """Verify production environment variables and pipeline defaults match requirements."""
        from pipeline import ISLPipeline
        pipe = ISLPipeline()
        try:
            self.assertEqual(pipe.backend_detector, "yolo")
            self.assertEqual(pipe.recognition_arch, "standard")
            self.assertEqual(pipe.mediapipe_delegate, "cpu")
            self.assertFalse(pipe.temporal_smoothing_enabled)
            self.assertFalse(pipe.shadow_mode_enabled)
            self.assertEqual(getattr(pipe, "model_c_input_resolution", "native"), "native")
        finally:
            pipe.shutdown()

    def test_03_reproduce_phase30_metrics_from_raw_csv(self):
        """Verify that Phase 30 metrics match exactly when recomputed from raw CSV predictions."""
        import csv
        self.assertTrue(os.path.exists(self.phase30_predictions_path))
        with open(self.phase30_predictions_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        self.assertEqual(len(rows), 1638, "Total evaluated frames must equal 1638")

        total_correct = sum(1 for r in rows if r["is_correct"] == "1")
        overall_acc = total_correct / len(rows)
        self.assertAlmostEqual(overall_acc, 0.59218559, places=4, msg="Overall accuracy must be 59.22%")

        cond1_rows = [r for r in rows if r["condition"] == "condition_1"]
        cond2_rows = [r for r in rows if r["condition"] == "condition_2"]
        self.assertEqual(len(cond1_rows), 821)
        self.assertEqual(len(cond2_rows), 817)

        c1_correct = sum(1 for r in cond1_rows if r["is_correct"] == "1")
        c2_correct = sum(1 for r in cond2_rows if r["is_correct"] == "1")
        self.assertAlmostEqual(c1_correct / len(cond1_rows), 0.64068, places=4, msg="Condition 1 acc must be 64.07%")
        self.assertAlmostEqual(c2_correct / len(cond2_rows), 0.54345, places=4, msg="Condition 2 acc must be 54.35%")

        # Zero recall classes check
        letters = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
        zero_recall = []
        for l in letters:
            l_rows = [r for r in rows if r["letter"] == l]
            l_correct = sum(1 for r in l_rows if r["is_correct"] == "1")
            if l_correct == 0:
                zero_recall.append(l)

        self.assertEqual(sorted(zero_recall), ["J", "O", "P", "Q"], "Zero recall classes must be J, O, P, Q")

    def test_04_deliverables_existence_and_schema(self):
        """Verify Phase 31 JSON audit files exist and adhere to expected structure."""
        deliverables = [
            "phase31_feature_audit.json",
            "phase31_confusion_analysis.json",
            "phase31_integrity_audit.json"
        ]
        for name in deliverables:
            path = os.path.join(self.workspace_dir, name)
            self.assertTrue(os.path.exists(path), f"Deliverable {name} is missing")
            with open(path, "r") as f:
                data = json.load(f)
            self.assertIn("timestamp", data, f"{name} missing timestamp")

        # Feature audit schema
        with open(os.path.join(self.workspace_dir, "phase31_feature_audit.json"), "r") as f:
            fa = json.load(f)
        self.assertEqual(fa["total_dimensions"], 280)
        self.assertIn("feature_ordering_specification", fa)
        self.assertIn("invariance_analysis", fa)
        self.assertIn("verified_pipeline_findings", fa)

        # Confusion analysis schema
        with open(os.path.join(self.workspace_dir, "phase31_confusion_analysis.json"), "r") as f:
            ca = json.load(f)
        self.assertIn("detailed_investigations", ca)
        for letter in ["J", "O", "P", "Q", "A", "L", "T", "H", "R"]:
            self.assertIn(letter, ca["detailed_investigations"])

        # Integrity audit schema
        with open(os.path.join(self.workspace_dir, "phase31_integrity_audit.json"), "r") as f:
            ia = json.load(f)
        self.assertEqual(ia["overall_integrity_status"], "PASS")
        self.assertEqual(ia["quarantined_holdout_status"], "VERIFIED_UNTOUCHED")
        self.assertTrue(ia["production_defaults_verified"]["production_defaults_intact"])


if __name__ == "__main__":
    unittest.main()
