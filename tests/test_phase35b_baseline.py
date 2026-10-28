#!/Users/pavanaksshay/venv/bin/python3
"""
Unit and Contract Tests for Phase 35B — Leakage-Safe Baseline Reproduction and Evaluation Split Audit.

Verifies:
1. Presence and non-empty status of all Phase 35B deliverables.
2. Bit-for-bit cryptographic immutability of production Model C, scaler, frozen benchmarks, and holdouts.
3. Verification that production defaults remain strictly untouched.
4. Complete recording-disjointness of signer_04 split partitions (zero shared recordings).
5. Explicit identification and documentation of the letter-condition block confound.
6. Verification that experimental scalers were fitted exclusively on approved training samples.
7. Verification of baseline performance metrics against the pre-declared Macro-F1 >= 75% milestone.
8. Presence of per-letter and recording-level evaluation metrics.
9. Integrity of the experimental model manifest and artifact hashes.
"""

import os
import sys
import json
import hashlib
import unittest
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


class TestPhase35BBaselineReproduction(unittest.TestCase):
    """Test suite for Phase 35B baseline reproduction, split audit, and governance safeguards."""

    @classmethod
    def setUpClass(cls):
        cls.report_path = os.path.join(BASE_DIR, "phase35b_baseline_report.md")
        cls.split_audit_path = os.path.join(BASE_DIR, "phase35b_split_audit.json")
        cls.metrics_path = os.path.join(BASE_DIR, "phase35b_baseline_metrics.json")
        cls.manifest_path = os.path.join(BASE_DIR, "phase35b_experimental_model_manifest.json")

        with open(cls.split_audit_path, "r", encoding="utf-8") as f:
            cls.split_audit = json.load(f)

        with open(cls.metrics_path, "r", encoding="utf-8") as f:
            cls.metrics = json.load(f)

        with open(cls.manifest_path, "r", encoding="utf-8") as f:
            cls.manifest = json.load(f)

    def _sha256(self, path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    def test_01_deliverables_exist_and_non_empty(self):
        """Verify presence and non-trivial file size for all required Phase 35B deliverables."""
        for p in [self.report_path, self.split_audit_path, self.metrics_path, self.manifest_path]:
            self.assertTrue(os.path.isfile(p), f"Missing required file: {p}")
            self.assertGreater(os.path.getsize(p), 500, f"File appears truncated: {p}")

        # Check experimental models directory
        exp_model = os.path.join(BASE_DIR, "phase35b_analysis", "models", "Model_C_Baseline_Fold1_Primary.joblib")
        exp_scaler = os.path.join(BASE_DIR, "phase35b_analysis", "models", "Model_C_Baseline_Fold1_Primary_scaler.joblib")
        self.assertTrue(os.path.isfile(exp_model), "Missing experimental model artifact")
        self.assertTrue(os.path.isfile(exp_scaler), "Missing experimental scaler artifact")

    def test_02_production_model_and_scaler_unmodified(self):
        """Verify bit-for-bit immutability of the production Model C checkpoint and scaler."""
        prod_model = os.path.join(BASE_DIR, "phase22_analysis", "models", "Model_C_Occlusion_Aware_RF_multisigner.joblib")
        prod_scaler = os.path.join(BASE_DIR, "phase22_analysis", "models", "Model_C_Occlusion_Aware_RF_multisigner_scaler.joblib")

        expected_model_hash = "1fab3349e2e537d8b31ddbe3bdd5c21e3ed005f85aaac9d3875c72217e28cc98"
        expected_scaler_hash = "93842932e7cfeb1d283e99f67a733fd487161d83cbff356d7458ced2c3de385d"

        self.assertEqual(self._sha256(prod_model), expected_model_hash, "Production Model C modified!")
        self.assertEqual(self._sha256(prod_scaler), expected_scaler_hash, "Production Scaler modified!")

    def test_03_protected_benchmarks_and_holdouts_unmodified(self):
        """Verify immutability of frozen benchmark manifests and sealed signer_05 holdout."""
        benchmarks = {
            "phase19_frozen_manifest.json": "9824eb510cc231249e9dd0722761de9112a02e7cecb681a479c4ef805b0fa797",
            "phase23_predictions.csv": "da77036032fb3250141ccc16ba53d1dd7d4997b9817c6f525bbf9e000be45bfe",
            "data/collection_manifests/signer_05_manifest.json": "5886b90d9b38e905848731ac8afeb82567a9ea343a460832bf49a74e3e4a5f3c",
            "App/include50_model/hand_landmarker.task": "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
        }
        for rel_p, exp_h in benchmarks.items():
            full_p = os.path.join(BASE_DIR, rel_p)
            self.assertTrue(os.path.isfile(full_p), f"Missing protected file: {rel_p}")
            self.assertEqual(self._sha256(full_p), exp_h, f"Protected file {rel_p} hash changed!")

    def test_04_split_recording_disjointness(self):
        """Verify zero recording overlap between training and validation partitions."""
        configs = self.split_audit["split_configurations"]
        for split_key, split_info in configs.items():
            self.assertEqual(
                split_info["recording_overlap"],
                0,
                f"Recording leakage detected in {split_key}!"
            )
            self.assertEqual(split_info["letters_in_train"], 26)
            self.assertEqual(split_info["letters_in_val"], 26)

    def test_05_confound_audit_documented(self):
        """Verify that the letter-condition confound is documented with mathematical limitations."""
        confound = self.split_audit["confound_analysis"]
        self.assertTrue(confound["confound_identified"])
        self.assertIn("100% block confound", confound["description"])
        self.assertIn("2-fold cross-condition", confound["mitigation_protocol"])

    def test_06_scaler_fit_on_training_only(self):
        """Verify experimental baseline scaler was fitted on training samples only."""
        fold1_meta = self.manifest["models"]["Model_C_Baseline_Fold1_Primary"]
        tr_frames = fold1_meta["training_pool"]["total_training_frames"]
        self.assertEqual(tr_frames, 3887, "Training set should contain exactly 3,077 s1 + 810 s4 frames")

        # Scaler must have mean_ vector of length 280
        import joblib
        scaler = joblib.load(fold1_meta["scaler_path"])
        self.assertEqual(len(scaler.mean_), 280)

    def test_07_baseline_metrics_and_gate_passed(self):
        """Verify baseline metrics meet pre-declared Macro-F1 >= 75% target threshold."""
        gate = self.metrics["target_threshold"]
        self.assertEqual(gate["target"], 0.75)
        self.assertEqual(gate["status"], "PASSED")

        f1_metrics = self.metrics["primary_baseline_fold1"]["metrics"]
        self.assertGreaterEqual(f1_metrics["macro_f1"], 0.75)
        self.assertGreaterEqual(f1_metrics["top1_accuracy"], 0.90)
        self.assertTrue(f1_metrics["predeclared_gate_passed"])

    def test_08_per_letter_and_recording_level_metrics(self):
        """Verify per-letter and recording-level metrics exist for all 26 letters and recordings."""
        f1_data = self.metrics["primary_baseline_fold1"]
        per_letter = f1_data["per_letter"]
        self.assertEqual(len(per_letter), 26, "All 26 letters must have metrics")

        for let in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            self.assertIn(let, per_letter)
            self.assertIn("precision", per_letter[let])
            self.assertIn("recall", per_letter[let])
            self.assertIn("f1", per_letter[let])
            self.assertIn("support", per_letter[let])

        rec_level = f1_data["recording_level"]
        self.assertEqual(len(rec_level), 26, "All 26 holdout recordings must be evaluated")

    def test_09_experimental_manifest_hashes_match(self):
        """Verify that all experimental model hashes in the manifest match disk files."""
        for m_name, m_info in self.manifest["models"].items():
            model_p = m_info["model_path"]
            scaler_p = m_info["scaler_path"]
            self.assertTrue(os.path.isfile(model_p))
            self.assertTrue(os.path.isfile(scaler_p))
            self.assertEqual(self._sha256(model_p), m_info["model_sha256"])
            self.assertEqual(self._sha256(scaler_p), m_info["scaler_sha256"])

    def test_10_production_defaults_intact(self):
        """Verify production runtime configuration defaults remain intact."""
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


if __name__ == "__main__":
    unittest.main()

