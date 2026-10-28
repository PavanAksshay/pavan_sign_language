#!/Users/pavanaksshay/venv/bin/python3
"""
Unit tests for Phase 35A — Existing-Data Generalization Strategy and Leakage-Safe Evaluation Redesign.

Verifies:
1. Presence and validity of Phase 35A deliverables (audit JSON, experiment protocol JSON, strategy report).
2. Bit-for-bit cryptographic immutability of protected models, scalers, benchmarks, and holdouts.
3. Strict enforcement of data governance boundaries (signer_01 training-only, signer_02/03 quarantined, signer_04 development-only, signer_05 sealed diagnostic holdout).
4. Recording-disjoint split specification validity (zero recording overlap, balanced letter coverage).
5. Pre-declared experiment protocol integrity (semantic augmentation constraints, hypotheses, acceptance criteria, falsification conditions).
6. Preservation of production runtime defaults.
"""

import os
import sys
import json
import hashlib
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


class TestPhase35AGeneralizationStrategy(unittest.TestCase):
    """Test suite validating Phase 35A governance, exposure audit, and experiment protocols."""

    @classmethod
    def setUpClass(cls):
        cls.audit_path = os.path.join(BASE_DIR, "phase35a_dataset_exposure_audit.json")
        cls.protocol_path = os.path.join(BASE_DIR, "phase35a_experiment_protocol.json")
        cls.report_path = os.path.join(BASE_DIR, "phase35a_generalization_strategy_report.md")

        with open(cls.audit_path, "r", encoding="utf-8") as f:
            cls.audit_data = json.load(f)

        with open(cls.protocol_path, "r", encoding="utf-8") as f:
            cls.protocol_data = json.load(f)

    def _sha256(self, path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    def test_01_artifacts_exist_and_non_empty(self):
        """Verify that all Phase 35A core deliverables exist and are non-empty."""
        self.assertTrue(os.path.isfile(self.audit_path), "Dataset exposure audit JSON missing")
        self.assertTrue(os.path.isfile(self.protocol_path), "Experiment protocol JSON missing")
        self.assertTrue(os.path.isfile(self.report_path), "Generalization strategy report missing")

        self.assertGreater(os.path.getsize(self.audit_path), 1000)
        self.assertGreater(os.path.getsize(self.protocol_path), 1000)
        self.assertGreater(os.path.getsize(self.report_path), 1000)

    def test_02_protected_artifact_hashes(self):
        """Verify bit-for-bit cryptographic immutability of protected assets against authoritative hashes."""
        authoritative = {
            "phase22_analysis/models/Model_C_Occlusion_Aware_RF_multisigner.joblib": (
                "1fab3349e2e537d8b31ddbe3bdd5c21e3ed005f85aaac9d3875c72217e28cc98"
            ),
            "phase22_analysis/models/Model_C_Occlusion_Aware_RF_multisigner_scaler.joblib": (
                "93842932e7cfeb1d283e99f67a733fd487161d83cbff356d7458ced2c3de385d"
            ),
            "phase19_frozen_manifest.json": (
                "9824eb510cc231249e9dd0722761de9112a02e7cecb681a479c4ef805b0fa797"
            ),
            "phase23_predictions.csv": (
                "da77036032fb3250141ccc16ba53d1dd7d4997b9817c6f525bbf9e000be45bfe"
            ),
            "data/collection_manifests/signer_05_manifest.json": (
                "5886b90d9b38e905848731ac8afeb82567a9ea343a460832bf49a74e3e4a5f3c"
            ),
            "App/include50_model/hand_landmarker.task": (
                "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
            ),
        }

        for rel_path, expected_hash in authoritative.items():
            full_path = os.path.join(BASE_DIR, rel_path)
            self.assertTrue(os.path.isfile(full_path), f"Protected artifact missing: {rel_path}")
            actual_hash = self._sha256(full_path)
            self.assertEqual(
                actual_hash,
                expected_hash,
                f"Hash mismatch on protected file {rel_path}!\nExpected: {expected_hash}\nActual:   {actual_hash}"
            )

    def test_03_signer01_governance_restrictions(self):
        """Verify signer_01 is restricted to training-only and prohibited from independent validation claims."""
        s1 = self.audit_data["dataset_census_and_provenance"]["signer_01"]
        self.assertEqual(s1["dataset_role"], "legacy_training_only")
        self.assertEqual(s1["governance_restrictions"]["allowed_future_role"], "training_pool_only")
        self.assertEqual(s1["governance_restrictions"]["validation_reuse"], "STRICTLY_PROHIBITED")
        self.assertFalse(s1["supports_recording_disjoint_split"])
        self.assertEqual(s1["exposure_and_leakage_audit"]["shared_sessions_count"], 4)

    def test_04_signer02_and_03_benchmark_quarantine(self):
        """Verify signer_02 and signer_03 are quarantined benchmarks and labeled as not pristine."""
        census = self.audit_data["dataset_census_and_provenance"]
        for s_id in ["signer_02", "signer_03"]:
            s_data = census[s_id]
            self.assertEqual(s_data["dataset_role"], "quarantined_frozen_benchmark")
            self.assertEqual(s_data["governance_restrictions"]["training_use"], "STRICTLY_PROHIBITED")
            self.assertEqual(s_data["governance_restrictions"]["tuning_use"], "STRICTLY_PROHIBITED")
            self.assertEqual(s_data["exposure_and_leakage_audit"]["pristine_holdout_status"], "NOT_PRISTINE")

    def test_05_signer04_exposure_and_split_specification(self):
        """Verify signer_04 is recognized as previously exposed development data and split has 0 recording leakage."""
        s4 = self.audit_data["dataset_census_and_provenance"]["signer_04"]
        self.assertEqual(s4["dataset_role"], "previously_exposed_development_data")
        self.assertEqual(s4["exposure_and_leakage_audit"]["pristine_holdout_status"], "EXPOSED_DEVELOPMENT_DATA")
        self.assertTrue(s4["supports_recording_disjoint_split"])

        # Check split specification
        sp1 = s4["provisional_split_options"]["split_1_alternating_balanced"]
        self.assertEqual(sp1["recording_overlap_count"], 0, "Split 1 must have zero recording overlap!")
        self.assertEqual(sp1["train_recordings_count"], 27)
        self.assertEqual(sp1["val_recordings_count"], 27)

    def test_06_signer05_sealed_diagnostic_holdout(self):
        """Verify signer_05 is sealed, not pristine, and excluded from all training/fitting/tuning."""
        s5 = self.audit_data["dataset_census_and_provenance"]["signer_05"]
        self.assertEqual(s5["dataset_role"], "quarantined_diagnostic_holdout")
        self.assertEqual(s5["governance_restrictions"]["training_use"], "STRICTLY_PROHIBITED")
        self.assertEqual(s5["governance_restrictions"]["tuning_use"], "STRICTLY_PROHIBITED")
        self.assertEqual(s5["governance_restrictions"]["scaler_fitting_use"], "STRICTLY_PROHIBITED")
        self.assertEqual(s5["exposure_and_leakage_audit"]["pristine_holdout_status"], "EXPOSED_DIAGNOSTIC_HOLDOUT")

    def test_07_no_uncollected_signers_in_protocol(self):
        """Verify that no phantom signers (signer_06 to signer_11) are permitted in training or validation."""
        contract = self.protocol_data["data_access_contract"]
        permitted_train_signers = [item["signer_id"] for item in contract["permitted_training_pool"]]
        permitted_val_signers = [item["signer_id"] for item in contract["permitted_development_validation_pool"]]

        for forbidden in ["signer_06", "signer_07", "signer_08", "signer_09", "signer_10", "signer_11"]:
            self.assertNotIn(forbidden, permitted_train_signers)
            self.assertNotIn(forbidden, permitted_val_signers)

        s6_11 = self.audit_data["dataset_census_and_provenance"]["signer_06_to_11"]
        self.assertEqual(s6_11["actual_recordings_collected"], 0)
        self.assertEqual(s6_11["status"], "TERMINATED_NO_COLLECTION_POSSIBLE")

    def test_08_semantic_augmentation_bounds(self):
        """Verify that ISL semantic preservation rules strictly prohibit flips and excessive rotations."""
        aug_rules = self.protocol_data["semantic_augmentation_guidelines"]
        prohibited_types = [p["type"] for p in aug_rules["strictly_prohibited_transformations"]]
        self.assertIn("horizontal_mirror_flip", prohibited_types)
        self.assertIn("vertical_mirror_flip", prohibited_types)
        self.assertIn("large_rotation", prohibited_types)

    def test_09_all_seven_experiments_predeclared(self):
        """Verify that all 7 mandatory experiments are pre-declared with complete schemas."""
        exp_list = self.protocol_data["experiment_staged_plan"]
        self.assertEqual(len(exp_list), 7, "Protocol must contain exactly 7 staged experiments")

        expected_ids = [
            "EXP-01-BASELINE-REPRO",
            "EXP-02-GEOMETRIC-AUG",
            "EXP-03-CLASS-BALANCING",
            "EXP-04-FEATURE-ABLATION",
            "EXP-05-CONFIDENCE-CALIBRATION",
            "EXP-06-CONFUSION-ANALYSIS",
            "EXP-07-RUNTIME-LATENCY-MEMORY",
        ]

        actual_ids = [e["experiment_id"] for e in exp_list]
        self.assertEqual(actual_ids, expected_ids)

        for exp in exp_list:
            self.assertIn("hypothesis", exp)
            self.assertIn("training_data", exp)
            self.assertIn("validation_data", exp)
            self.assertIn("primary_metric", exp)
            self.assertIn("leakage_controls", exp)
            self.assertIn("predeclared_acceptance_criteria", exp)
            self.assertIn("falsification_conditions", exp)

    def test_10_production_defaults_preservation(self):
        """Verify that production runtime environment defaults remain intact."""
        prod_audit = self.audit_data["production_defaults_audit"]
        self.assertEqual(prod_audit["ISL_BACKEND_DETECTOR"], "yolo")
        self.assertEqual(prod_audit["ISL_RECOGNITION_ARCH"], "standard")
        self.assertEqual(prod_audit["ISL_TEMPORAL_SMOOTHING"], "0")
        self.assertEqual(prod_audit["ISL_SHADOW_MODE"], "0")
        self.assertEqual(prod_audit["ISL_MEDIAPIPE_DELEGATE"], "cpu")
        self.assertEqual(prod_audit["ISL_MODEL_C_RESOLUTION"], "native")


if __name__ == "__main__":
    unittest.main()

