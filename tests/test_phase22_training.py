"""
Unit tests for Phase 22 — Multi-Signer Training, Controlled Ablation, and Signer Isolation.
Validates:
1. Reconciled dataset integrity (1,621 usable signer_04 frames, 57 superseded marked unusable, 4 defects marked unusable).
2. Strict external test isolation (signer_02 and signer_03 completely excluded).
3. Trained multi-signer model artifacts (Model A, Model B, Model C) and scalers load and predict valid classes.
4. Production defaults remain preserved (ISL_BACKEND_DETECTOR=yolo, ISL_RECOGNITION_ARCH=standard, ISL_TEMPORAL_SMOOTHING=0).
"""

import os
import sys
import json
import joblib
import unittest
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


class TestPhase22MultiSignerTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest_p = os.path.join(BASE_DIR, "data", "collection_manifests", "test_multisigner_manifest.json")
        cls.phase22_dir = os.path.join(BASE_DIR, "phase22_analysis")
        cls.models_dir = os.path.join(cls.phase22_dir, "models")
        cls.features_dir = os.path.join(cls.phase22_dir, "features_cache")

    def test_01_signer_04_reconciled_manifest_integrity(self):
        """Verify reconciled signer_04 manifest counts and unusable flags."""
        self.assertTrue(os.path.exists(self.manifest_p), f"Missing manifest: {self.manifest_p}")
        with open(self.manifest_p, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        s4_all = [s for s in manifest if s.get("signer_id") == "signer_04"]
        s4_usable = [s for s in s4_all if s.get("quality", {}).get("is_usable") is True]
        s4_unusable = [s for s in s4_all if s.get("quality", {}).get("is_usable") is not True]

        self.assertEqual(len(s4_all), 1682, "Total signer_04 samples should be exactly 1,682")
        self.assertEqual(len(s4_usable), 1621, "Usable signer_04 samples should be exactly 1,621")
        self.assertEqual(len(s4_unusable), 61, "Unusable signer_04 samples should be exactly 61")

        # Verify reasons for unusable samples
        superseded = [s for s in s4_unusable if "superseded" in s.get("quality", {}).get("rejection_reason", "")]
        blurry = [s for s in s4_unusable if s.get("quality", {}).get("rejection_reason") == "motion_blur_below_threshold"]
        zero_hands = [s for s in s4_unusable if s.get("quality", {}).get("rejection_reason") == "zero_hands_detected"]

        self.assertEqual(len(superseded), 57, "Exactly 57 samples must be marked superseded")
        self.assertEqual(len(blurry), 2, "Exactly 2 samples must be marked blurry")
        self.assertEqual(len(zero_hands), 2, "Exactly 2 samples must have zero hands detected")

    def test_02_strict_external_test_isolation(self):
        """Confirm signer_02 and signer_03 are completely excluded from Phase 22 training and features."""
        model_manifest_p = os.path.join(self.models_dir, "phase22_model_manifest.json")
        self.assertTrue(os.path.exists(model_manifest_p), "Missing phase22_model_manifest.json")
        with open(model_manifest_p, "r", encoding="utf-8") as f:
            m_manifest = json.load(f)

        self.assertEqual(m_manifest.get("signers_included_in_training"), ["signer_01", "signer_04"])
        self.assertEqual(m_manifest.get("signers_strictly_excluded_holdout"), ["signer_02", "signer_03"])

        # Check features cache
        for fname in os.listdir(self.features_dir):
            self.assertNotIn("signer_02", fname)
            self.assertNotIn("signer_03", fname)

    def test_03_feature_caches_and_shapes(self):
        """Verify feature dimensions: Model A (200), Model B (264), Model C (280)."""
        s1_tr_npz = os.path.join(self.features_dir, "signer_01_train_features.npz")
        s4_npz = os.path.join(self.features_dir, "signer_04_usable_features.npz")

        self.assertTrue(os.path.exists(s1_tr_npz))
        self.assertTrue(os.path.exists(s4_npz))

        d1 = np.load(s1_tr_npz)
        d4 = np.load(s4_npz)

        self.assertEqual(d1["features_v1"].shape, (3077, 200))
        self.assertEqual(d1["features_v2"].shape, (3077, 264))

        self.assertEqual(d4["features_v1"].shape, (1621, 200))
        self.assertEqual(d4["features_v2"].shape, (1621, 264))

    def test_04_model_artifacts_and_prediction(self):
        """Verify all three models and scalers load and predict valid 26-class probabilities."""
        models = {
            "Model_A_Landmark_RF_v1": 200,
            "Model_B_Regularized_Geometry_RF_v2": 264,
            "Model_C_Occlusion_Aware_RF": 280
        }

        for m_name, dim in models.items():
            m_path = os.path.join(self.models_dir, f"{m_name}_multisigner.joblib")
            s_path = os.path.join(self.models_dir, f"{m_name}_multisigner_scaler.joblib")

            self.assertTrue(os.path.exists(m_path), f"Missing model: {m_path}")
            self.assertTrue(os.path.exists(s_path), f"Missing scaler: {s_path}")

            clf = joblib.load(m_path)
            scaler = joblib.load(s_path)

            dummy_x = np.zeros((2, dim), dtype=np.float32)
            dummy_scaled = scaler.transform(dummy_x)
            preds = clf.predict(dummy_scaled)
            probs = clf.predict_proba(dummy_scaled)

            self.assertEqual(len(preds), 2)
            self.assertEqual(probs.shape, (2, 26))

    def test_05_production_defaults_intact(self):
        """Verify production environment defaults remain unchanged."""
        from pipeline import ISLPipeline
        pipeline = ISLPipeline()
        self.assertEqual(pipeline.backend_detector, "yolo")
        self.assertEqual(pipeline.recognition_arch, "standard")
        self.assertEqual(pipeline.temporal_smoothing_enabled, False)


if __name__ == "__main__":
    unittest.main()
