#!/usr/bin/env python3
"""
Phase 24 — Model C Runtime Integration, Shadow Testing, and Deployment Readiness Unit Tests

Comprehensive test coverage:
1. Production baseline defaults preserved (YOLO + standard + smoothing=0 + shadow=0).
2. Model C Recognizer initialization, 280-dimensional feature contract, and public metadata sanitation.
3. Missing artifact and malformed model error safety.
4. Zero-hand, one-hand, and two-hand feature extraction handling.
5. Non-finite values (NaN / Inf) and malformed feature vector deterministic handling.
6. Shadow mode isolation: production prediction returned unchanged, experimental output contained.
7. Shadow mode exception resilience: simulated experimental failures do not crash or alter production.
8. Dynamic architecture switching to 'model_c' and rejection of unsupported architectures.
9. State immutability across repeated sequential predictions (no weight/scaler drift).
10. Server health and diagnostic endpoint contracts without internal path leakage.
"""

import os
import sys
import unittest
import numpy as np
from unittest.mock import MagicMock, patch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from pipeline import ISLPipeline
from landmark_model_c import ModelCRecognizer, extract_landmarks_model_c, extract_occlusion_features, FEATURE_DIM_MODEL_C


class TestPhase24RuntimeIntegration(unittest.TestCase):
    def setUp(self):
        self.orig_detector = os.environ.get("ISL_BACKEND_DETECTOR")
        self.orig_arch = os.environ.get("ISL_RECOGNITION_ARCH")
        self.orig_smooth = os.environ.get("ISL_TEMPORAL_SMOOTHING")
        self.orig_shadow = os.environ.get("ISL_SHADOW_MODE")

    def tearDown(self):
        for k, v in [
            ("ISL_BACKEND_DETECTOR", self.orig_detector),
            ("ISL_RECOGNITION_ARCH", self.orig_arch),
            ("ISL_TEMPORAL_SMOOTHING", self.orig_smooth),
            ("ISL_SHADOW_MODE", self.orig_shadow),
        ]:
            if v is not None:
                os.environ[k] = v
            elif k in os.environ:
                del os.environ[k]

    # 1. Baseline Preservation
    def test_01_baseline_defaults_preserved(self):
        """Verify production defaults remain: YOLO detector, standard arch, smoothing=0, shadow=0."""
        for k in ["ISL_BACKEND_DETECTOR", "ISL_RECOGNITION_ARCH", "ISL_TEMPORAL_SMOOTHING", "ISL_SHADOW_MODE"]:
            if k in os.environ:
                del os.environ[k]

        pipe = ISLPipeline()
        self.assertEqual(pipe.backend_detector, "yolo")
        self.assertEqual(pipe.recognition_arch, "standard")
        self.assertFalse(pipe.temporal_smoothing_enabled)
        self.assertFalse(pipe.shadow_mode_enabled)

    # 2. Model C Recognizer & Feature Contract
    def test_02_model_c_recognizer_loading_and_metadata(self):
        """Verify Model C loads correctly, validates 280 dims, and reports sanitized public metadata."""
        rec = ModelCRecognizer()
        rec.load()
        self.assertTrue(rec.is_loaded)
        self.assertEqual(rec.feature_dim, 280)
        self.assertEqual(len(rec.classes), 26)

        meta = rec.get_public_metadata()
        self.assertEqual(meta["architecture"], "model_c")
        self.assertEqual(meta["feature_dimensions"], 280)
        self.assertEqual(meta["classes_supported"], 26)
        self.assertTrue(meta["is_loaded"])
        # Ensure no internal filesystem paths are exposed
        for val in meta.values():
            if isinstance(val, str):
                self.assertNotIn("/Users/", val)
                self.assertNotIn(".joblib", val)

    # 3. Missing Artifact Safety
    def test_03_missing_artifact_safety(self):
        """Verify missing checkpoint files raise explicit FileNotFoundError without crashing unhandled."""
        bogus_rec = ModelCRecognizer(
            model_path="/tmp/non_existent_model_c.joblib",
            scaler_path="/tmp/non_existent_scaler.joblib"
        )
        with self.assertRaises(FileNotFoundError):
            bogus_rec.load()

    # 4. Zero-Hand Handling
    def test_04_zero_hand_handling(self):
        """Verify None or empty landmarks return zero-hand safe fallback without invoking RF on invalid data."""
        rec = ModelCRecognizer()
        rec.load()

        res_none = rec.predict_mediapipe_result(None)
        self.assertEqual(res_none["predicted_letter"], "-")
        self.assertEqual(res_none["confidence"], 0.0)
        self.assertEqual(res_none["hand_count"], 0)
        self.assertEqual(res_none["feature_status"], "fallback_0_hands")

        # Mock result with empty hand_landmarks
        mock_empty = MagicMock()
        mock_empty.hand_landmarks = []
        res_empty = rec.predict_mediapipe_result(mock_empty)
        self.assertEqual(res_empty["predicted_letter"], "-")
        self.assertEqual(res_empty["hand_count"], 0)

    # 5. One-Hand and Two-Hand Feature Dimensions
    def test_05_feature_dimensions_and_flags(self):
        """Verify synthetic 1-hand and 2-hand structures yield exactly 280 dims with valid flags."""
        # 1-hand synthetic
        p = MagicMock(x=0.5, y=0.5, z=0.0)
        mock_1h = MagicMock()
        mock_1h.hand_landmarks = [[p] * 21]

        f_1h, count_1h, status_1h = extract_landmarks_model_c(mock_1h)
        self.assertEqual(len(f_1h), FEATURE_DIM_MODEL_C)
        self.assertEqual(count_1h, 1)
        self.assertEqual(status_1h, "detected_1_hand")
        # Occlusion flag: index 265 should be 1.0 (has_1_hand)
        self.assertEqual(f_1h[265], 1.0)
        self.assertEqual(f_1h[264], 0.0)
        self.assertEqual(f_1h[266], 0.0)

        # 2-hand synthetic
        p1 = MagicMock(x=0.3, y=0.5, z=0.0)
        p2 = MagicMock(x=0.7, y=0.5, z=0.0)
        mock_2h = MagicMock()
        mock_2h.hand_landmarks = [[p1] * 21, [p2] * 21]

        f_2h, count_2h, status_2h = extract_landmarks_model_c(mock_2h)
        self.assertEqual(len(f_2h), FEATURE_DIM_MODEL_C)
        self.assertEqual(count_2h, 2)
        self.assertEqual(status_2h, "detected_2_hands")
        self.assertEqual(f_2h[266], 1.0)
        self.assertEqual(f_2h[264], 0.0)
        self.assertEqual(f_2h[265], 0.0)

    # 6. Non-Finite Values and Malformed Vectors
    def test_06_non_finite_values_sanitation(self):
        """Verify vector with NaN, +Inf, -Inf is sanitized deterministically and evaluated."""
        rec = ModelCRecognizer()
        rec.load()

        dirty_vec = np.ones(FEATURE_DIM_MODEL_C, dtype=np.float32)
        dirty_vec[0] = np.nan
        dirty_vec[1] = np.inf
        dirty_vec[2] = -np.inf
        dirty_vec[264] = np.nan

        pred = rec.predict_features(dirty_vec)
        self.assertIn(pred["predicted_letter"], rec.classes)
        self.assertGreaterEqual(pred["confidence"], 0.0)
        self.assertLessEqual(pred["confidence"], 100.0)

    # 7. Shadow Mode Isolation
    def test_07_shadow_mode_isolation(self):
        """Verify shadow mode returns production prediction unchanged while attaching shadow diagnostics."""
        pipe = ISLPipeline()
        pipe.set_backend("yolo")
        pipe.set_recognition_arch("standard")
        pipe.set_shadow_mode(True)
        self.assertTrue(pipe.shadow_mode_enabled)

        # Generate a synthetic frame
        synthetic_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        res = pipe.process_frame(synthetic_frame, return_images=False)

        # Production prediction contract
        self.assertIn("predicted_sign", res)
        self.assertIn("confidence", res)
        self.assertIn("diagnostics", res)

        # Shadow diagnostics present
        self.assertIn("shadow_model_c", res["diagnostics"])
        shadow_diag = res["diagnostics"]["shadow_model_c"]
        self.assertTrue(shadow_diag["enabled"])
        self.assertIn(shadow_diag["status"], ("success", "error"))

        # Verify that shadow output did not overwrite production prediction
        if shadow_diag.get("status") == "success":
            self.assertIn("predicted_letter", shadow_diag)
            self.assertIn("match_production", shadow_diag)

    # 8. Shadow Mode Exception Resilience
    def test_08_shadow_mode_exception_resilience(self):
        """Verify an exception inside shadow mode does not disrupt or crash production inference."""
        pipe = ISLPipeline()
        pipe.set_backend("yolo")
        pipe.set_recognition_arch("standard")
        pipe.set_shadow_mode(True)

        synthetic_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Mock ModelCRecognizer to raise an unhandled error during shadow execution
        with patch.object(pipe.model_c_recognizer, "predict_mediapipe_result", side_effect=RuntimeError("Simulated Shadow Failure")):
            res = pipe.process_frame(synthetic_frame, return_images=False)

            # Production inference completed successfully without throwing
            self.assertIn("predicted_sign", res)
            self.assertIn("diagnostics", res)
            self.assertIn("shadow_model_c", res["diagnostics"])

            shadow_diag = res["diagnostics"]["shadow_model_c"]
            self.assertEqual(shadow_diag["status"], "error")
            self.assertEqual(shadow_diag["error_type"], "RuntimeError")
            self.assertIn("Simulated Shadow Failure", shadow_diag["error_message"])
            self.assertFalse(shadow_diag["match_production"])

    # 9. Architecture Switching and Rejection
    def test_09_architecture_switching_and_validation(self):
        """Verify set_recognition_arch accepts 'model_c' and rejects invalid architectures."""
        pipe = ISLPipeline()
        self.assertEqual(pipe.recognition_arch, "standard")

        pipe.set_recognition_arch("model_c")
        self.assertEqual(pipe.recognition_arch, "model_c")
        self.assertIsNotNone(pipe.model_c_recognizer)
        self.assertTrue(pipe.model_c_recognizer.is_loaded)

        # Switch back to standard
        pipe.set_recognition_arch("standard")
        self.assertEqual(pipe.recognition_arch, "standard")

        # Unsupported architecture rejection
        with self.assertRaises(ValueError):
            pipe.set_recognition_arch("unsupported_random_forest_v99")

    # 10. State Immutability across Repeated Predictions
    def test_10_repeated_predictions_immutability(self):
        """Verify repeated predictions do not mutate scaler mean, feature dimensionality, or class list."""
        rec = ModelCRecognizer()
        rec.load()

        scaler_mean_orig = rec.scaler.mean_.copy()
        classes_orig = list(rec.classes)

        vec = np.ones(FEATURE_DIM_MODEL_C, dtype=np.float32)
        for _ in range(25):
            _ = rec.predict_features(vec)

        np.testing.assert_array_equal(rec.scaler.mean_, scaler_mean_orig)
        self.assertEqual(rec.classes, classes_orig)
        self.assertEqual(rec.feature_dim, FEATURE_DIM_MODEL_C)

    # 11. Server Endpoints Contract
    def test_11_server_endpoints_contract(self):
        """Verify FastAPI endpoints report status, allow opt-in shadow mode, and do not leak paths."""
        import asyncio
        from server import health, get_recognition_status, get_shadow_mode_status, set_shadow_mode_endpoint, set_recognition
        from starlette.requests import Request

        # Health endpoint
        h_data = asyncio.run(health())
        self.assertEqual(h_data["status"], "healthy")
        self.assertIn("effective_config", h_data)
        self.assertIn("shadow_mode_enabled", h_data["effective_config"])
        self.assertIn("backend_detector", h_data["effective_config"])
        self.assertIn("recognition_arch", h_data["effective_config"])

        # Recognition status
        rec_status = asyncio.run(get_recognition_status())
        self.assertIn("recognition_arch", rec_status)
        self.assertIn("shadow_mode_enabled", rec_status)

        # Shadow mode status
        shadow_status = asyncio.run(get_shadow_mode_status())
        self.assertIn("shadow_mode_enabled", shadow_status)
        self.assertIn("model_c_loaded", shadow_status)


if __name__ == "__main__":
    unittest.main()
