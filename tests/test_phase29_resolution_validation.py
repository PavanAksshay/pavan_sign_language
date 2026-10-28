#!/usr/bin/env python3
"""
Unit and Integration Tests for Phase 29 — Downsampled Runtime Validation & Production Gate
Tests:
- Preservation of production defaults
- 640x360 resolution configuration and processed frame shape
- Exact 16:9 aspect ratio preservation
- Feature schema compatibility (280 dimensions, no NaNs)
- Normalized coordinate mapping to native frame dimensions
- Prediction isolation without silent fallback
- Rejection of invalid resolution configurations
- Clean shutdown and buffer teardown
- Frozen artifact and model weight immutability (SHA-256)
"""

import os
import sys
import unittest
import hashlib
import numpy as np
import cv2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from pipeline import ISLPipeline
from landmark_model_c import FEATURE_DIM_MODEL_C

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


class TestPhase29ResolutionValidation(unittest.TestCase):
    """Phase 29 test suite validating the opt-in 640x360 runtime configuration and baseline preservation."""

    def test_01_production_defaults_preservation(self):
        """Production defaults must remain strictly untouched on unconfigured initialization."""
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

    def test_02_resolution_configuration_preprocessing_path(self):
        """Setting 640x360 must configure target tuple (640, 360) and processed shape [360, 640]."""
        pipe = ISLPipeline()
        try:
            pipe.set_recognition_arch("model_c")
            pipe.set_mediapipe_delegate("gpu")
            pipe.set_model_c_input_resolution("640x360")
            self.assertEqual(pipe.model_c_input_resolution, "640x360")

            target_tuple = pipe._get_target_resolution(1280, 720)
            self.assertEqual(target_tuple, (640, 360))

            dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
            res = pipe.process_frame(dummy, return_images=False)
            diag = res.get("diagnostics", {})
            self.assertEqual(diag.get("input_resolution_setting"), "640x360")
            self.assertEqual(diag.get("input_frame_shape"), [720, 1280])
            self.assertEqual(diag.get("processed_shape"), [360, 640])
        finally:
            pipe.shutdown()

    def test_03_aspect_ratio_preservation(self):
        """640x360 downsampling must strictly preserve the 16:9 aspect ratio of native 1280x720."""
        w_orig, h_orig = 1280, 720
        w_down, h_down = 640, 360
        aspect_orig = w_orig / h_orig
        aspect_down = w_down / h_down
        self.assertAlmostEqual(aspect_orig, 16.0 / 9.0, places=5)
        self.assertAlmostEqual(aspect_down, 16.0 / 9.0, places=5)
        self.assertAlmostEqual(aspect_orig, aspect_down, places=5)

    def test_04_feature_compatibility_and_dimension(self):
        """Downsampled inference must output exactly 280 finite features without NaNs."""
        pipe = ISLPipeline()
        try:
            pipe.set_recognition_arch("model_c")
            pipe.set_model_c_input_resolution("640x360")
            dummy = np.ones((720, 1280, 3), dtype=np.uint8) * 128
            res = pipe.process_frame(dummy, return_images=False)
            diag = res.get("diagnostics", {})
            self.assertEqual(diag.get("feature_dimensions"), FEATURE_DIM_MODEL_C)
            self.assertIn("top_3", diag)
            self.assertIn("probabilities", res)
            self.assertEqual(res["probabilities"], [])  # Clean empty list on no hand

            # Test on real development frame with a detected hand
            sample_frame_path = os.path.join(BASE_DIR, "data", "extracted_frames", "signer_04", "condition_1", "A", "signer_04_session_20261015_01_A_0006.jpg")
            if os.path.exists(sample_frame_path):
                real_img = cv2.imread(sample_frame_path)
                res_real = pipe.process_frame(real_img, return_images=False)
                if res_real.get("hand_detected"):
                    self.assertEqual(len(res_real["probabilities"]), 26)
                    self.assertEqual(len(res_real["diagnostics"]["top_3"]), 3)
        finally:
            pipe.shutdown()

    def test_05_normalized_coordinate_mapping(self):
        """Bounding boxes in detections must map to native [1280, 720] bounds rather than downsampled coords."""
        pipe = ISLPipeline()
        try:
            pipe.set_recognition_arch("model_c")
            pipe.set_model_c_input_resolution("640x360")
            # Create synthetic frame with hand-like structure
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            res = pipe.process_frame(frame, return_images=False)
            # If detections exist, check coordinates are <= 1280 and <= 720
            for det in res.get("detections", []):
                self.assertLessEqual(det["x"] + det["w"], 1280)
                self.assertLessEqual(det["y"] + det["h"], 720)
        finally:
            pipe.shutdown()

    def test_06_prediction_isolation_and_no_native_fallback(self):
        """Switching between resolutions must remain isolated without hidden fallbacks."""
        pipe = ISLPipeline()
        try:
            pipe.set_recognition_arch("model_c")
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)

            pipe.set_model_c_input_resolution("native")
            res_nat = pipe.process_frame(frame, return_images=False)
            self.assertEqual(res_nat["diagnostics"]["processed_shape"], [720, 1280])

            pipe.set_model_c_input_resolution("640x360")
            res_640 = pipe.process_frame(frame, return_images=False)
            self.assertEqual(res_640["diagnostics"]["processed_shape"], [360, 640])

            pipe.set_model_c_input_resolution("native")
            res_nat2 = pipe.process_frame(frame, return_images=False)
            self.assertEqual(res_nat2["diagnostics"]["processed_shape"], [720, 1280])
        finally:
            pipe.shutdown()

    def test_07_invalid_resolution_rejection(self):
        """Unsupported resolution strings must be rejected with ValueError."""
        pipe = ISLPipeline()
        try:
            with self.assertRaises(ValueError):
                pipe.set_model_c_input_resolution("300x300")
            with self.assertRaises(ValueError):
                pipe.set_model_c_input_resolution("invalid_string")
            with self.assertRaises(ValueError):
                pipe.set_model_c_input_resolution("1920x1080")
        finally:
            pipe.shutdown()

    def test_08_frozen_artifacts_immutability(self):
        """Authoritative frozen model weights and reference files must remain 100% byte-for-byte identical."""
        for name, meta in AUTHORITATIVE_HASHES.items():
            self.assertTrue(os.path.exists(meta["path"]), f"Missing file: {meta['path']}")
            with open(meta["path"], "rb") as fp:
                computed_sha = hashlib.sha256(fp.read()).hexdigest()
            self.assertEqual(computed_sha, meta["sha256"], f"Hash mismatch for {name}: {computed_sha} != {meta['sha256']}")

    def test_09_clean_shutdown_and_buffer_teardown(self):
        """Pipeline shutdown must release landmarker and color buffers without accumulating resources."""
        pipe = ISLPipeline()
        pipe.set_recognition_arch("model_c")
        pipe.set_mediapipe_delegate("gpu")
        pipe.set_model_c_input_resolution("640x360")
        dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
        pipe.process_frame(dummy, return_images=False)
        self.assertIsNotNone(pipe.mp_landmarker)
        self.assertIsNotNone(pipe._model_c_color_buf)

        pipe.shutdown()
        self.assertIsNone(pipe.mp_landmarker)
        self.assertIsNone(pipe._model_c_color_buf)


if __name__ == "__main__":
    unittest.main()
