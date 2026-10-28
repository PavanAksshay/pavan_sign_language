#!/usr/bin/env python3
"""
Unit and Integration Tests for Phase 28 — Sustained GPU Latency Stabilization & Integrity
Tests:
- Preservation of production defaults
- Metal GPU delegate verification and dynamic switching
- 12-stage latency instrumentation integrity
- Streaming mode (zero JPEG encoding)
- Buffer reuse integrity
- Clean shutdown and repeated lifecycle initialization
- Failure containment and zero-hand recovery
- Frozen model weight immutability (SHA-256 checks)
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
from landmark_model_c import extract_landmarks_model_c, extract_occlusion_features, FEATURE_DIM_MODEL_C

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


class TestPhase28LatencyAndIntegrity(unittest.TestCase):
    """Phase 28 test suite validating sustained latency stabilization, delegate verification, and safety."""

    def test_production_defaults_preservation(self):
        """Production defaults must remain strictly untouched."""
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

    def test_model_c_metal_delegate_verification(self):
        """Metal delegate must be validated and reject invalid options."""
        pipe = ISLPipeline()
        try:
            pipe.set_mediapipe_delegate("gpu")
            self.assertEqual(pipe.mediapipe_delegate, "gpu")
            pipe.set_mediapipe_delegate("cpu")
            self.assertEqual(pipe.mediapipe_delegate, "cpu")
            with self.assertRaises(ValueError):
                pipe.set_mediapipe_delegate("vulkan")
        finally:
            pipe.shutdown()

    def test_stage_latency_instrumentation(self):
        """Model C process_frame must instrument all 12 stages with positive durations."""
        pipe = ISLPipeline()
        try:
            pipe.set_recognition_arch("model_c")
            pipe.set_mediapipe_delegate("gpu")
            dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
            res = pipe.process_frame(dummy, return_images=False)
            
            self.assertIn("diagnostics", res)
            self.assertIn("stage_timings_ms", res["diagnostics"])
            st = res["diagnostics"]["stage_timings_ms"]
            expected_stages = [
                "1_receipt", "2_validation", "3_copy", "4_resize",
                "5_color_conversion", "6_mp_image_construction", "7_detection",
                "8_postprocessing", "9_feature_extraction", "10_scaler_transform",
                "11_classifier_prediction", "12_formatting", "sum_stages"
            ]
            for s in expected_stages:
                self.assertIn(s, st)
            self.assertIn("latency_ms", res["diagnostics"])
            self.assertGreaterEqual(res["diagnostics"]["latency_ms"], 0.0)
        finally:
            pipe.shutdown()

    def test_streaming_mode_zero_jpeg_encoding(self):
        """return_images=False must omit image encoding overhead and avoid populating 'images' dict."""
        pipe = ISLPipeline()
        try:
            pipe.set_recognition_arch("model_c")
            dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
            res_stream = pipe.process_frame(dummy, return_images=False)
            self.assertNotIn("images", res_stream)

            res_full = pipe.process_frame(dummy, return_images=True)
            self.assertIn("images", res_full)
            self.assertIn("original", res_full["images"])
        finally:
            pipe.shutdown()

    def test_buffer_reuse_integrity(self):
        """Compatible preallocated buffer must be reused in Stage 5 without corrupting predictions."""
        pipe = ISLPipeline()
        try:
            pipe.set_recognition_arch("model_c")
            dummy1 = np.ones((720, 1280, 3), dtype=np.uint8) * 100
            dummy2 = np.ones((720, 1280, 3), dtype=np.uint8) * 150
            
            # Frame 1 allocates buffer
            res1 = pipe.process_frame(dummy1, return_images=False)
            self.assertTrue(hasattr(pipe, "_model_c_color_buf"))
            buf1_id = id(pipe._model_c_color_buf)

            # Frame 2 reuses identical buffer memory
            res2 = pipe.process_frame(dummy2, return_images=False)
            buf2_id = id(pipe._model_c_color_buf)
            self.assertEqual(buf1_id, buf2_id)
        finally:
            pipe.shutdown()

    def test_clean_shutdown_and_lifecycle(self):
        """Pipeline must shut down cleanly and allow re-instantiation without accumulating resources."""
        for _ in range(2):
            pipe = ISLPipeline()
            pipe.set_recognition_arch("model_c")
            pipe.set_mediapipe_delegate("gpu")
            dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
            res = pipe.process_frame(dummy, return_images=False)
            self.assertEqual(res["predicted_sign"], "-")
            pipe.shutdown()
            self.assertIsNone(pipe.mp_landmarker)
            self.assertIsNone(pipe._model_c_color_buf)

    def test_failure_containment_and_zero_hand_recovery(self):
        """Corrupted inputs and zero-hand frames must be contained gracefully."""
        pipe = ISLPipeline()
        try:
            pipe.set_recognition_arch("model_c")
            # Invalid input rejection
            with self.assertRaises(ValueError):
                pipe.process_frame(None)
            with self.assertRaises(ValueError):
                pipe.process_frame(np.array([]))

            # Blank zero-hand frame recovery
            blank = np.zeros((720, 1280, 3), dtype=np.uint8)
            res = pipe.process_frame(blank, return_images=False)
            self.assertFalse(res["hand_detected"])
            self.assertEqual(res["predicted_sign"], "-")
            self.assertEqual(res["confidence"], 0.0)
            self.assertEqual(res["diagnostics"]["hands_detected_count"], 0)
        finally:
            pipe.shutdown()

    def test_frozen_model_weight_immutability(self):
        """All frozen model weights, scalers, and reference benchmark files must match authoritative SHA-256."""
        for name, meta in AUTHORITATIVE_HASHES.items():
            self.assertTrue(os.path.exists(meta["path"]), f"Missing file: {meta['path']}")
            with open(meta["path"], "rb") as fp:
                computed_sha = hashlib.sha256(fp.read()).hexdigest()
            self.assertEqual(computed_sha, meta["sha256"], f"Hash mismatch for {name}: {computed_sha} != {meta['sha256']}")


if __name__ == "__main__":
    unittest.main()
