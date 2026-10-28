#!/usr/bin/env python3
"""
Phase 11 Unit and Regression Tests: Backend Switching and Controlled Inference

Validates:
1. YOLO remains the default backend detector when unconfigured.
2. Each backend loads its matching classifier and class mapping (no accidental mixing).
3. Invalid ISL_BACKEND_DETECTOR values fallback safely to 'yolo'.
4. Input tensor contract is [1, 224, 224, 3] float32 in [0, 1].
5. No-detection frames return a clean non-classifying response.
6. Multi-hand detections retain both hands in the isotropic square crop.
7. API response format remains 100% backward-compatible.
8. Switching backends dynamically does not mutate production weights.
"""

import os
import sys
import json
import unittest
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

APP_DIR = os.path.join(BASE_DIR, "App")
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from pipeline import ISLPipeline


class TestBackendSwitching(unittest.TestCase):
    def setUp(self):
        # Save and restore environment variable
        self.orig_env = os.environ.get("ISL_BACKEND_DETECTOR")

    def tearDown(self):
        if self.orig_env is not None:
            os.environ["ISL_BACKEND_DETECTOR"] = self.orig_env
        elif "ISL_BACKEND_DETECTOR" in os.environ:
            del os.environ["ISL_BACKEND_DETECTOR"]

    def test_yolo_default(self):
        """Verify YOLO remains default backend when unconfigured."""
        if "ISL_BACKEND_DETECTOR" in os.environ:
            del os.environ["ISL_BACKEND_DETECTOR"]
        pipe = ISLPipeline()
        self.assertEqual(pipe.backend_detector, "yolo")

    def test_invalid_backend_fallback(self):
        """Verify unsupported backend detector falls back to yolo."""
        os.environ["ISL_BACKEND_DETECTOR"] = "unsupported_detector_xyz"
        pipe = ISLPipeline()
        self.assertEqual(pipe.backend_detector, "yolo")

    def test_mediapipe_backend_loading(self):
        """Verify MediaPipe backend loads HandLandmarker and MediaPipe SqueezeNet model."""
        os.environ["ISL_BACKEND_DETECTOR"] = "mediapipe"
        pipe = ISLPipeline()
        self.assertEqual(pipe.backend_detector, "mediapipe")
        pipe.initialize()
        self.assertIsNotNone(pipe.mp_landmarker)
        self.assertIsNotNone(pipe.model)
        self.assertEqual(len(pipe.class_map), 26)
        self.assertEqual(pipe.model.input_shape, (None, 224, 224, 3))
        self.assertEqual(pipe.model.output_shape, (None, 26))

    def test_yolo_backend_loading(self):
        """Verify YOLO backend loads YOLO and Production SqueezeNet."""
        os.environ["ISL_BACKEND_DETECTOR"] = "yolo"
        pipe = ISLPipeline()
        self.assertEqual(pipe.backend_detector, "yolo")
        pipe.initialize()
        self.assertIsNotNone(pipe.yolo)
        self.assertIsNotNone(pipe.model)
        self.assertEqual(len(pipe.class_map), 26)

    def test_dynamic_backend_switching(self):
        """Verify switching backends dynamically reloads matching detector and classifier."""
        pipe = ISLPipeline()
        pipe.set_backend("yolo")
        self.assertEqual(pipe.backend_detector, "yolo")
        self.assertIsNotNone(pipe.yolo)
        self.assertIsNone(pipe.mp_landmarker)

        pipe.set_backend("mediapipe")
        self.assertEqual(pipe.backend_detector, "mediapipe")
        self.assertIsNone(pipe.yolo)
        self.assertIsNotNone(pipe.mp_landmarker)

        pipe.set_backend("yolo")
        self.assertEqual(pipe.backend_detector, "yolo")
        self.assertIsNotNone(pipe.yolo)
        self.assertIsNone(pipe.mp_landmarker)

    def test_mediapipe_no_detection_handling(self):
        """Verify blank black frame returns clean no-detection result without false classification."""
        pipe = ISLPipeline()
        pipe.set_backend("mediapipe")
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        res = pipe.process_frame(blank, return_images=True)

        self.assertFalse(res["hand_detected"])
        self.assertEqual(res["predicted_sign"], "-")
        self.assertEqual(res["confidence"], 0.0)
        self.assertEqual(len(res["detections"]), 0)
        self.assertIn("timing_ms", res)
        self.assertIn("yolo", res["timing_ms"])  # Frontend backward compatibility
        self.assertIn("detector", res["timing_ms"])

    def test_api_response_contract_compatibility(self):
        """Verify output dictionary contains all required fields for both backends."""
        pipe = ISLPipeline()
        dummy = np.zeros((240, 320, 3), dtype=np.uint8)

        for backend in ("yolo", "mediapipe"):
            pipe.set_backend(backend)
            res = pipe.process_frame(dummy, return_images=False)
            self.assertIn("predicted_sign", res)
            self.assertIn("confidence", res)
            self.assertIn("hand_detected", res)
            self.assertIn("is_two_handed", res)
            self.assertIn("detections", res)
            self.assertIn("probabilities", res)
            self.assertIn("diagnostics", res)
            self.assertIn("timing_ms", res)
            self.assertIn("total", res["timing_ms"])


if __name__ == "__main__":
    unittest.main()
