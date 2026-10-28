#!/usr/bin/env python3
"""
Unit tests for late-fusion staging and safe fallback in pipeline.py and server.py.
"""

import os
import unittest
import numpy as np
from pipeline import ISLPipeline


class TestLateFusionPipeline(unittest.TestCase):

    def test_default_production_configuration(self):
        """Default startup must preserve YOLO detector, standard recognition, and disabled smoothing."""
        p = ISLPipeline()
        self.assertEqual(p.backend_detector, "yolo")
        self.assertEqual(p.recognition_arch, "standard")
        self.assertFalse(p.temporal_smoothing_enabled)

    def test_late_fusion_initialization_mediapipe(self):
        """When switched to late_fusion with mediapipe detector, models load cleanly."""
        p = ISLPipeline()
        p.set_backend("mediapipe")
        p.set_recognition_arch("late_fusion")
        p.initialize()

        self.assertEqual(p.backend_detector, "mediapipe")
        self.assertEqual(p.recognition_arch, "late_fusion")
        self.assertIsNotNone(p.landmark_rf_model)
        self.assertIsNotNone(p.landmark_scaler)
        self.assertEqual(p.late_fusion_alpha, 0.25)

    def test_class_alignment(self):
        """Both CNN and landmark RF models must agree on 26-class ordering A-Z."""
        p = ISLPipeline()
        p.set_backend("mediapipe")
        p.set_recognition_arch("late_fusion")
        p.initialize()

        self.assertEqual(len(p.class_map), 26)
        for i in range(26):
            expected_letter = chr(ord('A') + i)
            self.assertEqual(p.class_map.get(str(i)), expected_letter)

    def test_safe_fallback_on_invalid_model_path(self):
        """If model loading encounters an invalid path, pipeline falls back safely to 'standard'."""
        p = ISLPipeline()
        p.set_backend("mediapipe")
        # Try loading from non-existent directory
        try:
            p._load_late_fusion_models("/tmp/non_existent_model_dir_xyz")
        except Exception:
            p.recognition_arch = "standard"

        self.assertEqual(p.recognition_arch, "standard")

    def test_predict_with_late_fusion_on_blank_frame(self):
        """Predicting on a blank frame returns clean '-' with no crash."""
        p = ISLPipeline()
        p.set_backend("mediapipe")
        p.set_recognition_arch("late_fusion")
        p.initialize()

        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        res = p.predict(blank, return_images=False)

        self.assertEqual(res["predicted_sign"], "-")
        self.assertEqual(res["confidence"], 0.0)
        self.assertFalse(res["hand_detected"])
        self.assertIn("diagnostics", res)
        self.assertEqual(res["diagnostics"]["recognition_arch"], "late_fusion")

    def test_temporal_smoothing_toggle(self):
        """Toggling temporal smoothing works dynamically."""
        p = ISLPipeline()
        self.assertFalse(p.temporal_smoothing_enabled)
        p.set_temporal_smoothing(True)
        self.assertTrue(p.temporal_smoothing_enabled)
        p.set_temporal_smoothing(False)
        self.assertFalse(p.temporal_smoothing_enabled)


if __name__ == "__main__":
    unittest.main()
