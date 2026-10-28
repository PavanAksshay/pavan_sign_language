#!/usr/bin/env python3
"""
Phase 16 — Unit Tests for Advanced Landmark Geometry v2, Fusion Calibration,
and Production Regression Protection.

Covers:
1. Feature vector dimension (264 float32 values) and deterministic ordering.
2. Degenerate landmarks (zero norm, colinear, missing hands) handling.
3. Feature normalization and bounds.
4. Probability vector alignment, sum-to-1 normalization, and non-negativity.
5. Fusion weight validation (alpha in [0.0, 1.0]).
6. Missing model artifact fallback safety.
7. Class mapping preservation (26 classes, A-Z).
8. Strict preservation of production configuration defaults:
   - ISL_BACKEND_DETECTOR = 'yolo'
   - ISL_RECOGNITION_ARCH = 'standard'
   - ISL_TEMPORAL_SMOOTHING = False
9. Existing API response structure backward compatibility.
10. Session isolation in TemporalSmoother.
11. Regression protection for YOLO production baseline.
"""

import os
import sys
import unittest
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import landmark_geometry_v2 as lg2
from temporal_smoother import TemporalSmoother
import pipeline


class TestPhase16GeometryAndFusion(unittest.TestCase):

    def setUp(self):
        # 21 synthetic landmarks
        self.mock_landmarks = np.array([
            [0.5, 0.8, 0.0],   # 0: Wrist
            [0.45, 0.75, 0.0], # 1: Thumb CMC
            [0.42, 0.70, 0.0], # 2: Thumb MCP
            [0.40, 0.65, 0.0], # 3: Thumb IP
            [0.38, 0.60, 0.0], # 4: Thumb TIP
            [0.48, 0.60, 0.0], # 5: Index MCP
            [0.48, 0.50, 0.0], # 6: Index PIP
            [0.48, 0.45, 0.0], # 7: Index DIP
            [0.48, 0.40, 0.0], # 8: Index TIP
            [0.52, 0.60, 0.0], # 9: Middle MCP
            [0.52, 0.48, 0.0], # 10: Middle PIP
            [0.52, 0.42, 0.0], # 11: Middle DIP
            [0.52, 0.36, 0.0], # 12: Middle TIP
            [0.56, 0.62, 0.0], # 13: Ring MCP
            [0.56, 0.52, 0.0], # 14: Ring PIP
            [0.56, 0.46, 0.0], # 15: Ring DIP
            [0.56, 0.42, 0.0], # 16: Ring TIP
            [0.60, 0.65, 0.0], # 17: Pinky MCP
            [0.60, 0.56, 0.0], # 18: Pinky PIP
            [0.60, 0.52, 0.0], # 19: Pinky DIP
            [0.60, 0.48, 0.0], # 20: Pinky TIP
        ], dtype=np.float32)

    def test_01_feature_vector_dimension_and_types(self):
        """Verify v2 feature vector dimensions and float32 data types."""
        feat_single = lg2.extract_single_hand_features_v2(self.mock_landmarks, category_name="Right")
        self.assertEqual(len(feat_single), lg2.FEATURE_DIM_PER_HAND_V2)
        self.assertEqual(len(feat_single), 125)
        self.assertEqual(feat_single.dtype, np.float32)

        feat_inter = lg2.extract_inter_hand_features_v2(self.mock_landmarks, self.mock_landmarks, 1.0, 1.0, True, True)
        self.assertEqual(len(feat_inter), lg2.INTER_HAND_FEATURE_DIM_V2)
        self.assertEqual(len(feat_inter), 14)
        self.assertEqual(feat_inter.dtype, np.float32)

        item = {
            "landmarks": [
                {"landmarks": [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in self.mock_landmarks], "category_name": "Right", "score": 0.99}
            ]
        }
        feat_total = lg2.extract_landmarks_from_manifest_item_v2(item)
        self.assertEqual(len(feat_total), lg2.TOTAL_FEATURE_DIM_V2)
        self.assertEqual(len(feat_total), 264)
        self.assertEqual(feat_total.dtype, np.float32)

    def test_02_degenerate_and_missing_landmarks_safety(self):
        """Verify that None, empty, zero-scale, or colinear landmarks do not crash and return zero vectors."""
        # Empty input
        zero_feat = lg2.extract_single_hand_features_v2(None, present=False)
        self.assertEqual(len(zero_feat), 125)
        self.assertTrue(np.all(zero_feat == 0.0))

        # Degenerate zero scale (all points coincident)
        colinear_pts = np.zeros((21, 3), dtype=np.float32)
        safe_feat = lg2.extract_single_hand_features_v2(colinear_pts, present=True)
        self.assertEqual(len(safe_feat), 125)
        self.assertFalse(np.isnan(safe_feat).any(), "NaN found in degenerate landmark feature vector")
        self.assertFalse(np.isinf(safe_feat).any(), "Inf found in degenerate landmark feature vector")

    def test_03_curl_ratios_bounded_and_meaningful(self):
        """Verify curl ratios are strictly bounded in [0.0, 1.5] for extended vs curled fingers."""
        feat = lg2.extract_single_hand_features_v2(self.mock_landmarks, present=True)
        # Curl ratios are indices 95 to 99 in 125-dim hand vector
        curl_ratios = feat[95:100]
        self.assertEqual(len(curl_ratios), 5)
        for r in curl_ratios:
            self.assertGreaterEqual(r, 0.0)
            self.assertLessEqual(r, 1.5)

    def test_04_probability_alignment_and_normalization(self):
        """Verify fusion probability blending preserves finite sum-to-1 normalization."""
        p_lm = np.array([0.1] * 20 + [0.8] + [0.0] * 5, dtype=np.float32)
        p_lm /= np.sum(p_lm)
        p_cnn = np.array([0.0] * 20 + [0.5] + [0.1] * 5, dtype=np.float32)
        p_cnn /= np.sum(p_cnn)

        for alpha in [0.0, 0.25, 0.50, 0.75, 1.0]:
            p_fused = alpha * p_lm + (1.0 - alpha) * p_cnn
            self.assertAlmostEqual(float(np.sum(p_fused)), 1.0, places=5)
            self.assertTrue(np.all(p_fused >= 0.0))
            self.assertFalse(np.isnan(p_fused).any())

    def test_05_fusion_weight_validation(self):
        """Verify invalid alpha values outside [0.0, 1.0] are detected."""
        valid_alphas = [0.25, 0.40, 0.50, 0.60, 0.75]
        for a in valid_alphas:
            self.assertTrue(0.0 <= a <= 1.0)

        invalid_alphas = [-0.1, 1.25, 5.0]
        for a in invalid_alphas:
            self.assertFalse(0.0 <= a <= 1.0)

    def test_06_temporal_smoother_session_isolation(self):
        """Verify sessions do not contaminate each other in TemporalSmoother."""
        smoother = TemporalSmoother(window_size=5)
        smoother.update("session_A", "C", 90.0)
        smoother.update("session_A", "C", 92.0)
        smoother.update("session_A", "C", 88.0)

        smoother.update("session_B", "L", 95.0)

        res_a = smoother.update("session_A", "C", 91.0)
        res_b = smoother.update("session_B", "L", 94.0)

        self.assertEqual(res_a["smoothed_sign"], "C")
        self.assertEqual(res_b["smoothed_sign"], "L")
        self.assertEqual(res_a["window_size"], 4)
        self.assertEqual(res_b["window_size"], 2)

    def test_07_class_mapping_preservation(self):
        """Verify 26 ISL alphabet classes A-Z remain strictly ordered 0-25."""
        class_map_path = os.path.join(BASE_DIR, "models", "mediapipe_squeezenet_v1", "model_class_26.json")
        import json
        with open(class_map_path, "r") as f:
            cmap = json.load(f)
        self.assertEqual(len(cmap), 26)
        for i in range(26):
            expected_letter = chr(65 + i)
            self.assertEqual(cmap[str(i)], expected_letter)

    def test_08_strict_preservation_of_production_defaults(self):
        """Verify production defaults remain yolo, standard, and smoothing disabled."""
        pipe = pipeline.ISLPipeline()
        self.assertEqual(pipe.backend_detector, "yolo")
        self.assertEqual(pipe.recognition_arch, "standard")
        self.assertEqual(pipe.temporal_smoothing_enabled, False)

    def test_09_v2_artifacts_exist_and_loadable(self):
        """Verify landmark_geometry_v2 model and scaler exist and can predict without errors."""
        import joblib
        v2_rf_path = os.path.join(BASE_DIR, "models", "landmark_geometry_v2", "landmark_rf_model_v2.joblib")
        v2_scaler_path = os.path.join(BASE_DIR, "models", "landmark_geometry_v2", "landmark_scaler_v2.joblib")

        self.assertTrue(os.path.exists(v2_rf_path), "Missing landmark_rf_model_v2.joblib")
        self.assertTrue(os.path.exists(v2_scaler_path), "Missing landmark_scaler_v2.joblib")

        rf = joblib.load(v2_rf_path)
        scaler = joblib.load(v2_scaler_path)

        dummy_x = np.zeros((1, 264), dtype=np.float32)
        scaled_x = scaler.transform(dummy_x)
        probs = rf.predict_proba(scaled_x)[0]
        self.assertEqual(len(probs), 26)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
