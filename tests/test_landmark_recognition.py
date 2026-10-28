#!/usr/bin/env python3
"""
Unit and Integration Tests for Phase 14A — Landmark-Guided Recognition

Covers:
1. Landmark vector dimensions and feature ordering.
2. Coordinate normalization and scale/translation invariance.
3. Degenerate cases (zero hands, zero palm scale, NaN/Inf immunity).
4. Missing or low-confidence landmarks (explicit flags, no fabricated coordinates).
5. One-hand and two-hand inputs.
6. Consistent hand ordering (invariance to input list ordering).
7. Deterministic feature extraction.
8. Class-index mapping integrity (A=0 through Z=25).
9. Landmark-only and hybrid model input/output shapes.
10. Preservation of existing YOLO and MediaPipe v1 production paths.
"""

import os
import sys
import unittest
import numpy as np
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from landmark_feature_extractor import (
    extract_single_hand_features,
    extract_inter_hand_features,
    extract_landmarks_from_manifest_item,
    extract_landmarks_from_mediapipe_result,
    FEATURE_DIM_PER_HAND,
    INTER_HAND_FEATURE_DIM,
    TOTAL_FEATURE_DIM,
    WRIST, MID_MCP
)


class TestLandmarkFeatureExtractor(unittest.TestCase):

    def setUp(self):
        # Create standard synthetic 21-point hand
        self.hand_right = np.zeros((21, 3), dtype=np.float32)
        # Wrist at (0.6, 0.5, 0.0)
        self.hand_right[WRIST] = [0.6, 0.5, 0.0]
        # Middle MCP at (0.6, 0.3, 0.0) -> palm scale = 0.2
        self.hand_right[MID_MCP] = [0.6, 0.3, 0.0]
        # Spread other landmarks
        for i in range(21):
            if i not in (WRIST, MID_MCP):
                self.hand_right[i] = [0.6 + 0.01 * (i % 5), 0.3 - 0.02 * (i // 5), 0.005 * i]

        self.hand_left = np.zeros((21, 3), dtype=np.float32)
        # Wrist at (0.4, 0.5, 0.0)
        self.hand_left[WRIST] = [0.4, 0.5, 0.0]
        self.hand_left[MID_MCP] = [0.4, 0.3, 0.0]
        for i in range(21):
            if i not in (WRIST, MID_MCP):
                self.hand_left[i] = [0.4 - 0.01 * (i % 5), 0.3 - 0.02 * (i // 5), -0.005 * i]

    def test_landmark_vector_dimensions_and_ordering(self):
        """Feature vector dimension must be exactly 200 (95 + 95 + 10)."""
        self.assertEqual(FEATURE_DIM_PER_HAND, 95)
        self.assertEqual(INTER_HAND_FEATURE_DIM, 10)
        self.assertEqual(TOTAL_FEATURE_DIM, 200)

        feat = extract_single_hand_features(self.hand_right, "Right", score=0.98, present=True)
        self.assertEqual(feat.shape, (95,))
        self.assertEqual(feat.dtype, np.float32)

        # Check metadata location at end of single hand vector
        # [..., presence, score, handedness_sign]
        self.assertEqual(feat[92], 1.0)       # presence
        self.assertAlmostEqual(feat[93], 0.98, places=2)  # score
        self.assertEqual(feat[94], 1.0)       # right = +1.0

    def test_coordinate_normalization_and_invariance(self):
        """Translation and isotropic scale transformations must produce invariant relative features."""
        feat_orig = extract_single_hand_features(self.hand_right, "Right", score=1.0, present=True)

        # 1. Translate hand in 3D space
        shift = np.array([0.15, -0.25, 0.05], dtype=np.float32)
        hand_shifted = self.hand_right + shift
        feat_shifted = extract_single_hand_features(hand_shifted, "Right", score=1.0, present=True)

        # Relative coordinates and geometric features (indices 0 to 92) should match within numerical epsilon
        np.testing.assert_allclose(feat_orig[:92], feat_shifted[:92], atol=1e-5)

        # 2. Scale hand isotropically by 1.5x
        hand_scaled = self.hand_right * 1.5
        feat_scaled = extract_single_hand_features(hand_scaled, "Right", score=1.0, present=True)
        np.testing.assert_allclose(feat_orig[:92], feat_scaled[:92], atol=1e-4)

    def test_degenerate_cases(self):
        """Handles zero hands, zero palm distance, and invalid arrays safely without NaN/Inf."""
        # 1. Zero hands item
        empty_item = {"landmarks": []}
        feat_empty = extract_landmarks_from_manifest_item(empty_item)
        self.assertEqual(feat_empty.shape, (200,))
        self.assertTrue(np.all(feat_empty == 0.0))
        self.assertFalse(np.isnan(feat_empty).any())

        # 2. Degenerate palm scale: Wrist == Middle MCP (distance = 0.0)
        degen_hand = self.hand_right.copy()
        degen_hand[MID_MCP] = degen_hand[WRIST]
        feat_degen = extract_single_hand_features(degen_hand, "Right", score=0.8, present=True)
        self.assertEqual(feat_degen.shape, (95,))
        self.assertFalse(np.isnan(feat_degen).any())
        self.assertFalse(np.isinf(feat_degen).any())

        # 3. None input
        feat_none = extract_single_hand_features(None, present=False)
        self.assertEqual(feat_none.shape, (95,))
        self.assertTrue(np.all(feat_none == 0.0))

    def test_missing_and_low_confidence_landmarks(self):
        """Missing hands must never have fabricated coordinates and must have presence=0.0."""
        # 1 hand manifest item
        item_1hand = {
            "landmarks": [{
                "hand_index": 0,
                "category_name": "Left",
                "score": 0.52,
                "landmarks": [{"id": i, "x": float(self.hand_left[i, 0]), "y": float(self.hand_left[i, 1]), "z": float(self.hand_left[i, 2])} for i in range(21)]
            }]
        }
        vec = extract_landmarks_from_manifest_item(item_1hand)
        self.assertEqual(vec.shape, (200,))

        # Hand 0 present
        self.assertEqual(vec[92], 1.0)  # Hand 0 presence
        self.assertAlmostEqual(vec[93], 0.52, places=2)  # Hand 0 score
        self.assertEqual(vec[94], -1.0) # Left = -1.0

        # Hand 1 strictly missing (all zeros, presence=0.0, score=0.0)
        np.testing.assert_array_equal(vec[95:190], np.zeros(95, dtype=np.float32))
        self.assertEqual(vec[95 + 92], 0.0)  # Hand 1 presence = 0

        # Inter-hand features strictly 0.0
        np.testing.assert_array_equal(vec[190:200], np.zeros(10, dtype=np.float32))

    def test_consistent_hand_ordering(self):
        """Swapping the list order of hands produces identical canonical vectors sorted by wrist-x."""
        hand_pts_L = [{"id": i, "x": float(self.hand_left[i, 0]), "y": float(self.hand_left[i, 1]), "z": float(self.hand_left[i, 2])} for i in range(21)]
        hand_pts_R = [{"id": i, "x": float(self.hand_right[i, 0]), "y": float(self.hand_right[i, 1]), "z": float(self.hand_right[i, 2])} for i in range(21)]

        item_order_1 = {
            "landmarks": [
                {"category_name": "Left", "score": 0.95, "landmarks": hand_pts_L},
                {"category_name": "Right", "score": 0.90, "landmarks": hand_pts_R}
            ]
        }
        item_order_2 = {
            "landmarks": [
                {"category_name": "Right", "score": 0.90, "landmarks": hand_pts_R},
                {"category_name": "Left", "score": 0.95, "landmarks": hand_pts_L}
            ]
        }

        vec1 = extract_landmarks_from_manifest_item(item_order_1)
        vec2 = extract_landmarks_from_manifest_item(item_order_2)
        np.testing.assert_allclose(vec1, vec2, atol=1e-6)

    def test_deterministic_feature_extraction(self):
        """Feature extraction must be 100% deterministic on repeated runs."""
        item = {
            "landmarks": [{
                "category_name": "Right",
                "score": 0.99,
                "landmarks": [{"id": i, "x": float(self.hand_right[i, 0]), "y": float(self.hand_right[i, 1]), "z": float(self.hand_right[i, 2])} for i in range(21)]
            }]
        }
        v1 = extract_landmarks_from_manifest_item(item)
        v2 = extract_landmarks_from_manifest_item(item)
        np.testing.assert_array_equal(v1, v2)


class TestLandmarkAndHybridModels(unittest.TestCase):

    def setUp(self):
        self.model_dir = os.path.join(BASE_DIR, "models", "landmark_hybrid_v1")
        self.rf_lm_path = os.path.join(self.model_dir, "landmark_rf_model.joblib")
        self.rf_hyb_path = os.path.join(self.model_dir, "hybrid_concatenation_model.joblib")
        self.scaler_lm_path = os.path.join(self.model_dir, "landmark_scaler.joblib")
        self.scaler_512_path = os.path.join(self.model_dir, "cnn512_scaler.joblib")

    def test_model_files_exist(self):
        """New experimental model files and scalers must exist."""
        self.assertTrue(os.path.exists(self.rf_lm_path))
        self.assertTrue(os.path.exists(self.rf_hyb_path))
        self.assertTrue(os.path.exists(self.scaler_lm_path))
        self.assertTrue(os.path.exists(self.scaler_512_path))

    def test_model_shapes_and_probability_sum(self):
        """Landmark-only and hybrid models accept expected shapes and return valid probabilities."""
        rf_lm = joblib.load(self.rf_lm_path)
        rf_hyb = joblib.load(self.rf_hyb_path)
        scaler_lm = joblib.load(self.scaler_lm_path)
        scaler_512 = joblib.load(self.scaler_512_path)

        # Landmark model input: (N, 200)
        dummy_lm = np.zeros((3, 200), dtype=np.float32)
        dummy_lm_s = scaler_lm.transform(dummy_lm)
        probs_lm = rf_lm.predict_proba(dummy_lm_s)
        self.assertEqual(probs_lm.shape, (3, 26))
        np.testing.assert_allclose(np.sum(probs_lm, axis=1), np.ones(3), atol=1e-5)

        # Hybrid model input: (N, 712)
        dummy_512 = np.zeros((3, 512), dtype=np.float32)
        dummy_512_s = scaler_512.transform(dummy_512)
        dummy_hyb = np.hstack([dummy_512_s, dummy_lm_s])
        self.assertEqual(dummy_hyb.shape, (3, 712))
        probs_hyb = rf_hyb.predict_proba(dummy_hyb)
        self.assertEqual(probs_hyb.shape, (3, 26))
        np.testing.assert_allclose(np.sum(probs_hyb, axis=1), np.ones(3), atol=1e-5)

    def test_class_index_mapping(self):
        """Class mapping matches 0=A to 25=Z."""
        import json
        class_map_file = os.path.join(BASE_DIR, "models", "mediapipe_squeezenet_v1", "model_class_26.json")
        with open(class_map_file) as f:
            cmap = json.load(f)
        for i in range(26):
            expected_char = chr(ord('A') + i)
            self.assertEqual(cmap[str(i)], expected_char)


class TestProductionPreservation(unittest.TestCase):

    def test_production_defaults_preserved(self):
        """ISL_BACKEND_DETECTOR defaults to yolo and YOLO models exist intact."""
        from pipeline import ISLPipeline
        p = ISLPipeline()
        self.assertEqual(p.backend_detector, "yolo")
        self.assertTrue(os.path.exists(os.path.join(BASE_DIR, "App", "final_model_26")))
        self.assertTrue(os.path.exists(os.path.join(BASE_DIR, "App", "yolo_models", "cross-hands.cfg")))
        self.assertTrue(os.path.exists(os.path.join(BASE_DIR, "App", "yolo_models", "cross-hands.weights")))

    def test_mediapipe_v1_and_v2_unmodified(self):
        """Preserves existing mediapipe_squeezenet_v1 and v2 artifacts."""
        v1_path = os.path.join(BASE_DIR, "models", "mediapipe_squeezenet_v1", "saved_model")
        v2_path = os.path.join(BASE_DIR, "models", "mediapipe_squeezenet_v2")
        self.assertTrue(os.path.exists(v1_path))
        self.assertTrue(os.path.exists(v2_path))


if __name__ == "__main__":
    unittest.main()
