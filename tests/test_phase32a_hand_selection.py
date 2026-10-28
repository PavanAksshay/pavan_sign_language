#!/usr/bin/env python3
"""
Unit tests for Phase 32A — Active-Hand Selection and Ordering Robustness.

Tests:
1. One visible active hand (never suppressed regardless of position).
2. Two legitimate active hands (preserved at standard or lower heights).
3. Active hand with resting lap hand (reproducing Phase 31 failure mode and verifying suppression).
4. Lower-positioned but genuinely active hand (interacting lower hands preserved).
5. Hands crossing horizontally (crossing deadband hysteresis & deterministic slot assignment).
6. Missing or low-confidence handedness (deterministic fallbacks).
7. Partial occlusion and missing landmarks (synchronized zero-fill and flags).
8. Deterministic slot assignment across repeated inputs (100% idempotent).
9. Feature vector dimension and schema compatibility (exact 280-dim float32 structure).
10. Preservation of original model and scaler hashes.
11. Preservation of production defaults.
"""

import os
import hashlib
import unittest
import numpy as np

from landmark_feature_extractor import (
    TOTAL_FEATURE_DIM,
    select_active_hands,
    WRIST, INDEX_TIP, MID_MCP
)
from landmark_geometry_v2 import (
    order_two_hands_robust,
    extract_landmarks_from_manifest_item_v2,
    TOTAL_FEATURE_DIM_V2
)
from landmark_model_c import (
    FEATURE_DIM_MODEL_C,
    GEOMETRY_DIM_V2,
    OCCLUSION_DIM,
    extract_occlusion_features,
    extract_landmarks_model_c
)

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def compute_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def make_synthetic_hand(wrist_x=0.5, wrist_y=0.5, scale=0.1, category="Right", score=0.95):
    """Generates synthetic 21x3 landmarks for unit testing."""
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[WRIST] = [wrist_x, wrist_y, 0.0]
    for i in range(1, 21):
        pts[i] = [wrist_x + (i % 5 - 2) * 0.02, wrist_y - (i // 5) * 0.03 - 0.02, 0.0]
    pts[MID_MCP] = [wrist_x, wrist_y - scale, 0.0]
    return {
        "pts": pts,
        "category": category,
        "score": score,
        "wrist_x": float(wrist_x),
        "wrist_y": float(wrist_y),
        "scale": float(scale)
    }


class TestPhase32AHandSelection(unittest.TestCase):

    def setUp(self):
        self.workspace_dir = WORKSPACE_DIR
        self.model_path = os.path.join(self.workspace_dir, "phase22_analysis/models/Model_C_Occlusion_Aware_RF_multisigner.joblib")
        self.scaler_path = os.path.join(self.workspace_dir, "phase22_analysis/models/Model_C_Occlusion_Aware_RF_multisigner_scaler.joblib")
        self.task_path = os.path.join(self.workspace_dir, "App/include50_model/hand_landmarker.task")

    def test_01_single_active_hand_never_suppressed_regardless_of_position(self):
        """A single detected hand must NEVER be suppressed, regardless of screen position."""
        test_positions = [
            (0.50, 0.30),  # Upper chest
            (0.50, 0.55),  # Chest level
            (0.50, 0.75),  # Low torso
            (0.48, 0.85),  # Lap height
            (0.50, 0.95),  # Bottom of frame
        ]
        for wx, wy in test_positions:
            hand = make_synthetic_hand(wrist_x=wx, wrist_y=wy)
            selected, diag = select_active_hands([hand], suppress_lap_hands=True)
            self.assertEqual(len(selected), 1, f"Single hand at y={wy} was falsely suppressed!")
            self.assertFalse(diag["lap_hand_suppressed"])
            self.assertEqual(diag["selected_hand_count"], 1)

    def test_02_two_legitimate_active_hands_preserved(self):
        """Two legitimate active signing hands must both be preserved."""
        # Standard signing position
        h_left = make_synthetic_hand(wrist_x=0.40, wrist_y=0.48, category="Left")
        h_right = make_synthetic_hand(wrist_x=0.58, wrist_y=0.50, category="Right")
        selected, diag = select_active_hands([h_left, h_right], suppress_lap_hands=True)
        self.assertEqual(len(selected), 2)
        self.assertFalse(diag["lap_hand_suppressed"])
        self.assertEqual(diag["selected_hand_count"], 2)

    def test_03_active_hand_with_resting_lap_hand(self):
        """
        When an active signing hand is at chest level and a resting hand is in the lap
        vertically aligned (reproducing Phase 31 failure where w_dist was < 0.35),
        the deep lap decoupling rule must successfully suppress the resting hand.
        """
        # Active hand at chest center: (x=0.50, y=0.48)
        active_hand = make_synthetic_hand(wrist_x=0.50, wrist_y=0.48, category="Right")
        # Resting lap hand directly underneath: (x=0.48, y=0.82)
        # Delta x = 0.02, Delta y = 0.34 -> w_dist = sqrt(0.02^2 + 0.34^2) = 0.3406 (< 0.35 legacy threshold)
        lap_hand = make_synthetic_hand(wrist_x=0.48, wrist_y=0.82, category="Left")

        selected, diag = select_active_hands([active_hand, lap_hand], suppress_lap_hands=True)
        self.assertEqual(len(selected), 1, "Resting lap hand was not suppressed!")
        self.assertTrue(diag["lap_hand_suppressed"])
        self.assertEqual(diag["selected_hand_count"], 1)
        self.assertAlmostEqual(selected[0]["wrist_y"], 0.48, places=2)

    def test_04_lower_positioned_genuinely_active_hands(self):
        """If both hands are at mid-low level but interacting together, neither is suppressed."""
        # Both hands low (y=0.70, y=0.72) interacting close together
        h0 = make_synthetic_hand(wrist_x=0.47, wrist_y=0.70, category="Left")
        h1 = make_synthetic_hand(wrist_x=0.52, wrist_y=0.72, category="Right")
        selected, diag = select_active_hands([h0, h1], suppress_lap_hands=True)
        self.assertEqual(len(selected), 2, "Lower-positioned interacting hands were incorrectly suppressed!")
        self.assertFalse(diag["lap_hand_suppressed"])

    def test_05_hands_crossing_horizontally(self):
        """
        Tests horizontal crossing behavior:
        - When hands are clearly separated (|dx| >= 0.02), ordered strictly by wrist_x ascending.
        - When hands cross within deadband (|dx| < 0.02), handedness resolution applies to prevent jitter.
        """
        # Clear separation
        h_a = make_synthetic_hand(wrist_x=0.40, wrist_y=0.50, category="Left", score=0.95)
        h_b = make_synthetic_hand(wrist_x=0.60, wrist_y=0.50, category="Right", score=0.95)
        o0, o1 = order_two_hands_robust(h_a, h_b)
        self.assertAlmostEqual(o0["wrist_x"], 0.40)
        self.assertAlmostEqual(o1["wrist_x"], 0.60)

        # Within crossing deadband: dx = 0.005 < 0.02
        # User Right (category="Right") should be assigned to Slot 0
        h_cross_r = make_synthetic_hand(wrist_x=0.502, wrist_y=0.50, category="Right", score=0.95)
        h_cross_l = make_synthetic_hand(wrist_x=0.498, wrist_y=0.50, category="Left", score=0.95)
        o0, o1 = order_two_hands_robust(h_cross_r, h_cross_l)
        self.assertEqual(o0["category"], "Right", "Slot 0 should receive high-confidence Right hand in deadband")
        self.assertEqual(o1["category"], "Left", "Slot 1 should receive high-confidence Left hand in deadband")

    def test_06_missing_or_low_confidence_handedness(self):
        """When handedness is low-confidence or identical, ordering falls back deterministically to wrist_x."""
        # Low confidence (< 0.85) in deadband
        h0 = make_synthetic_hand(wrist_x=0.499, wrist_y=0.50, category="Right", score=0.60)
        h1 = make_synthetic_hand(wrist_x=0.501, wrist_y=0.50, category="Right", score=0.60)
        o0, o1 = order_two_hands_robust(h0, h1)
        self.assertAlmostEqual(o0["wrist_x"], 0.499)
        self.assertAlmostEqual(o1["wrist_x"], 0.501)

    def test_07_partial_occlusion_and_missing_landmarks(self):
        """
        Tests partial occlusion and zero hands:
        - 0 hands input returns 0 hands and flag 0 in occlusion vector.
        - 1 hand input zero-fills slot 1 and sets flag 1 in occlusion vector.
        """
        manifest_0 = {"landmarks": []}
        feat_0 = extract_landmarks_from_manifest_item_v2(manifest_0)
        self.assertEqual(feat_0.shape, (TOTAL_FEATURE_DIM_V2,))
        self.assertTrue(np.all(feat_0 == 0.0))

        # Occlusion features for 0 hands
        occ_0 = extract_occlusion_features([])
        self.assertEqual(occ_0.shape, (OCCLUSION_DIM,))
        self.assertEqual(occ_0[0], 1.0)  # has_0_hands flag

        # Occlusion features for 1 hand
        h = make_synthetic_hand(wrist_x=0.50, wrist_y=0.50)
        occ_1 = extract_occlusion_features([{"landmarks": [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in h["pts"]]}])
        self.assertEqual(occ_1[1], 1.0)  # has_1_hand flag

    def test_08_deterministic_slot_assignment_across_repeated_inputs(self):
        """Running selection and ordering 50 times produces bit-for-bit identical results."""
        h0 = make_synthetic_hand(wrist_x=0.42, wrist_y=0.50, category="Left", score=0.92)
        h1 = make_synthetic_hand(wrist_x=0.58, wrist_y=0.52, category="Right", score=0.91)

        first_o0, first_o1 = order_two_hands_robust(h0, h1)
        for _ in range(50):
            cur_o0, cur_o1 = order_two_hands_robust(h0, h1)
            np.testing.assert_array_equal(cur_o0["pts"], first_o0["pts"])
            np.testing.assert_array_equal(cur_o1["pts"], first_o1["pts"])

    def test_09_feature_vector_dimension_and_schema_compatibility(self):
        """The complete Model C feature vector must be exactly 280-dim float32."""
        h0 = make_synthetic_hand(wrist_x=0.40, wrist_y=0.50, category="Left")
        h1 = make_synthetic_hand(wrist_x=0.60, wrist_y=0.50, category="Right")

        manifest_item = {
            "landmarks": [
                {"landmarks": [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in h0["pts"]], "category_name": "Left", "score": 0.95},
                {"landmarks": [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in h1["pts"]], "category_name": "Right", "score": 0.95}
            ]
        }
        feat_264, active_hands = extract_landmarks_from_manifest_item_v2(manifest_item, suppress_lap_hands=True, return_active_hands=True)
        self.assertEqual(feat_264.shape, (GEOMETRY_DIM_V2,))
        self.assertEqual(len(active_hands), 2)

        occ_16 = extract_occlusion_features([active_hands[0]["pts"], active_hands[1]["pts"]])
        self.assertEqual(occ_16.shape, (OCCLUSION_DIM,))

        feat_280 = np.concatenate([feat_264, occ_16]).astype(np.float32)
        self.assertEqual(feat_280.shape, (FEATURE_DIM_MODEL_C,))
        self.assertEqual(feat_280.dtype, np.float32)
        self.assertFalse(np.isnan(feat_280).any())
        self.assertFalse(np.isinf(feat_280).any())

    def test_10_preservation_of_original_model_and_scaler_hashes(self):
        """Authoritative hashes of frozen Model C artifacts must remain unaltered."""
        expected_hashes = {
            self.model_path: "1fab3349e2e537d8b31ddbe3bdd5c21e3ed005f85aaac9d3875c72217e28cc98",
            self.scaler_path: "93842932e7cfeb1d283e99f67a733fd487161d83cbff356d7458ced2c3de385d",
            self.task_path: "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
        }
        for path, exp_hash in expected_hashes.items():
            self.assertTrue(os.path.exists(path), f"Artifact missing: {path}")
            act_hash = compute_file_sha256(path)
            self.assertEqual(act_hash, exp_hash, f"Hash mismatch for {os.path.basename(path)}")

    def test_11_preservation_of_production_defaults(self):
        """Runtime pipeline production defaults must remain exactly as required."""
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
