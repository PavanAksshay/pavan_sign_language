#!/usr/bin/env python3
"""
Unit tests for Landmark Hand Selection and Lap-Hand Suppression in landmark_feature_extractor.py.
"""

import unittest
import numpy as np
from landmark_feature_extractor import (
    select_active_hands,
    extract_landmarks_from_manifest_item,
    TOTAL_FEATURE_DIM,
    WRIST,
    INDEX_TIP,
    MID_MCP
)


def make_dummy_hand(wrist_x=0.5, wrist_y=0.5, scale=0.1, category="Right", score=0.95):
    """Generates synthetic 21x3 landmarks for testing."""
    pts = np.zeros((21, 3), dtype=np.float32)
    # Wrist at (wrist_x, wrist_y, 0)
    pts[WRIST] = [wrist_x, wrist_y, 0.0]
    # Spread fingers slightly above wrist
    for i in range(1, 21):
        pts[i] = [wrist_x + (i % 5 - 2) * 0.02, wrist_y - (i // 5) * 0.03 - 0.02, 0.0]
    pts[MID_MCP] = [wrist_x, wrist_y - scale, 0.0]
    return {
        "pts": pts,
        "category": category,
        "score": score,
        "wrist_x": wrist_x,
        "scale": scale
    }


class TestLandmarkHandSelection(unittest.TestCase):

    def test_single_hand_low_position_preserved(self):
        """A single hand, even when low in the frame (e.g. wrist_y = 0.82), must NEVER be suppressed."""
        hand = make_dummy_hand(wrist_x=0.5, wrist_y=0.82)
        selected, diag = select_active_hands([hand], suppress_lap_hands=True)
        self.assertEqual(len(selected), 1)
        self.assertFalse(diag["lap_hand_suppressed"])
        self.assertEqual(diag["selected_hand_count"], 1)

    def test_resting_lap_hand_suppression(self):
        """When an active hand is at chest level (y=0.45) and a resting hand is at lap level (y=0.85), lap hand is suppressed."""
        active_hand = make_dummy_hand(wrist_x=0.5, wrist_y=0.45)
        lap_hand = make_dummy_hand(wrist_x=0.48, wrist_y=0.85)

        selected, diag = select_active_hands([active_hand, lap_hand], suppress_lap_hands=True)
        self.assertEqual(len(selected), 1)
        self.assertTrue(diag["lap_hand_suppressed"])
        self.assertAlmostEqual(selected[0]["wrist_y"], 0.45, places=2)

    def test_two_handed_valid_sign_preserved(self):
        """In a genuine two-handed sign near chest level, both hands are retained."""
        hand_left = make_dummy_hand(wrist_x=0.40, wrist_y=0.48, category="Left")
        hand_right = make_dummy_hand(wrist_x=0.58, wrist_y=0.50, category="Right")

        selected, diag = select_active_hands([hand_left, hand_right], suppress_lap_hands=True)
        self.assertEqual(len(selected), 2)
        self.assertFalse(diag["lap_hand_suppressed"])
        self.assertEqual(diag["selected_hand_count"], 2)

    def test_two_handed_low_interacting_hands_preserved(self):
        """If both hands are at mid-low level but interacting (close together), neither is suppressed."""
        hand_left = make_dummy_hand(wrist_x=0.45, wrist_y=0.68, category="Left")
        hand_right = make_dummy_hand(wrist_x=0.52, wrist_y=0.69, category="Right")

        selected, diag = select_active_hands([hand_left, hand_right], suppress_lap_hands=True)
        self.assertEqual(len(selected), 2)
        self.assertFalse(diag["lap_hand_suppressed"])

    def test_zero_hands_detected(self):
        """0 hands input returns empty selection and clean diagnostic."""
        selected, diag = select_active_hands([], suppress_lap_hands=True)
        self.assertEqual(len(selected), 0)
        self.assertFalse(diag["lap_hand_suppressed"])

    def test_feature_vector_dimension_is_constant(self):
        """Feature vector must always be 200 float32 dimensions regardless of suppression."""
        active_hand = make_dummy_hand(wrist_x=0.5, wrist_y=0.45)
        lap_hand = make_dummy_hand(wrist_x=0.48, wrist_y=0.85)

        manifest_item = {
            "landmarks": [
                {
                    "landmarks": [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in active_hand["pts"]],
                    "category_name": "Right",
                    "score": 0.95
                },
                {
                    "landmarks": [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in lap_hand["pts"]],
                    "category_name": "Left",
                    "score": 0.90
                }
            ]
        }

        feat, diag = extract_landmarks_from_manifest_item(manifest_item, suppress_lap_hands=True, return_diagnostics=True)
        self.assertEqual(feat.shape, (TOTAL_FEATURE_DIM,))
        self.assertEqual(feat.dtype, np.float32)
        self.assertTrue(diag["lap_hand_suppressed"])

        # And with suppression disabled
        feat_no_supp = extract_landmarks_from_manifest_item(manifest_item, suppress_lap_hands=False)
        self.assertEqual(feat_no_supp.shape, (TOTAL_FEATURE_DIM,))


if __name__ == "__main__":
    unittest.main()
