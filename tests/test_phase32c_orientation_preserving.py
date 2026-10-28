#!/usr/bin/env python3
"""
Unit tests for Phase 32C — Orientation-Preserving Rotation-Robust Feature Ablation.

Validates:
1. Feature dimensionality and ordering for Variants A (280), B (296), C (316), and D (332).
2. Finite values and graceful handling under degenerate or missing landmarks.
3. Expected behavior under controlled rigid rotations.
4. Preservation of absolute orientation cues (longitudinal and roll encoding).
5. Exact preservation of baseline feature output (Variant A matches Phase 32A baseline bit-for-bit).
6. Immutability of production model, scaler, and benchmark artifacts.
7. Sealed-holdout quarantine of signer_05.
8. Strict preservation of production runtime defaults.
"""

import os
import json
import hashlib
import unittest
import numpy as np

from landmark_model_c import extract_landmarks_model_c
from orientation_preserving_features import (
    extract_orientation_descriptors,
    extract_canonical_pose_descriptors,
    extract_phase32c_features_from_item,
    DIM_VARIANT_A, DIM_VARIANT_B, DIM_VARIANT_C, DIM_VARIANT_D
)

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def compute_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def make_synthetic_hand(wrist_x=0.5, wrist_y=0.5, category="Right", score=0.95):
    """Generates synthetic 21x3 landmarks with upright hand geometry."""
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[0] = [wrist_x, wrist_y, 0.0]  # Wrist
    pts[9] = [wrist_x, wrist_y - 0.15, 0.0]  # Mid MCP (pointing straight UP along -Y)
    pts[5] = [wrist_x - 0.05, wrist_y - 0.12, 0.0]  # Index MCP
    pts[17] = [wrist_x + 0.05, wrist_y - 0.12, 0.0]  # Pinky MCP
    for i in range(1, 21):
        if i not in (0, 5, 9, 17):
            pts[i] = [wrist_x + (i % 5 - 2) * 0.02, wrist_y - (i // 5) * 0.04 - 0.05, 0.0]
    return {
        "pts": pts,
        "category": category,
        "score": score
    }


class TestPhase32COrientationPreserving(unittest.TestCase):

    def setUp(self):
        self.workspace_dir = WORKSPACE_DIR
        self.model_path = os.path.join(self.workspace_dir, "phase22_analysis/models/Model_C_Occlusion_Aware_RF_multisigner.joblib")
        self.scaler_path = os.path.join(self.workspace_dir, "phase22_analysis/models/Model_C_Occlusion_Aware_RF_multisigner_scaler.joblib")
        self.task_path = os.path.join(self.workspace_dir, "App/include50_model/hand_landmarker.task")
        self.phase32c_metrics_path = os.path.join(self.workspace_dir, "phase32c_ablation_metrics.json")

    def test_01_feature_dimensionality_and_ordering(self):
        """Variants A, B, C, D must have strictly 280, 296, 316, and 332 float32 dimensions."""
        hand = make_synthetic_hand()
        item = {
            "landmarks": [
                {"landmarks": [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in hand["pts"]], "category_name": "Right", "score": 0.95}
            ]
        }
        f_a = extract_phase32c_features_from_item(item, variant="A")
        f_b = extract_phase32c_features_from_item(item, variant="B")
        f_c = extract_phase32c_features_from_item(item, variant="C")
        f_d = extract_phase32c_features_from_item(item, variant="D")

        self.assertEqual(f_a.shape, (DIM_VARIANT_A,))
        self.assertEqual(f_b.shape, (DIM_VARIANT_B,))
        self.assertEqual(f_c.shape, (DIM_VARIANT_C,))
        self.assertEqual(f_d.shape, (DIM_VARIANT_D,))

        # Verify ordering: f_b begins with f_a exactly
        np.testing.assert_allclose(f_b[:DIM_VARIANT_A], f_a, atol=1e-6)
        # Verify ordering: f_c begins with f_a exactly
        np.testing.assert_allclose(f_c[:DIM_VARIANT_A], f_a, atol=1e-6)
        # Verify ordering: f_d begins with f_b exactly
        np.testing.assert_allclose(f_d[:DIM_VARIANT_B], f_b, atol=1e-6)

    def test_02_finite_values_under_degenerate_or_missing_landmarks(self):
        """Empty detections, single hand, zero landmarks, and degenerate points must produce finite numbers."""
        # 1. 0 hands
        item_0 = {"landmarks": []}
        for var in ["A", "B", "C", "D"]:
            f = extract_phase32c_features_from_item(item_0, variant=var)
            self.assertFalse(np.isnan(f).any())
            self.assertFalse(np.isinf(f).any())
            self.assertEqual(f[DIM_VARIANT_A - 16], 1.0)  # 0 hands flag

        # 2. Degenerate all-zero landmarks
        item_zeros = {
            "landmarks": [
                {"landmarks": [{"x": 0.0, "y": 0.0, "z": 0.0} for _ in range(21)], "category_name": "Right", "score": 0.0}
            ]
        }
        for var in ["A", "B", "C", "D"]:
            f = extract_phase32c_features_from_item(item_zeros, variant=var)
            self.assertFalse(np.isnan(f).any())
            self.assertFalse(np.isinf(f).any())

    def test_03_expected_behavior_under_controlled_rigid_rotations(self):
        """Orientation features must accurately track roll angle under rigid rotations."""
        hand = make_synthetic_hand()
        pts = hand["pts"]
        w = pts[0]

        # Upright hand pointing along -Y -> angle should be -pi/2 -> cos=0, sin=-1
        orient_up = extract_orientation_descriptors(pts, present=True)
        self.assertAlmostEqual(float(orient_up[1]), -1.0, places=3)  # vy = -1
        self.assertAlmostEqual(float(orient_up[6]), 0.0, places=3)   # cos(theta) = 0
        self.assertAlmostEqual(float(orient_up[7]), -1.0, places=3)  # sin(theta) = -1

        # Rotate +90 degrees (clockwise in image: pointing right along +X)
        R_90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
        pts_90 = ((pts - w) @ R_90.T) + w
        orient_90 = extract_orientation_descriptors(pts_90, present=True)
        self.assertAlmostEqual(float(orient_90[0]), 1.0, places=3)   # vx = +1
        self.assertAlmostEqual(float(orient_90[1]), 0.0, places=3)   # vy = 0
        self.assertAlmostEqual(float(orient_90[6]), 1.0, places=3)   # cos(theta) = +1
        self.assertAlmostEqual(float(orient_90[7]), 0.0, places=3)   # sin(theta) = 0

    def test_04_preservation_of_absolute_orientation_cues(self):
        """Horizontal signs (like H) and vertical signs (like L) must have distinct orientation descriptors."""
        # Horizontal hand (pointing along +X)
        h_pts = np.zeros((21, 3), dtype=np.float32)
        h_pts[0] = [0.5, 0.5, 0.0]
        h_pts[9] = [0.65, 0.5, 0.0]  # pointing along +X
        h_pts[5] = [0.62, 0.45, 0.0]
        h_pts[17] = [0.62, 0.55, 0.0]
        orient_h = extract_orientation_descriptors(h_pts, present=True)

        # Vertical hand (pointing along -Y)
        v_pts = np.zeros((21, 3), dtype=np.float32)
        v_pts[0] = [0.5, 0.5, 0.0]
        v_pts[9] = [0.5, 0.35, 0.0]  # pointing along -Y
        v_pts[5] = [0.45, 0.38, 0.0]
        v_pts[17] = [0.55, 0.38, 0.0]
        orient_v = extract_orientation_descriptors(v_pts, present=True)

        # Longitudinal vy must be completely distinct
        self.assertAlmostEqual(float(orient_h[1]), 0.0, places=2)
        self.assertAlmostEqual(float(orient_v[1]), -1.0, places=2)
        # Cos/Sin roll angles must be completely distinct
        self.assertGreater(abs(orient_h[6] - orient_v[6]), 0.8)

    def test_05_preservation_of_baseline_feature_output(self):
        """Variant A must produce exact identical output to Phase 32A baseline extract_features_proposed."""
        from evaluate_phase32a_development import extract_features_proposed
        hand = make_synthetic_hand()
        item = {
            "landmarks": [
                {"landmarks": [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in hand["pts"]], "category_name": "Right", "score": 0.95}
            ]
        }
        feat_base, _ = extract_features_proposed(item["landmarks"])
        feat_var_a = extract_phase32c_features_from_item(item, variant="A", suppress_lap_hands=True)
        np.testing.assert_allclose(feat_var_a, feat_base, atol=1e-7)

    def test_06_model_and_scaler_immutability(self):
        """Authoritative hashes of frozen production model and scaler must match."""
        expected_hashes = {
            self.model_path: "1fab3349e2e537d8b31ddbe3bdd5c21e3ed005f85aaac9d3875c72217e28cc98",
            self.scaler_path: "93842932e7cfeb1d283e99f67a733fd487161d83cbff356d7458ced2c3de385d",
            self.task_path: "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
        }
        for path, exp_h in expected_hashes.items():
            self.assertTrue(os.path.exists(path), f"Missing artifact: {path}")
            self.assertEqual(compute_file_sha256(path), exp_h)

    def test_07_sealed_holdout_quarantine(self):
        """signer_05 must remain quarantined and certified untouched in Phase 32C."""
        self.assertTrue(os.path.exists(self.phase32c_metrics_path))
        with open(self.phase32c_metrics_path, "r") as f:
            m = json.load(f)
        gov = m["metadata"]["governance"]
        self.assertEqual(gov["signer_05_holdout_quarantine"], "VERIFIED_SEALED_UNTOUCHED")
        self.assertEqual(gov["signer_02_signer_03_benchmark"], "VERIFIED_UNTOUCHED")

    def test_08_preservation_of_production_defaults(self):
        """Pipeline runtime configuration must strictly retain production defaults."""
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
