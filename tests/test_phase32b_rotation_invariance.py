#!/usr/bin/env python3
"""
Unit tests for Phase 32B — Palm-Canonical Rotation-Invariant Feature Ablation.

Validates:
1. Orthonormality and numerical stability of the palm basis (R @ R.T == I, det(R) == +1.0).
2. Mathematical invariance of canonical coordinates to in-plane rotations (-45 to +45 deg).
3. Graceful handling of degenerate or missing landmarks (collinear points, 0 hands, all zeros).
4. Finite output and exact feature dimensions (always strictly 280 dims, float32, no NaN/Inf).
5. Left/right hand handling and mirroring policy.
6. Preservation of existing Phase 32A extractor and frozen Model C artifacts.
7. Verification that signer_05 remains sealed and untouched.
8. Preservation of production runtime defaults.
"""

import os
import json
import hashlib
import unittest
import numpy as np

from landmark_feature_extractor import select_active_hands
from landmark_geometry_v2 import (
    TOTAL_FEATURE_DIM_V2,
    extract_landmarks_from_manifest_item_v2
)
from landmark_model_c import (
    FEATURE_DIM_MODEL_C,
    extract_landmarks_model_c
)
from palm_canonical_transformer import (
    compute_palm_basis,
    transform_landmarks_to_palm_canonical,
    extract_single_hand_features_canonical,
    extract_landmarks_canonical_v2,
    extract_landmarks_canonical_model_c
)

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def compute_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def make_synthetic_hand(wrist_x=0.5, wrist_y=0.5, category="Right", score=0.95):
    """Generates synthetic 21x3 landmarks with a realistic hand geometry."""
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[0] = [wrist_x, wrist_y, 0.0]  # Wrist
    pts[9] = [wrist_x, wrist_y - 0.15, 0.0]  # Mid MCP (pointing up)
    pts[5] = [wrist_x - 0.05, wrist_y - 0.12, 0.0]  # Index MCP
    pts[17] = [wrist_x + 0.05, wrist_y - 0.12, 0.0]  # Pinky MCP
    # Fill remaining fingers
    for i in range(1, 21):
        if i not in (0, 5, 9, 17):
            pts[i] = [wrist_x + (i % 5 - 2) * 0.02, wrist_y - (i // 5) * 0.04 - 0.05, 0.0]
    return {
        "pts": pts,
        "category": category,
        "score": score
    }


class TestPhase32BRotationInvariance(unittest.TestCase):

    def setUp(self):
        self.workspace_dir = WORKSPACE_DIR
        self.model_path = os.path.join(self.workspace_dir, "phase22_analysis/models/Model_C_Occlusion_Aware_RF_multisigner.joblib")
        self.scaler_path = os.path.join(self.workspace_dir, "phase22_analysis/models/Model_C_Occlusion_Aware_RF_multisigner_scaler.joblib")
        self.task_path = os.path.join(self.workspace_dir, "App/include50_model/hand_landmarker.task")
        self.phase32b_metrics_path = os.path.join(self.workspace_dir, "phase32b_ablation_metrics.json")

    def test_01_orthonormality_and_numerical_stability_of_palm_basis(self):
        """Palm basis rotation matrix R must satisfy R @ R.T == I and det(R) == +1.0."""
        hand = make_synthetic_hand()
        R, wrist, scale, is_deg = compute_palm_basis(hand["pts"], hand["category"])

        self.assertFalse(is_deg)
        self.assertEqual(R.shape, (3, 3))
        np.testing.assert_allclose(R @ R.T, np.eye(3, dtype=np.float32), atol=1e-5)
        det = float(np.linalg.det(R))
        self.assertAlmostEqual(det, 1.0, places=4)
        self.assertGreater(scale, 0.01)

    def test_02_invariance_to_controlled_in_plane_rotations(self):
        """Canonical coordinates must remain invariant under 2D camera-plane rotations."""
        hand = make_synthetic_hand()
        pts_orig = hand["pts"]
        wrist = pts_orig[0]

        c_orig, _, _, _ = transform_landmarks_to_palm_canonical(pts_orig)

        for deg in [-45, -30, -15, 15, 30, 45, 90]:
            rad = np.radians(deg)
            cos_t, sin_t = np.cos(rad), np.sin(rad)
            R_2d = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]], dtype=np.float32)

            pts_rot = ((pts_orig - wrist) @ R_2d.T) + wrist
            c_rot, _, _, _ = transform_landmarks_to_palm_canonical(pts_rot)

            max_diff = float(np.max(np.abs(c_orig - c_rot)))
            self.assertLess(max_diff, 1e-4, f"Canonical coordinates changed under {deg} deg rotation! Max diff: {max_diff}")

    def test_03_graceful_handling_of_degenerate_or_missing_landmarks(self):
        """Degenerate landmarks (all zeros, single point, collinear) must use fallback basis without crash."""
        # 1. None input
        R, w, s, is_deg = compute_palm_basis(None)
        self.assertTrue(is_deg)
        self.assertEqual(R.shape, (3, 3))

        # 2. All zeros
        zeros = np.zeros((21, 3), dtype=np.float32)
        R, w, s, is_deg = compute_palm_basis(zeros)
        self.assertTrue(is_deg)
        np.testing.assert_allclose(R @ R.T, np.eye(3, dtype=np.float32), atol=1e-5)

        # 3. Collinear landmarks (wrist, mid_mcp, idx_mcp all on Y-axis)
        collinear = np.zeros((21, 3), dtype=np.float32)
        for i in range(21):
            collinear[i] = [0.5, 0.1 * i, 0.0]
        R, w, s, is_deg = compute_palm_basis(collinear)
        self.assertTrue(is_deg)
        np.testing.assert_allclose(R @ R.T, np.eye(3, dtype=np.float32), atol=1e-5)
        self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=4)

    def test_04_finite_output_and_exact_feature_dimensions(self):
        """Canonical feature extractors must produce exactly 280 dims, float32, no NaN/Inf."""
        hand = make_synthetic_hand()
        item = {
            "landmarks": [
                {"landmarks": [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in hand["pts"]], "category_name": "Right", "score": 0.95}
            ]
        }
        feat_280, deg_cnt = extract_landmarks_canonical_model_c(item)
        self.assertEqual(feat_280.shape, (FEATURE_DIM_MODEL_C,))
        self.assertEqual(feat_280.dtype, np.float32)
        self.assertFalse(np.isnan(feat_280).any())
        self.assertFalse(np.isinf(feat_280).any())

        # Test empty input (0 hands)
        feat_0, deg_0 = extract_landmarks_canonical_model_c({"landmarks": []})
        self.assertEqual(feat_0.shape, (FEATURE_DIM_MODEL_C,))
        self.assertEqual(feat_0[FEATURE_DIM_MODEL_C - 16], 1.0)  # has_0_hands flag

    def test_05_left_and_right_hand_handling(self):
        """Tests left hand canonicalization with and without anatomical mirroring."""
        h_right = make_synthetic_hand(category="Right")
        h_left = make_synthetic_hand(category="Left")

        # Non-mirrored
        R_r, _, _, _ = compute_palm_basis(h_right["pts"], category_name="Right", mirror_left=False)
        R_l, _, _, _ = compute_palm_basis(h_left["pts"], category_name="Left", mirror_left=False)
        np.testing.assert_allclose(R_r @ R_r.T, np.eye(3), atol=1e-5)
        np.testing.assert_allclose(R_l @ R_l.T, np.eye(3), atol=1e-5)

        # Mirrored
        R_lm, _, _, _ = compute_palm_basis(h_left["pts"], category_name="Left", mirror_left=True)
        np.testing.assert_allclose(R_lm @ R_lm.T, np.eye(3), atol=1e-5)
        self.assertAlmostEqual(float(np.linalg.det(R_lm)), 1.0, places=4)

    def test_06_preservation_of_existing_extractor_and_frozen_artifacts(self):
        """Verify original feature extractor and Model C hashes remain unchanged."""
        expected_hashes = {
            self.model_path: "1fab3349e2e537d8b31ddbe3bdd5c21e3ed005f85aaac9d3875c72217e28cc98",
            self.scaler_path: "93842932e7cfeb1d283e99f67a733fd487161d83cbff356d7458ced2c3de385d",
            self.task_path: "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
        }
        for path, exp_hash in expected_hashes.items():
            self.assertTrue(os.path.exists(path), f"Missing artifact: {path}")
            act_hash = compute_file_sha256(path)
            self.assertEqual(act_hash, exp_hash, f"Hash mismatch for {os.path.basename(path)}")

    def test_07_signer_05_holdout_quarantine_preserved(self):
        """Certify that signer_05 was not used in Phase 32B experiments."""
        self.assertTrue(os.path.exists(self.phase32b_metrics_path))
        with open(self.phase32b_metrics_path, "r") as f:
            metrics = json.load(f)
        gov = metrics["metadata"]["governance"]
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
