#!/usr/bin/env python3
"""
Phase 24 — Model C Runtime Recognizer and Feature Extractor
Encapsulates:
- 280-dimensional feature extraction (264-dim Geometry v2 + 16-dim Occlusion/Contact proxies)
- Robust missing-hand handling (0, 1, 2+ hands) and non-finite value sanitation
- Thread-safe, read-only inference using the frozen Phase 22 Model C Random Forest checkpoint
- Standardized dictionary output for live pipeline and shadow-mode evaluation
"""

import os
import sys
import time
import joblib
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from landmark_geometry_v2 import extract_landmarks_from_mediapipe_result_v2, TOTAL_FEATURE_DIM_V2

FEATURE_DIM_MODEL_C = 280
GEOMETRY_DIM_V2 = 264
OCCLUSION_DIM = 16

ALPHABET = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
I2C = {i: c for i, c in enumerate(ALPHABET)}
C2I = {c: i for i, c in enumerate(ALPHABET)}


def extract_occlusion_features(landmarks_list: Optional[List[Dict[str, Any]]]) -> np.ndarray:
    """
    Extracts the exact 16 occlusion-aware and inter-hand contact features defined in Phase 22.
    - Dimensions:
      0: has_0_hands flag
      1: has_1_hand flag
      2: has_2_hands flag
      3: 2D Bounding Box IoU
      4: Signed Depth Offset
      5-8: Tip-to-Palm Contact Proxies (dominant/subordinate indices and middle tips to palm plane)
      9: Knuckle-to-knuckle minimum distance
      10-12: Wrist-to-fingertip bilateral cross distances
      13: Bilateral palm scale ratio
      14-15: Cross-hand thumb-to-index cosine angles
    """
    if not landmarks_list or len(landmarks_list) == 0:
        vec = np.zeros(OCCLUSION_DIM, dtype=np.float32)
        vec[0] = 1.0  # has_0_hands flag
        return vec

    if len(landmarks_list) == 1:
        vec = np.zeros(OCCLUSION_DIM, dtype=np.float32)
        vec[1] = 1.0  # has_1_hand flag
        return vec

    # 2 or more hands detected
    try:
        if isinstance(landmarks_list[0], np.ndarray):
            h0 = landmarks_list[0].astype(np.float32)
            h1 = landmarks_list[1].astype(np.float32)
        else:
            hands = sorted(landmarks_list[:2], key=lambda h: h["landmarks"][0]["x"])
            h0 = np.array([[p["x"], p["y"], p["z"]] for p in hands[0]["landmarks"]], dtype=np.float32)
            h1 = np.array([[p["x"], p["y"], p["z"]] for p in hands[1]["landmarks"]], dtype=np.float32)

        scale0 = np.linalg.norm(h0[9] - h0[0]) + 1e-6
        scale1 = np.linalg.norm(h1[9] - h1[0]) + 1e-6
        mean_scale = (scale0 + scale1) / 2.0

        has_0 = 0.0
        has_1 = 0.0
        has_2 = 1.0

        # 2D Bounding Box IoU
        min0, max0 = np.min(h0[:, :2], axis=0), np.max(h0[:, :2], axis=0)
        min1, max1 = np.min(h1[:, :2], axis=0), np.max(h1[:, :2], axis=0)
        inter_min = np.maximum(min0, min1)
        inter_max = np.minimum(max0, max1)
        inter_dim = np.maximum(0.0, inter_max - inter_min)
        inter_area = inter_dim[0] * inter_dim[1]
        area0 = (max0[0] - min0[0]) * (max0[1] - min0[1])
        area1 = (max1[0] - min1[0]) * (max1[1] - min1[1])
        union_area = area0 + area1 - inter_area + 1e-7
        iou = float(inter_area / union_area)

        # Signed Depth Offset
        z_diff = float((np.mean(h0[:, 2]) - np.mean(h1[:, 2])) / mean_scale)

        # Tip-to-Palm Contact Proxies
        v_idx0 = h0[5] - h0[0]
        v_pnk0 = h0[17] - h0[0]
        norm0 = np.cross(v_idx0, v_pnk0)
        norm0 = norm0 / (np.linalg.norm(norm0) + 1e-7)

        v_idx1 = h1[5] - h1[0]
        v_pnk1 = h1[17] - h1[0]
        norm1 = np.cross(v_idx1, v_pnk1)
        norm1 = norm1 / (np.linalg.norm(norm1) + 1e-7)

        dist_tip1_to_palm0 = float(np.dot(h1[8] - h0[0], norm0) / mean_scale)
        dist_mid1_to_palm0 = float(np.dot(h1[12] - h0[0], norm0) / mean_scale)
        dist_tip0_to_palm1 = float(np.dot(h0[8] - h1[0], norm1) / mean_scale)
        dist_mid0_to_palm1 = float(np.dot(h0[12] - h1[0], norm1) / mean_scale)

        mcp_indices = [5, 9, 13, 17]
        knuckle_dists = [np.linalg.norm(h0[i] - h1[j]) / mean_scale for i in mcp_indices for j in mcp_indices]
        min_knuckle_dist = float(min(knuckle_dists))

        d_w0_idx1 = float(np.linalg.norm(h0[0] - h1[8]) / mean_scale)
        d_w1_idx0 = float(np.linalg.norm(h1[0] - h0[8]) / mean_scale)
        d_w0_w1 = float(np.linalg.norm(h0[0] - h1[0]) / mean_scale)

        scale_ratio = float(min(scale0, scale1) / max(scale0, scale1))

        t_v0 = (h0[4] - h0[2]) / (np.linalg.norm(h0[4] - h0[2]) + 1e-7)
        i_v1 = (h1[8] - h1[5]) / (np.linalg.norm(h1[8] - h1[5]) + 1e-7)
        t_v1 = (h1[4] - h1[2]) / (np.linalg.norm(h1[4] - h1[2]) + 1e-7)
        i_v0 = (h0[8] - h0[5]) / (np.linalg.norm(h0[8] - h0[5]) + 1e-7)
        cross_cos0 = float(np.clip(np.dot(t_v0, i_v1), -1.0, 1.0))
        cross_cos1 = float(np.clip(np.dot(t_v1, i_v0), -1.0, 1.0))

        vec = np.array([
            has_0, has_1, has_2,
            iou, z_diff,
            dist_tip1_to_palm0, dist_mid1_to_palm0, dist_tip0_to_palm1, dist_mid0_to_palm1,
            min_knuckle_dist,
            d_w0_idx1, d_w1_idx0, d_w0_w1,
            scale_ratio,
            cross_cos0, cross_cos1
        ], dtype=np.float32)
        return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        # Fallback if any coordinate array arithmetic fails
        vec = np.zeros(OCCLUSION_DIM, dtype=np.float32)
        vec[2] = 1.0
        return vec


def extract_landmarks_model_c(mediapipe_result: Any) -> Tuple[np.ndarray, int, str]:
    """
    Constructs the full 280-dimensional feature vector for Model C from a MediaPipe result.
    Returns:
        (features_280: np.ndarray, hand_count: int, feature_status: str)
    """
    if mediapipe_result is None:
        f_280 = np.zeros(FEATURE_DIM_MODEL_C, dtype=np.float32)
        f_280[GEOMETRY_DIM_V2] = 1.0  # has_0_hands
        return f_280, 0, "fallback_0_hands"

    # 1. Geometry v2 features (264 dimensions) and active hands
    try:
        f_264, active_hands = extract_landmarks_from_mediapipe_result_v2(mediapipe_result, return_active_hands=True)
        if len(f_264) != GEOMETRY_DIM_V2:
            f_264 = np.zeros(GEOMETRY_DIM_V2, dtype=np.float32)
            active_hands = []
    except Exception:
        f_264 = np.zeros(GEOMETRY_DIM_V2, dtype=np.float32)
        active_hands = []

    # 2. Occlusion features (16 dimensions) synchronized with active hands
    num_active = len(active_hands)
    if num_active >= 2:
        h0_pts = active_hands[0]["pts"]
        h1_pts = active_hands[1]["pts"]
        f_16 = extract_occlusion_features([h0_pts, h1_pts])
    elif num_active == 1:
        f_16 = np.zeros(OCCLUSION_DIM, dtype=np.float32)
        f_16[1] = 1.0  # has_1_hand flag
    else:
        f_16 = np.zeros(OCCLUSION_DIM, dtype=np.float32)
        f_16[0] = 1.0  # has_0_hands flag

    # 3. Concatenate and sanitize
    f_280 = np.concatenate([f_264, f_16], axis=0).astype(np.float32)
    f_280 = np.nan_to_num(f_280, nan=0.0, posinf=0.0, neginf=0.0)

    if len(f_280) != FEATURE_DIM_MODEL_C:
        padded = np.zeros(FEATURE_DIM_MODEL_C, dtype=np.float32)
        padded[:min(len(f_280), FEATURE_DIM_MODEL_C)] = f_280[:min(len(f_280), FEATURE_DIM_MODEL_C)]
        f_280 = padded

    status = "detected_2_hands" if num_active >= 2 else ("detected_1_hand" if num_active == 1 else "fallback_0_hands")
    return f_280, num_active, status


class ModelCRecognizer:
    """
    Thread-safe, read-only runtime inference engine for Model C.
    Loads checkpoint and scaler once upon initialization.
    """
    _instance = None

    def __init__(self, model_path: Optional[str] = None, scaler_path: Optional[str] = None):
        self.model_path = model_path or os.path.join(BASE_DIR, "phase22_analysis", "models", "Model_C_Occlusion_Aware_RF_multisigner.joblib")
        self.scaler_path = scaler_path or os.path.join(BASE_DIR, "phase22_analysis", "models", "Model_C_Occlusion_Aware_RF_multisigner_scaler.joblib")

        self.clf = None
        self.scaler = None
        self.is_loaded = False
        self.feature_dim = FEATURE_DIM_MODEL_C
        self.version = "1.0.0"
        self.schema_name = "v3_occlusion_280d"
        self.classes = ALPHABET

    def load(self):
        """Loads and validates the serialized model and scaler."""
        if self.is_loaded:
            return

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model C checkpoint not found at: {self.model_path}")
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Model C scaler not found at: {self.scaler_path}")

        loaded_clf = joblib.load(self.model_path)
        loaded_scaler = joblib.load(self.scaler_path)

        if not hasattr(loaded_clf, "predict_proba"):
            raise TypeError("Model C object does not implement predict_proba interface.")
        if getattr(loaded_clf, "n_features_in_", None) != FEATURE_DIM_MODEL_C:
            raise ValueError(f"Model C feature dimension mismatch: expected {FEATURE_DIM_MODEL_C}, got {getattr(loaded_clf, 'n_features_in_', None)}")

        if not hasattr(loaded_scaler, "transform"):
            raise TypeError("Scaler object does not implement transform interface.")
        if getattr(loaded_scaler, "mean_", None) is not None and len(loaded_scaler.mean_) != FEATURE_DIM_MODEL_C:
            raise ValueError(f"Scaler dimension mismatch: expected {FEATURE_DIM_MODEL_C}, got {len(loaded_scaler.mean_)}")

        # Phase 25 optimization: Set n_jobs=1 for single-sample inference to eliminate joblib thread-dispatch overhead
        configured_jobs = int(os.environ.get("ISL_MODEL_C_N_JOBS", "1"))
        if hasattr(loaded_clf, "n_jobs"):
            loaded_clf.n_jobs = configured_jobs

        self.clf = loaded_clf
        self.scaler = loaded_scaler
        self.n_jobs = getattr(loaded_clf, "n_jobs", 1)
        self.is_loaded = True

    @property
    def classifier(self):
        return self.clf

    def predict_features(self, features_280: np.ndarray) -> Dict[str, Any]:
        """
        Runs scaling and Random Forest inference on pre-extracted 280-dim feature vector.
        """
        if not self.is_loaded:
            self.load()

        t_start = time.perf_counter()
        # Input sanitation
        vec = np.asarray(features_280, dtype=np.float32)
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)
        if vec.shape[1] != FEATURE_DIM_MODEL_C:
            raise ValueError(f"Expected input shape (1, {FEATURE_DIM_MODEL_C}), got {vec.shape}")

        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

        # Scaling
        t_sc0 = time.perf_counter()
        scaled_x = self.scaler.transform(vec)
        scaler_latency_ms = round((time.perf_counter() - t_sc0) * 1000, 3)

        # Inference
        t_clf0 = time.perf_counter()
        probs = self.clf.predict_proba(scaled_x)[0]
        clf_latency_ms = round((time.perf_counter() - t_clf0) * 1000, 3)
        inf_latency_ms = round((time.perf_counter() - t_start) * 1000, 3)

        pred_idx = int(np.argmax(probs))
        pred_letter = I2C[pred_idx]
        confidence = round(float(probs[pred_idx]) * 100, 2)

        prob_dict = {I2C[i]: round(float(probs[i]) * 100, 2) for i in range(26)}
        top3_indices = np.argsort(probs)[-3:][::-1]
        top_3 = [(I2C[int(i)], round(float(probs[i]) * 100, 2)) for i in top3_indices]

        return {
            "predicted_letter": pred_letter,
            "confidence": confidence,
            "top_3": top_3,
            "probabilities": prob_dict,
            "scaler_latency_ms": scaler_latency_ms,
            "classifier_latency_ms": clf_latency_ms,
            "inference_latency_ms": inf_latency_ms
        }

    def predict_mediapipe_result(self, mediapipe_result: Any) -> Dict[str, Any]:
        """
        End-to-end inference from a MediaPipe HandLandmarker result.
        """
        t0 = time.perf_counter()
        features_280, hand_count, feat_status = extract_landmarks_model_c(mediapipe_result)
        feat_time_ms = round((time.perf_counter() - t0) * 1000, 2)

        if hand_count == 0:
            # Clean zero-hand return
            return {
                "predicted_letter": "-",
                "confidence": 0.0,
                "top_3": [],
                "probabilities": {c: 0.0 for c in ALPHABET},
                "hand_count": 0,
                "feature_status": feat_status,
                "feature_extraction_ms": feat_time_ms,
                "scaler_ms": 0.0,
                "classifier_ms": 0.0,
                "inference_ms": 0.0,
                "total_latency_ms": feat_time_ms
            }

        res = self.predict_features(features_280)
        tot_time_ms = round(feat_time_ms + res["inference_latency_ms"], 2)

        return {
            "predicted_letter": res["predicted_letter"],
            "confidence": res["confidence"],
            "top_3": res["top_3"],
            "probabilities": res["probabilities"],
            "hand_count": hand_count,
            "feature_status": feat_status,
            "feature_extraction_ms": feat_time_ms,
            "scaler_ms": res.get("scaler_latency_ms", 0.0),
            "classifier_ms": res.get("classifier_latency_ms", 0.0),
            "inference_ms": res["inference_latency_ms"],
            "total_latency_ms": tot_time_ms
        }

    def get_public_metadata(self) -> Dict[str, Any]:
        """Returns safe public metadata without exposing internal filesystem paths."""
        return {
            "recognizer": "Model C (Occlusion-Aware Geometry Random Forest)",
            "architecture": "model_c",
            "version": self.version,
            "feature_schema": self.schema_name,
            "feature_dimensions": self.feature_dim,
            "classes_supported": len(self.classes),
            "is_loaded": self.is_loaded
        }


# Global singleton instance
model_c_recognizer = ModelCRecognizer()
