#!/usr/bin/env python3
"""
Orientation-Preserving Rotation-Robust Feature Extractor for Phase 32C.

Retains the complete 280-dimensional camera-space reference representation
(preserving the phonemic orientation signals necessary to distinguish ISL signs like H, R, L, Y)
while exploring isolated complementary descriptors:

- Variant A (Baseline 32A): 280 dimensions (264d Geometry v2 in camera space + 16d Occlusion).
- Variant B (Baseline + Explicit Hand Orientation): 296 dimensions (280d baseline + 16d explicit global orientation vectors & roll angles).
- Variant C (Baseline + Complementary Canonical Pose): 316 dimensions (280d baseline + 36d rotation-invariant canonical fingertip coordinates).
- Variant D (Hybrid Decomposed Representation): 332 dimensions (280d baseline + 16d explicit orientation + 36d canonical pose).
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from landmark_feature_extractor import select_active_hands
from landmark_geometry_v2 import (
    WRIST, THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP,
    INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP,
    MID_MCP, MID_PIP, MID_DIP, MID_TIP,
    RING_MCP, RING_PIP, RING_DIP, RING_TIP,
    PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP,
    FEATURE_DIM_PER_HAND_V2,
    INTER_HAND_FEATURE_DIM_V2,
    TOTAL_FEATURE_DIM_V2,
    order_two_hands_robust,
    extract_single_hand_features_v2,
    extract_inter_hand_features_v2
)
from landmark_model_c import (
    FEATURE_DIM_MODEL_C,
    OCCLUSION_DIM,
    extract_occlusion_features
)
from palm_canonical_transformer import (
    compute_palm_basis,
    transform_landmarks_to_palm_canonical
)

ORIENTATION_DIM_PER_HAND = 8
ORIENTATION_DIM_TOTAL = 16

CANONICAL_POSE_DIM_PER_HAND = 18
CANONICAL_POSE_DIM_TOTAL = 36

DIM_VARIANT_A = FEATURE_DIM_MODEL_C                # 280
DIM_VARIANT_B = FEATURE_DIM_MODEL_C + 16          # 296
DIM_VARIANT_C = FEATURE_DIM_MODEL_C + 36          # 316
DIM_VARIANT_D = FEATURE_DIM_MODEL_C + 16 + 36     # 332


def extract_orientation_descriptors(
    pts_21x3: Optional[np.ndarray],
    present: bool = True
) -> np.ndarray:
    """
    Extracts 8 explicit orientation features from a single hand in camera coordinates:
    - Longitudinal unit vector (vx, vy, vz): 3 dims
    - Transverse unit vector (tx, ty, tz): 3 dims
    - 2D roll angle encoding (cos_theta, sin_theta): 2 dims
    """
    if not present or pts_21x3 is None:
        return np.zeros(ORIENTATION_DIM_PER_HAND, dtype=np.float32)

    pts = np.asarray(pts_21x3, dtype=np.float32)
    if pts.shape != (21, 3):
        return np.zeros(ORIENTATION_DIM_PER_HAND, dtype=np.float32)

    w = pts[WRIST]
    m_mcp = pts[MID_MCP]
    i_mcp = pts[INDEX_MCP]
    p_mcp = pts[PINKY_MCP]

    # 1. Longitudinal direction vector (Wrist -> Mid MCP)
    v_long = m_mcp - w
    len_long = float(np.linalg.norm(v_long))
    if len_long < 1e-4:
        v_long_u = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    else:
        v_long_u = (v_long / len_long).astype(np.float32)

    # 2. Transverse direction vector (Index MCP -> Pinky MCP)
    v_trans = p_mcp - i_mcp
    len_trans = float(np.linalg.norm(v_trans))
    if len_trans < 1e-4:
        v_trans_u = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        v_trans_u = (v_trans / len_trans).astype(np.float32)

    # 3. In-plane roll angle (direction of fingers in camera plane)
    theta_roll = float(np.arctan2(v_long_u[1], v_long_u[0]))
    cos_th = float(np.cos(theta_roll))
    sin_th = float(np.sin(theta_roll))

    feats = np.array([
        v_long_u[0], v_long_u[1], v_long_u[2],
        v_trans_u[0], v_trans_u[1], v_trans_u[2],
        cos_th, sin_th
    ], dtype=np.float32)

    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def extract_canonical_pose_descriptors(
    pts_21x3: Optional[np.ndarray],
    category_name: str = "unknown",
    present: bool = True
) -> np.ndarray:
    """
    Extracts 18 rotation-invariant palm-canonical coordinates for key pose landmarks:
    - 5 Fingertips: Thumb, Index, Middle, Ring, Pinky (15 dims)
    - Thumb IP joint (3 dims)
    """
    if not present or pts_21x3 is None:
        return np.zeros(CANONICAL_POSE_DIM_PER_HAND, dtype=np.float32)

    pts = np.asarray(pts_21x3, dtype=np.float32)
    if pts.shape != (21, 3):
        return np.zeros(CANONICAL_POSE_DIM_PER_HAND, dtype=np.float32)

    canon_pts, _, _, _ = transform_landmarks_to_palm_canonical(
        pts, category_name=category_name, mirror_left=False
    )

    pose_indices = [THUMB_TIP, INDEX_TIP, MID_TIP, RING_TIP, PINKY_TIP, THUMB_IP]
    pose_coords = canon_pts[pose_indices].flatten().astype(np.float32)

    assert len(pose_coords) == CANONICAL_POSE_DIM_PER_HAND
    return np.nan_to_num(pose_coords, nan=0.0, posinf=0.0, neginf=0.0)


def extract_phase32c_features_from_item(
    item: Dict[str, Any],
    variant: str = "A",
    suppress_lap_hands: bool = True
) -> np.ndarray:
    """
    Extracts features for a sample according to the requested experimental variant.
    Variants:
        "A": 280 dims (Phase 32A Baseline camera-space)
        "B": 296 dims (Baseline + 16d Explicit Orientation Descriptors)
        "C": 316 dims (Baseline + 36d Complementary Canonical Pose Descriptors)
        "D": 332 dims (Baseline + 16d Orientation + 36d Canonical Pose)
    """
    lms_list = item.get("landmarks", [])

    parsed_hands = []
    for h in lms_list:
        raw_pts = h.get("landmarks", [])
        if len(raw_pts) != 21:
            continue
        arr = np.array([[p["x"], p["y"], p["z"]] for p in raw_pts], dtype=np.float32)
        cat = h.get("category_name", "unknown")
        score = float(h.get("score", 1.0))
        scale = float(np.linalg.norm(arr[MID_MCP] - arr[WRIST]))
        if scale < 1e-4:
            scale = 1.0
        parsed_hands.append({
            "pts": arr,
            "category": cat,
            "score": score,
            "wrist_x": float(arr[WRIST, 0]),
            "wrist_y": float(arr[WRIST, 1]),
            "scale": scale
        })

    active_hands, _ = select_active_hands(parsed_hands, suppress_lap_hands=suppress_lap_hands)

    # 1. Geometry v2 (264 dims) in camera space
    if len(active_hands) == 0:
        feat0 = np.zeros(FEATURE_DIM_PER_HAND_V2, dtype=np.float32)
        feat1 = np.zeros(FEATURE_DIM_PER_HAND_V2, dtype=np.float32)
        inter = np.zeros(INTER_HAND_FEATURE_DIM_V2, dtype=np.float32)
        occ_16 = np.zeros(OCCLUSION_DIM, dtype=np.float32)
        occ_16[0] = 1.0  # 0 hands flag
        h0_pts, h1_pts = None, None
        h0_cat, h1_cat = "unknown", "unknown"
        h0_pres, h1_pres = False, False
    elif len(active_hands) == 1:
        h0 = active_hands[0]
        feat0 = extract_single_hand_features_v2(h0["pts"], h0["category"], h0["score"], present=True)
        feat1 = np.zeros(FEATURE_DIM_PER_HAND_V2, dtype=np.float32)
        inter = np.zeros(INTER_HAND_FEATURE_DIM_V2, dtype=np.float32)
        occ_16 = np.zeros(OCCLUSION_DIM, dtype=np.float32)
        occ_16[1] = 1.0  # 1 hand flag
        h0_pts, h1_pts = h0["pts"], None
        h0_cat, h1_cat = h0["category"], "unknown"
        h0_pres, h1_pres = True, False
    else:
        h0, h1 = order_two_hands_robust(active_hands[0], active_hands[1])
        feat0 = extract_single_hand_features_v2(h0["pts"], h0["category"], h0["score"], present=True)
        feat1 = extract_single_hand_features_v2(h1["pts"], h1["category"], h1["score"], present=True)
        inter = extract_inter_hand_features_v2(h0["pts"], h1["pts"], h0["scale"], h1["scale"], True, True)
        occ_16 = extract_occlusion_features([h0["pts"], h1["pts"]])
        h0_pts, h1_pts = h0["pts"], h1["pts"]
        h0_cat, h1_cat = h0["category"], h1["category"]
        h0_pres, h1_pres = True, True

    feat_264 = np.concatenate([feat0, feat1, inter]).astype(np.float32)
    feat_280 = np.concatenate([feat_264, occ_16]).astype(np.float32)

    if variant == "A":
        assert len(feat_280) == DIM_VARIANT_A
        return feat_280

    # 2. Candidate Extension 1: Explicit Hand Orientation (16 dims)
    orient0 = extract_orientation_descriptors(h0_pts, present=h0_pres)
    orient1 = extract_orientation_descriptors(h1_pts, present=h1_pres)
    orient_16 = np.concatenate([orient0, orient1]).astype(np.float32)

    if variant == "B":
        feat_296 = np.concatenate([feat_280, orient_16]).astype(np.float32)
        assert len(feat_296) == DIM_VARIANT_B
        return feat_296

    # 3. Candidate Extension 2: Complementary Palm-Canonical Pose (36 dims)
    pose0 = extract_canonical_pose_descriptors(h0_pts, category_name=h0_cat, present=h0_pres)
    pose1 = extract_canonical_pose_descriptors(h1_pts, category_name=h1_cat, present=h1_pres)
    pose_36 = np.concatenate([pose0, pose1]).astype(np.float32)

    if variant == "C":
        feat_316 = np.concatenate([feat_280, pose_36]).astype(np.float32)
        assert len(feat_316) == DIM_VARIANT_C
        return feat_316

    if variant == "D":
        feat_332 = np.concatenate([feat_280, orient_16, pose_36]).astype(np.float32)
        assert len(feat_332) == DIM_VARIANT_D
        return feat_332

    raise ValueError(f"Unknown variant '{variant}'. Supported: 'A', 'B', 'C', 'D'")
