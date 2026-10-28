#!/usr/bin/env python3
"""
Palm-Canonical Coordinate Transformer for Phase 32B.

Projects hand landmarks into a local anatomical palm coordinate frame:
- Origin: Wrist joint (landmark 0).
- Longitudinal Axis (Y): Vector from Wrist to Middle Finger MCP (landmark 9), normalized.
- Transverse / Lateral Axis (X): Orthonormal to Y and Palm Normal (Index MCP to Pinky MCP direction).
- Palm Normal Axis (Z): Cross product of Transverse and Longitudinal vectors (orthogonal to palm plane).
- Normalization: Distances normalized by palm reference scale ||P_mid_mcp - P_wrist||.

Provides:
- 100% mathematical invariance to in-plane hand rotation in camera space.
- Numerically stable Gram-Schmidt orthonormalization with fallback for collinear/degenerate cases.
- Safe handling of missing landmarks, 0-hand detections, and extreme values (no NaN/Inf).
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
    FINGER_JOINTS,
    _cosine_angle,
    order_two_hands_robust
)
from landmark_model_c import (
    FEATURE_DIM_MODEL_C,
    GEOMETRY_DIM_V2,
    OCCLUSION_DIM,
    extract_occlusion_features
)


def compute_palm_basis(
    pts_21x3: Optional[np.ndarray],
    category_name: str = "unknown",
    mirror_left: bool = False
) -> Tuple[np.ndarray, np.ndarray, float, bool]:
    """
    Computes the orthonormal palm coordinate frame from 21 hand landmarks.

    Returns:
        R: 3x3 orthonormal rotation matrix (rows are x_axis, y_axis, z_axis).
        origin: 3-dim wrist coordinate.
        scale: palm reference distance (wrist to middle MCP).
        is_degenerate: boolean indicating if fallback basis was used.
    """
    default_origin = np.zeros(3, dtype=np.float32)
    default_R = np.eye(3, dtype=np.float32)
    default_scale = 1.0

    if pts_21x3 is None:
        return default_R, default_origin, default_scale, True

    pts = np.asarray(pts_21x3, dtype=np.float32)
    if pts.shape != (21, 3) or not np.isfinite(pts).all():
        return default_R, default_origin, default_scale, True

    wrist = pts[WRIST]
    idx_mcp = pts[INDEX_MCP]
    mid_mcp = pts[MID_MCP]
    pnk_mcp = pts[PINKY_MCP]

    # 1. Longitudinal axis (Y): Wrist -> Mid MCP
    v_long = mid_mcp - wrist
    len_long = float(np.linalg.norm(v_long))
    is_degenerate = False

    if len_long < 1e-4:
        # Degenerate wrist-mcp distance: fallback to camera Y-axis
        y_axis = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        scale = 1.0
        is_degenerate = True
    else:
        y_axis = (v_long / len_long).astype(np.float32)
        scale = len_long

    # 2. Transverse vector: Index MCP -> Pinky MCP
    v_trans = pnk_mcp - idx_mcp
    len_trans = float(np.linalg.norm(v_trans))

    if len_trans < 1e-4:
        # Knuckles degenerate: construct arbitrary perpendicular to y_axis
        is_degenerate = True
        v_temp = np.array([1.0, 0.0, 0.0], dtype=np.float32) if abs(y_axis[0]) < 0.9 else np.array([0.0, 0.0, 1.0], dtype=np.float32)
        n = np.cross(v_temp, y_axis)
    else:
        n = np.cross(v_trans, y_axis)

    len_n = float(np.linalg.norm(n))
    if len_n < 1e-4:
        # Collinear longitudinal and transverse vectors
        is_degenerate = True
        v_temp = np.array([0.0, 0.0, 1.0], dtype=np.float32) if abs(y_axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        n = np.cross(v_temp, y_axis)
        len_n = float(np.linalg.norm(n))
        if len_n < 1e-6:
            n = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            len_n = 1.0

    z_axis = (n / len_n).astype(np.float32)

    # 3. Orthonormal lateral axis (X) = Y x Z
    x_axis = np.cross(y_axis, z_axis).astype(np.float32)
    len_x = float(np.linalg.norm(x_axis))
    if len_x < 1e-4:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        x_axis = x_axis / len_x

    # Re-normalize z to guarantee perfect right-handed orthonormality (det = +1.0)
    z_axis = np.cross(x_axis, y_axis).astype(np.float32)

    # 4. Optional Handedness Mirroring (reflect X and Z for Left hand to align anatomical radial/ulnar axes)
    if mirror_left and category_name.lower() == "left":
        x_axis = -x_axis
        z_axis = -z_axis

    R = np.vstack([x_axis, y_axis, z_axis]).astype(np.float32)
    return R, wrist, scale, is_degenerate


def transform_landmarks_to_palm_canonical(
    pts_21x3: Optional[np.ndarray],
    category_name: str = "unknown",
    mirror_left: bool = False
) -> Tuple[np.ndarray, np.ndarray, float, bool]:
    """
    Transforms 21x3 landmarks into canonical coordinates:
    P'_i = (1 / scale) * R * (P_i - wrist)
    """
    if pts_21x3 is None:
        return np.zeros((21, 3), dtype=np.float32), np.eye(3, dtype=np.float32), 1.0, True

    R, wrist, scale, is_deg = compute_palm_basis(pts_21x3, category_name, mirror_left)
    centered = pts_21x3 - wrist
    canonical = ((centered / scale) @ R.T).astype(np.float32)
    canonical = np.nan_to_num(canonical, nan=0.0, posinf=0.0, neginf=0.0)
    return canonical, R, scale, is_deg


def extract_single_hand_features_canonical(
    landmarks_21x3: Optional[np.ndarray],
    category_name: str = "unknown",
    score: float = 1.0,
    present: bool = True,
    mirror_left: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray], float, bool]:
    """
    Extracts 125 features from a single hand in palm-canonical coordinates.
    Returns:
        (features_125, R_matrix, palm_scale, is_degenerate)
    """
    if not present or landmarks_21x3 is None:
        return np.zeros(FEATURE_DIM_PER_HAND_V2, dtype=np.float32), None, 1.0, True

    pts = np.asarray(landmarks_21x3, dtype=np.float32)
    if pts.shape != (21, 3):
        return np.zeros(FEATURE_DIM_PER_HAND_V2, dtype=np.float32), None, 1.0, True

    canonical_pts, R, palm_scale, is_deg = transform_landmarks_to_palm_canonical(
        pts, category_name=category_name, mirror_left=mirror_left
    )

    # 1. Palm-canonical coordinates (21 x 3 = 63 dims)
    coords_feat = canonical_pts.flatten()

    # In canonical coordinates, the wrist is at (0,0,0) and scale is 1.0
    c_wrist = canonical_pts[WRIST]
    tip_indices = [THUMB_TIP, INDEX_TIP, MID_TIP, RING_TIP, PINKY_TIP]

    # 2. Fingertip-to-wrist normalized distances (5)
    d_tip_wrist = [np.linalg.norm(canonical_pts[t] - c_wrist) for t in tip_indices]

    # 3. Fingertip-to-MCP normalized distances (5)
    d_tip_mcp = [
        np.linalg.norm(canonical_pts[THUMB_TIP] - canonical_pts[THUMB_MCP]),
        np.linalg.norm(canonical_pts[INDEX_TIP] - canonical_pts[INDEX_MCP]),
        np.linalg.norm(canonical_pts[MID_TIP] - canonical_pts[MID_MCP]),
        np.linalg.norm(canonical_pts[RING_TIP] - canonical_pts[RING_MCP]),
        np.linalg.norm(canonical_pts[PINKY_TIP] - canonical_pts[PINKY_MCP]),
    ]

    # 4. Finger extension ratios (tip-to-wrist / mcp-to-wrist) (5)
    ext_ratios = [
        np.linalg.norm(canonical_pts[THUMB_TIP] - c_wrist) / (np.linalg.norm(canonical_pts[THUMB_MCP] - c_wrist) + 1e-6),
        np.linalg.norm(canonical_pts[INDEX_TIP] - c_wrist) / (np.linalg.norm(canonical_pts[INDEX_MCP] - c_wrist) + 1e-6),
        np.linalg.norm(canonical_pts[MID_TIP] - c_wrist) / (np.linalg.norm(canonical_pts[MID_MCP] - c_wrist) + 1e-6),
        np.linalg.norm(canonical_pts[RING_TIP] - c_wrist) / (np.linalg.norm(canonical_pts[RING_MCP] - c_wrist) + 1e-6),
        np.linalg.norm(canonical_pts[PINKY_TIP] - c_wrist) / (np.linalg.norm(canonical_pts[PINKY_MCP] - c_wrist) + 1e-6),
    ]

    # 5. Core inter-fingertip relative distances (6)
    inter_tip_dists = [
        np.linalg.norm(canonical_pts[THUMB_TIP] - canonical_pts[INDEX_TIP]),
        np.linalg.norm(canonical_pts[INDEX_TIP] - canonical_pts[MID_TIP]),
        np.linalg.norm(canonical_pts[MID_TIP] - canonical_pts[RING_TIP]),
        np.linalg.norm(canonical_pts[RING_TIP] - canonical_pts[PINKY_TIP]),
        np.linalg.norm(canonical_pts[THUMB_TIP] - canonical_pts[PINKY_TIP]),
        np.linalg.norm(canonical_pts[INDEX_TIP] - canonical_pts[PINKY_TIP]),
    ]

    # 6. PIP flexion angles (cosine at middle joint) (5)
    pip_flex_angles = [
        _cosine_angle(canonical_pts[THUMB_MCP] - canonical_pts[THUMB_IP], canonical_pts[THUMB_TIP] - canonical_pts[THUMB_IP]),
        _cosine_angle(canonical_pts[INDEX_MCP] - canonical_pts[INDEX_PIP], canonical_pts[INDEX_TIP] - canonical_pts[INDEX_PIP]),
        _cosine_angle(canonical_pts[MID_MCP] - canonical_pts[MID_PIP], canonical_pts[MID_TIP] - canonical_pts[MID_PIP]),
        _cosine_angle(canonical_pts[RING_MCP] - canonical_pts[RING_PIP], canonical_pts[RING_TIP] - canonical_pts[RING_PIP]),
        _cosine_angle(canonical_pts[PINKY_MCP] - canonical_pts[PINKY_PIP], canonical_pts[PINKY_TIP] - canonical_pts[PINKY_PIP]),
    ]

    # 7. Palm normal vector in canonical frame (3) -> (0, 0, 1) by definition
    palm_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # 8. Metadata features (3)
    h_sign = 1.0 if category_name.lower() == "right" else (-1.0 if category_name.lower() == "left" else 0.0)
    meta = [1.0, float(score), float(h_sign)]

    # 9. Finger Curl Ratios (5)
    curl_ratios = []
    for mcp_idx, pip_idx, dip_idx, tip_idx in FINGER_JOINTS:
        direct_dist = np.linalg.norm(canonical_pts[tip_idx] - canonical_pts[mcp_idx])
        bone1 = np.linalg.norm(canonical_pts[pip_idx] - canonical_pts[mcp_idx])
        bone2 = np.linalg.norm(canonical_pts[dip_idx] - canonical_pts[pip_idx])
        bone3 = np.linalg.norm(canonical_pts[tip_idx] - canonical_pts[dip_idx])
        total_len = bone1 + bone2 + bone3 + 1e-6
        curl_ratios.append(float(direct_dist / total_len))

    # 10. DIP Flexion Angles (5)
    dip_flex_angles = [
        _cosine_angle(canonical_pts[THUMB_CMC] - canonical_pts[THUMB_MCP], canonical_pts[THUMB_IP] - canonical_pts[THUMB_MCP]),
        _cosine_angle(canonical_pts[INDEX_PIP] - canonical_pts[INDEX_DIP], canonical_pts[INDEX_TIP] - canonical_pts[INDEX_DIP]),
        _cosine_angle(canonical_pts[MID_PIP] - canonical_pts[MID_DIP], canonical_pts[MID_TIP] - canonical_pts[MID_DIP]),
        _cosine_angle(canonical_pts[RING_PIP] - canonical_pts[RING_DIP], canonical_pts[RING_TIP] - canonical_pts[RING_DIP]),
        _cosine_angle(canonical_pts[PINKY_PIP] - canonical_pts[PINKY_DIP], canonical_pts[PINKY_TIP] - canonical_pts[PINKY_DIP]),
    ]

    # 11. MCP Base Flexion Angles (5)
    mcp_base_angles = [
        _cosine_angle(c_wrist - canonical_pts[THUMB_CMC], canonical_pts[THUMB_MCP] - canonical_pts[THUMB_CMC]),
        _cosine_angle(c_wrist - canonical_pts[INDEX_MCP], canonical_pts[INDEX_PIP] - canonical_pts[INDEX_MCP]),
        _cosine_angle(c_wrist - canonical_pts[MID_MCP], canonical_pts[MID_PIP] - canonical_pts[MID_MCP]),
        _cosine_angle(c_wrist - canonical_pts[RING_MCP], canonical_pts[RING_PIP] - canonical_pts[RING_MCP]),
        _cosine_angle(c_wrist - canonical_pts[PINKY_MCP], canonical_pts[PINKY_PIP] - canonical_pts[PINKY_MCP]),
    ]

    # 12. Signed Perpendicular Distance to Palm Plane (5) -> in canonical frame, this is simply the z-coordinate!
    palm_plane_depths = [float(canonical_pts[t][2]) for t in tip_indices]

    # 13. Additional Inter-Tip Distances (2)
    add_tip_dists = [
        float(np.linalg.norm(canonical_pts[THUMB_TIP] - canonical_pts[MID_TIP])),
        float(np.linalg.norm(canonical_pts[THUMB_TIP] - canonical_pts[RING_TIP])),
    ]

    # 14. Adjacent Finger Directional Spread Angles (4)
    v_thumb = canonical_pts[THUMB_TIP] - canonical_pts[THUMB_MCP]
    v_index = canonical_pts[INDEX_TIP] - canonical_pts[INDEX_MCP]
    v_mid = canonical_pts[MID_TIP] - canonical_pts[MID_MCP]
    v_ring = canonical_pts[RING_TIP] - canonical_pts[RING_MCP]
    v_pinky = canonical_pts[PINKY_TIP] - canonical_pts[PINKY_MCP]

    spread_angles = [
        _cosine_angle(v_thumb, v_index),
        _cosine_angle(v_index, v_mid),
        _cosine_angle(v_mid, v_ring),
        _cosine_angle(v_ring, v_pinky),
    ]

    # 15. Bounding Span and Aspect Ratio in palm frame (4)
    bbox_w = float(np.max(canonical_pts[:, 0]) - np.min(canonical_pts[:, 0]))
    bbox_h = float(np.max(canonical_pts[:, 1]) - np.min(canonical_pts[:, 1]))
    aspect = bbox_w / (bbox_h + 1e-6)
    palm_coverage = 1.0 / (bbox_h + 1e-6)
    span_feats = [bbox_w, bbox_h, aspect, palm_coverage]

    features = np.concatenate([
        coords_feat,
        np.array(d_tip_wrist, dtype=np.float32),
        np.array(d_tip_mcp, dtype=np.float32),
        np.array(ext_ratios, dtype=np.float32),
        np.array(inter_tip_dists, dtype=np.float32),
        np.array(pip_flex_angles, dtype=np.float32),
        palm_normal,
        np.array(meta, dtype=np.float32),
        np.array(curl_ratios, dtype=np.float32),
        np.array(dip_flex_angles, dtype=np.float32),
        np.array(mcp_base_angles, dtype=np.float32),
        np.array(palm_plane_depths, dtype=np.float32),
        np.array(add_tip_dists, dtype=np.float32),
        np.array(spread_angles, dtype=np.float32),
        np.array(span_feats, dtype=np.float32)
    ]).astype(np.float32)

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    assert len(features) == FEATURE_DIM_PER_HAND_V2
    return features, R, palm_scale, is_deg


def extract_inter_hand_features_canonical(
    pts0: Optional[np.ndarray],
    pts1: Optional[np.ndarray],
    scale0: float,
    scale1: float,
    present0: bool,
    present1: bool,
    R0: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Extracts 14 inter-hand features with relative displacement projected into Hand 0's palm frame.
    """
    if not (present0 and present1 and pts0 is not None and pts1 is not None):
        return np.zeros(INTER_HAND_FEATURE_DIM_V2, dtype=np.float32)

    mean_scale = (scale0 + scale1) / 2.0
    if mean_scale < 1e-4:
        mean_scale = 1.0

    w0, w1 = pts0[WRIST], pts1[WRIST]

    # 1. Wrist-to-wrist 3D distance (1)
    d_wrist = float(np.linalg.norm(w0 - w1) / mean_scale)

    # 2. Relative displacement projected onto Hand 0's palm coordinate frame (3)
    if R0 is not None:
        disp_cam = (w0 - w1) / mean_scale
        disp_canon = (R0 @ disp_cam).astype(np.float32)
        dx_wrist, dy_wrist, dz_wrist = float(disp_canon[0]), float(disp_canon[1]), float(disp_canon[2])
    else:
        dx_wrist = float((w0[0] - w1[0]) / mean_scale)
        dy_wrist = float((w0[1] - w1[1]) / mean_scale)
        dz_wrist = float((w0[2] - w1[2]) / mean_scale)

    # 3. Inter-hand fingertip contact distances (6)
    contacts = [
        float(np.linalg.norm(pts0[INDEX_TIP] - pts1[INDEX_TIP]) / mean_scale),
        float(np.linalg.norm(pts0[THUMB_TIP] - pts1[INDEX_TIP]) / mean_scale),
        float(np.linalg.norm(pts0[INDEX_TIP] - pts1[THUMB_TIP]) / mean_scale),
        float(np.linalg.norm(pts0[THUMB_TIP] - pts1[THUMB_TIP]) / mean_scale),
        float(np.linalg.norm(pts0[WRIST] - pts1[INDEX_TIP]) / mean_scale),
        float(np.linalg.norm(pts1[WRIST] - pts0[INDEX_TIP]) / mean_scale),
    ]

    # 4. Inter-palm normal alignment (cosine of normal vectors) (1)
    v1_0, v2_0 = pts0[INDEX_MCP] - w0, pts0[PINKY_MCP] - w0
    v1_1, v2_1 = pts1[INDEX_MCP] - w1, pts1[PINKY_MCP] - w1
    n0 = np.cross(v1_0, v2_0)
    n1 = np.cross(v1_1, v2_1)
    norm_align = _cosine_angle(n0, n1)

    # 5. Centroid-to-centroid distance (1)
    c0 = np.mean(pts0, axis=0)
    c1 = np.mean(pts1, axis=0)
    d_centroid = float(np.linalg.norm(c0 - c1) / mean_scale)

    # 6. Index0 to Middle1 contact distance (1)
    d_idx0_mid1 = float(np.linalg.norm(pts0[INDEX_TIP] - pts1[MID_TIP]) / mean_scale)

    # 7. Both present indicator (1)
    both_present = 1.0

    inter_features = [both_present, d_wrist, dx_wrist, dy_wrist, dz_wrist] + contacts + [norm_align, d_centroid, d_idx0_mid1]
    inter_features = np.nan_to_num(np.array(inter_features, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    assert len(inter_features) == INTER_HAND_FEATURE_DIM_V2
    return inter_features


def extract_landmarks_canonical_v2(
    item: Dict[str, Any],
    suppress_lap_hands: bool = True,
    mirror_left: bool = False,
    return_active_hands: bool = False
) -> Any:
    """
    Extracts 264-dim Geometry v2 features in Palm-Canonical coordinates.
    """
    lms_list = item.get("landmarks", [])
    if not lms_list:
        zeros = np.zeros(TOTAL_FEATURE_DIM_V2, dtype=np.float32)
        return (zeros, [], 0) if return_active_hands else zeros

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

    if len(parsed_hands) == 0:
        zeros = np.zeros(TOTAL_FEATURE_DIM_V2, dtype=np.float32)
        return (zeros, [], 0) if return_active_hands else zeros

    active_hands, _ = select_active_hands(parsed_hands, suppress_lap_hands=suppress_lap_hands)

    deg_count = 0
    if len(active_hands) == 1:
        h0 = active_hands[0]
        feat0, _, _, is_deg = extract_single_hand_features_canonical(
            h0["pts"], h0["category"], h0["score"], present=True, mirror_left=mirror_left
        )
        if is_deg:
            deg_count += 1
        feat1 = np.zeros(FEATURE_DIM_PER_HAND_V2, dtype=np.float32)
        inter = np.zeros(INTER_HAND_FEATURE_DIM_V2, dtype=np.float32)
        feat_264 = np.concatenate([feat0, feat1, inter]).astype(np.float32)
        return (feat_264, [h0], deg_count) if return_active_hands else feat_264

    # Two active hands: order robustly
    h0, h1 = order_two_hands_robust(active_hands[0], active_hands[1])

    feat0, R0, scale0, deg0 = extract_single_hand_features_canonical(
        h0["pts"], h0["category"], h0["score"], present=True, mirror_left=mirror_left
    )
    feat1, R1, scale1, deg1 = extract_single_hand_features_canonical(
        h1["pts"], h1["category"], h1["score"], present=True, mirror_left=mirror_left
    )
    if deg0:
        deg_count += 1
    if deg1:
        deg_count += 1

    inter = extract_inter_hand_features_canonical(
        h0["pts"], h1["pts"], scale0, scale1, True, True, R0=R0
    )

    feat_264 = np.concatenate([feat0, feat1, inter]).astype(np.float32)
    return (feat_264, [h0, h1], deg_count) if return_active_hands else feat_264


def extract_landmarks_canonical_model_c(
    item: Dict[str, Any],
    suppress_lap_hands: bool = True,
    mirror_left: bool = False
) -> Tuple[np.ndarray, int]:
    """
    Extracts full 280-dim feature vector in Palm-Canonical coordinates (264d geometry + 16d occlusion).
    Returns (features_280, degenerate_bases_count).
    """
    feat_264, active_hands, deg_count = extract_landmarks_canonical_v2(
        item, suppress_lap_hands=suppress_lap_hands, mirror_left=mirror_left, return_active_hands=True
    )
    num_active = len(active_hands)
    if num_active >= 2:
        occ_16 = extract_occlusion_features([active_hands[0]["pts"], active_hands[1]["pts"]])
    elif num_active == 1:
        occ_16 = np.zeros(OCCLUSION_DIM, dtype=np.float32)
        occ_16[1] = 1.0  # has_1_hand flag
    else:
        occ_16 = np.zeros(OCCLUSION_DIM, dtype=np.float32)
        occ_16[0] = 1.0  # has_0_hands flag

    feat_280 = np.concatenate([feat_264, occ_16]).astype(np.float32)
    feat_280 = np.nan_to_num(feat_280, nan=0.0, posinf=0.0, neginf=0.0)
    assert len(feat_280) == FEATURE_DIM_MODEL_C
    return feat_280, deg_count
