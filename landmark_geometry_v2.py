#!/usr/bin/env python3
"""
Phase 16 — Advanced Landmark Geometry Feature Extractor (v2)

Extends the 200-dimensional Phase 14A feature representation to 264 dimensions
to resolve fine-grained finger-shape confusions:
- N vs M (Ring finger curl ratio, Thumb-to-Ring tip distance, palm-plane signed depth)
- X vs V (Index DIP flexion angle, Index vs Middle curl ratios, finger separation angle)
- L vs Y/U (Thumb-Index spread angle, Thumb-Pinky spread angle, finger curl ratios)
- A vs I/G (Fist compaction ratio, pinky extension, thumb-to-index proximity)

Design Guarantees:
1. Scale-invariant: All distances normalized by palm scale (Wrist to Middle MCP).
2. Translation-invariant: All 3D coordinates centered relative to the wrist.
3. Rotation-invariant: Joint angles and curl ratios computed via dot products.
4. Deterministic Ordering: Hand 0 (Primary or Leftmost) -> Hand 1 (Secondary or Rightmost) -> Inter-hand.
5. Numerical Safety: Protected against division by zero (eps=1e-7) and degenerate detections.
6. Zero fabrication: Missing hands produce strictly zero-filled vectors with presence=0.0.

Feature Vector Dimensions:
- Hand 0 features: 125 dimensions
- Hand 1 features: 125 dimensions
- Inter-hand features: 14 dimensions
Total: 264 dimensions
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple

FEATURE_DIM_PER_HAND_V2 = 125
INTER_HAND_FEATURE_DIM_V2 = 14
TOTAL_FEATURE_DIM_V2 = FEATURE_DIM_PER_HAND_V2 * 2 + INTER_HAND_FEATURE_DIM_V2  # 264

# MediaPipe Hand Landmark Indices:
# 0: Wrist
# 1-4: Thumb (CMC=1, MCP=2, IP=3, TIP=4)
# 5-8: Index (MCP=5, PIP=6, DIP=7, TIP=8)
# 9-12: Middle (MCP=9, PIP=10, DIP=11, TIP=12)
# 13-16: Ring (MCP=13, PIP=14, DIP=15, TIP=16)
# 17-20: Pinky (MCP=17, PIP=18, DIP=19, TIP=20)
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MID_MCP, MID_PIP, MID_DIP, MID_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGER_JOINTS = [
    # (mcp, pip, dip, tip)
    (THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP),   # Thumb
    (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),   # Index
    (MID_MCP, MID_PIP, MID_DIP, MID_TIP),           # Middle
    (RING_MCP, RING_PIP, RING_DIP, RING_TIP),       # Ring
    (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP)    # Pinky
]


def _safe_norm(v: np.ndarray, eps: float = 1e-7) -> float:
    return float(np.linalg.norm(v)) + eps


def _cosine_angle(v1: np.ndarray, v2: np.ndarray, eps: float = 1e-7) -> float:
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm < eps:
        return 0.0
    return float(np.clip(dot / norm, -1.0, 1.0))


def extract_single_hand_features_v2(
    landmarks_21x3: Optional[np.ndarray],
    category_name: str = "unknown",
    score: float = 1.0,
    present: bool = True
) -> np.ndarray:
    """
    Extracts 125 geometric features from a single hand of 21 (x, y, z) landmarks.
    """
    if not present or landmarks_21x3 is None:
        return np.zeros(FEATURE_DIM_PER_HAND_V2, dtype=np.float32)

    pts = np.asarray(landmarks_21x3, dtype=np.float32)
    if pts.shape != (21, 3):
        return np.zeros(FEATURE_DIM_PER_HAND_V2, dtype=np.float32)

    wrist = pts[WRIST]
    mid_mcp = pts[MID_MCP]
    palm_scale = np.linalg.norm(mid_mcp - wrist)
    if palm_scale < 1e-4:
        palm_scale = 1.0

    # 1. Wrist-relative, scale-normalized 3D coordinates (21 x 3 = 63)
    coords_rel = (pts - wrist) / palm_scale
    coords_feat = coords_rel.flatten()

    # 2. Fingertip-to-wrist normalized distances (5)
    tip_indices = [THUMB_TIP, INDEX_TIP, MID_TIP, RING_TIP, PINKY_TIP]
    d_tip_wrist = [np.linalg.norm(pts[t] - wrist) / palm_scale for t in tip_indices]

    # 3. Fingertip-to-MCP normalized distances (5)
    d_tip_mcp = [
        np.linalg.norm(pts[THUMB_TIP] - pts[THUMB_MCP]) / palm_scale,
        np.linalg.norm(pts[INDEX_TIP] - pts[INDEX_MCP]) / palm_scale,
        np.linalg.norm(pts[MID_TIP] - pts[MID_MCP]) / palm_scale,
        np.linalg.norm(pts[RING_TIP] - pts[RING_MCP]) / palm_scale,
        np.linalg.norm(pts[PINKY_TIP] - pts[PINKY_MCP]) / palm_scale,
    ]

    # 4. Finger extension ratios (tip-to-wrist / mcp-to-wrist) (5)
    ext_ratios = [
        np.linalg.norm(pts[THUMB_TIP] - wrist) / (np.linalg.norm(pts[THUMB_MCP] - wrist) + 1e-6),
        np.linalg.norm(pts[INDEX_TIP] - wrist) / (np.linalg.norm(pts[INDEX_MCP] - wrist) + 1e-6),
        np.linalg.norm(pts[MID_TIP] - wrist) / (np.linalg.norm(pts[MID_MCP] - wrist) + 1e-6),
        np.linalg.norm(pts[RING_TIP] - wrist) / (np.linalg.norm(pts[RING_MCP] - wrist) + 1e-6),
        np.linalg.norm(pts[PINKY_TIP] - wrist) / (np.linalg.norm(pts[PINKY_MCP] - wrist) + 1e-6),
    ]

    # 5. Core inter-fingertip relative distances (6)
    # Pairs: (Thumb, Index), (Index, Mid), (Mid, Ring), (Ring, Pinky), (Thumb, Pinky), (Index, Pinky)
    inter_tip_dists = [
        np.linalg.norm(pts[THUMB_TIP] - pts[INDEX_TIP]) / palm_scale,
        np.linalg.norm(pts[INDEX_TIP] - pts[MID_TIP]) / palm_scale,
        np.linalg.norm(pts[MID_TIP] - pts[RING_TIP]) / palm_scale,
        np.linalg.norm(pts[RING_TIP] - pts[PINKY_TIP]) / palm_scale,
        np.linalg.norm(pts[THUMB_TIP] - pts[PINKY_TIP]) / palm_scale,
        np.linalg.norm(pts[INDEX_TIP] - pts[PINKY_TIP]) / palm_scale,
    ]

    # 6. PIP flexion angles (cosine at middle joint) (5)
    pip_flex_angles = [
        _cosine_angle(pts[THUMB_MCP] - pts[THUMB_IP], pts[THUMB_TIP] - pts[THUMB_IP]),
        _cosine_angle(pts[INDEX_MCP] - pts[INDEX_PIP], pts[INDEX_TIP] - pts[INDEX_PIP]),
        _cosine_angle(pts[MID_MCP] - pts[MID_PIP], pts[MID_TIP] - pts[MID_PIP]),
        _cosine_angle(pts[RING_MCP] - pts[RING_PIP], pts[RING_TIP] - pts[RING_PIP]),
        _cosine_angle(pts[PINKY_MCP] - pts[PINKY_PIP], pts[PINKY_TIP] - pts[PINKY_PIP]),
    ]

    # 7. Palm normal vector (3)
    v1 = pts[INDEX_MCP] - wrist
    v2 = pts[PINKY_MCP] - wrist
    norm_vec = np.cross(v1, v2)
    norm_len = np.linalg.norm(norm_vec)
    if norm_len > 1e-6:
        palm_normal = norm_vec / norm_len
    else:
        palm_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # 8. Metadata features: presence, confidence, handedness_sign (3)
    h_sign = 1.0 if category_name.lower() == "right" else (-1.0 if category_name.lower() == "left" else 0.0)
    meta = [1.0, float(score), float(h_sign)]

    # --- ADVANCED GEOMETRY FEATURES FOR N/M AND X/V (30 Dims) ---

    # 9. Finger Curl Ratios (5)
    # Ratio = ||tip - mcp|| / (sum of individual bone segment lengths)
    # Measures how tightly curled a finger is (approx 0.25 tightly curled, 1.0 straight extended)
    curl_ratios = []
    for mcp_idx, pip_idx, dip_idx, tip_idx in FINGER_JOINTS:
        direct_dist = np.linalg.norm(pts[tip_idx] - pts[mcp_idx])
        bone1 = np.linalg.norm(pts[pip_idx] - pts[mcp_idx])
        bone2 = np.linalg.norm(pts[dip_idx] - pts[pip_idx])
        bone3 = np.linalg.norm(pts[tip_idx] - pts[dip_idx])
        total_len = bone1 + bone2 + bone3 + 1e-6
        curl_ratios.append(float(direct_dist / total_len))

    # 10. DIP Flexion Angles (cosine at DIP joint) (5)
    # Directly detects hook/bent shapes (critical for X vs V!)
    dip_flex_angles = [
        _cosine_angle(pts[THUMB_CMC] - pts[THUMB_MCP], pts[THUMB_IP] - pts[THUMB_MCP]), # Thumb MCP angle
        _cosine_angle(pts[INDEX_PIP] - pts[INDEX_DIP], pts[INDEX_TIP] - pts[INDEX_DIP]),
        _cosine_angle(pts[MID_PIP] - pts[MID_DIP], pts[MID_TIP] - pts[MID_DIP]),
        _cosine_angle(pts[RING_PIP] - pts[RING_DIP], pts[RING_TIP] - pts[RING_DIP]),
        _cosine_angle(pts[PINKY_PIP] - pts[PINKY_DIP], pts[PINKY_TIP] - pts[PINKY_DIP]),
    ]

    # 11. MCP Base Flexion Angles (cosine of wrist-mcp vs mcp-pip) (5)
    mcp_base_angles = [
        _cosine_angle(wrist - pts[THUMB_CMC], pts[THUMB_MCP] - pts[THUMB_CMC]),
        _cosine_angle(wrist - pts[INDEX_MCP], pts[INDEX_PIP] - pts[INDEX_MCP]),
        _cosine_angle(wrist - pts[MID_MCP], pts[MID_PIP] - pts[MID_MCP]),
        _cosine_angle(wrist - pts[RING_MCP], pts[RING_PIP] - pts[RING_MCP]),
        _cosine_angle(wrist - pts[PINKY_MCP], pts[PINKY_PIP] - pts[PINKY_MCP]),
    ]

    # 12. Signed Perpendicular Distance to Palm Plane (5)
    # d = ((tip - wrist) . palm_normal) / palm_scale
    # Determines if fingers are folded inside palm or sitting above thumb (critical for N vs M)
    palm_plane_depths = [
        float(np.dot(pts[t] - wrist, palm_normal) / palm_scale) for t in tip_indices
    ]

    # 13. Additional Inter-Tip Distances: Thumb-to-Middle & Thumb-to-Ring (2)
    # Essential for N vs M (in M, ring tip touches thumb; in N, ring is curled away)
    add_tip_dists = [
        float(np.linalg.norm(pts[THUMB_TIP] - pts[MID_TIP]) / palm_scale),
        float(np.linalg.norm(pts[THUMB_TIP] - pts[RING_TIP]) / palm_scale),
    ]

    # 14. Adjacent Finger Directional Spread Angles (4)
    # Direction vectors from MCP to TIP
    v_thumb = pts[THUMB_TIP] - pts[THUMB_MCP]
    v_index = pts[INDEX_TIP] - pts[INDEX_MCP]
    v_mid = pts[MID_TIP] - pts[MID_MCP]
    v_ring = pts[RING_TIP] - pts[RING_MCP]
    v_pinky = pts[PINKY_TIP] - pts[PINKY_MCP]

    spread_angles = [
        _cosine_angle(v_thumb, v_index),  # Spread between thumb & index (L vs others)
        _cosine_angle(v_index, v_mid),    # Spread between index & middle (V vs U/fist)
        _cosine_angle(v_mid, v_ring),     # Spread between middle & ring (W vs others)
        _cosine_angle(v_ring, v_pinky),   # Spread between ring & pinky (Y vs others)
    ]

    # 15. Bounding Span and Aspect Ratio (4)
    bbox_w = (np.max(pts[:, 0]) - np.min(pts[:, 0])) / palm_scale
    bbox_h = (np.max(pts[:, 1]) - np.min(pts[:, 1])) / palm_scale
    aspect = bbox_w / (bbox_h + 1e-6)
    palm_coverage = palm_scale / (bbox_h + 1e-6)
    span_feats = [float(bbox_w), float(bbox_h), float(aspect), float(palm_coverage)]

    # Concatenate all 125 features
    features = np.concatenate([
        coords_feat,                                  # 63
        np.array(d_tip_wrist, dtype=np.float32),      # 5
        np.array(d_tip_mcp, dtype=np.float32),        # 5
        np.array(ext_ratios, dtype=np.float32),       # 5
        np.array(inter_tip_dists, dtype=np.float32),  # 6
        np.array(pip_flex_angles, dtype=np.float32),  # 5
        palm_normal.astype(np.float32),               # 3
        np.array(meta, dtype=np.float32),             # 3
        # V2 additions:
        np.array(curl_ratios, dtype=np.float32),      # 5
        np.array(dip_flex_angles, dtype=np.float32),  # 5
        np.array(mcp_base_angles, dtype=np.float32),  # 5
        np.array(palm_plane_depths, dtype=np.float32),# 5
        np.array(add_tip_dists, dtype=np.float32),    # 2
        np.array(spread_angles, dtype=np.float32),    # 4
        np.array(span_feats, dtype=np.float32)        # 4
    ])
    assert len(features) == FEATURE_DIM_PER_HAND_V2, f"Hand feature len mismatch: {len(features)} != {FEATURE_DIM_PER_HAND_V2}"
    return features.astype(np.float32)


def extract_inter_hand_features_v2(
    pts0: Optional[np.ndarray],
    pts1: Optional[np.ndarray],
    scale0: float,
    scale1: float,
    present0: bool,
    present1: bool
) -> np.ndarray:
    """
    Extracts 14 inter-hand interaction features when both hands are present.
    """
    if not (present0 and present1 and pts0 is not None and pts1 is not None):
        return np.zeros(INTER_HAND_FEATURE_DIM_V2, dtype=np.float32)

    mean_scale = (scale0 + scale1) / 2.0
    if mean_scale < 1e-4:
        mean_scale = 1.0

    w0, w1 = pts0[WRIST], pts1[WRIST]

    # 1. Wrist-to-wrist 3D distance (1)
    d_wrist = np.linalg.norm(w0 - w1) / mean_scale

    # 2. Relative horizontal, vertical, and depth displacement (3)
    dx_wrist = (w0[0] - w1[0]) / mean_scale
    dy_wrist = (w0[1] - w1[1]) / mean_scale
    dz_wrist = (w0[2] - w1[2]) / mean_scale

    # 3. Inter-hand fingertip contact distances (6)
    contacts = [
        np.linalg.norm(pts0[INDEX_TIP] - pts1[INDEX_TIP]) / mean_scale,
        np.linalg.norm(pts0[THUMB_TIP] - pts1[INDEX_TIP]) / mean_scale,
        np.linalg.norm(pts0[INDEX_TIP] - pts1[THUMB_TIP]) / mean_scale,
        np.linalg.norm(pts0[THUMB_TIP] - pts1[THUMB_TIP]) / mean_scale,
        np.linalg.norm(pts0[WRIST] - pts1[INDEX_TIP]) / mean_scale,
        np.linalg.norm(pts1[WRIST] - pts0[INDEX_TIP]) / mean_scale,
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
    d_centroid = np.linalg.norm(c0 - c1) / mean_scale

    # 6. Index0 to Middle1 contact distance (1)
    d_idx0_mid1 = np.linalg.norm(pts0[INDEX_TIP] - pts1[MID_TIP]) / mean_scale

    # 7. Both present indicator (1)
    both_present = 1.0

    inter_features = [both_present, d_wrist, dx_wrist, dy_wrist, dz_wrist] + contacts + [norm_align, d_centroid, d_idx0_mid1]
    assert len(inter_features) == INTER_HAND_FEATURE_DIM_V2, f"Inter-hand dim mismatch: {len(inter_features)} != {INTER_HAND_FEATURE_DIM_V2}"
    return np.array(inter_features, dtype=np.float32)


def order_two_hands_robust(
    h_a: Dict[str, Any],
    h_b: Dict[str, Any],
    deadband: float = 0.02
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Deterministically orders two active hands for Model C slot assignment.

    1. Primary Order: Screen-space wrist_x ascending (leftmost hand in camera view -> Slot 0, rightmost -> Slot 1),
       preserving exact alignment with Model C training.
    2. Crossing Deadband (|dx| < deadband): When wrists are horizontally close or crossing,
       resolves tie using high-confidence distinct handedness:
       - User Right hand (naturally on camera left) -> Slot 0
       - User Left hand (naturally on camera right) -> Slot 1
    3. Ambiguous Fallback: If handedness is identical, low-confidence (<0.85), or unknown,
       breaks tie deterministically by wrist_x (or wrist_y if exact tie), preventing random jitter.
    """
    dx = float(h_a.get("wrist_x", h_a["pts"][0, 0]) - h_b.get("wrist_x", h_b["pts"][0, 0]))
    if abs(dx) >= deadband:
        return (h_a, h_b) if dx < 0 else (h_b, h_a)

    cat_a = str(h_a.get("category", "unknown")).lower()
    cat_b = str(h_b.get("category", "unknown")).lower()
    score_a = float(h_a.get("score", 1.0))
    score_b = float(h_b.get("score", 1.0))

    if cat_a != cat_b and score_a >= 0.85 and score_b >= 0.85:
        if cat_a == "right" and cat_b == "left":
            return (h_a, h_b)
        elif cat_a == "left" and cat_b == "right":
            return (h_b, h_a)

    # Fallback for crossing tie: maintain wrist_x (or wrist_y if exactly equal)
    if abs(dx) > 1e-6:
        return (h_a, h_b) if dx < 0 else (h_b, h_a)
    wy_a = float(h_a.get("wrist_y", h_a["pts"][0, 1]))
    wy_b = float(h_b.get("wrist_y", h_b["pts"][0, 1]))
    return (h_a, h_b) if wy_a <= wy_b else (h_b, h_a)


def extract_landmarks_from_manifest_item_v2(
    item: Dict[str, Any],
    suppress_lap_hands: bool = True,
    return_active_hands: bool = False
) -> Any:
    """
    Extracts the full 264-dim feature vector from a split manifest dictionary item.
    """
    from landmark_feature_extractor import select_active_hands

    lms_list = item.get("landmarks", [])
    if not lms_list:
        zeros = np.zeros(TOTAL_FEATURE_DIM_V2, dtype=np.float32)
        return (zeros, []) if return_active_hands else zeros

    parsed_hands = []
    for h in lms_list:
        raw_pts = h.get("landmarks", [])
        if len(raw_pts) != 21:
            continue
        arr = np.array([[p["x"], p["y"], p["z"]] for p in raw_pts], dtype=np.float32)
        cat = h.get("category_name", "unknown")
        score = h.get("score", 1.0)
        wrist_x = arr[WRIST, 0]
        wrist_y = arr[WRIST, 1]
        scale = np.linalg.norm(arr[MID_MCP] - arr[WRIST])
        if scale < 1e-4:
            scale = 1.0
        parsed_hands.append({
            "pts": arr,
            "category": cat,
            "score": score,
            "wrist_x": wrist_x,
            "wrist_y": wrist_y,
            "scale": scale
        })

    if len(parsed_hands) == 0:
        zeros = np.zeros(TOTAL_FEATURE_DIM_V2, dtype=np.float32)
        return (zeros, []) if return_active_hands else zeros

    active_hands, _ = select_active_hands(parsed_hands, suppress_lap_hands=suppress_lap_hands)

    if len(active_hands) == 1:
        h0 = active_hands[0]
        feat0 = extract_single_hand_features_v2(h0["pts"], h0["category"], h0["score"], present=True)
        feat1 = np.zeros(FEATURE_DIM_PER_HAND_V2, dtype=np.float32)
        inter = np.zeros(INTER_HAND_FEATURE_DIM_V2, dtype=np.float32)
        feat_264 = np.concatenate([feat0, feat1, inter]).astype(np.float32)
        return (feat_264, [h0]) if return_active_hands else feat_264

    h0, h1 = order_two_hands_robust(active_hands[0], active_hands[1])

    feat0 = extract_single_hand_features_v2(h0["pts"], h0["category"], h0["score"], present=True)
    feat1 = extract_single_hand_features_v2(h1["pts"], h1["category"], h1["score"], present=True)
    inter = extract_inter_hand_features_v2(h0["pts"], h1["pts"], h0["scale"], h1["scale"], True, True)

    feat_264 = np.concatenate([feat0, feat1, inter]).astype(np.float32)
    return (feat_264, [h0, h1]) if return_active_hands else feat_264


def extract_landmarks_from_mediapipe_result_v2(
    detection_result: Any,
    suppress_lap_hands: bool = True,
    return_active_hands: bool = False
) -> Any:
    """
    Extracts the full 264-dim feature vector directly from a live MediaPipe HandLandmarker result.
    If return_active_hands is True, returns (features_264, active_hands).
    """
    from landmark_feature_extractor import select_active_hands

    if detection_result is None or not detection_result.hand_landmarks:
        zero_vec = np.zeros(TOTAL_FEATURE_DIM_V2, dtype=np.float32)
        return (zero_vec, []) if return_active_hands else zero_vec

    parsed_hands = []
    for h_idx, h_lms in enumerate(detection_result.hand_landmarks):
        if len(h_lms) != 21:
            continue
        arr = np.array([[lm.x, lm.y, lm.z] for lm in h_lms], dtype=np.float32)
        cat = "unknown"
        score = 1.0
        if detection_result.handedness and len(detection_result.handedness) > h_idx:
            cat = detection_result.handedness[h_idx][0].category_name
            score = float(detection_result.handedness[h_idx][0].score)
        wrist_x = arr[WRIST, 0]
        wrist_y = arr[WRIST, 1]
        scale = np.linalg.norm(arr[MID_MCP] - arr[WRIST])
        if scale < 1e-4:
            scale = 1.0
        parsed_hands.append({
            "pts": arr,
            "category": cat,
            "score": score,
            "wrist_x": wrist_x,
            "wrist_y": wrist_y,
            "scale": scale
        })

    if len(parsed_hands) == 0:
        zero_vec = np.zeros(TOTAL_FEATURE_DIM_V2, dtype=np.float32)
        return (zero_vec, []) if return_active_hands else zero_vec

    active_hands, _ = select_active_hands(parsed_hands, suppress_lap_hands=suppress_lap_hands)

    if len(active_hands) == 1:
        h0 = active_hands[0]
        feat0 = extract_single_hand_features_v2(h0["pts"], h0["category"], h0["score"], present=True)
        feat1 = np.zeros(FEATURE_DIM_PER_HAND_V2, dtype=np.float32)
        inter = np.zeros(INTER_HAND_FEATURE_DIM_V2, dtype=np.float32)
        res_vec = np.concatenate([feat0, feat1, inter]).astype(np.float32)
        return (res_vec, active_hands) if return_active_hands else res_vec

    h0, h1 = order_two_hands_robust(active_hands[0], active_hands[1])
    ordered_hands = [h0, h1]

    feat0 = extract_single_hand_features_v2(h0["pts"], h0["category"], h0["score"], present=True)
    feat1 = extract_single_hand_features_v2(h1["pts"], h1["category"], h1["score"], present=True)
    inter = extract_inter_hand_features_v2(h0["pts"], h1["pts"], h0["scale"], h1["scale"], True, True)

    res_vec = np.concatenate([feat0, feat1, inter]).astype(np.float32)
    return (res_vec, ordered_hands) if return_active_hands else res_vec
