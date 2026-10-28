#!/usr/bin/env python3
"""
Phase 14A — Landmark Feature Extractor

Extracts robust, scale-invariant, translation-invariant geometric features
from MediaPipe hand landmark detections for 26-class ISL recognition.

Design Principles:
1. Scale-invariant: All distances and coordinates are normalized by palm scale
   (Euclidean distance between Wrist landmark 0 and Middle MCP landmark 9).
2. Translation-invariant: All 3D joint coordinates are centered relative to the wrist.
3. Deterministic Canonical Hand Ordering:
   - When 0 hands are detected: 194-dim zero vector with presence flags = 0.
   - When 1 hand is detected: Hand 0 holds the detected hand (Primary Hand, presence=1.0).
     Hand 1 is strictly zero-padded (presence=0.0).
   - When 2 hands are detected: Hands are sorted deterministically by frame x-coordinate
     (Hand 0 = leftmost hand in camera view, Hand 1 = rightmost hand).
4. Explicit missing-hand indicators: Never fabricate coordinates for undetected hands.
5. Invariant to camera positioning and signer location: Does not use raw image coordinates.

Feature Dimension:
- Hand 0 features: 92 dimensions
- Hand 1 features: 92 dimensions
- Inter-hand interaction features: 10 dimensions
Total vector length: 194 float32 values
"""

import numpy as np


FEATURE_DIM_PER_HAND = 95
INTER_HAND_FEATURE_DIM = 10
TOTAL_FEATURE_DIM = FEATURE_DIM_PER_HAND * 2 + INTER_HAND_FEATURE_DIM  # 200

# Landmark Indices conforming to MediaPipe Hand Landmark spec:
# 0: Wrist
# 1-4: Thumb (CMC, MCP, IP, TIP)
# 5-8: Index (MCP, PIP, DIP, TIP)
# 9-12: Middle (MCP, PIP, DIP, TIP)
# 13-16: Ring (MCP, PIP, DIP, TIP)
# 17-20: Pinky (MCP, PIP, DIP, TIP)
WRIST = 0
THUMB_TIP, INDEX_TIP, MID_TIP, RING_TIP, PINKY_TIP = 4, 8, 12, 16, 20
INDEX_MCP, MID_MCP, RING_MCP, PINKY_MCP = 5, 9, 13, 17
INDEX_PIP, MID_PIP, RING_PIP, PINKY_PIP = 6, 10, 14, 18
THUMB_IP = 3


def _safe_norm(v, eps=1e-7):
    return float(np.linalg.norm(v)) + eps


def _cosine_angle(v1, v2, eps=1e-7):
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm < eps:
        return 0.0
    return float(np.clip(dot / norm, -1.0, 1.0))


def extract_single_hand_features(landmarks_21x3, category_name="unknown", score=1.0, present=True):
    """
    Extracts 92 geometric features from a single hand of 21 (x, y, z) landmarks.
    """
    if not present or landmarks_21x3 is None:
        return np.zeros(FEATURE_DIM_PER_HAND, dtype=np.float32)

    pts = np.asarray(landmarks_21x3, dtype=np.float32)
    if pts.shape != (21, 3):
        return np.zeros(FEATURE_DIM_PER_HAND, dtype=np.float32)

    wrist = pts[WRIST]
    mid_mcp = pts[MID_MCP]
    palm_scale = np.linalg.norm(mid_mcp - wrist)
    if palm_scale < 1e-4:
        palm_scale = 1.0

    # 1. Wrist-relative, scale-normalized 3D coordinates (21 x 3 = 63)
    coords_rel = (pts - wrist) / palm_scale
    coords_feat = coords_rel.flatten()  # 63

    # 2. Fingertip-to-wrist normalized distances (5)
    tip_indices = [THUMB_TIP, INDEX_TIP, MID_TIP, RING_TIP, PINKY_TIP]
    d_tip_wrist = [np.linalg.norm(pts[t] - wrist) / palm_scale for t in tip_indices]

    # 3. Fingertip-to-MCP normalized distances (5)
    mcp_indices = [INDEX_MCP, INDEX_MCP, MID_MCP, RING_MCP, PINKY_MCP]
    # For thumb, distance from thumb tip to thumb MCP (index 2)
    d_tip_mcp = [
        np.linalg.norm(pts[THUMB_TIP] - pts[2]) / palm_scale,
        np.linalg.norm(pts[INDEX_TIP] - pts[INDEX_MCP]) / palm_scale,
        np.linalg.norm(pts[MID_TIP] - pts[MID_MCP]) / palm_scale,
        np.linalg.norm(pts[RING_TIP] - pts[RING_MCP]) / palm_scale,
        np.linalg.norm(pts[PINKY_TIP] - pts[PINKY_MCP]) / palm_scale,
    ]

    # 4. Finger extension ratios (tip-to-wrist / mcp-to-wrist) (5)
    ext_ratios = [
        np.linalg.norm(pts[THUMB_TIP] - wrist) / (np.linalg.norm(pts[2] - wrist) + 1e-6),
        np.linalg.norm(pts[INDEX_TIP] - wrist) / (np.linalg.norm(pts[INDEX_MCP] - wrist) + 1e-6),
        np.linalg.norm(pts[MID_TIP] - wrist) / (np.linalg.norm(pts[MID_MCP] - wrist) + 1e-6),
        np.linalg.norm(pts[RING_TIP] - wrist) / (np.linalg.norm(pts[RING_MCP] - wrist) + 1e-6),
        np.linalg.norm(pts[PINKY_TIP] - wrist) / (np.linalg.norm(pts[PINKY_MCP] - wrist) + 1e-6),
    ]

    # 5. Inter-fingertip relative distances (6)
    # Pairs: (Thumb, Index), (Index, Mid), (Mid, Ring), (Ring, Pinky), (Thumb, Pinky), (Index, Pinky)
    inter_tip_dists = [
        np.linalg.norm(pts[THUMB_TIP] - pts[INDEX_TIP]) / palm_scale,
        np.linalg.norm(pts[INDEX_TIP] - pts[MID_TIP]) / palm_scale,
        np.linalg.norm(pts[MID_TIP] - pts[RING_TIP]) / palm_scale,
        np.linalg.norm(pts[RING_TIP] - pts[PINKY_TIP]) / palm_scale,
        np.linalg.norm(pts[THUMB_TIP] - pts[PINKY_TIP]) / palm_scale,
        np.linalg.norm(pts[INDEX_TIP] - pts[PINKY_TIP]) / palm_scale,
    ]

    # 6. Joint flexion angles (cosine at middle joint) (5)
    flex_angles = [
        _cosine_angle(pts[2] - pts[THUMB_IP], pts[THUMB_TIP] - pts[THUMB_IP]),
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
        palm_normal = (norm_vec / norm_len).tolist()
    else:
        palm_normal = [0.0, 0.0, 1.0]

    # 8. Metadata features: presence, confidence, handedness_sign (3)
    h_sign = 1.0 if category_name.lower() == "right" else (-1.0 if category_name.lower() == "left" else 0.0)
    meta = [1.0, float(score), float(h_sign)]

    features = np.concatenate([
        coords_feat,                       # 63
        np.array(d_tip_wrist, dtype=np.float32),      # 5
        np.array(d_tip_mcp, dtype=np.float32),        # 5
        np.array(ext_ratios, dtype=np.float32),       # 5
        np.array(inter_tip_dists, dtype=np.float32),  # 6
        np.array(flex_angles, dtype=np.float32),      # 5
        np.array(palm_normal, dtype=np.float32),      # 3
        np.array(meta, dtype=np.float32)              # 3
    ])  # Total = 92
    return features.astype(np.float32)


def extract_inter_hand_features(pts0, pts1, scale0, scale1, present0, present1):
    """
    Extracts 10 inter-hand interaction features when both hands are present.
    """
    if not (present0 and present1 and pts0 is not None and pts1 is not None):
        return np.zeros(INTER_HAND_FEATURE_DIM, dtype=np.float32)

    mean_scale = (scale0 + scale1) / 2.0
    if mean_scale < 1e-4:
        mean_scale = 1.0

    w0, w1 = pts0[WRIST], pts1[WRIST]

    # 1. Wrist-to-wrist 3D distance (1)
    d_wrist = np.linalg.norm(w0 - w1) / mean_scale

    # 2. Relative horizontal and vertical displacement (2)
    dx_wrist = (w0[0] - w1[0]) / mean_scale
    dy_wrist = (w0[1] - w1[1]) / mean_scale

    # 3. Inter-hand fingertip contact distances (6)
    # Key contact pairs in Indian Sign Language:
    # - Index0 to Index1 (touching index fingers)
    # - Thumb0 to Index1
    # - Index0 to Thumb1
    # - Thumb0 to Thumb1
    # - Wrist0 to Index1
    # - Wrist1 to Index0
    contacts = [
        np.linalg.norm(pts0[INDEX_TIP] - pts1[INDEX_TIP]) / mean_scale,
        np.linalg.norm(pts0[THUMB_TIP] - pts1[INDEX_TIP]) / mean_scale,
        np.linalg.norm(pts0[INDEX_TIP] - pts1[THUMB_TIP]) / mean_scale,
        np.linalg.norm(pts0[THUMB_TIP] - pts1[THUMB_TIP]) / mean_scale,
        np.linalg.norm(pts0[WRIST] - pts1[INDEX_TIP]) / mean_scale,
        np.linalg.norm(pts1[WRIST] - pts0[INDEX_TIP]) / mean_scale,
    ]

    # 4. Both present indicator (1)
    both_present = 1.0

    inter_features = [both_present, d_wrist, dx_wrist, dy_wrist] + contacts
    return np.array(inter_features, dtype=np.float32)


def select_active_hands(parsed_hands, suppress_lap_hands=True):
    """
    Distinguishes active signing hands from resting lap hands.
    Returns: (selected_hands, diagnostics)

    Rules:
    1. Single hand (0 or 1 hand detected): NEVER suppressed. A single low hand
       is a valid signing hand.
    2. Multiple hands (>=2 hands detected):
       - Analyzes vertical position, vertical gap, and inter-hand interaction.
       - A hand is suppressed ONLY if:
         * Its wrist is in the lower frame (wrist_y >= 0.72, min_y >= 0.60)
         * AND an active upper-body signing hand exists (centroid_y <= 0.65)
         * AND the vertical gap between active hand bottom and candidate top is >= 0.10
         * AND the hands are physically decoupled (wrist distance >= 0.35)
       - If hands are interacting or close, both are preserved as a genuine two-hand sign.
    """
    diag = {
        "initial_hand_count": len(parsed_hands),
        "selected_hand_count": len(parsed_hands),
        "lap_hand_suppressed": False,
        "suppression_reasons": [],
        "hand_details": []
    }

    if not parsed_hands:
        return [], diag

    for idx, h in enumerate(parsed_hands):
        pts = h["pts"]
        wy = float(pts[WRIST, 1])
        cy = float(np.mean(pts[:, 1]))
        min_y = float(np.min(pts[:, 1]))
        max_y = float(np.max(pts[:, 1]))
        h["wrist_y"] = wy
        h["centroid_y"] = cy
        h["min_y"] = min_y
        h["max_y"] = max_y
        diag["hand_details"].append({
            "index": idx,
            "wrist_y": round(wy, 3),
            "centroid_y": round(cy, 3),
            "min_y": round(min_y, 3),
            "max_y": round(max_y, 3),
            "score": round(float(h.get("score", 1.0)), 3)
        })

    # Rule 1: Single hand -> never suppress
    if len(parsed_hands) == 1 or not suppress_lap_hands:
        diag["suppression_reasons"].append("single_hand_or_suppression_disabled")
        return list(parsed_hands), diag

    # Rule 2: Multiple hands -> sort by centroid_y ascending (highest hand first)
    by_height = sorted(parsed_hands, key=lambda h: h["centroid_y"])
    active_hand = by_height[0]
    selected = [active_hand]

    for candidate in by_height[1:]:
        wy_c = candidate["wrist_y"]
        min_y_c = candidate["min_y"]
        cy_act = active_hand["centroid_y"]
        max_y_act = active_hand["max_y"]

        vertical_gap = min_y_c - max_y_act
        c_diff = candidate["centroid_y"] - cy_act
        w_dist = float(np.linalg.norm(active_hand["pts"][WRIST] - candidate["pts"][WRIST]))
        tip_dist = float(np.linalg.norm(active_hand["pts"][INDEX_TIP] - candidate["pts"][INDEX_TIP]))

        has_active_upper = (cy_act <= 0.65)

        # Standard lap suppression: candidate wrist in lap area, spatial gap from active hand, wide separation
        is_std_lap = (
            wy_c >= 0.75 and
            min_y_c >= 0.60 and
            (vertical_gap >= 0.10 or c_diff >= 0.25) and
            w_dist >= 0.35 and
            tip_dist >= 0.20
        )

        # Deep lap decoupling: candidate wrist is deep in frame, vertical gap exists, and fingertips decoupled
        is_deep_lap = (
            wy_c >= 0.78 and
            min_y_c >= 0.65 and
            vertical_gap >= 0.08 and
            w_dist >= 0.30 and
            tip_dist >= 0.20
        )

        if has_active_upper and (is_std_lap or is_deep_lap):
            mode_desc = "deep_lap" if is_deep_lap and not is_std_lap else "standard"
            reason = (f"lap_hand_suppressed ({mode_desc}): wrist_y={wy_c:.2f}, min_y={min_y_c:.2f}, "
                      f"active_y={cy_act:.2f}<=0.65, gap={vertical_gap:.2f}, wrist_dist={w_dist:.2f}>=0.30")
            diag["lap_hand_suppressed"] = True
            diag["suppression_reasons"].append(reason)
        else:
            selected.append(candidate)
            diag["suppression_reasons"].append(
                f"retained: gap={vertical_gap:.2f}, wrist_dist={w_dist:.2f}"
            )

    diag["selected_hand_count"] = len(selected)
    return selected, diag


def extract_landmarks_from_manifest_item(item, suppress_lap_hands=True, return_diagnostics=False):
    """
    Extracts the full 200-dim feature vector from a split manifest dictionary item.
    """
    lms_list = item.get("landmarks", [])
    num_hands = len(lms_list)

    empty_diag = {
        "initial_hand_count": 0,
        "selected_hand_count": 0,
        "lap_hand_suppressed": False,
        "suppression_reasons": ["no_landmarks"],
        "hand_details": []
    }

    if num_hands == 0:
        zero_feat = np.zeros(TOTAL_FEATURE_DIM, dtype=np.float32)
        return (zero_feat, empty_diag) if return_diagnostics else zero_feat

    # Parse hands
    parsed_hands = []
    for h in lms_list:
        raw_pts = h.get("landmarks", [])
        if len(raw_pts) != 21:
            continue
        arr = np.array([[p["x"], p["y"], p["z"]] for p in raw_pts], dtype=np.float32)
        cat = h.get("category_name", "unknown")
        score = h.get("score", 1.0)
        wrist_x = arr[WRIST, 0]
        scale = np.linalg.norm(arr[MID_MCP] - arr[WRIST])
        if scale < 1e-4:
            scale = 1.0
        parsed_hands.append({
            "pts": arr,
            "category": cat,
            "score": score,
            "wrist_x": wrist_x,
            "scale": scale
        })

    if len(parsed_hands) == 0:
        zero_feat = np.zeros(TOTAL_FEATURE_DIM, dtype=np.float32)
        return (zero_feat, empty_diag) if return_diagnostics else zero_feat

    # Apply lap hand suppression
    active_hands, diag = select_active_hands(parsed_hands, suppress_lap_hands=suppress_lap_hands)

    if len(active_hands) == 1:
        h0 = active_hands[0]
        feat0 = extract_single_hand_features(h0["pts"], h0["category"], h0["score"], present=True)
        feat1 = np.zeros(FEATURE_DIM_PER_HAND, dtype=np.float32)
        inter = np.zeros(INTER_HAND_FEATURE_DIM, dtype=np.float32)
        feat = np.concatenate([feat0, feat1, inter]).astype(np.float32)
        return (feat, diag) if return_diagnostics else feat

    # 2 or more hands -> sort by wrist x-coordinate (left-to-right in image)
    active_hands.sort(key=lambda h: h["wrist_x"])
    h0 = active_hands[0]
    h1 = active_hands[1]

    feat0 = extract_single_hand_features(h0["pts"], h0["category"], h0["score"], present=True)
    feat1 = extract_single_hand_features(h1["pts"], h1["category"], h1["score"], present=True)
    inter = extract_inter_hand_features(h0["pts"], h1["pts"], h0["scale"], h1["scale"], True, True)

    feat = np.concatenate([feat0, feat1, inter]).astype(np.float32)
    return (feat, diag) if return_diagnostics else feat


def extract_landmarks_from_mediapipe_result(detection_result, suppress_lap_hands=True, return_diagnostics=False):
    """
    Extracts the full 200-dim feature vector directly from a live MediaPipe HandLandmarker result.
    detection_result has .hand_landmarks and .handedness attributes.
    """
    empty_diag = {
        "initial_hand_count": 0,
        "selected_hand_count": 0,
        "lap_hand_suppressed": False,
        "suppression_reasons": ["no_detection_result"],
        "hand_details": []
    }

    if detection_result is None or not detection_result.hand_landmarks:
        zero_feat = np.zeros(TOTAL_FEATURE_DIM, dtype=np.float32)
        return (zero_feat, empty_diag) if return_diagnostics else zero_feat

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
        scale = np.linalg.norm(arr[MID_MCP] - arr[WRIST])
        if scale < 1e-4:
            scale = 1.0
        parsed_hands.append({
            "pts": arr,
            "category": cat,
            "score": score,
            "wrist_x": wrist_x,
            "scale": scale
        })

    if len(parsed_hands) == 0:
        zero_feat = np.zeros(TOTAL_FEATURE_DIM, dtype=np.float32)
        return (zero_feat, empty_diag) if return_diagnostics else zero_feat

    # Apply lap hand suppression
    active_hands, diag = select_active_hands(parsed_hands, suppress_lap_hands=suppress_lap_hands)

    if len(active_hands) == 1:
        h0 = active_hands[0]
        feat0 = extract_single_hand_features(h0["pts"], h0["category"], h0["score"], present=True)
        feat1 = np.zeros(FEATURE_DIM_PER_HAND, dtype=np.float32)
        inter = np.zeros(INTER_HAND_FEATURE_DIM, dtype=np.float32)
        feat = np.concatenate([feat0, feat1, inter]).astype(np.float32)
        return (feat, diag) if return_diagnostics else feat

    # 2 hands -> sort left-to-right
    active_hands.sort(key=lambda h: h["wrist_x"])
    h0 = active_hands[0]
    h1 = active_hands[1]

    feat0 = extract_single_hand_features(h0["pts"], h0["category"], h0["score"], present=True)
    feat1 = extract_single_hand_features(h1["pts"], h1["category"], h1["score"], present=True)
    inter = extract_inter_hand_features(h0["pts"], h1["pts"], h0["scale"], h1["scale"], True, True)

    feat = np.concatenate([feat0, feat1, inter]).astype(np.float32)
    return (feat, diag) if return_diagnostics else feat

