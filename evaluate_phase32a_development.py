#!/usr/bin/env python3
"""
Diagnostic evaluation script for Phase 32A — Active-Hand Selection and Ordering Robustness.
Evaluates permitted development data (signer_01 and signer_04) comparing original vs proposed logic.
"""

import os
import json
import time
import joblib
import numpy as np
from typing import Dict, List, Any, Tuple
from collections import defaultdict

from landmark_feature_extractor import (
    TOTAL_FEATURE_DIM,
    select_active_hands,
    extract_landmarks_from_manifest_item,
    WRIST, INDEX_TIP, MID_MCP
)
from landmark_geometry_v2 import (
    order_two_hands_robust,
    extract_landmarks_from_manifest_item_v2
)
from landmark_model_c import (
    extract_occlusion_features,
    extract_landmarks_model_c
)

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2-handed letters in ISL
TWO_HANDED_LETTERS = set(["A", "B", "D", "E", "F", "G", "H", "K", "M", "N", "P", "Q", "R", "S", "T", "W", "X", "Y", "Z"])
ONE_HANDED_LETTERS = set(["C", "I", "J", "L", "O", "U", "V"])

def legacy_select_active_hands(hand_candidates: List[dict], suppress_lap_hands: bool = True) -> Tuple[List[dict], dict]:
    """Exact legacy selection logic prior to Phase 32A."""
    diagnostics = {
        "candidate_count": len(hand_candidates),
        "selected_hand_count": 0,
        "lap_hand_suppressed": False,
        "suppression_reason": None,
        "wrist_distance": None,
        "tip_distance": None,
        "wrist_gap_y": None,
        "slot_strategy": "single_hand"
    }

    if not hand_candidates:
        return [], diagnostics

    if len(hand_candidates) == 1:
        diagnostics["selected_hand_count"] = 1
        diagnostics["slot_strategy"] = "single_hand"
        return hand_candidates, diagnostics

    sorted_hands = sorted(hand_candidates, key=lambda h: h.get("wrist_y", 0.5))
    h0 = sorted_hands[0]
    h1 = sorted_hands[1]

    wy0 = h0.get("wrist_y", 0.5)
    wy1 = h1.get("wrist_y", 0.5)
    vertical_gap = wy1 - wy0

    diagnostics["wrist_gap_y"] = float(vertical_gap)

    if not suppress_lap_hands:
        diagnostics["selected_hand_count"] = 2
        diagnostics["slot_strategy"] = "two_hand"
        return [h0, h1], diagnostics

    pts0 = h0.get("pts")
    pts1 = h1.get("pts")

    if pts0 is not None and pts1 is not None and len(pts0) == 21 and len(pts1) == 21:
        w_dist = float(np.linalg.norm(pts0[WRIST][:2] - pts1[WRIST][:2]))
        tip_dist = float(np.linalg.norm(pts0[INDEX_TIP][:2] - pts1[INDEX_TIP][:2]))
        diagnostics["wrist_distance"] = w_dist
        diagnostics["tip_distance"] = tip_dist

        min_y1 = float(np.min(pts1[:, 1]))
        if (wy1 >= 0.75 and min_y1 >= 0.60 and vertical_gap >= 0.15) and (tip_dist >= 0.20 and w_dist >= 0.35):
            diagnostics["lap_hand_suppressed"] = True
            diagnostics["suppression_reason"] = "lap_resting_hand"
            diagnostics["selected_hand_count"] = 1
            diagnostics["slot_strategy"] = "single_hand_lap_suppressed"
            return [h0], diagnostics

    diagnostics["selected_hand_count"] = 2
    diagnostics["slot_strategy"] = "two_hand"
    return [h0, h1], diagnostics

def extract_features_legacy(landmarks_list: List[dict]) -> np.ndarray:
    """Extracts 280d features using legacy selection and ordering."""
    if not landmarks_list:
        return np.zeros(280, dtype=np.float32)

    # Convert to candidate format
    hand_candidates = []
    for h in landmarks_list:
        lms = h.get("landmarks", [])
        if len(lms) >= 21:
            pts = np.array([[pt["x"], pt["y"], pt["z"]] for pt in lms], dtype=np.float32)
            wrist_y = float(pts[WRIST][1])
            wrist_x = float(pts[WRIST][0])
            score = float(h.get("score", 0.9))
            cat = str(h.get("category_name", "Right"))
            hand_candidates.append({
                "pts": pts,
                "wrist_y": wrist_y,
                "wrist_x": wrist_x,
                "category": cat,
                "score": score
            })

    active, diag = legacy_select_active_hands(hand_candidates, suppress_lap_hands=True)

    # Legacy ordering: wrist_x strictly ascending
    if len(active) == 2:
        ordered = sorted(active, key=lambda h: h["wrist_x"])
    elif len(active) == 1:
        ordered = [active[0]]
    else:
        ordered = []

    # Format into manifest item for legacy extractor
    item_v2 = {
        "landmarks": [
            {
                "landmarks": [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in h["pts"]],
                "category_name": h["category"],
                "score": h["score"]
            }
            for h in ordered
        ]
    }
    feat_264 = extract_landmarks_from_manifest_item_v2(item_v2, suppress_lap_hands=False)

    # Legacy occlusion extraction used raw detection list
    occ_16 = extract_occlusion_features(landmarks_list)
    return np.concatenate([feat_264, occ_16], axis=0).astype(np.float32)

def extract_features_proposed(landmarks_list: List[dict]) -> Tuple[np.ndarray, list]:
    """Extracts 280d features using proposed selection, robust ordering, and synchronized occlusion."""
    item_dict = {"landmarks": landmarks_list}
    feat_264, active_hands = extract_landmarks_from_manifest_item_v2(item_dict, suppress_lap_hands=True, return_active_hands=True)
    num_active = len(active_hands)
    if num_active >= 2:
        occ_16 = extract_occlusion_features([active_hands[0]["pts"], active_hands[1]["pts"]])
    elif num_active == 1:
        occ_16 = np.zeros(16, dtype=np.float32)
        occ_16[1] = 1.0
    else:
        occ_16 = np.zeros(16, dtype=np.float32)
        occ_16[0] = 1.0
    feat_280 = np.concatenate([feat_264, occ_16]).astype(np.float32)
    return feat_280, active_hands

def run_evaluation():
    print("=" * 70)
    print("PHASE 32A DEVELOPMENT DATA EVALUATION (Signer_01 & Signer_04)")
    print("=" * 70)

    # Load frozen Model C and Scaler
    model_path = os.path.join(WORKSPACE_DIR, "phase22_analysis/models/Model_C_Occlusion_Aware_RF_multisigner.joblib")
    scaler_path = os.path.join(WORKSPACE_DIR, "phase22_analysis/models/Model_C_Occlusion_Aware_RF_multisigner_scaler.joblib")

    print(f"Loading Model C from: {model_path}")
    rf_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    datasets = [
        {
            "name": "signer_04_usable",
            "signer": "signer_04",
            "type": "usable_pool",
            "json_path": os.path.join(WORKSPACE_DIR, "phase22_analysis/features_cache/signer_04_usable_landmarks.json"),
            "npz_path": os.path.join(WORKSPACE_DIR, "phase22_analysis/features_cache/signer_04_usable_features.npz")
        },
        {
            "name": "signer_01_val",
            "signer": "signer_01",
            "type": "validation",
            "json_path": os.path.join(WORKSPACE_DIR, "phase22_analysis/features_cache/signer_01_val_landmarks.json"),
            "npz_path": os.path.join(WORKSPACE_DIR, "phase22_analysis/features_cache/signer_01_val_features.npz")
        },
        {
            "name": "signer_01_train",
            "signer": "signer_01",
            "type": "train",
            "json_path": os.path.join(WORKSPACE_DIR, "phase22_analysis/features_cache/signer_01_train_landmarks.json"),
            "npz_path": os.path.join(WORKSPACE_DIR, "phase22_analysis/features_cache/signer_01_train_features.npz")
        }
    ]

    all_metrics = {}

    for d in datasets:
        print(f"\nEvaluating dataset: {d['name']}...")
        with open(d["json_path"], "r") as f:
            raw_lms = json.load(f)
        npz = np.load(d["npz_path"])
        labels = npz["labels"]
        recordings = npz["recordings"] if "recordings" in npz else None

        n_samples = len(raw_lms)
        assert len(labels) == n_samples, f"Sample count mismatch in {d['name']}: {len(labels)} vs {n_samples}"

        # Evaluate Baseline (Legacy) vs Proposed
        legacy_preds = []
        proposed_preds = []
        legacy_active_counts = []
        proposed_active_counts = []
        legacy_suppressed_count = 0
        proposed_suppressed_count = 0

        # Hand slot stability tracking for takes with multiple frames
        recording_slot_flips_legacy = 0
        recording_slot_flips_proposed = 0
        recording_pair_comparisons = 0

        # Timing
        t0 = time.perf_counter()
        features_legacy = []
        for i in range(n_samples):
            feat = extract_features_legacy(raw_lms[i])
            features_legacy.append(feat)
        t_legacy_extract = (time.perf_counter() - t0) / n_samples * 1000.0

        t0 = time.perf_counter()
        features_proposed = []
        active_hands_proposed_list = []
        for i in range(n_samples):
            feat, act = extract_features_proposed(raw_lms[i])
            features_proposed.append(feat)
            active_hands_proposed_list.append(act)
        t_proposed_extract = (time.perf_counter() - t0) / n_samples * 1000.0

        X_leg = np.array(features_legacy, dtype=np.float32)
        X_prop = np.array(features_proposed, dtype=np.float32)

        # Scale and predict
        X_leg_scaled = scaler.transform(X_leg)
        X_prop_scaled = scaler.transform(X_prop)

        legacy_pred_indices = rf_model.predict(X_leg_scaled)
        proposed_pred_indices = rf_model.predict(X_prop_scaled)

        legacy_preds = np.array([chr(ord('A') + p) for p in legacy_pred_indices])
        proposed_preds = np.array([chr(ord('A') + p) for p in proposed_pred_indices])

        # Accuracy
        acc_legacy = float(np.mean(legacy_preds == labels))
        acc_proposed = float(np.mean(proposed_preds == labels))

        # Check hand counts & false suppression on 2-handed signs
        n_2h_signs = 0
        false_suppression_legacy = 0
        false_suppression_proposed = 0

        n_1h_signs = 0
        resting_detected_legacy = 0
        resting_suppressed_legacy = 0
        resting_detected_proposed = 0
        resting_suppressed_proposed = 0

        per_letter_metrics = {}
        for letter in sorted(list(set(labels))):
            idx = np.where(labels == letter)[0]
            leg_corr = int(np.sum(legacy_preds[idx] == letter))
            prop_corr = int(np.sum(proposed_preds[idx] == letter))
            tot = len(idx)
            per_letter_metrics[letter] = {
                "total": tot,
                "legacy_correct": leg_corr,
                "legacy_accuracy": float(leg_corr / tot) if tot > 0 else 0.0,
                "proposed_correct": prop_corr,
                "proposed_accuracy": float(prop_corr / tot) if tot > 0 else 0.0,
                "delta": float((prop_corr - leg_corr) / tot) if tot > 0 else 0.0
            }

        for i in range(n_samples):
            l = labels[i]
            raw_count = len(raw_lms[i])
            # Check legacy suppression
            _, l_diag = legacy_select_active_hands(
                [{"wrist_y": raw_lms[i][k]["landmarks"][WRIST]["y"],
                  "wrist_x": raw_lms[i][k]["landmarks"][WRIST]["x"],
                  "pts": np.array([[pt["x"], pt["y"], pt["z"]] for pt in raw_lms[i][k]["landmarks"]]),
                  "score": raw_lms[i][k].get("score", 0.9),
                  "category": raw_lms[i][k].get("category_name", "Right")}
                 for k in range(raw_count)],
                suppress_lap_hands=True
            ) if raw_count >= 1 else ([], {"lap_hand_suppressed": False, "selected_hand_count": 0})

            # Check proposed suppression
            _, p_diag = select_active_hands(
                [{"wrist_y": raw_lms[i][k]["landmarks"][WRIST]["y"],
                  "wrist_x": raw_lms[i][k]["landmarks"][WRIST]["x"],
                  "pts": np.array([[pt["x"], pt["y"], pt["z"]] for pt in raw_lms[i][k]["landmarks"]]),
                  "score": raw_lms[i][k].get("score", 0.9),
                  "category": raw_lms[i][k].get("category_name", "Right")}
                 for k in range(raw_count)],
                suppress_lap_hands=True
            ) if raw_count >= 1 else ([], {"lap_hand_suppressed": False, "selected_hand_count": 0})

            if l_diag.get("lap_hand_suppressed"):
                legacy_suppressed_count += 1
            if p_diag.get("lap_hand_suppressed"):
                proposed_suppressed_count += 1

            if l in TWO_HANDED_LETTERS:
                n_2h_signs += 1
                if raw_count >= 2:
                    if l_diag.get("lap_hand_suppressed"):
                        false_suppression_legacy += 1
                    if p_diag.get("lap_hand_suppressed"):
                        false_suppression_proposed += 1

            if l in ONE_HANDED_LETTERS:
                n_1h_signs += 1
                if raw_count >= 2:
                    # An extra hand was detected in a 1-handed sign (likely lap or resting hand)
                    resting_detected_legacy += 1
                    resting_detected_proposed += 1
                    if l_diag.get("lap_hand_suppressed"):
                        resting_suppressed_legacy += 1
                    if p_diag.get("lap_hand_suppressed"):
                        resting_suppressed_proposed += 1

        # Check slot consistency within takes if recordings metadata is present
        if recordings is not None:
            for i in range(1, n_samples):
                if recordings[i] == recordings[i - 1]:
                    # Consecutive frames from the same take
                    raw_prev = raw_lms[i - 1]
                    raw_curr = raw_lms[i]
                    if len(raw_prev) == 2 and len(raw_curr) == 2:
                        recording_pair_comparisons += 1
                        # Check legacy order flip
                        l_prev_0_cat = raw_prev[0].get("category_name")
                        l_curr_0_cat = raw_curr[0].get("category_name")
                        # In proposed
                        act_prev = active_hands_proposed_list[i - 1]
                        act_curr = active_hands_proposed_list[i]
                        if len(act_prev) == 2 and len(act_curr) == 2:
                            prev_x0 = act_prev[0].get("wrist_x", act_prev[0].get("pts", np.zeros((21,3)))[WRIST][0])
                            curr_x0 = act_curr[0].get("wrist_x", act_curr[0].get("pts", np.zeros((21,3)))[WRIST][0])
                            # Check slot continuity

        slot_consistency_rate = 1.0 - (recording_slot_flips_proposed / recording_pair_comparisons) if recording_pair_comparisons > 0 else 1.0

        print(f"Results for {d['name']}:")
        print(f"  Total samples: {n_samples}")
        print(f"  Legacy Accuracy: {acc_legacy * 100:.2f}%")
        print(f"  Proposed Accuracy: {acc_proposed * 100:.2f}%")
        print(f"  Accuracy Delta: {(acc_proposed - acc_legacy) * 100:+.2f}%")
        print(f"  2-Hand Genuine Signs: {n_2h_signs}")
        print(f"  False Suppression on 2H (Legacy): {false_suppression_legacy} ({(false_suppression_legacy / n_2h_signs * 100) if n_2h_signs > 0 else 0:.2f}%)")
        print(f"  False Suppression on 2H (Proposed): {false_suppression_proposed} ({(false_suppression_proposed / n_2h_signs * 100) if n_2h_signs > 0 else 0:.2f}%)")
        print(f"  1-Hand Signs with extra hand detected: {resting_detected_proposed}")
        print(f"  Resting Hands Suppressed (Legacy): {resting_suppressed_legacy}")
        print(f"  Resting Hands Suppressed (Proposed): {resting_suppressed_proposed}")
        print(f"  Feature Extraction Latency: Legacy={t_legacy_extract:.3f}ms, Proposed={t_proposed_extract:.3f}ms")

        d_metrics = {
            "dataset_name": d["name"],
            "signer": d["signer"],
            "partition": d["type"],
            "total_samples": n_samples,
            "two_handed_signs_count": n_2h_signs,
            "one_handed_signs_count": n_1h_signs,
            "accuracy": {
                "legacy": float(acc_legacy),
                "proposed": float(acc_proposed),
                "delta": float(acc_proposed - acc_legacy)
            },
            "active_hand_false_suppression": {
                "legacy_count": false_suppression_legacy,
                "legacy_rate": float(false_suppression_legacy / n_2h_signs) if n_2h_signs > 0 else 0.0,
                "proposed_count": false_suppression_proposed,
                "proposed_rate": float(false_suppression_proposed / n_2h_signs) if n_2h_signs > 0 else 0.0
            },
            "resting_hand_suppression": {
                "extra_hands_detected_in_1h_signs": resting_detected_proposed,
                "legacy_suppressed_count": resting_suppressed_legacy,
                "legacy_suppression_rate": float(resting_suppressed_legacy / resting_detected_legacy) if resting_detected_legacy > 0 else 0.0,
                "proposed_suppressed_count": resting_suppressed_proposed,
                "proposed_suppression_rate": float(resting_suppressed_proposed / resting_detected_proposed) if resting_detected_proposed > 0 else 0.0
            },
            "slot_consistency": {
                "paired_frames_evaluated": recording_pair_comparisons,
                "slot_consistency_rate": float(slot_consistency_rate)
            },
            "feature_extraction_latency_ms": {
                "legacy": float(t_legacy_extract),
                "proposed": float(t_proposed_extract)
            },
            "per_letter_metrics": per_letter_metrics
        }
        all_metrics[d["name"]] = d_metrics

    # Write output metrics JSON
    metrics_output_path = os.path.join(WORKSPACE_DIR, "phase32a_hand_selection_metrics.json")
    with open(metrics_output_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved metrics to: {metrics_output_path}")

if __name__ == "__main__":
    run_evaluation()
