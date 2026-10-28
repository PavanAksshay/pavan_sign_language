#!/usr/bin/env python3
"""
Phase 22 — Feature Extraction and Caching Pipeline
Extracts and caches:
1. Legacy signer_01 features (v1 200-dim, v2 264-dim) from train_manifest.json and val_manifest.json.
2. Signer 04 features (v1 200-dim, v2 264-dim, and raw 3D landmarks) for all 1,621 usable frames.
3. Saves caches to phase22_analysis/features_cache/.
"""

import os
import sys
import json
import time
import numpy as np
import cv2
from typing import Dict, Any, List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from landmark_feature_extractor import extract_landmarks_from_mediapipe_result, extract_landmarks_from_manifest_item
from landmark_geometry_v2 import extract_landmarks_from_mediapipe_result_v2, extract_landmarks_from_manifest_item_v2

CACHE_DIR = os.path.join(BASE_DIR, "phase22_analysis", "features_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def extract_and_cache_all():
    print("=== PHASE 22: EXTRACTING & CACHING MULTI-SIGNER FEATURES ===")

    # 1. Process Legacy signer_01 Train Manifest
    train_manifest_p = os.path.join(BASE_DIR, "models", "mediapipe_squeezenet_v1", "splits", "train_manifest.json")
    print(f"\n1. Processing legacy signer_01 train manifest ({train_manifest_p})...")
    with open(train_manifest_p, "r", encoding="utf-8") as f:
        s1_train_data = json.load(f)

    s1_train_v1 = []
    s1_train_v2 = []
    s1_train_labels = []
    s1_train_ids = []
    s1_train_raw_lms = []

    for item in s1_train_data:
        f1 = extract_landmarks_from_manifest_item(item)
        f2 = extract_landmarks_from_manifest_item_v2(item)
        s1_train_v1.append(f1)
        s1_train_v2.append(f2)
        s1_train_labels.append(item["class_label"])
        s1_train_ids.append(item["file_id"])
        s1_train_raw_lms.append(item.get("landmarks", []))

    s1_train_v1 = np.array(s1_train_v1, dtype=np.float32)
    s1_train_v2 = np.array(s1_train_v2, dtype=np.float32)
    print(f"  signer_01 train: v1 shape={s1_train_v1.shape}, v2 shape={s1_train_v2.shape}")

    np.savez_compressed(
        os.path.join(CACHE_DIR, "signer_01_train_features.npz"),
        features_v1=s1_train_v1,
        features_v2=s1_train_v2,
        labels=np.array(s1_train_labels),
        sample_ids=np.array(s1_train_ids)
    )
    with open(os.path.join(CACHE_DIR, "signer_01_train_landmarks.json"), "w") as f:
        json.dump(s1_train_raw_lms, f)

    # 2. Process Legacy signer_01 Val Manifest
    val_manifest_p = os.path.join(BASE_DIR, "models", "mediapipe_squeezenet_v1", "splits", "val_manifest.json")
    print(f"\n2. Processing legacy signer_01 val manifest ({val_manifest_p})...")
    with open(val_manifest_p, "r", encoding="utf-8") as f:
        s1_val_data = json.load(f)

    s1_val_v1 = []
    s1_val_v2 = []
    s1_val_labels = []
    s1_val_ids = []
    s1_val_raw_lms = []

    for item in s1_val_data:
        f1 = extract_landmarks_from_manifest_item(item)
        f2 = extract_landmarks_from_manifest_item_v2(item)
        s1_val_v1.append(f1)
        s1_val_v2.append(f2)
        s1_val_labels.append(item["class_label"])
        s1_val_ids.append(item["file_id"])
        s1_val_raw_lms.append(item.get("landmarks", []))

    s1_val_v1 = np.array(s1_val_v1, dtype=np.float32)
    s1_val_v2 = np.array(s1_val_v2, dtype=np.float32)
    print(f"  signer_01 val: v1 shape={s1_val_v1.shape}, v2 shape={s1_val_v2.shape}")

    np.savez_compressed(
        os.path.join(CACHE_DIR, "signer_01_val_features.npz"),
        features_v1=s1_val_v1,
        features_v2=s1_val_v2,
        labels=np.array(s1_val_labels),
        sample_ids=np.array(s1_val_ids)
    )
    with open(os.path.join(CACHE_DIR, "signer_01_val_landmarks.json"), "w") as f:
        json.dump(s1_val_raw_lms, f)

    # 3. Process Usable signer_04 Frames
    manifest_p = os.path.join(BASE_DIR, "data", "collection_manifests", "test_multisigner_manifest.json")
    print(f"\n3. Processing usable signer_04 frames from manifest ({manifest_p})...")
    with open(manifest_p, "r", encoding="utf-8") as f:
        full_manifest = json.load(f)

    s4_usable = [s for s in full_manifest if s.get("signer_id") == "signer_04" and s.get("quality", {}).get("is_usable") is True]
    print(f"  Discovered {len(s4_usable)} usable signer_04 frames.")

    task_path = os.path.join(BASE_DIR, "App", "include50_model", "hand_landmarker.task")
    opt = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=task_path),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4
    )
    landmarker = vision.HandLandmarker.create_from_options(opt)

    s4_v1 = []
    s4_v2 = []
    s4_labels = []
    s4_sample_ids = []
    s4_conditions = []
    s4_recordings = []
    s4_frame_paths = []
    s4_raw_lms = []

    t0 = time.time()
    for idx, s in enumerate(s4_usable):
        rel_p = s["source"]["frame_path"]
        abs_p = os.path.join(BASE_DIR, rel_p)
        img = cv2.imread(abs_p)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = landmarker.detect(mp_img)

        f1 = extract_landmarks_from_mediapipe_result(res)
        f2 = extract_landmarks_from_mediapipe_result_v2(res)

        # Serialize landmarks for potential future use
        serialized_hands = []
        if res.hand_landmarks:
            for h_idx, h_pts in enumerate(res.hand_landmarks):
                cat = res.handedness[h_idx][0].category_name if res.handedness and len(res.handedness) > h_idx else "unknown"
                score = float(res.handedness[h_idx][0].score) if res.handedness and len(res.handedness) > h_idx else 1.0
                pts_list = [{"id": i, "x": float(p.x), "y": float(p.y), "z": float(p.z)} for i, p in enumerate(h_pts)]
                serialized_hands.append({
                    "hand_index": h_idx,
                    "category_name": cat,
                    "score": score,
                    "landmarks": pts_list
                })

        cond = "condition_1" if "condition_1" in rel_p else "condition_2"
        s4_v1.append(f1)
        s4_v2.append(f2)
        s4_labels.append(s["letter"])
        s4_sample_ids.append(s["sample_id"])
        s4_conditions.append(cond)
        s4_recordings.append(s["recording_id"])
        s4_frame_paths.append(rel_p)
        s4_raw_lms.append(serialized_hands)

        if (idx + 1) % 200 == 0 or (idx + 1) == len(s4_usable):
            elapsed = time.time() - t0
            print(f"    Extracted {idx + 1} / {len(s4_usable)} frames ({elapsed:.1f}s, {elapsed / (idx + 1) * 1000:.1f} ms/frame)")

    s4_v1 = np.array(s4_v1, dtype=np.float32)
    s4_v2 = np.array(s4_v2, dtype=np.float32)
    print(f"  signer_04: v1 shape={s4_v1.shape}, v2 shape={s4_v2.shape}")

    np.savez_compressed(
        os.path.join(CACHE_DIR, "signer_04_usable_features.npz"),
        features_v1=s4_v1,
        features_v2=s4_v2,
        labels=np.array(s4_labels),
        sample_ids=np.array(s4_sample_ids),
        conditions=np.array(s4_conditions),
        recordings=np.array(s4_recordings),
        frame_paths=np.array(s4_frame_paths)
    )
    with open(os.path.join(CACHE_DIR, "signer_04_usable_landmarks.json"), "w") as f:
        json.dump(s4_raw_lms, f)

    print("\nAll multi-signer features successfully extracted and cached to phase22_analysis/features_cache/!")


if __name__ == "__main__":
    extract_and_cache_all()
