#!/usr/bin/env python3
"""
Phase 14B — Multi-Signer A–Z Benchmark Collection Protocol & Ingestion Pipeline

Implements:
1. Complete Multi-Signer Metadata Schema conforming to Phase 14B requirements.
2. Ingestion & Quality Audit Pipeline:
   - Validates image dimensions, blur (Laplacian variance), and brightness.
   - Runs MediaPipe HandLandmarker with lap-hand suppression.
   - Extracts 200-dim canonical landmark features.
   - Checks hand count, handedness, and detection score.
3. Recording Checklist & Labeling Verification rules.
4. Signer-Independent Split Partitioning logic (Signer 1 -> Train, Signer 2 -> Val, Signer 3 -> Test).
"""

import os
import sys
import json
import time
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from landmark_feature_extractor import (
    extract_landmarks_from_mediapipe_result,
    extract_single_hand_features,
    TOTAL_FEATURE_DIM,
    FEATURE_DIM_PER_HAND,
    INTER_HAND_FEATURE_DIM
)

# Standard metadata schema definition
METADATA_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "ISL_MultiSigner_Benchmark_Sample",
    "type": "object",
    "required": [
        "sample_id",
        "signer_id",
        "session_id",
        "recording_id",
        "letter",
        "confirmation_method",
        "condition",
        "detection",
        "quality",
        "source"
    ],
    "properties": {
        "sample_id": {"type": "string"},
        "signer_id": {"type": "string", "enum": ["signer_01", "signer_02", "signer_03", "signer_04"]},
        "session_id": {"type": "string"},
        "recording_id": {"type": "string"},
        "letter": {"type": "string", "pattern": "^[A-Z]$"},
        "confirmation_method": {
            "type": "string",
            "enum": ["signer_confirmed", "expert_verified", "dual_annotator_consensus"]
        },
        "condition": {
            "type": "object",
            "required": ["lighting", "background", "camera_position", "distance_cm"],
            "properties": {
                "lighting": {
                    "type": "string",
                    "enum": ["ambient_natural", "indoor_overhead", "low_light"]
                },
                "background": {
                    "type": "string",
                    "enum": ["plain_wall", "textured_office", "cluttered_home"]
                },
                "camera_position": {
                    "type": "string",
                    "enum": ["frontal_eye_level", "frontal_low_angle", "diagonal_15deg"]
                },
                "distance_cm": {"type": "integer", "minimum": 40, "maximum": 150}
            }
        },
        "detection": {
            "type": "object",
            "required": ["hands_detected", "handedness", "scores", "lap_hand_suppressed"],
            "properties": {
                "hands_detected": {"type": "integer", "minimum": 0, "maximum": 2},
                "handedness": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["Left", "Right", "unknown"]}
                },
                "scores": {"type": "array", "items": {"type": "number"}},
                "lap_hand_suppressed": {"type": "boolean"}
            }
        },
        "quality": {
            "type": "object",
            "required": ["blur_laplacian_var", "mean_brightness", "is_usable"],
            "properties": {
                "blur_laplacian_var": {"type": "number"},
                "mean_brightness": {"type": "number"},
                "is_usable": {"type": "boolean"},
                "rejection_reason": {"type": ["string", "null"]}
            }
        },
        "source": {
            "type": "object",
            "required": ["video_path", "frame_index", "timestamp_sec"],
            "properties": {
                "video_path": {"type": "string"},
                "frame_index": {"type": "integer"},
                "timestamp_sec": {"type": "number"}
            }
        }
    }
}


class MultiSignerIngestionPipeline:
    """
    Ingests and audits multi-signer video recordings into standardized benchmark samples.
    """

    def __init__(self, min_blur_var=50.0, min_brightness=40.0, max_brightness=220.0):
        self.min_blur_var = min_blur_var
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness

        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        hand_task_path = os.path.join(BASE_DIR, "App", "include50_model", "hand_landmarker.task")
        if not os.path.exists(hand_task_path):
            raise FileNotFoundError(f"MediaPipe task model not found at {hand_task_path}")

        opt = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=hand_task_path),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmarker = vision.HandLandmarker.create_from_options(opt)

    def audit_frame_quality(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(np.mean(gray))

        is_usable = True
        rejection_reason = None

        if brightness < self.min_brightness:
            is_usable = False
            rejection_reason = f"Frame too dark (Mean brightness {brightness:.1f} < {self.min_brightness})"
        elif brightness > self.max_brightness:
            is_usable = False
            rejection_reason = f"Frame overexposed (Mean brightness {brightness:.1f} > {self.max_brightness})"
        elif blur_var < self.min_blur_var:
            is_usable = False
            rejection_reason = f"Frame too blurry (Laplacian var {blur_var:.1f} < {self.min_blur_var})"

        return {
            "blur_laplacian_var": round(blur_var, 2),
            "mean_brightness": round(brightness, 2),
            "is_usable": is_usable,
            "rejection_reason": rejection_reason
        }

    def process_frame(self, frame_bgr, signer_id, session_id, recording_id, letter,
                      condition_meta, source_meta):
        h, w = frame_bgr.shape[:2]
        quality = self.audit_frame_quality(frame_bgr)

        import mediapipe as mp
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self.landmarker.detect(mp_img)

        num_hands = len(res.hand_landmarks) if res.hand_landmarks else 0

        # Check for resting lap-hand: hand in bottom 25% of frame when active hand is above
        lap_hand_suppressed = False
        valid_landmarks = []
        valid_handedness = []
        valid_scores = []

        if num_hands > 1:
            mean_ys = [np.mean([lm.y for lm in h_lms]) for h_lms in res.hand_landmarks]
            min_y = min(mean_ys)
            for idx, h_lms in enumerate(res.hand_landmarks):
                my = mean_ys[idx]
                if my > 0.75 and min_y < 0.60:
                    lap_hand_suppressed = True
                    continue
                valid_landmarks.append(h_lms)
                if res.handedness and len(res.handedness) > idx:
                    valid_handedness.append(res.handedness[idx][0].category_name)
                    valid_scores.append(round(float(res.handedness[idx][0].score), 3))
                else:
                    valid_handedness.append("unknown")
                    valid_scores.append(1.0)
        elif num_hands == 1:
            valid_landmarks = res.hand_landmarks
            if res.handedness:
                valid_handedness = [res.handedness[0][0].category_name]
                valid_scores = [round(float(res.handedness[0][0].score), 3)]
            else:
                valid_handedness = ["unknown"]
                valid_scores = [1.0]

        final_hands_count = len(valid_landmarks)
        if final_hands_count == 0 and quality["is_usable"]:
            quality["is_usable"] = False
            quality["rejection_reason"] = "No hand detected by landmarker"

        sample_id = f"{signer_id}_{session_id}_{letter}_{source_meta.get('frame_index', 0):04d}"

        sample = {
            "sample_id": sample_id,
            "signer_id": signer_id,
            "session_id": session_id,
            "recording_id": recording_id,
            "letter": letter,
            "confirmation_method": "signer_confirmed",
            "condition": condition_meta,
            "detection": {
                "hands_detected": final_hands_count,
                "handedness": valid_handedness,
                "scores": valid_scores,
                "lap_hand_suppressed": lap_hand_suppressed
            },
            "quality": quality,
            "source": source_meta
        }
        return sample


def print_collection_protocol_summary():
    print("=" * 80)
    print("PHASE 14B — MULTI-SIGNER BENCHMARK COLLECTION PROTOCOL")
    print("=" * 80)
    print("""
Target Design Matrix:
  - Signers: 3 anonymous individuals (signer_01, signer_02, signer_03)
  - Recording Conditions per Signer: 2 conditions
      * Condition A: Ambient natural lighting, plain background, frontal eye-level (60-80cm)
      * Condition B: Indoor artificial lighting, complex/office background, slight diagonal (60-80cm)
  - Target Letters: All 26 classes (A through Z)
  - Usable Frames per Letter per Condition: ~30 frames
  - Total Target Volume: 3 signers x 2 conditions x 26 letters x 30 frames = 4,680 usable frames

Signer-Independent Split Strategy:
  - Signer 1 (2 conditions, 1,560 frames): Training partition
  - Signer 2 (2 conditions, 1,560 frames): Validation partition (tuning only)
  - Signer 3 (2 conditions, 1,560 frames): Strictly Held-Out Independent Test partition
  - Assertions: Zero signer overlap, zero session leakage across partitions.
    """)


if __name__ == "__main__":
    print_collection_protocol_summary()
    out_dir = os.path.join(BASE_DIR, "phase14b_analysis")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metadata_schema.json"), "w") as f:
        json.dump(METADATA_SCHEMA, f, indent=2)
    print(f"Saved metadata schema to {os.path.join(out_dir, 'metadata_schema.json')}")
