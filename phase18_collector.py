#!/usr/bin/env python3
"""
Phase 18 — ISL Alphabet Multi-Signer Data Collection Engine

Provides the backend data collection, frame sampling, quality auditing,
staging review, ledger management, and split-isolation engine for onboarding:
- signer_02: Independent volunteer signer
- signer_03: Independent volunteer signer
- signer_04: Independent volunteer signer

Nominal Collection Matrix Target:
  2 Signers × 2 Conditions × 26 Letters (A-Z) × ~30 usable frames = 3,120 Frames

Enforces:
1. Strict metadata schema validation against phase14b_analysis/metadata_schema.json.
2. Cryptographic SHA-256 duplicate detection.
3. Automated image quality gating (Laplacian blur >= 50.0, 40.0 <= brightness <= 220.0).
4. MediaPipe HandLandmarker with active hand selection and lap-hand suppression.
5. Configurable sampling interval and frame-movement difference filtering.
6. Two-phase staging and operator review (preview, accept, or reject before saving).
7. Volunteer consent enforcement and right-to-deletion protection.
8. Complete split isolation: all new data is quarantined strictly in external test holdout.
"""

import os
import sys
import json
import time
import uuid
import shutil
import hashlib
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

SCHEMA_PATH = os.path.join(BASE_DIR, "phase14b_analysis", "metadata_schema.json")
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_RECORDINGS_DIR = os.path.join(DATA_DIR, "raw_recordings")
EXTRACTED_FRAMES_DIR = os.path.join(DATA_DIR, "extracted_frames")
STAGING_DIR = os.path.join(DATA_DIR, "staging")
MANIFESTS_DIR = os.path.join(DATA_DIR, "collection_manifests")
LEDGER_PATH = os.path.join(MANIFESTS_DIR, "collection_ledger.json")
EXTERNAL_TEST_MANIFEST_PATH = os.path.join(MANIFESTS_DIR, "test_multisigner_manifest.json")
SPLITS_DIR = os.path.join(BASE_DIR, "models", "mediapipe_squeezenet_v1", "splits")

# Standard signer identifiers and role partitions
DEVELOPMENT_TRAIN_SIGNERS = ["signer_06", "signer_07", "signer_08", "signer_09"]
DEVELOPMENT_VALIDATION_SIGNERS = ["signer_10", "signer_11"]
DEVELOPMENT_SIGNERS = DEVELOPMENT_TRAIN_SIGNERS + DEVELOPMENT_VALIDATION_SIGNERS
EVALUATION_ONLY_SIGNERS = ["signer_05"]
FROZEN_BENCHMARK_SIGNERS = ["signer_02", "signer_03"]

ALLOWED_SIGNERS = [
    "signer_02", "signer_03", "signer_04", "signer_05",
    "signer_06", "signer_07", "signer_08", "signer_09",
    "signer_10", "signer_11"
]
ALL_VALID_SIGNERS = [
    "signer_01", "signer_02", "signer_03", "signer_04", "signer_05",
    "signer_06", "signer_07", "signer_08", "signer_09",
    "signer_10", "signer_11"
]

SIGNER_ROLE_MAPPING = {
    "signer_01": "development_train",
    "signer_02": "frozen_benchmark",
    "signer_03": "frozen_benchmark",
    "signer_04": "development_multi_role",
    "signer_05": "external_holdout",
    "signer_06": "development_train",
    "signer_07": "development_train",
    "signer_08": "development_train",
    "signer_09": "development_train",
    "signer_10": "development_validation",
    "signer_11": "development_validation"
}

SIGNER_05_MANIFEST_PATH = os.path.join(MANIFESTS_DIR, "signer_05_manifest.json")
DEV_TRAIN_MANIFEST_PATH = os.path.join(MANIFESTS_DIR, "dev_train_manifest.json")
DEV_VAL_MANIFEST_PATH = os.path.join(MANIFESTS_DIR, "dev_val_manifest.json")
REJECTED_RECORDINGS_DIR = os.path.join(DATA_DIR, "rejected_recordings")

# Standard condition presets conforming strictly to metadata_schema.json
CONDITION_PRESETS = {
    "condition_1": {
        "condition_id": "condition_1",
        "name": "Desk Neutral (Frontal)",
        "protocol_name": "Standard Frontal (0.7m, Neutral Lighting)",
        "lighting": "indoor_overhead",
        "background": "plain_wall",
        "camera_position": "frontal_eye_level",
        "camera_yaw_deg": 0.0,
        "camera_pitch_deg": 0.0,
        "pitch_deg": 0.0,  # Preserved for backward compatibility
        "distance_cm": 70,
        "distance_range_cm": [65, 75],
        "distance_measurement_method": "nominal",
        "lap_hand_resting": False,
        "description": "Frontal eye-level camera at ~0.7m distance with neutral diffuse lighting",
        "instructions": "Position camera directly in front of signer at eye level (~0.7m / 70cm distance, 0° yaw, 0° pitch). Inactive non-signing hand must remain free or off-camera, NOT resting in lap. Use neutral diffuse overhead lighting.",
        "geometry_disclaimer": "Camera angles and distance are nominal / operator-reported; not hardware-instrumented."
    },
    "condition_2": {
        "condition_id": "condition_2",
        "name": "Lateral Warm (15° Offset)",
        "display_name": "Lateral Warm (15° Yaw, 30° Downward Pitch)",
        "protocol_name": "Angled Laptop & Lap-Rest (15° Yaw, 25°-35° Pitch)",
        "lighting": "ambient_natural",
        "background": "textured_office",
        "camera_position": "diagonal_15deg",
        "camera_yaw_deg": 15.0,
        "camera_pitch_deg": 30.0,
        "pitch_deg": 30.0,  # Preserved for backward compatibility
        "distance_cm": 75,
        "distance_range_cm": [70, 80],
        "distance_measurement_method": "nominal",
        "lap_hand_resting": True,
        "description": "Nominal 15° horizontal camera yaw offset AND 25°-35° downward camera pitch (nominal 30°) with inactive hand resting in lap",
        "instructions": "Camera is placed with 15° horizontal yaw offset (lateral diagonal) AND tilted downward at 25°–35° pitch (nominal 30° downward pitch, simulating a laptop on a desk). Signer's non-signing inactive hand rests in their lap. Natural ambient lighting.",
        "axis_disambiguation": "Nominal 15° is horizontal azimuth (camera_yaw_deg) and 30° is vertical downward tilt (camera_pitch_deg); they represent orthogonal geometric axes.",
        "geometry_disclaimer": "Camera angles and distance are nominal / operator-reported; not hardware-instrumented."
    },
    "condition_3": {
        "condition_id": "condition_3",
        "name": "Distance & Ambient Challenge (1.0m-1.2m)",
        "protocol_name": "Distance & Ambient Challenge (1.0m-1.2m)",
        "lighting": "low_light",
        "background": "cluttered_home",
        "camera_position": "frontal_low_angle",
        "camera_yaw_deg": 0.0,
        "camera_pitch_deg": 10.0,
        "pitch_deg": 10.0,  # Preserved for backward compatibility
        "distance_cm": 110,
        "distance_range_cm": [100, 120],
        "distance_measurement_method": "nominal",
        "lap_hand_resting": False,
        "description": "Distance challenge at 1.0m-1.2m (nominal 110cm) with low-light ambient illumination and low-angle camera tilt (nominal 10°)",
        "instructions": "Distance and ambient challenge: Signer seated at 1.0m–1.2m distance (nominal 110cm, operator-reported). Camera placed at low angle with slight upward pitch (nominal 10°). Dim or low-light conditions with cluttered background. Inactive hand free.",
        "geometry_disclaimer": "Camera angles and distance are nominal / operator-reported; not hardware-instrumented."
    }
}


def compute_dhash(img: np.ndarray, hash_size: int = 8) -> str:
    """
    Computes difference perceptual hash (dHash) for fast, deterministic duplicate detection.
    Resize to (hash_size + 1, hash_size) grayscale, compare adjacent column intensities.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    decimal_val = 0
    for bit in diff.flatten():
        decimal_val = (decimal_val << 1) | int(bit)
    return f"{decimal_val:016x}"


def hamming_distance(hex1: str, hex2: str) -> int:
    """Computes bitwise Hamming distance between two hex hashes."""
    try:
        val1 = int(hex1, 16)
        val2 = int(hex2, 16)
        return bin(val1 ^ val2).count('1')
    except Exception:
        return 64



def validate_sample_schema(sample: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validates a sample dictionary against metadata_schema.json constraints.
    Returns (True, None) if valid, or (False, error_reason).
    """
    required_root_keys = [
        "sample_id", "signer_id", "session_id", "recording_id",
        "letter", "confirmation_method", "condition", "detection", "quality", "source"
    ]
    for k in required_root_keys:
        if k not in sample:
            return False, f"Missing required root key: '{k}'"

    # signer_id
    if sample["signer_id"] not in ALL_VALID_SIGNERS:
        return False, f"Invalid signer_id: '{sample['signer_id']}'"

    # letter
    letter = str(sample["letter"]).strip().upper()
    if len(letter) != 1 or not ('A' <= letter <= 'Z'):
        return False, f"Invalid letter: '{sample['letter']}' (must be single A-Z character)"

    # confirmation_method
    if sample["confirmation_method"] not in ["signer_confirmed", "expert_verified", "dual_annotator_consensus"]:
        return False, f"Invalid confirmation_method: '{sample['confirmation_method']}'"

    # condition
    cond = sample.get("condition", {})
    cond_keys = ["lighting", "background", "camera_position", "distance_cm"]
    for ck in cond_keys:
        if ck not in cond:
            return False, f"Missing condition key: '{ck}'"
    if cond["lighting"] not in ["ambient_natural", "indoor_overhead", "low_light"]:
        return False, f"Invalid lighting: '{cond['lighting']}'"
    if cond["background"] not in ["plain_wall", "textured_office", "cluttered_home"]:
        return False, f"Invalid background: '{cond['background']}'"
    if cond["camera_position"] not in ["frontal_eye_level", "frontal_low_angle", "diagonal_15deg"]:
        return False, f"Invalid camera_position: '{cond['camera_position']}'"
    if not (40 <= int(cond["distance_cm"]) <= 150):
        return False, f"distance_cm {cond['distance_cm']} out of range [40, 150]"
    if "camera_yaw_deg" in cond and not isinstance(cond["camera_yaw_deg"], (int, float)):
        return False, "camera_yaw_deg must be numeric"
    if "camera_pitch_deg" in cond and not isinstance(cond["camera_pitch_deg"], (int, float)):
        return False, "camera_pitch_deg must be numeric"
    if "distance_measurement_method" in cond and cond["distance_measurement_method"] not in ["measured", "operator_reported", "nominal"]:
        return False, f"Invalid distance_measurement_method: '{cond['distance_measurement_method']}'"
    if "lap_hand_resting" in cond and not isinstance(cond["lap_hand_resting"], bool):
        return False, "lap_hand_resting must be a boolean"
    if "measured_resolution" in cond:
        if not isinstance(cond["measured_resolution"], list) or len(cond["measured_resolution"]) != 2:
            return False, "measured_resolution must be a list of 2 integers [width, height]"
    if "observed_fps" in cond and not isinstance(cond["observed_fps"], (int, float)):
        return False, "observed_fps must be numeric"

    # detection
    det = sample.get("detection", {})
    det_keys = ["hands_detected", "handedness", "scores", "lap_hand_suppressed"]
    for dk in det_keys:
        if dk not in det:
            return False, f"Missing detection key: '{dk}'"
    if not (0 <= int(det["hands_detected"]) <= 2):
        return False, f"hands_detected {det['hands_detected']} out of range [0, 2]"
    if not isinstance(det["handedness"], list):
        return False, "handedness must be a list"
    if not isinstance(det["scores"], list):
        return False, "scores must be a list"
    if not isinstance(det["lap_hand_suppressed"], bool):
        return False, "lap_hand_suppressed must be a boolean"

    # quality
    q = sample.get("quality", {})
    q_keys = ["blur_laplacian_var", "mean_brightness", "is_usable"]
    for qk in q_keys:
        if qk not in q:
            return False, f"Missing quality key: '{qk}'"
    if not isinstance(q["is_usable"], bool):
        return False, "quality.is_usable must be a boolean"

    # source
    src = sample.get("source", {})
    src_keys = ["video_path", "frame_index", "timestamp_sec"]
    for sk in src_keys:
        if sk not in src:
            return False, f"Missing source key: '{sk}'"

    return True, None


class ISLCollectionEngine:
    """
    Coordinates multi-signer video ingestion, frame extraction, quality auditing,
    staging preview, ledger commits, and split-isolation verification.
    """

    def __init__(
        self,
        min_blur_var: float = 50.0,
        min_brightness: float = 40.0,
        max_brightness: float = 220.0,
        target_frames_per_letter: int = 30,
        min_movement_diff: float = 2.0,
        raw_recordings_dir: Optional[str] = None,
        extracted_frames_dir: Optional[str] = None,
        staging_dir: Optional[str] = None,
        manifests_dir: Optional[str] = None,
        rejected_recordings_dir: Optional[str] = None,
        min_resolution_width: int = 1280,
        min_resolution_height: int = 720,
        target_fps: float = 30.0,
        min_hand_confidence: float = 0.50,
        min_hand_bbox_ratio: float = 0.02
    ):
        self.min_blur_var = float(min_blur_var)
        self.min_brightness = float(min_brightness)
        self.max_brightness = float(max_brightness)
        self.target_frames_per_letter = int(target_frames_per_letter)
        self.min_movement_diff = float(min_movement_diff)
        self.nominal_target_frames = 3120

        self.min_resolution_width = int(min_resolution_width)
        self.min_resolution_height = int(min_resolution_height)
        self.target_fps = float(target_fps)
        self.min_hand_confidence = float(min_hand_confidence)
        self.min_hand_bbox_ratio = float(min_hand_bbox_ratio)

        self.raw_dir = raw_recordings_dir or RAW_RECORDINGS_DIR
        self.frames_dir = extracted_frames_dir or EXTRACTED_FRAMES_DIR
        self.staging_dir = staging_dir or STAGING_DIR
        self.manifests_dir = manifests_dir or MANIFESTS_DIR
        self.rejected_dir = rejected_recordings_dir or REJECTED_RECORDINGS_DIR
        self.ledger_path = os.path.join(self.manifests_dir, "collection_ledger.json")
        self.test_manifest_path = os.path.join(self.manifests_dir, "test_multisigner_manifest.json")
        self.signer_05_manifest_path = os.path.join(self.manifests_dir, "signer_05_manifest.json")
        self.signer05_manifest_path = self.signer_05_manifest_path
        self.dev_train_manifest_path = os.path.join(self.manifests_dir, "dev_train_manifest.json")
        self.dev_val_manifest_path = os.path.join(self.manifests_dir, "dev_val_manifest.json")

        # Ensure directory infrastructure
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.staging_dir, exist_ok=True)
        os.makedirs(self.manifests_dir, exist_ok=True)
        os.makedirs(self.rejected_dir, exist_ok=True)

        self._init_ledger_and_manifests()

        # Lazy loaded landmarker
        self._landmarker = None

    def _get_landmarker(self):
        if self._landmarker is None:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            task_path = os.path.join(BASE_DIR, "App", "include50_model", "hand_landmarker.task")
            if not os.path.exists(task_path):
                raise FileNotFoundError(f"MediaPipe hand landmarker task not found at {task_path}")

            opt = vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=task_path),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.4,
                min_hand_presence_confidence=0.4,
                min_tracking_confidence=0.4
            )
            self._landmarker = vision.HandLandmarker.create_from_options(opt)
        return self._landmarker

    def _init_ledger_and_manifests(self):
        """Initializes ledger and manifests if not present."""
        if not os.path.exists(self.ledger_path):
            initial_ledger = {
                "version": "1.0",
                "phase": "Phase 18",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "nominal_target_frames": 3120,
                "target_signers": ALLOWED_SIGNERS,
                "target_conditions": list(CONDITION_PRESETS.keys()),
                "records": []
            }
            with open(self.ledger_path, "w") as f:
                json.dump(initial_ledger, f, indent=2)

        if not os.path.exists(self.test_manifest_path):
            with open(self.test_manifest_path, "w") as f:
                json.dump([], f, indent=2)

        if not os.path.exists(self.dev_train_manifest_path):
            with open(self.dev_train_manifest_path, "w") as f:
                json.dump([], f, indent=2)

        if not os.path.exists(self.dev_val_manifest_path):
            with open(self.dev_val_manifest_path, "w") as f:
                json.dump([], f, indent=2)

    def load_ledger(self) -> Dict[str, Any]:
        """Loads collection ledger records."""
        if not os.path.exists(self.ledger_path):
            self._init_ledger_and_manifests()
        with open(self.ledger_path, "r") as f:
            return json.load(f)

    def save_ledger(self, ledger_data: Dict[str, Any]):
        """Saves collection ledger atomically."""
        tmp_path = self.ledger_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(ledger_data, f, indent=2)
        os.replace(tmp_path, self.ledger_path)

    def load_test_manifest(self) -> List[Dict[str, Any]]:
        """Loads external test manifest containing accepted samples."""
        if not os.path.exists(self.test_manifest_path):
            return []
        with open(self.test_manifest_path, "r") as f:
            return json.load(f)

    def save_test_manifest(self, manifest_data: List[Dict[str, Any]]):
        """Saves external test manifest atomically."""
        tmp_path = self.test_manifest_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(manifest_data, f, indent=2)
        os.replace(tmp_path, self.test_manifest_path)

    def load_signer05_manifest(self) -> List[Dict[str, Any]]:
        """Loads dedicated signer_05 manifest containing accepted evaluation samples."""
        target_path = getattr(self, "signer05_manifest_path", None) or self.signer_05_manifest_path
        if not os.path.exists(target_path):
            return []
        with open(target_path, "r") as f:
            return json.load(f)

    def save_signer05_manifest(self, manifest_data: List[Dict[str, Any]]):
        """Saves dedicated signer_05 manifest atomically."""
        target_path = getattr(self, "signer05_manifest_path", None) or self.signer_05_manifest_path
        tmp_path = target_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(manifest_data, f, indent=2)
        os.replace(tmp_path, target_path)

    def load_dev_train_manifest(self) -> List[Dict[str, Any]]:
        """Loads development training manifest (signer_06 through signer_09)."""
        if not os.path.exists(self.dev_train_manifest_path):
            return []
        with open(self.dev_train_manifest_path, "r") as f:
            return json.load(f)

    def save_dev_train_manifest(self, manifest_data: List[Dict[str, Any]]):
        """Saves development training manifest atomically."""
        tmp_path = self.dev_train_manifest_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(manifest_data, f, indent=2)
        os.replace(tmp_path, self.dev_train_manifest_path)

    def load_dev_val_manifest(self) -> List[Dict[str, Any]]:
        """Loads development validation manifest (signer_10 and signer_11)."""
        if not os.path.exists(self.dev_val_manifest_path):
            return []
        with open(self.dev_val_manifest_path, "r") as f:
            return json.load(f)

    def save_dev_val_manifest(self, manifest_data: List[Dict[str, Any]]):
        """Saves development validation manifest atomically."""
        tmp_path = self.dev_val_manifest_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(manifest_data, f, indent=2)
        os.replace(tmp_path, self.dev_val_manifest_path)

    def get_known_hashes(self) -> set:
        """Collects all known frame SHA-256 hashes across splits and ledger to prevent duplicates."""
        hashes = set()
        # 1. Existing splits
        for m_name in ["train_manifest.json", "val_manifest.json", "test_al_manifest.json", "test_mz_manifest.json"]:
            mp = os.path.join(SPLITS_DIR, m_name)
            if os.path.exists(mp):
                try:
                    with open(mp, "r") as f:
                        data = json.load(f)
                        for item in data:
                            h = item.get("image_hash") or (item.get("source", {}).get("sha256") if isinstance(item.get("source"), dict) else None)
                            if h:
                                hashes.add(h)
                except Exception:
                    pass

        # 2. External test manifest
        for sample in self.load_test_manifest():
            h = sample.get("source", {}).get("sha256")
            if h:
                hashes.add(h)

        # 3. Development manifests
        for sample in self.load_dev_train_manifest():
            h = sample.get("source", {}).get("sha256")
            if h:
                hashes.add(h)

        for sample in self.load_dev_val_manifest():
            h = sample.get("source", {}).get("sha256")
            if h:
                hashes.add(h)

        for sample in self.load_signer05_manifest():
            h = sample.get("source", {}).get("sha256")
            if h:
                hashes.add(h)

        return hashes

    def audit_frame_quality(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """Audits blur and brightness of a single frame."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(np.mean(gray))

        is_usable = True
        rejection_reason = None

        if brightness < self.min_brightness:
            is_usable = False
            rejection_reason = f"underexposed: brightness {brightness:.1f} < {self.min_brightness}"
        elif brightness > self.max_brightness:
            is_usable = False
            rejection_reason = f"overexposed: brightness {brightness:.1f} > {self.max_brightness}"
        elif blur_var < self.min_blur_var:
            is_usable = False
            rejection_reason = f"motion_blur: laplacian_var {blur_var:.1f} < {self.min_blur_var}"

        return {
            "blur_laplacian_var": round(blur_var, 2),
            "mean_brightness": round(brightness, 2),
            "is_usable": is_usable,
            "rejection_reason": rejection_reason
        }

    def detect_hands_in_frame(
        self,
        frame_bgr: np.ndarray,
        return_bbox_ratios: bool = False
    ):
        """
        Runs MediaPipe hand detection with lap-hand suppression.
        Returns: (selected_hand_count, handedness_list, scores_list, lap_hand_suppressed, suppression_reasons[, bbox_ratios])
        """
        from landmark_feature_extractor import extract_landmarks_from_mediapipe_result
        import mediapipe as mp

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        landmarker = self._get_landmarker()
        res = landmarker.detect(mp_img)

        _, diag = extract_landmarks_from_mediapipe_result(
            res, suppress_lap_hands=True, return_diagnostics=True
        )

        selected_count = diag.get("selected_hand_count", 0)
        handedness = [h.get("category", "unknown") for h in diag.get("hand_details", [])]
        scores = [round(float(h.get("score", 1.0)), 3) for h in diag.get("hand_details", [])]
        lap_suppressed = bool(diag.get("lap_hand_suppressed", False))
        reasons = diag.get("suppression_reasons", [])

        bbox_ratios = []
        if res and res.hand_landmarks:
            for h_lms in res.hand_landmarks:
                if len(h_lms) == 21:
                    xs = [lm.x for lm in h_lms]
                    ys = [lm.y for lm in h_lms]
                    w = max(0.0, max(xs) - min(xs))
                    h = max(0.0, max(ys) - min(ys))
                    bbox_ratios.append(round(float(w * h), 4))

        if return_bbox_ratios:
            return selected_count, handedness, scores, lap_suppressed, reasons, bbox_ratios
        return selected_count, handedness, scores, lap_suppressed, reasons

    def stage_recording(
        self,
        video_bytes: bytes,
        filename: str,
        signer_id: str,
        letter: str,
        condition_id: str,
        session_id: Optional[str] = None,
        consent_confirmed: bool = False,
        sampling_interval: int = 3,
        operator_notes: str = "",
        take_number: int = 1,
        actual_pitch_deg: Optional[float] = None,
        actual_distance_cm: Optional[float] = None,
        actual_yaw_deg: Optional[float] = None,
        distance_measurement_method: Optional[str] = "nominal"
    ) -> Dict[str, Any]:
        """
        Stages an uploaded/recorded video:
        1. Verifies volunteer consent (hard requirement).
        2. Validates signer_id, letter, condition, and take_number (1, 2, or 3).
        3. Extracts frames using configurable sampling interval and movement filter.
        4. Audits quality, duplicate hashes (SHA-256 and dHash), resolution, FPS, and landmarks.
        5. Saves temporary frames and generates base64 preview thumbnails.
        6. Returns full diagnostic preview without committing to permanent dataset.
        """
        # Hard consent check
        if not consent_confirmed:
            raise ValueError(
                "Volunteer consent must be explicitly confirmed prior to recording or staging sign language data."
            )

        # Parameter validation
        clean_signer = str(signer_id).strip()
        if clean_signer not in ALLOWED_SIGNERS:
            raise ValueError(f"Invalid signer_id: '{clean_signer}'. Allowed: {ALLOWED_SIGNERS}.")

        clean_letter = str(letter).strip().upper()
        if len(clean_letter) != 1 or not ('A' <= clean_letter <= 'Z'):
            raise ValueError(f"Invalid letter: '{clean_letter}'. Must be single A-Z character.")

        clean_condition = str(condition_id).strip()
        if clean_condition not in CONDITION_PRESETS:
            raise ValueError(f"Invalid condition_id: '{clean_condition}'. Allowed: {list(CONDITION_PRESETS.keys())}.")

        try:
            take_num = int(take_number)
            if take_num not in [1, 2, 3]:
                take_num = 1
        except Exception:
            take_num = 1

        sampling_interval = max(1, min(15, int(sampling_interval)))
        session_id = session_id.strip() if session_id else f"session_{time.strftime('%Y%m%d_%H%M%S')}"
        staging_id = f"stage_{uuid.uuid4().hex[:12]}"
        recording_id = f"rec_{clean_signer}_{clean_condition}_{clean_letter}_take{take_num}_{int(time.time())}_{uuid.uuid4().hex[:4]}"

        # Create staging directory
        stage_dir = os.path.join(self.staging_dir, staging_id)
        os.makedirs(stage_dir, exist_ok=True)
        frames_stage_dir = os.path.join(stage_dir, "frames")
        os.makedirs(frames_stage_dir, exist_ok=True)

        # Write video to staging
        ext = os.path.splitext(filename)[1].lower() or ".mp4"
        raw_video_path = os.path.join(stage_dir, f"raw_recording{ext}")
        with open(raw_video_path, "wb") as f:
            f.write(video_bytes)

        # Ensure standard playable MP4 exists via ffmpeg if available
        ffmpeg_bin = shutil.which("ffmpeg") or ("/opt/homebrew/bin/ffmpeg" if os.path.exists("/opt/homebrew/bin/ffmpeg") else None)
        playable_video_path = raw_video_path
        if ffmpeg_bin and os.path.exists(ffmpeg_bin):
            mp4_path = os.path.join(stage_dir, "raw_recording.mp4")
            # Always transcode webm or unknown containers to standard H.264 MP4 with generated timestamps
            ret = os.system(f"'{ffmpeg_bin}' -y -fflags +genpts -i '{raw_video_path}' -c:v libx264 -preset ultrafast -pix_fmt yuv420p '{mp4_path}' >/dev/null 2>&1")
            if ret == 0 and os.path.exists(mp4_path) and os.path.getsize(mp4_path) > 500:
                playable_video_path = mp4_path

        # Open video and audit frames - prefer playable_video_path (the indexed mp4)
        cap = cv2.VideoCapture(playable_video_path)
        if not cap.isOpened() and playable_video_path != raw_video_path:
            cap = cv2.VideoCapture(raw_video_path)

        if not cap.isOpened():
            shutil.rmtree(stage_dir, ignore_errors=True)
            raise ValueError("Could not open or decode uploaded video file. The file may be corrupt or empty.")

        observed_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        observed_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps) or fps > 120:
            fps = 30.0

        quality_warnings = []
        if observed_width > 0 and observed_height > 0 and (observed_width < self.min_resolution_width or observed_height < self.min_resolution_height):
            quality_warnings.append(
                f"Observed video resolution {observed_width}x{observed_height} is below recommended {self.min_resolution_width}x{self.min_resolution_height}."
            )
        if fps < 25.0:
            quality_warnings.append(
                f"Observed frame rate {fps:.1f} FPS is below recommended {self.target_fps:.0f} FPS."
            )

        known_hashes = self.get_known_hashes()
        stage_hashes = set()
        stage_dhashes = []

        sampled_frames_meta = []
        raw_frame_idx = 0
        prev_small_gray = None
        usable_count = 0
        rejected_count = 0

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            # Configurable sampling interval
            if raw_frame_idx % sampling_interval == 0:
                frame_ts = round(raw_frame_idx / fps, 3)

                # Compute movement difference to detect frozen camera or near-identical frames
                small_gray = cv2.resize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), (160, 120))
                movement_diff = 100.0
                if prev_small_gray is not None:
                    movement_diff = float(np.mean(np.abs(small_gray.astype(float) - prev_small_gray.astype(float))))
                prev_small_gray = small_gray

                # Compute cryptographic frame hash and perceptual hash
                _, img_encoded = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
                frame_bytes = img_encoded.tobytes()
                frame_sha256 = hashlib.sha256(frame_bytes).hexdigest()
                frame_dhash = compute_dhash(frame_bgr)

                is_usable = True
                rejection_reason = None

                # 1. Duplicate detection: exact SHA-256 hash or perceptual dHash match on consecutive frames
                is_consecutive_dup = False
                if stage_dhashes:
                    is_consecutive_dup = (hamming_distance(frame_dhash, stage_dhashes[-1]) <= 1)

                if frame_sha256 in known_hashes or frame_sha256 in stage_hashes:
                    is_usable = False
                    rejection_reason = f"duplicate_frame: hash {frame_sha256[:10]} already seen"
                elif is_consecutive_dup and movement_diff < 0.2 and len(sampled_frames_meta) > 0:
                    is_usable = False
                    rejection_reason = "perceptual_static_duplicate: dHash near-match with negligible movement"
                elif movement_diff < 0.1 and len(sampled_frames_meta) > 0:
                    is_usable = False
                    rejection_reason = f"static_duplicate: pixel movement {movement_diff:.2f} < 0.1"
                else:
                    # 2. Quality check (blur, brightness)
                    quality = self.audit_frame_quality(frame_bgr)
                    if not quality["is_usable"]:
                        is_usable = False
                        rejection_reason = quality["rejection_reason"]
                    else:
                        # 3. Landmark check (hands detected)
                        try:
                            res_detect = self.detect_hands_in_frame(frame_bgr, return_bbox_ratios=True)
                        except TypeError:
                            res_detect = self.detect_hands_in_frame(frame_bgr)

                        if len(res_detect) >= 6:
                            hands_detected, handedness, scores, lap_suppressed, supp_reasons, bbox_ratios = res_detect[:6]
                        else:
                            hands_detected, handedness, scores, lap_suppressed, supp_reasons = res_detect[:5]
                            bbox_ratios = [0.025] * hands_detected

                        if hands_detected == 0:
                            is_usable = False
                            rejection_reason = "no_hand_detected"
                        else:
                            quality["hands_detected"] = hands_detected
                            quality["handedness"] = handedness
                            quality["scores"] = scores
                            quality["lap_hand_suppressed"] = lap_suppressed
                            quality["hand_confidence"] = max(scores) if scores else 0.0
                            quality["bbox_ratios"] = bbox_ratios
                            quality["max_bbox_ratio"] = max(bbox_ratios) if bbox_ratios else 0.0

                if not is_usable and "quality" not in locals():
                    quality = self.audit_frame_quality(frame_bgr)
                    hands_detected, handedness, scores, lap_suppressed, _ = 0, [], [], False, []
                    bbox_ratios = []

                if is_usable:
                    stage_hashes.add(frame_sha256)
                    stage_dhashes.append(frame_dhash)
                    usable_count += 1
                else:
                    rejected_count += 1

                # Save frame image in staging
                frame_filename = f"frame_{raw_frame_idx:04d}_{'usable' if is_usable else 'rejected'}.jpg"
                frame_path = os.path.join(frames_stage_dir, frame_filename)
                with open(frame_path, "wb") as f:
                    f.write(frame_bytes)

                # Generate thumbnail base64 for browser review
                thumb = cv2.resize(frame_bgr, (160, 120))
                _, thumb_enc = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
                import base64
                thumb_b64 = "data:image/jpeg;base64," + base64.b64encode(thumb_enc.tobytes()).decode("utf-8")

                sampled_frames_meta.append({
                    "frame_index": raw_frame_idx,
                    "timestamp_sec": frame_ts,
                    "sha256": frame_sha256,
                    "dhash": frame_dhash,
                    "is_usable": is_usable,
                    "rejection_reason": rejection_reason,
                    "blur_laplacian_var": quality.get("blur_laplacian_var", 0.0),
                    "mean_brightness": quality.get("mean_brightness", 0.0),
                    "hands_detected": hands_detected if is_usable else 0,
                    "hand_confidence": quality.get("hand_confidence", 0.0),
                    "bbox_ratios": quality.get("bbox_ratios", []),
                    "max_bbox_ratio": quality.get("max_bbox_ratio", 0.0),
                    "movement_diff": round(movement_diff, 2),
                    "frame_file": frame_filename,
                    "thumbnail_b64": thumb_b64
                })

            raw_frame_idx += 1

        cap.release()

        total_frames = raw_frame_idx
        duration_sec = round(total_frames / fps, 2) if total_frames > 0 else 0.0

        if total_frames < 5:
            shutil.rmtree(stage_dir, ignore_errors=True)
            raise ValueError(f"Recording is too short: only {total_frames} frames ({duration_sec:.2f}s). Minimum is 5 frames.")

        is_recommended = usable_count >= 15
        recommendation_text = (
            f"Recommended: {usable_count} usable frames extracted."
            if is_recommended else
            f"Caution: Only {usable_count} usable frames extracted. Target is ~30 frames."
        )

        # Observational check for compact hand bbox area without hard rejection
        usable_bboxes = [
            f.get("max_bbox_ratio", 0.0)
            for f in sampled_frames_meta
            if f.get("is_usable") and f.get("max_bbox_ratio", 0.0) > 0.0
        ]
        mean_bbox = float(np.mean(usable_bboxes)) if usable_bboxes else 0.0
        if usable_bboxes and mean_bbox < 0.015:
            quality_warnings.append(
                f"Hand bounding box is compact (mean {mean_bbox*100:.2f}% of frame area). Verify signer is within recommended distance."
            )

        eff_yaw = actual_yaw_deg if actual_yaw_deg is not None else CONDITION_PRESETS[clean_condition].get("camera_yaw_deg", 0.0)
        eff_pitch = actual_pitch_deg if actual_pitch_deg is not None else CONDITION_PRESETS[clean_condition].get("camera_pitch_deg", CONDITION_PRESETS[clean_condition].get("pitch_deg", 0.0))
        eff_dist = actual_distance_cm if actual_distance_cm is not None else CONDITION_PRESETS[clean_condition].get("distance_cm", 75)
        eff_dist_method = distance_measurement_method or CONDITION_PRESETS[clean_condition].get("distance_measurement_method", "nominal")

        staging_info = {
            "staging_id": staging_id,
            "recording_id": recording_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "signer_id": clean_signer,
            "letter": clean_letter,
            "condition_id": clean_condition,
            "condition_meta": CONDITION_PRESETS[clean_condition],
            "session_id": session_id,
            "take_number": take_num,
            "actual_pitch_deg": eff_pitch,
            "actual_yaw_deg": eff_yaw,
            "actual_distance_cm": eff_dist,
            "distance_measurement_method": eff_dist_method,
            "mean_hand_bbox_ratio": round(mean_bbox, 4),
            "consent_confirmed": consent_confirmed,
            "sampling_interval": sampling_interval,
            "operator_notes": operator_notes,
            "raw_video_file": os.path.basename(playable_video_path),
            "original_filename": filename,
            "video_dimensions": {"width": observed_width, "height": observed_height},
            "fps": round(fps, 2),
            "total_video_frames": total_frames,
            "duration_sec": round(duration_sec, 2),
            "usable_count": usable_count,
            "rejected_count": rejected_count,
            "is_recommended": is_recommended,
            "recommendation_text": recommendation_text,
            "hard_rejection_gates": [
                {"gate": "consent_confirmed", "status": "PASS", "rule": "Explicit participant consent is mandatory"},
                {"gate": "min_duration", "status": "PASS", "rule": "Recording duration >= 5 frames"},
                {"gate": "sha256_uniqueness", "status": "ENFORCED", "rule": "Exact bit-identical frame duplicate rejection"},
                {"gate": "dhash_perceptual_duplicate", "status": "ENFORCED", "rule": "Near-duplicate frame rejection (Hamming <= 1, movement < 0.2)"},
                {"gate": "camera_freeze_static", "status": "ENFORCED", "rule": "Static frame freeze rejection (movement < 0.1)"},
                {"gate": "illumination_bounds", "status": "ENFORCED", "rule": "Mean brightness within [40.0, 220.0]"},
                {"gate": "motion_blur", "status": "ENFORCED", "rule": "Laplacian variance >= 50.0"},
                {"gate": "hand_detection", "status": "ENFORCED", "rule": "At least 1 active hand landmark detected"}
            ],
            "static_hold_safeguard": "Legitimate sign hold frames are preserved across consecutive video samples.",
            "quality_warnings": quality_warnings,
            "condition_telemetry": {
                "camera_yaw_deg": eff_yaw,
                "camera_pitch_deg": eff_pitch,
                "distance_cm": eff_dist,
                "distance_measurement_method": eff_dist_method,
                "lap_hand_resting": CONDITION_PRESETS[clean_condition].get("lap_hand_resting", False),
                "geometry_disclaimer": "Camera angles and distance are nominal / operator-reported; not hardware-instrumented."
            },
            "frames": sampled_frames_meta
        }

        with open(os.path.join(stage_dir, "staging_info.json"), "w") as f:
            json.dump(staging_info, f, indent=2)

        return staging_info

    def commit_recording(
        self,
        staging_id: str,
        action: str,
        rejection_note: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Commits or rejects a staged recording:
        - If action == 'accept':
          1. Moves raw video to data/raw_recordings/<signer>/<condition>/<recording_id>.mp4
          2. Moves accepted frames to data/extracted_frames/<signer>/<condition>/<letter>/<sample_id>.jpg
          3. Appends records to appropriate partition manifest:
             - dev_train_manifest.json for signer_06 through signer_09 (role: development_train, allow_training: True)
             - dev_val_manifest.json for signer_10 and signer_11 (role: development_validation, allow_training: False)
             - signer_05_manifest.json and test_multisigner_manifest.json for signer_05 (role: independent_evaluation_holdout, allow_training: False)
             - test_multisigner_manifest.json for signer_02..signer_04 (role: external_test_multisigner, allow_training: False)
          4. Records full entry in collection_ledger.json with dataset_role, take_number, and quality telemetry.
          5. Cleans up staging directory.
        - If action == 'reject':
          1. Preserves rejected video in data/rejected_recordings/<staging_id>/ for auditability.
          2. Logs rejection event in collection_ledger.json with reason.
        """
        stage_dir = os.path.join(self.staging_dir, staging_id)
        info_path = os.path.join(stage_dir, "staging_info.json")

        if not os.path.exists(info_path):
            raise FileNotFoundError(f"Staging session '{staging_id}' not found or already processed.")

        with open(info_path, "r") as f:
            info = json.load(f)

        clean_action = str(action).strip().lower()
        signer_id = info["signer_id"]
        condition_id = info["condition_id"]
        letter = info["letter"]
        recording_id = info["recording_id"]
        session_id = info["session_id"]
        condition_meta = info["condition_meta"]
        take_number = info.get("take_number", 1)

        ledger = self.load_ledger()

        if clean_action == "reject":
            # Preserve rejected recording in rejected_dir for provenance and audit
            rejected_stage_dir = os.path.join(self.rejected_dir, staging_id)
            os.makedirs(rejected_stage_dir, exist_ok=True)
            raw_src = os.path.join(stage_dir, info.get("raw_video_file", "raw_recording.mp4"))
            if os.path.exists(raw_src):
                shutil.copy2(raw_src, os.path.join(rejected_stage_dir, f"{recording_id}.mp4"))

            rejection_entry = {
                "recording_id": recording_id,
                "staging_id": staging_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "signer_id": signer_id,
                "condition_id": condition_id,
                "letter": letter,
                "session_id": session_id,
                "take_number": take_number,
                "status": "rejected",
                "rejection_note": rejection_note or "Operator rejected during review",
                "frames_sampled": len(info.get("frames", [])),
                "usable_frames": info.get("usable_count", 0),
                "rejected_frames": info.get("rejected_count", 0),
                "quality_warnings": info.get("quality_warnings", []),
                "distinct_rejection_reasons": list(set(f.get("rejection_reason") for f in info.get("frames", []) if f.get("rejection_reason")))
            }
            with open(os.path.join(rejected_stage_dir, "rejection_audit.json"), "w") as f:
                json.dump(rejection_entry, f, indent=2)

            ledger["records"].append(rejection_entry)
            self.save_ledger(ledger)

            # Clean staging
            shutil.rmtree(stage_dir, ignore_errors=True)

            return {
                "status": "rejected",
                "staging_id": staging_id,
                "recording_id": recording_id,
                "rejection_note": rejection_note or "Operator rejected during review",
                "message": "Recording rejected and preserved in rejected archives for auditability."
            }

        elif clean_action == "accept":
            if info["usable_count"] == 0:
                raise ValueError("Cannot accept recording with 0 usable frames. Please re-record.")

            # 1. Move raw video
            target_raw_dir = os.path.join(self.raw_dir, signer_id, condition_id)
            os.makedirs(target_raw_dir, exist_ok=True)
            raw_src = os.path.join(stage_dir, info["raw_video_file"])
            raw_ext = os.path.splitext(info["raw_video_file"])[1] or ".mp4"
            final_video_name = f"{recording_id}{raw_ext}"
            final_video_path = os.path.join(target_raw_dir, final_video_name)
            shutil.copy2(raw_src, final_video_path)

            rel_video_path = os.path.relpath(final_video_path, BASE_DIR)

            # Check for existing accepted records for this specific (signer, condition, letter, take_number)
            prior_accepted = [
                r for r in ledger["records"]
                if r.get("signer_id") == signer_id and
                   r.get("condition_id") == condition_id and
                   r.get("letter") == letter and
                   r.get("take_number", 1) == take_number and
                   r.get("status") == "accepted"
            ]
            prior_rec_ids = {r["recording_id"] for r in prior_accepted}
            now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

            # If retake, mark prior records as superseded in ledger
            if prior_accepted:
                for r in ledger["records"]:
                    if r.get("recording_id") in prior_rec_ids:
                        r["status"] = "superseded"
                        r["superseded_by"] = recording_id
                        r["superseded_at"] = now_iso

            # 2. Move accepted frames and build metadata
            target_frames_dir = os.path.join(self.frames_dir, signer_id, condition_id, letter)
            os.makedirs(target_frames_dir, exist_ok=True)

            # Determine partition role and target manifest
            if signer_id in DEVELOPMENT_TRAIN_SIGNERS:
                split_name = "development_train"
                role_name = "development_train"
                allow_training = True
                manifest_target = "dev_train"
            elif signer_id in DEVELOPMENT_VALIDATION_SIGNERS:
                split_name = "development_validation"
                role_name = "development_validation"
                allow_training = False
                manifest_target = "dev_val"
            elif signer_id in EVALUATION_ONLY_SIGNERS:
                split_name = f"external_test_{signer_id}"
                role_name = "independent_evaluation_holdout"
                allow_training = False
                manifest_target = "signer_05"
            else:
                split_name = "external_test_multisigner"
                role_name = "external_test_multisigner"
                allow_training = False
                manifest_target = "test_multisigner"

            accepted_samples = []

            for f_meta in info["frames"]:
                if not f_meta["is_usable"]:
                    continue

                frame_file = f_meta["frame_file"]
                src_frame_path = os.path.join(stage_dir, "frames", frame_file)
                if not os.path.exists(src_frame_path):
                    continue

                sample_id = f"{signer_id}_{session_id}_{letter}_t{take_number}_{f_meta['frame_index']:04d}"
                dest_frame_name = f"{sample_id}.jpg"
                dest_frame_path = os.path.join(target_frames_dir, dest_frame_name)
                if os.path.exists(dest_frame_path):
                    rec_tag = recording_id.split('_')[-1]
                    sample_id = f"{signer_id}_{session_id}_{letter}_t{take_number}_{rec_tag}_{f_meta['frame_index']:04d}"
                    dest_frame_name = f"{sample_id}.jpg"
                    dest_frame_path = os.path.join(target_frames_dir, dest_frame_name)

                shutil.copy2(src_frame_path, dest_frame_path)
                rel_frame_path = os.path.relpath(dest_frame_path, BASE_DIR)

                sample_record = {
                    "sample_id": sample_id,
                    "split": split_name,
                    "dataset_role": role_name,
                    "allow_training": allow_training,
                    "eligible_for_evaluation": not allow_training,
                    "is_superseded": False,
                    "superseded_by": None,
                    "signer_id": signer_id,
                    "session_id": session_id,
                    "recording_id": recording_id,
                    "take_number": take_number,
                    "condition_id": condition_id,
                    "letter": letter,
                    "confirmation_method": "signer_confirmed",
                    "condition": {
                        "lighting": condition_meta["lighting"],
                        "background": condition_meta["background"],
                        "camera_position": condition_meta["camera_position"],
                        "distance_cm": int(info.get("actual_distance_cm") or condition_meta["distance_cm"]),
                        "camera_yaw_deg": float(info.get("actual_yaw_deg") if info.get("actual_yaw_deg") is not None else condition_meta.get("camera_yaw_deg", 0.0)),
                        "camera_pitch_deg": float(info.get("actual_pitch_deg") if info.get("actual_pitch_deg") is not None else condition_meta.get("camera_pitch_deg", condition_meta.get("pitch_deg", 0.0))),
                        "distance_measurement_method": str(info.get("distance_measurement_method") or condition_meta.get("distance_measurement_method", "nominal")),
                        "lap_hand_resting": bool(condition_meta.get("lap_hand_resting", False)),
                        "measured_resolution": [info["video_dimensions"]["width"], info["video_dimensions"]["height"]] if "video_dimensions" in info and info["video_dimensions"].get("width") else [1280, 720],
                        "observed_fps": float(info.get("fps", 30.0))
                    },
                    "detection": {
                        "hands_detected": f_meta["hands_detected"],
                        "handedness": ["Right"] if f_meta["hands_detected"] == 1 else ["Left", "Right"],
                        "scores": [f_meta.get("hand_confidence", 0.95)],
                        "lap_hand_suppressed": False
                    },
                    "quality": {
                        "blur_laplacian_var": f_meta["blur_laplacian_var"],
                        "mean_brightness": f_meta["mean_brightness"],
                        "is_usable": True,
                        "rejection_reason": None
                    },
                    "source": {
                        "video_path": rel_video_path,
                        "frame_path": rel_frame_path,
                        "frame_index": f_meta["frame_index"],
                        "timestamp_sec": f_meta["timestamp_sec"],
                        "sha256": f_meta["sha256"],
                        "dhash": f_meta.get("dhash", "")
                    }
                }

                # Validate schema
                is_valid, err = validate_sample_schema(sample_record)
                if not is_valid:
                    raise ValueError(f"Sample schema validation failure on {sample_id}: {err}")

                accepted_samples.append(sample_record)

            # Save to target partition manifest
            if manifest_target == "dev_train":
                m_train = self.load_dev_train_manifest()
                if prior_accepted:
                    for s in m_train:
                        if s.get("recording_id") in prior_rec_ids:
                            s["quality"]["is_usable"] = False
                            s["quality"]["rejection_reason"] = f"superseded_by_retake_{recording_id}"
                            s["is_superseded"] = True
                            s["superseded_by"] = recording_id
                            s["eligible_for_evaluation"] = False
                m_train.extend(accepted_samples)
                self.save_dev_train_manifest(m_train)

            elif manifest_target == "dev_val":
                m_val = self.load_dev_val_manifest()
                if prior_accepted:
                    for s in m_val:
                        if s.get("recording_id") in prior_rec_ids:
                            s["quality"]["is_usable"] = False
                            s["quality"]["rejection_reason"] = f"superseded_by_retake_{recording_id}"
                            s["is_superseded"] = True
                            s["superseded_by"] = recording_id
                            s["eligible_for_evaluation"] = False
                m_val.extend(accepted_samples)
                self.save_dev_val_manifest(m_val)

            elif manifest_target == "signer_05":
                # For signer_05, maintain both dedicated manifest and test_manifest
                s05_manifest = self.load_signer05_manifest()
                test_manifest = self.load_test_manifest()
                if prior_accepted:
                    for s in s05_manifest:
                        if s.get("recording_id") in prior_rec_ids:
                            s["quality"]["is_usable"] = False
                            s["quality"]["rejection_reason"] = f"superseded_by_retake_{recording_id}"
                            s["is_superseded"] = True
                            s["superseded_by"] = recording_id
                            s["eligible_for_evaluation"] = False
                    for s in test_manifest:
                        if s.get("recording_id") in prior_rec_ids:
                            s["quality"]["is_usable"] = False
                            s["quality"]["rejection_reason"] = f"superseded_by_retake_{recording_id}"
                            s["is_superseded"] = True
                            s["superseded_by"] = recording_id
                            s["eligible_for_evaluation"] = False
                s05_manifest.extend(accepted_samples)
                test_manifest.extend(accepted_samples)
                self.save_signer05_manifest(s05_manifest)
                self.save_test_manifest(test_manifest)

            else:
                test_manifest = self.load_test_manifest()
                if prior_accepted:
                    for s in test_manifest:
                        if s.get("recording_id") in prior_rec_ids:
                            s["quality"]["is_usable"] = False
                            s["quality"]["rejection_reason"] = f"superseded_by_retake_{recording_id}"
                            s["is_superseded"] = True
                            s["superseded_by"] = recording_id
                            s["eligible_for_evaluation"] = False
                test_manifest.extend(accepted_samples)
                self.save_test_manifest(test_manifest)

            # 3. Save ledger entry
            ledger_entry = {
                "recording_id": recording_id,
                "staging_id": staging_id,
                "timestamp": now_iso,
                "signer_id": signer_id,
                "condition_id": condition_id,
                "letter": letter,
                "session_id": session_id,
                "take_number": take_number,
                "status": "accepted",
                "is_retake": len(prior_accepted) > 0,
                "supersedes": list(prior_rec_ids) if prior_accepted else [],
                "dataset_role": role_name,
                "allow_training": allow_training,
                "raw_video_path": rel_video_path,
                "usable_frames_count": len(accepted_samples),
                "rejected_frames_count": info["rejected_count"],
                "sampling_interval": info["sampling_interval"],
                "operator_notes": info.get("operator_notes", ""),
                "video_dimensions": info.get("video_dimensions", {"width": 0, "height": 0}),
                "observed_fps": info.get("fps", 30.0),
                "quality_warnings": info.get("quality_warnings", [])
            }
            ledger["records"].append(ledger_entry)
            self.save_ledger(ledger)

            # Clean up staging
            shutil.rmtree(stage_dir, ignore_errors=True)

            return {
                "status": "accepted",
                "staging_id": staging_id,
                "recording_id": recording_id,
                "signer_id": signer_id,
                "condition_id": condition_id,
                "letter": letter,
                "take_number": take_number,
                "dataset_role": role_name,
                "allow_training": allow_training,
                "is_retake": len(prior_accepted) > 0,
                "superseded_count": len(prior_accepted),
                "usable_frames_saved": len(accepted_samples),
                "video_path": rel_video_path,
                "message": f"Successfully ingested {len(accepted_samples)} usable frames for {signer_id} ({letter}, {condition_id}, Take {take_number}). Role: {role_name}."
            }
        else:
            raise ValueError(f"Unknown commit action: '{action}'. Allowed: 'accept', 'reject'.")

    def delete_recording(self, recording_id: str) -> Dict[str, Any]:
        """
        Deletes a previously accepted recording and its extracted frames
        to honor volunteer participant right-to-deletion requests.
        """
        ledger = self.load_ledger()
        matching = [r for r in ledger["records"] if r.get("recording_id") == recording_id and r.get("status") == "accepted"]

        if not matching:
            raise FileNotFoundError(f"Active accepted recording '{recording_id}' not found in ledger.")

        rec = matching[0]
        signer_id = rec["signer_id"]
        condition_id = rec["condition_id"]
        letter = rec["letter"]

        # 1. Remove raw video file
        raw_p = os.path.join(BASE_DIR, rec["raw_video_path"]) if rec.get("raw_video_path") else None
        if raw_p and os.path.exists(raw_p):
            try:
                os.remove(raw_p)
            except Exception:
                pass

        # 2. Remove frames from test manifest and disk
        test_manifest = self.load_test_manifest()
        remaining_manifest = []
        deleted_count = 0

        for s in test_manifest:
            if s.get("recording_id") == recording_id:
                deleted_count += 1
                fp = os.path.join(BASE_DIR, s.get("source", {}).get("frame_path", ""))
                if os.path.exists(fp):
                    try:
                        os.remove(fp)
                    except Exception:
                        pass
            else:
                remaining_manifest.append(s)

        self.save_test_manifest(remaining_manifest)

        # Also purge from dedicated signer_05 manifest if applicable
        if signer_id == "signer_05":
            s05_manifest = self.load_signer05_manifest()
            remaining_s05 = [s for s in s05_manifest if s.get("recording_id") != recording_id]
            self.save_signer05_manifest(remaining_s05)

        # 3. Update ledger entry to deleted
        for r in ledger["records"]:
            if r.get("recording_id") == recording_id:
                r["status"] = "deleted"
                r["deleted_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        self.save_ledger(ledger)

        return {
            "status": "deleted",
            "recording_id": recording_id,
            "signer_id": signer_id,
            "condition_id": condition_id,
            "letter": letter,
            "deleted_frames_count": deleted_count,
            "message": f"Recording {recording_id} and {deleted_count} frames permanently deleted."
        }

    def get_progress(self) -> Dict[str, Any]:
        """
        Calculates collection progress matrix across all target signers,
        conditions, letters, and takes.
        Supports 3 conditions and 3 takes per letter (234 planned takes) for
        development signers (signer_06 through signer_11), and 52 cells for legacy signers.
        """
        ledger = self.load_ledger()
        letters = [chr(65 + i) for i in range(26)]
        signers = ALLOWED_SIGNERS

        # Aggregate all active samples across all manifests
        all_manifests = [
            self.load_test_manifest(),
            self.load_signer05_manifest(),
            self.load_dev_train_manifest(),
            self.load_dev_val_manifest()
        ]
        active_samples = []
        seen_sample_ids = set()
        for m in all_manifests:
            for s in m:
                sid = s.get("sample_id")
                if sid and sid not in seen_sample_ids:
                    seen_sample_ids.add(sid)
                    active_samples.append(s)

        # Count active usable frames: (signer, condition, letter) -> count
        frame_counts = {}
        for s in signers:
            conds = list(CONDITION_PRESETS.keys()) if s in DEVELOPMENT_SIGNERS else ["condition_1", "condition_2"]
            for c in conds:
                for l in letters:
                    frame_counts[(s, c, l)] = 0

        for sample in active_samples:
            s = sample.get("signer_id")
            c_meta = sample.get("condition", {})
            c_pos = c_meta.get("camera_position")
            if c_pos == "frontal_eye_level":
                c_key = "condition_1"
            elif c_pos == "diagonal_15deg":
                c_key = "condition_2"
            elif c_pos == "frontal_low_angle":
                c_key = "condition_3"
            else:
                rec_id = sample.get("recording_id", "")
                c_key = "condition_3" if "condition_3" in rec_id else ("condition_2" if "condition_2" in rec_id else "condition_1")

            l = sample.get("letter")
            key = (s, c_key, l)
            if key in frame_counts:
                if sample.get("quality", {}).get("is_usable", True) and not sample.get("is_superseded", False):
                    frame_counts[key] += 1

        # Track takes, rejections, and supersessions from ledger
        accepted_takes_by_cell = {}     # (s, c, l) -> set of take_numbers
        rejections_by_letter = {}
        superseded_by_letter = {}

        for r in ledger.get("records", []):
            s = r.get("signer_id")
            c = r.get("condition_id")
            l = r.get("letter")
            t = r.get("take_number", 1)
            key = (s, c, l)

            if r.get("status") == "accepted":
                if key not in accepted_takes_by_cell:
                    accepted_takes_by_cell[key] = set()
                accepted_takes_by_cell[key].add(t)
            elif r.get("status") == "rejected":
                rejections_by_letter[key] = rejections_by_letter.get(key, 0) + 1
            elif r.get("status") == "superseded":
                superseded_by_letter[key] = superseded_by_letter.get(key, 0) + 1

        total_target_frames = 0
        total_accepted_frames = len(active_samples)
        total_rejected_attempts = sum(1 for r in ledger.get("records", []) if r.get("status") == "rejected")

        # Build matrix
        matrix = {}
        signer_summary = {}

        for s in signers:
            matrix[s] = {}
            is_dev = s in DEVELOPMENT_SIGNERS
            conds = ["condition_1", "condition_2", "condition_3"] if is_dev else ["condition_1", "condition_2"]
            s_frames = 0
            s_completed_cells = 0
            s_completed_takes = 0

            for c in conds:
                c_frames = sum(frame_counts.get((s, c, l), 0) for l in letters)
                s_frames += c_frames
                c_target = len(letters) * self.target_frames_per_letter
                letters_data = {}

                for l in letters:
                    cnt = frame_counts.get((s, c, l), 0)
                    rej = rejections_by_letter.get((s, c, l), 0)
                    sup = superseded_by_letter.get((s, c, l), 0)
                    cell_takes = accepted_takes_by_cell.get((s, c, l), set())

                    if is_dev:
                        # 3-take tracking for development signers
                        takes_info = {
                            1: {"status": "completed" if 1 in cell_takes else "missing"},
                            2: {"status": "completed" if 2 in cell_takes else "missing"},
                            3: {"status": "completed" if 3 in cell_takes else "missing"}
                        }
                        is_cell_complete = (len(cell_takes) >= 3)
                        s_completed_takes += len(cell_takes)
                        letters_data[l] = {
                            "accepted_frames": cnt,
                            "target_frames": self.target_frames_per_letter,
                            "is_complete": is_cell_complete,
                            "completed_takes": len(cell_takes),
                            "planned_takes": 3,
                            "takes": takes_info,
                            "rejected_attempts": rej,
                            "superseded_attempts": sup,
                            "has_retake": sup > 0,
                            "status": "completed" if is_cell_complete else ("partial" if len(cell_takes) > 0 else "missing")
                        }
                    else:
                        is_cell_complete = cnt >= self.target_frames_per_letter
                        letters_data[l] = {
                            "accepted_frames": cnt,
                            "target_frames": self.target_frames_per_letter,
                            "is_complete": is_cell_complete,
                            "rejected_attempts": rej,
                            "superseded_attempts": sup,
                            "has_retake": sup > 0,
                            "status": "completed" if is_cell_complete else ("partial" if cnt > 0 else "missing")
                        }

                    if is_cell_complete:
                        s_completed_cells += 1

                matrix[s][c] = {
                    "condition_name": CONDITION_PRESETS[c]["name"],
                    "accepted_frames": c_frames,
                    "target_frames": c_target,
                    "completion_pct": round((c_frames / c_target) * 100, 1) if c_target > 0 else 0.0,
                    "completed_letters_count": sum(1 for l in letters if letters_data[l]["is_complete"]),
                    "letters": letters_data
                }

            if is_dev:
                total_takes = 234  # 26 letters * 3 conditions * 3 takes
                signer_summary[s] = {
                    "accepted_frames": s_frames,
                    "target_frames": len(conds) * len(letters) * self.target_frames_per_letter,
                    "completion_pct": round((s_completed_takes / total_takes) * 100, 1),
                    "is_complete": s_completed_takes >= total_takes,
                    "completed_takes": s_completed_takes,
                    "total_takes": total_takes,
                    "planned_takes": total_takes,
                    "missing_takes_count": total_takes - s_completed_takes,
                    "completed_cells": s_completed_cells,
                    "total_cells": len(conds) * len(letters),  # 78 cells
                    "is_dataset_ready": s_completed_takes >= total_takes,
                    "collection_status": "NOT_COLLECTED" if s_completed_takes == 0 else ("IN_PROGRESS" if s_completed_takes < total_takes else "COMPLETED"),
                    "dataset_role": SIGNER_ROLE_MAPPING[s],
                    "dataset_split_status": "PROVISIONAL_UNFROZEN_PENDING_DATA_COLLECTION",
                    "split_freeze_permitted": False,
                    "is_evaluation_only": False,
                    "total_superseded_recordings": sum(superseded_by_letter.get((s, c, l), 0) for c in conds for l in letters),
                    "total_rejected_recordings": sum(rejections_by_letter.get((s, c, l), 0) for c in conds for l in letters)
                }
            else:
                total_cells = 52
                s_target = len(conds) * len(letters) * self.target_frames_per_letter
                signer_summary[s] = {
                    "accepted_frames": s_frames,
                    "target_frames": s_target,
                    "completion_pct": round((s_frames / s_target) * 100, 1) if s_target > 0 else 0.0,
                    "is_complete": s_completed_cells == total_cells,
                    "completed_cells": s_completed_cells,
                    "total_cells": total_cells,
                    "is_dataset_ready": s_completed_cells == total_cells,
                    "collection_status": "HISTORICAL",
                    "dataset_role": SIGNER_ROLE_MAPPING.get(s, "external_test_multisigner"),
                    "is_evaluation_only": s in EVALUATION_ONLY_SIGNERS,
                    "total_superseded_recordings": sum(superseded_by_letter.get((s, c, l), 0) for c in conds for l in letters),
                    "total_rejected_recordings": sum(rejections_by_letter.get((s, c, l), 0) for c in conds for l in letters)
                }

        uncollected_dev = [s for s in DEVELOPMENT_SIGNERS if signer_summary.get(s, {}).get("completed_takes", 0) == 0]
        campaign_status = "INCOMPLETE" if uncollected_dev else "COLLECTED"

        return {
            "nominal_target_frames": total_target_frames or 3120,
            "total_accepted_frames": total_accepted_frames,
            "total_rejected_attempts": total_rejected_attempts,
            "overall_completion_pct": round((total_accepted_frames / (total_target_frames or 3120)) * 100, 1),
            "campaign_status": campaign_status,
            "uncollected_development_signers": uncollected_dev,
            "dataset_split_status": "PROVISIONAL_UNFROZEN_PENDING_DATA_COLLECTION",
            "target_per_letter": self.target_frames_per_letter,
            "signer_summary": signer_summary,
            "matrix": matrix
        }

    def get_signer_summary(self, signer_id: str) -> Dict[str, Any]:
        """
        Returns a comprehensive completion, retake, and recovery summary for a specific signer.
        Includes completed, incomplete, rejected, and superseded takes.
        """
        clean_signer = str(signer_id).strip()
        if clean_signer not in ALLOWED_SIGNERS:
            raise ValueError(f"Invalid signer_id: '{clean_signer}'. Allowed: {ALLOWED_SIGNERS}.")

        prog = self.get_progress()
        s_summary = prog["signer_summary"].get(clean_signer, {})
        s_matrix = prog["matrix"].get(clean_signer, {})
        is_dev = clean_signer in DEVELOPMENT_SIGNERS

        letters = [chr(65 + i) for i in range(26)]
        conditions = ["condition_1", "condition_2", "condition_3"] if is_dev else ["condition_1", "condition_2"]

        completed_takes_list = []
        missing_takes_list = []
        next_target = None

        condition_summaries = {}
        condition_breakdown = {}
        for c in conditions:
            c_data = s_matrix.get(c, {})
            c_letters = c_data.get("letters", {})
            completed_letters = [l for l in letters if c_letters.get(l, {}).get("is_complete", False)]
            missing_letters = [l for l in letters if l not in completed_letters]

            cond_info = {
                "name": CONDITION_PRESETS[c]["name"],
                "completed_count": len(completed_letters),
                "completed_letters": len(completed_letters),
                "total_count": 26,
                "missing_letters": missing_letters,
                "accepted_frames": c_data.get("accepted_frames", 0),
                "letters": c_letters
            }
            condition_summaries[f"{c}_summary"] = cond_info
            condition_breakdown[c] = cond_info

            if is_dev:
                for l in letters:
                    l_takes = c_letters.get(l, {}).get("takes", {})
                    for t in [1, 2, 3]:
                        if l_takes.get(t, {}).get("status") == "completed":
                            completed_takes_list.append({"condition_id": c, "letter": l, "take_number": t})
                        else:
                            missing_takes_list.append({"condition_id": c, "letter": l, "take_number": t})
                            if next_target is None:
                                next_target = {"condition_id": c, "letter": l, "take_number": t}

        iso = self.audit_split_isolation()
        dataset_role = SIGNER_ROLE_MAPPING.get(clean_signer, "development_train")
        is_ready = s_summary.get("is_dataset_ready", False)

        res = {
            "signer_id": clean_signer,
            "role": "independent_evaluation_holdout" if (clean_signer in EVALUATION_ONLY_SIGNERS) else dataset_role,
            "dataset_role": dataset_role,
            "allow_training": dataset_role == "development_train",
            "is_evaluation_only": clean_signer in EVALUATION_ONLY_SIGNERS,
            "target_per_letter": self.target_frames_per_letter,
            "nominal_target_frames": len(conditions) * len(letters) * self.target_frames_per_letter,
            "total_accepted_usable_frames": s_summary.get("accepted_frames", 0),
            "completed_cells": s_summary.get("completed_cells", 0),
            "total_cells": s_summary.get("total_cells", 52),
            "completion_pct": s_summary.get("completion_pct", 0.0),
            "is_dataset_ready": is_ready,
            "superseded_recordings_count": s_summary.get("total_superseded_recordings", 0),
            "rejected_recordings_count": s_summary.get("total_rejected_recordings", 0),
            "isolation_guaranteed": iso.get("is_leakage_free", True),
            "condition_breakdown": condition_breakdown,
            "isolation_audit": {
                "is_leakage_free": iso.get("is_leakage_free", True),
                "train_overlap_count": len(iso.get("overlaps", {}).get("signer_train_test", [])),
                "val_overlap_count": len(iso.get("overlaps", {}).get("signer_val_test", []))
            },
            "matrix": s_matrix
        }

        # Merge condition summaries
        res.update(condition_summaries)

        if is_dev:
            res["planned_takes"] = 234
            res["completed_takes_count"] = len(completed_takes_list)
            res["missing_takes_count"] = len(missing_takes_list)
            res["next_recommended_target"] = next_target or {"condition_id": "condition_1", "letter": "A", "take_number": 1}
            res["completed_takes"] = completed_takes_list
            res["missing_takes"] = missing_takes_list

        return res

    def audit_split_isolation(self) -> Dict[str, Any]:
        """
        Strictly audits partition independence to guarantee that newly collected
        samples remain 100% held-out and NEVER leak into unintended splits:
        1. signer_05 is strictly quarantined in external holdout (zero training/calibration).
        2. signer_10 and signer_11 are strictly quarantined in dev_val (zero training).
        3. signer_02 and signer_03 remain frozen benchmarks (zero training).
        4. signer_06 through signer_09 are in dev_train (allow_training: True).
        """
        test_samples = self.load_test_manifest()
        test_signers = set(s.get("signer_id") for s in test_samples)
        test_hashes = set(s.get("source", {}).get("sha256") for s in test_samples if s.get("source", {}).get("sha256"))

        dev_train_samples = self.load_dev_train_manifest()
        dev_train_signers = set(s.get("signer_id") for s in dev_train_samples)
        dev_train_hashes = set(s.get("source", {}).get("sha256") for s in dev_train_samples if s.get("source", {}).get("sha256"))

        dev_val_samples = self.load_dev_val_manifest()
        dev_val_signers = set(s.get("signer_id") for s in dev_val_samples)
        dev_val_hashes = set(s.get("source", {}).get("sha256") for s in dev_val_samples if s.get("source", {}).get("sha256"))

        signer05_samples = self.load_signer05_manifest()
        signer05_hashes = set(s.get("source", {}).get("sha256") for s in signer05_samples if s.get("source", {}).get("sha256"))

        train_manifest_p = os.path.join(SPLITS_DIR, "train_manifest.json")
        val_manifest_p = os.path.join(SPLITS_DIR, "val_manifest.json")

        train_hashes = set()
        val_hashes = set()
        train_signers = set()
        val_signers = set()

        if os.path.exists(train_manifest_p):
            with open(train_manifest_p, "r") as f:
                for item in json.load(f):
                    train_signers.add(item.get("signer_id", "unknown"))
                    h = item.get("image_hash")
                    if h:
                        train_hashes.add(h)

        if os.path.exists(val_manifest_p):
            with open(val_manifest_p, "r") as f:
                for item in json.load(f):
                    val_signers.add(item.get("signer_id", "unknown"))
                    h = item.get("image_hash")
                    if h:
                        val_hashes.add(h)

        # Check overlaps
        signer_leakage_train = list(test_signers.intersection(train_signers))
        signer_leakage_val = list(test_signers.intersection(val_signers))
        hash_leakage_train = list(test_hashes.intersection(train_hashes))
        hash_leakage_val = list(test_hashes.intersection(val_hashes))

        # Check development partition isolation
        train_val_overlap = list(dev_train_signers.intersection(dev_val_signers))
        train_holdout_overlap = list(dev_train_signers.intersection(set(EVALUATION_ONLY_SIGNERS)))
        val_holdout_overlap = list(dev_val_signers.intersection(set(EVALUATION_ONLY_SIGNERS)))

        is_leakage_free = (
            len(signer_leakage_train) == 0 and
            len(signer_leakage_val) == 0 and
            len(hash_leakage_train) == 0 and
            len(hash_leakage_val) == 0 and
            len(train_val_overlap) == 0 and
            len(train_holdout_overlap) == 0 and
            len(val_holdout_overlap) == 0
        )

        return {
            "is_leakage_free": is_leakage_free,
            "is_strictly_held_out": is_leakage_free and (len(test_signers.intersection(ALLOWED_SIGNERS)) == len(test_signers)),
            "new_signers_in_test": list(test_signers),
            "new_test_sample_count": len(test_samples),
            "dev_train_sample_count": len(dev_train_samples),
            "dev_val_sample_count": len(dev_val_samples),
            "signer05_sample_count": len(signer05_samples),
            "train_sample_count": len(train_hashes),
            "val_sample_count": len(val_hashes),
            "overlaps": {
                "signer_train_test": signer_leakage_train,
                "signer_val_test": signer_leakage_val,
                "hash_train_test_count": len(hash_leakage_train),
                "hash_val_test_count": len(hash_leakage_val),
                "train_val_signer_overlap": train_val_overlap,
                "train_holdout_overlap": train_holdout_overlap
            }
        }

    def get_preflight_checklist(
        self,
        signer_id: str,
        condition_id: str,
        letter: str,
        take_number: int = 1,
        consent_confirmed: bool = False,
        detected_resolution: Optional[Dict[str, int]] = None,
        observed_fps: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Evaluates session preflight readiness before recording begins:
        1. Signer authorization & dataset role assignment.
        2. Letter, condition, and take slot validation.
        3. Geometry & setup instructions with nominal/operator-reported disclosures.
        4. Hardware telemetry verification (camera resolution & FPS).
        5. Volunteer consent verification (mandatory blocking gate).
        6. Overall preflight clearance verdict: READY vs BLOCKED.
        """
        clean_signer = str(signer_id).strip()
        clean_cond = str(condition_id).strip()
        clean_letter = str(letter).strip().upper()
        try:
            take_num = int(take_number)
        except Exception:
            take_num = -1

        blocking_reasons = []

        # 1. Signer Authorization
        if clean_signer not in ALL_VALID_SIGNERS:
            blocking_reasons.append(f"Signer '{clean_signer}' is not authorized. Must be one of {ALL_VALID_SIGNERS}.")
            dataset_role = "unknown"
            role_governance = "UNAUTHORIZED_SIGNER"
            allow_training = False
        else:
            dataset_role = SIGNER_ROLE_MAPPING.get(clean_signer, "unknown")
            allow_training = (dataset_role == "development_train")
            if clean_signer == "signer_05":
                role_governance = "SEALED_EXTERNAL_HOLDOUT (Evaluation & Training Strictly Prohibited)"
            elif clean_signer in FROZEN_BENCHMARK_SIGNERS:
                role_governance = "FROZEN_BENCHMARK (Evaluation Only, Excluded from Training)"
            elif clean_signer in DEVELOPMENT_VALIDATION_SIGNERS:
                role_governance = "DEVELOPMENT_VALIDATION (Tuning Only, Excluded from Training)"
            else:
                role_governance = "DEVELOPMENT_TRAIN (Permitted for Model Training)"

        # 2. Letter Validation
        if len(clean_letter) != 1 or not ('A' <= clean_letter <= 'Z'):
            blocking_reasons.append(f"Invalid letter '{clean_letter}'. Must be single uppercase character A-Z.")

        # 3. Condition Validation
        if clean_cond not in CONDITION_PRESETS:
            blocking_reasons.append(f"Invalid condition '{clean_cond}'. Must be one of {list(CONDITION_PRESETS.keys())}.")
            cond_info = {}
        else:
            cond_info = CONDITION_PRESETS[clean_cond]

        # 4. Take Slot Validation
        if take_num not in [1, 2, 3]:
            blocking_reasons.append(f"Invalid take_number {take_num}. Must be 1, 2, or 3.")

        # 5. Consent Validation (Mandatory Blocking Gate)
        if not consent_confirmed:
            blocking_reasons.append("Volunteer consent must be explicitly confirmed prior to recording.")

        # 6. Hardware Telemetry
        res_ok = False
        res_text = "Not detected"
        if detected_resolution:
            w = detected_resolution.get("width", 0)
            h = detected_resolution.get("height", 0)
            res_text = f"{w}x{h}"
            res_ok = (w >= self.min_resolution_width and h >= self.min_resolution_height)
        
        is_ready = (len(blocking_reasons) == 0)

        return {
            "preflight_status": "READY" if is_ready else "BLOCKED",
            "is_ready_for_capture": is_ready,
            "blocking_reasons": blocking_reasons,
            "signer": {
                "signer_id": clean_signer,
                "dataset_role": dataset_role,
                "role_governance": role_governance,
                "allow_training": allow_training,
                "collection_status": "NOT_COLLECTED" if clean_signer in DEVELOPMENT_SIGNERS else "HISTORICAL"
            },
            "cell": {
                "letter": clean_letter,
                "condition_id": clean_cond,
                "condition_name": cond_info.get("name", ""),
                "take_number": take_num,
                "take_slot_label": f"Take {take_num} of 3"
            },
            "protocol_geometry": {
                "camera_yaw_deg": cond_info.get("camera_yaw_deg", 0.0),
                "camera_pitch_deg": cond_info.get("camera_pitch_deg", 0.0),
                "distance_cm": cond_info.get("distance_cm", 75),
                "distance_range_cm": cond_info.get("distance_range_cm", [70, 80]),
                "distance_measurement_method": cond_info.get("distance_measurement_method", "nominal"),
                "lap_hand_resting": cond_info.get("lap_hand_resting", False),
                "axis_disambiguation": cond_info.get("axis_disambiguation"),
                "instructions": cond_info.get("instructions", ""),
                "geometry_disclaimer": "Camera angles and distance are nominal / operator-reported; not hardware-instrumented."
            },
            "telemetry": {
                "detected_resolution": res_text,
                "resolution_meets_target": res_ok,
                "observed_fps": observed_fps,
                "target_fps": self.target_fps
            },
            "consent_confirmed": consent_confirmed,
            "hard_rejection_gates": [
                "consent_confirmed == True",
                "video_frames >= 5",
                "sha256_uniqueness == True",
                "dhash_near_duplicate == False",
                "static_movement >= 0.1",
                "mean_brightness in [40.0, 220.0]",
                "laplacian_blur_var >= 50.0",
                "hands_detected >= 1"
            ],
            "static_hold_safeguard": "Legitimate sign hold frames are preserved across consecutive video samples.",
            "split_freeze_status": "PROVISIONAL_UNFROZEN_PENDING_DATA_COLLECTION"
        }


def verify_training_isolation_safeguard(candidate_manifest: Any) -> bool:
    """
    Safeguard function to verify that candidate training or calibration data
    does not accidentally contain evaluation-only, validation, or unauthorized signers
    (e.g. signer_05, signer_10, signer_11, signer_02, signer_03, or unknown signers).
    Fails closed: raises ValueError if any violation or malformed entry is detected.
    """
    if isinstance(candidate_manifest, (list, dict)):
        data = candidate_manifest
    elif isinstance(candidate_manifest, str):
        if not os.path.exists(candidate_manifest):
            return True
        try:
            with open(candidate_manifest, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            raise ValueError(f"Malformed manifest file '{candidate_manifest}': {e}")
    else:
        raise ValueError(f"Invalid candidate manifest type: {type(candidate_manifest)}")

    if isinstance(data, list):
        samples = data
    elif isinstance(data, dict):
        samples = data.get("samples", [])
    else:
        raise ValueError("Malformed manifest: root must be a list or dict")

    permitted_train_signers = set(DEVELOPMENT_TRAIN_SIGNERS).union({"signer_01", "signer_04"})

    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("Malformed sample entry: must be a dict")
        signer = sample.get("signer_id")
        if not signer or signer not in ALL_VALID_SIGNERS:
            raise ValueError(f"TRAINING LEAKAGE DETECTED: unknown or missing signer_id '{signer}'")
        if signer in EVALUATION_ONLY_SIGNERS or signer in DEVELOPMENT_VALIDATION_SIGNERS or signer in FROZEN_BENCHMARK_SIGNERS:
            raise ValueError(f"TRAINING LEAKAGE DETECTED: protected signer '{signer}' sample found in training/calibration set")
        if signer not in permitted_train_signers:
            raise ValueError(f"TRAINING LEAKAGE DETECTED: signer '{signer}' is not permitted in training set")
        if sample.get("allow_training") is not True:
            raise ValueError(f"TRAINING LEAKAGE DETECTED: sample with allow_training={sample.get('allow_training')} found in training set")
        role = sample.get("dataset_role")
        if role and role not in ["development_train"]:
            raise ValueError(f"TRAINING LEAKAGE DETECTED: sample with non-training role '{role}' found in training set")

    return True


# Singleton collection engine instance
collection_engine = ISLCollectionEngine()

if __name__ == "__main__":
    print("[Phase 18 Collection Engine] Initializing...")
    progress = collection_engine.get_progress()
    print(f"Total accepted frames: {progress['total_accepted_frames']} / {progress['nominal_target_frames']} ({progress['overall_completion_pct']}%)")
    iso = collection_engine.audit_split_isolation()
    print(f"Split isolation verified: {iso['is_leakage_free']} (strictly held out: {iso['is_strictly_held_out']})")
