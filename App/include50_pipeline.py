import os
import json
import base64
import numpy as np
import torch
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from App.include50_model.model_def import INCLUDE50LSTM

LABEL_DISPLAY_MAP = {
    "bank": "Bank",
    "biglarge": "Big / Large",
    "bird": "Bird",
    "black": "Black",
    "boy": "Boy",
    "brother": "Brother",
    "car": "Car",
    "cellphone": "Cell Phone",
    "court": "Court",
    "cow": "Cow",
    "death": "Death",
    "dog": "Dog",
    "dry": "Dry",
    "election": "Election",
    "fall": "Fall",
    "fan": "Fan",
    "father": "Father",
    "girl": "Girl",
    "good": "Good",
    "goodmorning": "Good Morning",
    "happy": "Happy",
    "hat": "Hat",
    "hello": "Hello",
    "hot": "Hot",
    "house": "House",
    "i": "I / Me",
    "it": "It",
    "long": "Long",
    "loud": "Loud",
    "monday": "Monday",
    "new": "New",
    "paint": "Paint",
    "pen": "Pen",
    "priest": "Priest",
    "quiet": "Quiet",
    "red": "Red",
    "shoes": "Shoes",
    "short": "Short",
    "smalllittle": "Small / Little",
    "storeorshop": "Store / Shop",
    "summer": "Summer",
    "teacher": "Teacher",
    "thankyou": "Thank You",
    "time": "Time",
    "trainticket": "Train Ticket",
    "tshirt": "T-Shirt",
    "white": "White",
    "window": "Window",
    "year": "Year",
    "youplural": "You (Plural)"
}

_CATALOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "include50_model", "word_catalog.json")
if os.path.exists(_CATALOG_PATH):
    with open(_CATALOG_PATH, "r") as _cf:
        WORD_CATALOG = json.load(_cf)
else:
    WORD_CATALOG = [
        {"id": "hello", "word": "Hello", "category": "Greetings & Polite"},
        {"id": "goodmorning", "word": "Good Morning", "category": "Greetings & Polite"},
        {"id": "thankyou", "word": "Thank You", "category": "Greetings & Polite"},
        {"id": "good", "word": "Good", "category": "Feelings & States"},
        {"id": "happy", "word": "Happy", "category": "Feelings & States"}
    ]

# Standard 21-point hand skeleton connection lines
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Wrist palm base
]

# Upper body pose connections (0 to 24)
POSE_CONNECTIONS = [
    (11, 12), # shoulders
    (11, 13), (13, 15), # left arm
    (12, 14), (14, 16), # right arm
    (11, 23), (12, 24), (23, 24) # torso
]

class INCLUDE50Pipeline:
    def __init__(self, max_buffer_size=32):
        self.model = None
        self.label_map = {}
        self.inv_label_map = {}
        self.display_map = LABEL_DISPLAY_MAP
        self.hand_detector = None
        self.pose_detector = None
        self.initialized = False
        self.max_buffer_size = max_buffer_size
        self.frame_buffer = []

    def initialize(self):
        if self.initialized:
            return

        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, "include50_model")
        weights_path = os.path.join(model_dir, "include50_lstm.pth")
        labels_path = os.path.join(model_dir, "label_map_include50.json")

        hand_task = os.path.join(model_dir, "hand_landmarker.task")
        pose_task = os.path.join(model_dir, "pose_landmarker.task")

        with open(labels_path, "r") as f:
            self.label_map = json.load(f)
            self.inv_label_map = {int(v): k for k, v in self.label_map.items()}

        self.model = INCLUDE50LSTM(
            input_size=134,
            hidden_size=256,
            num_layers=5,
            bidirectional=True,
            dropout=0.2,
            n_classes=len(self.label_map)
        )

        if os.path.exists(weights_path):
            ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
            state_dict = ckpt["model"] if "model" in ckpt else ckpt
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print(f"[INCLUDE-50] Model loaded successfully with {len(self.label_map)} word classes!")

        if os.path.exists(hand_task) and os.path.exists(pose_task):
            base_opt_h = python.BaseOptions(model_asset_path=hand_task)
            opt_h = vision.HandLandmarkerOptions(base_options=base_opt_h, num_hands=2)
            self.hand_detector = vision.HandLandmarker.create_from_options(opt_h)

            base_opt_p = python.BaseOptions(model_asset_path=pose_task)
            opt_p = vision.PoseLandmarkerOptions(base_options=base_opt_p)
            self.pose_detector = vision.PoseLandmarker.create_from_options(opt_p)
            print("[INCLUDE-50] MediaPipe Hand & Pose Landmarkers initialized!")

        self.initialized = True

    def reset_buffer(self):
        self.frame_buffer.clear()

    def extract_features(self, frame_bgr, draw_landmarks=False):
        """
        Extracts 134-dimensional feature vector (50 pose + 42 hand1 + 42 hand2).
        Optionally draws landmark overlays onto frame_bgr if requested.
        """
        if self.hand_detector is None or self.pose_detector is None:
            return np.zeros(134, dtype=np.float32), 0, False, None

        h, w, _ = frame_bgr.shape
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        hand_results = self.hand_detector.detect(mp_img)
        pose_results = self.pose_detector.detect(mp_img)

        # Pose upper body (25 keypoints x, y = 50 features)
        pose_xy = []
        pose_detected = False
        pose_coords = []
        if pose_results.pose_landmarks and len(pose_results.pose_landmarks) > 0:
            pose_detected = True
            for lm in pose_results.pose_landmarks[0][:25]:
                pose_xy.extend([lm.x, lm.y])
                if draw_landmarks:
                    pose_coords.append((int(lm.x * w), int(lm.y * h)))

        if len(pose_xy) < 50:
            pose_xy.extend([0.0] * (50 - len(pose_xy)))

        # Hand 1 and Hand 2 (21 keypoints x, y each = 42 + 42 = 84 features)
        h1_xy = [0.0] * 42
        h2_xy = [0.0] * 42
        hands_detected = 0
        hand_coords_list = []

        if hand_results.hand_landmarks:
            hands_detected = len(hand_results.hand_landmarks)
            if hands_detected > 0:
                h1_xy = []
                coords1 = []
                for lm in hand_results.hand_landmarks[0]:
                    h1_xy.extend([lm.x, lm.y])
                    if draw_landmarks:
                        coords1.append((int(lm.x * w), int(lm.y * h)))
                hand_coords_list.append(coords1)

            if hands_detected > 1:
                h2_xy = []
                coords2 = []
                for lm in hand_results.hand_landmarks[1]:
                    h2_xy.extend([lm.x, lm.y])
                    if draw_landmarks:
                        coords2.append((int(lm.x * w), int(lm.y * h)))
                hand_coords_list.append(coords2)

        annotated_b64 = None
        if draw_landmarks:
            annotated = frame_bgr.copy()
            # Draw pose connections
            if len(pose_coords) >= 25:
                for p1, p2 in POSE_CONNECTIONS:
                    if p1 < len(pose_coords) and p2 < len(pose_coords):
                        cv2.line(annotated, pose_coords[p1], pose_coords[p2], (220, 220, 220), 2)
                for pt in pose_coords[:25]:
                    cv2.circle(annotated, pt, 3, (0, 0, 0), -1)

            # Draw hand connections
            for coords in hand_coords_list:
                for c1, c2 in HAND_CONNECTIONS:
                    if c1 < len(coords) and c2 < len(coords):
                        cv2.line(annotated, coords[c1], coords[c2], (0, 0, 0), 2)
                for pt in coords:
                    cv2.circle(annotated, pt, 4, (255, 255, 255), -1)
                    cv2.circle(annotated, pt, 4, (0, 0, 0), 1)

            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

        vec = np.array(pose_xy[:50] + h1_xy[:42] + h2_xy[:42], dtype=np.float32)
        return vec, hands_detected, pose_detected, annotated_b64

    def predict_sequence(self, sequence_vectors):
        """Given sequence array of shape (N, 134), runs LSTM inference."""
        if self.model is None or len(sequence_vectors) == 0:
            return None

        if isinstance(sequence_vectors, list):
            sequence_vectors = np.array(sequence_vectors, dtype=np.float32)
        inp = torch.from_numpy(sequence_vectors).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(inp)
            probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()

        top_idx = int(np.argmax(probs))
        top_conf = float(probs[top_idx]) * 100.0
        raw_word = self.inv_label_map.get(top_idx, "Unknown")
        display_word = self.display_map.get(raw_word, raw_word.title())

        top_indices = np.argsort(probs)[::-1][:5]
        top_words = [
            {
                "raw_word": self.inv_label_map.get(int(i), ""),
                "display_word": self.display_map.get(self.inv_label_map.get(int(i), ""), self.inv_label_map.get(int(i), "").title()),
                "confidence": round(float(probs[i]) * 100.0, 1)
            }
            for i in top_indices
        ]

        return {
            "predicted_word": raw_word,
            "display_word": display_word,
            "confidence": round(top_conf, 1),
            "top_words": top_words
        }

    def process_frame(self, frame_bgr, add_to_buffer=True, return_annotated=False):
        """
        Processes a single streaming frame:
        - Extracts 134-d landmarks
        - Appends to rolling window buffer when hands are present
        - Runs prediction if buffer has sufficient context (>= 4 frames)
        - Emits completed gesture if user finishes and lowers hands
        """
        vec, hands_count, pose_detected, annotated_b64 = self.extract_features(
            frame_bgr, draw_landmarks=return_annotated
        )

        gesture_completed = False
        pred_res = None

        if hands_count > 0:
            if add_to_buffer:
                self.frame_buffer.append(vec)
                if len(self.frame_buffer) > self.max_buffer_size:
                    self.frame_buffer.pop(0)

            # Live prediction during signing once we have at least 4 frames
            if len(self.frame_buffer) >= 4:
                pred_res = self.predict_sequence(self.frame_buffer)
        else:
            # Hands dropped / not detected
            # If we had accumulated gesture frames (>= 4 frames), sign just finished!
            if len(self.frame_buffer) >= 4:
                pred_res = self.predict_sequence(self.frame_buffer)
                gesture_completed = True
            # Clear buffer after gesture completion or idle
            self.frame_buffer.clear()

        return {
            "hand_detected": hands_count > 0,
            "hands_count": hands_count,
            "pose_detected": pose_detected,
            "buffer_length": len(self.frame_buffer),
            "prediction": pred_res,
            "gesture_completed": gesture_completed,
            "annotated_image": annotated_b64
        }

    def process_clip(self, frames_list):
        """
        Processes a whole batch of frames (e.g. 25-40 frames of a sign gesture).
        """
        if not frames_list:
            return None

        vectors = []
        for f in frames_list:
            vec, _, _, _ = self.extract_features(f, draw_landmarks=False)
            vectors.append(vec)

        return self.predict_sequence(vectors)

word_pipeline_instance = INCLUDE50Pipeline()
