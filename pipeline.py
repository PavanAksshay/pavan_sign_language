import os
import sys
import time
import json
import base64
import cv2
import numpy as np

# Ensure App directory is in python path
APP_DIR = os.path.join(os.path.dirname(__file__), 'App')
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from yolo import YOLO
from preprocessing import skinDetector, resize
import tf_keras

class ISLPipeline:
    def __init__(self):
        self.yolo = None
        self.model = None
        self.class_map = {}
        self.face_cascade = None
        self.initialized = False

    def initialize(self):
        if self.initialized:
            return

        cfg_path = os.path.join(APP_DIR, 'yolo_models', 'cross-hands.cfg')
        weights_path = os.path.join(APP_DIR, 'yolo_models', 'cross-hands.weights')
        if not os.path.exists(weights_path) or os.path.getsize(weights_path) < 1000000:
            print('[Pipeline] cross-hands.weights not found. Auto-downloading (~246MB)...')
            import urllib.request
            weights_url = 'https://github.com/cansik/yolo-hand-detection/releases/download/pretrained/cross-hands.weights'
            urllib.request.urlretrieve(weights_url, weights_path)
            print('[Pipeline] Weights downloaded successfully!')
        # Load 26-class model if available, otherwise fallback to 10-class model
        model_26_path = os.path.join(APP_DIR, 'final_model_26')
        class_26_json_path = os.path.join(APP_DIR, 'model_class_26.json')
        if os.path.exists(model_26_path) and os.path.exists(class_26_json_path):
            model_path = model_26_path
            class_json_path = class_26_json_path
        else:
            model_path = os.path.join(APP_DIR, 'final_model')
            class_json_path = os.path.join(APP_DIR, 'model_class.json')

        print(f"[Pipeline] Loading YOLO from {cfg_path}...")
        self.yolo = YOLO(cfg_path, weights_path, ['hand'])
        self.yolo.confidence = 0.25
        self.yolo.threshold = 0.25

        print(f"[Pipeline] Loading SqueezeNet model from {model_path}...")
        self.model = tf_keras.models.load_model(model_path)

        with open(class_json_path, 'r') as f:
            self.class_map = json.load(f)

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        # Warm up SqueezeNet
        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        _ = self.model.predict(dummy, verbose=0)

        self.initialized = True
        print("[Pipeline] All models loaded & initialized successfully!")

    def detect_face(self, frame_bgr):
        if self.face_cascade is None:
            return False, []
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
        face_boxes = []
        for (x, y, w, h) in faces:
            face_boxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})
        return len(face_boxes) > 0, face_boxes

    def process_frame(self, frame_bgr, return_images=True):
        if not self.initialized:
            self.initialize()

        t_start = time.time()
        h, w = frame_bgr.shape[:2]

        # 1. Hand detection with YOLO
        t_yolo_start = time.time()
        iw, ih, inference_time, detections = self.yolo.inference(frame_bgr)
        t_yolo_end = time.time()

        annotated_frame = frame_bgr.copy()
        hand_detected = False
        box_coords = None
        box_info = []

        if len(detections) > 0:
            hand_detected = True
            # Sort detections by confidence
            detections = sorted(detections, key=lambda d: d[2], reverse=True)
            for det in detections:
                _, name, conf, bx, by, bw, bh = det
                box_info.append({
                    "name": name,
                    "confidence": round(float(conf) * 100, 1),
                    "x": int(bx), "y": int(by), "w": int(bw), "h": int(bh)
                })
                cv2.rectangle(annotated_frame, (bx, by), (bx + bw, by + bh), (0, 255, 128), 2)
                cv2.putText(annotated_frame, f"{name}: {conf*100:.1f}%", (bx, max(20, by - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 128), 2)

            # Use primary detection (or merged bbox if multiple)
            if len(detections) == 1:
                _, _, _, bx, by, bw, bh = detections[0]
                x, y, x1, y1 = bx, by, bx + bw, by + bh
            else:
                x = min(d[3] for d in detections)
                y = min(d[4] for d in detections)
                x1 = max(d[3] + d[5] for d in detections)
                y1 = max(d[4] + d[6] for d in detections)
            box_coords = (x, y, x1, y1)

        # 2. Early return if no hand detected: avoid running SqueezeNet on background/center crops
        if not hand_detected or box_coords is None:
            t_total_ms = round((time.time() - t_start) * 1000, 1)
            res = {
                "predicted_sign": "-",
                "confidence": 0.0,
                "hand_detected": False,
                "detections": [],
                "probabilities": [],
                "timing_ms": {
                    "yolo": round((t_yolo_end - t_yolo_start) * 1000, 1),
                    "skin_segmentation": 0.0,
                    "squeezenet": 0.0,
                    "total": t_total_ms
                }
            }
            if return_images:
                res["images"] = {
                    "original": self._encode_jpeg(frame_bgr),
                    "annotated": self._encode_jpeg(annotated_frame),
                    "cropped": "",
                    "segmented": ""
                }
            return res

        # 3. Crop Hand with bounding box padding
        x, y, x1, y1 = box_coords
        pad_x, pad_y = 40, 25
        x = max(0, x - pad_x)
        y = max(0, y - pad_y)
        x1 = min(w, x1 + pad_x)
        y1 = min(h, y1 + pad_y)
        cropped = frame_bgr[y:y1, x:x1]
        if cropped.size == 0 or cropped.shape[0] < 10 or cropped.shape[1] < 10:
            cropped = frame_bgr

        resized = cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_LINEAR)

        # 4. Skin Segmentation with fallback to raw RGB if mask is empty
        t_skin_start = time.time()
        detector = skinDetector(resized)
        segmented = detector.find_skin()
        t_skin_end = time.time()

        # The model was trained on natural RGB hand crops (via tf.keras image_dataset_from_directory).
        # OpenCV frames are in BGR format; converting BGR to RGB restores correct color channels
        # and eliminates the channel-swap distortion that caused predictions to collapse into V.
        inp_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # 5. SqueezeNet Sign Recognition
        t_nn_start = time.time()
        inp = (inp_rgb.astype(np.float32) / 255.0)
        inp = np.expand_dims(inp, axis=0)

        preds = self.model.predict(inp, verbose=0)[0]
        t_nn_end = time.time()

        # Build results
        top_idx = int(np.argmax(preds))
        predicted_sign = self.class_map.get(str(top_idx), f"Class {top_idx}")
        top_confidence = round(float(preds[top_idx]) * 100, 2)

        # Probabilities breakdown for all classes
        prob_breakdown = []
        for idx_str, sign_name in self.class_map.items():
            idx = int(idx_str)
            prob_breakdown.append({
                "class_id": idx,
                "sign": sign_name,
                "probability": round(float(preds[idx]) * 100, 2)
            })
        prob_breakdown = sorted(prob_breakdown, key=lambda x: x["probability"], reverse=True)

        total_time_ms = round((time.time() - t_start) * 1000, 1)

        result = {
            "predicted_sign": predicted_sign,
            "confidence": top_confidence,
            "hand_detected": hand_detected,
            "detections": box_info,
            "probabilities": prob_breakdown,
            "timing_ms": {
                "yolo": round((t_yolo_end - t_yolo_start) * 1000, 1),
                "skin_segmentation": round((t_skin_end - t_skin_start) * 1000, 1),
                "squeezenet": round((t_nn_end - t_nn_start) * 1000, 1),
                "total": total_time_ms
            }
        }

        if return_images:
            result["images"] = {
                "original": self._encode_jpeg(frame_bgr),
                "annotated": self._encode_jpeg(annotated_frame),
                "cropped": self._encode_jpeg(resized),
                "segmented": self._encode_jpeg(segmented)
            }

        return result

    def _encode_jpeg(self, img_bgr):
        _, buffer = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')

# Global singleton
pipeline_instance = ISLPipeline()
