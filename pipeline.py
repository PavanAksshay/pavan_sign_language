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
        self.yolo.confidence = 0.20
        self.yolo.threshold = 0.20

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

            # Determine whether to use primary hand or merge two hands:
            # In true 2-handed ISL gestures (e.g. A, B, P), both hands interact close together in the center.
            # If detections are far apart (e.g. active hand on left + resting arm on right),
            # merging them creates an absurd box across the entire torso!
            is_two_handed = False
            if len(detections) == 1:
                _, _, _, bx, by, bw, bh = detections[0]
                box_coords = (bx, by, bx + bw, by + bh)
            else:
                det1 = detections[0]  # highest confidence
                det2 = detections[1]  # second highest
                b1_x, b1_y, b1_w, b1_h = det1[3], det1[4], det1[5], det1[6]
                b2_x, b2_y, b2_w, b2_h = det2[3], det2[4], det2[5], det2[6]

                # Check horizontal gap between the two detections
                left_box = det1 if b1_x < b2_x else det2
                right_box = det2 if b1_x < b2_x else det1
                gap_x = right_box[3] - (left_box[3] + left_box[5])

                # Hands interacting in 2-handed signs must be close together (gap < 1.2 * hand width)
                max_hand_w = max(b1_w, b2_w)
                is_close_interaction = (gap_x < max_hand_w * 1.2)
                vertical_overlap = min(b1_y + b1_h, b2_y + b2_h) - max(b1_y, b2_y)
                is_vertically_aligned = (vertical_overlap > -max(b1_h, b2_h) * 0.6)

                if is_close_interaction and is_vertically_aligned and det2[2] >= 0.30:
                    # Genuine 2-handed gesture (touching or close)
                    is_two_handed = True
                    x = min(det1[3], det2[3])
                    y = min(det1[4], det2[4])
                    x1 = max(det1[3] + det1[5], det2[3] + det2[5])
                    y1 = max(det1[4] + det1[6], det2[4] + det2[6])
                    box_coords = (x, y, x1, y1)
                else:
                    # Single active hand gesture: use primary detection, ignore resting elbow/background
                    box_coords = (b1_x, b1_y, b1_x + b1_w, b1_y + b1_h)
            self._last_is_two_handed = is_two_handed
        else:
            self._last_is_two_handed = False

        # 2. Early return if no hand detected: avoid running SqueezeNet on background/center crops
        if not hand_detected or box_coords is None:
            t_total_ms = round((time.time() - t_start) * 1000, 1)
            # Center crop for stage visualizer
            side = min(w, h)
            sx = (w - side) // 2
            sy = (h - side) // 2
            c_crop = cv2.resize(frame_bgr[sy:sy + side, sx:sx + side], (224, 224))
            c_seg = skinDetector(c_crop).find_skin()
            res = {
                "predicted_sign": "-",
                "confidence": 0.0,
                "hand_detected": False,
                "is_two_handed": False,
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
                    "cropped": self._encode_jpeg(c_crop),
                    "segmented": self._encode_jpeg(c_seg)
                }
            return res

        # 3. Crop Hand Region
        # Crucial for recognition accuracy:
        # We apply clean proportional margins around the detected hand bounding box.
        x, y, x1, y1 = box_coords
        box_w = x1 - x
        box_h = y1 - y

        pad_x = max(10, int(box_w * 0.10))
        pad_y = max(10, int(box_h * 0.10))

        crop_x = max(0, x - pad_x)
        crop_y = max(0, y - pad_y)
        crop_x1 = min(w, x1 + pad_x)
        crop_y1 = min(h, y1 + pad_y)

        # Detect face to ensure hand crop does not inadvertently include the user's face
        has_face, face_boxes = self.detect_face(frame_bgr)
        if has_face and len(face_boxes) > 0:
            for fb in face_boxes:
                f_bottom = fb["y"] + fb["h"]
                if f_bottom <= y1 and crop_y < f_bottom:
                    crop_y = max(0, min(y, f_bottom))

        cropped = frame_bgr[crop_y:crop_y1, crop_x:crop_x1]
        if cropped.size == 0 or cropped.shape[0] < 10 or cropped.shape[1] < 10:
            cropped = frame_bgr

        # Aspect-Ratio Preserving Letterbox Resize:
        # Non-uniform stretching deforms curved hand gestures (e.g. C) into vertical sticks (V/S).
        # Letterboxing with black canvas preserves exact gesture geometry 100%.
        ch, cw = cropped.shape[:2]
        scale = 224.0 / max(cw, ch)
        nw = max(1, int(cw * scale))
        nh = max(1, int(ch * scale))
        scaled = cv2.resize(cropped, (nw, nh), interpolation=cv2.INTER_LINEAR)

        resized = np.zeros((224, 224, 3), dtype=np.uint8)
        dx = (224 - nw) // 2
        dy = (224 - nh) // 2
        resized[dy:dy+nh, dx:dx+nw] = scaled

        # Debug: save periodic crops for verification
        self._debug_frame_count = getattr(self, '_debug_frame_count', 0) + 1
        if self._debug_frame_count % 30 == 0:
            debug_dir = '/tmp/isl_debug'
            os.makedirs(debug_dir, exist_ok=True)
            idx = (self._debug_frame_count // 30) % 10
            cv2.imwrite(os.path.join(debug_dir, f'crop_{idx}.jpg'), resized)



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
            "is_two_handed": getattr(self, '_last_is_two_handed', False),
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
