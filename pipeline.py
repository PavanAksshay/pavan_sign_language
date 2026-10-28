import os
import sys
import time
import json
import base64
import queue
import threading
import cv2
import numpy as np

# Ensure App directory is in python path
APP_DIR = os.path.join(os.path.dirname(__file__), 'App')
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from yolo import YOLO
from preprocessing import skinDetector, resize
import tf_keras

class ShadowWorker:
    """
    Phase 25: Asynchronous, thread-safe background worker for Model C shadow-mode evaluation.
    Enforces a strict bounded queue (maxsize=1) with freshest-frame priority policy.
    Maintains detailed timing telemetry: queue wait time, shadow execution latency,
    frame age, and explicit dropped-frame accounting.
    """
    def __init__(self, pipeline_ref):
        self.pipeline = pipeline_ref
        self.queue = queue.Queue(maxsize=1)
        self.running = False
        self.thread = None
        self.submitted_count = 0
        self.processed_count = 0
        self.dropped_count = 0
        self.error_count = 0
        self.last_telemetry = None
        self._lock = threading.Lock()

    def start(self):
        with self._lock:
            if self.running:
                return
            self.running = True
            self.thread = threading.Thread(target=self._worker_loop, daemon=True, name="ISL-ShadowWorker")
            self.thread.start()

    def stop(self, timeout=5.0):
        with self._lock:
            if not self.running:
                return
            self.running = False
        try:
            self.queue.put_nowait(None)  # Sentinel to unblock queue.get
        except queue.Full:
            pass
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=timeout)

    def submit_task(self, frame_bgr, prod_res, prod_start_time, prod_finish_time):
        """
        Submits task to shadow queue in <0.1ms without blocking production response.
        If queue is full (shadow worker busy), drops oldest pending frame so worker
        always processes the freshest available frame.
        """
        if not self.running:
            self.start()

        submit_time = time.time()
        task = {
            "frame_bgr": frame_bgr.copy(),
            "production_prediction": prod_res.get("predicted_sign", "-"),
            "production_confidence": prod_res.get("confidence", 0.0),
            "prod_start_time": prod_start_time,
            "prod_finish_time": prod_finish_time,
            "submit_time": submit_time
        }
        with self._lock:
            self.submitted_count += 1
            try:
                self.queue.put_nowait(task)
            except queue.Full:
                try:
                    _ = self.queue.get_nowait()
                    self.dropped_count += 1
                except queue.Empty:
                    pass
                try:
                    self.queue.put_nowait(task)
                except queue.Full:
                    self.dropped_count += 1

    def _worker_loop(self):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        hand_task_path = os.path.join(APP_DIR, 'include50_model', 'hand_landmarker.task')
        delegate_enum = python.BaseOptions.Delegate.GPU if self.pipeline.mediapipe_delegate in ("gpu", "metal") else python.BaseOptions.Delegate.CPU
        base_opt = python.BaseOptions(model_asset_path=hand_task_path, delegate=delegate_enum)
        opt = vision.HandLandmarkerOptions(
            base_options=base_opt,
            num_hands=2,
            min_hand_detection_confidence=0.15,
            min_hand_presence_confidence=0.15
        )
        try:
            worker_landmarker = vision.HandLandmarker.create_from_options(opt)
        except Exception:
            base_opt = python.BaseOptions(model_asset_path=hand_task_path, delegate=python.BaseOptions.Delegate.CPU)
            worker_landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(base_options=base_opt, num_hands=2))

        while self.running:
            try:
                task = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if task is None or not self.running:
                break

            task_start_time = time.time()
            frame_bgr = task["frame_bgr"]
            prod_pred = task["production_prediction"]
            prod_conf = task["production_confidence"]
            prod_start = task["prod_start_time"]
            prod_finish = task["prod_finish_time"]
            submit_time = task["submit_time"]

            queue_wait_ms = round((task_start_time - submit_time) * 1000.0, 2)
            prod_response_latency_ms = round((prod_finish - prod_start) * 1000.0, 2)

            try:
                if self.pipeline.mediapipe_delegate in ("gpu", "metal"):
                    rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGBA, data=rgba)
                else:
                    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                t_det0 = time.time()
                res = worker_landmarker.detect(mp_img)
                det_ms = round((time.time() - t_det0) * 1000.0, 2)

                out = self.pipeline.model_c_recognizer.predict_mediapipe_result(res)
                task_finish_time = time.time()
                shadow_exec_ms = round((task_finish_time - task_start_time) * 1000.0, 2)
                frame_age_ms = round((task_finish_time - prod_start) * 1000.0, 2)

                shadow_letter = out["predicted_letter"]
                shadow_conf = out["confidence"]

                telemetry = {
                    "enabled": True,
                    "mode": "async",
                    "status": "success",
                    "model_name": "Model C (Occlusion-Aware Geometry Random Forest)",
                    "predicted_letter": shadow_letter,
                    "confidence": shadow_conf,
                    "top_3": out["top_3"],
                    "hand_count": out["hand_count"],
                    "feature_status": out["feature_status"],
                    "landmark_detection_latency_ms": det_ms,
                    "feature_extraction_latency_ms": out.get("feature_extraction_ms", 0.0),
                    "classifier_inference_latency_ms": out.get("inference_ms", 0.0),
                    "shadow_execution_latency_ms": shadow_exec_ms,
                    "queue_wait_ms": queue_wait_ms,
                    "frame_age_ms": frame_age_ms,
                    "production_response_latency_ms": prod_response_latency_ms,
                    "production_predicted_letter": prod_pred,
                    "production_confidence": prod_conf,
                    "match_production": (shadow_letter == prod_pred if prod_pred != "-" else False),
                    "queue_stats": {
                        "submitted": self.submitted_count,
                        "processed": self.processed_count + 1,
                        "dropped": self.dropped_count,
                        "drop_rate_pct": round(self.dropped_count / max(1, self.submitted_count) * 100, 2)
                    }
                }
                with self._lock:
                    self.processed_count += 1
                    self.last_telemetry = telemetry

            except Exception as e:
                task_finish_time = time.time()
                shadow_exec_ms = round((task_finish_time - task_start_time) * 1000.0, 2)
                frame_age_ms = round((task_finish_time - prod_start) * 1000.0, 2)
                telemetry = {
                    "enabled": True,
                    "mode": "async",
                    "status": "error",
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "shadow_execution_latency_ms": shadow_exec_ms,
                    "queue_wait_ms": queue_wait_ms,
                    "frame_age_ms": frame_age_ms,
                    "production_response_latency_ms": prod_response_latency_ms,
                    "match_production": False,
                    "queue_stats": {
                        "submitted": self.submitted_count,
                        "processed": self.processed_count + 1,
                        "dropped": self.dropped_count,
                        "errors": self.error_count + 1
                    }
                }
                with self._lock:
                    self.error_count += 1
                    self.processed_count += 1
                    self.last_telemetry = telemetry

    def get_telemetry(self):
        with self._lock:
            return {
                "running": self.running,
                "submitted_count": self.submitted_count,
                "processed_count": self.processed_count,
                "dropped_count": self.dropped_count,
                "error_count": self.error_count,
                "drop_rate_pct": round(self.dropped_count / max(1, self.submitted_count) * 100, 2) if self.submitted_count > 0 else 0.0,
                "last_result": self.last_telemetry
            }

class ISLPipeline:
    def __init__(self):
        self.yolo = None
        self.model = None
        self.class_map = {}
        self.face_cascade = None
        self.initialized = False
        self.frame_counter = 0
        self.diagnostics_enabled = os.environ.get("ISL_DIAGNOSTICS", "1") == "1"

        # Configurable Phase 4 Inference Controls (Reversible via env or attributes)
        self.filter_edge_detections = os.environ.get("ISL_FILTER_EDGE_DETECTIONS", "0") == "1"
        self.edge_bottom_margin = int(os.environ.get("ISL_EDGE_BOTTOM_MARGIN", "10"))
        self.edge_min_width = int(os.environ.get("ISL_EDGE_MIN_WIDTH", "60"))
        self.edge_max_aspect_ratio = float(os.environ.get("ISL_EDGE_MAX_ASPECT_RATIO", "2.5"))

        self.conservative_face_clamp = os.environ.get("ISL_CONSERVATIVE_FACE_CLAMP", "0") == "1"
        self.max_face_clamp_ratio = float(os.environ.get("ISL_MAX_FACE_CLAMP_RATIO", "0.15"))

        # Phase 5: Correct Face-Clamping Alignment and Primary Hand Safeguard
        self.face_clamp_require_horizontal_overlap = os.environ.get("ISL_FACE_CLAMP_REQUIRE_HORIZONTAL_OVERLAP", "0").lower() in ("1", "true", "yes")
        self.face_clamp_primary_hand_safeguard = os.environ.get("ISL_FACE_CLAMP_PRIMARY_SAFEGUARD", "0").lower() in ("1", "true", "yes")

        self.improved_two_hand_merge = os.environ.get("ISL_IMPROVED_TWO_HAND_MERGE", "0") == "1"
        self.second_hand_conf = float(os.environ.get("ISL_SECOND_HAND_CONF", "0.45"))
        self.merge_max_gap_ratio = float(os.environ.get("ISL_MERGE_MAX_GAP_RATIO", "0.8"))
        self.merge_min_area_ratio = float(os.environ.get("ISL_MERGE_MIN_AREA_RATIO", "0.25"))

        # Lap-hand spatial suppression (Phase 8: Enabled by default, explicit override to disable)
        self.suppress_lap_hand = os.environ.get("ISL_SUPPRESS_LAP_HAND", "1").lower() not in ("0", "false", "no", "off")
        self.lap_hand_min_y_ratio = float(os.environ.get("ISL_LAP_HAND_MIN_Y_RATIO", "0.75"))
        self.lap_hand_max_active_y_ratio = float(os.environ.get("ISL_LAP_HAND_MAX_ACTIVE_Y_RATIO", "0.60"))

        # Direct-resize preprocessing (Option A - default disabled)
        self.direct_resize = os.environ.get("ISL_DIRECT_RESIZE", "0").lower() in ("1", "true", "yes")

        # Square-crop preprocessing (Option C - default disabled)
        self.square_crop = os.environ.get("ISL_SQUARE_CROP", "0").lower() in ("1", "true", "yes")

        # Phase 11: Backend Detector selection ('yolo' production default, 'mediapipe' experimental)
        raw_backend = os.environ.get("ISL_BACKEND_DETECTOR", "yolo").strip().lower()
        if raw_backend in ("mediapipe", "mp"):
            self.backend_detector = "mediapipe"
        elif raw_backend in ("yolo", "darknet", ""):
            self.backend_detector = "yolo"
        else:
            print(f"[Pipeline WARNING] Unsupported ISL_BACKEND_DETECTOR='{raw_backend}'. Falling back safely to 'yolo'.", flush=True)
            self.backend_detector = "yolo"
        self.mp_landmarker = None

        # Phase 14C & Phase 24: Recognition Architecture selection ('standard' production default, 'late_fusion', 'model_c' experimental)
        raw_arch = os.environ.get("ISL_RECOGNITION_ARCH", "standard").strip().lower()
        if raw_arch in ("late_fusion", "latefusion", "fusion", "hybrid_late"):
            self.recognition_arch = "late_fusion"
        elif raw_arch in ("model_c", "model_c_occlusion", "model_c_occlusion_rf", "occlusion_rf"):
            self.recognition_arch = "model_c"
        elif raw_arch in ("standard", "cnn_only", "default", ""):
            self.recognition_arch = "standard"
        else:
            print(f"[Pipeline WARNING] Unsupported ISL_RECOGNITION_ARCH='{raw_arch}'. Falling back safely to 'standard'.", flush=True)
            self.recognition_arch = "standard"

        # Phase 24 & 25: Opt-in Shadow Mode evaluation (default disabled)
        self.shadow_mode_enabled = os.environ.get("ISL_SHADOW_MODE", "0").lower() in ("1", "true", "yes")
        self.shadow_async = os.environ.get("ISL_SHADOW_ASYNC", "0").lower() in ("1", "true", "yes")
        self.shadow_worker = None
        self.model_c_recognizer = None

        # Phase 25 & 26: MediaPipe execution delegate ('cpu' production default, 'gpu' / 'metal' experimental opt-in)
        raw_delegate = os.environ.get("ISL_MEDIAPIPE_DELEGATE", "cpu").strip().lower()
        self.mediapipe_delegate = "gpu" if raw_delegate in ("gpu", "metal") else "cpu"

        # Phase 26 & 27: Optional Model C Input Resolution Downsampling ('native', '960x540', '640x360', '640x480')
        self.model_c_input_resolution = os.environ.get(
            "ISL_MODEL_C_RESOLUTION",
            os.environ.get("ISL_MODEL_C_INPUT_RES", "native")
        ).strip().lower()

        # Phase 14C: Optional Temporal Smoothing (default disabled)
        self.temporal_smoothing_enabled = os.environ.get("ISL_TEMPORAL_SMOOTHING", "0").lower() in ("1", "true", "yes")
        from temporal_smoother import TemporalSmoother
        self.temporal_smoother = TemporalSmoother(window_size=5, max_idle_sec=2.0)

        # Late fusion components
        self.landmark_rf_model = None
        self.landmark_scaler = None
        self.late_fusion_alpha = 0.25
        self.hybrid_config = None

    def initialize(self):
        if self.initialized:
            return

        project_root = os.path.dirname(os.path.abspath(__file__))

        if self.backend_detector == "yolo":
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

            print(f"[Pipeline] Loading Production SqueezeNet model from {model_path}...")
            self.model = tf_keras.models.load_model(model_path)

            with open(class_json_path, 'r') as f:
                self.class_map = json.load(f)

            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(cascade_path)

        elif self.backend_detector == "mediapipe":
            hand_task_path = os.path.join(APP_DIR, 'include50_model', 'hand_landmarker.task')
            if not os.path.exists(hand_task_path):
                raise FileNotFoundError(f"MediaPipe hand landmarker task not found at {hand_task_path}")

            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            base_opt = python.BaseOptions(model_asset_path=hand_task_path)
            opt = vision.HandLandmarkerOptions(
                base_options=base_opt,
                num_hands=2,
                min_hand_detection_confidence=0.15,
                min_hand_presence_confidence=0.15
            )
            print(f"[Pipeline] Loading MediaPipe HandLandmarker from {hand_task_path}...")
            self.mp_landmarker = vision.HandLandmarker.create_from_options(opt)

            exp_model_path = os.path.join(project_root, 'models', 'mediapipe_squeezenet_v1', 'saved_model')
            exp_class_json_path = os.path.join(project_root, 'models', 'mediapipe_squeezenet_v1', 'model_class_26.json')
            if not os.path.exists(exp_model_path):
                exp_model_path = os.path.join(project_root, 'models', 'mediapipe_squeezenet_v1', 'mediapipe_squeezenet_v1.keras')

            print(f"[Pipeline] Loading MediaPipe-tuned SqueezeNet model from {exp_model_path}...")
            self.model = tf_keras.models.load_model(exp_model_path)

            with open(exp_class_json_path, 'r') as f:
                self.class_map = json.load(f)

            self.face_cascade = None

        # Load late-fusion models if requested
        if self.recognition_arch == "late_fusion":
            try:
                self._load_late_fusion_models(project_root)
            except Exception as e:
                print(f"[Pipeline WARNING] Failed to initialize late-fusion components: {e}. Falling back safely to 'standard' recognition path.", flush=True)
                self.recognition_arch = "standard"

        # Load Model C if requested or if shadow mode is enabled
        if self.recognition_arch == "model_c" or self.shadow_mode_enabled:
            try:
                self._load_model_c(project_root)
            except Exception as e:
                print(f"[Pipeline WARNING] Failed to initialize Model C components: {e}. Falling back safely.", flush=True)
                if self.recognition_arch == "model_c":
                    self.recognition_arch = "standard"
                self.shadow_mode_enabled = False

        # Warm up SqueezeNet
        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        _ = self.model.predict(dummy, verbose=0)

        self.initialized = True
        print(f"[Pipeline] Models initialized successfully! (Backend: {self.backend_detector.upper()})")
        print("[Pipeline] Effective Runtime Configuration:")
        print(f"  - ISL_BACKEND_DETECTOR: {self.backend_detector}")
        print(f"  - ISL_RECOGNITION_ARCH: {self.recognition_arch}")
        print(f"  - ISL_SHADOW_MODE: {self.shadow_mode_enabled}")
        print(f"  - ISL_TEMPORAL_SMOOTHING: {self.temporal_smoothing_enabled}")
        if self.backend_detector == "yolo":
            print(f"  - ISL_SUPPRESS_LAP_HAND: {self.suppress_lap_hand} (default=True, min_y_ratio={self.lap_hand_min_y_ratio}, max_active_y_ratio={self.lap_hand_max_active_y_ratio})")
            print(f"  - ISL_DIRECT_RESIZE: {self.direct_resize} (default=False)")
            print(f"  - ISL_SQUARE_CROP: {self.square_crop} (default=False)")
            print(f"  - ISL_FILTER_EDGE_DETECTIONS: {self.filter_edge_detections} (default=False)")
            print(f"  - ISL_CONSERVATIVE_FACE_CLAMP: {self.conservative_face_clamp} (default=False)")
            print(f"  - ISL_IMPROVED_TWO_HAND_MERGE: {self.improved_two_hand_merge} (default=False)")

    def _load_late_fusion_models(self, project_root: str):
        """Loads and verifies late fusion landmark RF model, scaler, and configuration."""
        import joblib
        models_dir = os.path.join(project_root, "models", "landmark_hybrid_v1")
        rf_path = os.path.join(models_dir, "landmark_rf_model.joblib")
        scaler_path = os.path.join(models_dir, "landmark_scaler.joblib")
        cfg_path = os.path.join(models_dir, "hybrid_config.json")

        if not os.path.exists(rf_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Missing late fusion models at {models_dir}")

        print(f"[Pipeline] Loading Landmark Random Forest from {rf_path}...")
        self.landmark_rf_model = joblib.load(rf_path)
        print(f"[Pipeline] Loading Landmark Scaler from {scaler_path}...")
        self.landmark_scaler = joblib.load(scaler_path)

        if os.path.exists(cfg_path):
            with open(cfg_path, "r") as f:
                self.hybrid_config = json.load(f)
                self.late_fusion_alpha = float(self.hybrid_config.get("late_fusion_alpha", 0.25))

        # Ensure MediaPipe HandLandmarker is available for landmark extraction
        if self.mp_landmarker is None:
            hand_task_path = os.path.join(APP_DIR, 'include50_model', 'hand_landmarker.task')
            if not os.path.exists(hand_task_path):
                raise FileNotFoundError(f"MediaPipe hand landmarker task not found at {hand_task_path}")

            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            base_opt = python.BaseOptions(model_asset_path=hand_task_path)
            opt = vision.HandLandmarkerOptions(
                base_options=base_opt,
                num_hands=2,
                min_hand_detection_confidence=0.15,
                min_hand_presence_confidence=0.15
            )
            print(f"[Pipeline] Loading auxiliary MediaPipe HandLandmarker for late fusion from {hand_task_path}...")
            self.mp_landmarker = vision.HandLandmarker.create_from_options(opt)

        # Ensure MediaPipe SqueezeNet is loaded if currently using production YOLO SqueezeNet
        if self.backend_detector == "yolo":
            exp_model_path = os.path.join(project_root, 'models', 'mediapipe_squeezenet_v1', 'saved_model')
            if not os.path.exists(exp_model_path):
                exp_model_path = os.path.join(project_root, 'models', 'mediapipe_squeezenet_v1', 'mediapipe_squeezenet_v1.keras')
            if os.path.exists(exp_model_path):
                print(f"[Pipeline] Loading auxiliary MediaPipe-tuned SqueezeNet for late fusion from {exp_model_path}...")
                self._mediapipe_cnn_model = tf_keras.models.load_model(exp_model_path)
            else:
                self._mediapipe_cnn_model = self.model
        else:
            self._mediapipe_cnn_model = self.model

        print(f"[Pipeline] Late-fusion models loaded successfully! (alpha={self.late_fusion_alpha})")

    def _load_model_c(self, project_root: str):
        """Loads and verifies Model C Recognizer (280-dim Occlusion-Aware Geometry RF)."""
        from landmark_model_c import model_c_recognizer
        self.model_c_recognizer = model_c_recognizer
        if not self.model_c_recognizer.is_loaded:
            print("[Pipeline] Loading Model C (Occlusion-Aware Geometry Random Forest, 280d)...", flush=True)
            self.model_c_recognizer.load()

        # Ensure MediaPipe HandLandmarker is available for landmark extraction
        if self.mp_landmarker is None:
            hand_task_path = os.path.join(APP_DIR, 'include50_model', 'hand_landmarker.task')
            if not os.path.exists(hand_task_path):
                raise FileNotFoundError(f"MediaPipe hand landmarker task not found at {hand_task_path}")

            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            delegate_enum = python.BaseOptions.Delegate.GPU if self.mediapipe_delegate in ("gpu", "metal") else python.BaseOptions.Delegate.CPU
            base_opt = python.BaseOptions(model_asset_path=hand_task_path, delegate=delegate_enum)
            opt = vision.HandLandmarkerOptions(
                base_options=base_opt,
                num_hands=2,
                min_hand_detection_confidence=0.15,
                min_hand_presence_confidence=0.15
            )
            print(f"[Pipeline] Loading auxiliary MediaPipe HandLandmarker for Model C (delegate={self.mediapipe_delegate}) from {hand_task_path}...")
            try:
                self.mp_landmarker = vision.HandLandmarker.create_from_options(opt)
            except Exception as e:
                if self.mediapipe_delegate in ("gpu", "metal"):
                    print(f"[Pipeline WARNING] Failed to initialize GPU delegate ({e}). Falling back safely to CPU.", flush=True)
                    self.mediapipe_delegate = "cpu"
                    base_opt = python.BaseOptions(model_asset_path=hand_task_path, delegate=python.BaseOptions.Delegate.CPU)
                    opt = vision.HandLandmarkerOptions(base_options=base_opt, num_hands=2, min_hand_detection_confidence=0.15, min_hand_presence_confidence=0.15)
                    self.mp_landmarker = vision.HandLandmarker.create_from_options(opt)
                else:
                    raise

        print("[Pipeline] Model C recognizer loaded successfully!")

    def set_recognition_arch(self, arch: str):
        """
        Dynamically switch recognition architecture ('standard', 'late_fusion', or 'model_c').
        """
        clean_a = str(arch).strip().lower()
        if clean_a in ("late_fusion", "latefusion", "fusion", "hybrid_late"):
            target = "late_fusion"
        elif clean_a in ("model_c", "model_c_occlusion", "model_c_occlusion_rf", "occlusion_rf"):
            target = "model_c"
        elif clean_a in ("standard", "cnn_only", "default", ""):
            target = "standard"
        else:
            raise ValueError(f"Unsupported recognition architecture '{arch}'. Must be 'standard', 'late_fusion', or 'model_c'.")

        project_root = os.path.dirname(os.path.abspath(__file__))
        if target == "model_c":
            if self.model_c_recognizer is None or not self.model_c_recognizer.is_loaded:
                try:
                    self._load_model_c(project_root)
                except Exception as e:
                    print(f"[Pipeline WARNING] Failed to switch to model_c: {e}. Remaining in standard mode.", flush=True)
                    self.recognition_arch = "standard"
                    return

        elif target == "late_fusion" and self.landmark_rf_model is None:
            try:
                self._load_late_fusion_models(project_root)
            except Exception as e:
                print(f"[Pipeline WARNING] Failed to switch to late_fusion: {e}. Remaining in standard mode.", flush=True)
                self.recognition_arch = "standard"
                return

        self.recognition_arch = target
        print(f"[Pipeline] Switched recognition architecture to '{self.recognition_arch}'.", flush=True)

    def set_shadow_mode(self, enabled: bool, async_mode: bool = None):
        """Enable or disable shadow-mode evaluation of Model C."""
        if async_mode is not None:
            self.shadow_async = bool(async_mode)
        if enabled:
            if self.model_c_recognizer is None or not self.model_c_recognizer.is_loaded:
                project_root = os.path.dirname(os.path.abspath(__file__))
                self._load_model_c(project_root)
            self.shadow_mode_enabled = True
            if self.shadow_async:
                if self.shadow_worker is None:
                    self.shadow_worker = ShadowWorker(self)
                self.shadow_worker.start()
            print(f"[Pipeline] Shadow-mode evaluation ENABLED (mode={'async' if self.shadow_async else 'sync'}).", flush=True)
        else:
            self.shadow_mode_enabled = False
            if self.shadow_worker:
                self.shadow_worker.stop()
            print("[Pipeline] Shadow-mode evaluation DISABLED.", flush=True)

    def get_shadow_telemetry(self):
        """Returns structured telemetry from background shadow worker."""
        if self.shadow_worker:
            return self.shadow_worker.get_telemetry()
        return {
            "running": False,
            "mode": "async" if self.shadow_async else "sync",
            "submitted_count": 0,
            "processed_count": 0,
            "dropped_count": 0,
            "error_count": 0,
            "drop_rate_pct": 0.0,
            "last_result": None
        }

    def shutdown(self):
        """Gracefully releases background threads and resources."""
        if self.shadow_worker:
            self.shadow_worker.stop()
        if hasattr(self, "mp_landmarker") and self.mp_landmarker is not None:
            try:
                self.mp_landmarker.close()
            except Exception:
                pass
            self.mp_landmarker = None
        if hasattr(self, "_model_c_color_buf"):
            self._model_c_color_buf = None

    def set_temporal_smoothing(self, enabled: bool):
        """Enable or disable rolling temporal smoothing."""
        self.temporal_smoothing_enabled = bool(enabled)
        print(f"[Pipeline] Temporal smoothing set to: {self.temporal_smoothing_enabled}", flush=True)

    def set_mediapipe_delegate(self, delegate: str):
        """
        Dynamically switch MediaPipe execution delegate ('cpu' or 'gpu' / 'metal')
        and safely reloads the landmarker instance.
        """
        clean_raw = str(delegate).strip().lower()
        if clean_raw in ("gpu", "metal"):
            clean_d = "gpu"
        elif clean_raw in ("cpu", "default"):
            clean_d = "cpu"
        else:
            raise ValueError(f"Unsupported MediaPipe delegate '{delegate}'. Must be 'cpu', 'gpu', or 'metal'.")
        if clean_d != self.mediapipe_delegate or self.mp_landmarker is None:
            self.mediapipe_delegate = clean_d
            if self.mp_landmarker is not None:
                try:
                    self.mp_landmarker.close()
                except Exception:
                    pass
                self.mp_landmarker = None
            if self.recognition_arch == "model_c" or self.shadow_mode_enabled or self.backend_detector == "mediapipe":
                project_root = os.path.dirname(os.path.abspath(__file__))
                self._load_model_c(project_root)
            print(f"[Pipeline] MediaPipe delegate switched to '{self.mediapipe_delegate}'.", flush=True)

    def set_model_c_input_resolution(self, resolution: str):
        """
        Sets opt-in downsampling resolution for Model C ('native', '960x540', '640x360', '640x480').
        """
        clean_res = str(resolution).strip().lower()
        if clean_res in ("native", "1280x720", "original", "none", ""):
            self.model_c_input_resolution = "native"
        elif clean_res in ("960x540", "640x360", "640x480"):
            self.model_c_input_resolution = clean_res
        else:
            raise ValueError(f"Unsupported resolution '{resolution}'. Must be 'native', '960x540', '640x360', or '640x480'.")
        print(f"[Pipeline] Model C input resolution set to '{self.model_c_input_resolution}'.", flush=True)

    def _get_target_resolution(self, w: int, h: int):
        res_cfg = getattr(self, "model_c_input_resolution", "native")
        if res_cfg in ("native", "1280x720", "original", None, ""):
            return None
        elif res_cfg == "960x540":
            return (960, 540)
        elif res_cfg == "640x360":
            return (640, 360)
        elif res_cfg == "640x480":
            return (640, 480)
        return None

    def set_backend(self, backend: str):
        """
        Dynamically switch backend detector ('yolo' or 'mediapipe') without stale state.
        """
        clean_b = str(backend).strip().lower()
        if clean_b in ("mediapipe", "mp"):
            target = "mediapipe"
        elif clean_b in ("yolo", "darknet"):
            target = "yolo"
        else:
            raise ValueError(f"Unsupported backend detector '{backend}'. Must be 'yolo' or 'mediapipe'.")

        if target != self.backend_detector or not self.initialized:
            print(f"[Pipeline] Switching backend detector from {self.backend_detector} to {target}...")
            self.backend_detector = target
            self.yolo = None
            self.mp_landmarker = None
            self.model = None
            self.class_map = {}
            self.initialized = False
            self.initialize()

    def detect_face(self, frame_bgr):
        if self.face_cascade is None:
            return False, []
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
        face_boxes = []
        for (x, y, w, h) in faces:
            face_boxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})
        return len(face_boxes) > 0, face_boxes

    def compute_square_crop(self, frame_shape, crop_coords):
        """
        Expands the bounding crop to a square using the larger of width and height: side = max(w, h).
        Centers the square on the original crop, subject to frame boundaries.
        When crop reaches frame boundary, shifts it to stay within bounds.
        If boundary cannot accommodate the square (e.g. at corner or side exceeds dimension),
        pads it so the square geometry is strictly preserved without stretching or cutting off gesture.
        Returns:
            (valid_x1, valid_y1, valid_x2, valid_y2): extraction slice coordinates within frame
            (sq_x1, sq_y1, side, side): logical square crop bounding box
            (pad_top, pad_bottom, pad_left, pad_right): padding needed to achieve side x side
        """
        frame_h, frame_w = frame_shape[:2]
        crop_x, crop_y, crop_x1, crop_y1 = crop_coords
        cw = max(1, crop_x1 - crop_x)
        ch = max(1, crop_y1 - crop_y)

        side = max(cw, ch)
        side = min(side, min(frame_w, frame_h))

        cx = crop_x + cw / 2.0
        cy = crop_y + ch / 2.0

        sq_x1 = int(round(cx - side / 2.0))
        sq_y1 = int(round(cy - side / 2.0))
        sq_x2 = sq_x1 + side
        sq_y2 = sq_y1 + side

        # Horizontal boundary shift
        if sq_x1 < 0:
            shift_x = -sq_x1
            sq_x1 += shift_x
            sq_x2 += shift_x
        elif sq_x2 > frame_w:
            shift_x = sq_x2 - frame_w
            sq_x1 -= shift_x
            sq_x2 -= shift_x

        # Vertical boundary shift
        if sq_y1 < 0:
            shift_y = -sq_y1
            sq_y1 += shift_y
            sq_y2 += shift_y
        elif sq_y2 > frame_h:
            shift_y = sq_y2 - frame_h
            sq_y1 -= shift_y
            sq_y2 -= shift_y

        pad_top = max(0, -sq_y1)
        pad_bottom = max(0, sq_y2 - frame_h)
        pad_left = max(0, -sq_x1)
        pad_right = max(0, sq_x2 - frame_w)

        valid_y1 = max(0, sq_y1)
        valid_y2 = min(frame_h, sq_y2)
        valid_x1 = max(0, sq_x1)
        valid_x2 = min(frame_w, sq_x2)

        return (valid_x1, valid_y1, valid_x2, valid_y2), (sq_x1, sq_y1, side, side), (pad_top, pad_bottom, pad_left, pad_right)

    @staticmethod
    def compute_landmark_crop_box(landmarks, frame_shape, margin_ratio=0.15, force_square=True):
        """
        Computes a localized bounding box from a list of landmarks (each having .x, .y attributes
        or (x, y) tuples normalized in [0, 1] or pixel coordinates).
        - Isolates palm & fingers from forearm (wrist is landmark 0).
        - Adds margin_ratio proportional to hand dimensions.
        - If force_square is True, expands bounding box to 1:1 aspect ratio with boundary shift.
        Returns:
            (cx1, cy1, cx2, cy2): clamped crop coordinates in pixel space.
        """
        frame_h, frame_w = frame_shape[:2]
        pts = []
        for lm in landmarks:
            if hasattr(lm, 'x') and hasattr(lm, 'y'):
                px = lm.x * frame_w if lm.x <= 1.0 else lm.x
                py = lm.y * frame_h if lm.y <= 1.0 else lm.y
            elif isinstance(lm, (tuple, list)) and len(lm) >= 2:
                px = lm[0] * frame_w if lm[0] <= 1.0 else lm[0]
                py = lm[1] * frame_h if lm[1] <= 1.0 else lm[1]
            else:
                continue
            pts.append((px, py))

        if not pts:
            return (0, 0, frame_w, frame_h)

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        bw = max(1.0, max_x - min_x)
        bh = max(1.0, max_y - min_y)

        if force_square:
            side = int(round(max(bw, bh) * (1.0 + 2.0 * margin_ratio)))
            side = max(1, min(side, min(frame_w, frame_h)))
            cx = (min_x + max_x) / 2.0
            cy = (min_y + max_y) / 2.0
            x1 = int(round(cx - side / 2.0))
            y1 = int(round(cy - side / 2.0))
            x2 = x1 + side
            y2 = y1 + side

            # Boundary shift
            if x1 < 0:
                shift = -x1
                x1 += shift
                x2 += shift
            elif x2 > frame_w:
                shift = x2 - frame_w
                x1 -= shift
                x2 -= shift

            if y1 < 0:
                shift = -y1
                y1 += shift
                y2 += shift
            elif y2 > frame_h:
                shift = y2 - frame_h
                y1 -= shift
                y2 -= shift

            ix1 = max(0, min(frame_w - side, x1))
            iy1 = max(0, min(frame_h - side, y1))
            ix2 = ix1 + side
            iy2 = iy1 + side
            return (ix1, iy1, ix2, iy2)
        else:
            mx = bw * margin_ratio
            my = bh * margin_ratio
            x1 = min_x - mx
            y1 = min_y - my
            x2 = max_x + mx
            y2 = max_y + my
            ix1 = int(round(max(0, min(frame_w - 1, x1))))
            iy1 = int(round(max(0, min(frame_h - 1, y1))))
            ix2 = int(round(max(ix1 + 1, min(frame_w, x2))))
            iy2 = int(round(max(iy1 + 1, min(frame_h, y2))))
            return (ix1, iy1, ix2, iy2)

    @staticmethod
    def compute_forearm_trimmed_box(box_coords, frame_shape, trim_bottom_ratio=0.15):
        """
        Trims bottom forearm portion from an elongated hand bounding box when aspect ratio indicates forearm inclusion.
        Returns:
            (x1, y1, x2, y2): trimmed bounding box coordinates.
        """
        frame_h, frame_w = frame_shape[:2]
        x1, y1, x2, y2 = box_coords
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        # If box is vertically elongated (h/w > 1.3), trim forearm region
        if h / w > 1.3:
            trim_px = int(h * trim_bottom_ratio)
            y2 = max(y1 + w, y2 - trim_px)
        return (max(0, x1), max(0, y1), min(frame_w, x2), min(frame_h, y2))

    def _log_diagnostics(self, d):
        if not d["hand_detected"]:
            print(f"[DIAG #{d['frame_id']:04d}] YOLO=0 | Hands=0 | FaceClamp=F | No hand detected | Latency={d['latency_ms']:.0f}ms", flush=True)
            return
        box = d["selected_box"]
        crop = d["crop_coords"]
        lb = d["letterbox_padding"]
        top3_str = ",".join([f"{s}:{p:.1f}%" for s, p in d["top_3"]])
        hands_str = "2H" if d["is_two_handed"] else "1H"
        fc_str = f"T(+{d['face_clamp_delta_y']}px)" if d["face_clamped"] else "F"
        lb_str = f"T(pad={lb['dx']},{lb['dy']})" if d["letterboxed"] else "F"
        lap_str = f" | LapSupp={d.get('lap_hands_suppressed', 0)}" if d.get('lap_hands_suppressed', 0) > 0 else ""
        if d.get("preprocessing_mode") == "square_crop" and d.get("square_crop_info"):
            sq = d["square_crop_info"]["square_crop_coords"]
            pad = d["square_crop_info"]["padding"]
            mode_str = f" | Mode=square_crop(sq=[{sq['x']},{sq['y']},{sq['w']},{sq['h']}],pad=[T:{pad['top']},B:{pad['bottom']},L:{pad['left']},R:{pad['right']}])"
        else:
            mode_str = f" | Mode={d.get('preprocessing_mode', 'letterbox')}"
        print(f"[DIAG #{d['frame_id']:04d}] YOLO={d['yolo_detections_count']} box=({box['x']},{box['y']},{box['w']},{box['h']}) conf={box['conf']:.2f} | "
              f"Hands={hands_str} | FaceClamp={fc_str} | Crop=({crop['x']},{crop['y']},{crop['w']},{crop['h']}) | "
              f"LB={lb_str}{lap_str}{mode_str} | Pred={d['predicted_letter']} ({d['confidence']:.1f}%) | "
              f"Top3=[{top3_str}] | Latency={d['latency_ms']:.0f}ms", flush=True)

    def process_frame(self, frame_bgr, return_images=True, session_id="default_session"):
        if self.recognition_arch == "model_c":
            if self.model_c_recognizer is None or not self.model_c_recognizer.is_loaded:
                project_root = os.path.dirname(os.path.abspath(__file__))
                self._load_model_c(project_root)
        elif not self.initialized:
            self.initialize()

        self.frame_counter += 1
        frame_id = self.frame_counter
        timestamp = time.time()
        t_start = timestamp

        if self.recognition_arch == "model_c":
            result = self._process_frame_model_c(frame_bgr, frame_id, timestamp, t_start, return_images=return_images, session_id=session_id)
        elif self.backend_detector == "mediapipe" or self.recognition_arch == "late_fusion":
            result = self._process_frame_mediapipe(frame_bgr, frame_id, timestamp, t_start, return_images=return_images, session_id=session_id)
        else:
            result = self._process_frame_yolo(frame_bgr, frame_id, timestamp, t_start, return_images=return_images, session_id=session_id)

        # Shadow-mode execution (if enabled and primary architecture is not model_c)
        if self.shadow_mode_enabled and self.recognition_arch != "model_c":
            if self.shadow_async:
                if self.shadow_worker is None:
                    self.shadow_worker = ShadowWorker(self)
                t_prod_finish = time.time()
                self.shadow_worker.submit_task(frame_bgr, result, t_start, t_prod_finish)
                tel = self.shadow_worker.get_telemetry()
                if "diagnostics" not in result or not isinstance(result["diagnostics"], dict):
                    result["diagnostics"] = {}
                if tel.get("last_result"):
                    result["diagnostics"]["shadow_model_c"] = tel["last_result"]
                else:
                    result["diagnostics"]["shadow_model_c"] = {
                        "enabled": True,
                        "mode": "async",
                        "status": "pending_first_frame",
                        "model_name": "Model C",
                        "match_production": False
                    }
            else:
                self._run_shadow_inference(frame_bgr, result)

        return result

    def predict(self, frame_bgr, return_images=True, session_id="default_session"):
        return self.process_frame(frame_bgr, return_images=return_images, session_id=session_id)

    def _process_frame_yolo(self, frame_bgr, frame_id, timestamp, t_start, return_images=True, session_id="default_session"):
        h, w = frame_bgr.shape[:2]

        # 1. Hand detection with YOLO
        t_yolo_start = time.time()
        iw, ih, inference_time, detections = self.yolo.inference(frame_bgr)
        t_yolo_end = time.time()

        annotated_frame = frame_bgr.copy()
        hand_detected = False
        box_coords = None
        box_info = []
        primary_conf = None
        rejected_detections = []
        raw_detections = list(detections)

        # Fix A: Reject implausible edge detections (e.g., bottom-edge forearm slivers)
        valid_detections = []
        for det in detections:
            _, name, conf, bx, by, bw, bh = det
            is_edge_sliver = False
            reject_reason = None

            if self.filter_edge_detections:
                # Condition 1: Contact with bottom edge of frame within margin
                touches_bottom = (by + bh) >= (h - self.edge_bottom_margin)
                # Condition 2: Suspicious non-hand geometry (narrow width or tall vertical sliver)
                is_narrow = (bw < self.edge_min_width)
                is_extreme_aspect = ((bh / max(1, bw)) > self.edge_max_aspect_ratio)

                # Apply conditions together: reject only if touching bottom edge AND exhibits suspicious geometry
                if touches_bottom and (is_narrow or is_extreme_aspect):
                    is_edge_sliver = True
                    aspect_val = bh / max(1, bw)
                    reject_reason = (f"bottom_edge({by+bh}>={h-self.edge_bottom_margin})_with_" +
                                     (f"narrow_w({bw}<{self.edge_min_width})" if is_narrow else f"high_aspect({aspect_val:.2f}>{self.edge_max_aspect_ratio})"))

            if is_edge_sliver:
                rejected_detections.append({
                    "name": name,
                    "confidence": round(float(conf) * 100, 1),
                    "x": int(bx), "y": int(by), "w": int(bw), "h": int(bh),
                    "reason": reject_reason
                })
            else:
                valid_detections.append(det)

        # Lap-hand spatial suppression: Ignore resting hands near bottom edge when elevated signing hand exists
        lap_hands_suppressed = 0
        if self.suppress_lap_hand and len(valid_detections) > 1:
            elevated_hands = [d for d in valid_detections if d[4] < self.lap_hand_max_active_y_ratio * h]
            lap_hands = [d for d in valid_detections if (d[4] + d[6]) >= self.lap_hand_min_y_ratio * h and d[4] >= (self.lap_hand_min_y_ratio - 0.10) * h]
            if elevated_hands and lap_hands:
                lap_hands_suppressed = len(lap_hands)
                valid_detections = [d for d in valid_detections if d not in lap_hands]
                if self.diagnostics_enabled:
                    for ld in lap_hands:
                        print(f"[LapSuppression] Suppressed resting lap hand: box=({ld[3]},{ld[4]},{ld[5]},{ld[6]}), conf={ld[2]*100:.1f}%", flush=True)

        detections = valid_detections

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

            # Fix C & Phase 5: Determine whether to use primary hand or merge two hands
            is_two_handed = False
            primary_hand_identified = False
            primary_hand_box = None
            primary_hand_h = None
            hand_ambiguity_recorded = False

            if len(detections) == 1:
                _, _, conf, bx, by, bw, bh = detections[0]
                box_coords = (bx, by, bx + bw, by + bh)
                primary_conf = conf
                primary_hand_box = {"x": int(bx), "y": int(by), "w": int(bw), "h": int(bh)}
                primary_hand_h = int(bh)
                primary_hand_identified = True
            else:
                det1 = detections[0]  # highest confidence
                det2 = detections[1]  # second highest
                b1_x, b1_y, b1_w, b1_h = det1[3], det1[4], det1[5], det1[6]
                b2_x, b2_y, b2_w, b2_h = det2[3], det2[4], det2[5], det2[6]

                # Check horizontal gap between the two detections
                left_box = det1 if b1_x < b2_x else det2
                right_box = det2 if b1_x < b2_x else det1
                gap_x = right_box[3] - (left_box[3] + left_box[5])
                vertical_overlap = min(b1_y + b1_h, b2_y + b2_h) - max(b1_y, b2_y)

                if self.improved_two_hand_merge:
                    # Stricter merge conditions to prevent distractor merges:
                    # 1. Higher confidence threshold for second hand (default >= 0.45 vs 0.30)
                    conf_ok = (det2[2] >= self.second_hand_conf)

                    # 2. Hands must be close / interacting (gap < 0.8 * min_hand_w)
                    min_hand_w = min(b1_w, b2_w)
                    is_close_interaction = (gap_x < min_hand_w * self.merge_max_gap_ratio)

                    # 3. Genuine vertical alignment: hands must be aligned vertically
                    min_hand_h = min(b1_h, b2_h)
                    is_vertically_aligned = (vertical_overlap > -min_hand_h * 0.15)

                    # 4. Area balance: prevent merging a tiny distractor box with a large hand
                    area1 = b1_w * b1_h
                    area2 = b2_w * b2_h
                    area_balanced = (min(area1, area2) / max(1, max(area1, area2))) >= self.merge_min_area_ratio

                    # 5. Distractor check: second detection should not touch bottom boundary
                    not_bottom_distractor = not (b2_y + b2_h >= h - self.edge_bottom_margin and (b2_h / max(1, b2_w)) > 2.0)

                    should_merge = (conf_ok and is_close_interaction and is_vertically_aligned and area_balanced and not_bottom_distractor)
                else:
                    # Baseline merge conditions
                    max_hand_w = max(b1_w, b2_w)
                    is_close_interaction = (gap_x < max_hand_w * 1.2)
                    is_vertically_aligned = (vertical_overlap > -max(b1_h, b2_h) * 0.6)
                    should_merge = (is_close_interaction and is_vertically_aligned and det2[2] >= 0.30)

                if should_merge:
                    is_two_handed = True
                    x = min(det1[3], det2[3])
                    y = min(det1[4], det2[4])
                    x1 = max(det1[3] + det1[5], det2[3] + det2[5])
                    y1 = max(det1[4] + det1[6], det2[4] + det2[6])
                    box_coords = (x, y, x1, y1)
                    primary_conf = (det1[2] + det2[2]) / 2.0

                    # Phase 5 Requirement 2: Primary hand determination for merged detections
                    # If det1 has distinctly higher confidence than det2 (>= 0.10 margin), det1 is primary active hand
                    if (det1[2] - det2[2]) >= 0.10:
                        primary_hand_box = {"x": int(b1_x), "y": int(b1_y), "w": int(b1_w), "h": int(b1_h)}
                        primary_hand_h = int(b1_h)
                        primary_hand_identified = True
                    else:
                        # Ambiguous which hand is primary active gesture:
                        # Do not silently guess; record ambiguity and use safe fallback: min(b1_h, b2_h)
                        primary_hand_box = {"x": int(b1_x), "y": int(b1_y), "w": int(b1_w), "h": int(b1_h)}
                        primary_hand_h = int(min(b1_h, b2_h))
                        primary_hand_identified = False
                        hand_ambiguity_recorded = True
                else:
                    # Single active hand gesture: use primary detection, ignore resting elbow/background
                    box_coords = (b1_x, b1_y, b1_x + b1_w, b1_y + b1_h)
                    primary_conf = det1[2]
                    primary_hand_box = {"x": int(b1_x), "y": int(b1_y), "w": int(b1_w), "h": int(b1_h)}
                    primary_hand_h = int(b1_h)
                    primary_hand_identified = True
            self._last_is_two_handed = is_two_handed
        else:
            self._last_is_two_handed = False
            primary_conf = None
            primary_hand_identified = False
            primary_hand_box = None
            primary_hand_h = None
            hand_ambiguity_recorded = False

        # 2. Early return if no hand detected
        if not hand_detected or box_coords is None:
            if self.temporal_smoothing_enabled:
                self.temporal_smoother.update(session_id, "-", 0.0, timestamp=timestamp)
            t_total_ms = round((time.time() - t_start) * 1000, 1)
            side = min(w, h)
            sx = (w - side) // 2
            sy = (h - side) // 2
            c_crop = cv2.resize(frame_bgr[sy:sy + side, sx:sx + side], (224, 224))
            c_seg = skinDetector(c_crop).find_skin()
            diag_info = {
                "frame_id": frame_id,
                "timestamp": round(timestamp, 3),
                "yolo_detections_count": len(detections),
                "raw_yolo_count": len(raw_detections),
                "edge_detections_rejected": len(rejected_detections),
                "rejected_detections": rejected_detections,
                "hand_detected": False,
                "is_two_handed": False,
                "selected_box": None,
                "face_clamped": False,
                "face_clamp_delta_y": 0,
                "excessive_face_clamp_avoided": False,
                "non_overlapping_clamps_ignored": 0,
                "face_clamp_decisions": [],
                "primary_hand_identified": False,
                "hand_ambiguity_recorded": False,
                "primary_hand_h": None,
                "crop_coords": None,
                "square_crop_info": None,
                "letterboxed": False,
                "letterbox_padding": None,
                "predicted_letter": "-",
                "confidence": 0.0,
                "top_3": [],
                "latency_ms": t_total_ms
            }
            if self.diagnostics_enabled:
                self._log_diagnostics(diag_info)
            res = {
                "predicted_sign": "-",
                "confidence": 0.0,
                "hand_detected": False,
                "is_two_handed": False,
                "detections": [],
                "probabilities": [],
                "diagnostics": diag_info,
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
        x, y, x1, y1 = box_coords
        box_w = x1 - x
        box_h = y1 - y

        pad_x = max(10, int(box_w * 0.10))
        pad_y = max(10, int(box_h * 0.10))

        crop_x = max(0, x - pad_x)
        crop_y = max(0, y - pad_y)
        crop_x1 = min(w, x1 + pad_x)
        crop_y1 = min(h, y1 + pad_y)

        # Fix B & Phase 5: Conservative Face Clamping Safeguard with Horizontal Overlap & Primary Hand Grounding
        orig_crop_y = crop_y
        face_clamped = False
        face_clamp_delta_y = 0
        excessive_clamp_avoided = False
        non_overlapping_clamps_ignored = 0
        face_clamp_decisions = []

        has_face, face_boxes = self.detect_face(frame_bgr)
        if has_face and len(face_boxes) > 0:
            for fb in face_boxes:
                face_x1 = fb["x"]
                face_x2 = fb["x"] + fb["w"]
                f_bottom = fb["y"] + fb["h"]

                # Phase 5 Requirement 1: Verify face bounding box horizontally overlaps the hand box:
                # max(hand_x1, face_x1) < min(hand_x2, face_x2)
                hand_x1 = x
                hand_x2 = x1
                has_horizontal_overlap = max(hand_x1, face_x1) < min(hand_x2, face_x2)

                proposed_crop_y = max(0, min(y, f_bottom))
                proposed_delta = int(proposed_crop_y - orig_crop_y)

                clamp_decision = {
                    "hand_box": {"x": int(x), "y": int(y), "w": int(box_w), "h": int(box_h)},
                    "face_box": fb,
                    "has_horizontal_overlap": has_horizontal_overlap,
                    "proposed_crop_y": proposed_crop_y,
                    "proposed_delta": proposed_delta,
                    "decision": "not_triggering"
                }

                if f_bottom <= y1 and crop_y < f_bottom:
                    # Require horizontal overlap if enabled
                    if self.face_clamp_require_horizontal_overlap and not has_horizontal_overlap:
                        non_overlapping_clamps_ignored += 1
                        clamp_decision["decision"] = "rejected_no_horizontal_overlap"
                        face_clamp_decisions.append(clamp_decision)
                        if self.diagnostics_enabled:
                            print(f"[FaceClamp] Ignored face without horizontal overlap: hand_x=[{hand_x1}..{hand_x2}], face_x=[{face_x1}..{face_x2}], proposed_delta={proposed_delta}px", flush=True)
                        continue

                    if proposed_delta > 0:
                        if self.conservative_face_clamp:
                            # Phase 5 Requirement 2: Apply clamp limit to primary hand
                            if self.face_clamp_primary_hand_safeguard:
                                if is_two_handed:
                                    safeguard_h = primary_hand_h if primary_hand_h is not None else box_h
                                    safeguard_target = "primary_hand" if primary_hand_identified else "ambiguous_fallback_min_h"
                                else:
                                    safeguard_h = box_h
                                    safeguard_target = "single_hand"
                            else:
                                safeguard_h = box_h
                                safeguard_target = "combined_box"

                            clamp_ratio = proposed_delta / max(1, safeguard_h)
                            if clamp_ratio > self.max_face_clamp_ratio:
                                excessive_clamp_avoided = True
                                clamp_decision["decision"] = f"rejected_excessive_clamp_{clamp_ratio*100:.1f}%_on_{safeguard_target}"
                                face_clamp_decisions.append(clamp_decision)
                                if self.diagnostics_enabled:
                                    print(f"[FaceClampSafeguard] Excessive clamp avoided on {safeguard_target}: delta={proposed_delta}px ({clamp_ratio*100:.1f}% > {self.max_face_clamp_ratio*100:.0f}% of h={safeguard_h}px). Retaining unclamped crop.", flush=True)
                                continue  # Retain unclamped crop

                        crop_y = proposed_crop_y
                        if crop_y != orig_crop_y:
                            face_clamped = True
                            face_clamp_delta_y = proposed_delta
                            clamp_decision["decision"] = "applied"

                face_clamp_decisions.append(clamp_decision)

        square_crop_info = None
        if self.square_crop:
            preprocessing_mode = "square_crop"
            (vx1, vy1, vx2, vy2), (sq_x, sq_y, sq_w, sq_h), (pt, pb, pl, pr) = self.compute_square_crop(
                frame_bgr.shape, (crop_x, crop_y, crop_x1, crop_y1)
            )
            cropped = frame_bgr[vy1:vy2, vx1:vx2]
            if pt > 0 or pb > 0 or pl > 0 or pr > 0:
                cropped = cv2.copyMakeBorder(cropped, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            if cropped.size == 0 or cropped.shape[0] < 10 or cropped.shape[1] < 10:
                cropped = frame_bgr
            ch, cw = cropped.shape[:2]
            resized = cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_LINEAR)
            nw = 224
            nh = 224
            dx = 0
            dy = 0
            letterboxed = False
            square_crop_info = {
                "enabled": True,
                "original_box": {"x": int(x), "y": int(y), "w": int(box_w), "h": int(box_h)},
                "square_crop_coords": {"x": int(sq_x), "y": int(sq_y), "w": int(sq_w), "h": int(sq_h)},
                "padding": {"top": int(pt), "bottom": int(pb), "left": int(pl), "right": int(pr)},
                "output_dimensions": {"w": 224, "h": 224}
            }
        else:
            cropped = frame_bgr[crop_y:crop_y1, crop_x:crop_x1]
            if cropped.size == 0 or cropped.shape[0] < 10 or cropped.shape[1] < 10:
                cropped = frame_bgr

            # Aspect-Ratio Preserving Letterbox Resize:
            # Non-uniform stretching deforms curved hand gestures (e.g. C) into vertical sticks (V/S).
            # Letterboxing with black canvas preserves exact gesture geometry 100%.
            ch, cw = cropped.shape[:2]
            if self.direct_resize:
                preprocessing_mode = "direct_resize"
                resized = cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_LINEAR)
                nw = 224
                nh = 224
                dx = 0
                dy = 0
                letterboxed = False
            else:
                preprocessing_mode = "letterbox"
                scale = 224.0 / max(cw, ch)
                nw = max(1, int(cw * scale))
                nh = max(1, int(ch * scale))
                scaled = cv2.resize(cropped, (nw, nh), interpolation=cv2.INTER_LINEAR)

                resized = np.zeros((224, 224, 3), dtype=np.uint8)
                dx = (224 - nw) // 2
                dy = (224 - nh) // 2
                resized[dy:dy+nh, dx:dx+nw] = scaled
                letterboxed = (nw != 224 or nh != 224)

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
        raw_sign = self.class_map.get(str(top_idx), f"Class {top_idx}")
        raw_confidence = round(float(preds[top_idx]) * 100, 2)

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
        top_3 = [(p["sign"], p["probability"]) for p in prob_breakdown[:3]]

        # Optional Temporal Smoothing
        if self.temporal_smoothing_enabled:
            smooth_res = self.temporal_smoother.update(
                session_id=session_id,
                raw_sign=raw_sign,
                raw_confidence=raw_confidence,
                timestamp=timestamp,
                prob_dict={p["sign"]: p["probability"] for p in prob_breakdown}
            )
            predicted_sign = smooth_res["smoothed_sign"]
            confidence = smooth_res["smoothed_confidence"]
        else:
            predicted_sign = raw_sign
            confidence = raw_confidence
            smooth_res = {
                "smoothed_sign": raw_sign,
                "smoothed_confidence": raw_confidence,
                "raw_sign": raw_sign,
                "raw_confidence": raw_confidence,
                "window_size": 1,
                "window_votes": {raw_sign: 1},
                "is_smoothed": False,
                "tie_broken": False,
                "session_id": session_id
            }

        total_time_ms = round((time.time() - t_start) * 1000, 1)

        diag_info = {
            "frame_id": frame_id,
            "timestamp": round(timestamp, 3),
            "backend_detector": "yolo",
            "recognition_arch": "standard",
            "temporal_smoothing_enabled": self.temporal_smoothing_enabled,
            "yolo_detections_count": len(detections),
            "raw_yolo_count": len(raw_detections),
            "edge_detections_rejected": len(rejected_detections),
            "rejected_detections": rejected_detections,
            "hand_detected": hand_detected,
            "lap_hands_suppressed": lap_hands_suppressed,
            "is_two_handed": getattr(self, '_last_is_two_handed', False),
            "selected_box": {
                "x": int(x), "y": int(y), "w": int(box_w), "h": int(box_h),
                "conf": round(float(primary_conf), 3) if primary_conf is not None else 0.0
            },
            "face_clamped": face_clamped,
            "face_clamp_delta_y": face_clamp_delta_y,
            "excessive_face_clamp_avoided": excessive_clamp_avoided,
            "non_overlapping_clamps_ignored": non_overlapping_clamps_ignored,
            "face_clamp_decisions": face_clamp_decisions,
            "primary_hand_identified": primary_hand_identified,
            "hand_ambiguity_recorded": hand_ambiguity_recorded,
            "primary_hand_h": primary_hand_h,
            "crop_coords": {
                "x": int(crop_x), "y": int(crop_y), 
                "w": int(crop_x1 - crop_x), "h": int(crop_y1 - crop_y)
            },
            "crop_dimensions": {"w": int(cw), "h": int(ch)},
            "square_crop_info": square_crop_info,
            "resized_dimensions": {"w": int(nw), "h": int(nh)},
            "preprocessing_mode": preprocessing_mode,
            "tensor_shape": [1, 224, 224, 3],
            "letterboxed": letterboxed,
            "letterbox_padding": {
                "dx": int(dx), "dy": int(dy), 
                "scaled_w": int(nw), "scaled_h": int(nh)
            },
            "raw_predicted_letter": raw_sign,
            "raw_confidence": raw_confidence,
            "predicted_letter": predicted_sign,
            "confidence": confidence,
            "is_smoothed": smooth_res.get("is_smoothed", False),
            "window_votes": smooth_res.get("window_votes", {}),
            "top_3": top_3,
            "latency_ms": total_time_ms
        }

        if self.diagnostics_enabled:
            self._log_diagnostics(diag_info)

        result = {
            "predicted_sign": predicted_sign,
            "confidence": confidence,
            "hand_detected": hand_detected,
            "is_two_handed": getattr(self, '_last_is_two_handed', False),
            "detections": box_info,
            "probabilities": prob_breakdown,
            "diagnostics": diag_info,
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

    def _process_frame_mediapipe(self, frame_bgr, frame_id, timestamp, t_start, return_images=True, session_id="default_session"):
        """
        MediaPipe HandLandmarker inference path with optional late-fusion and temporal smoothing.
        Supports:
        - Robust resting lap-hand suppression.
        - Decision-level probability fusion (ISL_RECOGNITION_ARCH=late_fusion).
        - Multi-session rolling temporal smoothing (ISL_TEMPORAL_SMOOTHING=1).
        """
        import mediapipe as mp
        from landmark_feature_extractor import (
            extract_landmarks_from_mediapipe_result,
            TOTAL_FEATURE_DIM
        )
        h, w = frame_bgr.shape[:2]

        # Convert to RGB and prepare MediaPipe Image
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        t_det_start = time.time()
        res = self.mp_landmarker.detect(mp_img)
        t_det_end = time.time()
        det_latency_ms = round((t_det_end - t_det_start) * 1000, 1)

        # Extract landmarks and apply resting-hand suppression
        landmark_features, landmark_diag = extract_landmarks_from_mediapipe_result(
            res, suppress_lap_hands=True, return_diagnostics=True
        )

        active_hands_count = landmark_diag["selected_hand_count"]
        raw_hands_count = landmark_diag["initial_hand_count"]
        hand_detected = (active_hands_count > 0)
        is_two_handed = (active_hands_count >= 2)
        self._last_is_two_handed = is_two_handed

        # 1. Zero hands detected -> Clean early return (no false classification)
        if not hand_detected:
            if self.temporal_smoothing_enabled:
                self.temporal_smoother.update(session_id, "-", 0.0, timestamp=timestamp)
            t_total_ms = round((time.time() - t_start) * 1000, 1)
            diag_info = {
                "frame_id": frame_id,
                "timestamp": round(timestamp, 3),
                "backend_detector": "mediapipe",
                "recognition_arch": self.recognition_arch,
                "temporal_smoothing_enabled": self.temporal_smoothing_enabled,
                "hands_detected_count": 0,
                "hand_detected": False,
                "is_two_handed": False,
                "lap_hand_suppressed": landmark_diag.get("lap_hand_suppressed", False),
                "suppression_reasons": landmark_diag.get("suppression_reasons", []),
                "selected_box": None,
                "crop_coords": None,
                "square_crop_info": None,
                "predicted_letter": "-",
                "confidence": 0.0,
                "top_3": [],
                "latency_ms": t_total_ms
            }
            if self.diagnostics_enabled:
                print(f"[DIAG #{frame_id:04d}] MediaPipe=0 | Hands=0 | No hand detected | Latency={t_total_ms:.0f}ms", flush=True)

            res_dict = {
                "predicted_sign": "-",
                "confidence": 0.0,
                "hand_detected": False,
                "is_two_handed": False,
                "detections": [],
                "probabilities": [],
                "diagnostics": diag_info,
                "timing_ms": {
                    "detector": det_latency_ms,
                    "yolo": det_latency_ms,  # Backward compatibility for frontend latency card
                    "skin_segmentation": 0.0,
                    "squeezenet": 0.0,
                    "total": t_total_ms
                }
            }
            if return_images:
                blank_224 = np.zeros((224, 224, 3), dtype=np.uint8)
                res_dict["images"] = {
                    "original": self._encode_jpeg(frame_bgr),
                    "annotated": self._encode_jpeg(frame_bgr),
                    "cropped": self._encode_jpeg(blank_224),
                    "segmented": self._encode_jpeg(blank_224)
                }
            return res_dict

        # 2. Extract active hand landmarks and compute joint bounding box
        all_pts = []
        box_info = []
        for h_idx, h_lms in enumerate(res.hand_landmarks):
            # If a hand was suppressed as a resting lap hand, exclude it from active bounding box calculation
            if landmark_diag.get("lap_hand_suppressed", False) and len(res.hand_landmarks) > 1:
                wy = float(h_lms[0].y)
                if wy >= 0.72:
                    continue  # Skip suppressed resting lap hand

            pts = [(lm.x * w, lm.y * h) for lm in h_lms]
            all_pts.extend(pts)
            hx_min, hx_max = min(p[0] for p in pts), max(p[0] for p in pts)
            hy_min, hy_max = min(p[1] for p in pts), max(p[1] for p in pts)
            score = round(float(res.handedness[h_idx][0].score), 3) if res.handedness and len(res.handedness) > h_idx else 0.95
            category = res.handedness[h_idx][0].category_name if res.handedness and len(res.handedness) > h_idx else "hand"
            box_info.append({
                "name": f"hand_{category.lower()}",
                "confidence": round(score * 100, 1),
                "x": int(hx_min),
                "y": int(hy_min),
                "w": int(max(1, hx_max - hx_min)),
                "h": int(max(1, hy_max - hy_min)),
                "score": score
            })

        if not all_pts:
            # Fallback if all points were somehow filtered
            for h_lms in res.hand_landmarks:
                all_pts.extend([(lm.x * w, lm.y * h) for lm in h_lms])

        # 3. Compute 1:1 Isotropic Square Crop with 15% Margin
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        bw = max(1.0, max_x - min_x)
        bh = max(1.0, max_y - min_y)

        # Expansion factor: 1.0 + 2*0.15 = 1.30
        side = int(round(max(bw, bh) * 1.30))
        side = max(1, side)
        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0
        sq_x1 = int(round(cx - side / 2.0))
        sq_y1 = int(round(cy - side / 2.0))
        sq_x2 = sq_x1 + side
        sq_y2 = sq_y1 + side

        # Deterministic boundary shifting
        if sq_x1 < 0:
            shift = -sq_x1
            sq_x1 += shift
            sq_x2 += shift
        elif sq_x2 > w:
            shift = sq_x2 - w
            sq_x1 -= shift
            sq_x2 -= shift

        if sq_y1 < 0:
            shift = -sq_y1
            sq_y1 += shift
            sq_y2 += shift
        elif sq_y2 > h:
            shift = sq_y2 - h
            sq_y1 -= shift
            sq_y2 -= shift

        ptop = max(0, -sq_y1)
        pbottom = max(0, sq_y2 - h)
        pleft = max(0, -sq_x1)
        pright = max(0, sq_x2 - w)

        vy1 = max(0, sq_y1)
        vy2 = min(h, sq_y2)
        vx1 = max(0, sq_x1)
        vx2 = min(w, sq_x2)

        crop = frame_bgr[vy1:vy2, vx1:vx2]
        if ptop > 0 or pbottom > 0 or pleft > 0 or pright > 0:
            crop = cv2.copyMakeBorder(crop, ptop, pbottom, pleft, pright, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
            resized = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)

        # 4. Neural Network Classification
        t_nn_start = time.time()
        crop_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        inp = (crop_rgb.astype(np.float32) / 255.0).reshape(1, 224, 224, 3)

        cnn_model = getattr(self, "_mediapipe_cnn_model", self.model)
        preds_cnn = cnn_model.predict(inp, verbose=0)[0]
        t_nn_end = time.time()
        nn_latency_ms = round((t_nn_end - t_nn_start) * 1000, 1)

        # 5. Late Fusion Architecture or Standard CNN
        p_landmark = None
        landmark_top1 = None
        landmark_conf = 0.0

        if self.recognition_arch == "late_fusion" and self.landmark_rf_model is not None and self.landmark_scaler is not None:
            t_rf_start = time.time()
            scaled_lms = self.landmark_scaler.transform([landmark_features])
            p_landmark = self.landmark_rf_model.predict_proba(scaled_lms)[0]
            t_rf_end = time.time()
            rf_latency_ms = round((t_rf_end - t_rf_start) * 1000, 1)

            # Decision-level fusion: P = alpha * P_landmark + (1 - alpha) * P_cnn
            alpha = self.late_fusion_alpha
            preds_final = alpha * p_landmark + (1.0 - alpha) * preds_cnn

            lm_top_idx = int(np.argmax(p_landmark))
            landmark_top1 = self.class_map.get(str(lm_top_idx), f"Class {lm_top_idx}")
            landmark_conf = round(float(p_landmark[lm_top_idx]) * 100, 2)
        else:
            preds_final = preds_cnn

        top_idx = int(np.argmax(preds_final))
        raw_sign = self.class_map.get(str(top_idx), f"Class {top_idx}")
        raw_confidence = round(float(preds_final[top_idx]) * 100, 2)

        prob_breakdown = []
        for idx_str, sign_name in self.class_map.items():
            idx = int(idx_str)
            prob_breakdown.append({
                "class_id": idx,
                "sign": sign_name,
                "probability": round(float(preds_final[idx]) * 100, 2)
            })
        prob_breakdown = sorted(prob_breakdown, key=lambda x: x["probability"], reverse=True)
        top_3 = [(p["sign"], p["probability"]) for p in prob_breakdown[:3]]

        # 6. Optional Temporal Smoothing
        if self.temporal_smoothing_enabled:
            smooth_res = self.temporal_smoother.update(
                session_id=session_id,
                raw_sign=raw_sign,
                raw_confidence=raw_confidence,
                timestamp=timestamp,
                prob_dict={p["sign"]: p["probability"] for p in prob_breakdown}
            )
            predicted_sign = smooth_res["smoothed_sign"]
            confidence = smooth_res["smoothed_confidence"]
        else:
            predicted_sign = raw_sign
            confidence = raw_confidence
            smooth_res = {
                "smoothed_sign": raw_sign,
                "smoothed_confidence": raw_confidence,
                "raw_sign": raw_sign,
                "raw_confidence": raw_confidence,
                "window_size": 1,
                "window_votes": {raw_sign: 1},
                "is_smoothed": False,
                "tie_broken": False,
                "session_id": session_id
            }

        total_time_ms = round((time.time() - t_start) * 1000, 1)

        cnn_top_idx = int(np.argmax(preds_cnn))
        cnn_top1 = self.class_map.get(str(cnn_top_idx), f"Class {cnn_top_idx}")
        cnn_conf = round(float(preds_cnn[cnn_top_idx]) * 100, 2)

        # 7. Build Diagnostic Info
        diag_info = {
            "frame_id": frame_id,
            "timestamp": round(timestamp, 3),
            "backend_detector": "mediapipe",
            "recognition_arch": self.recognition_arch,
            "temporal_smoothing_enabled": self.temporal_smoothing_enabled,
            "hands_detected_count": active_hands_count,
            "raw_hands_count": raw_hands_count,
            "hand_detected": True,
            "is_two_handed": is_two_handed,
            "lap_hand_suppressed": landmark_diag.get("lap_hand_suppressed", False),
            "suppression_reasons": landmark_diag.get("suppression_reasons", []),
            "selected_box": {
                "x": int(min_x), "y": int(min_y), "w": int(bw), "h": int(bh),
                "conf": box_info[0]["score"] if box_info else 1.0
            },
            "crop_coords": {"x": int(sq_x1), "y": int(sq_y1), "w": int(side), "h": int(side)},
            "square_crop_info": {
                "enabled": True,
                "square_crop_coords": {"x": int(sq_x1), "y": int(sq_y1), "w": int(side), "h": int(side)},
                "padding": {"top": int(ptop), "bottom": int(pbottom), "left": int(pleft), "right": int(pright)}
            },
            "preprocessing_mode": "mediapipe_square_crop",
            "tensor_shape": [1, 224, 224, 3],
            "cnn_top1": cnn_top1,
            "cnn_confidence": cnn_conf,
            "landmark_top1": landmark_top1,
            "landmark_confidence": landmark_conf,
            "raw_predicted_letter": raw_sign,
            "raw_confidence": raw_confidence,
            "predicted_letter": predicted_sign,
            "confidence": confidence,
            "is_smoothed": smooth_res.get("is_smoothed", False),
            "window_votes": smooth_res.get("window_votes", {}),
            "top_3": top_3,
            "latency_ms": total_time_ms
        }
        if self.diagnostics_enabled:
            hands_str = "2H" if is_two_handed else "1H"
            arch_str = f"Arch={self.recognition_arch}"
            smooth_str = f" | Smooth={predicted_sign}({confidence:.1f}%)" if self.temporal_smoothing_enabled else ""
            lap_str = " | LapSupp=T" if landmark_diag.get("lap_hand_suppressed", False) else ""
            top3_str = ",".join([f"{s}:{p:.1f}%" for s, p in top_3])
            print(f"[DIAG #{frame_id:04d}] MediaPipe={active_hands_count} ({hands_str}) | {arch_str}{lap_str} | "
                  f"Raw={raw_sign} ({raw_confidence:.1f}%){smooth_str} | Top3=[{top3_str}] | Latency={total_time_ms:.0f}ms", flush=True)

        annotated_frame = frame_bgr.copy()
        if return_images:
            cv2.rectangle(annotated_frame, (max(0, sq_x1), max(0, sq_y1)), (min(w, sq_x2), min(h, sq_y2)), (0, 255, 0), 2)
            for d in box_info:
                cv2.rectangle(annotated_frame, (d["x"], d["y"]), (d["x"] + d["w"], d["y"] + d["h"]), (255, 0, 0), 1)

        res_dict = {
            "predicted_sign": predicted_sign,
            "confidence": confidence,
            "hand_detected": True,
            "is_two_handed": is_two_handed,
            "detections": box_info,
            "probabilities": prob_breakdown,
            "diagnostics": diag_info,
            "timing_ms": {
                "detector": det_latency_ms,
                "yolo": det_latency_ms,  # Compatibility with frontend latency display
                "skin_segmentation": 0.0,
                "squeezenet": nn_latency_ms,
                "total": total_time_ms
            }
        }
        if return_images:
            res_dict["images"] = {
                "original": self._encode_jpeg(frame_bgr),
                "annotated": self._encode_jpeg(annotated_frame),
                "cropped": self._encode_jpeg(resized),
                "segmented": self._encode_jpeg(resized)
            }
        return res_dict

    def _run_shadow_inference(self, frame_bgr, prod_res):
        """
        Runs Model C in shadow mode on the same live frame without mutating or overriding production results.
        Injects structured comparison into prod_res['diagnostics']['shadow_model_c'].
        """
        t_shadow_start = time.time()
        try:
            if self.model_c_recognizer is None or not self.model_c_recognizer.is_loaded:
                project_root = os.path.dirname(os.path.abspath(__file__))
                self._load_model_c(project_root)

            import mediapipe as mp
            if self.mediapipe_delegate in ("gpu", "metal"):
                rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGBA, data=rgba)
            else:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            t_det_start = time.time()
            res = self.mp_landmarker.detect(mp_img)
            det_latency_ms = round((time.time() - t_det_start) * 1000, 2)

            model_c_out = self.model_c_recognizer.predict_mediapipe_result(res)
            total_shadow_ms = round((time.time() - t_shadow_start) * 1000, 2)

            shadow_letter = model_c_out["predicted_letter"]
            shadow_conf = model_c_out["confidence"]
            prod_pred = prod_res.get("predicted_sign", "-")
            prod_conf = prod_res.get("confidence", 0.0)

            shadow_diag = {
                "enabled": True,
                "status": "success",
                "model_name": "Model C (Occlusion-Aware Geometry Random Forest)",
                "version": self.model_c_recognizer.version,
                "predicted_letter": shadow_letter,
                "confidence": shadow_conf,
                "top_3": model_c_out["top_3"],
                "probabilities": model_c_out["probabilities"],
                "hand_count": model_c_out["hand_count"],
                "feature_status": model_c_out["feature_status"],
                "landmark_detection_latency_ms": det_latency_ms,
                "feature_extraction_latency_ms": model_c_out.get("feature_extraction_ms", 0.0),
                "classifier_inference_latency_ms": model_c_out.get("inference_ms", 0.0),
                "total_shadow_latency_ms": total_shadow_ms,
                "production_predicted_letter": prod_pred,
                "production_confidence": prod_conf,
                "match_production": (shadow_letter == prod_pred if prod_pred != "-" else False)
            }
        except Exception as e:
            total_shadow_ms = round((time.time() - t_shadow_start) * 1000, 2)
            shadow_diag = {
                "enabled": True,
                "status": "error",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "model_name": "Model C",
                "total_shadow_latency_ms": total_shadow_ms,
                "match_production": False
            }

        if "diagnostics" not in prod_res or not isinstance(prod_res["diagnostics"], dict):
            prod_res["diagnostics"] = {}
        prod_res["diagnostics"]["shadow_model_c"] = shadow_diag

    def _process_frame_model_c(self, frame_bgr, frame_id, timestamp, t_start, return_images=True, session_id="default_session"):
        """
        Full experimental recognition path using Model C (MediaPipe + 280d Geometry/Occlusion RF).
        Instruments all 12 execution stages from frame receipt to returned prediction.
        """
        import mediapipe as mp
        from landmark_model_c import extract_landmarks_model_c, ALPHABET

        t_pipeline_start = time.perf_counter()

        # Stage 1: Raw frame receipt
        t0 = time.perf_counter()
        t1 = time.perf_counter()
        stage_receipt_ms = (t1 - t0) * 1000.0

        # Stage 2: Input validation
        t0 = time.perf_counter()
        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0 or frame_bgr.ndim != 3 or frame_bgr.shape[2] not in (3, 4):
            raise ValueError("Invalid frame: empty or non-ndarray input received")
        h, w = frame_bgr.shape[:2]
        t1 = time.perf_counter()
        stage_val_ms = (t1 - t0) * 1000.0

        if self.model_c_recognizer is None or not self.model_c_recognizer.is_loaded:
            project_root = os.path.dirname(os.path.abspath(__file__))
            self._load_model_c(project_root)

        # Stage 3: Image copy or memory conversion (only if return_images is requested)
        t0 = time.perf_counter()
        annotated_frame = frame_bgr.copy() if return_images else None
        t1 = time.perf_counter()
        stage_copy_ms = (t1 - t0) * 1000.0

        # Stage 4: Resolution conversion or resizing (opt-in downsampling)
        t0 = time.perf_counter()
        target_res = self._get_target_resolution(w, h)
        if target_res is not None and target_res != (w, h):
            proc_frame = cv2.resize(frame_bgr, target_res, interpolation=cv2.INTER_LINEAR)
            proc_w, proc_h = target_res
        else:
            proc_frame = frame_bgr
            proc_w, proc_h = w, h
        t1 = time.perf_counter()
        stage_resize_ms = (t1 - t0) * 1000.0

        # Stage 5: BGR/RGB/RGBA conversion (with compatible buffer reuse)
        t0 = time.perf_counter()
        is_gpu = self.mediapipe_delegate in ("gpu", "metal")
        channels = 4 if is_gpu else 3
        conv_code = cv2.COLOR_BGR2RGBA if is_gpu else cv2.COLOR_BGR2RGB

        # Safe buffer reuse: preallocate or reallocate if dimensions/channels change
        if (getattr(self, "_model_c_color_buf", None) is None or
                self._model_c_color_buf.shape != (proc_h, proc_w, channels) or
                self._model_c_color_buf.dtype != np.uint8):
            self._model_c_color_buf = np.empty((proc_h, proc_w, channels), dtype=np.uint8)

        color_frame = cv2.cvtColor(proc_frame, conv_code, dst=self._model_c_color_buf)
        t1 = time.perf_counter()
        stage_color_conv_ms = (t1 - t0) * 1000.0

        # Stage 6: MediaPipe image construction
        t0 = time.perf_counter()
        if is_gpu:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGBA, data=color_frame)
        else:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=color_frame)
        t1 = time.perf_counter()
        stage_mp_image_ms = (t1 - t0) * 1000.0

        # Stage 7: HandLandmarker detection
        t0 = time.perf_counter()
        res = self.mp_landmarker.detect(mp_img)
        t1 = time.perf_counter()
        stage_detection_ms = (t1 - t0) * 1000.0

        # Stage 8: Landmark postprocessing (bounding boxes & annotations)
        t0 = time.perf_counter()
        box_info = []
        if res.hand_landmarks:
            for h_pts in res.hand_landmarks:
                xs = [p.x * w for p in h_pts]
                ys = [p.y * h for p in h_pts]
                min_x, max_x = max(0, int(min(xs))), min(w, int(max(xs)))
                min_y, max_y = max(0, int(min(ys))), min(h, int(max(ys)))
                box_info.append({
                    "name": "hand",
                    "score": 1.0,
                    "x": min_x,
                    "y": min_y,
                    "w": max_x - min_x,
                    "h": max_y - min_y
                })
                if return_images and annotated_frame is not None:
                    cv2.rectangle(annotated_frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
        t1 = time.perf_counter()
        stage_postproc_ms = (t1 - t0) * 1000.0

        # Stage 9: Geometry and occlusion feature extraction (280 dims)
        t0 = time.perf_counter()
        features_280, hand_count, feat_status = extract_landmarks_model_c(res)
        t1 = time.perf_counter()
        stage_feat_ms = (t1 - t0) * 1000.0

        hand_detected = (hand_count > 0)
        is_two_handed = (hand_count >= 2)
        self._last_is_two_handed = is_two_handed

        # Stage 10 & 11: Scaler transformation and Random Forest prediction
        stage_scaler_ms = 0.0
        stage_clf_ms = 0.0

        if not hand_detected:
            if self.temporal_smoothing_enabled:
                self.temporal_smoother.update(session_id, "-", 0.0, timestamp=timestamp)
            raw_sign = "-"
            raw_confidence = 0.0
            predicted_sign = "-"
            confidence = 0.0
            prob_breakdown = []
            smooth_res = {
                "smoothed_sign": "-",
                "smoothed_confidence": 0.0,
                "raw_sign": "-",
                "raw_confidence": 0.0,
                "window_size": 1,
                "window_votes": {"-": 1},
                "is_smoothed": False,
                "session_id": session_id
            }
            top_3 = []
        else:
            # Stage 10: Scaler transformation
            t0 = time.perf_counter()
            vec = np.asarray(features_280, dtype=np.float32).reshape(1, -1)
            scaled_x = self.model_c_recognizer.scaler.transform(vec)
            t1 = time.perf_counter()
            stage_scaler_ms = (t1 - t0) * 1000.0

            # Stage 11: Classifier prediction
            t0 = time.perf_counter()
            probs = self.model_c_recognizer.clf.predict_proba(scaled_x)[0]
            t1 = time.perf_counter()
            stage_clf_ms = (t1 - t0) * 1000.0

            pred_idx = int(np.argmax(probs))
            raw_sign = ALPHABET[pred_idx]
            raw_confidence = round(float(probs[pred_idx]) * 100, 2)
            prob_dict = {ALPHABET[i]: round(float(probs[i]) * 100, 2) for i in range(26)}
            top3_indices = np.argsort(probs)[-3:][::-1]
            top_3 = [(ALPHABET[int(i)], round(float(probs[i]) * 100, 2)) for i in top3_indices]

            prob_breakdown = [
                {"class_id": ord(c) - ord('A'), "sign": c, "probability": p}
                for c, p in prob_dict.items()
            ]
            prob_breakdown = sorted(prob_breakdown, key=lambda x: x["probability"], reverse=True)

            if self.temporal_smoothing_enabled:
                smooth_res = self.temporal_smoother.update(
                    session_id=session_id,
                    raw_sign=raw_sign,
                    raw_confidence=raw_confidence,
                    timestamp=timestamp,
                    prob_dict=prob_dict
                )
                predicted_sign = smooth_res["smoothed_sign"]
                confidence = smooth_res["smoothed_confidence"]
            else:
                predicted_sign = raw_sign
                confidence = raw_confidence
                smooth_res = {
                    "smoothed_sign": raw_sign,
                    "smoothed_confidence": raw_confidence,
                    "raw_sign": raw_sign,
                    "raw_confidence": raw_confidence,
                    "window_size": 1,
                    "window_votes": {raw_sign: 1},
                    "is_smoothed": False,
                    "session_id": session_id
                }

        # Stage 12: Result formatting and return
        t0 = time.perf_counter()

        stage_sum_ms = (
            stage_receipt_ms + stage_val_ms + stage_copy_ms +
            stage_resize_ms + stage_color_conv_ms + stage_mp_image_ms +
            stage_detection_ms + stage_postproc_ms + stage_feat_ms +
            stage_scaler_ms + stage_clf_ms
        )

        diag_info = {
            "frame_id": frame_id,
            "timestamp": round(timestamp, 3),
            "backend_detector": "mediapipe",
            "recognition_arch": "model_c",
            "mediapipe_delegate": self.mediapipe_delegate,
            "input_resolution_setting": getattr(self, "model_c_input_resolution", "native"),
            "input_frame_shape": [h, w],
            "processed_shape": [proc_h, proc_w],
            "temporal_smoothing_enabled": self.temporal_smoothing_enabled,
            "hands_detected_count": hand_count,
            "hand_detected": hand_detected,
            "is_two_handed": is_two_handed,
            "feature_status": feat_status,
            "feature_schema": self.model_c_recognizer.schema_name,
            "feature_dimensions": self.model_c_recognizer.feature_dim,
            "model_version": self.model_c_recognizer.version,
            "raw_predicted_letter": raw_sign,
            "raw_confidence": raw_confidence,
            "predicted_letter": predicted_sign,
            "confidence": confidence,
            "is_smoothed": smooth_res.get("is_smoothed", False),
            "window_votes": smooth_res.get("window_votes", {}),
            "top_3": top_3,
            "stage_timings_ms": {
                "1_receipt": round(stage_receipt_ms, 3),
                "2_validation": round(stage_val_ms, 3),
                "3_copy": round(stage_copy_ms, 3),
                "4_resize": round(stage_resize_ms, 3),
                "5_color_conversion": round(stage_color_conv_ms, 3),
                "6_mp_image_construction": round(stage_mp_image_ms, 3),
                "7_detection": round(stage_detection_ms, 3),
                "8_postprocessing": round(stage_postproc_ms, 3),
                "9_feature_extraction": round(stage_feat_ms, 3),
                "10_scaler_transform": round(stage_scaler_ms, 3),
                "11_classifier_prediction": round(stage_clf_ms, 3),
                "sum_stages": round(stage_sum_ms, 3)
            }
        }

        t1 = time.perf_counter()
        stage_formatting_ms = (t1 - t0) * 1000.0
        stage_sum_ms += stage_formatting_ms
        diag_info["stage_timings_ms"]["12_formatting"] = round(stage_formatting_ms, 3)
        diag_info["stage_timings_ms"]["sum_stages"] = round(stage_sum_ms, 3)

        total_elapsed_ms = (time.perf_counter() - t_pipeline_start) * 1000.0
        diag_info["latency_ms"] = round(total_elapsed_ms, 2)

        if self.diagnostics_enabled:
            hands_str = "2H" if is_two_handed else ("1H" if hand_detected else "0H")
            smooth_str = f" | Smooth={predicted_sign}({confidence:.1f}%)" if self.temporal_smoothing_enabled else ""
            top3_str = ",".join([f"{s}:{p:.1f}%" for s, p in top_3])
            print(f"[DIAG #{frame_id:04d}] ModelC={hand_count} ({hands_str}) | "
                  f"Raw={raw_sign} ({raw_confidence:.1f}%){smooth_str} | Top3=[{top3_str}] | Latency={total_elapsed_ms:.1f}ms", flush=True)

        res_dict = {
            "predicted_sign": predicted_sign,
            "confidence": confidence,
            "hand_detected": hand_detected,
            "is_two_handed": is_two_handed,
            "detections": box_info,
            "probabilities": prob_breakdown,
            "diagnostics": diag_info,
            "timing_ms": {
                "detector": round(stage_detection_ms, 2),
                "yolo": round(stage_detection_ms, 2),  # Frontend compatibility
                "skin_segmentation": 0.0,
                "squeezenet": round(stage_scaler_ms + stage_clf_ms, 2),
                "feature_extraction": round(stage_feat_ms, 2),
                "scaler_transform": round(stage_scaler_ms, 2),
                "model_c_classifier": round(stage_clf_ms, 2),
                "color_conversion": round(stage_color_conv_ms, 2),
                "resolution_resize": round(stage_resize_ms, 2),
                "stages_sum": round(stage_sum_ms, 2),
                "total": round(total_elapsed_ms, 2)
            }
        }
        if return_images and annotated_frame is not None:
            res_dict["images"] = {
                "original": self._encode_jpeg(frame_bgr),
                "annotated": self._encode_jpeg(annotated_frame),
                "cropped": "",
                "segmented": ""
            }
        elif return_images:
            res_dict["images"] = {
                "original": self._encode_jpeg(frame_bgr),
                "annotated": "",
                "cropped": "",
                "segmented": ""
            }
        return res_dict

    def _encode_jpeg(self, img_bgr):
        _, buffer = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')

# Global singleton
pipeline_instance = ISLPipeline()
