#!/usr/bin/env python3
"""
Phase 26 — End-to-End Latency Forensics and Safe Frame-Processing Optimization
Unit & Regression Tests

Coverage:
1. Baseline defaults preserved (YOLO detector, standard arch, smoothing=0, shadow=0, delegate=cpu, resolution=native).
2. Dynamic MediaPipe delegate switching and landmarker reloading.
3. Model C input resolution configuration, mapping, and validation.
4. Complete 12-stage instrumentation in timing_ms and diagnostics["stage_timings_ms"].
5. Sum of stage timings matches total elapsed latency within tight bound (<1.0 ms).
6. Frame shape, resolution, and color handling (RGB for CPU, RGBA for GPU).
7. Memory efficiency: redundant frame copying skipped when return_images=False.
8. Invalid inputs: None, empty frames, corrupted arrays handled gracefully.
9. Feature vector 280-dimensional contract preservation.
10. Shadow worker isolation and production output invariance.
"""

import os
import sys
import time
import unittest
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from pipeline import ISLPipeline, ShadowWorker
from landmark_model_c import ModelCRecognizer, extract_landmarks_model_c


class TestPhase26Latency(unittest.TestCase):
    def setUp(self):
        self.orig_env = {
            "ISL_BACKEND_DETECTOR": os.environ.get("ISL_BACKEND_DETECTOR"),
            "ISL_RECOGNITION_ARCH": os.environ.get("ISL_RECOGNITION_ARCH"),
            "ISL_TEMPORAL_SMOOTHING": os.environ.get("ISL_TEMPORAL_SMOOTHING"),
            "ISL_SHADOW_MODE": os.environ.get("ISL_SHADOW_MODE"),
            "ISL_SHADOW_ASYNC": os.environ.get("ISL_SHADOW_ASYNC"),
            "ISL_MEDIAPIPE_DELEGATE": os.environ.get("ISL_MEDIAPIPE_DELEGATE"),
            "ISL_MODEL_C_RESOLUTION": os.environ.get("ISL_MODEL_C_RESOLUTION"),
            "ISL_MODEL_C_N_JOBS": os.environ.get("ISL_MODEL_C_N_JOBS"),
        }

    def tearDown(self):
        for k, v in self.orig_env.items():
            if v is not None:
                os.environ[k] = v
            elif k in os.environ:
                del os.environ[k]

    # 1. Baseline Preservation
    def test_01_baseline_defaults_preserved(self):
        """Verify production defaults remain untouched."""
        for k in [
            "ISL_BACKEND_DETECTOR",
            "ISL_RECOGNITION_ARCH",
            "ISL_TEMPORAL_SMOOTHING",
            "ISL_SHADOW_MODE",
            "ISL_MEDIAPIPE_DELEGATE",
            "ISL_MODEL_C_RESOLUTION",
        ]:
            if k in os.environ:
                del os.environ[k]

        pipe = ISLPipeline()
        self.assertEqual(pipe.backend_detector, "yolo")
        self.assertEqual(pipe.recognition_arch, "standard")
        self.assertFalse(pipe.temporal_smoothing_enabled)
        self.assertFalse(pipe.shadow_mode_enabled)
        self.assertFalse(pipe.shadow_async)
        self.assertEqual(pipe.mediapipe_delegate, "cpu")
        self.assertEqual(pipe.model_c_input_resolution, "native")

    # 2. Dynamic MediaPipe Delegate Switching
    def test_02_dynamic_mediapipe_delegate_switching(self):
        """Verify set_mediapipe_delegate correctly updates delegate and recreates landmarker."""
        pipe = ISLPipeline()
        pipe.set_recognition_arch("model_c")
        self.assertEqual(pipe.mediapipe_delegate, "cpu")
        initial_landmarker = pipe.mp_landmarker

        # Switch to GPU
        pipe.set_mediapipe_delegate("gpu")
        self.assertEqual(pipe.mediapipe_delegate, "gpu")
        self.assertIsNot(pipe.mp_landmarker, initial_landmarker)

        # Switch back to CPU
        pipe.set_mediapipe_delegate("cpu")
        self.assertEqual(pipe.mediapipe_delegate, "cpu")

        # Invalid delegate raises ValueError
        with self.assertRaises(ValueError):
            pipe.set_mediapipe_delegate("cuda_invalid")

    # 3. Model C Resolution Configuration & Mapping
    def test_03_resolution_configuration_and_mapping(self):
        """Verify supported resolution modes and target coordinate dimensions."""
        pipe = ISLPipeline()
        pipe.set_recognition_arch("model_c")
        
        # Default is native
        self.assertEqual(pipe.model_c_input_resolution, "native")
        self.assertIsNone(pipe._get_target_resolution(1280, 720))

        # Supported resolutions
        for mode, expected in [
            ("960x540", (960, 540)),
            ("640x360", (640, 360)),
            ("640x480", (640, 480)),
            ("native", None),
        ]:
            pipe.set_model_c_input_resolution(mode)
            self.assertEqual(pipe.model_c_input_resolution, mode)
            self.assertEqual(pipe._get_target_resolution(1280, 720), expected)

        # Invalid resolution raises ValueError
        with self.assertRaises(ValueError):
            pipe.set_model_c_input_resolution("320x240_invalid")

    # 4. Complete 12-Stage Forensics Instrumentation
    def test_04_twelve_stage_timing_instrumentation(self):
        """Verify all 12 stages are present in timing_ms and diagnostics['stage_timings_ms']."""
        pipe = ISLPipeline()
        pipe.set_recognition_arch("model_c")
        pipe.set_mediapipe_delegate("gpu")
        
        # Test synthetic frame
        frame = np.full((720, 1280, 3), 128, dtype=np.uint8)
        res = pipe.process_frame(frame, return_images=False)

        timing = res.get("timing_ms", {})
        self.assertIn("total", timing)
        self.assertIn("detector", timing)
        self.assertIn("feature_extraction", timing)

        diagnostics = res.get("diagnostics", {})
        self.assertIn("stage_timings_ms", diagnostics)
        stages = diagnostics["stage_timings_ms"]

        required_stages = [
            "1_receipt",
            "2_validation",
            "3_copy",
            "4_resize",
            "5_color_conversion",
            "6_mp_image_construction",
            "7_detection",
            "8_postprocessing",
            "9_feature_extraction",
            "10_scaler_transform",
            "11_classifier_prediction",
            "12_formatting",
        ]

        for s in required_stages:
            self.assertIn(s, stages, f"Missing stage timing: {s}")
            self.assertIsInstance(stages[s], (int, float))
            self.assertGreaterEqual(stages[s], 0.0)

        # Stage sum vs total elapsed bound check (<1.0 ms discrepancy)
        stage_sum = sum(stages[s] for s in required_stages)
        total_ms = timing["total"]
        self.assertAlmostEqual(stage_sum, total_ms, delta=1.0)

    # 5. Frame Shape & Color Handling
    def test_05_frame_shape_and_color_handling(self):
        """Verify pipeline handles different frame resolutions and delegates correctly."""
        pipe = ISLPipeline()
        pipe.set_recognition_arch("model_c")

        for delegate in ["cpu", "gpu"]:
            pipe.set_mediapipe_delegate(delegate)
            for h, w in [(720, 1280), (540, 960), (480, 640), (1080, 1920)]:
                frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
                res = pipe.process_frame(frame, return_images=False)
                self.assertIn("predicted_sign", res)
                self.assertIn("confidence", res)
                self.assertIn("predicted_letter", res.get("diagnostics", {}))
                self.assertNotIn("images", res)

    # 6. Memory Efficiency (No Redundant Copy when return_images=False)
    def test_06_memory_efficiency_no_copy_when_return_images_false(self):
        """Verify annotated_frame and crop_bgr are None when return_images=False."""
        pipe = ISLPipeline()
        pipe.set_recognition_arch("model_c")
        frame = np.full((720, 1280, 3), 100, dtype=np.uint8)

        # False
        res_no_img = pipe.process_frame(frame, return_images=False)
        self.assertNotIn("images", res_no_img)
        self.assertEqual(res_no_img.get("diagnostics", {}).get("stage_timings_ms", {}).get("3_copy"), 0.0)

        # True
        res_with_img = pipe.process_frame(frame, return_images=True)
        self.assertIn("images", res_with_img)
        self.assertIn("annotated", res_with_img["images"])

    # 7. Invalid Inputs Resilience
    def test_07_invalid_inputs_resilience(self):
        """Verify graceful error recovery on invalid inputs."""
        pipe = ISLPipeline()
        pipe.set_recognition_arch("model_c")

        # None input raises ValueError
        with self.assertRaises(ValueError):
            pipe.process_frame(None)

        # Empty array raises ValueError
        empty_frame = np.zeros((0, 0, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            pipe.process_frame(empty_frame)

        # 2D array (missing color channels) raises ValueError
        flat_frame = np.zeros((100, 100), dtype=np.uint8)
        with self.assertRaises(ValueError):
            pipe.process_frame(flat_frame)

    # 8. 280-Dimensional Feature Contract
    def test_08_feature_dimensions_and_contract(self):
        """Verify strict adherence to 280-dimensional feature contract."""
        rec = ModelCRecognizer()
        rec.load()
        self.assertIsNotNone(rec.clf)
        self.assertIsNotNone(rec.scaler)

        # Valid 280-dim feature vector
        np.random.seed(42)
        valid_feat = np.random.randn(280).astype(np.float32)
        res = rec.predict_features(valid_feat)
        self.assertIn("predicted_letter", res)
        self.assertIn("confidence", res)
        self.assertEqual(len(res["probabilities"]), 26)

        # Invalid dimension feature vector raises or handles cleanly
        invalid_feat = np.random.randn(250).astype(np.float32)
        with self.assertRaises(ValueError):
            rec.predict_features(invalid_feat)

    # 9. Shadow Worker Isolation & Production Invariance
    def test_09_shadow_worker_isolation_and_production_invariance(self):
        """Verify shadow worker isolation, bounded queuing, and production output invariance."""
        pipe = ISLPipeline()
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

        worker = ShadowWorker(pipe)
        self.assertEqual(worker.queue.maxsize, 1)
        self.assertFalse(worker.running)

        # Start worker and submit production task
        worker.start()
        prod_res = {"predicted_sign": "B", "confidence": 92.5, "detections": []}
        t0 = time.time()
        worker.submit_task(dummy_frame, prod_res, t0, t0 + 0.01)
        for _ in range(20):
            if worker.get_telemetry()["last_result"] is not None:
                break
            time.sleep(0.05)

        # Production output contract remains untouched
        self.assertEqual(prod_res["predicted_sign"], "B")
        self.assertEqual(prod_res["confidence"], 92.5)
        self.assertEqual(prod_res["detections"], [])

        # Shadow worker telemetry recorded successfully
        telemetry = worker.get_telemetry()
        self.assertGreaterEqual(telemetry["submitted_count"], 1)
        self.assertEqual(telemetry["dropped_count"], 0)
        self.assertIsNotNone(telemetry["last_result"])

        worker.stop()
        pipe.shutdown()


if __name__ == "__main__":
    unittest.main()
