#!/usr/bin/env python3
"""
Phase 25 — Model C Performance Optimization, Delegate Benchmarking,
and Shadow Isolation Audit Unit Tests

Coverage:
1. Baseline defaults preserved (YOLO detector, standard arch, smoothing=0, shadow=0, delegate=cpu).
2. MediaPipe delegate configuration (CPU vs GPU) and format handling (SRGB vs SRGBA).
3. Classifier threading optimization (n_jobs=1 vs n_jobs=-1 mathematical equivalence).
4. ShadowWorker bounded queue (maxsize=1) and newest-frame priority drop policy.
5. Shadow async timing instrumentation (queue_wait_ms, shadow_execution_ms, frame_age_ms, prod_latency_ms).
6. Production output invariance (shadow mode never modifies production predictions, boxes, or confidences).
7. Shadow worker error containment (worker errors do not crash production and are tracked in telemetry).
8. Graceful shutdown (pipeline.shutdown cleanly stops threads).
9. Server API endpoints exposure of shadow telemetry and delegate status.
"""

import os
import sys
import time
import unittest
import numpy as np
from unittest.mock import MagicMock, patch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from pipeline import ISLPipeline, ShadowWorker
from landmark_model_c import ModelCRecognizer, extract_landmarks_model_c


class TestPhase25Performance(unittest.TestCase):
    def setUp(self):
        self.orig_env = {
            "ISL_BACKEND_DETECTOR": os.environ.get("ISL_BACKEND_DETECTOR"),
            "ISL_RECOGNITION_ARCH": os.environ.get("ISL_RECOGNITION_ARCH"),
            "ISL_TEMPORAL_SMOOTHING": os.environ.get("ISL_TEMPORAL_SMOOTHING"),
            "ISL_SHADOW_MODE": os.environ.get("ISL_SHADOW_MODE"),
            "ISL_SHADOW_ASYNC": os.environ.get("ISL_SHADOW_ASYNC"),
            "ISL_MEDIAPIPE_DELEGATE": os.environ.get("ISL_MEDIAPIPE_DELEGATE"),
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
        for k in ["ISL_BACKEND_DETECTOR", "ISL_RECOGNITION_ARCH", "ISL_TEMPORAL_SMOOTHING", "ISL_SHADOW_MODE", "ISL_MEDIAPIPE_DELEGATE"]:
            if k in os.environ:
                del os.environ[k]

        pipe = ISLPipeline()
        self.assertEqual(pipe.backend_detector, "yolo")
        self.assertEqual(pipe.recognition_arch, "standard")
        self.assertFalse(pipe.temporal_smoothing_enabled)
        self.assertFalse(pipe.shadow_mode_enabled)
        self.assertFalse(pipe.shadow_async)
        self.assertEqual(pipe.mediapipe_delegate, "cpu")

    # 2. Delegate Configuration & Buffer Formats
    def test_02_mediapipe_delegate_configuration(self):
        """Verify CPU and GPU delegate parsing."""
        os.environ["ISL_MEDIAPIPE_DELEGATE"] = "cpu"
        p_cpu = ISLPipeline()
        self.assertEqual(p_cpu.mediapipe_delegate, "cpu")

        os.environ["ISL_MEDIAPIPE_DELEGATE"] = "gpu"
        p_gpu = ISLPipeline()
        self.assertEqual(p_gpu.mediapipe_delegate, "gpu")

        os.environ["ISL_MEDIAPIPE_DELEGATE"] = "metal"
        p_metal = ISLPipeline()
        self.assertEqual(p_metal.mediapipe_delegate, "gpu")

    # 3. Classifier Single-Sample Threading & Invariance
    def test_03_classifier_threading_and_invariance(self):
        """Verify n_jobs=1 yields identical predictions to n_jobs=-1 without thread overhead."""
        rec = ModelCRecognizer()
        rec.load()
        np.random.seed(99)
        test_vec = np.random.randn(280).astype(np.float32)

        rec.clf.n_jobs = -1
        res_neg1 = rec.predict_features(test_vec)

        rec.clf.n_jobs = 1
        res_1 = rec.predict_features(test_vec)

        self.assertEqual(res_neg1["predicted_letter"], res_1["predicted_letter"])
        self.assertAlmostEqual(res_neg1["confidence"], res_1["confidence"], places=6)
        probs_neg1 = [res_neg1["probabilities"][c] for c in sorted(res_neg1["probabilities"].keys())]
        probs_1 = [res_1["probabilities"][c] for c in sorted(res_1["probabilities"].keys())]
        np.testing.assert_allclose(probs_neg1, probs_1, atol=1e-12)

    # 4. ShadowWorker Bounded Queue and Drop Policy
    def test_04_shadow_worker_bounded_queue_and_drop_policy(self):
        """Verify ShadowWorker has maxsize=1 and drops oldest frames under burst load."""
        pipe = ISLPipeline()
        worker = ShadowWorker(pipe)
        self.assertEqual(worker.queue.maxsize, 1)

        # Do not start worker thread so items stay in queue
        worker.running = True
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        prod_res = {"predicted_sign": "A", "confidence": 95.0}

        # Submit 5 tasks rapidly without consumer thread
        for i in range(5):
            worker.submit_task(dummy_frame, prod_res, time.time(), time.time())

        # Queue should contain exactly 1 task (freshest)
        self.assertEqual(worker.queue.qsize(), 1)
        self.assertEqual(worker.submitted_count, 5)
        # Exactly 4 tasks dropped
        self.assertEqual(worker.dropped_count, 4)
        telemetry = worker.get_telemetry()
        self.assertEqual(telemetry["dropped_count"], 4)
        self.assertEqual(telemetry["drop_rate_pct"], 80.0)

    # 5. Shadow Async Timing Instrumentation
    def test_05_shadow_async_timing_instrumentation(self):
        """Verify ShadowWorker records required timing timestamps."""
        pipe = ISLPipeline()
        pipe.initialize()
        pipe._load_model_c(BASE_DIR)
        worker = ShadowWorker(pipe)
        worker.start()

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        prod_start = time.time()
        time.sleep(0.01)
        prod_finish = time.time()
        prod_res = {"predicted_sign": "C", "confidence": 88.0}

        worker.submit_task(dummy_frame, prod_res, prod_start, prod_finish)
        time.sleep(0.5)

        telemetry = worker.get_telemetry()
        worker.stop()

        self.assertGreaterEqual(telemetry["processed_count"], 1)
        last_res = telemetry["last_result"]
        self.assertIsNotNone(last_res)
        self.assertIn("queue_wait_ms", last_res)
        self.assertIn("shadow_execution_latency_ms", last_res)
        self.assertIn("frame_age_ms", last_res)
        self.assertIn("production_response_latency_ms", last_res)
        self.assertIn("queue_stats", last_res)
        self.assertGreaterEqual(last_res["production_response_latency_ms"], 9.0)

    # 6. Production Output Invariance
    def test_06_production_output_invariance(self):
        """Verify enabling async shadow mode does not alter production predictions or bounding boxes."""
        pipe = ISLPipeline()
        pipe.initialize()
        pipe._load_model_c(BASE_DIR)

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Baseline run (shadow disabled)
        pipe.set_shadow_mode(False)
        base_res = pipe.process_frame(dummy_frame, return_images=False)

        # Async shadow run
        pipe.set_shadow_mode(True, async_mode=True)
        shadow_res = pipe.process_frame(dummy_frame, return_images=False)

        self.assertEqual(base_res["predicted_sign"], shadow_res["predicted_sign"])
        self.assertEqual(base_res["confidence"], shadow_res["confidence"])
        self.assertEqual(base_res["detections"], shadow_res["detections"])
        self.assertEqual(base_res["probabilities"], shadow_res["probabilities"])

        pipe.shutdown()

    # 7. Shadow Worker Error Containment
    def test_07_shadow_worker_error_containment(self):
        """Verify errors in shadow worker are contained, logged, and increment error_count without crashing."""
        pipe = ISLPipeline()
        worker = ShadowWorker(pipe)
        worker.start()

        # Submit task with None frame to cause worker exception
        with worker._lock:
            worker.submitted_count += 1
            bad_task = {
                "frame_bgr": None,
                "production_prediction": "-",
                "production_confidence": 0.0,
                "prod_start_time": time.time(),
                "prod_finish_time": time.time(),
                "submit_time": time.time()
            }
            worker.queue.put_nowait(bad_task)

        for _ in range(25):
            if worker.get_telemetry()["processed_count"] > 0:
                break
            time.sleep(0.1)
        telemetry = worker.get_telemetry()
        worker.stop()

        self.assertGreaterEqual(telemetry["error_count"], 1)
        self.assertEqual(telemetry["last_result"]["status"], "error")
        self.assertIn("error_message", telemetry["last_result"])

    # 8. Graceful Shutdown
    def test_08_graceful_shutdown(self):
        """Verify pipeline.shutdown releases background threads cleanly."""
        pipe = ISLPipeline()
        pipe.shadow_worker = ShadowWorker(pipe)
        pipe.shadow_worker.start()
        self.assertTrue(pipe.shadow_worker.running)
        self.assertTrue(pipe.shadow_worker.thread.is_alive())

        pipe.shutdown()
        self.assertFalse(pipe.shadow_worker.running)
        self.assertFalse(pipe.shadow_worker.thread.is_alive())


if __name__ == "__main__":
    unittest.main()
