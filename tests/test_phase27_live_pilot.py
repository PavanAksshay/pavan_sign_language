#!/usr/bin/env python3
"""
Unit and Integration Tests for Phase 27 — Controlled Live Pilot and Deployment Readiness
"""

import os
import sys
import json
import unittest
import asyncio
import numpy as np
from starlette.requests import Request

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from pipeline import ISLPipeline
from server import health, get_backend_status, get_recognition_status, set_delegate_endpoint, set_resolution_endpoint


def make_mock_request(payload: dict) -> Request:
    return Request(
        {"type": "http", "method": "POST", "headers": [(b"content-type", b"application/json")]},
        lambda: asyncio.sleep(0, {"type": "http.request", "body": json.dumps(payload).encode("utf-8")})
    )


class TestPhase27LivePilot(unittest.TestCase):

    def test_01_production_defaults_preserved(self):
        """Verifies that fresh ISLPipeline preserves strict production defaults."""
        pipe = ISLPipeline()
        try:
            self.assertEqual(pipe.backend_detector, "yolo")
            self.assertEqual(pipe.recognition_arch, "standard")
            self.assertFalse(pipe.temporal_smoothing_enabled)
            self.assertFalse(pipe.shadow_mode_enabled)
            self.assertEqual(pipe.mediapipe_delegate, "cpu")
            self.assertEqual(pipe.model_c_input_resolution, "native")
        finally:
            pipe.shutdown()

    def test_02_server_health_reports_delegate_and_resolution(self):
        """Verifies FastAPI endpoints report mediapipe_delegate and model_c_input_resolution."""
        data = asyncio.run(health())
        self.assertEqual(data["status"], "healthy")
        self.assertIn("mediapipe_delegate", data)
        self.assertIn("model_c_input_resolution", data)
        self.assertIn("effective_config", data)
        self.assertIn("mediapipe_delegate", data["effective_config"])
        self.assertIn("model_c_input_resolution", data["effective_config"])

    def test_03_server_backend_status_reports_configuration(self):
        """Verifies /api/backend/status accurately reports delegate and resolution."""
        data = asyncio.run(get_backend_status())
        self.assertIn("backend_detector", data)
        self.assertIn("recognition_arch", data)
        self.assertIn("mediapipe_delegate", data)
        self.assertIn("model_c_input_resolution", data)

    def test_04_server_recognition_status(self):
        """Verifies /api/recognition/status accurately reports delegate and resolution."""
        data = asyncio.run(get_recognition_status())
        self.assertIn("recognition_arch", data)
        self.assertIn("mediapipe_delegate", data)
        self.assertIn("model_c_input_resolution", data)

    def test_05_pilot_configuration_selection(self):
        """Verifies explicit selection of Model C pilot configuration."""
        pipe = ISLPipeline()
        try:
            pipe.set_recognition_arch("model_c")
            pipe.set_mediapipe_delegate("gpu")
            pipe.set_model_c_input_resolution("native")

            self.assertEqual(pipe.recognition_arch, "model_c")
            self.assertEqual(pipe.mediapipe_delegate, "gpu")
            self.assertEqual(pipe.model_c_input_resolution, "native")
        finally:
            pipe.shutdown()

    def test_06_clean_switch_back_to_production(self):
        """Verifies switching from pilot Model C back to production configuration is clean."""
        pipe = ISLPipeline()
        try:
            # Switch to pilot
            pipe.set_recognition_arch("model_c")
            pipe.set_mediapipe_delegate("gpu")
            # Switch back to production
            pipe.set_recognition_arch("standard")
            pipe.set_mediapipe_delegate("cpu")

            self.assertEqual(pipe.recognition_arch, "standard")
            self.assertEqual(pipe.mediapipe_delegate, "cpu")
        finally:
            pipe.shutdown()

    def test_07_delegate_setter_validation(self):
        """Verifies set_mediapipe_delegate validates input options properly."""
        pipe = ISLPipeline()
        try:
            pipe.set_mediapipe_delegate("gpu")
            self.assertEqual(pipe.mediapipe_delegate, "gpu")
            pipe.set_mediapipe_delegate("metal")
            self.assertEqual(pipe.mediapipe_delegate, "gpu")
            pipe.set_mediapipe_delegate("cpu")
            self.assertEqual(pipe.mediapipe_delegate, "cpu")

            with self.assertRaises(ValueError):
                pipe.set_mediapipe_delegate("invalid_delegate")
        finally:
            pipe.shutdown()

    def test_08_resolution_setter_validation(self):
        """Verifies set_model_c_input_resolution validates resolution strings properly."""
        pipe = ISLPipeline()
        try:
            pipe.set_model_c_input_resolution("native")
            self.assertEqual(pipe.model_c_input_resolution, "native")
            pipe.set_model_c_input_resolution("960x540")
            self.assertEqual(pipe.model_c_input_resolution, "960x540")
            pipe.set_model_c_input_resolution("640x360")
            self.assertEqual(pipe.model_c_input_resolution, "640x360")
            pipe.set_model_c_input_resolution("640x480")
            self.assertEqual(pipe.model_c_input_resolution, "640x480")

            with self.assertRaises(ValueError):
                pipe.set_model_c_input_resolution("1920x1080")
        finally:
            pipe.shutdown()

    def test_09_server_endpoints_delegate_and_resolution(self):
        """Verifies POST /api/delegate/set and POST /api/resolution/set endpoints."""
        req_gpu = make_mock_request({"delegate": "gpu"})
        resp_gpu = asyncio.run(set_delegate_endpoint(req_gpu))
        self.assertEqual(resp_gpu["mediapipe_delegate"], "gpu")

        req_res = make_mock_request({"resolution": "960x540"})
        resp_res = asyncio.run(set_resolution_endpoint(req_res))
        self.assertEqual(resp_res["model_c_input_resolution"], "960x540")

        # Restore defaults
        req_cpu = make_mock_request({"delegate": "cpu"})
        asyncio.run(set_delegate_endpoint(req_cpu))
        req_nat = make_mock_request({"resolution": "native"})
        asyncio.run(set_resolution_endpoint(req_nat))

    def test_10_error_containment_and_recovery(self):
        """Verifies graceful handling of invalid frames and recovery on valid frame."""
        pipe = ISLPipeline()
        try:
            pipe.set_recognition_arch("model_c")
            pipe.set_mediapipe_delegate("gpu")
            pipe.diagnostics_enabled = False

            # Blank black frame
            blank = np.zeros((720, 1280, 3), dtype=np.uint8)
            res_blank = pipe.process_frame(blank)
            self.assertFalse(res_blank["hand_detected"])
            self.assertIn(res_blank["predicted_sign"], (None, "-"))

            # Immediate recovery on subsequent valid synthetic pattern
            pattern = np.ones((720, 1280, 3), dtype=np.uint8) * 128
            res_pat = pipe.process_frame(pattern)
            self.assertIn("hand_detected", res_pat)
            self.assertIn("timing_ms", res_pat)
        finally:
            pipe.shutdown()

    def test_11_sustained_inference_latency_gate(self):
        """Verifies sustained inference on Model C satisfies the 100 ms target."""
        pipe = ISLPipeline()
        try:
            pipe.set_recognition_arch("model_c")
            pipe.set_mediapipe_delegate("gpu")
            pipe.diagnostics_enabled = False

            test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            latencies = []
            for _ in range(15):
                res = pipe.process_frame(test_frame)
                lat = res["timing_ms"]["total"]
                latencies.append(lat)

            p95 = np.percentile(latencies, 95)
            self.assertLess(p95, 100.0, f"Sustained inference p95 exceeded 100 ms: {p95} ms")
        finally:
            pipe.shutdown()

    def test_12_collection_ledger_and_manifest_integrity(self):
        """Verifies collection ledger contains all 54 signer_04 recordings."""
        ledger_path = os.path.join(BASE_DIR, "data", "collection_manifests", "collection_ledger.json")
        self.assertTrue(os.path.exists(ledger_path))
        with open(ledger_path, "r") as f:
            ledger = json.load(f)

        s4_records = [r for r in ledger["records"] if r.get("signer_id") == "signer_04"]
        self.assertEqual(len(s4_records), 54)
        for r in s4_records:
            self.assertEqual(r["status"], "accepted")
            self.assertIn("raw_video_path", r)
            self.assertTrue(os.path.exists(os.path.join(BASE_DIR, r["raw_video_path"])))


if __name__ == "__main__":
    unittest.main()
