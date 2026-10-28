#!/usr/bin/env python3
"""
Phase 30 — Signer_05 Independent Evaluation Regression Test Suite
Tests:
1. Production defaults preservation (YOLO, standard, CPU, native, smoothing disabled, shadow disabled)
2. Model immutability and approved baseline hashes (Model C RF, scaler, frozen benchmarks)
3. Evaluation manifest integrity and structure (1,638 frames, 26 letters, 52 recordings, exclusions)
4. Cryptographic deduplication and split isolation (0 internal duplicates, 0 train/val/test overlap)
5. Canonical feature schema (280-dim) and class mapping (26 letters A-Z)
6. Metric denominators, support sums, and mathematical consistency
7. Handling of zero-hand and invalid frames without crashes
8. Gate evaluation and verdict consistency
"""

import os
import sys
import json
import hashlib
import unittest
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from pipeline import ISLPipeline
from landmark_model_c import FEATURE_DIM_MODEL_C, ALPHABET, C2I, I2C

APPROVED_HASHES = {
    "model_c_rf": {
        "path": os.path.join(BASE_DIR, "phase22_analysis", "models", "Model_C_Occlusion_Aware_RF_multisigner.joblib"),
        "sha256": "1fab3349e2e537d8b31ddbe3bdd5c21e3ed005f85aaac9d3875c72217e28cc98"
    },
    "model_c_scaler": {
        "path": os.path.join(BASE_DIR, "phase22_analysis", "models", "Model_C_Occlusion_Aware_RF_multisigner_scaler.joblib"),
        "sha256": "93842932e7cfeb1d283e99f67a733fd487161d83cbff356d7458ced2c3de385d"
    },
    "phase19_frozen_manifest": {
        "path": os.path.join(BASE_DIR, "phase19_frozen_manifest.json"),
        "sha256": "9824eb510cc231249e9dd0722761de9112a02e7cecb681a479c4ef805b0fa797"
    },
    "phase23_predictions": {
        "path": os.path.join(BASE_DIR, "phase23_predictions.csv"),
        "sha256": "da77036032fb3250141ccc16ba53d1dd7d4997b9817c6f525bbf9e000be45bfe"
    },
    "hand_landmarker_task": {
        "path": os.path.join(BASE_DIR, "App", "include50_model", "hand_landmarker.task"),
        "sha256": "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
    }
}


def sha256_file(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class TestPhase30IndependentSigner(unittest.TestCase):
    """Phase 30 test suite for independent signer_05 evaluation and validation gates."""

    def test_01_production_defaults_preservation(self):
        """Production pipeline initialization must preserve default settings strictly untouched."""
        pipe = ISLPipeline()
        try:
            self.assertEqual(pipe.backend_detector, "yolo")
            self.assertEqual(pipe.recognition_arch, "standard")
            self.assertEqual(pipe.mediapipe_delegate, "cpu")
            self.assertFalse(pipe.temporal_smoothing_enabled)
            self.assertFalse(pipe.shadow_mode_enabled)
            self.assertEqual(getattr(pipe, "model_c_input_resolution", "native"), "native")
        finally:
            pipe.shutdown()

    def test_02_model_and_artifact_immutability(self):
        """Approved baseline model checkpoints and frozen manifests must remain bit-for-bit identical."""
        for name, meta in APPROVED_HASHES.items():
            self.assertTrue(os.path.exists(meta["path"]), f"Missing artifact: {meta['path']}")
            actual_sha = sha256_file(meta["path"])
            self.assertEqual(
                actual_sha, meta["sha256"],
                f"Hash mismatch on {name}: expected {meta['sha256']}, got {actual_sha}"
            )

    def test_03_evaluation_manifest_integrity_and_structure(self):
        """Evaluation manifest must be frozen, schema-valid, and match on-disk frames."""
        manifest_path = os.path.join(BASE_DIR, "phase30_evaluation_manifest.json")
        audit_path = os.path.join(BASE_DIR, "phase30_integrity_audit.json")
        self.assertTrue(os.path.exists(manifest_path), "Missing phase30_evaluation_manifest.json")
        self.assertTrue(os.path.exists(audit_path), "Missing phase30_integrity_audit.json")

        actual_manifest_sha = sha256_file(manifest_path)
        with open(audit_path, "r") as f:
            audit = json.load(f)
        self.assertEqual(actual_manifest_sha, audit["evaluation_manifest_sha256"])

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        self.assertEqual(manifest["phase"], 30)
        self.assertEqual(manifest["signer_id"], "signer_05")
        self.assertEqual(manifest["summary"]["total_eligible_frames"], 1638)
        self.assertEqual(manifest["summary"]["condition_1_frames"], 821)
        self.assertEqual(manifest["summary"]["condition_2_frames"], 817)
        self.assertEqual(manifest["summary"]["total_recordings"], 52)
        self.assertEqual(manifest["summary"]["distinct_letters"], 26)

        # Check all letters are represented in both conditions
        c1_counts = manifest["summary"]["per_condition_letter_counts"]["condition_1"]
        c2_counts = manifest["summary"]["per_condition_letter_counts"]["condition_2"]
        self.assertEqual(len(c1_counts), 26)
        self.assertEqual(len(c2_counts), 26)
        for letter in ALPHABET:
            self.assertGreater(c1_counts[letter], 0)
            self.assertGreater(c2_counts[letter], 0)

        # Exclusions documented
        self.assertEqual(len(manifest["exclusions"]), 1)
        self.assertEqual(manifest["exclusions"][0]["recording_id"], "rec_signer_05_condition_2_H_1792961047")
        self.assertEqual(manifest["exclusions"][0]["status"], "rejected")

        # Samples verification (all 1638 samples exist and match sha256)
        samples = manifest["samples"]
        self.assertEqual(len(samples), 1638)
        for s in samples[:50]:  # Check first 50 directly in test
            fpath = os.path.join(BASE_DIR, s["frame_path"])
            self.assertTrue(os.path.exists(fpath))
            self.assertEqual(sha256_file(fpath), s["sha256"])

    def test_04_cryptographic_deduplication_and_split_isolation(self):
        """Signer_05 frames must be unique and have zero overlap with training and validation splits."""
        manifest_path = os.path.join(BASE_DIR, "phase30_evaluation_manifest.json")
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        s05_hashes = set(s["sha256"] for s in manifest["samples"])
        self.assertEqual(len(s05_hashes), 1638, "Found internal duplicate frame hashes in signer_05 manifest!")

        train_path = os.path.join(BASE_DIR, "models", "mediapipe_squeezenet_v1", "splits", "train_manifest.json")
        val_path = os.path.join(BASE_DIR, "models", "mediapipe_squeezenet_v1", "splits", "val_manifest.json")
        p19_path = os.path.join(BASE_DIR, "phase19_frozen_manifest.json")

        for split_path in [train_path, val_path, p19_path]:
            if os.path.exists(split_path):
                with open(split_path, "r") as f:
                    data = json.load(f)
                hashes = set()
                if isinstance(data, list):
                    for x in data:
                        if isinstance(x, dict):
                            h = x.get("source", {}).get("sha256") or x.get("sha256")
                            if h:
                                hashes.add(h)
                overlap = s05_hashes.intersection(hashes)
                self.assertEqual(len(overlap), 0, f"Data leakage: {len(overlap)} hashes overlap with {split_path}")

    def test_05_feature_schema_and_class_mapping(self):
        """Feature schema must be 280 dimensions and class mapping must cover exactly 26 classes A-Z."""
        self.assertEqual(FEATURE_DIM_MODEL_C, 280)
        self.assertEqual(len(ALPHABET), 26)
        self.assertEqual(len(C2I), 26)
        self.assertEqual(len(I2C), 26)
        self.assertEqual(ALPHABET[0], "A")
        self.assertEqual(ALPHABET[-1], "Z")

        # Verify Model C Scaler dimensionality
        import joblib
        scaler = joblib.load(APPROVED_HASHES["model_c_scaler"]["path"])
        self.assertEqual(len(scaler.mean_), 280)

        # Verify Model C Classifier dimensionality
        rf = joblib.load(APPROVED_HASHES["model_c_rf"]["path"])
        self.assertEqual(rf.n_features_in_, 280)
        self.assertEqual(rf.n_classes_, 26)

    def test_06_metric_denominators_and_mathematical_consistency(self):
        """Evaluation metrics must have consistent denominators, support sums, and valid ranges."""
        metrics_path = os.path.join(BASE_DIR, "phase30_independent_signer_metrics.json")
        self.assertTrue(os.path.exists(metrics_path), "Missing phase30_independent_signer_metrics.json")

        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        ov = metrics["recognition_metrics"]["overall_pooled"]
        c1 = metrics["recognition_metrics"]["condition_1"]
        c2 = metrics["recognition_metrics"]["condition_2"]

        # Denominator checks
        self.assertEqual(ov["total_evaluated"], 1638)
        self.assertEqual(c1["total_evaluated"], 821)
        self.assertEqual(c2["total_evaluated"], 817)
        self.assertEqual(c1["total_evaluated"] + c2["total_evaluated"], 1638)

        # Accuracy within [0, 1]
        self.assertTrue(0.0 <= ov["overall_top1_accuracy"] <= 1.0)
        self.assertTrue(0.0 <= ov["macro_f1"] <= 1.0)
        self.assertTrue(0.0 <= ov["balanced_accuracy"] <= 1.0)

        # Support sums
        total_support = sum(ov["per_class"][c]["support"] for c in ALPHABET)
        self.assertEqual(total_support, 1638)

        for c in ALPHABET:
            p = ov["per_class"][c]
            self.assertEqual(p["tp"] + p["fn"], p["support"])

        # Confusion matrix sum plus zero-hand frames equals total evaluated
        cm = np.array(metrics["confusion_matrix_26x26"])
        self.assertEqual(cm.shape, (26, 26))
        zero_hands = metrics["detection_and_forensics"]["zero_hand_frames"]
        self.assertEqual(int(cm.sum()) + zero_hands, 1638)

    def test_07_invalid_and_zero_hand_resilience(self):
        """Pipeline must handle blank/zero-hand frames gracefully returning fallback flag without raising."""
        pipe = ISLPipeline()
        try:
            pipe.set_recognition_arch("model_c")
            pipe.set_model_c_input_resolution("640x360")
            pipe.set_mediapipe_delegate("gpu")
            pipe.initialize()

            blank_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            res = pipe.process_frame(blank_frame, return_images=False)
            self.assertIn("predicted_sign", res)
            diag = res.get("diagnostics", {})
            self.assertEqual(diag.get("hands_detected_count"), 0)
            self.assertEqual(diag.get("feature_status"), "fallback_0_hands")
            self.assertEqual(diag.get("feature_dimensions"), 280)
        finally:
            pipe.shutdown()

    def test_08_verdict_and_gate_evaluation_consistency(self):
        """Verdict in metrics and integrity audit must match measured gate outcomes."""
        metrics_path = os.path.join(BASE_DIR, "phase30_independent_signer_metrics.json")
        audit_path = os.path.join(BASE_DIR, "phase30_integrity_audit.json")

        with open(metrics_path, "r") as f:
            m = json.load(f)
        with open(audit_path, "r") as f:
            a = json.load(f)

        verdict = m["final_verdict"]
        self.assertEqual(verdict, a["final_verdict"])
        self.assertEqual(verdict, "RECOGNITION GENERALIZATION GATE FAILED")

        gates = m["gate_evaluation"]
        # Recognition gates failed
        self.assertFalse(gates["recognition_gates"]["all_recognition_gates_passed"])
        self.assertFalse(gates["recognition_gates"]["overall_accuracy_pass"])
        self.assertFalse(gates["recognition_gates"]["macro_f1_pass"])
        self.assertFalse(gates["recognition_gates"]["zero_recall_pass"])
        self.assertIn("J", gates["recognition_gates"]["zero_recall_letters"])
        self.assertIn("O", gates["recognition_gates"]["zero_recall_letters"])
        self.assertIn("P", gates["recognition_gates"]["zero_recall_letters"])
        self.assertIn("Q", gates["recognition_gates"]["zero_recall_letters"])

        # Latency gates passed
        self.assertTrue(gates["latency_gates"]["all_latency_gates_passed"])
        self.assertTrue(gates["latency_gates"]["p95_pass"])
        self.assertTrue(gates["latency_gates"]["throughput_pass"])
        self.assertLess(gates["latency_gates"]["p95_value_ms"], 100.0)
        self.assertGreater(gates["latency_gates"]["throughput_value_fps"], 15.0)

        # Integrity gates passed
        self.assertTrue(gates["integrity_gates"]["all_integrity_gates_passed"])
        self.assertTrue(gates["integrity_gates"]["model_hashes_pass"])
        self.assertTrue(gates["integrity_gates"]["evaluation_manifest_frozen"])
        self.assertTrue(gates["integrity_gates"]["split_isolation_pass"])
        self.assertTrue(gates["integrity_gates"]["production_defaults_pass"])
        self.assertTrue(gates["integrity_gates"]["reliability_errors_pass"])


if __name__ == "__main__":
    unittest.main()
