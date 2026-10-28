#!/usr/bin/env python3
"""
Unit and integration tests for Phase 19 Frozen Multi-Signer Benchmark Evaluation.

Verifies:
1. Frozen benchmark manifest integrity, SHA-256 hashes, and version declarations.
2. Complete evaluation of all seven candidate models on 3,329 external frames.
3. 26-class confusion matrices and per-letter metric completeness.
4. Subgroup breakdown coverage (signers, conditions, 4-way matrix).
5. Raw prediction records export integrity.
6. Acceptance criteria gates evaluation.
7. Unaltered production pipeline defaults.
"""

import os
import sys
import json
import csv
import unittest

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FROZEN_MANIFEST_PATH = os.path.join(BASE_DIR, "phase19_frozen_manifest.json")
BENCHMARK_RESULTS_PATH = os.path.join(BASE_DIR, "phase19_analysis", "phase19_benchmark_results.json")
PREDICTIONS_CSV_PATH = os.path.join(BASE_DIR, "phase19_analysis", "phase19_predictions_raw.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "phase19_analysis", "plots")

EXPECTED_MODELS = [
    "1_Production_YOLO_Baseline",
    "2_MediaPipe_v1_CNN",
    "3_Landmark_Only_Geometry_v1",
    "4_Landmark_Only_Geometry_v2",
    "5_Late_Fusion_Baseline_alpha_0.25",
    "6_Late_Fusion_Calibrated_alpha_0.75",
    "7_Hybrid_Concatenation_RF"
]

ALPHABET = [chr(i) for i in range(ord('A'), ord('Z') + 1)]


class TestPhase19Benchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert os.path.exists(FROZEN_MANIFEST_PATH), f"Missing frozen manifest: {FROZEN_MANIFEST_PATH}"
        with open(FROZEN_MANIFEST_PATH, "r") as f:
            cls.manifest = json.load(f)

    def test_01_frozen_manifest_structure(self):
        """Verifies phase 19 manifest tracks all 7 models, dataset hashes, and environment."""
        self.assertIn("phase", self.manifest)
        self.assertIn("environment", self.manifest)
        self.assertIn("dataset", self.manifest)
        self.assertIn("frozen_models", self.manifest)

        dataset = self.manifest["dataset"]
        self.assertEqual(dataset["total_frames"], 3329)
        self.assertEqual(dataset["session_id"], "session_20261022_01")
        self.assertEqual(dataset["signers"], ["signer_02", "signer_03"])
        self.assertEqual(dataset["conditions"], ["condition_1", "condition_2"])

        models = self.manifest["frozen_models"]
        for m in EXPECTED_MODELS:
            self.assertIn(m, models, f"Model {m} missing from frozen manifest")

    def test_02_production_defaults_unaltered(self):
        """Verifies that production defaults in pipeline.py remain strictly unchanged."""
        from pipeline import ISLPipeline
        p = ISLPipeline()
        self.assertEqual(p.backend_detector, "yolo")
        self.assertEqual(p.recognition_arch, "standard")
        self.assertFalse(p.temporal_smoothing_enabled)

    def test_03_benchmark_results_and_denominators(self):
        """Verifies benchmark results file exists and has full coverage across all 7 models."""
        if not os.path.exists(BENCHMARK_RESULTS_PATH):
            self.skipTest("Benchmark results not yet generated; skipping until execution finishes.")

        with open(BENCHMARK_RESULTS_PATH, "r") as f:
            results = json.load(f)

        self.assertIn("aggregate_evaluation", results)
        agg = results["aggregate_evaluation"]
        for m in EXPECTED_MODELS:
            self.assertIn(m, agg, f"Model {m} missing from aggregate results")
            met = agg[m]
            self.assertEqual(met["attempted_samples"], 3329)
            self.assertIn("conditional_top1_accuracy", met)
            self.assertIn("macro_f1", met)
            self.assertIn("confusion_matrix", met)
            self.assertEqual(len(met["confusion_matrix"]), 26)
            self.assertEqual(len(met["per_class"]), 26)

            for ltr in ALPHABET:
                self.assertIn(ltr, met["per_class"])
                self.assertGreaterEqual(met["per_class"][ltr]["support"], 120)

    def test_04_subgroup_evaluation_coverage(self):
        """Verifies subgroup breakdown across signers, conditions, and 4-way matrices."""
        if not os.path.exists(BENCHMARK_RESULTS_PATH):
            self.skipTest("Benchmark results not yet generated; skipping until execution finishes.")

        with open(BENCHMARK_RESULTS_PATH, "r") as f:
            results = json.load(f)

        self.assertIn("subgroup_evaluation", results)
        sub = results["subgroup_evaluation"]
        expected_groups = [
            "signer_02", "signer_03",
            "condition_1", "condition_2",
            "signer_02_condition_1", "signer_02_condition_2",
            "signer_03_condition_1", "signer_03_condition_2"
        ]
        for g in expected_groups:
            self.assertIn(g, sub, f"Subgroup {g} missing from results")
            for m in EXPECTED_MODELS:
                self.assertIn(m, sub[g])

    def test_05_raw_predictions_csv(self):
        """Verifies that the CSV containing raw per-frame predictions exists and has 3,329 rows."""
        if not os.path.exists(PREDICTIONS_CSV_PATH):
            self.skipTest("Raw predictions CSV not yet generated; skipping until execution finishes.")

        with open(PREDICTIONS_CSV_PATH, "r") as f:
            reader = list(csv.reader(f))

        # 1 header + 3329 data rows = 3330
        self.assertEqual(len(reader), 3330, f"Expected 3,330 rows in CSV, got {len(reader)}")

    def test_06_acceptance_gates_evaluation(self):
        """Verifies acceptance criteria gates are evaluated and formatted properly."""
        if not os.path.exists(BENCHMARK_RESULTS_PATH):
            self.skipTest("Benchmark results not yet generated; skipping until execution finishes.")

        with open(BENCHMARK_RESULTS_PATH, "r") as f:
            results = json.load(f)

        self.assertIn("acceptance_criteria_gates", results)
        gates = results["acceptance_criteria_gates"]
        for g_id in ["gate_1_macro_f1", "gate_2_n_recall", "gate_3_x_recall", "gate_4_l_accuracy", "gate_5_latency", "gate_6_regression_safety"]:
            self.assertIn(g_id, gates)
            self.assertIn(gates[g_id]["status"], ["PASS", "FAIL", "NOT_VERIFIED"])


if __name__ == "__main__":
    unittest.main()
