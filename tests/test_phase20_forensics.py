#!/usr/bin/env python3
"""
Unit tests for Phase 20 Error Forensics and Multi-Signer Training Strategy.
Verifies metric reconciliation, gate denominators, discrepancy forensics,
artifact preservation, and production configuration safety.
"""

import os
import csv
import json
import unittest
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestPhase20Forensics(unittest.TestCase):

    def setUp(self):
        self.csv_path = os.path.join(BASE_DIR, "phase19_analysis", "phase19_predictions_raw.csv")
        self.bench_json_path = os.path.join(BASE_DIR, "phase19_analysis", "phase19_benchmark_results.json")
        self.diagnostics_json_path = os.path.join(BASE_DIR, "phase20_analysis", "phase20_diagnostics.json")
        self.manifest_path = os.path.join(BASE_DIR, "phase19_frozen_manifest.json")
        self.report_path = os.path.join(BASE_DIR, "phase19_frozen_external_benchmark_report.md")

        self.assertTrue(os.path.exists(self.csv_path), "Raw predictions CSV must exist")
        self.assertTrue(os.path.exists(self.bench_json_path), "Benchmark results JSON must exist")
        self.assertTrue(os.path.exists(self.diagnostics_json_path), "Diagnostics JSON must exist")
        self.assertTrue(os.path.exists(self.manifest_path), "Frozen manifest must exist")
        self.assertTrue(os.path.exists(self.report_path), "Benchmark report must exist")

        with open(self.bench_json_path) as f:
            self.bench_json = json.load(f)

        with open(self.diagnostics_json_path) as f:
            self.diagnostics = json.load(f)

        self.raw_rows = []
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                self.raw_rows.append(r)

    def test_raw_predictions_reconciliation(self):
        """Verifies that recomputing Top-1 and Macro-F1 from CSV matches JSON within 0.05%."""
        y_true = np.array([r["ground_truth"] for r in self.raw_rows])
        classes = sorted(list(set(y_true)))

        col_map = {
            "3_Landmark_Only_Geometry_v1": "pred_m3_landmark_v1",
            "4_Landmark_Only_Geometry_v2": "pred_m4_geom_v2",
            "6_Late_Fusion_Calibrated_alpha_0.75": "pred_m6_late_fuse_calib"
        }

        for m_name, col in col_map.items():
            y_pred = np.array([r[col] for r in self.raw_rows])
            valid_mask = (y_pred != "NO_DETECTION") & (y_pred != "") & (y_pred != "None")
            recomp_t1 = accuracy_score(y_true[valid_mask], y_pred[valid_mask]) * 100.0
            
            _, _, f1, _ = precision_recall_fscore_support(
                y_true[valid_mask], y_pred[valid_mask], labels=classes, zero_division=0
            )
            recomp_f1 = float(np.mean(f1)) * 100.0

            json_entry = self.bench_json["aggregate_evaluation"][m_name]
            json_t1 = json_entry["conditional_top1_accuracy"]
            json_f1 = json_entry["macro_f1"]

            self.assertAlmostEqual(recomp_t1, json_t1, delta=0.05,
                                   msg=f"{m_name} Top-1 reconciliation mismatch")
            self.assertAlmostEqual(recomp_f1, json_f1, delta=0.05,
                                   msg=f"{m_name} Macro-F1 reconciliation mismatch")

    def test_gate_denominators(self):
        """Verifies that denominators for N, X, L are strictly 128, 129, and 128."""
        counts = {}
        for r in self.raw_rows:
            gt = r["ground_truth"]
            counts[gt] = counts.get(gt, 0) + 1

        self.assertEqual(counts["N"], 128, "Letter N denominator must be 128")
        self.assertEqual(counts["X"], 129, "Letter X denominator must be 129")
        self.assertEqual(counts["L"], 128, "Letter L denominator must be 128")
        self.assertEqual(len(self.raw_rows), 3329, "Total sample count must be 3,329")

    def test_discrepancy_static_vs_video_n(self):
        """Reconciles the zero static N recall in Model 4 with continuous video results."""
        n_rows = [r for r in self.raw_rows if r["ground_truth"] == "N"]
        self.assertEqual(len(n_rows), 128)

        # Model 4 (Geometry v2) has 0 correct predictions on N across all 128 static frames
        m4_correct = sum(1 for r in n_rows if r["pred_m4_geom_v2"] == "N")
        self.assertEqual(m4_correct, 0, "Model 4 static recall on N must be 0")

        # Model 6 (Calibrated Late Fusion) has 25 correct predictions, all on signer_03_condition_2
        m6_correct = sum(1 for r in n_rows if r["pred_m6_late_fuse_calib"] == "N")
        self.assertEqual(m6_correct, 25, "Model 6 static recall on N must be 25/128")

        # Confirm all 25 correct frames for M6 belong to rec_signer_03_condition_2_N
        m6_correct_rows = [r for r in n_rows if r["pred_m6_late_fuse_calib"] == "N"]
        rec_ids = set(r["recording_id"] for r in m6_correct_rows)
        self.assertEqual(len(rec_ids), 1)
        self.assertTrue("signer_03_condition_2" in list(rec_ids)[0])

    def test_yolo_latency_audit(self):
        """Verifies YOLO latency breakdown audit is properly documented in diagnostics."""
        yolo_audit = self.diagnostics.get("yolo_latency_audit", {})
        self.assertIn("measured_empirical_latencies", yolo_audit)
        meas = yolo_audit["measured_empirical_latencies"]
        self.assertGreater(meas["total_e2e_median_ms"], 500.0, "YOLO latency exceeds 100ms gate")
        self.assertGreater(meas["total_e2e_p95_ms"], 1000.0, "YOLO p95 latency exceeds 100ms gate")

    def test_production_defaults_intact(self):
        """Ensures production defaults in pipeline.py are strictly preserved."""
        from pipeline import ISLPipeline
        p = ISLPipeline()
        self.assertEqual(p.backend_detector, "yolo")
        self.assertEqual(p.recognition_arch, "standard")
        self.assertFalse(p.temporal_smoothing_enabled)


if __name__ == "__main__":
    unittest.main()
