#!/usr/bin/env python3
"""
Unit tests for Phase 21C Signer 04 Manifest Reconciliation and Quality Cleanup.
Verifies:
1. Manifest reconciliation: 1,682 total samples = 1,621 usable + 57 superseded + 4 quality rejected.
2. Exact 1:1 physical disk parity for all 1,621 usable samples.
3. Coverage across all 52 cells (26 letters x 2 conditions), all meeting >= 16 usable frames.
4. Correct authoritative session metadata ('session_20261022_01') across manifest and ledger.
5. Exact 4 quality-rejected samples identified with explicit reasons.
6. Zero cryptographic data leakage and zero intra-dataset duplicates.
7. 100% schema compliance across all manifest samples.
"""

import os
import sys
import json
import hashlib
import unittest
from collections import defaultdict, Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from phase18_collector import validate_sample_schema


class TestPhase21CReconciliation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.manifest_path = os.path.join(BASE_DIR, "data", "collection_manifests", "test_multisigner_manifest.json")
        cls.ledger_path = os.path.join(BASE_DIR, "data", "collection_manifests", "collection_ledger.json")
        cls.results_json_path = os.path.join(BASE_DIR, "phase21_analysis", "phase21c_audit_results.json")

        assert os.path.exists(cls.manifest_path), f"Missing {cls.manifest_path}"
        assert os.path.exists(cls.ledger_path), f"Missing {cls.ledger_path}"
        assert os.path.exists(cls.results_json_path), f"Missing {cls.results_json_path}"

        with open(cls.manifest_path, "r", encoding="utf-8") as f:
            cls.manifest = json.load(f)

        with open(cls.ledger_path, "r", encoding="utf-8") as f:
            cls.ledger = json.load(f)

        with open(cls.results_json_path, "r", encoding="utf-8") as f:
            cls.audit_results = json.load(f)

        cls.s4_samples = [s for s in cls.manifest if s.get("signer_id") == "signer_04"]
        cls.s4_usable = [s for s in cls.s4_samples if s.get("quality", {}).get("is_usable") is True]
        cls.s4_superseded = [s for s in cls.s4_samples if "superseded" in (s.get("quality", {}).get("rejection_reason") or "")]
        cls.s4_quality_rejected = [s for s in cls.s4_samples if s.get("quality", {}).get("is_usable") is False and "superseded" not in (s.get("quality", {}).get("rejection_reason") or "")]

    def test_01_manifest_sample_breakdown(self):
        """Verifies exact reconciliation breakdown: 1,682 = 1,621 usable + 57 superseded + 4 quality-rejected."""
        self.assertEqual(len(self.s4_samples), 1682, "Total signer_04 manifest samples must be 1,682")
        self.assertEqual(len(self.s4_superseded), 57, "Must have exactly 57 superseded samples")
        self.assertEqual(len(self.s4_quality_rejected), 4, "Must have exactly 4 quality-rejected samples")
        self.assertEqual(len(self.s4_usable), 1621, "Must have exactly 1,621 usable samples")
        self.assertEqual(len(self.s4_usable) + len(self.s4_superseded) + len(self.s4_quality_rejected), 1682)

    def test_02_usable_frames_physical_disk_parity(self):
        """Verifies that every single one of the 1,621 usable frames exists on disk and matches SHA-256."""
        self.assertEqual(len(self.s4_usable), 1621)
        for s in self.s4_usable:
            rel_p = s["source"]["frame_path"]
            abs_p = os.path.join(BASE_DIR, rel_p)
            self.assertTrue(os.path.exists(abs_p), f"Missing frame on disk: {rel_p}")
            with open(abs_p, "rb") as fp:
                file_hash = hashlib.sha256(fp.read()).hexdigest()
            self.assertEqual(file_hash, s["source"]["sha256"], f"SHA-256 mismatch for {rel_p}")

    def test_03_coverage_matrix_52_cells_complete(self):
        """Verifies all 52 cells retain >= 16 usable frames (Condition 1 min=26, Condition 2 min=31)."""
        alphabet = [chr(65 + i) for i in range(26)]
        cells = defaultdict(int)
        for s in self.s4_usable:
            cond = "condition_1" if "condition_1" in s["source"]["frame_path"] else "condition_2"
            cells[(cond, s["letter"])] += 1

        self.assertEqual(len(cells), 52, f"Expected 52 cells, found {len(cells)}")
        for (cond, let), count in cells.items():
            self.assertGreaterEqual(count, 16, f"Cell ({cond}, {let}) has only {count} frames (< 16)")
            if cond == "condition_1":
                self.assertGreaterEqual(count, 26, f"Cell ({cond}, {let}) has < 26 frames")
            else:
                self.assertGreaterEqual(count, 31, f"Cell ({cond}, {let}) has < 31 frames")

    def test_04_quality_rejection_reasons(self):
        """Verifies the 4 quality-rejected samples have correct recorded reasons."""
        self.assertEqual(len(self.s4_quality_rejected), 4)
        rejected_by_id = {s["sample_id"]: s for s in self.s4_quality_rejected}

        # D_0003 blur
        self.assertIn("signer_04_session_20261015_01_D_0003", rejected_by_id)
        d3 = rejected_by_id["signer_04_session_20261015_01_D_0003"]
        self.assertEqual(d3["quality"]["rejection_reason"], "motion_blur_below_threshold")
        self.assertEqual(d3["quality"]["blur_laplacian_var"], 49.77)

        # W_0003 blur
        self.assertIn("signer_04_session_20261015_01_W_0003", rejected_by_id)
        w3 = rejected_by_id["signer_04_session_20261015_01_W_0003"]
        self.assertEqual(w3["quality"]["rejection_reason"], "motion_blur_below_threshold")
        self.assertEqual(w3["quality"]["blur_laplacian_var"], 49.81)

        # W_0078 zero hands
        self.assertIn("signer_04_session_20261015_01_W_0078", rejected_by_id)
        w78 = rejected_by_id["signer_04_session_20261015_01_W_0078"]
        self.assertEqual(w78["quality"]["rejection_reason"], "zero_hands_detected")
        self.assertEqual(w78["detection"]["hands_detected"], 0)

        # W_0081 zero hands
        self.assertIn("signer_04_session_20261015_01_W_0081", rejected_by_id)
        w81 = rejected_by_id["signer_04_session_20261015_01_W_0081"]
        self.assertEqual(w81["quality"]["rejection_reason"], "zero_hands_detected")
        self.assertEqual(w81["detection"]["hands_detected"], 0)

    def test_05_authoritative_session_metadata(self):
        """Verifies session metadata is corrected to 'session_20261022_01' across all signer_04 records."""
        for s in self.s4_samples:
            self.assertEqual(s["session_id"], "session_20261022_01", f"Stale session_id in sample {s['sample_id']}")

        s4_ledger_recs = [r for r in self.ledger.get("records", []) if r.get("signer_id") == "signer_04"]
        self.assertEqual(len(s4_ledger_recs), 54)
        for r in s4_ledger_recs:
            self.assertEqual(r["session_id"], "session_20261022_01", f"Stale session_id in ledger record {r['recording_id']}")

    def test_06_zero_historical_leakage_and_duplicates(self):
        """Verifies zero SHA-256 overlap with historical splits or Phase 19 frozen test partition."""
        hist_splits = {
            "train": os.path.join(BASE_DIR, "models", "mediapipe_squeezenet_v1", "splits", "train_manifest.json"),
            "val": os.path.join(BASE_DIR, "models", "mediapipe_squeezenet_v1", "splits", "val_manifest.json"),
            "test_al": os.path.join(BASE_DIR, "models", "mediapipe_squeezenet_v1", "splits", "test_al_manifest.json"),
            "test_mz": os.path.join(BASE_DIR, "models", "mediapipe_squeezenet_v1", "splits", "test_mz_manifest.json")
        }
        hist_hashes = set()
        for name, p in hist_splits.items():
            if os.path.exists(p):
                with open(p, "r") as fp:
                    data = json.load(fp)
                samples = data if isinstance(data, list) else data.get("samples", [])
                for s in samples:
                    h = s.get("source", {}).get("sha256") or s.get("sha256")
                    if h:
                        hist_hashes.add(h)

        usable_hashes = set(s["source"]["sha256"] for s in self.s4_usable)
        self.assertEqual(len(usable_hashes), 1621, "Intra-signer duplicate hashes detected!")
        leakage = usable_hashes.intersection(hist_hashes)
        self.assertEqual(len(leakage), 0, f"Found {len(leakage)} leaked hashes into historical splits!")

        # Phase 19 frozen test partition (first 3,329 frames)
        p19_hashes = set(s["source"]["sha256"] for s in self.manifest[:3329])
        p19_leakage = usable_hashes.intersection(p19_hashes)
        self.assertEqual(len(p19_leakage), 0, f"Found {len(p19_leakage)} leaked hashes into Phase 19 frozen test!")

    def test_07_full_schema_compliance(self):
        """Verifies all signer_04 samples pass strict schema validation against metadata_schema.json."""
        for s in self.s4_samples:
            is_valid, err = validate_sample_schema(s)
            self.assertTrue(is_valid, f"Schema error on {s.get('sample_id')}: {err}")


if __name__ == "__main__":
    unittest.main()
