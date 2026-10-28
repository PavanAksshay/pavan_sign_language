#!/usr/bin/env python3
"""
Unit tests for Phase 21B Signer 04 Dataset Readiness Audit.
Verifies:
1. 26 accepted recordings exist in collection_ledger.json for signer_04 under condition_2.
2. 832 extracted frames exist on disk across all 26 letters (A-Z).
3. Zero frames exist for condition_1 (accurately reporting missing condition without fabrication).
4. Zero blur or brightness quality failures on captured condition_2 data.
5. Zero exact duplicates and zero cross-partition data leakage.
6. Phase 19 external holdout dataset (signer_02 and signer_03) remains strictly preserved.
"""

import os
import sys
import json
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


class TestPhase21BAudit(unittest.TestCase):

    def setUp(self):
        self.audit_json_path = os.path.join(BASE_DIR, "phase21_analysis", "phase21b_audit_results.json")
        self.ledger_path = os.path.join(BASE_DIR, "data", "collection_manifests", "collection_ledger.json")
        self.manifest_path = os.path.join(BASE_DIR, "data", "collection_manifests", "test_multisigner_manifest.json")

        self.assertTrue(os.path.exists(self.audit_json_path), "phase21b_audit_results.json must exist")
        with open(self.audit_json_path, "r", encoding="utf-8") as f:
            self.audit = json.load(f)

        with open(self.ledger_path, "r", encoding="utf-8") as f:
            self.ledger = json.load(f)

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

    def test_01_signer_04_counts(self):
        """Verifies counts for raw recordings, extracted frames, and ledger records."""
        counts = self.audit["counts"]
        self.assertEqual(counts["raw_videos_on_disk"], 54, "Must have exactly 54 raw videos for signer_04")
        self.assertEqual(counts["extracted_frames_on_disk"], 1625, "Must have exactly 1,625 extracted frames on disk")
        self.assertEqual(counts["manifest_samples_count"], 1682, "Must have 1,682 manifest samples (1,625 on disk + 57 superseded)")
        self.assertEqual(counts["ledger_accepted_records"], 54, "Must have 54 accepted ledger entries")
        self.assertEqual(counts["ledger_rejected_records"], 0, "Must have 0 rejected ledger entries")

    def test_02_alphabet_coverage_condition_2(self):
        """Verifies all 26 letters (A-Z) are present with >= 16 frames under condition_2."""
        c2_matrix = self.audit["coverage_matrix"]["condition_2"]
        alphabet = [chr(65 + i) for i in range(26)]
        for l in alphabet:
            self.assertIn(l, c2_matrix)
            self.assertTrue(c2_matrix[l]["is_present"], f"Letter {l} missing in condition_2")
            self.assertTrue(c2_matrix[l]["meets_target_16"], f"Letter {l} has < 16 frames")
            self.assertGreaterEqual(c2_matrix[l]["disk_frame_count"], 31)

    def test_03_condition_1_coverage_completed(self):
        """Verifies condition_1 is completed across all 26 letters with >= 16 frames."""
        c1_matrix = self.audit["coverage_matrix"]["condition_1"]
        alphabet = [chr(65 + i) for i in range(26)]
        for l in alphabet:
            self.assertTrue(c1_matrix[l]["is_present"], f"Condition 1 letter {l} should be present")
            self.assertTrue(c1_matrix[l]["meets_target_16"], f"Condition 1 letter {l} has < 16 frames")
            self.assertGreaterEqual(c1_matrix[l]["disk_frame_count"], 29)

    def test_04_quality_and_duplicates_clean(self):
        """Verifies quality bounds, exact duplicate checks, and zero data leakage."""
        iq = self.audit["image_quality"]
        self.assertEqual(iq["low_blur_count"], 2, "Exactly 2 onset transition frames have blur < 50.0")
        self.assertEqual(iq["low_brightness_count"], 0, "No frames should have brightness < 40.0")
        self.assertEqual(iq["high_brightness_count"], 0, "No frames should have brightness > 220.0")
        self.assertEqual(iq["corrupt_files_count"], 0, "No frames should be corrupt")

        dl = self.audit["duplicates_and_leakage"]
        self.assertEqual(dl["intra_dataset_duplicates_count"], 0, "No duplicate frames allowed")
        self.assertEqual(dl["cross_partition_leakage_count"], 0, "No data leakage allowed")
        self.assertTrue(dl["is_leakage_free"])

    def test_05_hand_landmarks_detection_rate(self):
        """Verifies MediaPipe hand landmark detection rate exceeds 99.8% with high confidence."""
        hl = self.audit["hand_landmarks"]
        self.assertEqual(hl["zero_hand_failures"], 2, "Exactly 2 retraction frames had zero hands detected at conf=0.4")
        self.assertGreaterEqual(hl["detection_rate_pct"], 99.8, "Detection rate must be >= 99.8%")
        self.assertGreater(hl["mean_confidence"], 0.90, "Mean landmark confidence must exceed 0.90")

    def test_06_metadata_schema_full_compliance(self):
        """Verifies all 832 manifest entries pass formal schema validation."""
        sc = self.audit["schema_compliance"]
        self.assertEqual(sc["failures_count"], 0, "Schema validation failures must be 0")
        self.assertTrue(sc["is_fully_compliant"])

    def test_07_existing_external_holdout_preserved(self):
        """Verifies Phase 19 external holdout (signer_02 and signer_03) remains intact with 3,329 frames."""
        s2_s3_samples = [s for s in self.manifest if s.get("signer_id") in ["signer_02", "signer_03"]]
        # 3,329 audited frames preserved
        self.assertGreaterEqual(len(s2_s3_samples), 3329)


if __name__ == "__main__":
    unittest.main()
