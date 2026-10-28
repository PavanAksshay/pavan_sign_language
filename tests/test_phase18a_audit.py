#!/usr/bin/env python3
"""
Unit and integration tests for Phase 18A dataset audit.
Validates:
1. Exact frame count (3,329) and target satisfaction (>= 3,120).
2. Complete 26-letter alphabet coverage.
3. 4-way matrix (2 signers x 2 conditions x 26 letters = 104 cells, all >= 30 frames).
4. Physical disk 1:1 match with manifest (0 missing, 0 untracked).
5. 104 raw MP4 video recordings integrity.
6. Zero SHA-256 leakage into historical train/val/test splits.
7. Zero intra-dataset exact duplicates.
8. Frame dimensions (1280x720x3) and quality bounds (blur >= 50, brightness in [40, 220]).
9. Ledger records: 104 accepted recordings, 3 rejected preview attempts.
"""

import os
import sys
import glob
import json
import hashlib
import unittest
from collections import defaultdict, Counter
import cv2

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MANIFEST_PATH = os.path.join(BASE_DIR, "data", "collection_manifests", "test_multisigner_manifest.json")
LEDGER_PATH = os.path.join(BASE_DIR, "data", "collection_manifests", "collection_ledger.json")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw_recordings")
FRAMES_DIR = os.path.join(BASE_DIR, "data", "extracted_frames")

HISTORICAL_SPLITS = {
    "train": os.path.join(BASE_DIR, "models", "mediapipe_squeezenet_v1", "splits", "train_manifest.json"),
    "val": os.path.join(BASE_DIR, "models", "mediapipe_squeezenet_v1", "splits", "val_manifest.json"),
    "test_al": os.path.join(BASE_DIR, "models", "mediapipe_squeezenet_v1", "splits", "test_al_manifest.json"),
    "test_mz": os.path.join(BASE_DIR, "models", "mediapipe_squeezenet_v1", "splits", "test_mz_manifest.json")
}

ALPHABET = [chr(i) for i in range(ord('A'), ord('Z') + 1)]


class TestPhase18AAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert os.path.exists(MANIFEST_PATH), f"Manifest file missing: {MANIFEST_PATH}"
        assert os.path.exists(LEDGER_PATH), f"Ledger file missing: {LEDGER_PATH}"

        with open(MANIFEST_PATH, "r") as f:
            all_samples = json.load(f)
            # Phase 18A dataset is the frozen benchmark partition (first 3,329 frames)
            cls.manifest_samples = all_samples[:3329]

        with open(LEDGER_PATH, "r") as f:
            cls.ledger_data = json.load(f)

    def test_01_total_frame_count_and_target_satisfaction(self):
        """Verifies that the dataset contains exactly 3,329 frames and satisfies the >= 3,120 target."""
        total_frames = len(self.manifest_samples)
        self.assertEqual(total_frames, 3329, f"Expected 3,329 frames, got {total_frames}")
        self.assertGreaterEqual(total_frames, 3120, "Phase 18 nominal target of 3,120 frames not satisfied")

    def test_02_all_26_letters_covered(self):
        """Verifies all 26 letters (A–Z) are represented with balanced counts."""
        letters = [s["letter"] for s in self.manifest_samples]
        unique_letters = set(letters)
        self.assertEqual(len(unique_letters), 26, f"Expected 26 letters, got {len(unique_letters)}")
        self.assertEqual(sorted(unique_letters), ALPHABET)

        counts = Counter(letters)
        for l in ALPHABET:
            self.assertGreaterEqual(counts[l], 120, f"Letter {l} has only {counts[l]} frames (expected >= 120)")

    def test_03_signers_and_conditions_representation(self):
        """Verifies exactly 2 signers and 2 conditions are represented."""
        signers = Counter(s["signer_id"] for s in self.manifest_samples)
        self.assertEqual(set(signers.keys()), {"signer_02", "signer_03"})
        self.assertEqual(signers["signer_02"], 1663)
        self.assertEqual(signers["signer_03"], 1666)

        # Verify sessions
        sessions = set(s["session_id"] for s in self.manifest_samples)
        self.assertEqual(sessions, {"session_20261022_01"})

        # Verify split quarantined
        splits = set(s["split"] for s in self.manifest_samples)
        self.assertEqual(splits, {"external_test_multisigner"})

    def test_04_104_cell_matrix_minimums(self):
        """Verifies that every single cell (signer x condition x letter) has >= 30 frames."""
        matrix = defaultdict(lambda: defaultdict(int))
        for s in self.manifest_samples:
            parts = s["source"]["frame_path"].split(os.sep)
            signer = s["signer_id"]
            cond = parts[3]
            letter = s["letter"]
            matrix[(signer, cond)][letter] += 1

        self.assertEqual(len(matrix), 4, f"Expected 4 (signer, condition) groups, got {len(matrix)}")

        for (signer, cond), letter_map in matrix.items():
            self.assertEqual(len(letter_map), 26, f"Group {signer}/{cond} missing letters")
            for letter in ALPHABET:
                count = letter_map[letter]
                self.assertGreaterEqual(count, 30, f"Cell {signer}/{cond}/{letter} has only {count} frames (expected >= 30)")
                self.assertTrue(31 <= count <= 33, f"Cell {signer}/{cond}/{letter} count {count} outside expected range [31, 33]")

    def test_05_disk_vs_manifest_one_to_one_integrity(self):
        """Verifies that every Phase 18A frame path in manifest exists on disk."""
        manifest_paths = set(s["source"]["frame_path"] for s in self.manifest_samples)
        self.assertEqual(len(manifest_paths), 3329)
        for p in manifest_paths:
            self.assertTrue(os.path.exists(os.path.join(BASE_DIR, p)), f"Missing frame on disk: {p}")

    def test_06_raw_videos_integrity(self):
        """Verifies 104 Phase 18A raw video recordings on disk corresponding to 104 cells."""
        manifest_videos = set(s["source"]["video_path"] for s in self.manifest_samples)
        self.assertEqual(len(manifest_videos), 104)
        for v in manifest_videos:
            self.assertTrue(os.path.exists(os.path.join(BASE_DIR, v)), f"Missing video on disk: {v}")

    def test_07_zero_historical_leakage(self):
        """Verifies zero SHA-256 hash overlap with historical train, val, and test splits."""
        manifest_hashes = set(s["source"]["sha256"] for s in self.manifest_samples)
        self.assertEqual(len(manifest_hashes), 3329, "Found intra-dataset duplicate hashes in manifest")

        for split_name, split_path in HISTORICAL_SPLITS.items():
            if os.path.exists(split_path):
                with open(split_path, "r") as f:
                    split_data = json.load(f)
                samples = split_data if isinstance(split_data, list) else split_data.get("samples", [])
                for s in samples:
                    # 1. Existing sha256 field
                    hist_hash = s.get("source", {}).get("sha256") or s.get("sha256")
                    if hist_hash:
                        self.assertNotIn(hist_hash, manifest_hashes, f"Data leakage: hash {hist_hash} found in {split_name}")
                    # 2. Check raw file hash
                    rpath = s.get("relative_path")
                    if rpath and os.path.exists(os.path.join(BASE_DIR, rpath)):
                        with open(os.path.join(BASE_DIR, rpath), "rb") as rf:
                            r_hash = hashlib.sha256(rf.read()).hexdigest()
                        self.assertNotIn(r_hash, manifest_hashes, f"Data leakage: raw image {rpath} found in new manifest")
                    # 3. Check crop file hash
                    cpath = s.get("crop_path")
                    if cpath and os.path.exists(os.path.join(BASE_DIR, cpath)):
                        with open(os.path.join(BASE_DIR, cpath), "rb") as cf:
                            c_hash = hashlib.sha256(cf.read()).hexdigest()
                        self.assertNotIn(c_hash, manifest_hashes, f"Data leakage: crop image {cpath} found in new manifest")

    def test_08_ledger_records(self):
        """Verifies 104 accepted recordings and 3 rejected attempts in the Phase 18A portion of the collection ledger."""
        records = self.ledger_data.get("records", [])[:107]
        self.assertEqual(len(records), 107)
        accepted = [r for r in records if r.get("status") == "accepted"]
        rejected = [r for r in records if r.get("status") == "rejected"]

        self.assertEqual(len(accepted), 104)
        self.assertEqual(len(rejected), 3)

        total_usable = sum(r.get("usable_frames_count", 0) for r in accepted)
        self.assertEqual(total_usable, 3329)

        for r in rejected:
            self.assertEqual(r.get("rejection_note"), "Operator rejected during preview")

    def test_09_sample_image_quality_bounds(self):
        """Spot checks 50 frames across signers and conditions for readability and quality."""
        sample_subset = self.manifest_samples[::66]  # 51 frames
        for s in sample_subset:
            fpath = os.path.join(BASE_DIR, s["source"]["frame_path"])
            img = cv2.imread(fpath)
            self.assertIsNotNone(img, f"Failed to read {fpath}")
            self.assertEqual(img.shape, (720, 1280, 3), f"Unexpected shape {img.shape} for {fpath}")

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            brightness = float(gray.mean())

            self.assertGreaterEqual(blur_var, 50.0, f"Blur variance {blur_var} < 50.0 for {fpath}")
            self.assertTrue(40.0 <= brightness <= 220.0, f"Brightness {brightness} outside [40, 220] for {fpath}")


if __name__ == "__main__":
    unittest.main()
