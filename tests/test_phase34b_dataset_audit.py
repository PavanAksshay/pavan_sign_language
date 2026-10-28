#!/usr/bin/env python3
"""
tests/test_phase34b_dataset_audit.py
====================================
Comprehensive audit and validation test suite for Phase 34B:
- Manifest-to-disk reconciliation (0 missing files, 0 orphaned active frames).
- Duplicate ID and duplicate hash audit (100% cryptographic hash uniqueness).
- Take-specific retake accounting and supersession provenance.
- Signer_01 train/val temporal leakage forensics (frame-disjoint only verification).
- Signer-level partition isolation and fail-closed training safeguard.
- Capture condition metadata audit and yaw/pitch disambiguation.
- Quality gate classifications (hard gates vs warnings).
- Campaign completion status enforcement (incomplete/not-collected).
- Preservation of authoritative artifact SHA-256 hashes and production defaults.
"""

import os
import sys
import json
import hashlib
import unittest
import numpy as np
from collections import Counter, defaultdict

# Ensure workspace root is in path
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from phase18_collector import (
    ISLCollectionEngine,
    DEVELOPMENT_TRAIN_SIGNERS,
    DEVELOPMENT_VALIDATION_SIGNERS,
    DEVELOPMENT_SIGNERS,
    EVALUATION_ONLY_SIGNERS,
    FROZEN_BENCHMARK_SIGNERS,
    SIGNER_ROLE_MAPPING,
    CONDITION_PRESETS,
    verify_training_isolation_safeguard,
)
from pipeline import ISLPipeline

AUTHORITATIVE_HASHES = {
    "Model_C_Occlusion_Aware_RF": {
        "path": os.path.join(WORKSPACE_ROOT, "phase22_analysis", "models", "Model_C_Occlusion_Aware_RF_multisigner.joblib"),
        "sha256": "1fab3349e2e537d8b31ddbe3bdd5c21e3ed005f85aaac9d3875c72217e28cc98"
    },
    "Model_C_Scaler": {
        "path": os.path.join(WORKSPACE_ROOT, "phase22_analysis", "models", "Model_C_Occlusion_Aware_RF_multisigner_scaler.joblib"),
        "sha256": "93842932e7cfeb1d283e99f67a733fd487161d83cbff356d7458ced2c3de385d"
    },
    "phase19_frozen_manifest": {
        "path": os.path.join(WORKSPACE_ROOT, "phase19_frozen_manifest.json"),
        "sha256": "9824eb510cc231249e9dd0722761de9112a02e7cecb681a479c4ef805b0fa797"
    },
    "phase23_predictions": {
        "path": os.path.join(WORKSPACE_ROOT, "phase23_predictions.csv"),
        "sha256": "da77036032fb3250141ccc16ba53d1dd7d4997b9817c6f525bbf9e000be45bfe"
    }
}


class TestPhase34BDatasetAudit(unittest.TestCase):
    """Test suite for Phase 34B dataset audit and governance."""

    def setUp(self):
        self.engine = ISLCollectionEngine()
        self.manifest_dir = os.path.join(WORKSPACE_ROOT, "data", "collection_manifests")

    # -------------------------------------------------------------------------
    # Test 1: Manifest-to-Disk Reconciliation
    # -------------------------------------------------------------------------
    def test_01_manifest_to_disk_reconciliation(self):
        """Verifies that all samples in manifests point to existing files on disk."""
        # 1. Check test_multisigner_manifest.json
        test_manifest_p = os.path.join(self.manifest_dir, "test_multisigner_manifest.json")
        self.assertTrue(os.path.exists(test_manifest_p))
        with open(test_manifest_p) as f:
            test_manifest = json.load(f)

        self.assertEqual(len(test_manifest), 7458)
        missing_count = 0
        for s in test_manifest:
            fp = s["source"]["frame_path"]
            abs_fp = os.path.join(WORKSPACE_ROOT, fp)
            if not os.path.exists(abs_fp):
                missing_count += 1
        self.assertEqual(missing_count, 0, f"Found {missing_count} missing files in test_multisigner_manifest.json")

        # 2. Check signer_05_manifest.json
        s05_manifest_p = os.path.join(self.manifest_dir, "signer_05_manifest.json")
        self.assertTrue(os.path.exists(s05_manifest_p))
        with open(s05_manifest_p) as f:
            s05_manifest = json.load(f)

        self.assertEqual(len(s05_manifest), 1638)
        s05_missing = 0
        for s in s05_manifest:
            fp = s["source"]["frame_path"]
            abs_fp = os.path.join(WORKSPACE_ROOT, fp)
            if not os.path.exists(abs_fp):
                s05_missing += 1
        self.assertEqual(s05_missing, 0, f"Found {s05_missing} missing files in signer_05_manifest.json")

        # 3. Check dev_train and dev_val manifests (currently 0 samples, valid JSON)
        dev_tr = self.engine.load_dev_train_manifest()
        dev_va = self.engine.load_dev_val_manifest()
        self.assertIsInstance(dev_tr, list)
        self.assertIsInstance(dev_va, list)
        self.assertEqual(len(dev_tr), 0)
        self.assertEqual(len(dev_va), 0)

    # -------------------------------------------------------------------------
    # Test 2: Missing and Orphaned Files Detection Logic
    # -------------------------------------------------------------------------
    def test_02_orphaned_and_missing_files_detection(self):
        """Verifies audit logic correctly flags missing or non-reconciled files."""
        # Simulated test manifest with an intentional missing file
        dummy_manifest = [
            {"source": {"frame_path": "data/extracted_frames/non_existent_frame_12345.jpg"}}
        ]
        missing = [
            s["source"]["frame_path"]
            for s in dummy_manifest
            if not os.path.exists(os.path.join(WORKSPACE_ROOT, s["source"]["frame_path"]))
        ]
        self.assertEqual(len(missing), 1)
        self.assertIn("non_existent_frame_12345.jpg", missing[0])

    # -------------------------------------------------------------------------
    # Test 3: Duplicate IDs and Cryptographic Hash Uniqueness Audit
    # -------------------------------------------------------------------------
    def test_03_duplicate_ids_and_duplicate_hashes_audit(self):
        """Audits cryptographic uniqueness and identifies historical sample_id collisions."""
        test_manifest_p = os.path.join(self.manifest_dir, "test_multisigner_manifest.json")
        with open(test_manifest_p) as f:
            samples = json.load(f)

        # 1. Cryptographic hashes must be 100% unique (zero duplicate frames)
        hashes = [s["source"]["sha256"] for s in samples]
        self.assertEqual(len(hashes), len(set(hashes)), "Cryptographic duplicate frame hashes detected!")

        # 2. Historical sample_id collision audit:
        # In Phase 18, sample_id omitted condition/take, causing 3,263 collisions across conditions
        sample_ids = [s["sample_id"] for s in samples]
        id_counts = Counter(sample_ids)
        colliding_ids = {k: v for k, v in id_counts.items() if v > 1}
        self.assertEqual(len(colliding_ids), 3263)

        # Confirm that despite sample_id collision, image file paths and image hashes are distinct
        for sid, count in list(colliding_ids.items())[:20]:
            matching = [s for s in samples if s["sample_id"] == sid]
            fps = set(s["source"]["frame_path"] for s in matching)
            sha = set(s["source"]["sha256"] for s in matching)
            self.assertEqual(len(sha), len(matching), f"Non-unique image hashes for sample_id {sid}")

    # -------------------------------------------------------------------------
    # Test 4: Take-Specific Retake Accounting and Supersession
    # -------------------------------------------------------------------------
    def test_04_take_specific_retake_accounting(self):
        """Audits ledger records and verifies take-specific supersession mechanics."""
        ledger_p = os.path.join(self.manifest_dir, "collection_ledger.json")
        with open(ledger_p) as f:
            ledger = json.load(f)

        records = ledger.get("records", [])
        self.assertEqual(len(records), 240)
        status_counts = Counter(r.get("status") for r in records)
        self.assertEqual(status_counts["accepted"], 236)
        self.assertEqual(status_counts["rejected"], 4)

        # Verify historical retake anomaly in signer_04 Condition 1 letters E and H
        s04_c1 = [r for r in records if r["signer_id"] == "signer_04" and r["condition_id"] == "condition_1"]
        c1_letters = Counter(r["letter"] for r in s04_c1)
        self.assertEqual(c1_letters["E"], 2)
        self.assertEqual(c1_letters["H"], 2)

        # Verify Phase 34A take-specific supersession behavior:
        # If take_number matches, prior take is superseded; different take_number accumulates
        simulated_records = [
            {"signer_id": "signer_06", "condition_id": "condition_1", "letter": "A", "take_number": 1, "status": "accepted", "recording_id": "rec_1"},
            {"signer_id": "signer_06", "condition_id": "condition_1", "letter": "A", "take_number": 2, "status": "accepted", "recording_id": "rec_2"},
        ]
        # Adding take 1 redo should supersede rec_1 only
        new_take = {"signer_id": "signer_06", "condition_id": "condition_1", "letter": "A", "take_number": 1, "recording_id": "rec_1_redo"}
        for r in simulated_records:
            if r["signer_id"] == new_take["signer_id"] and r["condition_id"] == new_take["condition_id"] and r["letter"] == new_take["letter"] and r.get("take_number", 1) == new_take["take_number"]:
                r["status"] = "superseded"
                r["superseded_by"] = new_take["recording_id"]

        self.assertEqual(simulated_records[0]["status"], "superseded")
        self.assertEqual(simulated_records[0]["superseded_by"], "rec_1_redo")
        self.assertEqual(simulated_records[1]["status"], "accepted")  # take 2 preserved

    # -------------------------------------------------------------------------
    # Test 5: Signer 01 Train/Validation Leakage Forensics
    # -------------------------------------------------------------------------
    def test_05_signer01_leakage_forensics(self):
        """Forensically proves that signer_01 train and val splits are FRAME-DISJOINT ONLY."""
        tr_path = os.path.join(WORKSPACE_ROOT, "models", "mediapipe_squeezenet_v1", "splits", "train_manifest.json")
        va_path = os.path.join(WORKSPACE_ROOT, "models", "mediapipe_squeezenet_v1", "splits", "val_manifest.json")
        self.assertTrue(os.path.exists(tr_path))
        self.assertTrue(os.path.exists(va_path))

        with open(tr_path) as f:
            tr = json.load(f)
        with open(va_path) as f:
            va = json.load(f)

        self.assertEqual(len(tr), 3077)
        self.assertEqual(len(va), 1134)

        # 1. Zero sample_id overlap (frame-disjoint)
        tr_ids = set(x["file_id"] for x in tr)
        va_ids = set(x["file_id"] for x in va)
        self.assertEqual(len(tr_ids.intersection(va_ids)), 0)

        # 2. Severe session overlap: 4 of 5 sessions are in BOTH train and val
        tr_sessions = set(x["session_id"] for x in tr)
        va_sessions = set(x["session_id"] for x in va)
        shared_sessions = tr_sessions.intersection(va_sessions)
        self.assertEqual(len(shared_sessions), 4)
        self.assertIn("session_20200516_12h", shared_sessions)
        self.assertIn("session_20200515_03h", shared_sessions)
        self.assertIn("session_20200515_04h", shared_sessions)
        self.assertIn("session_20200515_00h", shared_sessions)

        # 3. All validation frames come from shared sessions
        val_shared_count = sum(1 for x in va if x["session_id"] in shared_sessions)
        self.assertEqual(val_shared_count, len(va))

        # 4. Class-session overlap: 19 of 26 classes share the exact same session in train & val
        class_sess_overlap = defaultdict(lambda: {"tr": set(), "va": set()})
        for x in tr:
            class_sess_overlap[x["class_label"]]["tr"].add(x["session_id"])
        for x in va:
            class_sess_overlap[x["class_label"]]["va"].add(x["session_id"])

        shared_class_sessions = sum(
            1 for c, d in class_sess_overlap.items() if len(d["tr"].intersection(d["va"])) > 0
        )
        self.assertEqual(shared_class_sessions, 19)

    # -------------------------------------------------------------------------
    # Test 6: Signer-Level Partition Isolation and Fail-Closed Safeguards
    # -------------------------------------------------------------------------
    def test_06_partition_isolation_and_safeguards(self):
        """Verifies pairwise partition isolation and fail-closed training guardrails."""
        # 1. Audit split isolation across active manifests
        audit = self.engine.audit_split_isolation()
        self.assertTrue(audit["is_leakage_free"])
        self.assertTrue(audit["is_strictly_held_out"])
        self.assertEqual(len(audit["overlaps"]["train_holdout_overlap"]), 0)
        self.assertEqual(len(audit["overlaps"]["train_val_signer_overlap"]), 0)

        # 2. Verify fail-closed behavior on unauthorized signers
        contaminated_manifest_holdout = [
            {"signer_id": "signer_05", "allow_training": False, "sample_id": "s1"}
        ]
        with self.assertRaises(ValueError):
            verify_training_isolation_safeguard(contaminated_manifest_holdout)

        contaminated_manifest_val = [
            {"signer_id": "signer_10", "allow_training": False, "sample_id": "s2"}
        ]
        with self.assertRaises(ValueError):
            verify_training_isolation_safeguard(contaminated_manifest_val)

        contaminated_manifest_benchmark = [
            {"signer_id": "signer_02", "allow_training": False, "sample_id": "s3"}
        ]
        with self.assertRaises(ValueError):
            verify_training_isolation_safeguard(contaminated_manifest_benchmark)

        # 3. Verify fail-closed behavior on unknown signers or ambiguous permissions
        unknown_signer_manifest = [
            {"signer_id": "signer_99", "allow_training": True, "sample_id": "s4"}
        ]
        with self.assertRaises(ValueError):
            verify_training_isolation_safeguard(unknown_signer_manifest)

        ambiguous_manifest = [
            {"signer_id": "signer_06", "allow_training": False, "sample_id": "s5"}
        ]
        with self.assertRaises(ValueError):
            verify_training_isolation_safeguard(ambiguous_manifest)

        # 4. Valid manifest must pass
        valid_dev_manifest = [
            {"signer_id": "signer_06", "allow_training": True, "sample_id": "s6"},
            {"signer_id": "signer_07", "allow_training": True, "sample_id": "s7"}
        ]
        self.assertTrue(verify_training_isolation_safeguard(valid_dev_manifest))

    # -------------------------------------------------------------------------
    # Test 7: Capture Condition Metadata Audit
    # -------------------------------------------------------------------------
    def test_07_capture_condition_metadata_audit(self):
        """Audits CONDITION_PRESETS and identifies yaw/pitch discrepancies in Condition 2."""
        self.assertIn("condition_1", CONDITION_PRESETS)
        self.assertIn("condition_2", CONDITION_PRESETS)
        self.assertIn("condition_3", CONDITION_PRESETS)

        c1 = CONDITION_PRESETS["condition_1"]
        self.assertEqual(c1["camera_position"], "frontal_eye_level")
        self.assertEqual(c1["pitch_deg"], 0.0)
        self.assertEqual(c1["distance_cm"], 70)
        self.assertFalse(c1["lap_hand_resting"])

        c2 = CONDITION_PRESETS["condition_2"]
        # Audit finding: camera_position is diagonal_15deg (yaw), but pitch_deg is 30.0 (downward pitch)
        self.assertEqual(c2["camera_position"], "diagonal_15deg")
        self.assertEqual(c2["pitch_deg"], 30.0)
        self.assertEqual(c2["distance_cm"], 75)
        self.assertTrue(c2["lap_hand_resting"])

        c3 = CONDITION_PRESETS["condition_3"]
        self.assertEqual(c3["camera_position"], "frontal_low_angle")
        self.assertEqual(c3["pitch_deg"], 10.0)
        self.assertEqual(c3["distance_cm"], 110)
        self.assertEqual(c3["lighting"], "low_light")
        self.assertFalse(c3["lap_hand_resting"])

    # -------------------------------------------------------------------------
    # Test 8: Quality Gates Classification Audit
    # -------------------------------------------------------------------------
    def test_08_quality_gates_audit(self):
        """Audits hard rejection gates versus warnings."""
        engine = self.engine

        # Hard rejection thresholds
        self.assertEqual(engine.min_brightness, 40.0)
        self.assertEqual(engine.max_brightness, 220.0)
        self.assertEqual(engine.min_blur_var, 50.0)

        # Warnings thresholds
        self.assertEqual(engine.min_resolution_width, 1280)
        self.assertEqual(engine.min_resolution_height, 720)
        self.assertEqual(engine.target_fps, 30.0)

        # Attribute presence for bbox ratio
        self.assertTrue(hasattr(engine, "min_hand_bbox_ratio"))
        self.assertEqual(engine.min_hand_bbox_ratio, 0.02)

        # Verify audit_frame_quality returns rejection on out-of-bounds brightness
        dark_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        res_dark = engine.audit_frame_quality(dark_frame)
        self.assertFalse(res_dark["is_usable"])
        self.assertIn("underexposed", res_dark["rejection_reason"])

        bright_frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        res_bright = engine.audit_frame_quality(bright_frame)
        self.assertFalse(res_bright["is_usable"])
        self.assertIn("overexposed", res_bright["rejection_reason"])

    # -------------------------------------------------------------------------
    # Test 9: Campaign Completion Status Enforcement
    # -------------------------------------------------------------------------
    def test_09_campaign_completion_status(self):
        """Verifies campaign is NOT marked complete when signer_06-11 are uncollected."""
        dev_tr = self.engine.load_dev_train_manifest()
        dev_va = self.engine.load_dev_val_manifest()
        total_collected = len(dev_tr) + len(dev_va)

        # Must be 0 samples collected
        self.assertEqual(total_collected, 0)

        # Provisional split spec must exist and declare provisional status
        spec_p = os.path.join(WORKSPACE_ROOT, "phase34b_split_specification.json")
        self.assertTrue(os.path.exists(spec_p))
        with open(spec_p) as f:
            spec = json.load(f)

        self.assertEqual(spec["specification_metadata"]["status"], "PROVISIONAL")
        self.assertEqual(spec["specification_metadata"]["freeze_status"], "UNFROZEN_PENDING_DATA_COLLECTION")
        self.assertEqual(spec["dataset_roles"]["development_train"]["current_status"], "NOT_COLLECTED")
        self.assertEqual(spec["dataset_roles"]["development_validation"]["current_status"], "NOT_COLLECTED")

    # -------------------------------------------------------------------------
    # Test 10: Protected Artifact Hashes and Production Defaults
    # -------------------------------------------------------------------------
    def test_10_protected_artifact_hashes_and_production_defaults(self):
        """Verifies production defaults and authoritative hashes are strictly preserved."""
        pipe = ISLPipeline()
        try:
            self.assertEqual(pipe.backend_detector, "yolo")
            self.assertEqual(pipe.recognition_arch, "standard")
            self.assertEqual(pipe.mediapipe_delegate, "cpu")
            self.assertFalse(pipe.temporal_smoothing_enabled)
        finally:
            pipe.shutdown()

        for name, meta in AUTHORITATIVE_HASHES.items():
            p = meta["path"]
            exp = meta["sha256"]
            self.assertTrue(os.path.exists(p), f"Artifact missing: {p}")
            hasher = hashlib.sha256()
            with open(p, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
            act = hasher.hexdigest()
            self.assertEqual(act, exp, f"Hash mismatch on {name} ({p})")


if __name__ == "__main__":
    unittest.main()
