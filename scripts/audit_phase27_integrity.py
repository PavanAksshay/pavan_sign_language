#!/usr/bin/env python3
"""
Phase 27 — Cryptographic Integrity and Baseline Audit Engine
Verifies that all frozen artifacts from Phases 19, 22, 23, 24, 25, and 26 remain unaltered,
and audits production defaults.
"""

import os
import sys
import json
import hashlib
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

EXPECTED_HASHES = {
    "phase19_frozen_manifest.json": "9824eb510cc231249e9dd0722761de9112a02e7cecb681a479c4ef805b0fa797",
    "phase19_frozen_external_benchmark_report.md": "b35a1b1ab0fae37da1b8b472551b78c95a4b1ac6cf92cf148ca87fe45e4d6010",
    "phase22_training_report.md": "e90cabd514f2f7877ec18cd0d8541fe270d135c213b5e4f760e7acdb0345af7f",
    "phase22_analysis/models/Model_C_Occlusion_Aware_RF_multisigner.joblib": "1fab3349e2e537d8b31ddbe3bdd5c21e3ed005f85aaac9d3875c72217e28cc98",
    "phase22_analysis/models/Model_C_Occlusion_Aware_RF_multisigner_scaler.joblib": "93842932e7cfeb1d283e99f67a733fd487161d83cbff356d7458ced2c3de385d",
    "phase23_external_evaluation_report.md": "3f215e9a4a24870eaf67c76884b523b157e2414a7f38072944668537cc1cc128",
    "phase23_predictions.csv": "da77036032fb3250141ccc16ba53d1dd7d4997b9817c6f525bbf9e000be45bfe",
    "phase23_external_metrics.json": "48468fa3008f564e4fc66663205ee0811b9da1b9add9d6adc111f7a0431e5534",
    "phase23_integrity_audit.json": "ee8ba55bf0c863d004fd3859ae0b440748bc9a5fba139ce4c6bfa8a421c1cbd0",
    "phase24_runtime_integration_report.md": "bc73ee59730a96c12c49ef56e2b19bfbacd45c733b0cea713982c2464ebf789c",
    "phase24_runtime_metrics.json": "aad2781a9fd8b4c558e374dd1588eea7858fca59c93a1cab1bb2b2aec9988ae1",
    "phase24_integrity_audit.json": "a8053f32d9a63ab0ca5e71016494fad03a51453caf000f3e210ad0b07916c82f",
    "phase25_performance_report.md": "6b4cb9b05a569b72e43623e8656817db4e05f9a441a34381283d2d4b6931bdd8",
    "phase25_runtime_metrics.json": "2eb05c491c6320d3df0e231db3c11007ca7b822efc1ec2420cac983f063e013d",
    "phase25_integrity_audit.json": "f11949bc14d59fc6602637be4c25b44cfaa94ae5557a272fe351d7a3cbdf2e44",
    "phase26_latency_forensics_report.md": "1c417730407a56367877b4adfa69bf747d514ca1db69a63020530e93f506df7d",
    "phase26_runtime_metrics.json": "b992b601855c6ec4140e7a443b70dfa2365bdcc0d59b1e402604bd18f78bf9d0",
    "phase26_integrity_audit.json": "729ff2fa76245d4fbb720cb65fb20a72d5266443ffd50e6175550b5d1cf516cd"
}

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=" * 70)
    print("Phase 27 — Cryptographic Integrity and Baseline Audit")
    print("=" * 70)
    audit_results = {}
    all_matched = True

    for rel_path, exp_hash in EXPECTED_HASHES.items():
        abs_path = os.path.join(BASE_DIR, rel_path)
        if not os.path.exists(abs_path):
            print(f"❌ MISSING: {rel_path}")
            audit_results[rel_path] = {
                "exists": False,
                "matches_baseline": False,
                "expected": exp_hash
            }
            all_matched = False
            continue

        calc_hash = sha256_file(abs_path)
        matches = (calc_hash == exp_hash)
        if matches:
            print(f"✓ MATCH: {rel_path}")
        else:
            print(f"❌ MISMATCH: {rel_path} (got {calc_hash}, expected {exp_hash})")
            all_matched = False

        audit_results[rel_path] = {
            "sha256": calc_hash,
            "matches_baseline": matches
        }

    # Verify production defaults in pipeline.py
    from pipeline import ISLPipeline
    pipe = ISLPipeline()
    prod_defaults = {
        "ISL_BACKEND_DETECTOR": pipe.backend_detector,
        "ISL_RECOGNITION_ARCH": pipe.recognition_arch,
        "ISL_TEMPORAL_SMOOTHING": pipe.temporal_smoothing_enabled,
        "ISL_SHADOW_MODE": pipe.shadow_mode_enabled,
        "ISL_MEDIAPIPE_DELEGATE": pipe.mediapipe_delegate,
        "ISL_MODEL_C_RESOLUTION": pipe.model_c_input_resolution
    }
    defaults_ok = (
        prod_defaults["ISL_BACKEND_DETECTOR"] == "yolo" and
        prod_defaults["ISL_RECOGNITION_ARCH"] == "standard" and
        prod_defaults["ISL_TEMPORAL_SMOOTHING"] is False and
        prod_defaults["ISL_SHADOW_MODE"] is False and
        prod_defaults["ISL_MEDIAPIPE_DELEGATE"] == "cpu" and
        prod_defaults["ISL_MODEL_C_RESOLUTION"] == "native"
    )

    out_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phase": "Phase 27 — Controlled Live Pilot, Continuous-Signing Evaluation, and Deployment Readiness Integrity Audit",
        "all_baseline_artifacts_unmodified": all_matched,
        "production_defaults_verified": prod_defaults,
        "production_defaults_pass": defaults_ok,
        "artifacts_audited": audit_results
    }

    out_file = os.path.join(BASE_DIR, "phase27_integrity_audit.json")
    with open(out_file, "w") as f:
        json.dump(out_data, f, indent=2)

    print(f"\nAll baseline artifacts unmodified: {all_matched}")
    print(f"Production defaults verified: {defaults_ok}")
    print(f"Audit saved to {out_file}")

if __name__ == "__main__":
    main()
