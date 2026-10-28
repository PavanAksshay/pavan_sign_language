#!/usr/bin/env python3
"""
Phase 28 — Cryptographic Integrity and Configuration Safety Audit
Validates SHA-256 hashes of all frozen historical artifacts (Phases 19-27)
and confirms strict preservation of production defaults.
Outputs: phase28_integrity_audit.json
"""

import os
import sys
import json
import hashlib
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from pipeline import ISLPipeline

AUTHORITATIVE_ARTIFACTS = {
    # Phase 19
    "phase19_frozen_manifest.json": "9824eb510cc231249e9dd0722761de9112a02e7cecb681a479c4ef805b0fa797",
    "phase19_frozen_external_benchmark_report.md": "b35a1b1ab0fae37da1b8b472551b78c95a4b1ac6cf92cf148ca87fe45e4d6010",
    "phase19_analysis/phase19_predictions_raw.csv": "54bd1739ef73239e42c6bf2b15686d5fcffff8613668c2b55eafbacf14395649",

    # Phase 22
    "phase22_training_report.md": "e90cabd514f2f7877ec18cd0d8541fe270d135c213b5e4f760e7acdb0345af7f",
    "phase22_analysis/models/Model_C_Occlusion_Aware_RF_multisigner.joblib": "1fab3349e2e537d8b31ddbe3bdd5c21e3ed005f85aaac9d3875c72217e28cc98",
    "phase22_analysis/models/Model_C_Occlusion_Aware_RF_multisigner_scaler.joblib": "93842932e7cfeb1d283e99f67a733fd487161d83cbff356d7458ced2c3de385d",

    # Phase 23
    "phase23_external_evaluation_report.md": "3f215e9a4a24870eaf67c76884b523b157e2414a7f38072944668537cc1cc128",
    "phase23_predictions.csv": "da77036032fb3250141ccc16ba53d1dd7d4997b9817c6f525bbf9e000be45bfe",
    "phase23_external_metrics.json": "48468fa3008f564e4fc66663205ee0811b9da1b9add9d6adc111f7a0431e5534",
    "phase23_integrity_audit.json": "ee8ba55bf0c863d004fd3859ae0b440748bc9a5fba139ce4c6bfa8a421c1cbd0",

    # Phase 24
    "phase24_runtime_integration_report.md": "bc73ee59730a96c12c49ef56e2b19bfbacd45c733b0cea713982c2464ebf789c",
    "phase24_runtime_metrics.json": "aad2781a9fd8b4c558e374dd1588eea7858fca59c93a1cab1bb2b2aec9988ae1",
    "phase24_integrity_audit.json": "a8053f32d9a63ab0ca5e71016494fad03a51453caf000f3e210ad0b07916c82f",

    # Phase 25
    "phase25_performance_report.md": "6b4cb9b05a569b72e43623e8656817db4e05f9a441a34381283d2d4b6931bdd8",
    "phase25_runtime_metrics.json": "2eb05c491c6320d3df0e231db3c11007ca7b822efc1ec2420cac983f063e013d",
    "phase25_integrity_audit.json": "f11949bc14d59fc6602637be4c25b44cfaa94ae5557a272fe351d7a3cbdf2e44",

    # Phase 26
    "phase26_latency_forensics_report.md": "1c417730407a56367877b4adfa69bf747d514ca1db69a63020530e93f506df7d",
    "phase26_runtime_metrics.json": "b992b601855c6ec4140e7a443b70dfa2365bdcc0d59b1e402604bd18f78bf9d0",
    "phase26_integrity_audit.json": "729ff2fa76245d4fbb720cb65fb20a72d5266443ffd50e6175550b5d1cf516cd",

    # Phase 27
    "phase27_live_pilot_report.md": "003e6be72f12ce3ba0c647687e8a372df26f5defb6ba014c15046eef51574236",
    "phase27_pilot_metrics.json": "092dce67e32b2f57915f2f0deb75948cb79b79fa5e036928372923f8c36c5355",
    "phase27_video_summaries.json": "fecb859359cff6565a065ca11adde3b540775ff93681b3e74a8f4f9033707b05",
    "phase27_integrity_audit.json": "ee4701daaecbd09426ff2ceabe04f9a31f1621584ec817bc556a603e4956f4c1"
}


def main():
    print("=" * 75)
    print("Phase 28 — Cryptographic Integrity & Production Defaults Audit")
    print("=" * 75)

    all_match = True
    audit_results = {}

    for rel_path, expected_hash in AUTHORITATIVE_ARTIFACTS.items():
        abs_path = os.path.join(BASE_DIR, rel_path)
        if not os.path.exists(abs_path):
            print(f"[MISSING] {rel_path}")
            audit_results[rel_path] = {"exists": False, "matches": False}
            all_match = False
            continue

        with open(abs_path, "rb") as fp:
            data = fp.read()
            actual_hash = hashlib.sha256(data).hexdigest()

        matches = (actual_hash == expected_hash)
        if not matches:
            all_match = False
            print(f"[MISMATCH] {rel_path}: expected {expected_hash}, got {actual_hash}")
        else:
            print(f"[OK] {rel_path} (100% SHA-256 match)")

        audit_results[rel_path] = {
            "exists": True,
            "sha256": actual_hash,
            "matches_baseline": matches
        }

    # Verify production defaults
    pipe = ISLPipeline()
    prod_audit = {
        "ISL_BACKEND_DETECTOR": pipe.backend_detector,
        "ISL_RECOGNITION_ARCH": pipe.recognition_arch,
        "ISL_TEMPORAL_SMOOTHING": pipe.temporal_smoothing_enabled,
        "ISL_SHADOW_MODE": pipe.shadow_mode_enabled,
        "ISL_MEDIAPIPE_DELEGATE": pipe.mediapipe_delegate,
        "ISL_MODEL_C_RESOLUTION": getattr(pipe, "model_c_input_resolution", "native")
    }
    pipe.shutdown()

    prod_pass = (
        prod_audit["ISL_BACKEND_DETECTOR"] == "yolo" and
        prod_audit["ISL_RECOGNITION_ARCH"] == "standard" and
        prod_audit["ISL_TEMPORAL_SMOOTHING"] is False and
        prod_audit["ISL_SHADOW_MODE"] is False and
        prod_audit["ISL_MEDIAPIPE_DELEGATE"] == "cpu" and
        prod_audit["ISL_MODEL_C_RESOLUTION"] == "native"
    )

    print(f"\nProduction Defaults Audit: {'PASS' if prod_pass else 'FAIL'}")
    for k, v in prod_audit.items():
        print(f"  {k} = {v}")

    out_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phase": "Phase 28 — Sustained GPU Latency Stabilization and Benchmark Integrity Audit",
        "all_baseline_artifacts_unmodified": all_match,
        "production_defaults_verified": prod_audit,
        "production_defaults_pass": prod_pass,
        "artifacts_audited": audit_results,
        "overall_audit_passed": all_match and prod_pass
    }

    out_path = os.path.join(BASE_DIR, "phase28_integrity_audit.json")
    with open(out_path, "w") as fp:
        json.dump(out_data, fp, indent=2)
    print(f"\nWrote integrity audit to: {out_path}")
    print(f"Overall Audit Status: {'PASSED' if (all_match and prod_pass) else 'FAILED'}")
    print("=" * 75)


if __name__ == "__main__":
    main()
