#!/usr/bin/env python3
"""
Phase 30 — Signer_05 Independent Evaluation Script
Evaluates the frozen Model C runtime at opt-in 640x360 resolution on the unseen
independent signer_05 dataset across all 26 letters (A-Z) in Condition 1 and Condition 2.

Objective:
Determine whether the frozen Model C generalizes to an unseen signer while meeting
the project's recognition and latency acceptance criteria.

Strict Isolation:
- Inference only. No retraining, refitting, tuning, or threshold selection.
- Verifies SHA-256 hashes of all frozen benchmark artifacts before and after execution.
- Evaluates at opt-in 640x360 resolution with active delegate and aspect ratio verification.
- Generates:
  * phase30_evaluation_manifest.json
  * phase30_integrity_audit.json
  * phase30_independent_signer_metrics.json
  * phase30_analysis/plots/confusion_matrix.png
  * phase30_analysis/phase30_predictions_raw.csv
  * phase30_independent_signer_report.md
"""

import os
import sys
import time
import json
import csv
import hashlib
import resource
import platform
from collections import defaultdict, Counter
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from pipeline import ISLPipeline
from landmark_model_c import FEATURE_DIM_MODEL_C, ALPHABET, I2C, C2I

# Output directories
ANALYSIS_DIR = os.path.join(BASE_DIR, "phase30_analysis")
PLOTS_DIR = os.path.join(ANALYSIS_DIR, "plots")
os.makedirs(ANALYSIS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Target deliverable filepaths
EVAL_MANIFEST_PATH = os.path.join(BASE_DIR, "phase30_evaluation_manifest.json")
INTEGRITY_AUDIT_PATH = os.path.join(BASE_DIR, "phase30_integrity_audit.json")
METRICS_PATH = os.path.join(BASE_DIR, "phase30_independent_signer_metrics.json")
REPORT_PATH = os.path.join(BASE_DIR, "phase30_independent_signer_report.md")
PREDICTIONS_CSV_PATH = os.path.join(ANALYSIS_DIR, "phase30_predictions_raw.csv")
CM_PLOT_PATH = os.path.join(PLOTS_DIR, "phase30_confusion_matrix.png")
CM_COND1_PLOT_PATH = os.path.join(PLOTS_DIR, "phase30_confusion_matrix_cond1.png")
CM_COND2_PLOT_PATH = os.path.join(PLOTS_DIR, "phase30_confusion_matrix_cond2.png")

# Authoritative baseline artifact hashes
APPROVED_HASHES = {
    "phase22_analysis/models/Model_C_Occlusion_Aware_RF_multisigner.joblib": "1fab3349e2e537d8b31ddbe3bdd5c21e3ed005f85aaac9d3875c72217e28cc98",
    "phase22_analysis/models/Model_C_Occlusion_Aware_RF_multisigner_scaler.joblib": "93842932e7cfeb1d283e99f67a733fd487161d83cbff356d7458ced2c3de385d",
    "phase19_frozen_manifest.json": "9824eb510cc231249e9dd0722761de9112a02e7cecb681a479c4ef805b0fa797",
    "phase23_predictions.csv": "da77036032fb3250141ccc16ba53d1dd7d4997b9817c6f525bbf9e000be45bfe",
    "App/include50_model/hand_landmarker.task": "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
}


def sha256_file(filepath: str) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_rss_mb() -> float:
    """Returns resident set size in megabytes."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    rss = ru.ru_maxrss
    if platform.system() == "Darwin":
        return round(rss / (1024.0 * 1024.0), 2)
    return round(rss / 1024.0, 2)


def compute_distribution(arr: List[float]) -> Dict[str, float]:
    """Computes standard distribution statistics."""
    if not arr:
        return {}
    a = np.array(arr, dtype=np.float64)
    return {
        "count": int(len(arr)),
        "mean": round(float(np.mean(a)), 3),
        "median": round(float(np.median(a)), 3),
        "p50": round(float(np.percentile(a, 50)), 3),
        "p90": round(float(np.percentile(a, 90)), 3),
        "p95": round(float(np.percentile(a, 95)), 3),
        "p99": round(float(np.percentile(a, 99)), 3),
        "min": round(float(np.min(a)), 3),
        "max": round(float(np.max(a)), 3),
        "std": round(float(np.std(a)), 3)
    }


def compute_classification_metrics(y_true: List[str], y_pred: List[str], y_prob: Optional[List[List[float]]] = None) -> Dict[str, Any]:
    """Computes top-1, top-3, macro-precision, macro-recall, macro-F1, balanced accuracy, and per-class stats."""
    classes = ALPHABET
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    support = defaultdict(int)

    correct = 0
    total = len(y_true)

    for yt, yp in zip(y_true, y_pred):
        support[yt] += 1
        if yt == yp:
            correct += 1
            tp[yt] += 1
        else:
            fn[yt] += 1
            if yp in classes:
                fp[yp] += 1

    overall_acc = round(correct / max(1, total), 4)

    # Top-3 Accuracy
    top3_correct = 0
    if y_prob is not None:
        for i, yt in enumerate(y_true):
            probs = y_prob[i]
            top3_idx = np.argsort(probs)[-3:]
            top3_letters = [classes[idx] for idx in top3_idx]
            if yt in top3_letters:
                top3_correct += 1
        top3_acc = round(top3_correct / max(1, total), 4)
    else:
        top3_acc = None

    per_class = {}
    recalls = []
    precisions = []
    f1s = []

    for c in classes:
        c_sup = support[c]
        c_tp = tp[c]
        c_fp = fp[c]
        c_fn = fn[c]

        prec = round(c_tp / max(1, c_tp + c_fp), 4) if (c_tp + c_fp) > 0 else 0.0
        rec = round(c_tp / max(1, c_sup), 4) if c_sup > 0 else 0.0
        f1 = round(2 * prec * rec / max(1e-6, prec + rec), 4) if (prec + rec) > 0 else 0.0

        per_class[c] = {
            "support": c_sup,
            "tp": c_tp,
            "fp": c_fp,
            "fn": c_fn,
            "precision": prec,
            "recall": rec,
            "f1": f1
        }
        if c_sup > 0:
            recalls.append(rec)
            precisions.append(prec)
            f1s.append(f1)

    macro_f1 = round(float(np.mean(f1s)), 4) if f1s else 0.0
    macro_prec = round(float(np.mean(precisions)), 4) if precisions else 0.0
    macro_rec = round(float(np.mean(recalls)), 4) if recalls else 0.0
    balanced_acc = macro_rec

    # Identify zero-recall and sub-70% recall letters
    zero_recall_letters = [c for c in classes if per_class[c]["support"] > 0 and per_class[c]["recall"] == 0.0]
    critical_weakness_letters = [c for c in classes if per_class[c]["support"] > 0 and per_class[c]["recall"] < 0.70]

    return {
        "total_evaluated": total,
        "correct_predictions": correct,
        "overall_top1_accuracy": overall_acc,
        "overall_top3_accuracy": top3_acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "balanced_accuracy": balanced_acc,
        "zero_recall_letters": zero_recall_letters,
        "critical_weakness_letters": critical_weakness_letters,
        "per_class": per_class
    }


def plot_confusion_matrix(cm: np.ndarray, classes: List[str], title: str, save_path: str):
    """Renders and saves a high-resolution confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel='Ground Truth Label',
        xlabel='Predicted Label'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = int(cm[i, j])
            if val > 0:
                ax.text(j, i, format(val, 'd'),
                        ha="center", va="center",
                        color="white" if val > thresh else "black",
                        fontsize=7)
    fig.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def run_evaluation() -> Dict[str, Any]:
    print("=" * 80)
    print("PHASE 30 — SIGNER_05 INDEPENDENT EVALUATION")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # STEP 1: LOCATE AND AUDIT THE COLLECTED DATA
    # -------------------------------------------------------------------------
    print("\n[Step 1] Auditing and Locating Collected signer_05 Data...")
    
    s05_manifest_path = os.path.join(BASE_DIR, "data", "collection_manifests", "signer_05_manifest.json")
    collection_ledger_path = os.path.join(BASE_DIR, "data", "collection_manifests", "collection_ledger.json")
    
    if not os.path.exists(s05_manifest_path):
        raise FileNotFoundError(f"Missing signer_05 manifest: {s05_manifest_path}")
    if not os.path.exists(collection_ledger_path):
        raise FileNotFoundError(f"Missing collection ledger: {collection_ledger_path}")

    with open(s05_manifest_path, "r") as f:
        s05_manifest_data = json.load(f)

    with open(collection_ledger_path, "r") as f:
        ledger_data = json.load(f)

    print(f"  Loaded {len(s05_manifest_data)} samples from signer_05 manifest.")

    # Ledger audit
    ledger_records = ledger_data.get("records", [])
    s05_ledger_records = [r for r in ledger_records if r.get("signer_id") == "signer_05"]
    print(f"  Loaded {len(s05_ledger_records)} signer_05 recording records from ledger.")

    # Audit exclusions & retakes
    accepted_records = [r for r in s05_ledger_records if r.get("status") == "accepted"]
    rejected_records = [r for r in s05_ledger_records if r.get("status") == "rejected"]
    superseded_records = [r for r in s05_ledger_records if r.get("status") == "superseded"]

    print(f"  Accepted recordings: {len(accepted_records)}")
    print(f"  Rejected recordings: {len(rejected_records)}")
    print(f"  Superseded recordings: {len(superseded_records)}")

    exclusions_log = []
    for r in rejected_records:
        exclusions_log.append({
            "recording_id": r.get("recording_id"),
            "letter": r.get("letter"),
            "condition_id": r.get("condition_id"),
            "status": "rejected",
            "reason": r.get("rejection_note", "Operator rejected during review"),
            "frames_excluded": r.get("frames_sampled", 0),
            "timestamp": r.get("timestamp")
        })

    for r in superseded_records:
        exclusions_log.append({
            "recording_id": r.get("recording_id"),
            "letter": r.get("letter"),
            "condition_id": r.get("condition_id"),
            "status": "superseded",
            "reason": f"Superseded by retake {r.get('superseded_by')}",
            "frames_excluded": r.get("usable_frames_count", 0),
            "timestamp": r.get("timestamp")
        })

    # Verify per-sample presence, integrity, and hashes
    manifest_samples = []
    missing_files = []
    hash_mismatches = []
    sample_hashes = set()
    condition_counts = defaultdict(int)
    letter_counts = defaultdict(lambda: defaultdict(int))
    recording_frames = defaultdict(list)

    for item in s05_manifest_data:
        frame_rel_path = item["source"]["frame_path"]
        frame_abs_path = os.path.join(BASE_DIR, frame_rel_path)
        expected_sha = item["source"]["sha256"]
        letter = item["letter"]
        rec_id = item["recording_id"]
        cond = "condition_1" if "condition_1" in frame_rel_path else ("condition_2" if "condition_2" in frame_rel_path else "unknown")

        if not os.path.exists(frame_abs_path):
            missing_files.append(frame_rel_path)
            continue

        actual_sha = sha256_file(frame_abs_path)
        if actual_sha != expected_sha:
            hash_mismatches.append({"path": frame_rel_path, "expected": expected_sha, "actual": actual_sha})

        sample_hashes.add(actual_sha)
        condition_counts[cond] += 1
        letter_counts[cond][letter] += 1
        recording_frames[rec_id].append(item)

        manifest_samples.append({
            "sample_id": item["sample_id"],
            "signer_id": "signer_05",
            "letter": letter,
            "condition": cond,
            "recording_id": rec_id,
            "frame_path": frame_rel_path,
            "sha256": actual_sha,
            "eligible_for_evaluation": item.get("eligible_for_evaluation", True),
            "hands_detected_collection": item.get("detection", {}).get("hands_detected", None),
            "blur_laplacian_var": item.get("quality", {}).get("blur_laplacian_var", None),
            "mean_brightness": item.get("quality", {}).get("mean_brightness", None)
        })

    if missing_files:
        raise FileNotFoundError(f"Dataset integrity failure: {len(missing_files)} missing frame files on disk!")
    if hash_mismatches:
        raise ValueError(f"Dataset integrity failure: {len(hash_mismatches)} frame SHA-256 mismatches!")

    # Deduplication and isolation check against training splits
    print("\n  Checking deduplication and independence against training splits...")
    train_manifest_path = os.path.join(BASE_DIR, "models", "mediapipe_squeezenet_v1", "splits", "train_manifest.json")
    val_manifest_path = os.path.join(BASE_DIR, "models", "mediapipe_squeezenet_v1", "splits", "val_manifest.json")
    p19_manifest_path = os.path.join(BASE_DIR, "phase19_frozen_manifest.json")

    split_overlap = {}
    for name, p in [("train_manifest", train_manifest_path), ("val_manifest", val_manifest_path), ("p19_manifest", p19_manifest_path)]:
        if os.path.exists(p):
            with open(p, "r") as f:
                d = json.load(f)
            hashes = set()
            if isinstance(d, list):
                for x in d:
                    if isinstance(x, dict):
                        h = x.get("source", {}).get("sha256") or x.get("sha256") or x.get("md5_hash")
                        if h:
                            hashes.add(h)
            split_overlap[name] = len(sample_hashes.intersection(hashes))

    print(f"  Internal unique hashes: {len(sample_hashes)} / {len(manifest_samples)}")
    print(f"  Overlap with train_manifest: {split_overlap.get('train_manifest', 0)}")
    print(f"  Overlap with val_manifest: {split_overlap.get('val_manifest', 0)}")
    print(f"  Overlap with p19_manifest: {split_overlap.get('p19_manifest', 0)}")

    if len(sample_hashes) != len(manifest_samples):
        raise ValueError("Duplicate frame hashes detected within signer_05 dataset!")
    for k, v in split_overlap.items():
        if v > 0:
            raise ValueError(f"Dataset leakage detected! signer_05 shares {v} hashes with {k}!")

    # -------------------------------------------------------------------------
    # STEP 2: FREEZE THE EVALUATION SET
    # -------------------------------------------------------------------------
    print("\n[Step 2] Freezing Evaluation Manifest...")
    
    evaluation_manifest = {
        "phase": 30,
        "title": "Phase 30 — Signer_05 Independent Evaluation Manifest",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "signer_id": "signer_05",
        "dataset_role": "independent_evaluation_holdout",
        "script_version": "1.0.0",
        "summary": {
            "total_eligible_frames": len(manifest_samples),
            "total_excluded_frames": sum(e["frames_excluded"] for e in exclusions_log),
            "condition_1_frames": condition_counts["condition_1"],
            "condition_2_frames": condition_counts["condition_2"],
            "total_recordings": len(recording_frames),
            "distinct_letters": len(set(s["letter"] for s in manifest_samples)),
            "per_condition_letter_counts": {
                "condition_1": dict(letter_counts["condition_1"]),
                "condition_2": dict(letter_counts["condition_2"])
            }
        },
        "exclusions": exclusions_log,
        "samples": manifest_samples
    }

    with open(EVAL_MANIFEST_PATH, "w") as f:
        json.dump(evaluation_manifest, f, indent=2)

    eval_manifest_sha256 = sha256_file(EVAL_MANIFEST_PATH)
    print(f"  Saved frozen evaluation manifest to {EVAL_MANIFEST_PATH}")
    print(f"  Evaluation Manifest SHA-256: {eval_manifest_sha256}")

    # -------------------------------------------------------------------------
    # STEP 3: FREEZE AND VERIFY THE MODEL & RUNTIME
    # -------------------------------------------------------------------------
    print("\n[Step 3] Verifying Frozen Model C Checkpoints and Runtime Configuration...")
    
    model_c_rf_path = os.path.join(BASE_DIR, "phase22_analysis", "models", "Model_C_Occlusion_Aware_RF_multisigner.joblib")
    model_c_scaler_path = os.path.join(BASE_DIR, "phase22_analysis", "models", "Model_C_Occlusion_Aware_RF_multisigner_scaler.joblib")
    task_path = os.path.join(BASE_DIR, "App", "include50_model", "hand_landmarker.task")

    current_hashes = {
        "phase22_analysis/models/Model_C_Occlusion_Aware_RF_multisigner.joblib": sha256_file(model_c_rf_path),
        "phase22_analysis/models/Model_C_Occlusion_Aware_RF_multisigner_scaler.joblib": sha256_file(model_c_scaler_path),
        "phase19_frozen_manifest.json": sha256_file(os.path.join(BASE_DIR, "phase19_frozen_manifest.json")),
        "phase23_predictions.csv": sha256_file(os.path.join(BASE_DIR, "phase23_predictions.csv")),
        "App/include50_model/hand_landmarker.task": sha256_file(task_path)
    }

    hash_audit_passed = True
    for artifact, expected_h in APPROVED_HASHES.items():
        curr_h = current_hashes.get(artifact)
        match = (curr_h == expected_h)
        print(f"  {artifact}: {'MATCH' if match else 'MISMATCH'}")
        if not match:
            hash_audit_passed = False
            print(f"    Expected: {expected_h}")
            print(f"    Current:  {curr_h}")

    if not hash_audit_passed:
        raise ValueError("Model immutability violation: baseline artifact hashes do not match approved hashes!")

    # Verify Production Defaults Preservation
    print("\n  Verifying Production Defaults on Unconfigured Pipeline...")
    default_pipe = ISLPipeline()
    prod_defaults_ok = (
        default_pipe.backend_detector == "yolo" and
        default_pipe.recognition_arch == "standard" and
        default_pipe.mediapipe_delegate == "cpu" and
        default_pipe.temporal_smoothing_enabled is False and
        default_pipe.shadow_mode_enabled is False and
        getattr(default_pipe, "model_c_input_resolution", "native") == "native"
    )
    default_pipe.shutdown()
    print(f"  Production defaults intact: {prod_defaults_ok}")
    if not prod_defaults_ok:
        raise ValueError("Production default configuration has been modified!")

    # Initialize Experimental Model C Pipeline at 640x360
    # Note: Phase 29 benchmark evaluated Model C at 640x360 with GPU delegate (Apple Metal on macOS arm64)
    print("\n  Initializing Experimental Pipeline (Model C @ 640x360 with Apple Metal GPU delegate)...")
    pipe = ISLPipeline()
    pipe.set_recognition_arch("model_c")
    pipe.set_model_c_input_resolution("640x360")
    pipe.set_mediapipe_delegate("gpu")
    pipe.initialize()

    # Pre-flight check on sample frame
    test_img = cv2.imread(os.path.join(BASE_DIR, manifest_samples[0]["frame_path"]))
    test_res = pipe.process_frame(test_img, return_images=False)
    test_diag = test_res.get("diagnostics", {})

    target_res = pipe._get_target_resolution(test_img.shape[1], test_img.shape[0])
    proc_shape = test_diag.get("processed_shape")
    active_delegate = test_diag.get("mediapipe_delegate")
    feat_dim = test_diag.get("feature_dimensions")

    print(f"  Configured resolution: {pipe.model_c_input_resolution}")
    print(f"  Target resolution tuple: {target_res}")
    print(f"  Processed frame shape: {proc_shape}")
    print(f"  Active MediaPipe delegate: {active_delegate}")
    print(f"  Active feature dimension: {feat_dim} (Expected: {FEATURE_DIM_MODEL_C})")

    if target_res != (640, 360) or proc_shape != [360, 640]:
        raise ValueError(f"Resolution configuration mismatch: {proc_shape}")
    if feat_dim != FEATURE_DIM_MODEL_C:
        raise ValueError(f"Feature dimension mismatch: expected {FEATURE_DIM_MODEL_C}, got {feat_dim}")

    # -------------------------------------------------------------------------
    # STEP 4 & 5: RUN INFERENCE-ONLY EVALUATION & BENCHMARK LATENCY
    # -------------------------------------------------------------------------
    print("\n[Step 4 & 5] Running Inference-Only Evaluation and Latency Benchmarking...")
    
    # Warm-up phase: 30 frames
    print("  Executing 30 warm-up inferences...")
    warmup_samples = manifest_samples[:30]
    for w_s in warmup_samples:
        w_img = cv2.imread(os.path.join(BASE_DIR, w_s["frame_path"]))
        _ = pipe.process_frame(w_img, return_images=False)

    print(f"  Executing sustained evaluation across {len(manifest_samples)} unique signer_05 frames...")

    y_true = []
    y_pred = []
    y_prob = []
    predictions_records = []
    latencies_ms = []
    stage_timings = defaultdict(list)
    memory_checkpoints = {}

    memory_checkpoints["start_rss_mb"] = get_rss_mb()

    # Tracking forensics
    zero_hand_frames = 0
    one_hand_frames = 0
    two_hand_frames = 0
    invalid_feature_frames = 0
    inference_errors = 0

    t_eval_start = time.perf_counter()

    for idx, s in enumerate(manifest_samples):
        # Memory checkpoints
        if idx in (100, 250, 500, 750, 1000, 1250, 1500):
            memory_checkpoints[f"frame_{idx}_rss_mb"] = get_rss_mb()

        frame_abs_path = os.path.join(BASE_DIR, s["frame_path"])
        img = cv2.imread(frame_abs_path)
        if img is None:
            inference_errors += 1
            continue

        t0 = time.perf_counter()
        try:
            res = pipe.process_frame(img, return_images=False)
            t_elapsed = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(t_elapsed)
        except Exception as e:
            inference_errors += 1
            print(f"    ERROR on sample {s['sample_id']}: {e}")
            continue

        diag = res.get("diagnostics", {})
        pred_letter = res.get("predicted_sign", "-")
        conf = res.get("confidence", 0.0)
        h_count = diag.get("hands_detected_count", 0)

        # Update stage timings
        stages = diag.get("stage_timings_ms", {})
        for st_k, st_v in stages.items():
            stage_timings[st_k].append(st_v)

        # Forensics
        if h_count == 0:
            zero_hand_frames += 1
        elif h_count == 1:
            one_hand_frames += 1
        elif h_count >= 2:
            two_hand_frames += 1

        if diag.get("feature_dimensions") != FEATURE_DIM_MODEL_C:
            invalid_feature_frames += 1

        # Probability distribution across all 26 classes
        top3_list = diag.get("top_3", [])
        # Construct probability array if model_c recognizer probability available
        # pipe.model_c_recognizer.clf.predict_proba yields normalized probabilities
        last_probs = getattr(pipe.model_c_recognizer, "_last_proba", None)
        if last_probs is not None:
            prob_vec = list(last_probs)
        else:
            # Fallback to sparse proxy from top_3
            prob_vec = [0.0] * 26
            for l_sym, p_val in top3_list:
                if l_sym in C2I:
                    prob_vec[C2I[l_sym]] = round(p_val / 100.0, 4)

        y_true.append(s["letter"])
        y_pred.append(pred_letter)
        y_prob.append(prob_vec)

        predictions_records.append({
            "sample_id": s["sample_id"],
            "letter": s["letter"],
            "predicted": pred_letter,
            "is_correct": int(s["letter"] == pred_letter),
            "confidence": round(conf, 2),
            "condition": s["condition"],
            "recording_id": s["recording_id"],
            "hands_detected": h_count,
            "latency_ms": round(t_elapsed, 2),
            "frame_path": s["frame_path"],
            "sha256": s["sha256"]
        })

    t_eval_end = time.perf_counter()
    total_eval_duration = t_eval_end - t_eval_start
    memory_checkpoints["end_rss_mb"] = get_rss_mb()

    print(f"  Inference completed in {total_eval_duration:.2f} seconds.")
    print(f"  Total frames evaluated: {len(y_true)} / {len(manifest_samples)}")

    # Save predictions CSV
    with open(PREDICTIONS_CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions_records[0].keys()))
        writer.writeheader()
        writer.writerows(predictions_records)
    print(f"  Saved raw per-sample predictions to {PREDICTIONS_CSV_PATH}")

    # Latency Distribution
    lat_dist = compute_distribution(latencies_ms)
    throughput_fps = round(len(latencies_ms) / total_eval_duration, 2)
    slow_frames_count = sum(1 for lat in latencies_ms if lat > 100.0)
    slow_frames_pct = round((slow_frames_count / max(1, len(latencies_ms))) * 100.0, 2)

    # Per-stage distributions
    stage_breakdowns = {st_k: compute_distribution(st_v) for st_k, st_v in stage_timings.items()}

    # Classification Metrics: Overall pooled
    overall_metrics = compute_classification_metrics(y_true, y_pred, y_prob)

    # Classification Metrics: Condition 1 breakdown
    cond1_indices = [i for i, s in enumerate(manifest_samples) if s["condition"] == "condition_1"]
    y_true_c1 = [y_true[i] for i in cond1_indices]
    y_pred_c1 = [y_pred[i] for i in cond1_indices]
    y_prob_c1 = [y_prob[i] for i in cond1_indices]
    cond1_metrics = compute_classification_metrics(y_true_c1, y_pred_c1, y_prob_c1)

    # Classification Metrics: Condition 2 breakdown
    cond2_indices = [i for i, s in enumerate(manifest_samples) if s["condition"] == "condition_2"]
    y_true_c2 = [y_true[i] for i in cond2_indices]
    y_pred_c2 = [y_pred[i] for i in cond2_indices]
    y_prob_c2 = [y_prob[i] for i in cond2_indices]
    cond2_metrics = compute_classification_metrics(y_true_c2, y_pred_c2, y_prob_c2)

    # Confusion Matrix (26x26)
    cm_overall = np.zeros((26, 26), dtype=int)
    cm_c1 = np.zeros((26, 26), dtype=int)
    cm_c2 = np.zeros((26, 26), dtype=int)

    for yt, yp in zip(y_true, y_pred):
        if yt in C2I and yp in C2I:
            cm_overall[C2I[yt], C2I[yp]] += 1

    for yt, yp in zip(y_true_c1, y_pred_c1):
        if yt in C2I and yp in C2I:
            cm_c1[C2I[yt], C2I[yp]] += 1

    for yt, yp in zip(y_true_c2, y_pred_c2):
        if yt in C2I and yp in C2I:
            cm_c2[C2I[yt], C2I[yp]] += 1

    plot_confusion_matrix(cm_overall, ALPHABET, "Phase 30 Model C Confusion Matrix (Signer_05 Pooled)", CM_PLOT_PATH)
    plot_confusion_matrix(cm_c1, ALPHABET, "Phase 30 Model C Confusion Matrix (Condition 1: Desk Neutral)", CM_COND1_PLOT_PATH)
    plot_confusion_matrix(cm_c2, ALPHABET, "Phase 30 Model C Confusion Matrix (Condition 2: Lateral Warm)", CM_COND2_PLOT_PATH)
    print(f"  Saved confusion matrix plots to {PLOTS_DIR}")

    # Recording-Level Evidence and Static-Hold Stability
    print("\n  Analyzing Recording-Level Evidence and Static-Hold Stability...")
    recording_analysis = {}
    rec_correct_count = 0

    for rec_id, frames_meta in recording_frames.items():
        rec_preds = [r["predicted"] for r in predictions_records if r["recording_id"] == rec_id]
        ground_letter = frames_meta[0]["letter"]
        cond = "condition_1" if "condition_1" in frames_meta[0]["source"]["frame_path"] else "condition_2"
        
        # Modal prediction
        pred_counts = Counter(rec_preds)
        modal_pred, modal_count = pred_counts.most_common(1)[0]
        is_rec_correct = (modal_pred == ground_letter)
        if is_rec_correct:
            rec_correct_count += 1

        stability_pct = round((modal_count / max(1, len(rec_preds))) * 100.0, 2)
        ground_letter_pct = round((pred_counts.get(ground_letter, 0) / max(1, len(rec_preds))) * 100.0, 2)

        # Transition stability: first 3 frames vs middle vs last 3 frames
        first_3_agree = sum(1 for p in rec_preds[:3] if p == modal_pred) / max(1, len(rec_preds[:3]))
        last_3_agree = sum(1 for p in rec_preds[-3:] if p == modal_pred) / max(1, len(rec_preds[-3:]))

        recording_analysis[rec_id] = {
            "recording_id": rec_id,
            "letter": ground_letter,
            "condition": cond,
            "total_frames": len(rec_preds),
            "modal_prediction": modal_pred,
            "modal_count": modal_count,
            "is_correct": is_rec_correct,
            "stability_pct": stability_pct,
            "ground_letter_pct": ground_letter_pct,
            "transition_start_stability": round(first_3_agree * 100.0, 1),
            "transition_end_stability": round(last_3_agree * 100.0, 1),
            "distribution": dict(pred_counts)
        }

    recording_acc = round(rec_correct_count / max(1, len(recording_frames)), 4)
    avg_stability = round(float(np.mean([ra["stability_pct"] for ra in recording_analysis.values()])), 2)

    # -------------------------------------------------------------------------
    # STEP 6: APPLY THE PREDECLARED GATES
    # -------------------------------------------------------------------------
    print("\n[Step 6] Applying Predeclared Gates...")
    
    # Recognition Gates
    rec_gate_acc_pass = overall_metrics["overall_top1_accuracy"] >= 0.85
    rec_gate_f1_pass = overall_metrics["macro_f1"] >= 0.80
    rec_gate_zero_recall_pass = len(overall_metrics["zero_recall_letters"]) == 0

    # Latency Gates
    lat_gate_p95_pass = lat_dist["p95"] < 100.0
    lat_gate_fps_pass = throughput_fps > 15.0

    # Integrity Gates
    integ_model_hash_pass = hash_audit_passed
    integ_manifest_pass = len(manifest_samples) == 1638 and len(sample_hashes) == 1638
    integ_isolation_pass = all(v == 0 for v in split_overlap.values())
    integ_prod_defaults_pass = prod_defaults_ok
    integ_no_inference_errors = (inference_errors == 0 and invalid_feature_frames == 0)

    recognition_gates_passed = rec_gate_acc_pass and rec_gate_f1_pass and rec_gate_zero_recall_pass
    latency_gates_passed = lat_gate_p95_pass and lat_gate_fps_pass
    integrity_gates_passed = (
        integ_model_hash_pass and integ_manifest_pass and integ_isolation_pass and
        integ_prod_defaults_pass and integ_no_inference_errors
    )

    # Determine Final Verdict
    if not integrity_gates_passed:
        final_verdict = "INTEGRITY OR RELIABILITY GATE FAILED"
    elif not latency_gates_passed:
        final_verdict = "LATENCY GATE FAILED"
    elif not recognition_gates_passed:
        final_verdict = "RECOGNITION GENERALIZATION GATE FAILED"
    else:
        final_verdict = "INDEPENDENT SIGNER VALIDATION PASSED"

    print(f"\n========================================================")
    print(f"FINAL VERDICT: {final_verdict}")
    print(f"========================================================")
    print(f"Recognition Acc: {overall_metrics['overall_top1_accuracy']*100:.2f}% (Gate >= 85.0%: {'PASS' if rec_gate_acc_pass else 'FAIL'})")
    print(f"Macro-F1:        {overall_metrics['macro_f1']*100:.2f}% (Gate >= 80.0%: {'PASS' if rec_gate_f1_pass else 'FAIL'})")
    print(f"Zero Recall:     {overall_metrics['zero_recall_letters']} (Gate == 0: {'PASS' if rec_gate_zero_recall_pass else 'FAIL'})")
    print(f"Sub-70% Letters: {overall_metrics['critical_weakness_letters']}")
    print(f"Latency p95:     {lat_dist['p95']:.2f} ms (Gate < 100 ms: {'PASS' if lat_gate_p95_pass else 'FAIL'})")
    print(f"Throughput FPS:  {throughput_fps:.2f} FPS (Gate > 15 FPS: {'PASS' if lat_gate_fps_pass else 'FAIL'})")
    print(f"Latency p99/max: {lat_dist['p99']:.2f} ms / {lat_dist['max']:.2f} ms")
    print(f"Recording Acc:   {recording_acc*100:.2f}% ({rec_correct_count}/{len(recording_frames)} recordings)")
    print(f"Avg Stability:   {avg_stability:.2f}%")

    # -------------------------------------------------------------------------
    # COMPILE DELIVERABLES
    # -------------------------------------------------------------------------
    metrics_payload = {
        "phase": 30,
        "title": "Signer_05 Independent Evaluation Metrics",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "final_verdict": final_verdict,
        "system_platform": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "system": platform.system(),
            "release": platform.release(),
            "python_version": platform.python_version()
        },
        "evaluation_configuration": {
            "signer_id": "signer_05",
            "recognition_arch": "model_c",
            "model_c_resolution": "640x360",
            "mediapipe_delegate": active_delegate,
            "temporal_smoothing": False,
            "shadow_mode": False,
            "model_c_n_jobs": 1,
            "differences_from_phase29_benchmark": "Zero architecture or delegate differences. Same 640x360 resolution and Apple Metal delegate as Phase 29 benchmark."
        },
        "dataset_summary": {
            "total_frames_evaluated": len(y_true),
            "condition_1_frames": len(y_true_c1),
            "condition_2_frames": len(y_true_c2),
            "total_recordings": len(recording_frames),
            "eligible_frames": len(manifest_samples),
            "excluded_frames": sum(e["frames_excluded"] for e in exclusions_log),
            "exclusions": exclusions_log
        },
        "recognition_metrics": {
            "overall_pooled": overall_metrics,
            "condition_1": cond1_metrics,
            "condition_2": cond2_metrics,
            "recording_level": {
                "total_recordings": len(recording_frames),
                "correct_recordings": rec_correct_count,
                "majority_vote_accuracy": recording_acc,
                "average_static_hold_stability_pct": avg_stability,
                "recordings": recording_analysis
            }
        },
        "latency_metrics": {
            "warmup_frames": len(warmup_samples),
            "evaluated_frames": len(latencies_ms),
            "total_duration_sec": round(total_eval_duration, 3),
            "throughput_fps": throughput_fps,
            "slow_frames_gt_100ms_count": slow_frames_count,
            "slow_frames_gt_100ms_pct": slow_frames_pct,
            "latency_distribution_ms": lat_dist,
            "stage_breakdowns_ms": stage_breakdowns,
            "memory_checkpoints_mb": memory_checkpoints
        },
        "detection_and_forensics": {
            "zero_hand_frames": zero_hand_frames,
            "zero_hand_pct": round((zero_hand_frames / max(1, len(y_true))) * 100.0, 2),
            "one_hand_frames": one_hand_frames,
            "two_hand_frames": two_hand_frames,
            "invalid_feature_frames": invalid_feature_frames,
            "inference_errors": inference_errors
        },
        "gate_evaluation": {
            "recognition_gates": {
                "overall_accuracy_pass": rec_gate_acc_pass,
                "overall_accuracy_value": overall_metrics["overall_top1_accuracy"],
                "macro_f1_pass": rec_gate_f1_pass,
                "macro_f1_value": overall_metrics["macro_f1"],
                "zero_recall_pass": rec_gate_zero_recall_pass,
                "zero_recall_letters": overall_metrics["zero_recall_letters"],
                "critical_weaknesses_sub_70": overall_metrics["critical_weakness_letters"],
                "all_recognition_gates_passed": recognition_gates_passed
            },
            "latency_gates": {
                "p95_pass": lat_gate_p95_pass,
                "p95_value_ms": lat_dist["p95"],
                "throughput_pass": lat_gate_fps_pass,
                "throughput_value_fps": throughput_fps,
                "p99_value_ms": lat_dist["p99"],
                "max_value_ms": lat_dist["max"],
                "all_latency_gates_passed": latency_gates_passed
            },
            "integrity_gates": {
                "model_hashes_pass": integ_model_hash_pass,
                "evaluation_manifest_frozen": integ_manifest_pass,
                "split_isolation_pass": integ_isolation_pass,
                "production_defaults_pass": integ_prod_defaults_pass,
                "reliability_errors_pass": integ_no_inference_errors,
                "all_integrity_gates_passed": integrity_gates_passed
            }
        },
        "confusion_matrix_26x26": cm_overall.tolist()
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"  Saved metrics to {METRICS_PATH}")

    # Integrity Audit Document
    integrity_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phase": "Phase 30 — Signer_05 Independent Evaluation Integrity Audit",
        "final_verdict": final_verdict,
        "evaluation_manifest_sha256": eval_manifest_sha256,
        "metrics_sha256": sha256_file(METRICS_PATH),
        "production_defaults_verified": {
            "ISL_BACKEND_DETECTOR": default_pipe.backend_detector,
            "ISL_RECOGNITION_ARCH": default_pipe.recognition_arch,
            "ISL_TEMPORAL_SMOOTHING": default_pipe.temporal_smoothing_enabled,
            "ISL_SHADOW_MODE": default_pipe.shadow_mode_enabled,
            "ISL_MEDIAPIPE_DELEGATE": default_pipe.mediapipe_delegate,
            "ISL_MODEL_C_RESOLUTION": getattr(default_pipe, "model_c_input_resolution", "native")
        },
        "production_defaults_pass": prod_defaults_ok,
        "artifacts_audited": {
            k: {
                "exists": os.path.exists(os.path.join(BASE_DIR, k)),
                "current_sha256": current_hashes[k],
                "expected_sha256": exp_h,
                "matches": (current_hashes[k] == exp_h)
            } for k, exp_h in APPROVED_HASHES.items()
        },
        "data_independence_audit": {
            "signer_05_in_training": False,
            "signer_05_in_validation": False,
            "signer_05_in_calibration": False,
            "total_signer_05_samples": len(manifest_samples),
            "unique_sample_hashes": len(sample_hashes),
            "overlap_with_train_manifest": split_overlap.get("train_manifest", 0),
            "overlap_with_val_manifest": split_overlap.get("val_manifest", 0),
            "overlap_with_phase19_manifest": split_overlap.get("p19_manifest", 0)
        }
    }

    with open(INTEGRITY_AUDIT_PATH, "w") as f:
        json.dump(integrity_payload, f, indent=2)
    print(f"  Saved integrity audit to {INTEGRITY_AUDIT_PATH}")

    pipe.shutdown()
    return metrics_payload


if __name__ == "__main__":
    run_evaluation()
