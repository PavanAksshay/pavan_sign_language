#!/usr/bin/env python3
"""
Phase 15 — Multi-Model Benchmark across Independent and Provisional Partitions

Compares:
1. Production Baseline (YOLO on video / Production SqueezeNet on crops)
2. MediaPipe v1 CNN
3. Landmark-Only Random Forest
4. Late-Fusion Ensemble (alpha=0.25)
5. Hybrid Concatenation Random Forest

Evaluates across:
- Independent Test Set A–L (May 14, 381 crops)
- Provisional Test Set M–Z (May 16, 720 crops)
- Preserved Continuous Webcam Videos (Frontal C, Second C, Standard L, Webcam A)

Computes:
- Attempted vs successfully processed samples
- Detection and landmark failure counts
- Top-1 and Top-3 accuracy
- Macro-F1 and Balanced accuracy
- Per-class Precision, Recall, F1, Support
- 26x26 A–Z Confusion Matrices
- Priority confusion pair audit (N vs M, X vs V, L vs Y/U, E vs J, O vs Y, A vs G/I)
- Video temporal stability (Flip rate, Max run, Majority vote, Recovery time, Detection-loss)
- End-to-end CPU latencies
"""

import os
import sys
import json
import time
import cv2
import numpy as np
import joblib
import tf_keras
from collections import Counter, defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from pipeline import ISLPipeline
from temporal_smoother import TemporalSmoother
from landmark_feature_extractor import (
    extract_landmarks_from_manifest_item,
    extract_landmarks_from_mediapipe_result,
    TOTAL_FEATURE_DIM
)

SPLITS_DIR = os.path.join(BASE_DIR, "models", "mediapipe_squeezenet_v1", "splits")
CNN_MODEL_DIR = os.path.join(BASE_DIR, "models", "mediapipe_squeezenet_v1")
CNN_SAVED_MODEL = os.path.join(CNN_MODEL_DIR, "saved_model")
PROD_SAVED_MODEL = os.path.join(BASE_DIR, "App", "final_model_26")
HYBRID_MODEL_DIR = os.path.join(BASE_DIR, "models", "landmark_hybrid_v1")
OUT_DIR = os.path.join(BASE_DIR, "phase15_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

LETTERS = [chr(ord('A') + i) for i in range(26)]
C2I = {ltr: idx for idx, ltr in enumerate(LETTERS)}


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def get_dir_size_mb(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return round(total / (1024 * 1024), 2)


def compute_comprehensive_metrics(y_true, y_probs, attempted_count, failed_count, latencies=None):
    num_classes = 26
    y_pred = np.argmax(y_probs, axis=1)
    top3_indices = [np.argsort(p)[-3:][::-1].tolist() for p in y_probs]

    successful_count = len(y_true)
    top1_correct = (y_true == y_pred)
    cond_top1_acc = float(np.mean(top1_correct)) * 100.0 if successful_count > 0 else 0.0
    end_to_end_top1_acc = (float(np.sum(top1_correct)) / attempted_count * 100.0) if attempted_count > 0 else 0.0

    top3_correct = [y_true[i] in top3_indices[i] for i in range(successful_count)]
    cond_top3_acc = float(np.mean(top3_correct)) * 100.0 if successful_count > 0 else 0.0
    end_to_end_top3_acc = (float(np.sum(top3_correct)) / attempted_count * 100.0) if attempted_count > 0 else 0.0

    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        confusion[t, p] += 1

    per_class = {}
    recalls = []
    precisions = []
    f1s = []

    for c_idx in range(num_classes):
        c_name = LETTERS[c_idx]
        c_total = int(np.sum(y_true == c_idx))
        c_correct = int(np.sum((y_true == c_idx) & (y_pred == c_idx)))

        tp = c_correct
        fp = int(np.sum((y_true != c_idx) & (y_pred == c_idx)))
        fn = int(np.sum((y_true == c_idx) & (y_pred != c_idx)))

        prec = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        if c_total > 0:
            recalls.append(rec)
            precisions.append(prec)
            f1s.append(f1)

        per_class[c_name] = {
            "class_index": c_idx,
            "total_samples": c_total,
            "correct_top1": c_correct,
            "top1_accuracy": round((c_correct / c_total * 100.0) if c_total > 0 else 0.0, 2),
            "precision": round(prec, 2),
            "recall": round(rec, 2),
            "f1_score": round(f1, 2)
        }

    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    balanced_acc = float(np.mean(recalls)) if recalls else 0.0

    return {
        "attempted_samples": attempted_count,
        "processed_samples": successful_count,
        "failure_count": failed_count,
        "conditional_top1_accuracy": round(cond_top1_acc, 2),
        "conditional_top3_accuracy": round(cond_top3_acc, 2),
        "end_to_end_top1_accuracy": round(end_to_end_top1_acc, 2),
        "end_to_end_top3_accuracy": round(end_to_end_top3_acc, 2),
        "macro_f1": round(macro_f1, 2),
        "balanced_accuracy": round(balanced_acc, 2),
        "mean_latency_ms": round(float(np.mean(latencies)), 2) if latencies else None,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist()
    }


def audit_priority_confusion_pairs(confusion_matrix):
    """
    Examines specific known confusion pairs across A-Z:
    N vs M, X vs V, L vs Y/U, E vs J, O vs Y, A vs G/I
    """
    pairs = [
        ("N", "M"),
        ("M", "N"),
        ("X", "V"),
        ("V", "X"),
        ("L", "Y"),
        ("L", "U"),
        ("E", "J"),
        ("J", "E"),
        ("O", "Y"),
        ("A", "G"),
        ("A", "I"),
    ]
    results = {}
    for true_c, pred_c in pairs:
        t_idx = C2I[true_c]
        p_idx = C2I[pred_c]
        count = int(confusion_matrix[t_idx][p_idx])
        total = int(sum(confusion_matrix[t_idx]))
        rate = round((count / total * 100.0) if total > 0 else 0.0, 2)
        results[f"{true_c}_confused_as_{pred_c}"] = {
            "true_class": true_c,
            "predicted_class": pred_c,
            "error_count": count,
            "total_true_samples": total,
            "confusion_rate_pct": rate
        }
    return results


def evaluate_static_splits():
    print("\n--- Loading Models for Static Evaluation ---")
    t0 = time.time()
    # 1. Base SqueezeNet model for feature extraction
    base_cnn = tf_keras.models.load_model(CNN_SAVED_MODEL, compile=False)
    gap_512 = tf_keras.layers.GlobalAveragePooling2D()(base_cnn.get_layer("concatenate_7").output)
    feature_model = tf_keras.Model(
        inputs=base_cnn.input,
        outputs=[gap_512, base_cnn.output]
    )

    # 2. Production SqueezeNet
    prod_cnn = tf_keras.models.load_model(PROD_SAVED_MODEL, compile=False)

    # 3. Landmark RF & Scaler
    rf_lm = joblib.load(os.path.join(HYBRID_MODEL_DIR, "landmark_rf_model.joblib"))
    scaler_lm = joblib.load(os.path.join(HYBRID_MODEL_DIR, "landmark_scaler.joblib"))

    # 4. Hybrid Concatenation RF & Scalers
    rf_hyb = joblib.load(os.path.join(HYBRID_MODEL_DIR, "hybrid_concatenation_model.joblib"))
    scaler_512 = joblib.load(os.path.join(HYBRID_MODEL_DIR, "cnn512_scaler.joblib"))

    print(f"Loaded all models in {time.time() - t0:.2f}s")

    split_manifests = [
        ("Independent Test A-L (May 14)", "test_al_manifest.json", "historical_signer_01", "independent_session"),
        ("Provisional Test M-Z (May 16)", "test_mz_manifest.json", "historical_signer_01", "provisional_session_leakage")
    ]

    static_results = {}

    for split_label, manifest_name, signer_id, eval_type in split_manifests:
        mpath = os.path.join(SPLITS_DIR, manifest_name)
        print(f"\nEvaluating on {split_label} ({mpath})...")
        with open(mpath, "r") as f:
            manifest = json.load(f)

        attempted = len(manifest)
        X_lm = []
        imgs = []
        y_true = []
        failed = 0

        for item in manifest:
            lm_vec = extract_landmarks_from_manifest_item(item)
            cpath = item.get("crop_path")
            if cpath and not os.path.isabs(cpath):
                img_path = os.path.join(BASE_DIR, cpath)
            elif cpath and os.path.exists(cpath):
                img_path = cpath
            else:
                img_path = item.get("file_path", "")
                if not os.path.isabs(img_path) or not os.path.exists(img_path):
                    img_path = os.path.join(BASE_DIR, item.get("relative_path", ""))

            if not os.path.exists(img_path):
                failed += 1
                continue

            img = cv2.imread(img_path)
            if img is None:
                failed += 1
                continue

            if img.shape[:2] != (224, 224):
                img = cv2.resize(img, (224, 224))

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            imgs.append(rgb)
            X_lm.append(lm_vec)
            y_true.append(C2I[item["class_label"]])

        X_lm = np.array(X_lm, dtype=np.float32)
        y_true = np.array(y_true, dtype=np.int32)
        X_imgs = np.array(imgs, dtype=np.float32)
        processed = len(y_true)

        # 1. Feature extraction from MediaPipe SqueezeNet v1
        t_start = time.time()
        f512_raw, probs_mp_cnn = feature_model.predict(X_imgs, batch_size=32, verbose=0)
        t_cnn = (time.time() - t_start) / processed * 1000

        # 2. Production SqueezeNet predictions
        t_start = time.time()
        probs_prod_cnn = prod_cnn.predict(X_imgs, batch_size=32, verbose=0)
        t_prod = (time.time() - t_start) / processed * 1000

        # 3. Landmark-Only RF
        t_start = time.time()
        X_lm_scaled = scaler_lm.transform(X_lm)
        probs_lm = rf_lm.predict_proba(X_lm_scaled)
        t_lm = (time.time() - t_start) / processed * 1000

        # 4. Late-Fusion Ensemble (alpha=0.25)
        probs_late_fusion = 0.25 * probs_lm + 0.75 * probs_mp_cnn

        # 5. Hybrid Concatenation RF
        t_start = time.time()
        f512_scaled = scaler_512.transform(f512_raw)
        X_hyb = np.hstack([f512_scaled, X_lm_scaled])
        probs_hyb = rf_hyb.predict_proba(X_hyb)
        t_hyb = (time.time() - t_start) / processed * 1000

        models_eval = {
            "Production Baseline SqueezeNet": compute_comprehensive_metrics(y_true, probs_prod_cnn, attempted, failed, [t_prod]*processed),
            "MediaPipe v1 CNN": compute_comprehensive_metrics(y_true, probs_mp_cnn, attempted, failed, [t_cnn]*processed),
            "Landmark-Only RF": compute_comprehensive_metrics(y_true, probs_lm, attempted, failed, [t_lm]*processed),
            "Late-Fusion Ensemble (alpha=0.25)": compute_comprehensive_metrics(y_true, probs_late_fusion, attempted, failed, [t_cnn + t_lm]*processed),
            "Hybrid Concatenation RF": compute_comprehensive_metrics(y_true, probs_hyb, attempted, failed, [t_cnn + t_hyb]*processed),
        }

        # Audit confusion pairs for late fusion
        lf_confusion = models_eval["Late-Fusion Ensemble (alpha=0.25)"]["confusion_matrix"]
        confusion_audit = audit_priority_confusion_pairs(lf_confusion)

        static_results[split_label] = {
            "manifest_name": manifest_name,
            "signer_id": signer_id,
            "evaluation_type": eval_type,
            "models": models_eval,
            "priority_confusion_audit": confusion_audit
        }

        print(f"  Summary for {split_label}:")
        for mname, mres in models_eval.items():
            print(f"    - {mname:35s} | Top-1: {mres['conditional_top1_accuracy']:5.2f}% | Top-3: {mres['conditional_top3_accuracy']:5.2f}% | Macro-F1: {mres['macro_f1']:5.2f}%")

    return static_results


def evaluate_continuous_webcam_videos():
    print("\n--- Evaluating Models on Continuous Webcam Videos ---")
    videos = [
        {"name": "Frontal C", "path": "recording_C_20261014_164912.mp4", "gt": "C"},
        {"name": "Second C", "path": "recording_C_20261014_165234.mp4", "gt": "C"},
        {"name": "Standard L", "path": "recording_L_20261014_165754.mp4", "gt": "L"},
        {"name": "Webcam A", "path": "recording_A_20261014_165529.mp4", "gt": "A"}
    ]

    # Initialize pipelines
    yolo_pipe = ISLPipeline()
    yolo_pipe.backend_detector = "yolo"
    yolo_pipe.recognition_arch = "standard"
    yolo_pipe.initialize()

    mp_cnn_pipe = ISLPipeline()
    mp_cnn_pipe.backend_detector = "mediapipe"
    mp_cnn_pipe.recognition_arch = "standard"
    mp_cnn_pipe.initialize()

    lf_raw_pipe = ISLPipeline()
    lf_raw_pipe.backend_detector = "mediapipe"
    lf_raw_pipe.recognition_arch = "late_fusion"
    lf_raw_pipe.temporal_smoothing_enabled = False
    lf_raw_pipe.initialize()

    lf_smooth_pipe = ISLPipeline()
    lf_smooth_pipe.backend_detector = "mediapipe"
    lf_smooth_pipe.recognition_arch = "late_fusion"
    lf_smooth_pipe.temporal_smoothing_enabled = True
    lf_smooth_pipe.initialize()

    video_results = {}

    for v in videos:
        vname = v["name"]
        vpath = os.path.join(BASE_DIR, v["path"])
        gt = v["gt"]

        if not os.path.exists(vpath):
            print(f"Skipping {vname}: {vpath} not found")
            continue

        cap = cv2.VideoCapture(vpath)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        total_frames = len(frames)
        print(f"\nProcessing Video {vname} (GT: {gt}, {total_frames} frames)...")

        model_preds = {
            "YOLO Baseline": [],
            "MediaPipe v1 CNN": [],
            "Landmark-Only RF": [],
            "Late-Fusion Raw": [],
            "Late-Fusion + Smoothed": []
        }
        model_latencies = {
            "YOLO Baseline": [],
            "MediaPipe v1 CNN": [],
            "Landmark-Only RF": [],
            "Late-Fusion Raw": [],
            "Late-Fusion + Smoothed": []
        }
        detection_failures = {
            "YOLO Baseline": 0,
            "MediaPipe v1 CNN": 0,
            "Landmark-Only RF": 0,
            "Late-Fusion Raw": 0,
            "Late-Fusion + Smoothed": 0
        }

        lf_smooth_pipe.temporal_smoother.reset()

        for idx, frame in enumerate(frames):
            # 1. YOLO
            t0 = time.time()
            res_yolo = yolo_pipe.predict(frame, return_images=False)
            t_yolo = (time.time() - t0) * 1000
            model_preds["YOLO Baseline"].append(res_yolo["predicted_sign"])
            model_latencies["YOLO Baseline"].append(t_yolo)
            if not res_yolo["hand_detected"]:
                detection_failures["YOLO Baseline"] += 1

            # 2. CNN v1
            t0 = time.time()
            res_cnn = mp_cnn_pipe.predict(frame, return_images=False)
            t_cnn = (time.time() - t0) * 1000
            model_preds["MediaPipe v1 CNN"].append(res_cnn["predicted_sign"])
            model_latencies["MediaPipe v1 CNN"].append(t_cnn)
            if not res_cnn["hand_detected"]:
                detection_failures["MediaPipe v1 CNN"] += 1

            # 3. Late-Fusion Raw & Landmark-Only
            t0 = time.time()
            res_lf = lf_raw_pipe.predict(frame, return_images=False)
            t_lf = (time.time() - t0) * 1000
            model_preds["Late-Fusion Raw"].append(res_lf["predicted_sign"])
            model_latencies["Late-Fusion Raw"].append(t_lf)

            lm_sign = res_lf["diagnostics"].get("landmark_top1", "-")
            model_preds["Landmark-Only RF"].append(lm_sign)
            model_latencies["Landmark-Only RF"].append(t_lf)

            if not res_lf["hand_detected"]:
                detection_failures["Late-Fusion Raw"] += 1
                detection_failures["Landmark-Only RF"] += 1

            # 4. Late-Fusion + Smoothed
            t0 = time.time()
            res_smooth = lf_smooth_pipe.predict(frame, return_images=False, session_id=f"vid_{vname}")
            t_smooth = (time.time() - t0) * 1000
            model_preds["Late-Fusion + Smoothed"].append(res_smooth["predicted_sign"])
            model_latencies["Late-Fusion + Smoothed"].append(t_smooth)
            if not res_smooth["hand_detected"]:
                detection_failures["Late-Fusion + Smoothed"] += 1

        # Compute video-level metrics
        video_results[vname] = {
            "ground_truth": gt,
            "total_frames": total_frames,
            "signer_id": "webcam_signer_01",
            "condition": "desktop_webcam_live",
            "models": {}
        }

        for mname, preds in model_preds.items():
            correct_frames = sum(1 for p in preds if p == gt)
            frame_acc = round((correct_frames / total_frames) * 100, 2)
            flips = sum(1 for i in range(1, len(preds)) if preds[i] != preds[i - 1])
            flip_rate = round((flips / max(1, len(preds) - 1)) * 100, 2)

            # Max stable run of correct predictions
            max_run = 0
            cur_run = 0
            for p in preds:
                if p == gt:
                    cur_run += 1
                    max_run = max(max_run, cur_run)
                else:
                    cur_run = 0

            # Recovery time after wrong prediction (average frames to return to correct GT)
            recovery_times = []
            in_error = False
            error_len = 0
            for p in preds:
                if p != gt:
                    in_error = True
                    error_len += 1
                else:
                    if in_error:
                        recovery_times.append(error_len)
                        in_error = False
                        error_len = 0
            avg_recovery_frames = round(float(np.mean(recovery_times)), 1) if recovery_times else None

            # Majority class
            counts = Counter(preds)
            majority_class, majority_count = counts.most_common(1)[0]
            rec_acc = 100.0 if majority_class == gt else 0.0

            mean_lat = round(float(np.mean(model_latencies[mname])), 1)

            video_results[vname]["models"][mname] = {
                "frame_accuracy": frame_acc,
                "majority_class": majority_class,
                "majority_count": majority_count,
                "majority_vote_accuracy": rec_acc,
                "flip_rate": flip_rate,
                "max_stable_run": max_run,
                "avg_recovery_frames": avg_recovery_frames,
                "detection_failures": detection_failures[mname],
                "mean_latency_ms": mean_lat,
                "distribution": counts.most_common(5)
            }

            print(f"  {mname:25s} | FrameAcc: {frame_acc:5.1f}% | Maj: {majority_class} ({rec_acc:3.0f}%) | FlipRate: {flip_rate:4.1f}% | MaxRun: {max_run:3d} | Lat: {mean_lat:4.1f}ms")

    return video_results


def main():
    print("=" * 80)
    print("PHASE 15 — COMPREHENSIVE MULTI-MODEL BENCHMARK & EVALUATION")
    print("=" * 80)

    static_eval = evaluate_static_splits()
    video_eval = evaluate_continuous_webcam_videos()

    # Measure disk footprint
    footprint = {
        "YOLOv3_cross_hands": {
            "config_bytes": os.path.getsize(os.path.join(BASE_DIR, "App", "yolo_models", "cross-hands.cfg")),
            "weights_bytes": os.path.getsize(os.path.join(BASE_DIR, "App", "yolo_models", "cross-hands.weights")),
            "total_mb": round((os.path.getsize(os.path.join(BASE_DIR, "App", "yolo_models", "cross-hands.cfg")) + os.path.getsize(os.path.join(BASE_DIR, "App", "yolo_models", "cross-hands.weights"))) / (1024*1024), 2)
        },
        "MediaPipe_HandLandmarker_Task": {
            "total_mb": round(os.path.getsize(os.path.join(BASE_DIR, "App", "include50_model", "hand_landmarker.task")) / (1024*1024), 2)
        },
        "MediaPipe_SqueezeNet_v1": {
            "total_mb": get_dir_size_mb(CNN_SAVED_MODEL)
        },
        "Production_SqueezeNet": {
            "total_mb": get_dir_size_mb(PROD_SAVED_MODEL)
        },
        "Landmark_Random_Forest": {
            "total_mb": round(os.path.getsize(os.path.join(HYBRID_MODEL_DIR, "landmark_rf_model.joblib")) / (1024*1024), 2)
        },
        "Hybrid_Concatenation_Random_Forest": {
            "total_mb": round(os.path.getsize(os.path.join(HYBRID_MODEL_DIR, "hybrid_concatenation_model.joblib")) / (1024*1024), 2)
        }
    }

    full_evaluation = {
        "timestamp": "2026-10-15T19:30:00Z",
        "phase": "Phase 15",
        "model_disk_footprint": footprint,
        "static_splits_evaluation": static_eval,
        "continuous_webcam_video_evaluation": video_eval
    }

    out_file = os.path.join(OUT_DIR, "existing_models_evaluation.json")
    with open(out_file, "w") as f:
        json.dump(full_evaluation, f, indent=2, cls=NpEncoder)

    print(f"\n[Phase 15] Saved complete evaluation to {out_file}")


if __name__ == "__main__":
    main()
