#!/usr/bin/env python3
"""
Phase 20 Diagnostics & Metrics Reconciliation Script.
Recomputes all benchmark metrics from phase19_predictions_raw.csv,
compares against phase19_benchmark_results.json,
audits denominators, analyzes class confusions and failures,
investigates YOLO latency breakdown, and checks video evaluations.
"""

import os
import sys
import csv
import json
import time
import numpy as np
from collections import defaultdict, Counter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, balanced_accuracy_score

def load_predictions_csv(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def reconcile_all_metrics():
    print("=== 1. RECONCILING BENCHMARK METRICS FROM RAW PREDICTIONS ===")
    csv_path = "phase19_analysis/phase19_predictions_raw.csv"
    json_path = "phase19_analysis/phase19_benchmark_results.json"
    
    rows = load_predictions_csv(csv_path)
    print(f"Loaded {len(rows)} rows from {csv_path}")
    
    with open(json_path) as f:
        bench_json = json.load(f)
        
    y_true = np.array([r["ground_truth"] for r in rows])
    classes = sorted(list(set(y_true)))
    print(f"Ground truth classes ({len(classes)}): {classes}")
    
    # Check class support
    class_counts = Counter(y_true)
    print(f"Class support counts: min={min(class_counts.values())}, max={max(class_counts.values())}, total={sum(class_counts.values())}")
    
    model_col_map = {
        "1_Production_YOLO_Baseline": "pred_m1_yolo_prod",
        "2_MediaPipe_v1_CNN": "pred_m2_mediapipe_cnn",
        "3_Landmark_Only_Geometry_v1": "pred_m3_landmark_v1",
        "4_Landmark_Only_Geometry_v2": "pred_m4_geom_v2",
        "5_Late_Fusion_Baseline_alpha_0.25": "pred_m5_late_fuse_base",
        "6_Late_Fusion_Calibrated_alpha_0.75": "pred_m6_late_fuse_calib",
        "7_Hybrid_Concatenation_RF": "pred_m7_hybrid_rf"
    }
    
    reconciled = {}
    
    for m_name, col in model_col_map.items():
        y_pred = np.array([r[col] for r in rows])
        
        valid_mask = (y_pred != "NO_DETECTION") & (y_pred != "") & (y_pred != "None")
        n_valid = int(np.sum(valid_mask))
        n_fail = int(len(y_pred) - n_valid)
        
        # End to end vs conditional
        acc_cond = float(accuracy_score(y_true[valid_mask], y_pred[valid_mask])) * 100.0 if n_valid > 0 else 0.0
        acc_e2e = float(np.sum(y_true == y_pred)) / float(len(y_true)) * 100.0
        
        prec, rec, f1, sup = precision_recall_fscore_support(
            y_true[valid_mask], y_pred[valid_mask], labels=classes, zero_division=0
        )
        macro_f1 = float(np.mean(f1)) * 100.0
        macro_prec = float(np.mean(prec)) * 100.0
        macro_rec = float(np.mean(rec)) * 100.0
        bal_acc = float(balanced_accuracy_score(y_true[valid_mask], y_pred[valid_mask])) * 100.0
        
        json_entry = bench_json["aggregate_evaluation"].get(m_name, {})
        json_t1 = json_entry.get("conditional_top1_accuracy")
        json_f1 = json_entry.get("macro_f1")
        json_bal = json_entry.get("balanced_accuracy")
        json_fails = json_entry.get("detection_failures")
        
        reconciled[m_name] = {
            "recomputed_top1": round(acc_cond, 2),
            "json_top1": json_t1,
            "top1_diff": round(acc_cond - (json_t1 or 0), 4),
            "recomputed_macro_f1": round(macro_f1, 2),
            "json_macro_f1": json_f1,
            "macro_f1_diff": round(macro_f1 - (json_f1 or 0), 4),
            "recomputed_balanced_acc": round(bal_acc, 2),
            "json_balanced_acc": json_bal,
            "valid_samples": n_valid,
            "failed_samples": n_fail,
            "json_failed_samples": json_fails,
            "per_class": {
                c: {
                    "precision": round(float(p) * 100.0, 2),
                    "recall": round(float(r) * 100.0, 2),
                    "f1": round(float(f) * 100.0, 2),
                    "support": int(s)
                }
                for c, p, r, f, s in zip(classes, prec, rec, f1, sup)
            }
        }
        
        print(f"Model: {m_name}")
        print(f"  Top-1: Recomp={acc_cond:.2f}%, JSON={json_t1}%, Diff={acc_cond - (json_t1 or 0):.4f}%")
        print(f"  Macro-F1: Recomp={macro_f1:.2f}%, JSON={json_f1}%, Diff={macro_f1 - (json_f1 or 0):.4f}%")
        print(f"  Balanced Acc: Recomp={bal_acc:.2f}%, JSON={json_bal}%, Diff={bal_acc - (json_bal or 0):.4f}%")
        print(f"  Valid / Failures: {n_valid} / {n_fail} (JSON fails: {json_fails})")
        
    return rows, reconciled, bench_json

def audit_acceptance_gate_denominators(rows, bench_json):
    print("\n=== 2. AUDIT OF ACCEPTANCE GATE DENOMINATORS (N, X, L) ===")
    
    classes_of_interest = ["N", "X", "L"]
    model_col_map = {
        "1_Production_YOLO_Baseline": "pred_m1_yolo_prod",
        "2_MediaPipe_v1_CNN": "pred_m2_mediapipe_cnn",
        "3_Landmark_Only_Geometry_v1": "pred_m3_landmark_v1",
        "4_Landmark_Only_Geometry_v2": "pred_m4_geom_v2",
        "5_Late_Fusion_Baseline_alpha_0.25": "pred_m5_late_fuse_base",
        "6_Late_Fusion_Calibrated_alpha_0.75": "pred_m6_late_fuse_calib",
        "7_Hybrid_Concatenation_RF": "pred_m7_hybrid_rf"
    }
    
    gate_details = {}
    
    for letter in classes_of_interest:
        sub_rows = [r for r in rows if r["ground_truth"] == letter]
        total_support = len(sub_rows)
        signer_counts = Counter(r["signer_id"] for r in sub_rows)
        cond_counts = Counter(r["condition_id"] for r in sub_rows)
        
        print(f"\n--- Class '{letter}' Audit ---")
        print(f"Total ground-truth samples (Denominator): {total_support}")
        print(f"By Signer: {dict(signer_counts)}")
        print(f"By Condition: {dict(cond_counts)}")
        
        gate_details[letter] = {
            "denominator": total_support,
            "signer_breakdown": dict(signer_counts),
            "condition_breakdown": dict(cond_counts),
            "model_recalls": {}
        }
        
        for m_name, col in model_col_map.items():
            correct = sum(1 for r in sub_rows if r[col] == letter)
            recall = (correct / total_support) * 100.0 if total_support > 0 else 0.0
            
            # Top confused predictions
            preds = [r[col] for r in sub_rows]
            confused_counts = Counter(preds)
            
            gate_details[letter]["model_recalls"][m_name] = {
                "correct": correct,
                "recall": round(recall, 2),
                "confusion_distribution": confused_counts.most_common(5)
            }
            print(f"  {m_name[:32]:<32} | Recall: {recall:>5.2f}% ({correct}/{total_support}) | Top preds: {confused_counts.most_common(3)}")
            
    return gate_details

def analyze_reconcile_video_vs_static_n(rows, bench_json):
    print("\n=== 3. RECONCILING STATIC N RECALL (0%) vs CONTINUOUS VIDEO N ACCURACY (75.8%) ===")
    
    # 1. Static frames of N
    n_rows = [r for r in rows if r["ground_truth"] == "N"]
    print(f"Total static N frames extracted across all 4 recordings: {len(n_rows)}")
    
    # Check recording breakdown for N
    by_rec = defaultdict(list)
    for r in n_rows:
        by_rec[r["recording_id"]].append(r)
        
    for rec_id, rec_list in by_rec.items():
        signer = rec_list[0]["signer_id"]
        cond = rec_list[0]["condition_id"]
        m4_preds = Counter(r["pred_m4_geom_v2"] for r in rec_list)
        m6_preds = Counter(r["pred_m6_late_fuse_calib"] for r in rec_list)
        m3_preds = Counter(r["pred_m3_landmark_v1"] for r in rec_list)
        m2_preds = Counter(r["pred_m2_mediapipe_cnn"] for r in rec_list)
        print(f"\nRecording: {rec_id} (Signer: {signer}, Cond: {cond}, Extracted Frames: {len(rec_list)})")
        print(f"  M4 (Geometry v2) predictions: {dict(m4_preds)}")
        print(f"  M3 (Landmark v1) predictions: {dict(m3_preds)}")
        print(f"  M6 (Late Fuse Calib) predictions: {dict(m6_preds)}")
        print(f"  M2 (MP CNN) predictions: {dict(m2_preds)}")
        
    # Check the video evaluation entry in bench_json
    vid_eval = bench_json.get("video_temporal_evaluation", {})
    print("\nContinuous Video Evaluation entries for N:")
    for k, v in vid_eval.items():
        if "_N" in k:
            print(f"Key: {k}")
            print(f"  Path: {v.get('video_path')}")
            print(f"  Total video frames evaluated: {v.get('total_frames')}")
            print(f"  M6 Raw Acc: {v.get('m6_calib_raw', {}).get('frame_accuracy')}% (Majority: {v.get('m6_calib_raw', {}).get('majority_prediction')})")
            print(f"  M6 Smoothed Acc: {v.get('m6_calib_smoothed', {}).get('frame_accuracy')}% (Majority: {v.get('m6_calib_smoothed', {}).get('majority_prediction')})")
            print(f"  M4 Raw Acc: {v.get('m4_geom_raw', {}).get('frame_accuracy')}% (Majority: {v.get('m4_geom_raw', {}).get('majority_prediction')})")

def analyze_class_failures_and_geometry_vs_landmark(rows, bench_json):
    print("\n=== 4. CLASS FAILURE ANALYSIS & M3 (Landmark v1) vs M4 (Geometry v2) ===")
    
    classes = sorted(list(set(r["ground_truth"] for r in rows)))
    
    # Check per-class recall for M3 and M4
    m3_recalls = {}
    m4_recalls = {}
    diffs = {}
    
    m3_info = bench_json["aggregate_evaluation"]["3_Landmark_Only_Geometry_v1"]["per_class"]
    m4_info = bench_json["aggregate_evaluation"]["4_Landmark_Only_Geometry_v2"]["per_class"]
    
    zero_recall_m4 = []
    zero_recall_m3 = []
    
    for c in classes:
        r3 = m3_info[c]["recall"]
        r4 = m4_info[c]["recall"]
        m3_recalls[c] = r3
        m4_recalls[c] = r4
        diffs[c] = round(r3 - r4, 2)
        if r4 == 0.0:
            zero_recall_m4.append(c)
        if r3 == 0.0:
            zero_recall_m3.append(c)
            
    print(f"Zero recall classes in M4 (Geometry v2): {zero_recall_m4} (Count: {len(zero_recall_m4)})")
    print(f"Zero recall classes in M3 (Landmark v1): {zero_recall_m3} (Count: {len(zero_recall_m3)})")
    
    print("\nClasses where M3 outperforms M4 by > 10%:")
    for c, d in sorted(diffs.items(), key=lambda x: x[1], reverse=True):
        if d > 10.0:
            print(f"  Class {c}: M3 = {m3_recalls[c]}%, M4 = {m4_recalls[c]}% (Delta: +{d}%)")
            
    print("\nClasses where M4 outperforms M3 by > 10%:")
    for c, d in sorted(diffs.items(), key=lambda x: x[1]):
        if d < -10.0:
            print(f"  Class {c}: M4 = {m4_recalls[c]}%, M3 = {m3_recalls[c]}% (Delta: +{-d}%)")

    # Detailed confusion analysis for D, E, F, I, N, Q, R, X
    print("\nConfusion Distribution for Zero-Recall Classes in M4:")
    target_classes = ["D", "E", "F", "I", "N", "Q", "R", "X", "M"]
    for tc in target_classes:
        sub = [r for r in rows if r["ground_truth"] == tc]
        m4_preds = Counter(r["pred_m4_geom_v2"] for r in sub)
        m3_preds = Counter(r["pred_m3_landmark_v1"] for r in sub)
        print(f"\nClass '{tc}' (N={len(sub)}):")
        print(f"  M4 (Geom v2) Top Confusions: {m4_preds.most_common(4)}")
        print(f"  M3 (Landmark v1) Top Confusions: {m3_preds.most_common(4)}")

if __name__ == "__main__":
    rows, reconciled, bench_json = reconcile_all_metrics()
    audit_acceptance_gate_denominators(rows, bench_json)
    analyze_reconcile_video_vs_static_n(rows, bench_json)
    analyze_class_failures_and_geometry_vs_landmark(rows, bench_json)
