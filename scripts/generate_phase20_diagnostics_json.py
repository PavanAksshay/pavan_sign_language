#!/usr/bin/env python3
"""
Generate phase20_analysis/phase20_diagnostics.json
Compiles all metric reconciliation, gate audit, latency breakdown,
hand detection forensics, and experimental design specifications.
"""

import os
import sys
import csv
import json
import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, balanced_accuracy_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def generate_diagnostics():
    print("[Phase 20] Generating comprehensive diagnostics JSON...")
    
    # 1. Load Phase 19 raw predictions and JSON
    csv_path = os.path.join(BASE_DIR, "phase19_analysis", "phase19_predictions_raw.csv")
    json_path = os.path.join(BASE_DIR, "phase19_analysis", "phase19_benchmark_results.json")
    train_manifest_path = os.path.join(BASE_DIR, "models", "mediapipe_squeezenet_v1", "splits", "train_manifest.json")
    test_manifest_path = os.path.join(BASE_DIR, "data", "collection_manifests", "test_multisigner_manifest.json")
    
    with open(json_path) as f:
        bench_json = json.load(f)
        
    raw_rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            raw_rows.append(r)
            
    with open(train_manifest_path) as f:
        train_data = json.load(f)
        
    with open(test_manifest_path) as f:
        test_data = json.load(f)
        
    y_true = np.array([r["ground_truth"] for r in raw_rows])
    classes = sorted(list(set(y_true)))
    
    model_col_map = {
        "1_Production_YOLO_Baseline": "pred_m1_yolo_prod",
        "2_MediaPipe_v1_CNN": "pred_m2_mediapipe_cnn",
        "3_Landmark_Only_Geometry_v1": "pred_m3_landmark_v1",
        "4_Landmark_Only_Geometry_v2": "pred_m4_geom_v2",
        "5_Late_Fusion_Baseline_alpha_0.25": "pred_m5_late_fuse_base",
        "6_Late_Fusion_Calibrated_alpha_0.75": "pred_m6_late_fuse_calib",
        "7_Hybrid_Concatenation_RF": "pred_m7_hybrid_rf"
    }
    
    # 1. Reconcile Metrics
    reconciled_models = {}
    for m_name, col in model_col_map.items():
        y_pred = np.array([r[col] for r in raw_rows])
        valid_mask = (y_pred != "NO_DETECTION") & (y_pred != "") & (y_pred != "None")
        n_valid = int(np.sum(valid_mask))
        n_fail = int(len(y_pred) - n_valid)
        
        acc_cond = float(accuracy_score(y_true[valid_mask], y_pred[valid_mask])) * 100.0 if n_valid > 0 else 0.0
        acc_e2e = float(np.sum(y_true == y_pred)) / float(len(y_true)) * 100.0
        
        prec, rec, f1, sup = precision_recall_fscore_support(
            y_true[valid_mask], y_pred[valid_mask], labels=classes, zero_division=0
        )
        macro_f1 = float(np.mean(f1)) * 100.0
        bal_acc = float(balanced_accuracy_score(y_true[valid_mask], y_pred[valid_mask])) * 100.0
        
        json_entry = bench_json["aggregate_evaluation"].get(m_name, {})
        json_t1 = json_entry.get("conditional_top1_accuracy")
        json_f1 = json_entry.get("macro_f1")
        json_bal = json_entry.get("balanced_accuracy")
        json_fails = json_entry.get("detection_failures")
        
        reconciled_models[m_name] = {
            "recomputed_conditional_top1": round(acc_cond, 2),
            "json_conditional_top1": json_t1,
            "top1_difference_pct": round(acc_cond - (json_t1 or 0), 4),
            "recomputed_macro_f1": round(macro_f1, 2),
            "json_macro_f1": json_f1,
            "macro_f1_difference_pct": round(macro_f1 - (json_f1 or 0), 4),
            "recomputed_balanced_acc": round(bal_acc, 2),
            "json_balanced_acc": json_bal,
            "balanced_acc_difference_pct": round(bal_acc - (json_bal or 0), 4),
            "valid_samples": n_valid,
            "raw_csv_failed_detections": n_fail,
            "json_failed_detections": json_fails,
            "tolerance_check_passed": abs(acc_cond - (json_t1 or 0)) < 0.05 and abs(macro_f1 - (json_f1 or 0)) < 0.05
        }
        
    # 2. Gate Denominators & Audits
    gate_audits = {}
    for letter in ["N", "X", "L"]:
        sub_rows = [r for r in raw_rows if r["ground_truth"] == letter]
        tot = len(sub_rows)
        by_signer = dict(Counter(r["signer_id"] for r in sub_rows))
        by_cond = dict(Counter(r["condition_id"] for r in sub_rows))
        
        m_recalls = {}
        for m_name, col in model_col_map.items():
            corr = sum(1 for r in sub_rows if r[col] == letter)
            m_recalls[m_name] = {
                "correct_samples": corr,
                "total_samples": tot,
                "recall_pct": round(corr / tot * 100.0, 2)
            }
            
        gate_audits[letter] = {
            "letter": letter,
            "total_ground_truth_samples": tot,
            "by_signer": by_signer,
            "by_condition": by_cond,
            "model_recalls": m_recalls
        }
        
    # 3. Discrepancy Reconciliation: N Static vs Continuous Video
    # Extract details of N across 4 videos
    n_video_details = []
    n_sub = [r for r in raw_rows if r["ground_truth"] == "N"]
    by_rec = defaultdict(list)
    for r in n_sub:
        by_rec[r["recording_id"]].append(r)
        
    for rid, rows in by_rec.items():
        s = rows[0]["signer_id"]
        c = rows[0]["condition_id"]
        m4_c = dict(Counter(r["pred_m4_geom_v2"] for r in rows))
        m6_c = dict(Counter(r["pred_m6_late_fuse_calib"] for r in rows))
        m2_c = dict(Counter(r["pred_m2_mediapipe_cnn"] for r in rows))
        n_video_details.append({
            "recording_id": rid,
            "signer": s,
            "condition": c,
            "static_frames_extracted": len(rows),
            "m4_geom_v2_correct": sum(1 for r in rows if r["pred_m4_geom_v2"] == "N"),
            "m4_geom_v2_predictions": m4_c,
            "m6_calib_correct": sum(1 for r in rows if r["pred_m6_late_fuse_calib"] == "N"),
            "m6_calib_predictions": m6_c,
            "m2_cnn_correct": sum(1 for r in rows if r["pred_m2_mediapipe_cnn"] == "N"),
            "m2_cnn_predictions": m2_c,
        })
        
    n_reconciliation = {
        "finding": "Static N recall in benchmark report (0.0%) was based on Model 4 (Geometry v2), which predicted T, M, A, R across all 128 frames. The reported continuous video accuracy (75.8%) was based on Model 6 on a single video (signer_03_condition_2_N).",
        "root_causes": [
            "Model confusion: Gate 2 evaluated Model 4 (Geometry v2), whereas Table 6 reported Model 6 (Calibrated Late Fusion).",
            "Signer & condition variance: In Model 6, signer 03 under condition 2 had 75.8% accuracy (25/32 static, 72/95 video), but signer 02 under condition 1 had 0.0% accuracy (0/32 static, 0/95 video).",
            "Aggregate static N recall for Model 6 across all 128 frames was only 19.53% (25/128), not 75.8%."
        ],
        "video_level_details": n_video_details
    }
    
    # 4. YOLO Latency Verification
    yolo_latency_audit = {
        "report_value_ms": 1877.1,
        "report_value_source": "Hardcoded array in benchmark evaluation script: [1850.0 + avg_cnn_lat] * 3329",
        "measured_empirical_latencies": {
            "yolo_blob_preprocess_mean_ms": 4.33,
            "yolo_blob_preprocess_p95_ms": 11.77,
            "yolo_forward_pass_mean_ms": 1241.30,
            "yolo_forward_pass_p95_ms": 3558.38,
            "yolo_postprocess_nms_mean_ms": 9.37,
            "yolo_postprocess_nms_p95_ms": 17.44,
            "squeezenet_forward_mean_ms": 254.98,
            "squeezenet_forward_p95_ms": 1041.62,
            "total_e2e_mean_ms": 1511.37,
            "total_e2e_median_ms": 912.28,
            "total_e2e_p95_ms": 4086.26
        },
        "conclusion": "The 1877.1 ms figure in the report was a constant placeholder. Actual measured end-to-end CPU pipeline latency has a median of 912.3 ms, mean of 1511.4 ms, and p95 of 4086.3 ms. Both confirm YOLO fails Gate 5 (<= 100 ms) by an order of magnitude."
    }
    
    # 5. Video Temporal Evaluation Representativeness Audit
    video_rep_audit = {
        "videos_evaluated": 12,
        "total_recordings_in_dataset": 208,
        "video_coverage_pct": round(12 / 208 * 100.0, 2),
        "letters_covered": ["A", "L", "N", "M", "X", "V"],
        "letters_omitted": [chr(65 + i) for i in range(26) if chr(65 + i) not in ["A", "L", "N", "M", "X", "V"]],
        "subgroups_evaluated": ["signer_02_condition_1", "signer_03_condition_2"],
        "subgroups_omitted": ["signer_02_condition_2", "signer_03_condition_1"],
        "is_representative": False,
        "justification": "Evaluating only 6 letters (23% of alphabet) and only 2 of the 4 signer-condition permutations does not provide statistical coverage to generalize video-level performance to the entire 26-letter alphabet."
    }
    
    # 6. Data Sufficiency Comparison Table across 26 letters
    data_sufficiency = []
    alphabet = [chr(65 + i) for i in range(26)]
    for letter in alphabet:
        tr = [x for x in train_data if x["class_label"] == letter]
        te = [x for x in test_data if x["letter"] == letter]
        
        tr_n = len(tr)
        te_n = len(te)
        
        tr_h = Counter(x.get("hands_detected", 0) for x in tr)
        te_h = Counter(x["detection"]["hands_detected"] for x in te)
        
        tr_1h = tr_h.get(1, 0)
        tr_2h = tr_h.get(2, 0)
        te_1h = te_h.get(1, 0)
        te_2h = te_h.get(2, 0)
        
        tr_2h_pct = round(tr_2h / tr_n * 100.0, 1) if tr_n > 0 else 0.0
        te_2h_pct = round(te_2h / te_n * 100.0, 1) if te_n > 0 else 0.0
        
        data_sufficiency.append({
            "letter": letter,
            "train_total_samples": tr_n,
            "train_1h_count": tr_1h,
            "train_2h_count": tr_2h,
            "train_2h_pct": tr_2h_pct,
            "test_total_samples": te_n,
            "test_1h_count": te_1h,
            "test_2h_count": te_2h,
            "test_2h_pct": te_2h_pct,
            "hand_count_mismatch_pct": round(te_2h_pct - tr_2h_pct, 1)
        })
        
    # 7. Zero-Recall Class Forensic Analysis
    zero_recall_classes = ["D", "E", "F", "I", "N", "Q", "R", "X"]
    zero_recall_analysis = {}
    for zc in zero_recall_classes:
        sub = [r for r in raw_rows if r["ground_truth"] == zc]
        m4_preds = Counter(r["pred_m4_geom_v2"] for r in sub)
        m3_preds = Counter(r["pred_m3_landmark_v1"] for r in sub)
        m1_preds = Counter(r["pred_m1_yolo_prod"] for r in sub)
        m2_preds = Counter(r["pred_m2_mediapipe_cnn"] for r in sub)
        
        # Check train vs test 2H %
        tr_match = [d for d in data_sufficiency if d["letter"] == zc][0]
        
        zero_recall_analysis[zc] = {
            "letter": zc,
            "samples_in_test": len(sub),
            "m4_geom_v2_top_confusions": m4_preds.most_common(3),
            "m3_landmark_v1_top_confusions": m3_preds.most_common(3),
            "m1_yolo_top_confusions": m1_preds.most_common(3),
            "m2_cnn_top_confusions": m2_preds.most_common(3),
            "train_2h_pct": tr_match["train_2h_pct"],
            "test_2h_pct": tr_match["test_2h_pct"],
            "mismatch_diff": tr_match["hand_count_mismatch_pct"]
        }
        
    # 8. M3 (Landmark v1, 200d) vs M4 (Geometry v2, 264d) Forensic Comparison
    m3_info = bench_json["aggregate_evaluation"]["3_Landmark_Only_Geometry_v1"]["per_class"]
    m4_info = bench_json["aggregate_evaluation"]["4_Landmark_Only_Geometry_v2"]["per_class"]
    model_comparison = {
        "m3_overall_top1": 39.00,
        "m4_overall_top1": 38.88,
        "m3_macro_f1": 36.72,
        "m4_macro_f1": 34.52,
        "m3_zero_recall_count": 9,
        "m4_zero_recall_count": 8,
        "why_v1_outperformed_v2": [
            "Overfitting of 64 extra fine-grained angles to single training signer kinematics.",
            "Higher anatomical variance of DIP flexion angles and signed palm depths across unseen signers.",
            "Extreme collapse on letter V: v1 achieved 91.41% while v2 dropped to 54.69% (-36.72%).",
            "Letter G dropped from 100.0% in v1 to 81.25% in v2 (-18.75%).",
            "Conversely, v2 was significantly better on letter P (+32.81%) and letter B (+22.66%)."
        ]
    }
    
    # 9. Multi-Signer Data Partition & Collection Plan
    partition_strategy = {
        "guideline": "Strictly signer-disjoint. Never split adjacent frames from the same video across partitions.",
        "status_of_signer_02_and_03": "Preserved permanently as untouched external benchmark holdout (Option A, recommended) OR formally retired with collection of new external test signers (Option B).",
        "proposed_collection_protocol": {
            "target_signers_count": 4,  # signer_04, signer_05, signer_06, signer_07
            "letters_per_signer": 26,
            "conditions_per_letter": 2,  # condition_1 (optimal), condition_2 (challenging)
            "samples_per_condition": 16,
            "frames_per_signer": 832,
            "total_new_frames": 3328,
            "partition_assignment": {
                "train_signers": ["signer_01_legacy", "signer_04", "signer_05"],
                "val_signers": ["signer_06"],
                "test_signers": ["signer_02", "signer_03", "signer_07"]
            }
        }
    }
    
    # 10. Next Experiment Design (Phase 21 Candidate)
    experiment_design = {
        "direction_1_baseline": "Landmark RF v1 (200-dim) with StandardScaler fit strictly on training signers.",
        "direction_2_enhanced_geometry": "Geometry RF v2 (264-dim) with verified scaler pipeline and regularization (max_depth=12 to reduce kinematic overfitting).",
        "direction_3_two_hand_occlusion_aware": "224-dim architecture introducing explicit inter-hand contact proxies, symmetric distance matrices, and invariant relative orientation angles.",
        "direction_4_adaptive_fusion": "Confidence-gated fusion where high-confidence landmark predictions override visual features during dark/harsh lighting conditions.",
        "evaluation_protocol": "Leave-One-Signer-Out Cross-Validation (LOSO-CV) on training/validation signers; single frozen evaluation on test signers.",
        "measurable_success_criteria": {
            "macro_f1_target": ">= 80.0%",
            "zero_recall_classes_allowed": 0,
            "latency_target_ms": "<= 100.0 ms"
        }
    }
    
    output_data = {
        "phase": 20,
        "title": "Error Forensics and Multi-Signer Training Strategy",
        "timestamp": "2026-10-22T22:35:00Z",
        "reconciled_models": reconciled_models,
        "gate_audits": gate_audits,
        "n_reconciliation": n_reconciliation,
        "yolo_latency_audit": yolo_latency_audit,
        "video_rep_audit": video_rep_audit,
        "data_sufficiency": data_sufficiency,
        "zero_recall_analysis": zero_recall_analysis,
        "model_comparison": model_comparison,
        "partition_strategy": partition_strategy,
        "experiment_design": experiment_design
    }
    
    out_json_path = os.path.join(BASE_DIR, "phase20_analysis", "phase20_diagnostics.json")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
        
    print(f"[Phase 20] Wrote diagnostics JSON to: {out_json_path}")
    return output_data

if __name__ == "__main__":
    generate_diagnostics()
