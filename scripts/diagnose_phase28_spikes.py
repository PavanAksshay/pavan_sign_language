#!/usr/bin/env python3
"""
Phase 28 — GPU Latency & Synchronization Spike Diagnostic Probe
Instruments 1,000 unique development frames from signer_04 across all stages.
"""

import os
import sys
import time
import glob
import json
import resource
import platform
import cv2
import numpy as np
from typing import Dict, Any, List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from pipeline import ISLPipeline


def get_rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    rss = ru.ru_maxrss
    if platform.system() == "Darwin":
        return round(rss / (1024.0 * 1024.0), 2)
    return round(rss / 1024.0, 2)


def compute_stats(arr: List[float]) -> Dict[str, float]:
    if not arr:
        return {}
    a = np.array(arr, dtype=np.float64)
    return {
        "count": len(arr),
        "mean": round(float(np.mean(a)), 3),
        "median": round(float(np.median(a)), 3),
        "p90": round(float(np.percentile(a, 90)), 3),
        "p95": round(float(np.percentile(a, 95)), 3),
        "p99": round(float(np.percentile(a, 99)), 3),
        "min": round(float(np.min(a)), 3),
        "max": round(float(np.max(a)), 3),
        "std": round(float(np.std(a)), 3)
    }


def main():
    print("=" * 75)
    print("Phase 28 — Diagnostic Probe: GPU Latency & Synchronization Spikes")
    print("=" * 75)

    # 1. Collect unique signer_04 frames
    frame_paths = sorted(glob.glob(os.path.join(BASE_DIR, "data", "extracted_frames", "signer_04", "*", "*", "*.jpg")))
    print(f"Total available unique signer_04 frames: {len(frame_paths)}")
    assert len(frame_paths) >= 1000, f"Expected at least 1,000 frames, found {len(frame_paths)}"

    # Select 30 warm-up frames + 1,000 evaluation frames (1,030 total unique frames)
    warmup_paths = frame_paths[:30]
    eval_paths = frame_paths[30:1030]
    print(f"Using {len(warmup_paths)} warm-up frames and {len(eval_paths)} unique timed evaluation frames.")

    # 2. Initialize pipeline with Model C and Metal GPU delegate
    pipe = ISLPipeline()
    pipe.set_backend("yolo")  # Production baseline detector preserved
    pipe.set_recognition_arch("model_c")
    pipe.set_mediapipe_delegate("gpu")
    pipe.set_model_c_input_resolution("native")
    pipe.diagnostics_enabled = False

    print(f"\nPipeline initialized:")
    print(f"  Recognition arch: {pipe.recognition_arch}")
    print(f"  MediaPipe delegate: {pipe.mediapipe_delegate}")
    print(f"  Model C input resolution: {pipe.model_c_input_resolution}")

    # 3. Warm-up
    print("\nExecuting warm-up (30 frames)...", flush=True)
    for p in warmup_paths:
        img = cv2.imread(p)
        if img is not None:
            pipe.process_frame(img, return_images=False)
    print(f"Warm-up complete. Initial RSS: {get_rss_mb()} MB")

    # 4. Timed Evaluation (1,000 unique frames)
    # Testing streaming mode (return_images=False) to isolate core pipeline
    total_latencies = []
    stage_breakdowns: Dict[str, List[float]] = {
        "1_receipt": [],
        "2_validation": [],
        "3_copy": [],
        "4_resize": [],
        "5_color_conversion": [],
        "6_mp_image_construction": [],
        "7_detection": [],
        "8_postprocessing": [],
        "9_feature_extraction": [],
        "10_scaler_transform": [],
        "11_classifier_prediction": [],
        "12_formatting": []
    }
    hand_counts = []
    rss_checkpoints = {}
    rss_checkpoints["start_rss_mb"] = get_rss_mb()

    # Track slow frame details
    slow_frames = []

    print(f"\nProcessing {len(eval_paths)} unique frames...", flush=True)
    t_bench_start = time.perf_counter()

    for idx, p in enumerate(eval_paths):
        img = cv2.imread(p)
        if img is None:
            continue

        t0 = time.perf_counter()
        res = pipe.process_frame(img, return_images=False)
        t_elapsed = (time.perf_counter() - t0) * 1000.0
        total_latencies.append(t_elapsed)

        diag = res.get("diagnostics", {})
        st = diag.get("stage_timings_ms", {})
        h_cnt = diag.get("hands_detected_count", 0)
        hand_counts.append(h_cnt)

        for k in stage_breakdowns:
            if k in st:
                stage_breakdowns[k].append(st[k])

        # Track any frame exceeding 80 ms
        if t_elapsed > 80.0:
            slow_frames.append({
                "frame_index": idx,
                "path": os.path.basename(p),
                "total_ms": round(t_elapsed, 2),
                "hands_detected": h_cnt,
                "stage_timings": st
            })

        if (idx + 1) in (100, 250, 500, 750, 1000):
            rss_checkpoints[f"frame_{idx + 1}_rss_mb"] = get_rss_mb()
            print(f"  Checkpoint [{idx + 1}/{len(eval_paths)}]: RSS = {rss_checkpoints[f'frame_{idx + 1}_rss_mb']} MB, Last Latency = {t_elapsed:.1f} ms")

    total_bench_duration = time.perf_counter() - t_bench_start
    throughput_fps = round(len(total_latencies) / total_bench_duration, 2)
    rss_checkpoints["end_rss_mb"] = get_rss_mb()

    pipe.shutdown()

    # 5. Compute and display statistics
    overall_stats = compute_stats(total_latencies)
    print("\n" + "=" * 75)
    print("DIAGNOSTIC BENCHMARK RESULTS (1,000 Unique Frames, Native 1280x720, GPU)")
    print("=" * 75)
    print(f"Total Timed Frames: {overall_stats['count']}")
    print(f"Throughput: {throughput_fps} FPS")
    print(f"Latency Distribution (ms):")
    print(f"  Median (p50): {overall_stats['median']} ms")
    print(f"  p90:          {overall_stats['p90']} ms")
    print(f"  p95:          {overall_stats['p95']} ms")
    print(f"  p99:          {overall_stats['p99']} ms")
    print(f"  Min / Max:    {overall_stats['min']} ms / {overall_stats['max']} ms")
    print(f"  Mean (Std):   {overall_stats['mean']} ms (+/- {overall_stats['std']} ms)")

    p95_pass = overall_stats['p95'] < 100.0
    print(f"\nTarget Gate (p95 < 100 ms): {'PASSED' if p95_pass else 'FAILED'} (Measured: {overall_stats['p95']} ms)")

    print("\n--- Per-Stage Breakdown (Mean / p95 ms) ---")
    for k, v in stage_breakdowns.items():
        if v:
            st_stats = compute_stats(v)
            print(f"  {k:28s}: Mean={st_stats['mean']:6.2f} ms | Median={st_stats['median']:6.2f} ms | p95={st_stats['p95']:6.2f} ms | Max={st_stats['max']:6.2f} ms")

    print("\n--- Hand Count Correlation ---")
    h_arr = np.array(hand_counts)
    lat_arr = np.array(total_latencies)
    for h in sorted(set(hand_counts)):
        mask = (h_arr == h)
        sub_lats = lat_arr[mask].tolist()
        sub_stats = compute_stats(sub_lats)
        print(f"  Hands={h} (N={len(sub_lats):4d}): Median={sub_stats['median']:5.2f} ms | p95={sub_stats['p95']:5.2f} ms | Max={sub_stats['max']:5.2f} ms")

    print("\n--- Slow Frame Analysis (Top 5 Slowest Frames) ---")
    slowest_5 = sorted(slow_frames, key=lambda x: x["total_ms"], reverse=True)[:5]
    for s in slowest_5:
        print(f"  Frame {s['frame_index']} ({s['path']}): Total={s['total_ms']} ms, Hands={s['hands_detected']}")
        print(f"    Detection={s['stage_timings'].get('7_detection')} ms, ColorConv={s['stage_timings'].get('5_color_conversion')} ms, FeatExt={s['stage_timings'].get('9_feature_extraction')} ms")

    print("\n--- Memory RSS Profile ---")
    for k, v in rss_checkpoints.items():
        print(f"  {k}: {v} MB")

    out_diag_path = os.path.join(BASE_DIR, "phase28_diagnostic_probe.json")
    with open(out_diag_path, "w") as f:
        json.dump({
            "overall_stats": overall_stats,
            "throughput_fps": throughput_fps,
            "p95_gate_passed": p95_pass,
            "stage_breakdowns": {k: compute_stats(v) for k, v in stage_breakdowns.items() if v},
            "hand_count_breakdowns": {
                f"hands_{h}": compute_stats(lat_arr[h_arr == h].tolist())
                for h in sorted(set(hand_counts))
            },
            "rss_checkpoints": rss_checkpoints,
            "total_slow_frames_gt_80ms": len(slow_frames),
            "slowest_5_frames": slowest_5
        }, f, indent=2)
    print(f"\nDiagnostic results saved to {out_diag_path}")


if __name__ == "__main__":
    main()
