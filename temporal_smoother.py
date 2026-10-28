#!/usr/bin/env python3
"""
Phase 14C — Temporal Smoother Component

Provides thread-safe, multi-session rolling temporal smoothing (majority-vote
and confidence-weighted tie-breaking) for sign language recognition.

Requirements:
- Independent buffer state per active client/session.
- Automatic reset on session expiry or idle timeout (configurable max_idle_sec).
- Explicit missing-detection handling (immediate clearing on sustained missing hand).
- Deterministic tie-breaking (highest cumulative confidence in window, then most recent).
- Non-blocking and isolated per session.
- Rich diagnostics: exposes raw prediction, smoothed prediction, vote counts, and latency.
"""

import time
import threading
from collections import deque, Counter
from typing import Dict, List, Optional, Tuple, Any


class SessionBuffer:
    """Stores rolling predictions for a single client session."""

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.history: deque = deque(maxlen=window_size)
        self.last_timestamp: float = time.time()
        self.consecutive_empty: int = 0

    def append(self, sign: str, confidence: float, timestamp: float, prob_dict: Optional[Dict[str, float]] = None):
        self.last_timestamp = timestamp
        if sign == "-" or sign is None:
            self.consecutive_empty += 1
            if self.consecutive_empty >= 2:
                # Sustained missing hand -> clear history to prevent stale ghost predictions
                self.history.clear()
        else:
            self.consecutive_empty = 0
            self.history.append({
                "sign": sign,
                "confidence": float(confidence),
                "timestamp": timestamp,
                "prob_dict": prob_dict or {}
            })

    def clear(self):
        self.history.clear()
        self.consecutive_empty = 0
        self.last_timestamp = time.time()


class TemporalSmoother:
    """
    Thread-safe multi-session temporal smoother with majority-vote and
    confidence-weighted deterministic tie-breaking.
    """

    def __init__(self, window_size: int = 5, max_idle_sec: float = 2.0, min_confidence_floor: float = 15.0):
        self.window_size = max(1, int(window_size))
        self.max_idle_sec = float(max_idle_sec)
        self.min_confidence_floor = float(min_confidence_floor)
        self._sessions: Dict[str, SessionBuffer] = {}
        self._lock = threading.Lock()

    def _get_or_create_session(self, session_id: str, current_time: float) -> SessionBuffer:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionBuffer(self.window_size)
            return self._sessions[session_id]

        buf = self._sessions[session_id]
        # Check idle timeout
        if current_time - buf.last_timestamp > self.max_idle_sec:
            buf.clear()
        return buf

    def update(
        self,
        session_id: str,
        raw_sign: str,
        raw_confidence: float,
        timestamp: Optional[float] = None,
        prob_dict: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Appends a new frame prediction and computes the temporally smoothed sign.
        Returns a rich dictionary containing both raw and smoothed results.
        """
        now = time.time() if timestamp is None else float(timestamp)
        clean_sid = str(session_id) if session_id else "default_session"

        with self._lock:
            buf = self._get_or_create_session(clean_sid, now)
            buf.append(raw_sign, raw_confidence, now, prob_dict)

            # If no history or current sign is empty
            if not buf.history:
                return {
                    "smoothed_sign": raw_sign,
                    "smoothed_confidence": raw_confidence,
                    "raw_sign": raw_sign,
                    "raw_confidence": raw_confidence,
                    "window_size": 0,
                    "window_votes": {},
                    "is_smoothed": False,
                    "tie_broken": False,
                    "session_id": clean_sid
                }

            # If only 1 sample in history
            if len(buf.history) == 1:
                item = buf.history[-1]
                return {
                    "smoothed_sign": item["sign"],
                    "smoothed_confidence": item["confidence"],
                    "raw_sign": raw_sign,
                    "raw_confidence": raw_confidence,
                    "window_size": 1,
                    "window_votes": {item["sign"]: 1},
                    "is_smoothed": False,
                    "tie_broken": False,
                    "session_id": clean_sid
                }

            # Count votes
            signs = [entry["sign"] for entry in buf.history]
            vote_counts = Counter(signs)
            max_votes = max(vote_counts.values())

            # Find candidates with the maximum vote count
            candidates = [s for s, count in vote_counts.items() if count == max_votes]

            tie_broken = False
            if len(candidates) == 1:
                winner = candidates[0]
            else:
                # Deterministic Tie-breaking:
                # 1. Highest cumulative confidence across the window
                tie_broken = True
                cand_conf = {c: sum(e["confidence"] for e in buf.history if e["sign"] == c) for c in candidates}
                sorted_by_conf = sorted(cand_conf.items(), key=lambda x: x[1], reverse=True)

                if len(sorted_by_conf) > 1 and sorted_by_conf[0][1] > sorted_by_conf[1][1]:
                    winner = sorted_by_conf[0][0]
                else:
                    # 2. Most recent frame's prediction among the tied candidates
                    winner = None
                    for entry in reversed(buf.history):
                        if entry["sign"] in candidates:
                            winner = entry["sign"]
                            break
                    if winner is None:
                        winner = candidates[0]

            # Compute smoothed confidence (mean confidence of winner frames in window)
            winner_confs = [e["confidence"] for e in buf.history if e["sign"] == winner]
            smoothed_conf = round(float(np_mean(winner_confs)), 2)

            return {
                "smoothed_sign": winner,
                "smoothed_confidence": smoothed_conf,
                "raw_sign": raw_sign,
                "raw_confidence": raw_confidence,
                "window_size": len(buf.history),
                "window_votes": dict(vote_counts),
                "is_smoothed": (winner != raw_sign),
                "tie_broken": tie_broken,
                "session_id": clean_sid
            }

    def reset(self, session_id: Optional[str] = None):
        """Resets the buffer for a specific session or all sessions."""
        with self._lock:
            if session_id:
                if session_id in self._sessions:
                    self._sessions[session_id].clear()
            else:
                self._sessions.clear()

    def get_session_count(self) -> int:
        with self._lock:
            return len(self._sessions)


def np_mean(vals: List[float]) -> float:
    if not vals:
        return 0.0
    return sum(vals) / len(vals)
