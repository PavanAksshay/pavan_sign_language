#!/usr/bin/env python3
"""
Unit tests for TemporalSmoother in temporal_smoother.py.
"""

import time
import unittest
import threading
from temporal_smoother import TemporalSmoother


class TestTemporalSmoother(unittest.TestCase):

    def setUp(self):
        self.smoother = TemporalSmoother(window_size=5, max_idle_sec=1.0)

    def test_single_frame_passthrough(self):
        """Single initial frame passes through directly."""
        res = self.smoother.update("s1", "C", 85.0)
        self.assertEqual(res["smoothed_sign"], "C")
        self.assertEqual(res["raw_sign"], "C")
        self.assertFalse(res["is_smoothed"])
        self.assertEqual(res["window_size"], 1)

    def test_majority_vote_smoothing(self):
        """A transient flicker (e.g. C, C, G, C, C) is smoothed to C."""
        for sign in ["C", "C", "G", "C", "C"]:
            res = self.smoother.update("s1", sign, 80.0)

        self.assertEqual(res["raw_sign"], "C")
        self.assertEqual(res["smoothed_sign"], "C")
        self.assertEqual(res["window_votes"]["C"], 4)
        self.assertEqual(res["window_votes"]["G"], 1)

        # Immediate single-frame flip to G should still smooth to C (since C has 3 votes vs G's 2)
        res = self.smoother.update("s1", "G", 75.0)
        self.assertEqual(res["raw_sign"], "G")
        self.assertEqual(res["smoothed_sign"], "C")  # Smoothed over window
        self.assertTrue(res["is_smoothed"])

    def test_session_isolation(self):
        """Client sessions must be strictly isolated."""
        # Client 1 feeds 'L'
        for _ in range(5):
            self.smoother.update("client_1", "L", 90.0)

        # Client 2 feeds 'C'
        for _ in range(5):
            self.smoother.update("client_2", "C", 88.0)

        res1 = self.smoother.update("client_1", "L", 92.0)
        res2 = self.smoother.update("client_2", "C", 89.0)

        self.assertEqual(res1["smoothed_sign"], "L")
        self.assertEqual(res2["smoothed_sign"], "C")
        self.assertEqual(self.smoother.get_session_count(), 2)

    def test_deterministic_tie_breaking(self):
        """Tied vote counts break ties deterministically by cumulative confidence."""
        smoother = TemporalSmoother(window_size=4, max_idle_sec=5.0)
        # Class A: 2 votes @ 90% = 180
        # Class B: 2 votes @ 60% = 120
        smoother.update("s_tie", "A", 90.0)
        smoother.update("s_tie", "B", 60.0)
        smoother.update("s_tie", "A", 90.0)
        res = smoother.update("s_tie", "B", 60.0)

        self.assertEqual(res["window_votes"]["A"], 2)
        self.assertEqual(res["window_votes"]["B"], 2)
        self.assertTrue(res["tie_broken"])
        self.assertEqual(res["smoothed_sign"], "A")  # A wins by higher cumulative confidence

    def test_sustained_missing_hand_clears_buffer(self):
        """If hand is missing for 2 consecutive frames, buffer is cleared."""
        for _ in range(5):
            self.smoother.update("s_empty", "L", 90.0)

        # Frame 1 missing
        self.smoother.update("s_empty", "-", 0.0)
        # Frame 2 missing -> buffer clears
        res = self.smoother.update("s_empty", "-", 0.0)

        self.assertEqual(res["smoothed_sign"], "-")
        self.assertEqual(res["window_size"], 0)

    def test_idle_timeout_reset(self):
        """If gap between frames exceeds max_idle_sec, buffer auto-resets."""
        smoother = TemporalSmoother(window_size=5, max_idle_sec=0.1)
        smoother.update("s_idle", "L", 90.0, timestamp=100.0)
        smoother.update("s_idle", "L", 90.0, timestamp=100.05)

        # Gap of 0.2s > 0.1s
        res = smoother.update("s_idle", "A", 80.0, timestamp=100.3)
        self.assertEqual(res["smoothed_sign"], "A")
        self.assertEqual(res["window_size"], 1)

    def test_concurrent_threads_safety(self):
        """Ensure thread safety under rapid concurrent updates."""
        errors = []

        def worker(session_name):
            try:
                for i in range(50):
                    sign = "A" if i % 2 == 0 else "B"
                    self.smoother.update(session_name, sign, 70.0 + i)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(f"thread_{k}",)) for k in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)
        self.assertEqual(self.smoother.get_session_count(), 5)


if __name__ == "__main__":
    unittest.main()
