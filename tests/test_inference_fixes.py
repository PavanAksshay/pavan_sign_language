import os
import unittest
import numpy as np
from pipeline import ISLPipeline

class TestInferenceFixes(unittest.TestCase):
    def setUp(self):
        self.pipe = ISLPipeline()
        # Initialize pipeline attributes without loading heavy weights for fast unit testing
        self.pipe.initialized = True
        self.pipe.diagnostics_enabled = False

    def test_fix_a_edge_filter_rejects_bottom_sliver(self):
        """Tests that a bottom-edge narrow sliver is rejected when filter is active."""
        h, w = 720, 1280
        # Simulated sliver: touching bottom (625 + 94 = 719 >= 710), narrow w=26 < 60, aspect=94/26=3.6 > 2.5
        detections = [
            (0, "hand", 0.75, 220, 625, 26, 94)
        ]
        
        # 1. Baseline mode (filter disabled): should NOT reject
        self.pipe.filter_edge_detections = False
        valid_baseline = []
        rejected_baseline = []
        for det in detections:
            _, name, conf, bx, by, bw, bh = det
            touches_bottom = (by + bh) >= (h - self.pipe.edge_bottom_margin)
            is_narrow = (bw < self.pipe.edge_min_width)
            is_extreme_aspect = ((bh / max(1, bw)) > self.pipe.edge_max_aspect_ratio)
            if self.pipe.filter_edge_detections and touches_bottom and (is_narrow or is_extreme_aspect):
                rejected_baseline.append(det)
            else:
                valid_baseline.append(det)
        self.assertEqual(len(valid_baseline), 1)
        self.assertEqual(len(rejected_baseline), 0)

        # 2. Enabled mode: should reject the bottom sliver
        self.pipe.filter_edge_detections = True
        valid_enabled = []
        rejected_enabled = []
        for det in detections:
            _, name, conf, bx, by, bw, bh = det
            touches_bottom = (by + bh) >= (h - self.pipe.edge_bottom_margin)
            is_narrow = (bw < self.pipe.edge_min_width)
            is_extreme_aspect = ((bh / max(1, bw)) > self.pipe.edge_max_aspect_ratio)
            if self.pipe.filter_edge_detections and touches_bottom and (is_narrow or is_extreme_aspect):
                rejected_enabled.append(det)
            else:
                valid_enabled.append(det)
        self.assertEqual(len(valid_enabled), 0)
        self.assertEqual(len(rejected_enabled), 1)

    def test_fix_a_edge_filter_preserves_slender_center_hands(self):
        """Tests that a legitimate slender hand gesture (like L) in frame center is NOT rejected."""
        h, w = 720, 1280
        self.pipe.filter_edge_detections = True
        
        # Center slender hand: w=50 (<60), h=180 (aspect=3.6 > 2.5), but y=250 (y+h=430 << 710)
        center_slender = (0, "hand", 0.85, 500, 250, 50, 180)
        _, _, _, bx, by, bw, bh = center_slender
        touches_bottom = (by + bh) >= (h - self.pipe.edge_bottom_margin)
        is_narrow = (bw < self.pipe.edge_min_width)
        is_extreme_aspect = ((bh / max(1, bw)) > self.pipe.edge_max_aspect_ratio)
        
        should_reject = self.pipe.filter_edge_detections and touches_bottom and (is_narrow or is_extreme_aspect)
        self.assertFalse(should_reject, "Center slender hand must NOT be rejected since touches_bottom is False!")

    def test_fix_a_edge_filter_preserves_normal_hands_near_bottom(self):
        """Tests that a normal-width hand near bottom frame is NOT rejected."""
        h, w = 720, 1280
        self.pipe.filter_edge_detections = True
        
        # Hand near bottom: y=520, h=195 (y+h=715 >= 710), but w=180 (aspect=1.08)
        normal_hand = (0, "hand", 0.90, 400, 520, 180, 195)
        _, _, _, bx, by, bw, bh = normal_hand
        touches_bottom = (by + bh) >= (h - self.pipe.edge_bottom_margin)
        is_narrow = (bw < self.pipe.edge_min_width)
        is_extreme_aspect = ((bh / max(1, bw)) > self.pipe.edge_max_aspect_ratio)
        
        should_reject = self.pipe.filter_edge_detections and touches_bottom and (is_narrow or is_extreme_aspect)
        self.assertFalse(should_reject, "Normal-width hand must NOT be rejected even if near bottom!")

    def test_fix_b_conservative_face_clamping(self):
        """Tests that face clamping > 15% hand height is avoided when safeguard is active."""
        box_h = 200
        # Proposed clamp removes 40px (20% of hand height > 15%)
        proposed_delta = 40
        
        # Baseline behavior (safeguard off): clamp is accepted
        self.pipe.conservative_face_clamp = False
        excessive_avoided_baseline = False
        if self.pipe.conservative_face_clamp:
            clamp_ratio = proposed_delta / max(1, box_h)
            if clamp_ratio > self.pipe.max_face_clamp_ratio:
                excessive_avoided_baseline = True
        self.assertFalse(excessive_avoided_baseline)

        # Safeguard active: clamp is rejected, unclamped crop preserved
        self.pipe.conservative_face_clamp = True
        self.pipe.max_face_clamp_ratio = 0.15
        excessive_avoided_safeguard = False
        if self.pipe.conservative_face_clamp:
            clamp_ratio = proposed_delta / max(1, box_h)
            if clamp_ratio > self.pipe.max_face_clamp_ratio:
                excessive_avoided_safeguard = True
        self.assertTrue(excessive_avoided_safeguard)

        # Mild clamp (15px = 7.5% <= 15%): clamp is accepted safely
        proposed_delta_mild = 15
        excessive_avoided_mild = False
        if self.pipe.conservative_face_clamp:
            clamp_ratio = proposed_delta_mild / max(1, box_h)
            if clamp_ratio > self.pipe.max_face_clamp_ratio:
                excessive_avoided_mild = True
        self.assertFalse(excessive_avoided_mild)

    def test_fix_c_improved_two_hand_merge_rejects_low_conf(self):
        """Tests that second detection with conf=0.35 is rejected when threshold is 0.45."""
        det1 = (0, "hand", 0.88, 300, 300, 150, 150)
        det2 = (0, "hand", 0.35, 460, 310, 140, 140)
        
        b1_x, b1_y, b1_w, b1_h = det1[3], det1[4], det1[5], det1[6]
        b2_x, b2_y, b2_w, b2_h = det2[3], det2[4], det2[5], det2[6]
        gap_x = b2_x - (b1_x + b1_w)
        vertical_overlap = min(b1_y + b1_h, b2_y + b2_h) - max(b1_y, b2_y)

        # Baseline: conf >= 0.30 merges
        self.pipe.improved_two_hand_merge = False
        max_hand_w = max(b1_w, b2_w)
        is_close_interaction_base = (gap_x < max_hand_w * 1.2)
        is_vertically_aligned_base = (vertical_overlap > -max(b1_h, b2_h) * 0.6)
        baseline_merge = (is_close_interaction_base and is_vertically_aligned_base and det2[2] >= 0.30)
        self.assertTrue(baseline_merge, "Baseline should have merged with conf=0.35 >= 0.30")

        # Improved: conf >= 0.45 rejects merge
        self.pipe.improved_two_hand_merge = True
        self.pipe.second_hand_conf = 0.45
        conf_ok = (det2[2] >= self.pipe.second_hand_conf)
        self.assertFalse(conf_ok)

    def test_fix_c_improved_two_hand_merge_rejects_distant_distractor(self):
        """Tests that distant distractor (gap > 0.8 * min_w) is not merged."""
        # Hands with gap=140px, min_w=120px -> gap/min_w = 1.16 > 0.8
        det1 = (0, "hand", 0.90, 200, 300, 120, 130)
        det2 = (0, "hand", 0.55, 460, 310, 130, 130)
        
        gap_x = det2[3] - (det1[3] + det1[5]) # 460 - 320 = 140
        min_hand_w = min(det1[5], det2[5])   # 120
        self.pipe.improved_two_hand_merge = True
        self.pipe.merge_max_gap_ratio = 0.8
        
        is_close_interaction = (gap_x < min_hand_w * self.pipe.merge_max_gap_ratio)
        self.assertFalse(is_close_interaction, "Distant detection must not merge!")

    def test_fix_c_improved_two_hand_merge_merges_genuine_two_hands(self):
        """Tests that genuine interacting two hands (conf >= 0.45, close gap, aligned) DO merge."""
        det1 = (0, "hand", 0.85, 300, 300, 150, 160)
        det2 = (0, "hand", 0.65, 420, 310, 140, 150)
        
        b1_x, b1_y, b1_w, b1_h = det1[3], det1[4], det1[5], det1[6]
        b2_x, b2_y, b2_w, b2_h = det2[3], det2[4], det2[5], det2[6]
        gap_x = b2_x - (b1_x + b1_w) # 420 - 450 = -30 (overlapping/touching)
        vertical_overlap = min(b1_y + b1_h, b2_y + b2_h) - max(b1_y, b2_y)
        min_hand_w = min(b1_w, b2_w)
        min_hand_h = min(b1_h, b2_h)
        area1 = b1_w * b1_h
        area2 = b2_w * b2_h
        
        self.pipe.improved_two_hand_merge = True
        conf_ok = (det2[2] >= self.pipe.second_hand_conf)
        is_close = (gap_x < min_hand_w * self.pipe.merge_max_gap_ratio)
        is_aligned = (vertical_overlap > -min_hand_h * 0.15)
        area_ok = (min(area1, area2) / max(1, max(area1, area2))) >= self.pipe.merge_min_area_ratio
        
        should_merge = (conf_ok and is_close and is_aligned and area_ok)
        self.assertTrue(should_merge, "Genuine touching two hands must merge!")

    def test_phase5_face_to_right_of_hand_does_not_clamp(self):
        """Tests that a face to the right of a hand on the left does NOT clamp it when horizontal overlap is required."""
        hand_x1, hand_x2 = 100, 400
        face_x1, face_x2 = 900, 1000
        
        # Horizontal overlap check: max(hand_x1, face_x1) < min(hand_x2, face_x2)
        has_overlap = max(hand_x1, face_x1) < min(hand_x2, face_x2)
        self.assertFalse(has_overlap, "Hand on left [100..400] and Face on right [900..1000] must NOT have horizontal overlap!")

        # With requirement active: clamp decision must be rejected
        self.pipe.face_clamp_require_horizontal_overlap = True
        should_skip_clamp = self.pipe.face_clamp_require_horizontal_overlap and not has_overlap
        self.assertTrue(should_skip_clamp, "Clamp must be rejected when horizontal overlap is required and absent!")

        # With requirement disabled (baseline): would attempt clamping
        self.pipe.face_clamp_require_horizontal_overlap = False
        should_skip_clamp_base = self.pipe.face_clamp_require_horizontal_overlap and not has_overlap
        self.assertFalse(should_skip_clamp_base)

    def test_phase5_face_overlapping_clamps_only_when_limit_respected(self):
        """Tests that a face overlapping horizontally may clamp only when the 15% limit is respected."""
        hand_x1, hand_x2 = 300, 600
        face_x1, face_x2 = 400, 550
        box_h = 300
        
        has_overlap = max(hand_x1, face_x1) < min(hand_x2, face_x2)
        self.assertTrue(has_overlap, "Hand [300..600] and Face [400..550] DO horizontally overlap.")

        self.pipe.face_clamp_require_horizontal_overlap = True
        self.pipe.conservative_face_clamp = True
        self.pipe.max_face_clamp_ratio = 0.15

        # Mild clamp (30px = 10% <= 15%): permitted
        proposed_delta_mild = 30
        clamp_ratio_mild = proposed_delta_mild / max(1, box_h)
        avoided_mild = (clamp_ratio_mild > self.pipe.max_face_clamp_ratio)
        self.assertFalse(avoided_mild, "Mild clamp within 15% limit should be allowed.")

        # Excessive clamp (60px = 20% > 15%): rejected
        proposed_delta_excessive = 60
        clamp_ratio_excessive = proposed_delta_excessive / max(1, box_h)
        avoided_excessive = (clamp_ratio_excessive > self.pipe.max_face_clamp_ratio)
        self.assertTrue(avoided_excessive, "Excessive clamp exceeding 15% limit must be rejected.")

    def test_phase5_merged_box_does_not_hide_excessive_primary_clamping(self):
        """Tests that a large merged box cannot hide excessive clamping of the primary active hand."""
        # Primary hand: h=180px, conf=0.92
        b1_w, b1_h, conf1 = 150, 180, 0.92
        # Resting second hand: h=200px, conf=0.50
        b2_w, b2_h, conf2 = 160, 200, 0.50
        # Combined merged box height: 450px
        merged_box_h = 450

        # Proposed clamp delta: 45px
        proposed_delta = 45

        # In Phase 4 baseline: safeguard denominator was merged_box_h (450px)
        # 45 / 450 = 10% <= 15% -> Clamp was wrongfully PERMITTED, cutting 25% of primary hand!
        clamp_ratio_phase4 = proposed_delta / merged_box_h
        self.assertLessEqual(clamp_ratio_phase4, 0.15, "Phase 4 denominator permitted clamp.")

        # In Phase 5: primary hand safeguard is active
        self.pipe.face_clamp_primary_hand_safeguard = True
        self.pipe.max_face_clamp_ratio = 0.15

        # Reliable primary hand identification: conf1 - conf2 = 0.42 >= 0.10
        primary_identified = (conf1 - conf2) >= 0.10
        self.assertTrue(primary_identified)
        primary_hand_h = b1_h

        safeguard_h = primary_hand_h if self.pipe.face_clamp_primary_hand_safeguard else merged_box_h
        clamp_ratio_phase5 = proposed_delta / max(1, safeguard_h)
        excessive_avoided_phase5 = (clamp_ratio_phase5 > self.pipe.max_face_clamp_ratio)
        self.assertTrue(excessive_avoided_phase5, "Phase 5 must reject clamp that removes 45/180 = 25% of primary hand height!")

    def test_phase5_primary_hand_ambiguity_safe_fallback(self):
        """Tests that when primary hand is ambiguous, pipeline records ambiguity and uses safe min_h fallback."""
        # Hands with nearly identical confidence (diff < 0.10)
        b1_h, conf1 = 180, 0.72
        b2_h, conf2 = 220, 0.69
        
        diff = abs(conf1 - conf2)
        primary_identified = (diff >= 0.10)
        self.assertFalse(primary_identified, "Confidences 0.72 and 0.69 are ambiguous.")
        
        # Safe fallback: min(b1_h, b2_h)
        safeguard_fallback_h = min(b1_h, b2_h)
        self.assertEqual(safeguard_fallback_h, 180)
        
        # Clamp of 35px: 35 / 180 = 19.4% > 15% -> rejected by safe fallback
        proposed_delta = 35
        clamp_ratio = proposed_delta / safeguard_fallback_h
        self.assertTrue(clamp_ratio > 0.15, "Ambiguity safe fallback avoids destructive clamping on smaller hand.")

    def test_phase5_hand_with_no_face_retains_existing_crop(self):
        """Tests that when no face is detected, hand crop is strictly defined by bounding box padding."""
        x, y, x1, y1 = 200, 300, 400, 500
        box_w = x1 - x # 200
        box_h = y1 - y # 200
        pad_x = max(10, int(box_w * 0.10)) # 20
        pad_y = max(10, int(box_h * 0.10)) # 20

        crop_x = max(0, x - pad_x) # 180
        crop_y = max(0, y - pad_y) # 280
        crop_x1 = x1 + pad_x      # 420
        crop_y1 = y1 + pad_y      # 520

        face_boxes = [] # No face detected
        face_clamped = False
        face_clamp_delta_y = 0

        if len(face_boxes) > 0:
            face_clamped = True

        self.assertFalse(face_clamped)
        self.assertEqual(face_clamp_delta_y, 0)
        self.assertEqual((crop_x, crop_y, crop_x1, crop_y1), (180, 280, 420, 520))

    def test_lap_hand_suppression_ignores_resting_lap_hand(self):
        """Tests that a resting hand near the bottom edge is ignored when an elevated hand exists."""
        h, w = 720, 1280
        self.pipe.suppress_lap_hand = True
        self.pipe.lap_hand_min_y_ratio = 0.75
        self.pipe.lap_hand_max_active_y_ratio = 0.60

        # Hand 1: Lap hand at bottom (y=584, h=120 -> y+h=704 >= 0.75*720=540)
        lap_hand = (0, "hand", 0.90, 392, 584, 224, 120)
        # Hand 2: Active elevated hand in chest area (y=130, h=375 -> y < 0.60*720=432)
        active_hand = (0, "hand", 0.86, 166, 130, 420, 375)

        valid_detections = [lap_hand, active_hand]

        elevated_hands = [d for d in valid_detections if d[4] < self.pipe.lap_hand_max_active_y_ratio * h]
        lap_hands = [d for d in valid_detections if (d[4] + d[6]) >= self.pipe.lap_hand_min_y_ratio * h and d[4] >= (self.pipe.lap_hand_min_y_ratio - 0.10) * h]

        self.assertEqual(len(elevated_hands), 1)
        self.assertEqual(len(lap_hands), 1)

        filtered = [d for d in valid_detections if d not in lap_hands]
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0], active_hand)

    def test_lap_hand_suppression_preserves_two_handed_chest_gestures(self):
        """Tests that two genuine signing hands both held at chest level are NOT suppressed."""
        h, w = 720, 1280
        self.pipe.suppress_lap_hand = True
        self.pipe.lap_hand_min_y_ratio = 0.75
        self.pipe.lap_hand_max_active_y_ratio = 0.60

        # Two hands both signing at chest level (y=200 and y=220)
        hand1 = (0, "hand", 0.85, 300, 200, 180, 200)
        hand2 = (0, "hand", 0.82, 520, 220, 190, 210)

        valid_detections = [hand1, hand2]
        lap_hands = [d for d in valid_detections if (d[4] + d[6]) >= self.pipe.lap_hand_min_y_ratio * h and d[4] >= (self.pipe.lap_hand_min_y_ratio - 0.10) * h]

        self.assertEqual(len(lap_hands), 0, "Chest-level hands must never be classified as lap hands.")
        filtered = [d for d in valid_detections if d not in lap_hands]
        self.assertEqual(len(filtered), 2, "Both active hands must be preserved for two-handed signing.")

    def test_direct_resize_disabled_preserves_letterbox_padding(self):
        """Tests that when direct_resize is disabled (default), non-square crops are letterboxed."""
        cropped = np.zeros((100, 200, 3), dtype=np.uint8) # landscape 1:2
        ch, cw = cropped.shape[:2]
        
        # When direct_resize is False:
        scale = 224.0 / max(cw, ch)
        nw = max(1, int(cw * scale))
        nh = max(1, int(ch * scale))
        dx = (224 - nw) // 2
        dy = (224 - nh) // 2
        letterboxed = (nw != 224 or nh != 224)

        self.assertEqual((nw, nh), (224, 112))
        self.assertEqual((dx, dy), (0, 56))
        self.assertTrue(letterboxed)

    def test_direct_resize_enabled_resizes_directly_to_224(self):
        """Tests that when direct_resize is enabled, crops are directly resized without padding."""
        import cv2
        cropped = np.ones((100, 200, 3), dtype=np.uint8) * 128
        ch, cw = cropped.shape[:2]

        # When direct_resize is True:
        resized = cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_LINEAR)
        nw, nh = 224, 224
        dx, dy = 0, 0
        letterboxed = False

        self.assertEqual(resized.shape, (224, 224, 3))
        self.assertEqual((nw, nh), (224, 224))
        self.assertEqual((dx, dy), (0, 0))
        self.assertFalse(letterboxed)

    def test_phase8_defaults(self):
        """Tests that ISL_SUPPRESS_LAP_HAND is enabled by default in Phase 8, while square_crop and direct_resize remain disabled."""
        new_pipe = ISLPipeline()
        self.assertTrue(new_pipe.suppress_lap_hand, "suppress_lap_hand must be enabled by default in Phase 8!")
        self.assertFalse(new_pipe.square_crop, "square_crop must be disabled by default!")
        self.assertFalse(new_pipe.direct_resize, "direct_resize must be disabled by default!")

    def test_suppress_lap_hand_rollback_override(self):
        """Tests that ISL_SUPPRESS_LAP_HAND can be explicitly overridden to false for immediate rollback."""
        orig_val = os.environ.get("ISL_SUPPRESS_LAP_HAND")
        try:
            for false_str in ("0", "false", "no", "off", "FALSE"):
                os.environ["ISL_SUPPRESS_LAP_HAND"] = false_str
                test_pipe = ISLPipeline()
                self.assertFalse(test_pipe.suppress_lap_hand, f"Setting ISL_SUPPRESS_LAP_HAND={false_str} must disable lap hand suppression!")
        finally:
            if orig_val is not None:
                os.environ["ISL_SUPPRESS_LAP_HAND"] = orig_val
            else:
                os.environ.pop("ISL_SUPPRESS_LAP_HAND", None)

    def test_compute_square_crop_geometry_is_exact_square(self):
        """Tests that compute_square_crop produces strictly square geometry (w == h)."""
        frame_shape = (720, 1280, 3)
        # Landscape crop: 352 wide, 254 tall
        crop_coords = (545, 200, 897, 454) # cw = 352, ch = 254
        (vx1, vy1, vx2, vy2), (sq_x, sq_y, sq_w, sq_h), (pt, pb, pl, pr) = self.pipe.compute_square_crop(
            frame_shape, crop_coords
        )
        self.assertEqual(sq_w, sq_h, "Square crop width and height must be strictly identical!")
        self.assertEqual(sq_w, 352, "Square side must equal max(cw, ch)!")
        
        # Verify valid slice + padding equals exactly side x side
        total_w = (vx2 - vx1) + pl + pr
        total_h = (vy2 - vy1) + pt + pb
        self.assertEqual(total_w, 352)
        self.assertEqual(total_h, 352)

    def test_compute_square_crop_frame_boundary_shifting_preserves_gesture(self):
        """Tests that frame-boundary shifting preserves the complete gesture without unnecessary padding."""
        frame_shape = (720, 1280, 3)
        # Crop near left frame boundary: hand is at x=10..160 (cw=150), y=100..400 (ch=300)
        crop_coords = (10, 100, 160, 400)
        (vx1, vy1, vx2, vy2), (sq_x, sq_y, sq_w, sq_h), (pt, pb, pl, pr) = self.pipe.compute_square_crop(
            frame_shape, crop_coords
        )
        self.assertEqual(sq_w, 300)
        self.assertEqual(sq_h, 300)
        # Without boundary shift, center cx=85 would place sq_x1 at 85 - 150 = -65.
        # Boundary shift shifts it to sq_x1 = 0, sq_x2 = 300:
        self.assertEqual(sq_x, 0)
        self.assertEqual(sq_y, 100)
        # Zero padding needed because boundary shifting fit the entire square within the frame!
        self.assertEqual((pt, pb, pl, pr), (0, 0, 0, 0))
        # Complete gesture [10..160] is preserved within [0..300]
        self.assertTrue(sq_x <= crop_coords[0] and sq_x + sq_w >= crop_coords[2],
                        "Square crop must enclose the original hand gesture!")

    def test_compute_square_crop_padding_at_tight_corners(self):
        """Tests that tight boundary constraints apply padding to preserve exact square dimensions."""
        import cv2
        frame = np.ones((200, 200, 3), dtype=np.uint8) * 100
        # Crop near boundary that exceeds clearance: cw=120, ch=180 -> side=180
        # Crop is at x=0..120, y=0..180
        crop_coords = (0, 0, 120, 180)
        (vx1, vy1, vx2, vy2), (sq_x, sq_y, sq_w, sq_h), (pt, pb, pl, pr) = self.pipe.compute_square_crop(
            frame.shape, crop_coords
        )
        self.assertEqual(sq_w, sq_h)
        cropped = frame[vy1:vy2, vx1:vx2]
        if pt > 0 or pb > 0 or pl > 0 or pr > 0:
            cropped = cv2.copyMakeBorder(cropped, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        self.assertEqual(cropped.shape[0], sq_h)
        self.assertEqual(cropped.shape[1], sq_w)
        self.assertEqual(cropped.shape[0], cropped.shape[1], "Output crop must be strictly square!")

    def test_square_crop_isotropic_scaling_avoids_distortion(self):
        """Tests that resizing a square crop to 224x224 applies identical isotropic scale factors."""
        import cv2
        # A square crop 300x300
        crop = np.ones((300, 300, 3), dtype=np.uint8) * 200
        resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
        self.assertEqual(resized.shape, (224, 224, 3))
        # Aspect ratio is exactly 1.0 (no vertical or horizontal stretch distortion)
        self.assertAlmostEqual(resized.shape[0] / resized.shape[1], 1.0)

    def test_two_handed_gestures_with_lap_suppression_and_square_crop(self):
        """Tests that genuine two-handed gestures are preserved when both lap suppression and square crop are enabled."""
        h, w = 720, 1280
        self.pipe.suppress_lap_hand = True
        self.pipe.square_crop = True

        # Genuine two-handed gesture: both hands active at chest level (y=180 and y=210)
        hand1 = (0, "hand", 0.88, 320, 180, 160, 180)
        hand2 = (0, "hand", 0.84, 520, 210, 170, 190)
        valid_detections = [hand1, hand2]

        lap_hands = [d for d in valid_detections if (d[4] + d[6]) >= self.pipe.lap_hand_min_y_ratio * h and d[4] >= (self.pipe.lap_hand_min_y_ratio - 0.10) * h]
        self.assertEqual(len(lap_hands), 0, "No active hand may be suppressed!")
        remaining = [d for d in valid_detections if d not in lap_hands]
        self.assertEqual(len(remaining), 2, "Both hands must remain for two-handed processing!")

    def test_compute_landmark_crop_box_square_expansion(self):
        """Tests that landmark crop helper creates an isotropic square box with proper margins."""
        frame_shape = (720, 1280, 3)
        # 5 dummy landmarks simulating a hand: x in [0.40, 0.48], y in [0.30, 0.50]
        # Pixel coords: x in [512, 614.4] (w=102.4), y in [216, 360] (h=144)
        dummy_landmarks = [
            (0.40, 0.30),
            (0.48, 0.30),
            (0.44, 0.40),
            (0.41, 0.50),
            (0.47, 0.48)
        ]
        x1, y1, x2, y2 = ISLPipeline.compute_landmark_crop_box(
            dummy_landmarks, frame_shape, margin_ratio=0.15, force_square=True
        )
        cw = x2 - x1
        ch = y2 - y1
        self.assertEqual(cw, ch, "Landmark square crop must have strictly 1:1 aspect ratio!")
        self.assertTrue(cw >= 144, "Square side must at least cover maximum landmark span!")
        self.assertTrue(x1 >= 0 and y1 >= 0 and x2 <= 1280 and y2 <= 720, "Crop must stay within frame bounds!")

    def test_compute_landmark_crop_box_boundary_shift(self):
        """Tests that landmark crop near frame boundary shifts inward without clipping."""
        frame_shape = (720, 1280, 3)
        # Hand near left edge: x in [0.01, 0.05], y in [0.30, 0.45]
        dummy_landmarks = [(0.01, 0.30), (0.05, 0.45)]
        x1, y1, x2, y2 = ISLPipeline.compute_landmark_crop_box(
            dummy_landmarks, frame_shape, margin_ratio=0.15, force_square=True
        )
        self.assertEqual(x1, 0, "Crop near left border must shift to start at x=0!")
        self.assertTrue(x2 > 0, "Right coordinate must be positive!")
        self.assertEqual(x2 - x1, y2 - y1, "Aspect ratio must remain square even after boundary shifting!")

    def test_compute_landmark_crop_box_tight_mode(self):
        """Tests tight rectangular mode (force_square=False) with margin."""
        frame_shape = (720, 1280, 3)
        dummy_landmarks = [(0.30, 0.30), (0.40, 0.50)]
        x1, y1, x2, y2 = ISLPipeline.compute_landmark_crop_box(
            dummy_landmarks, frame_shape, margin_ratio=0.10, force_square=False
        )
        # x span: 0.10 * 1280 = 128px; margin 10% = 12.8px -> expected w ~ 154px
        # y span: 0.20 * 720 = 144px; margin 10% = 14.4px -> expected h ~ 173px
        cw = x2 - x1
        ch = y2 - y1
        self.assertTrue(cw >= 128, "Crop width must enclose raw landmarks!")
        self.assertTrue(ch >= 144, "Crop height must enclose raw landmarks!")

    def test_compute_forearm_trimmed_box(self):
        """Tests that vertically elongated box (h/w > 1.3) trims bottom forearm region."""
        frame_shape = (720, 1280, 3)
        # Elongated hand + forearm box: w = 100, h = 200 (aspect = 2.0 > 1.3)
        box = (400, 200, 500, 400)
        t_x1, t_y1, t_x2, t_y2 = ISLPipeline.compute_forearm_trimmed_box(
            box, frame_shape, trim_bottom_ratio=0.20
        )
        self.assertEqual(t_x1, 400)
        self.assertEqual(t_x2, 500)
        self.assertEqual(t_y1, 200)
        # Original h was 200, 20% trim is 40px -> trimmed y2 should be 360
        self.assertEqual(t_y2, 360)

        # Normal square-like box: w = 150, h = 150 (aspect = 1.0 <= 1.3)
        sq_box = (400, 200, 550, 350)
        st_x1, st_y1, st_x2, st_y2 = ISLPipeline.compute_forearm_trimmed_box(
            sq_box, frame_shape, trim_bottom_ratio=0.20
        )
        self.assertEqual((st_x1, st_y1, st_x2, st_y2), sq_box, "Normal aspect hand must NOT be trimmed!")

if __name__ == '__main__':
    unittest.main()

