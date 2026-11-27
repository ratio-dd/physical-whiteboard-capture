"""Deterministic camera-free tests through BoardEstimator's public interface."""

import unittest

import cv2
import numpy as np

from vision import BoardEstimator, enhance_board


FULL = ((0, 0), (1, 0), (1, 1), (0, 1))
SKEWED = ((0.27, 0.08), (0.90, 0.18), (0.97, 0.91), (0.05, 0.83))
EPOCH = 1_700_000_000_000


class VisionTest(unittest.TestCase):
    def setUp(self):
        self.blank = np.full((192, 320, 3), 230, np.uint8)
        self.est = BoardEstimator(FULL, 320, 192)
        self.now = EPOCH

    def sample(self, frame):
        result = self.est.update(frame, self.now)
        self.assertEqual(set(result), {"held", "observed_at", "glare"})
        self.assertLessEqual(0, result["held"])
        self.assertLessEqual(result["held"], 1)
        self.assertTrue(0 <= result["glare"] <= 1)
        self.assertIsInstance(result["observed_at"], int)
        self.now += 150
        return result

    def settle(self, frame):
        for _ in range(9):
            result = self.sample(frame)
        return result

    def test_invalid_calibrations(self):
        for corners in (
            FULL[:3], [(0, 0), (1, 0), (float("nan"), 1), (0, 1)],
            [(0, 0), (1.01, 0), (1, 1), (0, 1)], [FULL[i] for i in (0, 2, 1, 3)],
            FULL[::-1], (FULL[0], FULL[1], FULL[1], FULL[3]),
            ((0, 0), (1, 0), (0.1, 0.1), (0, 1)),
            tuple(np.add((0.5, 0.5), np.array(FULL) * 0.0001)),
            SKEWED[1:] + SKEWED[:1], SKEWED[-1:] + SKEWED[:-1],
        ):
            with self.subTest(corners=corners), self.assertRaises(ValueError):
                BoardEstimator(corners, 320, 192)
        for width, height in ((0, 192), (320.5, 192), (320, False)):
            with self.subTest(size=(width, height)), self.assertRaises(ValueError):
                BoardEstimator(FULL, width, height)

    def test_invalid_samples_leave_state_intact(self):
        self.settle(self.blank)
        before, known = self.est.board.copy(), self.est.known.copy()
        for frame, stamp in (
            (self.blank[..., 0], self.now), (self.blank[:100], self.now),
            (self.blank.astype(np.float32), self.now), (None, self.now),
            (self.blank, float("nan")), (self.blank, -1), (self.blank, EPOCH),
        ):
            with self.subTest(stamp=stamp), self.assertRaises(ValueError):
                self.est.update(frame, stamp)
        np.testing.assert_array_equal(self.est.board, before)
        np.testing.assert_array_equal(self.est.known, known)

    def test_perspective_maps_colored_landmarks_under_local_shading(self):
        # A bright gradient and colored crosses exist in the real board plane.
        # Warp them out to the camera, then ask the estimator to recover them.
        h, w = self.blank.shape[:2]
        brightness = np.linspace(143, 237, w, dtype=np.float32)
        surface = np.broadcast_to(brightness[None, :, None], self.blank.shape).astype(np.uint8, order="C")
        markers = [(0.22, 0.31, 2), (0.76, 0.65, 0)]
        for x, y, channel in markers:
            color = [20, 20, 20]
            color[channel] = 158
            cv2.drawMarker(surface, (round(x*(w-1)), round(y*(h-1))), color,
                           markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
        camera_size = (400, 240)
        corners = np.float32(SKEWED) * np.float32([399, 239])
        transform = cv2.getPerspectiveTransform(np.float32(FULL) * np.float32([w-1, h-1]), corners)
        frame = cv2.warpPerspective(surface, transform, camera_size, borderValue=(30, 30, 30))
        self.est = BoardEstimator(SKEWED, *camera_size)
        self.settle(frame)
        self.assertEqual(self.est.board.dtype, np.uint8)
        self.assertEqual(self.est.known.dtype, np.uint8)
        H, W = self.est.known.shape
        self.assertEqual(self.est.board.shape, (H, W, 3))
        for x, y, channel in markers:
            X, Y = round(x*(W-1)), round(y*(H-1))
            patch = self.est.board[Y-3:Y+4, X-3:X+4].astype(int)
            self.assertEqual(self.est.known[Y, X], 255)
            self.assertGreater((patch[..., channel] - patch[..., 1]).max(), 55)
        self.assertTrue(np.all(self.est.known[H//2, W//8:W*7//8] == 255))
        enhanced_paper = self.est.board[H//2, W//8:W*7//8, 1]
        self.assertLess(np.ptp(enhanced_paper), 18)
        self.assertGreater(np.min(enhanced_paper), 205)

    def test_confirmed_changes_use_earliest_visible_sample_then_idle_uses_current_time(self):
        self.assertFalse(np.any(self.est.known))
        result = self.sample(self.blank)
        self.assertEqual(result["observed_at"], EPOCH)
        self.assertFalse(np.any(self.est.known))
        for _ in range(12):
            result = self.sample(self.blank)
            if self.est.known.all():
                break
        self.assertTrue(np.all(self.est.known == 255))
        self.assertEqual(result["observed_at"], EPOCH)

        written = self.blank.copy()
        cv2.putText(written, "A+B", (58, 112), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (28, 38, 164), 3)
        ink = written[..., 0] < 100
        for frame, writing in ((written, True), (self.blank, False)):
            first_seen = self.now
            self.sample(frame)
            # Neither a one-frame stroke nor a one-frame return to white commits.
            self.assertEqual(bool(np.mean(self.est.board[..., 0][ink]) > 190), writing)
            for _ in range(12):
                result = self.sample(frame)
                dark = np.mean(self.est.board[..., 0][ink]) < 110
                if dark == writing:
                    break
            self.assertEqual(dark, writing)
            self.assertEqual(result["observed_at"], first_seen)
            current = self.now
            self.assertEqual(self.sample(frame)["observed_at"], current)

    def test_stationary_foreground_holds_old_marks_while_clear_tiles_change(self):
        annotated = self.blank.copy()
        cv2.line(annotated, (30, 96), (240, 96), (140, 40, 15), 2)
        self.settle(annotated)
        before = self.est.board.copy()
        for color in ((40, 70, 115), (252, 252, 252), (194, 194, 194)):
            with self.subTest(color=color):
                occluded = annotated.copy()
                cv2.rectangle(occluded, (10, 28), (150, 171), color, -1)
                cv2.line(occluded, (203, 64), (295, 64), (24, 26, 26), 2)
                result = self.settle(occluded)
                self.assertGreater(result["held"], 0.1)
                self.assertLess(result["held"], 0.85)
                np.testing.assert_array_equal(self.est.board[40:150, 32:138], before[40:150, 32:138])
                self.assertTrue(np.all(self.est.known == 255))
                self.assertLess(np.max(self.est.board[64, 208:287]), 65)
                self.settle(annotated)
                np.testing.assert_array_equal(self.est.board, before)

    def test_initially_occluded_tiles_stay_unknown_and_recover_after_reveal(self):
        occluded = self.blank.copy()
        occluded[:, :155] = (30, 52, 86)
        result = self.settle(occluded)
        self.assertGreater(result["held"], 0.30)
        self.assertTrue(np.all(self.est.known[:, :140] == 0))
        self.assertTrue(np.all(self.est.known[:, 180:] == 255))
        revealed = self.blank.copy()
        cv2.line(revealed, (25, 52), (105, 52), (175, 36, 32), 2)
        self.sample(revealed)
        self.assertEqual(self.est.known[52, 60], 0)
        self.settle(revealed)
        self.assertTrue(np.all(self.est.known == 255))
        self.assertGreater(int(self.est.board[52, 70, 0]) - int(self.est.board[52, 70, 2]), 90)

    def test_person_touching_board_rim_does_not_block_the_clear_side(self):
        frame = self.blank.copy()
        cv2.rectangle(frame, (0, 0), (319, 191), (40, 40, 40), 2)
        frame[60:, :100] = (45, 65, 145)
        self.settle(frame)
        self.assertEqual(self.est.known[100, 70], 0)
        self.assertTrue(np.all(self.est.known[20:170, 140:300] == 255))

    def test_moving_arm_and_single_frame_strokes_are_held(self):
        self.settle(self.blank)
        before = self.est.board.copy()
        for x in range(30, 220, 12):
            frame = self.blank.copy()
            cv2.line(frame, (x, 191), (x + 48, 66), (122, 154, 192), 22)
            cv2.line(frame, (20, x % 92 + 5), (100, x % 92 + 5), (20, 24, 30), 2)
            self.assertGreater(self.sample(frame)["held"], 0)
            np.testing.assert_array_equal(self.est.board, before)
        self.settle(self.blank)
        np.testing.assert_array_equal(self.est.board, before)

    def test_two_tiles_confirm_together_with_different_first_observations(self):
        self.settle(self.blank)
        first_seen = self.now
        left = self.blank.copy()
        cv2.line(left, (24, 42), (74, 42), (30, 30, 30), 2)
        both = left.copy()
        cv2.line(both, (214, 112), (264, 112), (30, 30, 30), 2)
        for frame, delay in ((left, 0), (both, 50), (both, 100)):
            self.est.update(frame, first_seen + delay)
        result = self.est.update(both, first_seen + 1000)
        self.assertLess(self.est.board[42, 50].max(), 60)
        self.assertLess(self.est.board[112, 240].max(), 60)
        self.assertEqual(result["observed_at"], first_seen)

    def test_jpeg_noise_does_not_repeatedly_commit_unchanged_writing(self):
        rng = np.random.default_rng(123)
        events = []
        for i in range(48):
            frame = self.blank.copy()
            if 8 <= i < 37:
                cv2.putText(frame, "B=2", (106, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.96,
                            (31, 43, 155), 2)
            if 17 <= i <= 24:
                cv2.rectangle(frame, (53, 64), (160, 144), (90, 150, 180), -1)
            noise = rng.integers(-3, 4, (192, 320, 1), dtype=np.int16)
            frame = np.clip(frame + noise, 0, 255).astype(np.uint8)
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75 + i % 9])
            self.assertTrue(ok)
            board, known = self.est.board.copy(), self.est.known.copy()
            result = self.sample(cv2.imdecode(jpeg, cv2.IMREAD_COLOR))
            if not np.array_equal(board, self.est.board) or not np.array_equal(known, self.est.known):
                events.append((result["observed_at"] - EPOCH) // 150)
            else:
                self.assertEqual(result["observed_at"], EPOCH + i * 150)
        # Only initial visibility, writing, and erasure actually changed state.
        self.assertEqual(events, [0, 8, 37])


if __name__ == "__main__":
    unittest.main()
