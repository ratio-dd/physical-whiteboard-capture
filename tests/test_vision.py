"""Deterministic camera-free tests through BoardEstimator's public interface."""

import unittest

import cv2
import numpy as np

from vision import BoardEstimator, enhance_board


FULL = ((0, 0), (1, 0), (1, 1), (0, 1))
SKEWED = ((0.27, 0.08), (0.90, 0.18), (0.97, 0.91), (0.05, 0.83))
EPOCH = 1_700_000_000_000


def jpeg_frame(frame, quality=86):
    ok, data = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


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

    def test_stable_window_keeps_persistent_stroke_tips_and_rejects_isolated_specks(self):
        self.settle(self.blank)
        first_seen = self.now
        # Each isolated pixel lies in a different changing tile. It is small
        # enough on its own to fall within the tile's JPEG/motion tolerance.
        specks = np.arange(35, 295, 30)
        tips, flashes = specks + 10, specks + 20
        for elapsed in (0, 67, 133, 200, 267, 333):
            frame = self.blank.copy()
            frame[82:85, 25:309] = 30
            if elapsed == 0:
                frame[88, specks] = 30
            else:
                frame[85, tips] = 30
            if elapsed == 333:
                frame[87, flashes] = 30
            result = self.est.update(frame, first_seen + elapsed)
            if elapsed < 300:
                self.assertGreater(self.est.board[83, 124].min(), 200)
        self.assertEqual(result["observed_at"], first_seen)
        self.assertGreater(self.est.board[88, specks].min(), 200)
        self.assertLess(self.est.board[85, tips].max(), 60)
        self.assertGreater(self.est.board[87, flashes].min(), 200)

    def test_old_stain_and_faint_ink_survive_new_writing_and_its_erasure(self):
        h, w = self.blank.shape[:2]
        y, x = np.indices((h, w))
        light = 232 + 7 * x / w - 12 * np.exp(-(((x - 245) / 41)**2 + ((y - 130) / 30)**2))
        paper = np.broadcast_to(light[..., None], self.blank.shape).astype(np.uint8, order="C")
        # The scuff and the old red mark are both present at first observation.
        paper[31:34, 40:125] -= 18
        paper[64:67, 40:125] -= np.array([24, 20, 2], np.uint8)
        paper[9:19, 6:290] = 255
        marked = paper.copy()
        marked[99:102, 40:125] -= 18
        base, marked = (jpeg_frame(cv2.GaussianBlur(f, (5, 5), 0.7), 78)
                        for f in (paper, marked))
        self.settle(base)
        before = self.est.board.copy()
        self.assertLess(np.median(before[32, 50:115, 1]), np.median(before[25, 50:115, 1]) - 10)
        self.assertGreater(self.sample(marked)["glare"], 0)
        self.settle(marked)
        diff = before[..., 1].astype(int) - self.est.board[..., 1]
        self.assertGreater(np.mean(diff[100, 50:115] > 8), 0.95)
        np.testing.assert_array_equal(self.est.board[:85], before[:85])
        np.testing.assert_array_equal(self.est.board[118:], before[118:])
        self.settle(base)
        np.testing.assert_array_equal(self.est.board, before)

    def test_short_false_stroke_and_paper_colored_cover_do_not_enter_history_state(self):
        marked = self.blank.copy()
        cv2.line(marked, (30, 95), (290, 95), (22, 22, 22), 3)
        self.settle(marked)
        baseline = self.est.board.copy()
        # The cover hides existing ink and carries a false narrow dark mark.
        covered = self.blank.copy()
        cv2.line(covered, (190, 48), (255, 48), (30, 30, 30), 2)
        for i in range(15):
            elapsed = round(i * 1000 / 15)
            self.est.update(covered if elapsed < 240 else marked, self.now + elapsed)
            np.testing.assert_array_equal(self.est.board, baseline)
        self.now += 1000
        self.settle(self.blank)
        self.assertGreater(self.est.board[95, 30:291].min(), 200)

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

    def test_fixed_glare_from_first_frame_preserves_readable_low_contrast_ink(self):
        rng = np.random.default_rng(13)
        h, w = self.blank.shape[:2]
        light = 232 + np.linspace(-6, 7, w)[None, :, None] + rng.normal(0, 1.25, (h, w, 1))
        frame = np.broadcast_to(light, (h, w, 3)).copy()
        frame[49:52, 45:146] -= (15, 15, 15)
        frame[71:74, 190:278] -= (24, 21, 2)
        frame[10:23, 12:301] = 255
        # Some ink remains readable inside the clipped stripe, in the same tile.
        frame[14:18, 40:110] = (235, 235, 235)
        frame[14:18, 175:245] = (223, 230, 250)
        decoded = jpeg_frame(np.clip(frame, 0, 255).astype(np.uint8), 88)
        plain, enhanced = enhance_board(decoded, contrast=1), enhance_board(decoded)
        y0, y1 = (cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(float) for im in (plain, enhanced))

        def separation(arr, row, cols):
            return np.median(arr[row - 8, cols]) - np.median(arr[row, cols])

        for row, cols in ((50, slice(55, 138)), (72, slice(202, 268))):
            self.assertGreater(separation(y1, row, cols), 1.6 * separation(y0, row, cols))

        def red_separation(im):
            red = im[..., 2].astype(float) - im[..., 0]
            return -separation(red, 72, slice(202, 268))

        self.assertGreater(red_separation(enhanced), 1.5 * red_separation(plain))
        noise = np.s_[120:170, 140:165]
        self.assertLessEqual(y1[noise].std(), y0[noise].std() + 0.5)

        self.assertGreater(self.sample(decoded)["glare"], 0.03)
        result = self.settle(decoded)
        self.assertLess(result["glare"], 0.08)
        self.assertTrue(np.all(self.est.known[12:21, 125:155] == 0))
        self.assertTrue(np.all(self.est.known[30:] == 255))
        for x in (75, 200):
            self.assertEqual(self.est.known[16, x], 255)
            np.testing.assert_array_equal(self.est.board[16, x], enhanced[16, x])
        np.testing.assert_array_equal(self.est.board[35:], enhanced[35:])
        self.settle(self.blank)
        self.assertTrue(np.all(self.est.known == 255))

    def test_fixed_bright_band_does_not_suppress_faint_red_transition(self):
        """The live-camera reproduction must advance past the blank frame.

        This is deliberately the weak RGB(225, 204, 206) mark and an
        independent clipped band from the live-evaluation fixture.  It guards
        the estimator boundary without weakening the global change thresholds.
        """
        base = np.full((480, 640, 3), 224, np.uint8)
        base[40:140, 300:420] = 252
        marked = base.copy()
        cv2.line(marked, (70, 220), (230, 270), (206, 204, 225), 8)
        blank = jpeg_frame(base, 98)
        red = jpeg_frame(marked, 98)
        estimator = BoardEstimator(FULL, 640, 480)
        for index in range(4):
            estimator.update(blank, EPOCH + index * 100)
        before = estimator.board.copy()
        for index in range(4):
            result = estimator.update(red, EPOCH + 400 + index * 100)
        self.assertFalse(np.array_equal(before, estimator.board))
        self.assertEqual(result["observed_at"], EPOCH + 400)
        self.assertGreater(int(np.count_nonzero(estimator.board[..., 2] > estimator.board[..., 0] + 2)), 100)

    def test_glare_on_known_ink_holds_then_confirms_real_erasure_at_reveal(self):
        blank = np.full_like(self.blank, 242)
        marked = blank.copy()
        cv2.line(marked, (30, 8), (168, 8), (25, 25, 25), 2)
        cv2.line(marked, (184, 8), (285, 8), (52, 56, 174), 2)
        ink = marked[..., 0] < 100
        band = marked.copy()
        band[:16, 20:300] = 255
        marked, band, blank = map(jpeg_frame, (marked, band, blank))
        self.settle(marked)
        original = self.est.board[ink].copy()
        for reveal_writing in (True, False):
            r = self.settle(band)
            self.assertAlmostEqual(r["glare"], 16 * 280 / ink.size, delta=0.003)
            self.assertTrue(np.all(self.est.known[ink] == 255))
            np.testing.assert_array_equal(self.est.board[ink], original)
            if reveal_writing:
                self.assertEqual(self.settle(marked)["glare"], 0)
                np.testing.assert_array_equal(self.est.board[ink], original)
        first_clear = self.now
        self.sample(blank)
        np.testing.assert_array_equal(self.est.board[ink], original)
        for _ in range(9):
            r = self.sample(blank)
            if np.min(self.est.board[ink]) > 215:
                break
        self.assertGreater(np.min(self.est.board[ink]), 215)
        self.assertEqual(r["observed_at"], first_clear)

    def test_bright_white_paper_and_gradual_highlight_keep_visible_writing(self):
        w = self.blank.shape[1]
        gradient = np.clip(np.linspace(223, 258, w), 0, 255).astype(np.uint8)
        graded = np.broadcast_to(gradient[None, :, None], self.blank.shape).copy()
        for paper in (np.full_like(self.blank, 255), graded):
            self.est = BoardEstimator(FULL, 320, 192)
            self.assertEqual(self.settle(jpeg_frame(paper))["glare"], 0)
            self.assertTrue(np.all(self.est.known == 255))
            self.assertGreater(int(self.est.board.min()), 216)
            # Include a line near the edge of pure white paper: ink alone is
            # insufficient evidence that the white area above it is a reflection.
            paper[19:22] -= np.array([30, 23, 3], np.uint8)
            r = self.settle(jpeg_frame(paper))
            self.assertEqual(r["glare"], 0)
            self.assertTrue(np.all(self.est.known == 255))
            self.assertGreater(int(self.est.board[20, 130, 2]) - int(self.est.board[20, 130, 0]), 30)

    def test_inclined_clipped_stripe_and_smooth_shadow_on_same_sheet(self):
        h, w = self.blank.shape[:2]
        y, x = np.indices((h, w))
        light = 232 - 31 * np.exp(-(((y - 108) / 37)**2 + ((x - 105) / 86)**2))
        frame = np.broadcast_to(light[..., None], self.blank.shape).astype(np.uint8, order="C")
        stripe = np.array([(13, 0), (43, 0), (298, 39), (298, 48)], np.int32)
        cv2.fillConvexPoly(frame, stripe, (255, 255, 255))
        decoded = jpeg_frame(frame)
        result = self.settle(decoded)
        self.assertTrue(0.02 < result["glare"] < 0.05)
        self.assertEqual(self.est.known[22, 165], 0)
        self.assertTrue(np.all(self.est.known[63:] == 255))
        shade = self.est.board[63:].astype(int)
        light_only = enhance_board(decoded, contrast=1)[63:].astype(int)
        self.assertLessEqual(np.percentile(np.abs(shade - light_only), 99), 2)
        self.assertGreater(shade.min(), 218)

    def _independent_jpeg(self, frame, index, quality=88):
        """Encode each sample separately so the test cannot reuse one JPEG."""
        rng = np.random.default_rng(8000 + index)
        noise = rng.integers(-2, 3, frame.shape[:2] + (1,), dtype=np.int16)
        noisy = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        ok, encoded = cv2.imencode(".jpg", noisy, [cv2.IMWRITE_JPEG_QUALITY, quality])
        self.assertTrue(ok)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        self.assertIsNotNone(decoded)
        return decoded

    def _weak_lines(self, base):
        frame = base.copy()
        # One-pixel marks close to the low-contrast reproduction used in the
        # capture fixtures. The second mark is chromatic without assuming a
        # particular ink hue in production.
        cv2.line(frame, (35, 65), (285, 65), (222, 222, 222), 1, cv2.LINE_AA)
        cv2.line(frame, (35, 95), (285, 95), (209, 209, 228), 1, cv2.LINE_AA)
        return frame

    def test_one_pixel_jpeg_ink_commits_and_erases_with_first_observation(self):
        """Independent Q88 samples must create and remove both weak marks."""
        base = np.full_like(self.blank, 230)
        self.est = BoardEstimator(FULL, 320, 192)
        for i in range(4):
            self.est.update(self._independent_jpeg(base, i), EPOCH + i * 150)
        before = self.est.board.copy()
        written = self._weak_lines(base)
        write_stamp = EPOCH + 1000
        write_changes = []
        for i in range(4):
            result = self.est.update(self._independent_jpeg(written, 20 + i), write_stamp + i * 150)
            write_changes.append((result, self.est.board.copy()))
        self.assertTrue(any(not np.array_equal(before, board) for _, board in write_changes))
        first_write = next(result for result, board in write_changes if not np.array_equal(before, board))
        self.assertEqual(first_write["observed_at"], write_stamp)
        committed = write_changes[-1][1]
        self.assertLess(np.median(committed[65, 55:265]), np.median(before[65, 55:265]) - 2)
        color_delta = (committed[95, 55:265, 2].astype(int)
                       - committed[95, 55:265, 0].astype(int))
        self.assertGreater(np.median(color_delta), 2)

        erase_stamp = write_stamp + 900
        erase_changes = []
        for i in range(4):
            result = self.est.update(self._independent_jpeg(base, 40 + i), erase_stamp + i * 150)
            erase_changes.append((result, self.est.board.copy()))
        final = erase_changes[-1][1]
        self.assertTrue(any(not np.array_equal(committed, board) for _, board in erase_changes))
        first_erase = next(result for result, board in erase_changes if not np.array_equal(committed, board))
        self.assertEqual(first_erase["observed_at"], erase_stamp)
        for y in (65, 95):
            restored = np.abs(final[y, 55:265].astype(int) - before[y, 55:265])
            self.assertLessEqual(np.percentile(restored.max(axis=1), 90), 2)
        np.testing.assert_array_equal(final[110:], before[110:])

    def test_pending_line_does_not_survive_empty_or_move(self):
        """An interrupted pending mark starts its window at its new position."""
        base = np.full_like(self.blank, 230)
        self.est = BoardEstimator(FULL, 320, 192)
        for i in range(4):
            self.est.update(self._independent_jpeg(base, i), EPOCH + i * 150)
        before = self.est.board.copy()
        old = base.copy()
        cv2.line(old, (24, 52), (124, 52), (222, 222, 222), 1, cv2.LINE_AA)
        moved = base.copy()
        cv2.line(moved, (180, 118), (280, 118), (209, 209, 228), 1, cv2.LINE_AA)
        first = EPOCH + 1000
        self.est.update(self._independent_jpeg(old, 10), first)
        self.est.update(self._independent_jpeg(base, 11), first + 150)
        np.testing.assert_array_equal(self.est.board, before)
        changes = []
        for i in range(4):
            result = self.est.update(self._independent_jpeg(moved, 30 + i), first + 300 + i * 150)
            changes.append((result, self.est.board.copy()))
        changed = next((result for result, board in changes if np.min(board[115:122, 180:280]) < 235), None)
        self.assertIsNotNone(changed)
        self.assertEqual(changed["observed_at"], first + 300)
        self.assertGreater(np.min(self.est.board[48:56, 24:124]), 215)

    def test_repeated_timestamp_does_not_count_as_three_samples(self):
        base = np.full_like(self.blank, 230)
        self.est = BoardEstimator(FULL, 320, 192)
        for i in range(4):
            self.est.update(self._independent_jpeg(base, i), EPOCH + i * 150)
        before = self.est.board.copy()
        written = base.copy()
        cv2.line(written, (35, 65), (285, 65), (222, 222, 222), 1, cv2.LINE_AA)
        repeated = EPOCH + 1000
        for i in range(5):
            self.est.update(self._independent_jpeg(written, 200 + i), repeated)
        np.testing.assert_array_equal(self.est.board, before)
        results = []
        for i in range(1, 4):
            result = self.est.update(self._independent_jpeg(written, 220 + i), repeated + 400 + i * 100)
            if i == 1:
                np.testing.assert_array_equal(self.est.board, before)
            results.append((result, self.est.board.copy()))
        first_change = next(result for result, board in results if not np.array_equal(board, before))
        self.assertEqual(first_change["observed_at"], repeated)

    def test_static_dust_and_light_photo_do_not_update_board(self):
        """A repeated dirty/glare frame is a negative, not a new writing event."""
        base = np.full_like(self.blank, 230)
        self.est = BoardEstimator(FULL, 320, 192)
        for i in range(4):
            self.est.update(self._independent_jpeg(base, i), EPOCH + i * 150)
        before = self.est.board.copy()
        dirty = base.copy()
        for x, y in ((24, 28), (89, 44), (152, 31), (264, 83), (301, 141)):
            dirty[y, x] = (150, 150, 150)
        dirty[12:22, 16:305] = 255
        timestamps = [EPOCH + 1000 + i * 150 for i in range(5)]
        results = [self.est.update(self._independent_jpeg(dirty, 100 + i), stamp)
                   for i, stamp in enumerate(timestamps)]
        np.testing.assert_array_equal(self.est.board, before)
        self.assertTrue(all(result["observed_at"] == stamp for result, stamp in zip(results, timestamps)))


if __name__ == "__main__":
    unittest.main()
