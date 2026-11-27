import unittest
import cv2
import numpy as np

from ink_geometry import extract_paths


class InkGeometryTest(unittest.TestCase):
    def setUp(self):
        self.board = np.full((240, 360, 3), 238, np.uint8)
        self.known = np.full((240, 360), 255, np.uint8)

    def test_extracts_dark_line_and_colored_line(self):
        cv2.line(self.board, (35, 60), (300, 70), (45, 45, 45), 5)
        cv2.line(self.board, (80, 160), (240, 205), (50, 70, 210), 5)
        paths = extract_paths(self.board, self.known)
        self.assertGreaterEqual(len(paths), 2)
        self.assertTrue(any(path["kind"] == "line" for path in paths))
        self.assertTrue(any(path["color"] != "#000000" for path in paths))
        for path in paths:
            self.assertGreaterEqual(path["width"], 1)
            self.assertEqual(path["evidence"], "visible_component")

    def test_loop_is_closed_and_unknown_is_ignored(self):
        cv2.rectangle(self.board, (90, 80), (230, 180), (35, 35, 35), 5)
        self.known[:70] = 0
        paths = extract_paths(self.board, self.known)
        self.assertTrue(any(path["closed"] for path in paths))
        self.assertTrue(all(not any(y < 70 for _, y in path["points"]) for path in paths))

    def test_noise_and_blank_board_do_not_create_paths(self):
        noisy = self.board.copy()
        rng = np.random.default_rng(4)
        noisy[:] = np.clip(noisy.astype(int) + rng.normal(0, 2, noisy.shape), 0, 255).astype(np.uint8)
        self.assertEqual(extract_paths(noisy, self.known), [])

    def test_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError): extract_paths(self.board[..., 0], self.known)
        with self.assertRaises(ValueError): extract_paths(self.board, np.ones(self.known.shape, np.uint8))

    def test_curved_and_branching_marks_are_centerlines(self):
        arc = np.array([[60 + int(40 * np.cos(t)), 120 + int(40 * np.sin(t))]
                        for t in np.linspace(np.pi, 1.5 * np.pi, 24)], np.int32)
        cv2.polylines(self.board, [arc], False, (35, 35, 35), 5)
        cv2.line(self.board, (190, 80), (260, 160), (35, 35, 35), 5)
        cv2.line(self.board, (260, 80), (190, 160), (35, 35, 35), 5)
        paths = extract_paths(self.board, self.known)
        self.assertGreaterEqual(len(paths), 2)
        self.assertTrue(any(len(path["points"]) > 4 for path in paths))
        self.assertTrue(all(path["kind"] in {"line", "polyline", "loop"} for path in paths))

    def test_unknown_stripe_cuts_the_reported_geometry(self):
        cv2.line(self.board, (30, 30), (330, 30), (25, 25, 25), 5)
        self.known[20:40, 165:195] = 0
        paths = extract_paths(self.board, self.known)
        self.assertTrue(paths)
        for path in paths:
            self.assertFalse(any(165 <= x < 195 and 20 <= y <= 40
                                 for x, y in path["points"]))

    def test_long_marks_have_local_width_not_length_width(self):
        for start, end in [((3, 30), (350, 30)), ((40, 3), (40, 210)), ((3, 30), (350, 80))]:
            board = np.full((240, 360, 3), 238, np.uint8)
            cv2.line(board, start, end, (35, 35, 35), 7)
            paths = extract_paths(board, self.known)
            self.assertTrue(paths)
            self.assertLess(max(path["width"] for path in paths), 12)

    def test_outer_two_pixel_rail_is_suppressed_but_inward_mark_survives(self):
        board = np.full((240, 360, 3), 238, np.uint8)
        cv2.line(board, (359, 90), (359, 220), (30, 30, 30), 2)
        cv2.line(board, (358, 100), (330, 180), (30, 30, 30), 4)
        paths = extract_paths(board, self.known)
        self.assertTrue(paths)
        self.assertTrue(any(min(x for x, _ in p["points"]) < 350 for p in paths))
        self.assertFalse(any(all(x >= 357 for x, _ in p["points"]) for p in paths))


if __name__ == "__main__":
    unittest.main()
