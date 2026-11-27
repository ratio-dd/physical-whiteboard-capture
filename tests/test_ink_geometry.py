import unittest
import cv2
import numpy as np

from ink_geometry import extract_paths, make_path_extractor


def render_layers(paths, size):
    """Fill editable rings at 3× before reducing to board pixels, like an SVG."""
    h, w = size
    canvas = np.full((3*h, 3*w, 3), 255, np.uint8)
    for path in paths:
        for layer in path["layers"]:
            rings = [np.rint(np.array(ring)*3 - 0.5).astype(np.int32)
                     for ring in layer["outline"]]
            cv2.fillPoly(canvas, rings, tuple(bytes.fromhex(layer["color"][1:]))[::-1])
    return cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA)


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

    def test_global_coordinates_non_square_board_and_holes(self):
        board = np.full((127, 431, 3), 241, np.uint8)
        cv2.rectangle(board, (62, 18), (104, 38), (52,)*3, 2)
        cv2.ellipse(board, (302, 84), (23, 11), 0, 0, 360, (88, 93, 206), 2)
        paths = extract_paths(board, np.full(board.shape[:2], 255, np.uint8))
        image = render_layers(paths, board.shape[:2])
        self.assertEqual(len(paths), 2)
        for path, (cx, cy) in zip(paths, ((83,28), (302,84))):
            points = np.vstack(path["outline"])
            center = (points.min(axis=0) + points.max(axis=0)) / 2
            np.testing.assert_allclose(center, [cx+0.5, cy+0.5], atol=1)
            self.assertGreaterEqual(len(path["outline"]), 2)  # Outer boundary and hole.
            self.assertTrue(np.all(image[cy, cx] == 255))
        b, g, r = cv2.split(image.astype(int))
        self.assertGreater(int(np.count_nonzero((r-g > 30) & (r-b > 30))), 150)

    def test_connected_inks_retain_both_colors_and_taper(self):
        cv2.fillConvexPoly(self.board, np.array([[35, 117], [285, 111], [285, 124]], np.int32),
                           (50, 50, 50))
        cv2.line(self.board, (155, 55), (155, 175), (151, 155, 217), 3)
        paths = extract_paths(self.board, self.known)
        image = render_layers(paths, self.known.shape)
        self.assertTrue(all(s["centerlines"] and s["outline"] for s in paths))
        self.assertLess(int(image[117, 238].max()), 125)
        self.assertGreater(int(image[149, 155, 2]) - int(image[149, 155, 1]), 35)
        thin = np.count_nonzero(image[94:140, 70, 0] < 180)
        thick = np.count_nonzero(image[94:140, 264, 0] < 180)
        self.assertGreater(thin, 0)
        self.assertGreater(thick, 2*thin)

    def test_gradient_flat_tiles_and_glare_shoulders_do_not_turn_into_ink(self):
        y, x = np.indices(self.known.shape)
        diffuse = 228 + 9*x/360 - 18*np.exp(-(((x-218)/35)**2 + ((y-164)/27)**2))
        board = np.broadcast_to(diffuse[..., None], self.board.shape).astype(np.uint8, order="C")
        board[85:99, 78:104] -= 16
        board[31:38, 40:290] -= 11
        board[38:44, 40:290] = 255
        board[44:50, 40:290] -= 11
        decoded = cv2.imdecode(cv2.imencode(".jpg", board, [cv2.IMWRITE_JPEG_QUALITY, 82])[1], 1)
        self.assertEqual(extract_paths(decoded, self.known), [])
        # Real ink may be oblique or close to the physical top edge.
        cv2.line(decoded, (12, 8), (95, 8), (175,175,175), 1)
        cv2.line(decoded, (188, 95), (235, 148), (147,147,147), 1)
        visible = render_layers(extract_paths(decoded, self.known), self.known.shape)
        self.assertGreater(np.mean(np.min(visible[6:11, 16:91, 0], axis=0) < 208), 0.95)
        self.assertLess(int(visible[122,212].min()), 215)

    def test_parallel_real_stroke_next_to_slanted_highlight_survives(self):
        board = np.full((180, 370, 3), 231, np.uint8)
        known = np.full(board.shape[:2], 255, np.uint8)
        cv2.line(board, (45, 34), (270, 72), (218,)*3, 13, cv2.LINE_AA)
        cv2.line(board, (45, 34), (270, 72), (255,)*3, 5, cv2.LINE_AA)

        def jpeg(image):
            return cv2.imdecode(cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 84])[1], 1)

        self.assertEqual(extract_paths(jpeg(board), known), [])
        cv2.line(board, (65, 49), (223, 76), (150,)*3, 2, cv2.LINE_AA)
        visible = render_layers(extract_paths(jpeg(board), known), known.shape)
        xs = np.arange(70, 221)
        ys = np.rint(49 + (xs - 65)*27/158).astype(int)
        self.assertGreater(np.mean(visible[ys, xs, 0] < 190), 0.95)

    def test_horizontal_glare_touching_unknown_paper_keeps_a_parallel_pen(self):
        board = np.full((180, 410, 3), 231, np.uint8)
        known = np.full(board.shape[:2], 255, np.uint8)
        cv2.line(board, (35, 64), (380, 72), (218,)*3, 13, cv2.LINE_AA)
        cv2.line(board, (35, 64), (380, 72), (255,)*3, 5, cv2.LINE_AA)
        known[:73, 375:] = 0
        board[known == 0] = 255

        def paths():
            encoded = cv2.imencode(".jpg", board, [cv2.IMWRITE_JPEG_QUALITY, 84])[1]
            return extract_paths(cv2.imdecode(encoded, 1), known)

        self.assertEqual(paths(), [])
        cv2.line(board, (55, 77), (214, 80), (140,)*3, 2, cv2.LINE_AA)
        visible = render_layers(paths(), known.shape)
        xs = np.arange(58, 211)
        ys = np.rint(77 + (xs - 55)*3/159).astype(int)
        self.assertGreater(np.mean(visible[ys, xs, 0] < 190), 0.95)

    def test_session_cache_uses_local_evidence_and_global_coordinates(self):
        cached = make_path_extractor()
        board = self.board.copy()
        cv2.line(board, (72, 76), (195, 80), (96, 105, 176), 3)
        original = extract_paths(board, self.known)
        self.assertEqual(cached(board, self.known), original)
        altered_result = cached(board, self.known)
        altered_result[0]["layers"][0]["outline"][0][0][0] = -42
        self.assertEqual(cached(board, self.known), original)
        for dx, color, paper, cut in ((0, (96,105,176), 219, False),
                                      (23, (96,105,176), 238, False),
                                      (23, (176,105,96), 238, False),
                                      (23, (176,105,96), 238, True)):
            frame = np.full_like(board, paper)
            cv2.line(frame, (72+dx, 76), (195+dx, 80), color, 3)
            known = self.known.copy()
            if cut:
                known[:, 125:135] = 0
            self.assertEqual(cached(frame, known), extract_paths(frame, known))

    def test_neural_recovers_measured_red_gap_and_keeps_empty_gap_white(self):
        known = np.full((85, 180), 255, np.uint8)
        board = np.full((*known.shape, 3), 238, np.uint8)
        cv2.line(board, (20, 41), (140, 41), (209, 209, 228), 2, cv2.LINE_AA)
        board[:, 74:83] = 238
        extractor = make_path_extractor(mode="neural")
        for contrast in (0, 3):
            if contrast:
                cv2.line(board, (74, 41), (82, 41), (235, 235, 237), 2, cv2.LINE_AA)
            classic = render_layers(extract_paths(board, known), known.shape)
            paths = extract_paths(board, known, mode="neural")
            self.assertEqual(extractor(board, known), paths)
            neural = render_layers(paths, known.shape)
            np.testing.assert_array_equal(neural[41, 35], classic[41, 35])
            self.assertEqual(tuple(classic[41, 78]), (255, 255, 255))
            if contrast:
                blue, green, red = map(int, neural[41, 78])
                self.assertLess(blue, 235)
                self.assertGreater(red, max(blue, green) + 10)
            else:
                self.assertTrue(np.all(neural[40:43, 76:81] == 255))

    def test_neural_and_classic_ignore_arbitrary_unknown_storage(self):
        known = self.known.copy()
        known[:, 180:] = 0
        cv2.line(self.board, (110, 42), (195, 78), (180, 180, 202), 2)
        noise = np.random.default_rng(44).integers(0, 256, self.board.shape, dtype=np.uint8)
        modified = np.where(known[..., None] == 0, noise, self.board)
        for mode in ("classic", "neural"):
            extractor = make_path_extractor(mode=mode)
            self.assertEqual(extractor(self.board, known), extractor(modified, known))
        with self.assertRaises(ValueError):
            make_path_extractor(mode="unknown")


if __name__ == "__main__":
    unittest.main()
