import sys
import tempfile
import unittest
from contextlib import closing
from unittest import mock
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
import ink
from history import History


def path(points, color="#202020", width=3, kind="line"):
    return {"points": points, "color": color, "width": width, "closed": False, "kind": kind}


def closed_path(points, color="#202020", width=3, kind="loop"):
    value = path(points, color, width, kind)
    value["closed"] = True
    return value


class InkTimelineTests(unittest.TestCase):
    def build_from_paths(self, sequence, shape=(24, 32), stamps=None, created_at=1000):
        """Build a timeline with explicit seq-to-path fixture data."""
        extractor = lambda: lambda board, known: sequence[int(board[0, 0, 0])]
        with mock.patch.object(ink, "_geometry_extractor", extractor):
            with tempfile.TemporaryDirectory() as temp:
                with closing(History(Path(temp) / "ink.sqlite3")) as history:
                    board = np.full((*shape, 3), 255, np.uint8)
                    board[0, 0] = (0, 0, 0)
                    known = np.full(shape, 255, np.uint8)
                    session = history.create(board, known, created_at)
                    for seq, stamp in enumerate(stamps or range(created_at + 100, created_at + 100 * len(sequence), 100), 1):
                        if seq >= len(sequence):
                            break
                        board = board.copy()
                        board[0, 0] = (seq, 0, 0)
                        history.record(session, board, known, stamp, stamp + 10)
                    return ink.build_ink(history, session)

    def test_stable_ids_updates_and_removes(self):
        result = self.build_from_paths([
            [], [path([[2, 5], [10, 5]])], [path([[2, 5], [14, 5]])], []], shape=(20, 20))
        changes = [change for event in result["events"] for change in event["changes"]]
        adds = [change for change in changes if change["op"] == "add"]
        updates = [change for change in changes if change["op"] == "update"]
        removes = [change for change in changes if change["op"] == "remove"]
        self.assertEqual(len(adds), 1)
        self.assertEqual(adds[0]["id"], updates[0]["id"])
        self.assertEqual(removes[0]["id"], adds[0]["id"])

    def test_empty_session_has_seq_boundary(self):
        with tempfile.TemporaryDirectory() as temp:
            with closing(History(Path(temp) / "ink.sqlite3")) as history:
                board = np.full((8, 9, 3), 255, np.uint8)
                known = np.full((8, 9), 255, np.uint8)
                session = history.create(board, known, 1000)
                result = ink.build_ink(history, session)
        self.assertEqual(result["through_seq"], 0)
        self.assertEqual(result["width"], 9)
        self.assertEqual(result["height"], 8)
        self.assertEqual(result["events"], [])

    def test_direction_is_not_claimed_from_point_order(self):
        result = self.build_from_paths([[], [path([[2, 5], [10, 5]])], [path([[2, 5], [14, 5]])]], shape=(10, 20))
        updates = [c for e in result["events"] for c in e["changes"] if c["op"] == "update"]
        self.assertEqual(updates[0]["stroke"]["direction_source"], "inferred")

    def test_reversed_geometry_order_does_not_create_update(self):
        result = self.build_from_paths([[], [path([[2, 5], [14, 5]])], [path([[14, 5], [2, 5]])]], shape=(10, 20))
        changes = [c for e in result["events"] for c in e["changes"]]
        self.assertEqual([c["op"] for c in changes], ["add"])

    def test_near_equivalent_identity_is_declared_ambiguous(self):
        result = self.build_from_paths([
            [],
            [path([[2, 5], [14, 5]]), path([[2, 5], [14, 5]])],
            [path([[2, 5], [14, 5]])],
        ])
        changes = [change for event in result["events"] for change in event["changes"]]
        self.assertEqual(sum(change["op"] == "add" for change in changes), 3)
        self.assertTrue(any(change.get("stroke", {}).get("identity_ambiguous") for change in changes[2:]))

    def test_color_noise_keeps_local_extension_identity(self):
        result = self.build_from_paths([
            [],
            [path([[3, 5], [12, 5]], color="#202020")],
            [path([[3, 5], [16, 5]], color="#212121")],
        ])
        changes = [change for event in result["events"] for change in event["changes"]]
        self.assertEqual([change["op"] for change in changes], ["add", "update"])
        self.assertEqual(changes[0]["id"], changes[1]["id"])

    def test_shortening_a_path_is_not_reported_as_forward_writing(self):
        result = self.build_from_paths([
            [],
            [path([[3, 5], [18, 5]])],
            [path([[3, 5], [12, 5]])],
        ])
        update = [change for event in result["events"] for change in event["changes"] if change["op"] == "update"][0]
        self.assertEqual(update["stroke"]["direction"], "unknown")
        self.assertEqual(update["stroke"]["direction_source"], "unknown")

    def test_adjacent_lines_keep_one_to_one_ids_after_local_edit(self):
        result = self.build_from_paths([
            [],
            [path([[3, 5], [14, 5]]), path([[3, 8], [14, 8]])],
            [path([[3, 5], [18, 5]]), path([[3, 8], [14, 8]])],
            [path([[3, 5], [18, 5]])],
        ])
        changes = [change for event in result["events"] for change in event["changes"]]
        adds = [change for change in changes if change["op"] == "add"]
        removes = [change for change in changes if change["op"] == "remove"]
        updates = [change for change in changes if change["op"] == "update"]
        self.assertEqual(len(adds), 2)
        self.assertEqual(len(removes), 1)
        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0]["id"], adds[0]["id"])
        self.assertEqual(removes[0]["id"], adds[1]["id"])

    def test_identical_style_different_positions_are_distinct_strokes(self):
        result = self.build_from_paths([
            [],
            [path([[3, 4], [10, 4]]), path([[3, 14], [10, 14]])],
            [path([[3, 4], [12, 4]]), path([[3, 14], [12, 14]])],
        ])
        changes = [change for event in result["events"] for change in event["changes"]]
        adds = [change for change in changes if change["op"] == "add"]
        updates = [change for change in changes if change["op"] == "update"]
        self.assertEqual(len({change["id"] for change in adds}), 2)
        self.assertEqual({change["id"] for change in updates}, {change["id"] for change in adds})

    def test_closed_path_reversal_preserves_identity_and_style(self):
        loop = [[4, 4], [12, 4], [12, 12], [4, 12]]
        result = self.build_from_paths([
            [],
            [closed_path(loop, width=5)],
            [closed_path(list(reversed(loop)), width=5)],
        ])
        changes = [change for event in result["events"] for change in event["changes"]]
        self.assertEqual([change["op"] for change in changes], ["add"])
        self.assertEqual(changes[0]["stroke"]["kind"], "loop")

    def test_existing_geometry_at_seq_zero_is_replayable(self):
        result = self.build_from_paths([
            [path([[4, 4], [12, 4]])],
            [path([[4, 4], [12, 4]])],
        ])
        visible = [change for event in result["events"] if event["seq"] == 0 for change in event["changes"]]
        self.assertEqual(len(visible), 1)
        self.assertEqual(visible[0]["op"], "add")

    def test_synthetic_parallel_raster_only_extends_upper_line(self):
        """Actual raster extraction must keep the lower line stable."""
        with tempfile.TemporaryDirectory() as temp:
            with closing(History(Path(temp) / "ink.sqlite3")) as history:
                blank = np.full((120, 180, 3), 255, np.uint8)
                known = np.full((120, 180), 255, np.uint8)
                session = history.create(blank, known, 1000)
                first = blank.copy()
                cv2.line(first, (20, 40), (100, 40), (30, 30, 30), 5)
                cv2.line(first, (20, 52), (100, 52), (30, 30, 30), 5)
                upper = blank.copy()
                cv2.line(upper, (20, 40), (130, 40), (30, 30, 30), 5)
                cv2.line(upper, (20, 52), (100, 52), (30, 30, 30), 5)
                history.record(session, first, known, 1100, 1100)
                history.record(session, upper, known, 1200, 1200)
                result = ink.build_ink(history, session)
        events = {event["seq"]: event["changes"] for event in result["events"]}
        first_adds = [change for change in events[1] if change["op"] == "add"]
        self.assertEqual(len(first_adds), 2)
        upper_id = min(first_adds, key=lambda change: change["stroke"]["points"][0][1])["id"]
        lower_id = max(first_adds, key=lambda change: change["stroke"]["points"][0][1])["id"]
        updates = [change for change in events[2] if change["op"] == "update"]
        self.assertEqual([change["id"] for change in updates], [upper_id])
        self.assertNotIn(lower_id, {change["id"] for change in updates})
        upper_points = next(change for change in updates if change["id"] == upper_id)["stroke"]["points"]
        self.assertGreater(max(point[0] for point in upper_points), 110)

    def test_synthetic_partial_erase_does_not_remove_parallel_sibling(self):
        with tempfile.TemporaryDirectory() as temp:
            with closing(History(Path(temp) / "ink.sqlite3")) as history:
                blank = np.full((120, 180, 3), 255, np.uint8)
                known = np.full((120, 180), 255, np.uint8)
                session = history.create(blank, known, 1000)
                first = blank.copy()
                cv2.line(first, (20, 40), (100, 40), (30, 30, 30), 5)
                cv2.line(first, (20, 52), (100, 52), (30, 30, 30), 5)
                erased = first.copy()
                cv2.line(erased, (52, 52), (68, 52), (255, 255, 255), 9)
                history.record(session, first, known, 1100, 1100)
                history.record(session, erased, known, 1200, 1200)
                result = ink.build_ink(history, session)
        first_changes = result["events"][0]["changes"]
        second_changes = result["events"][1]["changes"]
        first_ids = {change["id"] for change in first_changes}
        self.assertEqual(len(first_ids), 2)
        self.assertFalse(any(change["op"] == "remove" for change in second_changes))
        second_ids = [change["id"] for change in second_changes]
        self.assertEqual(len(second_ids), len(set(second_ids)))
        self.assertTrue(first_ids & set(second_ids), "the surviving lower line must retain its ID")

    def test_real_seq_zero_known_black_line_extends_without_direction_claim(self):
        with tempfile.TemporaryDirectory() as temp:
            with closing(History(Path(temp) / "ink.sqlite3")) as history:
                board = np.full((80, 120, 3), 255, np.uint8)
                known = np.full((80, 120), 255, np.uint8)
                known[:, 72:] = 0
                cv2.line(board, (15, 30), (65, 30), (30, 30, 30), 5)
                session = history.create(board, known, 1000)
                extended = board.copy()
                visible = np.full_like(known, 255)
                cv2.line(extended, (15, 30), (95, 30), (30, 30, 30), 5)
                history.record(session, extended, visible, 1100, 1100)
                result = ink.build_ink(history, session)
        seq_zero = [change for event in result["events"] if event["seq"] == 0 for change in event["changes"]]
        self.assertTrue(seq_zero and all(change["op"] == "add" for change in seq_zero))
        later = [change for event in result["events"] if event["seq"] == 1 for change in event["changes"]]
        self.assertTrue(later)
        self.assertTrue(all(change["stroke"]["direction"] == "unknown" for change in later))

    def test_new_branch_after_unchanged_centerline_forces_unknown_direction(self):
        center = path([[3, 5], [14, 5]])
        extended = path([[3, 5], [18, 5]])
        branched = path([[3, 5], [18, 5]])
        branched["centerlines"] = [
            {"points": [[3, 5], [18, 5]], "closed": False},
            {"points": [[14, 5], [14, 11]], "closed": False},
        ]
        result = self.build_from_paths([[], [center], [extended], [branched]])
        updates = [change for event in result["events"] for change in event["changes"] if change["op"] == "update"]
        self.assertTrue(updates)
        self.assertEqual(updates[-1]["stroke"]["direction"], "unknown")

    def test_reverse_direction_and_style_change_preserve_new_vertex_order(self):
        original = path([[3, 5], [14, 5]], color="#202020")
        extended = path([[3, 5], [18, 5]], color="#202020")
        changed = path([[18, 5], [3, 5]], color="#aa3322")
        result = self.build_from_paths([[], [original], [extended], [changed]])
        changes = [change for event in result["events"] for change in event["changes"]]
        latest = changes[-1]
        self.assertEqual(latest["stroke"]["points"], [[18.0, 5.0], [3.0, 5.0]])
        self.assertEqual(latest["stroke"]["color"], "#aa3322")
        self.assertEqual(latest["stroke"]["direction"], "reverse")

    def test_long_sequence_reconstructs_each_confirmed_seq_across_snapshot_boundary(self):
        sequence = [[]]
        for seq in range(1, 35):
            sequence.append([path([[3, 5], [12 + seq, 5]])])
        result = self.build_from_paths(sequence, shape=(40, 60))
        self.assertEqual(result["through_seq"], 34)
        self.assertEqual([event["seq"] for event in result["events"]], list(range(1, 35)))
        current = {}
        stable_id = None
        for event in result["events"]:
            for change in event["changes"]:
                if change["op"] == "remove":
                    current.pop(change["id"], None)
                else:
                    current[change["id"]] = change["stroke"]
            self.assertEqual(len(current), 1)
            stroke_id = next(iter(current))
            stable_id = stable_id or stroke_id
            self.assertEqual(stroke_id, stable_id)
            self.assertEqual(current[stroke_id]["points"], sequence[event["seq"]][0]["points"])

    def test_identical_coincident_copies_stay_stable_while_count_agrees(self):
        line = path([[5, 5], [15, 5]])
        result = self.build_from_paths([[line, line], [line, line]])
        self.assertEqual([event["seq"] for event in result["events"]], [0])
        changes = result["events"][0]["changes"]
        self.assertEqual([change["op"] for change in changes], ["add", "add"])
        self.assertNotEqual(changes[0]["id"], changes[1]["id"])

    def test_reappearing_geometry_reuses_id_only_if_board_and_mask_stayed_unchanged(self):
        stroke = path([[12, 8], [27, 8]])
        def extract(board, known):
            return [stroke] if board[0, 0, 0] in (0, 3) else []

        for interruption in ("detection", "erase", "visibility"):
            with self.subTest(interruption=interruption), tempfile.TemporaryDirectory() as temp:
                with closing(History(Path(temp) / "ink.sqlite3")) as history:
                    board = np.full((20, 40, 3), 255, np.uint8)
                    board[0, 0] = 0
                    board[8, 12:28] = 0
                    known = np.full(board.shape[:2], 255, np.uint8)
                    session = history.create(board, known, 1000)
                    for seq in (1, 2, 3):
                        board[0, 0] = seq
                        board[8, 12:28] = 255 if seq == 1 and interruption == "erase" else 0
                        known[8, 12:28] = 0 if seq == 1 and interruption == "visibility" else 255
                        history.record(session, board, known, 1000 + seq, 1000 + seq)
                    with mock.patch.object(ink, "_geometry_extractor", return_value=extract):
                        timeline = ink.build_ink(history, session)
                changes = [change for event in timeline["events"] for change in event["changes"]]
                self.assertEqual([c["op"] for c in changes], ["add", "remove", "add"])
                old, new = changes[0]["stroke"], changes[-1]["stroke"]
                self.assertEqual(old["id"] == new["id"], interruption == "detection")
                self.assertEqual(new["first_seq"], 0 if interruption == "detection" else 3)

    def test_overlapping_objects_with_distinct_style_keep_ids(self):
        result = self.build_from_paths([
            [path([[5, 5], [15, 5]], width=3), path([[5, 5], [15, 5]], width=5)],
            [path([[5, 5], [18, 5]], width=3), path([[5, 5], [18, 5]], width=5)],
        ])
        changes = [change for event in result["events"] for change in event["changes"]]
        adds = [change for change in changes if change["op"] == "add"]
        updates = [change for change in changes if change["op"] == "update"]
        self.assertEqual(len(adds), 2)
        self.assertEqual({change["id"] for change in updates}, {change["id"] for change in adds})

    def test_overlapping_colors_swapped_in_extractor_order_keep_ids(self):
        red = path([[5, 5], [15, 5]], color="#cc2211")
        blue = path([[5, 5], [15, 5]], color="#1122cc")
        result = self.build_from_paths([[red, blue], [path([[5, 5], [18, 5]], color="#1122cc"),
                                        path([[5, 5], [18, 5]], color="#cc2211")]])
        changes = [change for event in result["events"] for change in event["changes"]]
        adds = {change["stroke"]["color"]: change["id"] for change in changes if change["op"] == "add"}
        updates = {change["stroke"]["color"]: change["id"] for change in changes if change["op"] == "update"}
        self.assertEqual(adds, updates)

    def test_optional_geometry_fields_survive_add_and_update(self):
        outer = [[3, 3], [17, 3], [17, 17], [3, 17], [3, 3]]
        hole = [[7, 7], [13, 7], [13, 13], [7, 13], [7, 7]]
        first = path([[4, 4], [16, 16]])
        first.update({"outline": [outer, hole],
                      "centerlines": [{"points": [[4, 4], [16, 16]], "closed": False}],
                      "layers": [{"outline": [outer, hole], "color": "#cc2211"},
                                 {"outline": [outer], "color": "#1122cc"}],
                      "appearance": {"opacity": 0.7, "cap": "round"}, "area": 42.5,
                      "evidence": "visible_component"})
        changed = dict(first)
        changed["layers"] = [{"outline": [outer, hole], "color": "#33aa55"},
                              {"outline": [outer], "color": "#1122cc"}]
        result = self.build_from_paths([[first], [changed]])
        changes = [change for event in result["events"] for change in event["changes"]]
        self.assertEqual([change["op"] for change in changes], ["add", "update"])
        self.assertEqual(changes[0]["id"], changes[1]["id"])
        for change, original in zip(changes, (first, changed)):
            self.assertEqual({k: change["stroke"][k] for k in original}, original)


if __name__ == "__main__":
    unittest.main()
