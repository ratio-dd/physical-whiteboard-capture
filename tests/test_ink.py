import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
import ink
from history import History


def path(points, color="#202020", width=3, kind="line"):
    return {"points": points, "color": color, "width": width, "closed": False, "kind": kind}


class InkTimelineTests(unittest.TestCase):
    def test_stable_ids_updates_and_removes(self):
        sequence = [
            [],
            [path([[2, 5], [10, 5]])],
            [path([[2, 5], [14, 5]])],
            [],
        ]
        original = ink._geometry_extractor
        ink._geometry_extractor = lambda: (lambda board, known: sequence.pop(0))
        try:
            with tempfile.TemporaryDirectory() as temp:
                history = History(Path(temp) / "ink.sqlite3")
                board = np.full((20, 20, 3), 255, np.uint8)
                known = np.full((20, 20), 255, np.uint8)
                session = history.create(board, known, 1000)
                for stamp in (1100, 1200, 1300, 1400):
                    board[stamp // 100 - 10, 0] = 0
                    history.record(session, board, known, stamp, stamp + 10)
                result = ink.build_ink(history, session)
            changes = [change for event in result["events"] for change in event["changes"]]
            adds = [change for change in changes if change["op"] == "add"]
            updates = [change for change in changes if change["op"] == "update"]
            removes = [change for change in changes if change["op"] == "remove"]
            self.assertEqual(len(adds), 1)
            self.assertEqual(adds[0]["id"], updates[0]["id"])
            self.assertEqual(removes[0]["id"], adds[0]["id"])
        finally:
            ink._geometry_extractor = original

    def test_empty_session_has_seq_boundary(self):
        with tempfile.TemporaryDirectory() as temp:
            history = History(Path(temp) / "ink.sqlite3")
            board = np.full((8, 9, 3), 255, np.uint8)
            known = np.full((8, 9), 255, np.uint8)
            session = history.create(board, known, 1000)
            result = ink.build_ink(history, session)
        self.assertEqual(result["through_seq"], 0)
        self.assertEqual(result["width"], 9)
        self.assertEqual(result["height"], 8)
        self.assertEqual(result["events"], [])

    def test_direction_is_not_claimed_from_point_order(self):
        sequence = [[path([[2, 5], [10, 5]])], [path([[2, 5], [14, 5]])]]
        original = ink._geometry_extractor
        ink._geometry_extractor = lambda: (lambda board, known: sequence.pop(0))
        try:
            with tempfile.TemporaryDirectory() as temp:
                history = History(Path(temp) / "ink.sqlite3")
                board = np.full((10, 20, 3), 255, np.uint8)
                known = np.full((10, 20), 255, np.uint8)
                session = history.create(board, known, 1000)
                board[1, 0] = 0
                history.record(session, board, known, 1100, 1110)
                board[2, 0] = 0
                history.record(session, board, known, 1200, 1210)
                result = ink.build_ink(history, session)
            updates = [c for e in result["events"] for c in e["changes"] if c["op"] == "update"]
            self.assertEqual(updates[0]["stroke"]["direction_source"], "inferred")
        finally:
            ink._geometry_extractor = original

    def test_reversed_geometry_order_does_not_create_update(self):
        sequence = [[path([[2, 5], [14, 5]])], [path([[14, 5], [2, 5]])]]
        original = ink._geometry_extractor
        ink._geometry_extractor = lambda: (lambda board, known: sequence.pop(0))
        try:
            with tempfile.TemporaryDirectory() as temp:
                history = History(Path(temp) / "ink.sqlite3")
                board = np.full((10, 20, 3), 255, np.uint8)
                known = np.full((10, 20), 255, np.uint8)
                session = history.create(board, known, 1000)
                board[1, 0] = 0
                history.record(session, board, known, 1100, 1110)
                board[2, 0] = 0
                history.record(session, board, known, 1200, 1210)
                result = ink.build_ink(history, session)
            changes = [c for e in result["events"] for c in e["changes"]]
            self.assertEqual([c["op"] for c in changes], ["add"])
        finally:
            ink._geometry_extractor = original

    def test_near_equivalent_identity_is_declared_ambiguous(self):
        old = [path([[2, 5], [14, 5]]), path([[2, 5], [14, 5]])]
        new = [path([[2, 5], [14, 5]])]
        matching, ambiguous = ink._match(old, new, 20.0)
        self.assertEqual(matching, {})
        self.assertEqual(ambiguous, {0})


if __name__ == "__main__":
    unittest.main()
