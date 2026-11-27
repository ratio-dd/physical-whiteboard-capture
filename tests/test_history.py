import base64
import sqlite3
import struct
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from threading import Barrier

import numpy as np
import cv2

from history import History


class HistoryTests(unittest.TestCase):
    def setUp(self):
        folder = self.enterContext(tempfile.TemporaryDirectory())
        self.path = Path(folder) / "history.sqlite3"
        self.history = self.open_history()
        self.board = np.full((18, 24, 3), 255, np.uint8)
        self.known = np.zeros(self.board.shape[:2], np.uint8)
        self.sid = self.history.create(self.board, self.known, created_at=1234)

    def open_history(self):
        return self.enterContext(closing(History(self.path)))

    def query(self, sql, *parameters):
        with closing(sqlite3.connect(self.path, isolation_level=None)) as connection:
            return connection.execute(sql, parameters).fetchall()

    def assert_state(self, result, seq, board, known):
        actual_seq, actual_board, actual_known = result
        self.assertEqual(actual_seq, seq)
        np.testing.assert_array_equal(actual_board, board)
        np.testing.assert_array_equal(actual_known, known)
        self.assertEqual(actual_board.dtype, np.uint8)
        self.assertEqual(actual_known.dtype, np.uint8)

    def test_initial_state_noop_and_array_ownership(self):
        self.assertIsInstance(self.sid, str)
        self.assertEqual(self.history.sessions(), [{"id": self.sid, "seq": 0, "created_at": 1234}])
        self.assertEqual(self.history.events(self.sid), [])
        self.assertEqual(self.history.record(self.sid, self.board, self.known, 1500, 1600), 0)
        _, saved_board, saved_known = self.history.state(self.sid)
        saved_board.fill(80)
        saved_known.fill(255)
        self.assert_state(self.history.state(self.sid), 0, self.board, self.known)
        self.board[5, 7] = (12, 87, 210)
        self.known[5, 7] = 255
        expected = self.board.copy(), self.known.copy()
        self.assertEqual(self.history.record(self.sid, self.board, self.known, 1700, 1800), 1)
        self.board.fill(64)
        self.known.fill(0)
        self.assert_state(self.history.state(self.sid), 1, *expected)
        self.assertEqual(len(self.history.events(self.sid)), 1)

    def test_event_kinds_and_timestamps(self):
        self.known[:8] = 255
        for seq, kind in enumerate(("first_visible", "add", "update", "erase"), start=1):
            if kind == "add":
                self.board[2, 1] = (38, 80, 179)
                self.board[3, 8] = 248  # Paper brightness also changed near the stroke.
            elif kind == "update":
                self.board[2, 1] = (119, 9, 76)
            elif kind == "erase":
                self.board[2, 1] = 250
            observed_at = 1770000000000 + seq * 100
            confirmed_at = observed_at + 60
            self.assertEqual(self.history.record(self.sid, self.board, self.known,
                                                 observed_at, confirmed_at), seq)
            self.assertEqual(self.history.events(self.sid)[-1], {
                "seq": seq, "observed_at": observed_at, "confirmed_at": confirmed_at, "kind": kind,
            })

    def test_every_sequence_after_closing_across_multiple_snapshots(self):
        frames = [(self.board.copy(), self.known.copy())]
        for seq in range(1, 68):
            y, x = divmod(seq, self.board.shape[1])
            if seq % 3:
                self.board[y, x] = (seq, 255 - seq, 2 * seq)
            if seq % 4:
                self.known[y, x] = 255
            else:
                self.board[y, x] = (10, seq, 19)
            self.assertEqual(self.history.record(self.sid, self.board, self.known, seq, seq + 1), seq)
            frames.append((self.board.copy(), self.known.copy()))
        self.history.close()
        reopened = self.open_history()
        for seq, (board, known) in enumerate(frames):
            with self.subTest(seq=seq):
                self.assert_state(reopened.state(self.sid, seq), seq, board, known)
        self.assert_state(reopened.state(self.sid), 67, *frames[-1])
        self.assertEqual(self.query("SELECT seq FROM snapshots WHERE session_id=? ORDER BY seq", self.sid),
                         [(0,), (32,), (64,)])
        self.assertEqual(self.query("SELECT seq FROM patches WHERE seq IN (32, 64)"), [])
        self.assertEqual(self.query("PRAGMA integrity_check"), [("ok",)])

    def test_separated_dots_store_only_their_changed_pixels(self):
        self.board[0, 0] = 0
        self.known[-1, -1] = 255
        self.history.record(self.sid, self.board, self.known, 1, 2)
        patches = self.query("SELECT x, y, board_png, known_png FROM patches WHERE session_id=?",
                             self.sid)
        h, w = self.known.shape
        self.assertEqual({(x, y) for x, y, _, _ in patches}, {(0, 0), (w - 1, h - 1)})
        for _, _, board_png, known_png in patches:
            for blob in (board_png, known_png):
                self.assertEqual(struct.unpack(">II", blob[16:24]), (1, 1))
        self.assert_state(self.open_history().state(self.sid), 1, self.board, self.known)

    def test_dot_and_diagonal_with_same_bounding_box_origin(self):
        self.board[0, 0] = 0
        self.board[np.arange(16), np.arange(15, -1, -1)] = 0
        self.history.record(self.sid, self.board, self.known, 1, 2)
        self.assert_state(self.open_history().state(self.sid), 1, self.board, self.known)

    def test_invalid_session_sequence_and_images(self):
        for call in (self.history.state, self.history.events):
            with self.assertRaises(KeyError):
                call("missing")
        for seq in (-1, 1, 0.5, True):
            with self.subTest(seq=seq), self.assertRaises(ValueError):
                self.history.state(self.sid, seq)
        for board, known in ((self.board[:, :, :2], self.known), (self.board[:-1], self.known[:-1]),
                             (self.board, self.known + 1), (self.board, self.known.astype(bool))):
            with self.assertRaises(ValueError):
                self.history.record(self.sid, board, known, 1, 2)
        self.assertEqual(self.history.events(self.sid), [])

    def test_concurrent_readers_and_writers_across_connections(self):
        connections = (self.history, self.open_history())
        frames = [(np.full_like(self.board, i), np.full_like(self.known, (i % 2) * 255))
                  for i in range(16)]
        with ThreadPoolExecutor(max_workers=6) as pool:
            writes, reads = [], []
            for i, (board, known) in enumerate(frames):
                writer, reader = connections[i % 2], connections[1 - i % 2]
                writes.append(pool.submit(writer.record, self.sid, board, known, i, i + 1))
                reads.append(pool.submit(reader.state, self.sid))
        saved = {future.result(): frame for future, frame in zip(writes, frames)}
        self.assertEqual(sorted(saved), list(range(1, 17)))
        saved[0] = (self.board, self.known)
        for seq, frame in saved.items():
            self.assert_state(self.history.state(self.sid, seq), seq, *frame)
        for future in reads:
            result = future.result()
            self.assert_state(result, result[0], *saved[result[0]])

    def test_simultaneous_first_open_and_isolated_sessions(self):
        path = self.path.with_name("parallel.sqlite3")
        barrier = Barrier(6)

        def create(i):
            barrier.wait(timeout=5)
            with closing(History(path)) as history:
                board = np.full_like(self.board, i)
                return history.create(board, self.known, i)

        with ThreadPoolExecutor(max_workers=6) as pool:
            ids = list(pool.map(create, range(6)))
        with closing(History(path)) as history:
            self.assertEqual({row["id"] for row in history.sessions()}, set(ids))
            for i, sid in enumerate(ids):
                self.assert_state(history.state(sid), 0, np.full_like(self.board, i), self.known)

    def test_failed_delta_rolls_back_and_can_be_retried(self):
        self.query("CREATE TRIGGER fail_commit BEFORE UPDATE OF seq ON sessions "
                   "BEGIN SELECT RAISE(ABORT, 'injected failure'); END")
        changed = self.board.copy()
        changed[3, 8] = (20, 144, 90)
        with self.assertRaises(sqlite3.IntegrityError):
            self.history.record(self.sid, changed, self.known, 1, 2)
        reopened = self.open_history()
        self.assert_state(reopened.state(self.sid), 0, self.board, self.known)
        self.assertEqual(reopened.events(self.sid), [])
        self.assertEqual(self.query("SELECT seq FROM patches"), [])
        self.query("DROP TRIGGER fail_commit")
        self.assertEqual(self.history.record(self.sid, changed, self.known, 1, 2), 1)
        self.assert_state(reopened.state(self.sid), 1, changed, self.known)

    def test_process_exit_during_snapshot_transaction(self):
        for seq in range(1, 32):
            self.board[0, 0] = seq
            self.history.record(self.sid, self.board, self.known, seq, seq)
        self.history.close()
        # Exit after the PNG and event have been inserted, before advancing the head.
        script = """
import os
import sys
from history import History
history = History(sys.argv[1])
sid = sys.argv[2]
_, board, known = history.state(sid)
board[0, 0] = 32
history._db.create_function("interrupt_commit", 0, lambda: os._exit(77))
history._db.execute("CREATE TEMP TRIGGER interrupt BEFORE UPDATE OF seq ON sessions "
                    "BEGIN SELECT interrupt_commit(); END")
history.record(sid, board, known, 32, 32)
"""
        child = subprocess.run([sys.executable, "-c", script, str(self.path), self.sid],
                               cwd=Path(__file__).resolve().parents[1], timeout=20)
        self.assertEqual(child.returncode, 77)
        reopened = self.open_history()
        self.assert_state(reopened.state(self.sid), 31, self.board, self.known)
        self.assertEqual(len(reopened.events(self.sid)), 31)
        self.assertEqual(self.query("SELECT seq FROM snapshots ORDER BY seq"), [(0,)])
        self.board[0, 0] = 32
        self.assertEqual(reopened.record(self.sid, self.board, self.known, 32, 32), 32)
        self.assert_state(self.open_history().state(self.sid), 32, self.board, self.known)

    def test_replay_segments_match_state_across_snapshot_boundaries(self):
        frames = [(self.board.copy(), self.known.copy())]
        for seq in range(1, 68):
            y, x = divmod(seq, self.board.shape[1])
            self.board[y, x] = (seq, 255 - seq, 2 * seq)
            self.known[y, x] = 255
            self.assertEqual(self.history.record(self.sid, self.board, self.known, seq, seq), seq)
            frames.append((self.board.copy(), self.known.copy()))
        reopened = self.open_history()
        for target in (0, 1, 31, 32, 33, 63, 64, 67):
            with self.subTest(target=target):
                segment = reopened.replay(self.sid, target)
                start = (target // 32) * 32
                end = min(start + 31, 67)
                self.assertEqual((segment["start"], segment["end"]), (start, end))
                self.assertEqual([frame["seq"] for frame in segment["frames"]],
                                 list(range(start + 1, end + 1)))
                board = cv2.imdecode(np.frombuffer(base64.b64decode(segment["snapshot"]["board"]), np.uint8), cv2.IMREAD_COLOR)
                known = cv2.imdecode(np.frombuffer(base64.b64decode(segment["snapshot"]["known"]), np.uint8), cv2.IMREAD_GRAYSCALE)
                for frame in segment["frames"]:
                    for patch in frame["patches"]:
                        patch_board = cv2.imdecode(np.frombuffer(base64.b64decode(patch["board"]), np.uint8), cv2.IMREAD_COLOR)
                        patch_known = cv2.imdecode(np.frombuffer(base64.b64decode(patch["known"]), np.uint8), cv2.IMREAD_GRAYSCALE)
                        height, width = patch_known.shape
                        board[patch["y"]:patch["y"] + height, patch["x"]:patch["x"] + width] = patch_board
                        known[patch["y"]:patch["y"] + height, patch["x"]:patch["x"] + width] = patch_known
                    expected_board, expected_known = frames[frame["seq"]]
                    np.testing.assert_array_equal(board, expected_board)
                    np.testing.assert_array_equal(known, expected_known)
                self.assert_state(reopened.state(self.sid, target), target, *frames[target])

    def test_replay_reuses_sparse_png_and_validates_target(self):
        changed = self.board.copy()
        changed[1, 2] = (1, 2, 3)
        changed[-2, -3] = (4, 5, 6)
        self.history.record(self.sid, changed, self.known, 1, 2)
        replay = self.history.replay(self.sid, 1)
        self.assertEqual(len(replay["frames"]), 1)
        self.assertEqual(len(replay["frames"][0]["patches"]), 2)
        self.assertTrue(all(patch["board"] and patch["known"] for patch in replay["frames"][0]["patches"]))
        with self.assertRaises(KeyError):
            self.history.replay("missing", 0)
        for invalid in (-1, 2, 0.5, True):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.history.replay(self.sid, invalid)


if __name__ == "__main__":
    unittest.main()
