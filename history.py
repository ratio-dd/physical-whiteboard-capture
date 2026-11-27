"""SQLite history for rectified BGR whiteboards and their known-pixel masks."""

import base64
import sqlite3
import threading
import uuid

import cv2
import numpy as np


_SNAPSHOT_INTERVAL = 32
_TILE_SIZE = 64
_PAPER_MIN_CHANNEL = 235

_SCHEMA = """
BEGIN IMMEDIATE;
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    created_at INTEGER NOT NULL,
    seq INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    session_id TEXT NOT NULL REFERENCES sessions(id),
    seq INTEGER NOT NULL,
    observed_at INTEGER NOT NULL,
    confirmed_at INTEGER NOT NULL,
    kind TEXT NOT NULL,
    PRIMARY KEY (session_id, seq)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS snapshots (
    session_id TEXT NOT NULL REFERENCES sessions(id),
    seq INTEGER NOT NULL,
    board_png BLOB NOT NULL,
    known_png BLOB NOT NULL,
    PRIMARY KEY (session_id, seq)
);
CREATE TABLE IF NOT EXISTS patches (
    session_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    x INTEGER NOT NULL,
    y INTEGER NOT NULL,
    board_png BLOB NOT NULL,
    known_png BLOB NOT NULL,
    PRIMARY KEY (session_id, seq, y, x),
    FOREIGN KEY (session_id, seq) REFERENCES events(session_id, seq)
);
COMMIT;
"""


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _validate_images(board, known):
    if (not isinstance(board, np.ndarray) or board.dtype != np.uint8
            or board.ndim != 3 or board.shape[2] != 3 or board.size == 0):
        raise ValueError("board must be a nonempty H×W×3 uint8 BGR image")
    if (not isinstance(known, np.ndarray) or known.dtype != np.uint8
            or known.shape != board.shape[:2]):
        raise ValueError("known must be an H×W uint8 mask matching board")
    if cv2.countNonZero(cv2.inRange(known, 1, 254)):
        raise ValueError("known mask values must be 0 or 255")


def _png(image):
    ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise ValueError("could not encode history image")
    return encoded.tobytes()


def _decode(data, mode):
    image = cv2.imdecode(np.frombuffer(data, np.uint8), mode)
    if image is None:
        raise sqlite3.DatabaseError("invalid PNG in whiteboard history")
    return image


def _replay_images(board_png, known_png):
    return {"board": base64.b64encode(board_png).decode("ascii"),
            "known": base64.b64encode(known_png).decode("ascii")}


def _changed_pixels(old_board, old_known, board, known):
    same_color = cv2.inRange(cv2.absdiff(old_board, board), (0, 0, 0), (0, 0, 0))
    return cv2.bitwise_or(cv2.bitwise_not(same_color), cv2.bitwise_xor(old_known, known))


def _event_kind(old_board, old_known, board, known, changed):
    selected = changed != 0
    before, after = old_known[selected], known[selected]
    if np.all(before == 0) and np.all(after == 255):
        return "first_visible"
    if not np.all(before == 255) or not np.all(after == 255):
        return "update"
    # Paper-to-paper brightness variation in a committed camera tile
    # should not turn an otherwise pure addition or erasure into an update.
    had_ink = np.any(old_board[selected] < _PAPER_MIN_CHANNEL)
    has_ink = np.any(board[selected] < _PAPER_MIN_CHANNEL)
    if not had_ink and has_ink:
        return "add"
    if had_ink and not has_ink:
        return "erase"
    return "update"


def _regions(changed, left=0, top=0):
    """Trim and bisect sparse regions; sibling rectangles never overlap."""
    dx, dy, w, h = cv2.boundingRect(changed)
    if not w:
        return
    x, y = left + dx, top + dy
    tile = changed[dy:dy + h, dx:dx + w]
    count = cv2.countNonZero(tile)
    # Group nearby changes when separate PNG headers would dominate the data.
    dense = 2 * count >= w * h
    compact = max(w, h) <= _TILE_SIZE and count >= 8 and 8 * count >= w * h
    if dense or compact:
        yield x, y, w, h
    elif w >= h:
        mid = w // 2
        yield from _regions(tile[:, :mid], x, y)
        yield from _regions(tile[:, mid:], x + mid, y)
    else:
        mid = h // 2
        yield from _regions(tile[:mid, :], x, y)
        yield from _regions(tile[mid:, :], x, y + mid)


class History:
    """Append-only sessions. Public methods may be called from HTTP threads."""

    def __init__(self, db_path):
        self._lock = threading.Lock()
        self._cache = None  # Owned arrays for one committed (session_id, seq).
        self._db = sqlite3.connect(db_path, timeout=30, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        try:
            try:
                self._db.execute("PRAGMA journal_mode = WAL")
            except sqlite3.OperationalError as error:
                # Simultaneous opens can race the journal-mode switch. The
                # current SQLite journal also provides atomic transactions.
                if error.sqlite_errorcode & 0xFF != sqlite3.SQLITE_BUSY:
                    raise
            self._db.execute("PRAGMA synchronous = FULL")
            self._db.execute("PRAGMA foreign_keys = ON")
            self._db.executescript(_SCHEMA)
        except BaseException:
            self._db.close()
            raise

    def close(self):
        with self._lock:
            self._db.close()
            self._cache = None

    def __del__(self):
        # Tests and short-lived derivation helpers may not own an explicit
        # lifecycle.  Close the SQLite handle during interpreter cleanup so
        # those callers do not leak connections or emit ResourceWarning.
        try:
            db = getattr(self, "_db", None)
            if db is not None:
                db.close()
        except Exception:
            pass

    def create(self, board, known, created_at):
        _validate_images(board, known)
        created_at = _integer(created_at, "created_at")
        session_id = uuid.uuid4().hex
        board, known = board.copy(), known.copy()
        with self._lock:
            with self._db:
                self._db.execute("BEGIN IMMEDIATE")
                self._db.execute("INSERT INTO sessions VALUES (?, ?, 0)", (session_id, created_at))
                self._save_snapshot(session_id, 0, board, known)
            self._cache = (session_id, 0, board, known)
        return session_id

    def _save_snapshot(self, session_id, seq, board, known):
        self._db.execute("INSERT INTO snapshots VALUES (?, ?, ?, ?)",
                         (session_id, seq, _png(board), _png(known)))

    def _head_seq(self, session_id):
        row = self._db.execute("SELECT seq FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            raise KeyError(session_id)
        return row[0]

    def _restore(self, session_id, seq):
        row = self._db.execute(
            "SELECT seq, board_png, known_png FROM snapshots "
            "WHERE session_id = ? AND seq <= ? ORDER BY seq DESC LIMIT 1", (session_id, seq)
        ).fetchone()
        if row is None:
            raise sqlite3.DatabaseError("missing whiteboard snapshot")
        base, board_data, known_data = row
        board = _decode(board_data, cv2.IMREAD_COLOR)
        known = _decode(known_data, cv2.IMREAD_GRAYSCALE)
        patches = self._db.execute(
            "SELECT x, y, board_png, known_png FROM patches "
            "WHERE session_id = ? AND seq > ? AND seq <= ? ORDER BY seq, y, x",
            (session_id, base, seq),
        )
        for x, y, board_data, known_data in patches:
            tile = _decode(board_data, cv2.IMREAD_COLOR)
            mask = _decode(known_data, cv2.IMREAD_GRAYSCALE)
            h, w = mask.shape
            board[y:y + h, x:x + w] = tile
            known[y:y + h, x:x + w] = mask
        return board, known

    def _head_images(self, session_id, seq):
        if self._cache is None or self._cache[:2] != (session_id, seq):
            self._cache = (session_id, seq, *self._restore(session_id, seq))
        return self._cache[2:]

    def record(self, id, board, known, observed_at, confirmed_at):
        _validate_images(board, known)
        observed_at = _integer(observed_at, "observed_at")
        confirmed_at = _integer(confirmed_at, "confirmed_at")
        with self._lock:
            with self._db:
                # Reserve the write before reading the head; other History
                # instances sharing this file can never diff against a stale head.
                self._db.execute("BEGIN IMMEDIATE")
                seq = self._head_seq(id)
                old_board, old_known = self._head_images(id, seq)
                if board.shape != old_board.shape:
                    raise ValueError("board dimensions differ from the initial snapshot")
                changed = _changed_pixels(old_board, old_known, board, known)
                if not cv2.countNonZero(changed):
                    return seq
                seq += 1
                board, known = board.copy(), known.copy()
                kind = _event_kind(old_board, old_known, board, known, changed)
                self._db.execute("INSERT INTO events VALUES (?, ?, ?, ?, ?)",
                                 (id, seq, observed_at, confirmed_at, kind))
                if seq % _SNAPSHOT_INTERVAL == 0:
                    self._save_snapshot(id, seq, board, known)
                else:
                    for x, y, w, h in _regions(changed):
                        region = np.s_[y:y + h, x:x + w]
                        self._db.execute("INSERT INTO patches VALUES (?, ?, ?, ?, ?, ?)",
                                         (id, seq, x, y, _png(board[region]), _png(known[region])))
                self._db.execute("UPDATE sessions SET seq = ? WHERE id = ?", (seq, id))
            self._cache = (id, seq, board, known)
        return seq

    def state(self, id, seq=None):
        with self._lock:
            head = self._head_seq(id)
            target = head if seq is None else _integer(seq, "seq")
            if not 0 <= target <= head:
                raise ValueError(f"seq must be between 0 and {head}")
            if target == head:
                board, known = self._head_images(id, head)
                return head, board.copy(), known.copy()
            return target, *self._restore(id, target)

    def sessions(self):
        with self._lock:
            return [dict(row) for row in self._db.execute(
                "SELECT id, created_at, seq FROM sessions ORDER BY created_at DESC, id"
            )]

    def events(self, id):
        with self._lock:
            self._head_seq(id)
            return [dict(row) for row in self._db.execute(
                "SELECT seq, observed_at, confirmed_at, kind FROM events "
                "WHERE session_id = ? ORDER BY seq", (id,)
            )]

    def replay(self, id, seq):
        """Return the checkpoint and subsequent patches for the segment containing seq."""
        target = _integer(seq, "seq")
        with self._lock:
            head = self._head_seq(id)
            if not 0 <= target <= head:
                raise ValueError(f"seq must be between 0 and {head}")
            start = (target // _SNAPSHOT_INTERVAL) * _SNAPSHOT_INTERVAL
            end = min(start + _SNAPSHOT_INTERVAL - 1, head)
            snapshot = self._db.execute(
                "SELECT board_png, known_png FROM snapshots "
                "WHERE session_id = ? AND seq = ?", (id, start)
            ).fetchone()
            if snapshot is None:
                raise sqlite3.DatabaseError("missing whiteboard segment snapshot")

            patch_rows = self._db.execute(
                "SELECT seq, x, y, board_png, known_png FROM patches "
                "WHERE session_id = ? AND seq > ? AND seq <= ? ORDER BY seq, y, x",
                (id, start, end),
            ).fetchall()
        # Encoding may run concurrently with capture. The selected rows belong
        # to immutable, committed events and no longer need the database lock.
        frames = [{"seq": n, "patches": []} for n in range(start + 1, end + 1)]
        for n, x, y, board_png, known_png in patch_rows:
            frames[n - start - 1]["patches"].append({"x": x, "y": y, **_replay_images(board_png, known_png)})
        return {"start": start, "end": end, "snapshot": _replay_images(*snapshot), "frames": frames}
