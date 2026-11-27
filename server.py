"""Local HTTP service for the physical whiteboard MVP.

The service deliberately listens on loopback.  It is intended to be opened by
the local browser workbench, while the durable state lives in a SQLite file.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import threading
import time
import math
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from history import History
from vision import BoardEstimator


MAX_FRAME_BYTES = 12 * 1024 * 1024
MAX_JSON_BYTES = 128 * 1024
MAX_BOARD_PIXELS = 8_000_000
SESSION_IDLE_SECONDS = 3600


class InkPending(Exception):
    def __init__(self, session_id: str, seq: int):
        super().__init__("ink derivation is still running")
        self.session_id, self.seq = session_id, seq


def _now_ms() -> int:
    return int(time.time() * 1000)


def _png_b64(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


@dataclass
class Session:
    id: str
    estimator: BoardEstimator | None
    created_at: int
    stopped: bool = False
    lock: threading.RLock = field(default_factory=threading.RLock)
    seq: int = 0
    last_activity: float = field(default_factory=time.monotonic)


class WhiteboardApp:
    def __init__(self, data_dir: str | os.PathLike[str], web_root: str | os.PathLike[str] | None = None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.history = History(self.data_dir / "whiteboard.sqlite3")
        self.web_root = Path(web_root) if web_root else Path(__file__).parent / "web"
        self.sessions: dict[str, Session] = {}
        self.lock = threading.RLock()
        self.ink_cache: OrderedDict[tuple[str, int, str], dict] = OrderedDict()
        self.ink_jobs: dict[tuple[str, int, str], object] = {}
        self.ink_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ink")

    def close(self) -> None:
        self.ink_executor.shutdown(wait=True, cancel_futures=True)
        self.history.close()

    def create(self, corners: list[list[float]], width: int, height: int) -> Session:
        # The estimator owns geometric validation and gives callers a useful
        # ValueError for malformed or self-intersecting quadrilaterals.
        estimator = BoardEstimator(corners, width, height)
        created = _now_ms()
        persisted_id = self.history.create(estimator.board, estimator.known, created)
        # History is the source of truth for the identifier.  Keeping the
        # generated id in both layers makes restart and concurrent requests
        # deterministic even if a storage backend chooses its own id format.
        sid = str(persisted_id)
        session = Session(sid, estimator, created)
        with self.lock:
            self._cleanup_sessions()
            self.sessions[sid] = session
        return session

    def _cleanup_sessions(self) -> None:
        cutoff = time.monotonic() - SESSION_IDLE_SECONDS
        for sid, session in list(self.sessions.items()):
            if session.last_activity < cutoff:
                with session.lock:
                    session.stopped = True
                    session.estimator = None
                del self.sessions[sid]

    def get(self, sid: str) -> Session:
        with self.lock:
            self._cleanup_sessions()
            session = self.sessions.get(sid)
        if session is None:
            raise KeyError("session not found")
        return session

    def process(self, session: Session, jpeg: bytes, observed_at: int | None, seen_seq: int | None = None) -> dict:
        if len(jpeg) > MAX_FRAME_BYTES:
            raise ValueError("frame is too large")
        array = np.frombuffer(jpeg, dtype=np.uint8)
        frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("invalid JPEG frame")
        with session.lock:
            if session.stopped:
                raise ValueError("session is stopped")
            session.last_activity = time.monotonic()
            stamp = int(observed_at if observed_at is not None else _now_ms())
            estimator = session.estimator
            if estimator is None:
                raise ValueError("session is stopped")
            result = estimator.update(frame, stamp)
            first_seen = int(result.get("observed_at", stamp))
            confirmed = max(_now_ms(), first_seen)
            previous_seq = session.seq
            session.seq = self.history.record(session.id, estimator.board, estimator.known, first_seen, confirmed)
            response = {
                "seq": session.seq,
                "held": float(result.get("held", 0.0)),
                # Fraction of rectified pixels currently classified as glare.
                # Older estimators do not expose it, so clients receive 0.
                "glare": float(result.get("glare", 0.0)),
            }
            if seen_seq != session.seq or previous_seq != session.seq:
                response["board"] = _png_b64(estimator.board)
                response["unknown"] = _png_b64(255 - estimator.known)
            return response

    def state(self, sid: str, seq: int | None = None) -> dict:
        # History remains readable after a process restart, when no in-memory
        # Session object exists.
        actual, board, known = self.history.state(sid, seq)
        return {"seq": actual, "board": _png_b64(board), "unknown": _png_b64(255 - known)}

    def ink(self, sid: str, seq: int | None = None, mode: str = "classic", wait_seconds: float = 0.35) -> dict:
        """Derive vector ink outside the pixel-history write path.

        Results are cached only for the eight most recently requested
        session/sequence pairs.  A repeated request therefore does not repeat
        an expensive derivation or grow memory without bound.
        """
        head = self.history.state(sid, None)[0]
        target = head if seq is None else int(seq)
        if mode not in {"classic", "neural"}:
            raise ValueError("mode must be classic or neural")
        key = (sid, target, mode)
        with self.lock:
            cached = self.ink_cache.get(key)
            if cached is not None:
                self.ink_cache.move_to_end(key)
                return cached
        with self.lock:
            job = self.ink_jobs.get(key)
            if job is None:
                try:
                    from ink import build_ink
                except ImportError as exc:
                    raise RuntimeError("ink derivation is unavailable") from exc
                def derive():
                    try:
                        return build_ink(self.history, sid, target, mode=mode)
                    except TypeError:
                        if mode != "classic":
                            raise RuntimeError("neural ink derivation is unavailable")
                        return build_ink(self.history, sid, target)
                job = self.ink_executor.submit(derive)
                self.ink_jobs[key] = job
        try:
            try:
                result = job.result(timeout=wait_seconds)
            except FutureTimeout as exc:
                raise InkPending(sid, target) from exc
            if not isinstance(result, dict):
                raise RuntimeError("invalid ink derivation result")
            result.setdefault("recognition_mode", mode)
            with self.lock:
                self.ink_cache[key] = result
                self.ink_cache.move_to_end(key)
                while len(self.ink_cache) > 8:
                    self.ink_cache.popitem(last=False)
                self.ink_jobs.pop(key, None)
        except InkPending:
            # The worker is still running. Keep the in-flight entry so later
            # polls join this exact Future instead of starting a duplicate.
            raise
        except Exception:
            with self.lock:
                self.ink_jobs.pop(key, None)
            raise
        return result


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "PhysicalWhiteboard/0.1"

    @property
    def app(self) -> WhiteboardApp:
        return self.server.app  # type: ignore[attr-defined]

    def _send_json(self, value: object, status: int = HTTPStatus.OK) -> None:
        payload = json.dumps(value, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _error(self, status: int, message: str) -> None:
        self._send_json({"error": message}, status)

    def _read_body(self, limit: int) -> bytes:
        try:
            length = int(self.headers.get("Content-Length", "-1"))
        except ValueError:
            raise ValueError("invalid Content-Length")
        if length < 0 or length > limit:
            raise ValueError("request body is too large or missing length")
        body = self.rfile.read(length)
        if len(body) != length:
            raise ValueError("incomplete request body")
        return body

    def _json_body(self) -> dict:
        try:
            value = json.loads(self._read_body(MAX_JSON_BYTES).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("JSON body must be an object")
        return value

    def do_OPTIONS(self) -> None:
        if not self._allowed_request():
            self._error(HTTPStatus.FORBIDDEN, "origin not allowed")
            return
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _allowed_request(self) -> bool:
        host = self.headers.get("Host", "")
        expected_hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        if host not in expected_hosts:
            return False
        origin = self.headers.get("Origin")
        if origin is None:
            return True
        return origin in {f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"}

    def do_GET(self) -> None:
        if not self._allowed_request():
            self._error(HTTPStatus.FORBIDDEN, "origin not allowed")
            return
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        try:
            if path == "/api/sessions":
                self._send_json({"sessions": self.app.history.sessions()})
                return
            parts = path.split("/")
            if len(parts) == 5 and parts[1:3] == ["api", "sessions"] and parts[4] == "events":
                self._send_json({"events": self.app.history.events(parts[3])})
                return
            if len(parts) == 5 and parts[1:3] == ["api", "sessions"] and parts[4] == "state":
                raw_seq = parse_qs(parsed.query).get("seq", [None])[0]
                seq = int(raw_seq) if raw_seq is not None else None
                self._send_json(self.app.state(parts[3], seq))
                return
            if len(parts) == 5 and parts[1:3] == ["api", "sessions"] and parts[4] == "replay":
                raw_seq = parse_qs(parsed.query).get("seq", [None])[0]
                if raw_seq is None:
                    raise ValueError("seq is required")
                segment = self.app.history.replay(parts[3], int(raw_seq))
                self._send_json(segment)
                return
            if len(parts) == 5 and parts[1:3] == ["api", "sessions"] and parts[4] == "ink":
                raw_seq = parse_qs(parsed.query).get("seq", [None])[0]
                seq = int(raw_seq) if raw_seq is not None else None
                mode = parse_qs(parsed.query).get("mode", ["classic"])[0]
                try:
                    self._send_json(self.app.ink(parts[3], seq, mode))
                except InkPending as pending:
                    self._send_json({"status": "pending", "session_id": pending.session_id, "seq": pending.seq}, HTTPStatus.ACCEPTED)
                return
            if path.startswith("/api/"):
                self._error(HTTPStatus.NOT_FOUND, "route not found")
                return
            self._serve_static(path)
        except (KeyError, ValueError) as exc:
            self._error(HTTPStatus.NOT_FOUND if isinstance(exc, KeyError) else HTTPStatus.BAD_REQUEST, str(exc))
        except RuntimeError as exc:
            self._error(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
        except Exception:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal server error")

    def _serve_static(self, path: str) -> None:
        relative = "index.html" if path == "/" else path.lstrip("/")
        root = self.app.web_root.resolve()
        candidate = (root / relative).resolve()
        if root not in candidate.parents and candidate != root:
            self._error(HTTPStatus.NOT_FOUND, "not found")
            return
        if not candidate.is_file():
            self._error(HTTPStatus.NOT_FOUND, "not found")
            return
        data = candidate.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(candidate.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        if not self._allowed_request():
            self._error(HTTPStatus.FORBIDDEN, "origin not allowed")
            return
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        try:
            if path == "/api/sessions":
                body = self._json_body()
                corners = body.get("corners")
                width, height = body.get("width"), body.get("height")
                if (not isinstance(width, int) or isinstance(width, bool) or
                        not isinstance(height, int) or isinstance(height, bool) or
                        width <= 0 or height <= 0 or width * height > MAX_BOARD_PIXELS):
                    raise ValueError("corners, width and height are required")
                if not isinstance(corners, list) or len(corners) != 4:
                    raise ValueError("corners, width and height are required")
                if any(not isinstance(point, list) or len(point) != 2 or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in point) for point in corners):
                    raise ValueError("corners must contain finite numeric pairs")
                session = self.app.create(corners, width, height)
                self._send_json({"id": session.id}, HTTPStatus.CREATED)
                return
            parts = path.split("/")
            if len(parts) == 5 and parts[1:3] == ["api", "sessions"] and parts[4] == "frames":
                sid = parts[3]
                query_t = parse_qs(parsed.query).get("t", [None])[0]
                observed = int(query_t) if query_t is not None else None
                query_seen = parse_qs(parsed.query).get("seen_seq", [None])[0]
                seen_seq = int(query_seen) if query_seen is not None else None
                if seen_seq is not None and seen_seq < 0:
                    raise ValueError("seen_seq must be non-negative")
                if self.headers.get("Content-Type", "").split(";", 1)[0].lower() not in {"image/jpeg", "image/jpg"}:
                    raise ValueError("frame Content-Type must be image/jpeg")
                payload = self._read_body(MAX_FRAME_BYTES)
                if not payload:
                    raise ValueError("empty JPEG frame")
                result = self.app.process(self.app.get(sid), payload, observed, seen_seq)
                self._send_json(result)
                return
            if len(parts) == 5 and parts[1:3] == ["api", "sessions"] and parts[4] == "stop":
                session = self.app.get(parts[3])
                with session.lock:
                    session.stopped = True
                    session.estimator = None
                self._send_json({"id": session.id})
                return
            self._error(HTTPStatus.NOT_FOUND, "route not found")
        except KeyError as exc:
            self._error(HTTPStatus.NOT_FOUND, str(exc))
        except ValueError as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal server error")

    def log_message(self, fmt: str, *args: object) -> None:
        return


def make_server(app: WhiteboardApp, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), RequestHandler)
    server.app = app  # type: ignore[attr-defined]
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Physical whiteboard local service")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-dir", default=str(Path(__file__).parent / "data"))
    args = parser.parse_args()
    server = make_server(WhiteboardApp(args.data_dir), port=args.port)
    print(f"Physical whiteboard listening on http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        server.app.close()  # type: ignore[attr-defined]


if __name__ == "__main__":
    main()
