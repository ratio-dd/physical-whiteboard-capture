"""Local HTTP service for the physical whiteboard MVP.

The service deliberately listens on loopback.  It is intended to be opened by
the local browser workbench, while the durable state lives in a SQLite file.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
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
from live_ink.session import LiveInkSession, LiveSessionStore


MAX_FRAME_BYTES = 12 * 1024 * 1024
MAX_JSON_BYTES = 128 * 1024
MAX_BOARD_PIXELS = 8_000_000
SESSION_IDLE_SECONDS = 3600
EXPERIMENT_PACKAGE_MODES = {
    "yolo-experiment",
    "yolo-mps-experiment",
    "source-color-experiment",
    "lecturenet-experiment",
}


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
    live_session: LiveInkSession | None = None
    live_pending: int = 0
    live_drain: threading.Condition | None = None

    def __post_init__(self):
        self.live_drain = threading.Condition(self.lock)


class WhiteboardApp:
    def __init__(self, data_dir: str | os.PathLike[str], web_root: str | os.PathLike[str] | None = None,
                 replay_packages_dir: str | os.PathLike[str] | None = None,
                 live_runtime_factory=None,
                 live_data_dir: str | os.PathLike[str] | None = None,
        live_model_version: str = "LectureNet-V34-FCN-proposal_text+source-evidence-v1",
                 live_weights_sha256: str = ""):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.history = History(self.data_dir / "whiteboard.sqlite3")
        self.web_root = Path(web_root) if web_root else Path(__file__).parent / "web"
        self.replay_packages_dir = Path(replay_packages_dir) if replay_packages_dir else self.data_dir / "replay-packages"
        self.sessions: dict[str, Session] = {}
        self.lock = threading.RLock()
        self.ink_cache: OrderedDict[tuple[str, int, str], dict] = OrderedDict()
        self.ink_jobs: dict[tuple[str, int, str], object] = {}
        self.ink_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ink")
        # Explicitly injected only; normal Mac startup has no model/weights.
        self.live_runtime_factory = live_runtime_factory
        self.live_data_dir = Path(live_data_dir) if live_data_dir else self.data_dir / "live-ink"
        self.live_model_version = live_model_version
        self.live_weights_sha256 = live_weights_sha256

    def _package_path(self, sid: str, filename: str) -> Path:
        root = self.replay_packages_dir.resolve()
        candidate = (root / sid / filename).resolve()
        if root not in candidate.parents or candidate.suffix != ".json":
            raise ValueError("invalid replay package path")
        return candidate

    def _read_package(self, path: Path, sid: str, head: int) -> dict | None:
        try:
            package = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not isinstance(package, dict) or package.get("version") != 1:
            return None
        if package.get("session_id") != sid or package.get("recognition_mode") not in EXPERIMENT_PACKAGE_MODES:
            return None
        if package.get("session_version") != head or package.get("through_seq") != head:
            return None
        if not isinstance(package.get("events"), list) or not isinstance(package.get("model_version"), str):
            return None
        return package

    def replay_packages(self, sid: str) -> dict:
        head = self.history.state(sid, None)[0]
        directory = (self.replay_packages_dir / sid)
        packages = []
        live = LiveSessionStore.load_package(self.live_data_dir, sid)
        if live is not None and live.get("through_seq") == head:
            packages.append({"mode": "lecturenet-live", "seq": head,
                             "model_version": live.get("model_version", ""),
                             "weights_sha256": live.get("weights_sha256", ""),
                             "proposal_head": live.get("proposal_head", ""),
                             "session_version": head})
        if directory.is_dir():
            for path in sorted(directory.glob("*.json")):
                package = self._read_package(path, sid, head)
                if package is None:
                    continue
                mode = package["recognition_mode"]
                packages.append({"mode": mode, "seq": head,
                                 "model_version": package["model_version"],
                                 "weights_sha256": package.get("weights_sha256", ""),
                                 "session_version": head})
        return {"packages": packages}

    def _precomputed_ink(self, sid: str, head: int, mode: str) -> dict:
        # The unsuffixed name is the original CPU package and remains readable.
        names = [f"{head}.json"] if mode == "yolo-experiment" else []
        names.append(f"{head}.{mode}.json")
        package = None
        for name in names:
            package = self._read_package(self._package_path(sid, name), sid, head)
            if package is not None and package.get("recognition_mode") == mode:
                break
        else:
            package = None
        if package is None:
            raise RuntimeError("YOLO 实验回放包不可用")
        return package

    def close(self) -> None:
        with self.lock:
            sessions = list(self.sessions.values())
        with self.lock:
            live_sessions = [s.live_session for s in self.sessions.values() if s.live_session is not None]
            sessions_by_live = {s.live_session: s for s in self.sessions.values() if s.live_session is not None}
        for live in live_sessions:
            try:
                session = sessions_by_live[live]
                with session.lock:
                    session.stopped = True; session.estimator = None
                    while session.live_pending:
                        session.live_drain.wait(.05)
                live.stop()
            except RuntimeError:
                pass
        self.ink_executor.shutdown(wait=True, cancel_futures=True)
        self.history.close()

    def create(self, corners: list[list[float]], width: int, height: int,
               recognition_mode: str | None = None) -> Session:
        # The estimator owns geometric validation and gives callers a useful
        # ValueError for malformed or self-intersecting quadrilaterals.
        estimator = BoardEstimator(corners, width, height)
        if recognition_mode == "lecturenet-live" and self.live_runtime_factory is None:
            raise RuntimeError("lecturenet-live is unavailable: local MPS model is not configured")
        created = _now_ms()
        persisted_id = self.history.create(estimator.board, estimator.known, created)
        # History is the source of truth for the identifier.  Keeping the
        # generated id in both layers makes restart and concurrent requests
        # deterministic even if a storage backend chooses its own id format.
        sid = str(persisted_id)
        live = None
        if recognition_mode == "lecturenet-live":
            live = LiveInkSession(self.live_data_dir, sid, width, height, recognition_mode,
                                  self.live_model_version, self.live_weights_sha256,
                                  self.live_runtime_factory)
        session = Session(sid, estimator, created, live_session=live)
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
        live_submit = None
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
            if session.live_session is not None and session.seq > previous_seq:
                # Submit outside session.lock below: OrderedInkRuntime's
                # lossless bounded queue may briefly apply backpressure.
                live_submit = (session.live_session, session.seq, first_seen,
                               estimator.board.copy(), estimator.known.copy())
                session.live_pending += 1
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
        if live_submit is not None:
            live, live_seq, live_stamp, live_board, live_known = live_submit
            try:
                live.submit(live_seq, live_stamp, live_board, live_known)
            finally:
                with session.lock:
                    session.live_pending -= 1
                    session.live_drain.notify_all()
            response["ink_live"] = live.response(live_seq - 1)
        elif session.live_session is not None:
            response["ink_live"] = session.live_session.response(session.seq - 1)
        return response

    def live_ink(self, sid: str, after_seq: int = 0) -> dict:
        with self.lock:
            session = self.sessions.get(sid)
        if session is not None and session.live_session is not None:
            return session.live_session.response(after_seq)
        persisted = self.live_data_dir / sid / "metadata.json"
        if persisted.is_file():
            try:
                value = json.loads(persisted.read_text(encoding="utf-8"))
                seq = int(value.get("through_seq", 0)); persisted_state = value.get("state", "stopped")
                if persisted_state == "running": persisted_state = "interrupted"
                value.update({"seq": seq, "updated": seq > after_seq,
                    "status": persisted_state, "state": persisted_state, "lag": 0, "depth": 0,
                    "submitted": seq, "completed": seq, "dropped": 0})
                if seq > after_seq:
                    svg = self.live_data_dir / sid / f"seq-{seq:06d}.svg"
                    value["svg"] = svg.read_text(encoding="utf-8") if svg.is_file() else ""
                return value
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        return {"enabled": False, "state": "disabled", "submitted_seq": 0,
                "processed_seq": 0, "pending": 0, "error": None, "result": None, "flushed": False}

    def live_package(self, sid: str, seq: int | None = None) -> dict:
        with self.lock:
            session = self.sessions.get(sid)
        if session is not None and session.live_session is not None:
            package = session.live_session.store.package(seq)
        else:
            package = LiveSessionStore.load_package(self.live_data_dir, sid, seq)
        if package is None:
            raise RuntimeError("live LectureNet package is unavailable")
        return package

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
        if mode not in {"classic", "neural", "lecturenet-live", *EXPERIMENT_PACKAGE_MODES}:
            raise ValueError("unsupported ink mode")
        if mode == "lecturenet-live":
            return self.live_package(sid, target)
        if mode in EXPERIMENT_PACKAGE_MODES:
            result = self._precomputed_ink(sid, head, mode)
            result.setdefault("recognition_mode", mode)
            return result
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
            if len(parts) == 5 and parts[1:3] == ["api", "sessions"] and parts[4] == "ink-packages":
                self._send_json(self.app.replay_packages(parts[3]))
                return
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
            if len(parts) == 5 and parts[1:3] == ["api", "sessions"] and parts[4] in {"live-ink", "ink-live"}:
                after = int(parse_qs(parsed.query).get("after_seq", ["0"])[0])
                wait_ms = min(100, max(0, int(parse_qs(parsed.query).get("wait_ms", ["0"])[0])))
                deadline = time.monotonic() + wait_ms / 1000
                result = self.app.live_ink(parts[3], after)
                while not result.get("updated") and time.monotonic() < deadline:
                    time.sleep(.02); result = self.app.live_ink(parts[3], after)
                self._send_json(result)
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
                mode = body.get("recognition_mode")
                if mode is not None and mode != "lecturenet-live":
                    raise ValueError("unsupported recognition_mode")
                session = self.app.create(corners, width, height, mode)
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
                    while session.live_pending:
                        session.live_drain.wait(.05)
                if session.live_session is not None:
                    session.live_session.stop()
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


def _make_lecturenet_runtime_factory(model_root: Path, weights: Path):
    """Build the explicit live factory without importing ML at normal startup."""
    if not model_root.is_dir() or not weights.is_file():
        return None
    def factory(sink):
        import sys
        runtime_root = Path(__file__).parent
        sys.path[:0] = [str(runtime_root / "experiments/faint-color-ink"),
                       str(runtime_root / "experiments/lecturemath-ink")]
        from live_ink.svg import geometry_svg
        from evidence_memory import MemoryConfig
        from pipeline import FaintColorPipeline, PipelineConfig
        from live_ink.runtime import LectureNetMPSBackend, OrderedInkRuntime
        backend = LectureNetMPSBackend(model_root, weights, output_mode="proposal_text")
        return OrderedInkRuntime(
            backend,
            lambda: FaintColorPipeline(PipelineConfig(memory=MemoryConfig(threshold=0.5))),
            lambda result, width, height: geometry_svg(result.vector_paths, width, height),
            max_queue=2, backpressure="block", result_sink=sink)
    return factory


def main() -> None:
    parser = argparse.ArgumentParser(description="Physical whiteboard local service")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-dir", default=str(Path(__file__).parent / "data"))
    parser.add_argument("--replay-packages", default=None, help="optional local precomputed ink package directory")
    cache = Path(os.environ.get("LECTURENET_CACHE", str(Path(__file__).parent / ".cache/lecturenet-v34")))
    parser.add_argument("--live-model-root", default=str(cache / "lecturemath/ACCESS2021_release"))
    parser.add_argument("--live-weights", default=str(cache / "LectureNet_model_BIN_V34_final.dat"))
    args = parser.parse_args()
    live_root, live_weights = Path(args.live_model_root), Path(args.live_weights)
    live_factory = _make_lecturenet_runtime_factory(live_root, live_weights)
    live_hash = hashlib.sha256(live_weights.read_bytes()).hexdigest() if live_weights.is_file() else ""
    server = make_server(WhiteboardApp(args.data_dir, replay_packages_dir=args.replay_packages,
                                       live_runtime_factory=live_factory,
                                       live_model_version="LectureNet-V34-FCN-proposal_text+source-evidence-v1",
                                       live_weights_sha256=live_hash), port=args.port)
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
