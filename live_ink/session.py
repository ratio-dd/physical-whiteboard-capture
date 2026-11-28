"""Persistent adapter between OrderedInkRuntime and the HTTP session API."""
from __future__ import annotations
import json, os, threading, math
from pathlib import Path
from typing import Any, Callable
from .runtime import OrderedInkRuntime, RuntimeResult


class LiveSessionStore:
    def __init__(self, root: str | os.PathLike[str], session_id: str, width: int, height: int,
                 mode: str, model_version: str, weights_sha256: str, proposal_head: str = "proposal_text"):
        self.root = Path(root) / session_id; self.root.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.root / "metadata.json"; self.lock = threading.RLock()
        self.events_path = self.root / "events.json"  # legacy read-only compatibility
        self.events_dir = self.root / "events"; self.events_dir.mkdir(exist_ok=True)
        self.legacy_events = json.loads(self.events_path.read_text(encoding="utf-8")) if self.events_path.is_file() else None
        self.previous = {}; self.next_id = 1
        self.metadata = {"version": 1, "session_id": session_id, "width": width, "height": height,
                         "recognition_mode": mode, "model_version": model_version,
                         "weights_sha256": weights_sha256, "proposal_head": proposal_head,
                         "through_seq": 0, "state": "running"}
        self._write(self.meta_path, self.metadata)

    @staticmethod
    def _write(path: Path, value: Any) -> None:
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        temp.replace(path)

    def write_result(self, result: RuntimeResult) -> None:
        with self.lock:
            paths = getattr(result.pipeline_result, "vector_paths", []) or []
            current = {}
            for path in paths:
                points = path.get("points", []) if isinstance(path, dict) else []
                if points:
                    center = (sum(float(p[0]) for p in points) / len(points), sum(float(p[1]) for p in points) / len(points))
                else:
                    center = (0.0, 0.0)
                best = None; distance = 1e9
                for ident, old in self.previous.items():
                    if ident in current: continue
                    d = math.hypot(center[0] - old[0], center[1] - old[1])
                    if d < distance: best, distance = ident, d
                ident = best if best is not None and distance <= 24 else f"stroke-{self.next_id:04d}"
                if ident == f"stroke-{self.next_id:04d}": self.next_id += 1
                current[ident] = center
            changes = []
            by_id = {}
            for ident, path in zip(current, paths):
                stroke = dict(path); stroke["id"] = ident
                by_id[ident] = stroke
                changes.append({"op": "update" if ident in self.previous else "add", "id": ident, "stroke": stroke})
            for ident in self.previous:
                if ident not in current: changes.append({"op": "remove", "id": ident})
            event = {"seq": result.seq, "observed_at": result.observed_at, "changes": changes}
            self._write(self.events_dir / f"seq-{result.seq:06d}.json", event)
            self.previous = current
            self._write(self.root / f"seq-{result.seq:06d}.svg", result.svg)
            self._write(self.root / f"seq-{result.seq:06d}.json", {"seq": result.seq, "observed_at": result.observed_at, "arrival_to_svg_ms": result.arrival_to_svg_ms})
            self.metadata["through_seq"] = result.seq; self.metadata["latest_svg"] = str(self.root / f"seq-{result.seq:06d}.svg")
            self._write(self.meta_path, self.metadata)

    def package(self, through: int | None = None) -> dict[str, Any]:
        with self.lock:
            head = int(self.metadata.get("through_seq", 0)); target = head if through is None else min(int(through), head)
            events = [event for event in self._events() if event["seq"] <= target]
            return {"version": 1, "session_id": self.metadata["session_id"], "session_version": target,
                    "through_seq": target, "width": self.metadata["width"], "height": self.metadata["height"],
                    "recognition_mode": self.metadata["recognition_mode"], "model_version": self.metadata["model_version"],
                    "weights_sha256": self.metadata["weights_sha256"], "proposal_head": self.metadata.get("proposal_head", ""),
                    "event_count": len(events), "complete": len(events) == target, "events": events}

    def _events(self) -> list[dict[str, Any]]:
        files = sorted(self.events_dir.glob("seq-*.json"))
        if files:
            return [json.loads(path.read_text(encoding="utf-8")) for path in files]
        return list(self.legacy_events or [])

    @staticmethod
    def load_package(root: str | os.PathLike[str], session_id: str, through: int | None = None) -> dict[str, Any] | None:
        directory = Path(root) / session_id
        try:
            metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
            event_files = sorted((directory / "events").glob("seq-*.json"))
            if event_files:
                events = [json.loads(path.read_text(encoding="utf-8")) for path in event_files]
            else:
                events = json.loads((directory / "events.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        head = int(metadata.get("through_seq", 0)); target = head if through is None else min(int(through), head)
        state = metadata.get("state", "stopped")
        if state == "running": state = "interrupted"
        filtered = [event for event in events if event["seq"] <= target]
        return {"version": 1, "session_id": session_id, "session_version": target, "through_seq": target,
                "width": metadata["width"], "height": metadata["height"],
                "recognition_mode": metadata["recognition_mode"], "model_version": metadata["model_version"],
                "weights_sha256": metadata.get("weights_sha256", ""), "proposal_head": metadata.get("proposal_head", ""),
                "state": state, "event_count": len(filtered), "complete": len(filtered) == target,
                "events": filtered}

    def finish(self, state: str = "stopped") -> None:
        with self.lock:
            self.metadata["state"] = state; self._write(self.meta_path, self.metadata)

    def response(self, after_seq: int = 0, stats: dict[str, Any] | None = None) -> dict[str, Any]:
        with self.lock:
            through = int(self.metadata.get("through_seq", 0)); out = dict(self.metadata)
            submitted = int((stats or {}).get("submitted", 0)); out.update({"seq": through, "updated": through > int(after_seq), "lag": max(0, submitted - through), "depth": int((stats or {}).get("queue_depth", 0))})
            if through > int(after_seq):
                path = self.root / f"seq-{through:06d}.svg"; out["svg"] = path.read_text(encoding="utf-8") if path.is_file() else ""
            return out


class LiveInkSession:
    def __init__(self, root: str | os.PathLike[str], session_id: str, width: int, height: int,
                 mode: str, model_version: str, weights_sha256: str, runtime_factory: Callable):
        self.store = LiveSessionStore(root, session_id, width, height, mode, model_version, weights_sha256, "proposal_text")
        self.submit_condition = threading.Condition()
        self.next_submit_seq = 1
        self.pending_submissions = {}
        try:
            self.runtime = runtime_factory(self.store.write_result); self.runtime.start(session_id)
        except Exception:
            self.store.finish("failed")
            raise

    def submit(self, seq: int, observed_at: int, board, known) -> None:
        # HTTP requests may finish decoding out of order after the estimator
        # commits History. Hold each copied state until its predecessor has
        # entered OrderedInkRuntime; no session.lock is held while the bounded
        # runtime queue applies backpressure.
        with self.submit_condition:
            self.pending_submissions[int(seq)] = (observed_at, board, known)
            while int(seq) != self.next_submit_seq:
                if self.runtime.stats().state in {"failed", "stopped"}:
                    self.pending_submissions.pop(int(seq), None)
                    raise RuntimeError("live ink runtime stopped before ordered submit")
                self.submit_condition.wait(.05)
            observed_at, board, known = self.pending_submissions.pop(int(seq))
            self.runtime.submit(int(seq), observed_at, board, known)
            self.next_submit_seq += 1
            self.submit_condition.notify_all()

    def response(self, after_seq: int = 0) -> dict[str, Any]:
        s = self.runtime.stats()
        out = self.store.response(after_seq, {"submitted": s.submitted, "queue_depth": s.queue_depth})
        out.update({"status": s.state, "error": s.error, "submitted": s.submitted,
                    "completed": s.completed, "dropped": s.dropped})
        return out

    def stop(self) -> dict[str, Any]:
        try:
            self.runtime.stop(drain=True, timeout=30)
            if self.runtime.stats().state == "failed":
                self.store.finish("failed")
                raise RuntimeError(self.runtime.stats().error or "live ink runtime failed")
            self.store.finish("stopped")
        except Exception as exc:
            try:
                self.store.finish("failed")
            except Exception:
                pass
            raise RuntimeError(str(exc)) from exc
        return self.response()
