"""Build a conservative, editable ink timeline from confirmed board states.
The raster history remains authoritative.  This module derives a vector view
for analysis and export, and deliberately marks stroke direction as unknown
unless two successive confirmed states provide extension evidence.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _geometry_extractor():
    try:
        from ink_geometry import extract_paths
    except ImportError:
        from .ink_geometry import extract_paths  # type: ignore
    return extract_paths


def _points(stroke: dict[str, Any]) -> np.ndarray:
    value = np.asarray(stroke.get("points", []), dtype=np.float32)
    return value.reshape((-1, 2)) if value.size else np.empty((0, 2), np.float32)


def _canonical(points: np.ndarray, closed: bool = False) -> np.ndarray:
    """Remove contour/PCA sign flips from otherwise identical geometry."""
    if closed or len(points) < 2:
        return points
    if tuple(points[0]) > tuple(points[-1]):
        return points[::-1].copy()
    return points


def _bbox(points: np.ndarray) -> tuple[float, float, float, float]:
    lo, hi = points.min(axis=0), points.max(axis=0)
    return float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    if not len(a) or not len(b):
        return 0.0
    ax0, ay0, ax1, ay1 = _bbox(a)
    bx0, by0, bx1, by1 = _bbox(b)
    ix0, iy0, ix1, iy1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = max(1.0, (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter)
    return inter / union


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    """Approximate symmetric point distance in board pixels."""
    if not len(a) or not len(b):
        return math.inf
    def one(x: np.ndarray, y: np.ndarray) -> float:
        return float(np.min(np.linalg.norm(x[:, None, :] - y[None, :, :], axis=2), axis=1).mean())
    return (one(a, b) + one(b, a)) / 2.0


def _match(old: list[dict[str, Any]], new: list[dict[str, Any]], diagonal: float):
    pairs: list[tuple[int, int, float]] = []
    exact: dict[tuple, list[int]] = {}
    for oi, previous in enumerate(old):
        key = (previous.get("color"), round(float(previous.get("width", 0)), 2),
               bool(previous.get("closed")), tuple(tuple(round(float(v), 2) for v in point)
               for point in _canonical(_points(previous), bool(previous.get("closed")))))
        exact.setdefault(key, []).append(oi)
    exact_pairs: list[tuple[int, int, float]] = []
    exact_old, exact_new = set(), set()
    for ni, current in enumerate(new):
        key = (current.get("color"), round(float(current.get("width", 0)), 2),
               bool(current.get("closed")), tuple(tuple(round(float(v), 2) for v in point)
               for point in _canonical(_points(current), bool(current.get("closed")))))
        candidates = exact.get(key, [])
        if len(candidates) == 1:
            exact_pairs.append((candidates[0], ni, 0.0))
            exact_old.add(candidates[0])
            exact_new.add(ni)
    for oi, previous in enumerate(old):
        if oi in exact_old:
            continue
        a = _points(previous)
        for ni, current in enumerate(new):
            b = _points(current)
            if not len(a) or not len(b):
                continue
            if previous.get("color") != current.get("color"):
                continue
            distance = _distance(a, b)
            # A changed line usually keeps most of its box.  Allow a modest
            # displacement for perspective noise but reject unrelated marks.
            score = distance / max(1.0, diagonal) + (1.0 - _iou(a, b)) * 0.10
            if score <= 0.25:
                pairs.append((oi, ni, score))
    pairs.sort(key=lambda item: item[2])
    # If two old strokes are effectively indistinguishable for a new stroke,
    # do not invent an identity.  A conservative remove+add is preferable to
    # silently swapping two parallel lines and fabricating a pen direction.
    ambiguous: set[int] = set()
    for ni in range(len(new)):
        if ni in exact_new:
            continue
        candidates = [score for oi, candidate_ni, score in pairs if candidate_ni == ni]
        if len(candidates) > 1 and candidates[1] - candidates[0] < 0.03:
            ambiguous.add(ni)
    used_old, used_new, result = set(exact_old), set(exact_new), {ni: (oi, 0.0) for oi, ni, _ in exact_pairs}
    for oi, ni, score in pairs:
        if ni in ambiguous:
            continue
        if oi not in used_old and ni not in used_new:
            used_old.add(oi)
            used_new.add(ni)
            result[ni] = (oi, score)
    return result, ambiguous


def _direction(previous: np.ndarray, current: np.ndarray) -> tuple[str, str]:
    """Infer direction only when an endpoint clearly extends."""
    if len(previous) < 2 or len(current) < 2:
        return "unknown", "unknown"
    scale = max(2.0, float(np.linalg.norm(previous[-1] - previous[0])))
    start_delta = float(np.linalg.norm(current[0] - previous[0]))
    end_delta = float(np.linalg.norm(current[-1] - previous[-1]))
    if end_delta > max(2.0, start_delta * 1.8) and end_delta > scale * 0.025:
        return "forward", "inferred"
    if start_delta > max(2.0, end_delta * 1.8) and start_delta > scale * 0.025:
        return "reverse", "inferred"
    return "unknown", "unknown"


def _normalise(path: dict[str, Any], stroke_id: str, first_seq: int) -> dict[str, Any]:
    closed = bool(path.get("closed", False))
    points = [[round(float(x), 3), round(float(y), 3)] for x, y in _canonical(_points(path), closed)]
    value = {
        "id": stroke_id,
        "points": points,
        "color": str(path.get("color", "#222222")),
        "width": round(float(path.get("width", 1.0)), 3),
        "closed": closed,
        "kind": str(path.get("kind", "polyline")),
        "first_seq": int(first_seq),
        "direction": "unknown",
        "direction_source": "unknown",
    }
    return value


def build_ink(history, session_id: str, upto_seq: int | None = None) -> dict[str, Any]:
    """Derive an editable timeline from one History session.

    ``history.state`` is intentionally used as the stable public read API. It
    is cached by History for the head and remains correct after a reopen. This
    routine never mutates the database or treats vector point order as proof of
    a real pen direction.
    """
    sessions = {str(row["id"]): row for row in history.sessions()}
    if session_id not in sessions:
        raise KeyError(session_id)
    events = history.events(session_id)
    head = int(sessions[session_id]["seq"])
    target = head if upto_seq is None else int(upto_seq)
    if target < 0 or target > head:
        raise ValueError("upto_seq is outside the session")
    events = [event for event in events if int(event["seq"]) <= target]
    extractor = _geometry_extractor()
    _, initial_board, _ = history.state(session_id, 0)
    height, width = initial_board.shape[:2]
    active: dict[str, dict[str, Any]] = {}
    output_events: list[dict[str, Any]] = []
    next_id = 1
    previous_paths: list[dict[str, Any]] = []
    previous_ids: list[str] = []

    for event in events:
        seq = int(event["seq"])
        _, board, known = history.state(session_id, seq)
        height, width = board.shape[:2]
        paths = extractor(board, known)
        matching, ambiguous = _match(previous_paths, paths, math.hypot(width, height))
        changes = []
        current_ids: list[str] = []
        for index, path in enumerate(paths):
            if index in matching:
                old_index, _ = matching[index]
                stroke_id = previous_ids[old_index]
                stroke = _normalise(path, stroke_id, active[stroke_id]["first_seq"])
                direction, source = _direction(_points(previous_paths[old_index]), _points(path))
                if direction != "unknown":
                    stroke["direction"], stroke["direction_source"] = direction, source
                current_ids.append(stroke_id)
                if stroke != active[stroke_id]:
                    changes.append({"op": "update", "id": stroke_id, "stroke": stroke})
                active[stroke_id] = stroke
            else:
                stroke_id = f"stroke-{next_id:04d}"
                next_id += 1
                stroke = _normalise(path, stroke_id, seq)
                if index in ambiguous:
                    stroke["identity_ambiguous"] = True
                active[stroke_id] = stroke
                current_ids.append(stroke_id)
                changes.append({"op": "add", "id": stroke_id, "stroke": stroke})
        for old_index, stroke_id in enumerate(previous_ids):
            if old_index not in {value[0] for value in matching.values()}:
                active.pop(stroke_id, None)
                changes.append({"op": "remove", "id": stroke_id})
        if changes:
            output_events.append({
                "seq": seq,
                "observed_at": int(event["observed_at"]),
                "confirmed_at": int(event["confirmed_at"]),
                "changes": changes,
            })
        previous_paths, previous_ids = paths, current_ids

    return {
        "version": 1,
        "session_id": session_id,
        "width": width,
        "height": height,
        "through_seq": target,
        "events": output_events,
    }
