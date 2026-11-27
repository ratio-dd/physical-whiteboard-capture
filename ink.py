"""Track visible ink components across confirmed states of a physical whiteboard."""

from __future__ import annotations

import base64
from collections import Counter, OrderedDict, defaultdict
from functools import cached_property

import cv2
import numpy as np


def _geometry_extractor():
    if __package__:
        from .ink_geometry import make_path_extractor
    else:
        from ink_geometry import make_path_extractor
    return make_path_extractor()


def _line_key(points, closed=False):
    """Order-independent comparison key. Export the author's coordinates unchanged."""
    line = tuple(map(tuple, points))
    if not closed or len(line) < 2:
        return min(line, line[::-1])
    if line[0] == line[-1]:
        line = line[:-1]
    first = min(line)
    # A ring has neither a privileged starting vertex nor a traversal direction.
    return min(order[i:] + order[:i] for order in (line, line[::-1])
               for i, point in enumerate(order) if point == first)


def _ring_keys(rings):
    return tuple(sorted(_line_key(ring, True) for ring in rings))


class _Shape:
    def __init__(self, path):
        self.path = path
        self.line = _line_key(path["points"], path.get("closed", False))
        self.rings = _ring_keys(path.get("outline", []))
        self.branches = tuple(sorted((branch.get("closed", False),
                                      _line_key(branch["points"], branch.get("closed", False)))
                                     for branch in path.get("centerlines", [])))
        # Rings within an even-odd fill may be reordered, paint layers may not.
        self.layers = tuple((layer["color"], _ring_keys(layer["outline"])) for layer in path.get("layers", []))
        self.style = {k: v for k, v in path.items() if k not in ("points", "outline", "centerlines", "layers")}

    @property
    def location(self):
        return ("outline", self.rings) if self.rings else ("line", self.line)

    def same(self, other):
        return (self.line == other.line and self.rings == other.rings
                and self.branches == other.branches and self.layers == other.layers
                and self.style == other.style)

    @cached_property
    def support(self):
        """Tight, board-aligned ink footprint for matching changed objects."""
        rings = self.path.get("outline", [])
        branches = self.path.get("centerlines") or [self.path]
        contours = rings or [branch["points"] for branch in branches]
        polygons = [np.rint(points).astype(np.int32) for points in contours if len(points)]
        if not polygons:
            return 0, 0, np.zeros((0, 0), np.uint8), 0
        x, y, w, h = cv2.boundingRect(np.concatenate(polygons))
        local = [poly - np.array([x, y], np.int32) for poly in polygons]
        pixels = np.zeros((h, w), np.uint8)
        if rings:
            # Drawing all rings together retains the holes under the even-odd rule.
            cv2.fillPoly(pixels, local, 1)
        else:
            for poly, branch in zip(local, branches):
                cv2.polylines(pixels, [poly], bool(branch.get("closed")), 1, 1)
        return x, y, pixels, cv2.countNonZero(pixels)

    def coverage(self, other):
        x, y, a, area_a = self.support
        u, v, b, area_b = other.support
        left, top = max(x, u), max(y, v)
        right, bottom = min(x + a.shape[1], u + b.shape[1]), min(y + a.shape[0], v + b.shape[0])
        if right <= left or bottom <= top or not min(area_a, area_b):
            return 0.0, 0.0
        a = a[top - y:bottom - y, left - x:right - x]
        b = b[top - v:bottom - v, left - u:right - u]
        common = cv2.countNonZero(cv2.bitwise_and(a, b))
        return common / area_a, common / area_b


def _match(old, new):
    """Reserve exact geometry, then match overlapping edits one-to-one."""
    old_locations, new_locations = defaultdict(list), defaultdict(list)
    for i, shape in enumerate(old):
        old_locations[shape.location].append(i)
    for i, shape in enumerate(new):
        new_locations[shape.location].append(i)

    matches = {}
    for location, indices in new_locations.items():
        previous = old_locations.get(location, [])
        if len(indices) == len(previous) == 1:
            matches[indices[0]] = previous[0]
        elif previous:
            # Coincident objects can still be distinguished by their paint.
            # Preserve identical copies as a group only while their multiplicity agrees.
            unclaimed = list(previous)
            for ni in indices:
                equivalents = [oi for oi in unclaimed if new[ni].same(old[oi])]
                if not equivalents:
                    continue
                if len(previous) != len(indices):
                    copies = sum(new[ni].same(new[i]) for i in indices)
                    if len(equivalents) != 1 or copies != 1:
                        continue
                matches[ni] = equivalents[0]
                unclaimed.remove(equivalents[0])
            unmatched = [ni for ni in indices if ni not in matches]
            if len(unclaimed) == len(unmatched) == 1:
                matches[unmatched[0]] = unclaimed[0]

    used = set(matches.values())
    candidates = []
    ambiguous = set()
    remaining = [i for i in range(len(old)) if i not in used]
    for ni, current in enumerate(new):
        if ni in matches:
            continue
        options = []
        for oi in remaining:
            old_fraction, new_fraction = old[oi].coverage(current)
            # Require actual shared board pixels. A close parallel line or
            # similar colour elsewhere is insufficient evidence of identity.
            if max(old_fraction, new_fraction) < 0.55 or min(old_fraction, new_fraction) < 0.10:
                continue
            dice = 2 * old_fraction * new_fraction / (old_fraction + new_fraction)
            options.append((dice, oi, ni))
        options.sort(reverse=True)
        if not options:
            continue
        close = [item for item in options if options[0][0] - item[0] < 0.04]
        if len(close) > 1:
            # Overlapping pens may have the same footprint. Use a unique
            # matching appearance only to settle otherwise indistinct overlap.
            close = [item for item in close
                     if all(old[item[1]].path.get(k) == current.path.get(k) for k in ("color", "width"))]
        if len(close) == 1:
            candidates.append(close[0])
        else:
            ambiguous.add(ni)

    # On a split only the strongest surviving fragment can carry the old ID.
    # On an indistinguishable merge start a new object with uncertain identity.
    for _, oi, ni in sorted(candidates, reverse=True):
        if oi not in used:
            matches[ni] = oi
            used.add(oi)
    return matches, ambiguous


def _direction(old, new, previous_known, previous_direction):
    """Endpoint extension evidence, only on previously visible, unbranched ink."""
    if (old.path.get("closed") or new.path.get("closed")
            or len(old.branches) > 1 or len(new.branches) > 1):
        return "unknown"
    if old.line == new.line and old.branches == new.branches:
        if old.path["points"] == new.path["points"]:
            return previous_direction
        return {"forward": "reverse", "reverse": "forward"}.get(previous_direction, "unknown")
    a = np.asarray(old.path["points"], np.float64)
    b = np.asarray(new.path["points"], np.float64)
    if min(len(a), len(b)) < 2:
        return "unknown"
    growth = np.linalg.norm(np.diff(b, axis=0), axis=1).sum() - np.linalg.norm(np.diff(a, axis=0), axis=1).sum()
    if growth < 2 or old.coverage(new)[0] < 0.90:
        return "unknown"
    x, y, support, _ = new.support
    was_known = previous_known[y:y + support.shape[0], x:x + support.shape[1]]
    if was_known.shape != support.shape or np.any((was_known == 0) & (support != 0)):
        return "unknown"
    ends = a[[0, -1]]
    target = b[[0, -1]]
    if np.linalg.norm(ends[::-1] - target, axis=1).sum() < np.linalg.norm(ends - target, axis=1).sum():
        ends = ends[::-1]
    start, end = np.linalg.norm(ends - target, axis=1)
    if start <= 1.5 and end >= max(2, start * 2):
        return "forward"
    if end <= 1.5 and start >= max(2, end * 2):
        return "reverse"
    return "unknown"


def _last_pixel_change(board, known, previous_board, previous_known, last_change, seq):
    # Record changes even if a later frame restores exactly the same pixels.
    equal_bgr = cv2.inRange(cv2.absdiff(board, previous_board), (0, 0, 0), (0, 0, 0))
    equal_mask = cv2.compare(known, previous_known, cv2.CMP_EQ)
    last_change[cv2.bitwise_and(equal_bgr, equal_mask) == 0] = seq


def _reappeared(shape, retired, last_change):
    entry = retired.pop(shape.location, None)
    if entry is None:
        return None
    stroke_id, first_seq, last_seen = entry
    x, y, pixels, _ = shape.support
    h, w = pixels.shape
    # Include the antialiased boundary. Reuse is conservative when neighboring
    # content inside the same small bounding rectangle changed during the gap.
    window = last_change[max(0, y - 1):y + h + 1, max(0, x - 1):x + w + 1]
    if window.size and window.max() <= last_seen:
        return stroke_id, first_seq
    return None


def _decode_pair(item):
    def image(key, mode):
        data = np.frombuffer(base64.b64decode(item[key]), np.uint8)
        return cv2.imdecode(data, mode)
    return image("board", cv2.IMREAD_COLOR), image("known", cv2.IMREAD_GRAYSCALE)


def _states(history, session_id, target):
    """Advance through each stored snapshot/patch once, using the public replay API."""
    if target == 0 or not hasattr(history, "replay"):
        for seq in range(target + 1):
            yield history.state(session_id, seq)
        return
    seq = 0
    while seq <= target:
        segment = history.replay(session_id, seq)
        board, known = _decode_pair(segment["snapshot"])
        yield seq, board, known
        for frame in segment["frames"]:
            seq = frame["seq"]
            if seq > target:
                return
            for patch in frame["patches"]:
                pixels, mask = _decode_pair(patch)
                x, y = patch["x"], patch["y"]
                h, w = mask.shape
                board[y:y + h, x:x + w] = pixels
                known[y:y + h, x:x + w] = mask
            yield seq, board, known
        seq = segment["end"] + 1


def build_ink(history, session_id: str, upto_seq: int | None = None) -> dict:
    """Return v1 add/update/remove events; each ID denotes a visible ink object."""
    session = next((s for s in history.sessions() if s["id"] == session_id), None)
    if session is None:
        raise KeyError(session_id)
    head = session["seq"]
    target = head if upto_seq is None else int(upto_seq)
    if not 0 <= target <= head:
        raise ValueError("upto_seq is outside the session")
    stamps = {e["seq"]: e for e in history.events(session_id)}
    stamps[0] = {"observed_at": session["created_at"], "confirmed_at": session["created_at"]}
    extract = _geometry_extractor()
    old_shapes, old_strokes = [], []
    previous_board = previous_known = None
    retired = OrderedDict()
    output = []
    next_id = 1
    for seq, board, known in _states(history, session_id, target):
        if seq == 0:
            last_change = np.zeros(known.shape, np.int64)
        else:
            _last_pixel_change(board, known, previous_board, previous_known, last_change, seq)
        paths = extract(board, known) if cv2.countNonZero(known) else []
        shapes = [_Shape(path) for path in paths]
        matches, ambiguous = _match(old_shapes, shapes)
        active_locations = Counter(s.location for s in shapes)
        changes, strokes = [], []
        for ni, shape in enumerate(shapes):
            oi = matches.get(ni)
            if oi is None:
                identity = None
                if ni not in ambiguous and active_locations[shape.location] == 1:
                    identity = _reappeared(shape, retired, last_change)
                if identity is None:
                    identity = f"stroke-{next_id:04d}", seq
                    next_id += 1
                stroke_id, first_seq = identity
                stroke = {**shape.path, "id": stroke_id, "first_seq": first_seq,
                          "direction": "unknown", "direction_source": "unknown"}
                if ni in ambiguous:
                    stroke["identity_ambiguous"] = True
                op = "add"
            else:
                previous = old_strokes[oi]
                old = old_shapes[oi]
                if old.same(shape):
                    strokes.append(previous)
                    shapes[ni] = old
                    continue
                stroke_id = previous["id"]
                direction = _direction(old, shape, previous_known, previous["direction"])
                stroke = {**shape.path, "id": stroke_id, "first_seq": previous["first_seq"],
                          "direction": direction,
                          "direction_source": "inferred" if direction != "unknown" else "unknown"}
                if previous.get("identity_ambiguous"):
                    stroke["identity_ambiguous"] = True
                op = "update"
            strokes.append(stroke)
            changes.append({"op": op, "id": stroke_id, "stroke": stroke})
        inherited = set(matches.values())
        counts = Counter(s.location for s in old_shapes)
        for location in active_locations:
            retired.pop(location, None)
        for i, previous in enumerate(old_strokes):
            if i in inherited:
                continue
            changes.append({"op": "remove", "id": previous["id"]})
            location = old_shapes[i].location
            if counts[location] == 1 and location not in active_locations and not previous.get("identity_ambiguous"):
                retired[location] = previous["id"], previous["first_seq"], seq - 1
                retired.move_to_end(location)
        while len(retired) > 256:
            retired.popitem(last=False)
        if changes:
            output.append({"seq": seq, "observed_at": stamps[seq]["observed_at"],
                           "confirmed_at": stamps[seq]["confirmed_at"], "changes": changes})
        old_shapes, old_strokes = shapes, strokes
        previous_board, previous_known = board.copy(), known.copy()
    height, width = known.shape
    return {"version": 1, "session_id": session_id, "width": width, "height": height,
            "through_seq": target, "events": output}
