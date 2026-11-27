"""Extract conservative, image-space electronic ink paths from a board state.

This module deliberately reports geometry that is visible in one confirmed
board image.  It does not infer pen order or invent pixels in unknown areas.
"""
from __future__ import annotations

import cv2
import numpy as np


def _mask(board: np.ndarray, known: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    # A large closing estimates the local board colour without requiring a
    # second reference frame.  The saturation branch preserves red/blue ink.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    # Reject broad low-frequency paper variation.  A dark candidate must be
    # both locally darker and sufficiently separated from the white surface.
    dark = ((background.astype(np.int16) - gray.astype(np.int16)) > 18) & (gray < 220)
    hsv = cv2.cvtColor(board, cv2.COLOR_BGR2HSV)
    coloured = (hsv[..., 1] > 42) & (hsv[..., 2] < 245) & (gray < 235)
    mask = (dark | coloured) & (known == 255)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return mask


def _colour(board: np.ndarray, pixels: np.ndarray) -> str:
    if pixels.size == 0:
        return "#000000"
    bgr = np.median(board[pixels[:, 1], pixels[:, 0]], axis=0)
    rgb = np.clip(np.rint(bgr[::-1]), 0, 255).astype(int)
    return "#%02x%02x%02x" % tuple(rgb)


def _thin(binary: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning, keeping the 8-connected topology."""
    image = (binary > 0).astype(np.uint8)
    changed = True
    while changed:
        changed = False
        for phase in (0, 1):
            p = np.pad(image, 1)
            n = [p[1 + dy:1 + dy + image.shape[0], 1 + dx:1 + dx + image.shape[1]]
                 for dx, dy in ((0, -1), (1, -1), (1, 0), (1, 1),
                                (0, 1), (-1, 1), (-1, 0), (-1, -1))]
            transitions = sum((n[i] == 0) & (n[(i + 1) % 8] == 1) for i in range(8))
            count = sum(n)
            if phase == 0:
                condition = (n[0] * n[2] * n[4] == 0) & (n[2] * n[4] * n[6] == 0)
            else:
                condition = (n[0] * n[2] * n[6] == 0) & (n[0] * n[4] * n[6] == 0)
            remove = (image == 1) & (count >= 2) & (count <= 6) & (transitions == 1) & condition
            if np.any(remove):
                image[remove] = 0
                changed = True
    return image


def _component_paths(component: np.ndarray) -> list[tuple[list[list[int]], bool, str, float]]:
    ys, xs = np.where(component)
    points = np.column_stack((xs, ys)).astype(np.float32)
    if len(points) < 2:
        return []
    # Morphological skeletonization gives a centreline without requiring the
    # optional contrib/ximgproc package.  It preserves curves and junctions,
    # while the original component area remains available for pen width.
    skeleton = _thin(component)
    sy, sx = np.where(skeleton)
    if len(sx) < 2:
        return []
    pixels = {(int(x), int(y)) for x, y in zip(sx, sy)}
    neighbours = lambda p: [q for q in ((p[0] + dx, p[1] + dy)
                                        for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                                        if (dx or dy)) if q in pixels]
    degree = {p: len(neighbours(p)) for p in pixels}
    ends = [p for p, d in degree.items() if d == 1]
    node_pixels = {p for p, d in degree.items() if d != 2}
    # Contract adjacent junction/endpoint pixels into node clusters. This
    # removes 2x2 diagonal ambiguity before walking graph edges.
    clusters = []
    remaining = set(node_pixels)
    while remaining:
        seed = remaining.pop(); stack = [seed]; cluster = {seed}
        while stack:
            q = stack.pop()
            for r in neighbours(q):
                if r in remaining:
                    remaining.remove(r); cluster.add(r); stack.append(r)
        clusters.append(cluster)
    node_of = {p: i for i, cluster in enumerate(clusters) for p in cluster}
    visited_edges = set(); paths = []
    # Components are tightly cropped and often touch their crop edges. Without
    # an explicit zero border, distanceTransform treats a long horizontal mark
    # as if it continued forever and reports a radius proportional to length.
    padded = cv2.copyMakeBorder(component.astype(np.uint8), 1, 1, 1, 1,
                                cv2.BORDER_CONSTANT, value=0)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
    def add_path(points_path, closed):
        if len(points_path) < 2: return
        radii = [distance[y, x] for x, y in points_path if 0 <= y < distance.shape[0] and 0 <= x < distance.shape[1]]
        width = max(1.0, float(2 * np.median(radii))) if radii else 1.0
        paths.append(([[int(x), int(y)] for x, y in points_path], closed,
                      "loop" if closed else ("line" if len(points_path) == 2 else "polyline"), width))
    if clusters:
        for cluster_id, cluster in enumerate(clusters):
            for start in cluster:
                for first in neighbours(start):
                    if first in cluster: continue
                    key = tuple(sorted((cluster_id, node_of[first]))) if first in node_of else None
                    current, previous = first, start; path = [start, current]
                    while current not in node_of:
                        options = [q for q in neighbours(current) if q != previous]
                        if not options: break
                        previous, current = current, options[0]; path.append(current)
                    end_id = node_of.get(current)
                    edge_key = (cluster_id, end_id) if end_id is not None else (cluster_id, tuple(path[-1]))
                    edge_key = tuple(sorted(edge_key, key=str))
                    if edge_key in visited_edges: continue
                    visited_edges.add(edge_key); add_path(path, False)
    else:
        # A pure loop has degree-two pixels only. Walk once until returning.
        start = min(pixels); path = [start]; previous = None; current = start
        while True:
            options = [q for q in neighbours(current) if q != previous and q != start]
            if not options: break
            previous, current = current, options[0]; path.append(current)
        add_path(path, True)
    # A pure loop can have corner pixels with degree 3 under 8-connectivity,
    # so use the absence of true endpoints as the topology signal as well.
    if not ends and paths:
        first = paths[0]
        paths[0] = (first[0], True, "loop", first[3])
    elif len(ends) == 2:
        paths = [(points, closed, "line" if not closed else kind, width)
                 for points, closed, kind, width in paths]
    return paths


def extract_paths(board: np.ndarray, known: np.ndarray) -> list[dict]:
    """Return conservative vector-like paths in rectified board coordinates.

    ``known`` is a uint8 mask with values 0 or 255.  Paths are independent
    visible components, with no temporal or pen-order claim.
    """
    if not isinstance(board, np.ndarray) or board.dtype != np.uint8 or board.ndim != 3 or board.shape[2] != 3:
        raise ValueError("board must be a uint8 BGR image")
    if not isinstance(known, np.ndarray) or known.dtype != np.uint8 or known.shape != board.shape[:2]:
        raise ValueError("known must be an H×W uint8 mask matching board")
    if np.any((known != 0) & (known != 255)):
        raise ValueError("known values must be 0 or 255")
    mask = _mask(board, known)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    paths: list[dict] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x, y, w, h = (int(stats[label, i]) for i in range(4))
        if area < 8 or w < 2 or h < 2 or area > board.shape[0] * board.shape[1] * 0.25:
            continue
        # Camera/warp borders often leave a 1–2 px bright or dark rail.  A
        # rail that hugs the outer edge for most of the board and never turns
        # inward is acquisition artefact, not electronic ink.  Keep near-edge
        # strokes with real width or an inward diagonal extension.
        touches_edge = x <= 1 or y <= 1 or x + w >= board.shape[1] - 1 or y + h >= board.shape[0] - 1
        rail = min(w, h) <= 3 and max(w, h) >= 20
        if touches_edge and rail:
            continue
        component = labels[y:y + h, x:x + w] == label
        contours, _ = cv2.findContours(component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        pixel_points = np.column_stack(np.where(component)[::-1]) + np.array([x, y])
        for points, closed, kind, width in _component_paths(component):
            if len(points) < 2:
                continue
            points = [[float(px + x), float(py + y)] for px, py in points]
            xs_path = [point[0] for point in points]
            ys_path = [point[1] for point in points]
            # Apply the rail test to the actual walked edge too. A diagonal
            # touching the border is retained because its path enters inward.
            if (max(xs_path) >= board.shape[1] - 2 and min(xs_path) >= board.shape[1] - 3
                    and max(ys_path) - min(ys_path) >= 20) or (
                    max(ys_path) >= board.shape[0] - 2 and min(ys_path) >= board.shape[0] - 3
                    and max(xs_path) - min(xs_path) >= 20):
                continue
            paths.append({"points": points, "color": _colour(board, pixel_points),
                          "width": round(width, 2), "closed": bool(closed), "kind": kind,
                          "evidence": "visible_component", "area": area})
    paths.sort(key=lambda item: (item["points"][0][1], item["points"][0][0]))
    return paths
