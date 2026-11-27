"""Extract conservative, image-space electronic ink paths from a board state.

This module deliberately reports geometry that is visible in one confirmed
board image.  It does not infer pen order or invent pixels in unknown areas.
"""
from __future__ import annotations

from collections import OrderedDict
import json

import cv2
import numpy as np


def _maximum_channel(image):
    b, g, r = cv2.split(image)
    return cv2.max(b, cv2.max(g, r))


def _glare_shoulders(smooth, known, strength):
    """Measure stripe-adjacent marks against paper along the stripe itself."""
    bright = cv2.inRange(smooth, (249, 249, 249), (255, 255, 255))
    luma = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    highlight = cv2.morphologyEx(luma, cv2.MORPH_TOPHAT, np.ones((13, 13), np.uint8))
    bright[(luma >= 244) & (highlight >= 5)] = 255
    # White placeholders can connect a thin reflection to a large rectangle,
    # changing its measured aspect ratio even far from the unknown pixels.
    bright = cv2.bitwise_and(bright, known)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((3, 7), np.uint8))
    bright = cv2.bitwise_and(bright, known)
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = known.shape
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        short, long = sorted(rect[1])
        if short < 1 or long < max(30, 6 * short) or short > min(w, h) * 0.15:
            continue
        # Follow the long side, regardless of the camera's roll angle.
        corners = cv2.boxPoints(rect)
        sides = np.roll(corners, -1, axis=0) - corners
        tangent = sides[np.argmax(np.sum(sides * sides, axis=1))] / long
        dx, dy = np.rint(tangent * 12).astype(int)
        kernel = np.zeros((2 * abs(dy) + 1, 2 * abs(dx) + 1), np.uint8)
        cv2.line(kernel, (abs(dx)-dx, abs(dy)-dy), (abs(dx)+dx, abs(dy)+dy), 1)
        pad = max(7, min(18, int(short) + 4))
        # A specular streak tapers below the clipping level at its two ends.
        # Test these shoulders against the same oriented background as its core.
        tx, ty = np.rint(tangent * min(32, 0.55 * long)).astype(int)
        extension = np.zeros((2*abs(ty)+1, 2*abs(tx)+1), np.uint8)
        cv2.line(extension, (abs(tx)-tx, abs(ty)-ty), (abs(tx)+tx, abs(ty)+ty), 1)
        x, y, ww, hh = cv2.boundingRect(contour)
        px, py = pad + 13 + abs(tx), pad + 13 + abs(ty)
        x0, y0 = max(0, x-px), max(0, y-py)
        roi = np.s_[y0:min(h,y+hh+py), x0:min(w,x+ww+px)]
        stripe = np.zeros(strength[roi].shape, np.uint8)
        cv2.drawContours(stripe, [contour], 0, 1, -1, offset=(-x0, -y0))
        stripe = cv2.dilate(stripe, extension)
        near = cv2.dilate(stripe, np.ones((2*pad+1, 2*pad+1), np.uint8)) != 0
        along = cv2.morphologyEx(smooth[roi], cv2.MORPH_CLOSE, kernel)
        contrast = _maximum_channel(cv2.subtract(along, smooth[roi]))
        permitted = cv2.max(cv2.subtract(contrast, 5), cv2.subtract(strength[roi], 30))
        strength[roi][near] = np.minimum(strength[roi][near], permitted[near])


def _flat_patch_mask(detail, deficit):
    """Find filled, low-contrast neutral rectangles caused by tile exposure steps."""
    _, labels, stats, _ = cv2.connectedComponentsWithStats((detail >= 6).astype(np.uint8), 8)
    flat = np.zeros(len(stats), bool)
    for label in range(1, len(stats)):
        x, y, w, h, area = map(int, stats[label])
        if min(w, h) < 7 or area < 0.88 * w * h:
            continue
        roi = np.s_[y:y+h, x:x+w]
        values = deficit[roi][labels[roi] == label]
        level = values.max(axis=1)
        if np.percentile(level, 95) <= 30 and np.percentile(level-values.min(axis=1), 90) <= 5:
            flat[label] = True
    return flat[labels]


def _faint_ridges(response):
    """Keep low-amplitude tails supported along a line and lighter on its flanks."""
    result = np.zeros(response.shape, bool)
    for dx, dy in ((4, 0), (4, 2), (3, 3), (2, 4), (0, 4), (-2, 4), (-3, 3), (-4, 2)):
        kernel = np.zeros((9, 9), np.float32)
        cv2.line(kernel, (4-dx, 4-dy), (4+dx, 4+dy), 1)
        average = cv2.filter2D(response, cv2.CV_32F, kernel / kernel.sum())
        # Compare with parallel lines on each side of the candidate.
        nx, ny = np.rint(np.array([-dy, dx]) * (3 / np.hypot(dx, dy))).astype(int)
        sample = np.zeros((2*abs(ny)+1, 2*abs(nx)+1), np.uint8)
        sample[abs(ny)+ny, abs(nx)+nx] = sample[abs(ny)-ny, abs(nx)-nx] = 1
        flank = cv2.dilate(average, sample, borderType=cv2.BORDER_REPLICATE)
        result |= (average >= 4.0) & (average - flank >= 2.5)
    return (response >= 3) & result


def _ink_response(board, known):
    smooth = cv2.GaussianBlur(board, (3, 3), 0.65)
    ridge = np.zeros(known.shape, np.uint8)
    # A stroke has lighter shoulders on both sides in at least one direction.
    # A single exposure step or smooth shading normally has only one.
    for radius in (2, 4, 7):
        for dx, dy in ((radius, 0), (0, radius), (radius, radius), (radius, -radius)):
            kernel = np.zeros((2 * abs(dy) + 1, 2 * abs(dx) + 1), np.uint8)
            kernel[abs(dy) + dy, abs(dx) + dx] = 1
            kernel[abs(dy) - dy, abs(dx) - dx] = 1
            shoulders = cv2.erode(smooth, kernel, borderType=cv2.BORDER_REPLICATE)
            ridge = cv2.max(ridge, _maximum_channel(cv2.subtract(shoulders, smooth)))

    paper = cv2.morphologyEx(smooth, cv2.MORPH_CLOSE, np.ones((23, 23), np.uint8))
    deficit = cv2.subtract(paper, smooth)
    # A border exposure seam shifts an entire scanline's paper color. Subtract
    # the common offset, preserving local contrast of genuine near-edge marks.
    depth = max(2, round(min(known.shape) * 0.045))
    for view in (deficit, deficit.swapaxes(0, 1)):
        for strip in (view[:depth], view[-depth:]):
            level = np.median(strip, axis=1, keepdims=True).astype(np.uint8)
            strip[:] = np.maximum(strip, level) - level
    detail = _maximum_channel(deficit)
    # Rescue centers of broad, high-contrast marker strokes beyond the ridge scales.
    ridge = cv2.max(ridge, cv2.subtract(detail, 28))
    _glare_shoulders(smooth, known, ridge)
    valid = (known == 255) & ~_flat_patch_mask(detail, deficit)
    weak = (ridge >= 5) & (detail >= 7) & valid
    # The high-contrast core already has a measured silhouette. Do not grow
    # JPEG ringing beside it while following a genuinely fading stroke.
    edge = cv2.dilate((ridge >= 30).astype(np.uint8), np.ones((5, 5), np.uint8))
    weak |= _faint_ridges(cv2.min(ridge, detail)) & valid & (edge == 0)
    strong = (ridge >= 12) & (detail >= 13) & valid
    count, labels, stats, _ = cv2.connectedComponentsWithStats(weak.astype(np.uint8), 8)
    seeds = np.bincount(labels[strong], minlength=count)
    keep = (seeds >= 2) & (stats[:, cv2.CC_STAT_AREA] >= 5)
    keep[0] = False
    return keep[labels].astype(np.uint8), deficit


def _mask(board: np.ndarray, known: np.ndarray) -> np.ndarray:
    return _ink_response(board, known)[0]


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


def _centerlines(component, x0, y0):
    """Walk each skeleton *edge* once, including parallel routes at junctions."""
    yy, xx = np.where(_thin(component))
    pixels = set(zip(map(int, xx), map(int, yy)))
    graph = {}
    for x, y in sorted(pixels):
        adjacent = []
        for dx, dy in ((-1,0), (0,-1), (0,1), (1,0), (-1,-1), (-1,1), (1,-1), (1,1)):
            q = (x+dx, y+dy)
            if q not in pixels:
                continue
            # In an L of three pixels the diagonal shortcut creates a false loop.
            if dx and dy and ((x+dx,y) in pixels or (x,y+dy) in pixels):
                continue
            adjacent.append(q)
        graph[(x,y)] = adjacent

    used = set()
    branches = []
    starting = [p for p in graph if len(graph[p]) != 2]
    for start in starting + list(graph):
        for neighbour in graph[start]:
            first_edge = tuple(sorted((start, neighbour)))
            if first_edge in used:
                continue
            used.add(first_edge)
            path = [start, neighbour]
            previous, current = path
            while current != start and len(graph[current]) == 2:
                a, b = graph[current]
                following = b if a == previous else a
                edge = tuple(sorted((current, following)))
                if edge in used:
                    break
                used.add(edge)
                path.append(following)
                previous, current = current, following
            closed = path[0] == path[-1]
            pts = np.asarray(path[:-1] if closed else path, np.float32)
            if len(pts) < 2:
                continue
            length = cv2.arcLength(pts, closed)
            pts = cv2.approxPolyDP(pts, 0.5, closed).reshape(-1, 2)
            if len(pts) >= 2:
                points = (pts + (x0 + 0.5, y0 + 0.5)).tolist()
                branches.append((length, {"points": points, "closed": closed}))
    if not branches:
        # A round dot may thin to one pixel. Give its observed longest axis a
        # short centerline for older consumers; the outline retains its footprint.
        y, x = np.where(component)
        cx, cy = x.mean() + x0 + 0.5, y.mean() + y0 + 0.5
        branches.append((0, {"points": [[round(cx-0.25,2), round(cy,2)],
                                        [round(cx+0.25,2), round(cy,2)]], "closed": False}))
    branches.sort(key=lambda item: -item[0])
    return [branch for _, branch in branches]


def _rings(mask, x, y):
    # Interpolate the binary footprint on a finer grid, padding with background.
    # A pixel spans [x,x+1] in SVG; its center is at x+0.5 in board coordinates.
    scale = 3
    padded = np.pad(mask.astype(np.float32), 1)
    high = cv2.resize(padded, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    contours, _ = cv2.findContours((high >= 0.42).astype(np.uint8), cv2.RETR_LIST,
                                    cv2.CHAIN_APPROX_SIMPLE)
    rings = []
    for contour in contours:
        pts = cv2.approxPolyDP(contour, 0.33 * scale, True).reshape(-1, 2)
        if len(pts) >= 3:
            pts = (pts.astype(np.float64) + 0.5) / scale + (x - 1, y - 1)
            rings.append(np.round(pts, 2).tolist())
    return rings


def _pigments(component, deficit):
    """Split touching inks only when coherent regions support distinct hues."""
    peak = _maximum_channel(deficit).astype(np.float32)
    spread = peak - np.min(deficit, axis=2)
    hue = cv2.cvtColor(255 - deficit, cv2.COLOR_BGR2HSV)[..., 0].astype(np.int16)
    hue = 1 + ((hue + 15) // 30) % 6
    chromatic = (spread >= 7) & (spread >= 0.33 * peak)
    neutral = (spread <= np.maximum(4, 0.17 * peak)) & (peak >= 18)
    types = np.where(chromatic, hue, np.where(neutral, 0, -1))
    types[~component] = -1
    minimum = max(9, round(component.sum() * 0.017))
    seeds = []
    for ink in np.unique(types[types >= 0]):
        sample = (types == ink).astype(np.uint8)
        if sample.sum() < minimum:
            continue
        count, labels, stats, _ = cv2.connectedComponentsWithStats(sample, 8)
        significant = stats[:, cv2.CC_STAT_AREA] >= max(4, minimum // 4)
        significant[0] = False
        sample = significant[labels]
        if sample.sum() >= minimum:
            seeds.append((ink, sample))
    if len(seeds) < 2:
        yield component, (seeds[0][0] if seeds else None)
        return
    # Weak antialiasing at a crossing takes the hue of its nearest reliable
    # color sample. Keep the original foreground support unchanged.
    distances = [cv2.distanceTransform((~seed).astype(np.uint8), cv2.DIST_L2, 3)
                 for _, seed in seeds]
    owner = np.argmin(distances, axis=0)
    for i, (ink, _) in enumerate(seeds):
        yield component & (owner == i), ink


def _display_color(direction, strength):
    gain = min(245 / max(strength, 1), 1.1 + 3.6 * np.exp(-strength / 87))
    # Expand the lowest reproducible optical densities on the white SVG page.
    density = min(245, strength * gain + min(26, 1.6*strength) * np.exp(-strength/29))
    rgb = np.rint(255 - direction[::-1] * density).clip(0, 255).astype(int)
    return "#%02x%02x%02x" % tuple(rgb)


def _layers(component, deficit, outline, x, y, ink):
    samples = deficit[component].astype(np.float32)
    strengths = samples.max(axis=1)
    top = max(1, float(np.percentile(strengths, 93)))
    core = strengths >= max(1, top * 0.60)
    if ink == 0 or not core.any() or (ink is None and top < 18):
        direction = np.ones(3)
    else:
        direction = np.median(samples[core] / strengths[core, None], axis=0)
    pigment = _display_color(direction, top)
    levels = [0, max(11, top * 0.24), max(16, top * 0.47), max(21, top * 0.73)]
    pixel_strength = _maximum_channel(deficit)
    layers = []
    for i, low in enumerate(levels):
        band = strengths >= low
        if i < len(levels)-1:
            band &= strengths < levels[i+1]
        if not band.any():
            continue
        rings = outline if i == 0 else _rings(component & (pixel_strength >= low), x, y)
        if rings:
            color = _display_color(direction, float(np.median(strengths[band])))
            layers.append({"outline": rings, "color": color})
    return pigment, layers


def _stroke(component, deficit, x, y, ink):
    outline = _rings(component, x, y)
    if not outline:
        return None
    centerlines = _centerlines(component, x, y)
    primary = centerlines[0]
    padded = np.pad(component.astype(np.uint8), 1)
    radius = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1,1:-1]
    width = max(1, round(2 * float(np.percentile(radius[component], 75)) - 1, 2))
    color, layers = _layers(component, deficit, outline, x, y, ink)
    pts = np.array(primary["points"], np.float32)
    straight = cv2.arcLength(pts, False) <= 1.045 * np.linalg.norm(pts[-1] - pts[0])
    kind = "loop" if primary["closed"] else "line" if straight else "polyline"
    return {**primary, "centerlines": centerlines, "outline": outline, "layers": layers,
            "color": color, "width": width, "kind": kind, "area": int(component.sum()),
            "evidence": "visible_component"}


def _border_rail(x, y, w, h, W, H):
    horizontal = y <= 2 or y + h >= H - 2
    vertical = x <= 2 or x + w >= W - 2
    short = min(W, H)
    depth = max(4, short * (0.08 if horizontal and vertical else 0.045))
    return ((vertical and w <= depth and h >= 2*w)
            or (horizontal and h <= depth and w >= 2*h)
            or (horizontal and w >= W*0.55 and h <= short*0.08)
            or (vertical and h >= H*0.55 and w <= short*0.08))


def _component_strokes(component, local, x, y, W, H):
    paths = []
    for part, ink in _pigments(component, local):
        number, regions, bounds, _ = cv2.connectedComponentsWithStats(part.astype(np.uint8), 8)
        for region in range(1, number):
            u, v, ww, hh, size = map(int, bounds[region])
            if size < 4 or _border_rail(x+u, y+v, ww, hh, W, H):
                continue
            sub = np.s_[v:v+hh, u:u+ww]
            stroke = _stroke(regions[sub] == region, local[sub], x+u, y+v, ink)
            if stroke is not None:
                paths.append(stroke)
    return paths


class _ComponentCache:
    """Per-derivation LRU, bounded by both entries and serialized bytes."""

    def __init__(self):
        self.entries = OrderedDict()
        self.size = 0

    def extract(self, component, local, x, y, width, height):
        # Global paper color, glare, seams, known mask, and hysteresis have
        # already been evaluated. The remaining pure function reads only this
        # support, its on-support BGR deficits, and the rectangle coordinates.
        support = np.packbits(component, axis=None).tobytes()
        color = local[component].tobytes()
        key = (x, y, width, height, component.shape, support, color)
        cached = self.entries.get(key)
        if cached is not None:
            self.entries.move_to_end(key)
            return json.loads(cached[0])
        strokes = _component_strokes(component, local, x, y, width, height)
        encoded = json.dumps(strokes, separators=(",", ":"), allow_nan=False).encode()
        size = len(support) + len(color) + len(encoded)
        budget = 6 * 1024 * 1024
        if size <= budget:
            while self.entries and (self.size + size > budget or len(self.entries) >= 128):
                _, (_, evicted_size) = self.entries.popitem(last=False)
                self.size -= evicted_size
            self.entries[key] = (encoded, size)
            self.size += size
        return strokes


def make_path_extractor():
    """Return a fresh, bounded extractor for one session's confirmed states."""
    cache = _ComponentCache()

    def extract(board, known):
        return _extract_paths(board, known, cache)

    return extract


def extract_paths(board: np.ndarray, known: np.ndarray) -> list[dict]:
    """Editable visible ink groups in absolute rectified-board pixel coordinates.

    outline and layer outlines are closed, even-odd rings (including holes).
    Layer order is light to dark. points is the primary measured centerline;
    centerlines includes the other branches, with no direction or timing claim.
    """
    return _extract_paths(board, known, None)


def _extract_paths(board, known, cache):
    if not isinstance(board, np.ndarray) or board.dtype != np.uint8 or board.ndim != 3 or board.shape[2] != 3:
        raise ValueError("board must be a uint8 BGR image")
    if not isinstance(known, np.ndarray) or known.dtype != np.uint8 or known.shape != board.shape[:2]:
        raise ValueError("known must be an H×W uint8 mask matching board")
    if board.size == 0 or np.any((known != 0) & (known != 255)):
        raise ValueError("expected a nonempty board and a binary known mask")
    mask, deficit = _ink_response(board, known)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    paths = []
    H, W = known.shape
    for label in range(1, count):
        x, y, w, h, area = map(int, stats[label])
        if area > H * W * 0.25:
            continue
        if _border_rail(x, y, w, h, W, H):
            continue
        component = labels[y:y+h, x:x+w] == label
        local = deficit[y:y+h, x:x+w]
        # Remove only very long runs on the outermost pixels of a mixed
        # component; a pen mark entering from the rim remains measurable.
        if x <= 2 or x+w >= W-2 or y <= 2 or y+h >= H-2:
            ys, xs = np.ogrid[y:y+h, x:x+w]
            strip = component.astype(np.uint8)
            v = cv2.morphologyEx(strip, cv2.MORPH_OPEN, np.ones((23,1), np.uint8))
            h_run = cv2.morphologyEx(strip, cv2.MORPH_OPEN, np.ones((1,23), np.uint8))
            outer = ((v != 0) & ((xs < 3) | (xs >= W-3))) | (
                (h_run != 0) & ((ys < 3) | (ys >= H-3)))
            component &= ~outer
        if cache is None:
            strokes = _component_strokes(component, local, x, y, W, H)
        else:
            strokes = cache.extract(component, local, x, y, W, H)
        paths.extend(strokes)
    return paths
