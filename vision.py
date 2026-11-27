"""Estimate a fixed camera's light-colored writing surface from BGR frames.

Stable-frame count and duration control confirmation latency. Decreasing the
foreground thickness or area threshold holds narrower occluders and more thick
drawings. An occluder matching the board's appearance can resemble an erasure.
Recalibrate the four corners whenever the camera moves.
"""

from numbers import Integral, Real

import cv2
import numpy as np


# Keep resolution proportional to the observed board and bound per-frame work.
_MAX_BOARD_EDGE = 1600
_ANALYSIS_EDGE = 384
_STABLE_FRAMES = 3
_STABLE_MS = 300
_PIXEL_DELTA = 18  # Luminance tolerance in the enhanced image.
_CHROMA_DELTA = 22
_PAPER_DELTA = 34
_SURFACE_DELTA = 17  # Raw luminance levels relative to a previously clear tile.
_MIN_FOREGROUND_AREA = 0.003
_FOREGROUND_THICKNESS = 0.055  # Fraction of the shorter board edge.
_INK_CONTRAST = 2.8
_INK_NOISE_FLOOR = 4  # Local BGR levels after smoothing.
_GLARE_CLIP = 251  # All three raw channels must approach the sensor/JPEG ceiling.


def _integer(value, label, minimum, maximum):
    if (isinstance(value, bool) or not isinstance(value, Integral)
            or not minimum <= int(value) <= maximum):
        raise ValueError(f"{label} must be an integer in [{minimum}, {maximum}]")
    return int(value)


def _geometry(corners, width, height):
    try:
        points = np.asarray(corners)
        if points.shape != (4, 2) or points.dtype.kind not in "fiu":
            raise ValueError("expected four numeric (x, y) corners")
        points = points.astype(np.float64)
    except (TypeError, OverflowError) as exc:
        raise ValueError("expected four normalized corners") from exc
    if not np.isfinite(points).all() or np.any((points < 0) | (points > 1)):
        raise ValueError("corners must be finite normalized coordinates in [0, 1]")

    points *= (width - 1, height - 1)
    edges = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(edges, axis=1)
    following = np.roll(edges, -1, axis=0)
    cross = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    # Positive cross products require a convex clockwise polygon in image space.
    # Require at least one source pixel of altitude at each vertex as well.
    if np.any(lengths < 2) or np.any(cross < np.maximum(lengths, np.roll(lengths, -1))):
        raise ValueError("board corners must form a convex, non-degenerate quadrilateral")
    tl, tr, br, bl = points
    right = (tr + br - tl - bl) / np.ptp(points, axis=0)
    down = (bl + br - tl - tr) / np.ptp(points, axis=0)
    if right[0] <= 0 or down[1] <= 0 or right[0] + down[1] < abs(right[1] - down[0]):
        raise ValueError("expected top-left, top-right, bottom-right, bottom-left")

    spans = np.array([max(lengths[0], lengths[2]), max(lengths[1], lengths[3])])
    spans *= min(1.0, (_MAX_BOARD_EDGE - 1) / spans.max())
    w, h = np.maximum(3, np.rint(spans).astype(int) + 1)
    destination = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    transform = cv2.getPerspectiveTransform(points.astype(np.float32), destination)
    return transform, (int(w), int(h))


def _ellipse(diameter):
    size = max(3, int(round(diameter)) | 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _shading_plane(luminance, quantile):
    """Estimate an illumination gradient from local slopes across the board.

    Adjacent sample differences estimate the typical shading slope even when a
    contiguous group of samples belongs to a person. Use an upper quantile for
    paper brightness and the median for a change in overall exposure.
    """
    h, w = luminance.shape
    grid = cv2.resize(luminance, (min(20, w), min(16, h)),
                      interpolation=cv2.INTER_AREA).astype(np.float32)
    rows, cols = grid.shape
    slope_x = np.median(np.diff(grid, axis=1)) * cols
    slope_y = np.median(np.diff(grid, axis=0)) * rows
    x = (np.arange(cols, dtype=np.float32) + 0.5) / cols
    y = (np.arange(rows, dtype=np.float32) + 0.5) / rows
    offset = np.percentile(grid - x * slope_x - y[:, None] * slope_y, quantile)
    x = (np.arange(w, dtype=np.float32) + 0.5) / w
    y = (np.arange(h, dtype=np.float32) + 0.5) / h
    return offset + slope_x * x + slope_y * y[:, None]


def _glare_bands(small, luma):
    """Locate elongated near-white plateaus bordered by visibly darker paper."""
    clipped = cv2.inRange(small, (_GLARE_CLIP,) * 3, (255,) * 3)
    bands = np.zeros_like(clipped)
    if not clipped.any() or clipped.all():
        return bands
    connected = cv2.morphologyEx(clipped, cv2.MORPH_CLOSE, _ellipse(3))
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Fill thin dark marks in the *surround* used for the contrast check, so a
    # white area bordered only by dark lettering cannot masquerade as glare.
    paper = cv2.morphologyEx(luma, cv2.MORPH_CLOSE, _ellipse(5))
    h, w = luma.shape
    margin = max(4, round(min(w, h) * 0.035))
    ring_kernel = _ellipse(2 * margin + 1)
    for contour in contours:
        area = cv2.contourArea(contour)
        if not 0.001 * luma.size <= area <= 0.40 * luma.size:
            continue
        thin, long = sorted(cv2.minAreaRect(contour)[1])
        if thin < 2 or long < max(12, 2.5 * thin) or area < 0.45 * long * thin:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        x0, y0 = max(0, x - margin), max(0, y - margin)
        roi = np.s_[y0:min(h, y + height + margin), x0:min(w, x + width + margin)]
        core = np.zeros_like(bands[roi])
        cv2.drawContours(core, [contour], 0, 1, cv2.FILLED, offset=(-x0, -y0))
        surround = cv2.dilate(core, ring_kernel) > core
        if surround.any() and np.median(paper[roi][surround]) <= 245:
            bands[roi] |= core
    return bands


# Opposite sides of a one-pixel mark usually extend along one of these tangents.
# Evaluate expensive geometry only where the signed local ink deficit changes.
_WEAK_DELTA = 5
_NO_EVIDENCE = 255
_LINE_KERNELS = tuple(np.ascontiguousarray(k) for k in (
    np.ones((1, 7), np.uint8), np.ones((7, 1), np.uint8),
    np.eye(5, dtype=np.uint8), np.fliplr(np.eye(5, dtype=np.uint8))))


def _line_changes(rise, fall, valid):
    result = np.zeros_like(rise)
    for change in (rise, fall):
        mask = ((change >= _WEAK_DELTA - 1) & valid).astype(np.uint8)
        x, y, w, h = cv2.boundingRect(mask)
        if not w:
            continue
        roi = np.s_[max(0, y - 4):y + h + 4, max(0, x - 4):x + w + 4]
        local = mask[roi]
        line = np.zeros_like(local)
        for kernel in _LINE_KERNELS:
            line |= cv2.morphologyEx(local, cv2.MORPH_OPEN, kernel,
                                     borderType=cv2.BORDER_CONSTANT, borderValue=0)
        if cv2.countNonZero(line):
            seeds = local & (change[roi] >= _WEAK_DELTA)
            core = cv2.dilate(seeds, np.ones((5, 5), np.uint8))
            # Broad changes of texture/exposure lack two clearer flanks.
            broad = cv2.morphologyEx(local, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            result[roi] |= line & core & (broad == 0)
    return result


def _enhance(rect, small_luma, contrast, measure_lines=False):
    short = min(small_luma.shape)
    light = cv2.morphologyEx(small_luma, cv2.MORPH_CLOSE, _ellipse(max(9, short * 0.12)))
    light = cv2.GaussianBlur(light, (0, 0), max(1, short * 0.015))
    h, w = rect.shape[:2]
    light = cv2.resize(light, (w, h), interpolation=cv2.INTER_LINEAR)
    gain = 246.0 / np.maximum(light.astype(np.float32), 96)
    adjusted = cv2.split(rect)
    level = None
    if contrast != 1:
        # A narrow per-channel closing estimates the paper across a stroke.
        # Use only deficits present both at this pixel and in a denoised copy.
        smooth = cv2.GaussianBlur(rect, (3, 3), 0.7)
        size = max(7, min(23, round(min(w, h) * 0.035))) | 1
        paper = cv2.morphologyEx(smooth, cv2.MORPH_CLOSE, np.ones((size, size), np.uint8))
        raw_detail = cv2.subtract(paper, rect)
        detail = cv2.min(cv2.subtract(paper, smooth), raw_detail)
        channels = cv2.split(detail)
        b, g, r = channels
        detail_max = cv2.max(b, cv2.max(g, r))
        strength = detail_max.astype(np.float32)
        weight = np.maximum(strength - _INK_NOISE_FLOOR, 0) / np.maximum(strength, 1)
        # Strong edges already have contrast. Taper across their immediate
        # neighborhood as well, to avoid magnifying JPEG ringing around ink.
        b, g, r = cv2.split(raw_detail)
        # JPEG 4:2:0 color blocks can ring several pixels beyond the luma edge.
        raw_max = cv2.max(b, cv2.max(g, r))
        nearby_edge = cv2.dilate(raw_max, np.ones((17, 17), np.uint8))
        if measure_lines:
            # Keep the largest local channel deficit of a one-pixel mark.
            level = raw_max
            ink_edge = cv2.dilate(nearby_edge, np.ones((17, 17), np.uint8))
            level[(ink_edge > 55) | (cv2.inRange(rect, (250,) * 3, (255,) * 3) != 0)] = _NO_EVIDENCE
        taper = np.clip((128 - nearby_edge.astype(np.float32)) / 64, 0, 1)
        weight *= (contrast - 1) * taper
        adjusted = [cv2.subtract(channel, cv2.multiply(deficit, weight, dtype=cv2.CV_32F),
                                 dtype=cv2.CV_32F)
                    for channel, deficit in zip(adjusted, channels)]
    enhanced = cv2.merge([cv2.multiply(channel, gain, dtype=cv2.CV_8U) for channel in adjusted])
    return enhanced, level


def _analysis_size(w, h):
    scale = min(1, _ANALYSIS_EDGE / max(w, h))
    return max(3, round(w * scale)), max(3, round(h * scale))


def enhance_board(rectified_bgr, *, contrast=_INK_CONTRAST):
    """Enhance one rectified BGR image. contrast=1 applies only light flattening.

    The estimator calls the same spatial operator on every frame. No history
    or replacement pixels are used to enhance low-contrast visible marks.
    """
    if (not isinstance(rectified_bgr, np.ndarray) or rectified_bgr.dtype != np.uint8
            or rectified_bgr.ndim != 3 or rectified_bgr.shape[2] != 3
            or min(rectified_bgr.shape[:2]) < 3):
        raise ValueError("expected a uint8 rectified BGR image of at least 3 by 3 pixels")
    if isinstance(contrast, bool) or not isinstance(contrast, Real) or not 1 <= contrast <= 4:
        raise ValueError("contrast must be a number between 1 and 4")
    h, w = rectified_bgr.shape[:2]
    small = cv2.resize(rectified_bgr, _analysis_size(w, h), interpolation=cv2.INTER_AREA)
    return _enhance(rectified_bgr, cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), contrast)[0]


def _obstructions(luma, cr, cb, thickness, reference, valid, bands):
    # Work on raw light and color, before local whitening can conceal a shirt.
    # Exclude the specular plateau from the illumination fit. This background
    # statistic is only for classification and is never drawn onto the board.
    background = luma
    if bands.any():
        background = np.where(bands, np.median(luma[bands == 0]), luma).astype(np.uint8)
        valid = valid & (bands == 0)
    paper = _shading_plane(background, 85)
    colored = (cv2.absdiff(cr, 128) > 27) | (cv2.absdiff(cb, 128) > 27)
    atypical = (luma < np.maximum(85, paper - _PAPER_DELTA)) | colored
    if valid.any():
        # A near-white sleeve can be quite unlike the previously seen surface
        # even if it is the brightest object in this frame. Discount a lighting
        # change only if the great majority of known samples agree on it.
        delta = luma.astype(np.float32) - reference
        delta[~valid] = 0
        exposure = _shading_plane(delta, 50)
        residual = delta - exposure
        if np.mean(np.abs(residual[valid]) <= 10) >= 0.90:
            delta = residual
        atypical |= valid & (np.abs(delta) >= _SURFACE_DELTA)
    atypical &= bands == 0
    # Bridge seams in textured clothing. Seek thick cores before filling their
    # silhouettes, so loops in marker handwriting do not become solid blobs.
    solid = cv2.morphologyEx(atypical.astype(np.uint8), cv2.MORPH_CLOSE, _ellipse(3))
    core = cv2.erode(solid, thickness, borderType=cv2.BORDER_REPLICATE)
    blocked = np.zeros_like(solid)
    if core.any():
        # Recover clothing and fingers near thick cores. Do not flood along a
        # connected marker stroke or board rim into an otherwise clear region.
        nearby = cv2.dilate(core, thickness, iterations=3)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(solid & nearby)
        large = stats[:, cv2.CC_STAT_AREA] >= solid.size * _MIN_FOREGROUND_AREA
        large[0] = False
        outlines, _ = cv2.findContours(large[labels].astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blocked, outlines, -1, 1, cv2.FILLED)
        blocked = cv2.dilate(blocked, _ellipse(3))
    return blocked


def _different_pixels(a, b):
    # Camera noise and JPEG chroma ringing fluctuate at marker edges. Compare
    # a small neighborhood with separate luminance and chroma tolerances.
    # Also require a difference at the pixel itself so an adjacent tile cannot
    # be marked as changed merely because its neighbor contains a new stroke.
    raw_close = cv2.inRange(cv2.absdiff(a, b), (0, 0, 0), (8, 8, 8))
    a = cv2.cvtColor(cv2.GaussianBlur(a, (5, 5), 1.0), cv2.COLOR_BGR2YCrCb)
    b = cv2.cvtColor(cv2.GaussianBlur(b, (5, 5), 1.0), cv2.COLOR_BGR2YCrCb)
    tolerance = (_PIXEL_DELTA, _CHROMA_DELTA, _CHROMA_DELTA)
    structural_close = cv2.inRange(cv2.absdiff(a, b), (0, 0, 0), tolerance)
    return ((raw_close | structural_close) == 0).astype(np.uint8)


def _stable_observation(first, middle, last):
    """Choose an observed BGR triplet per pixel from three clear samples.

    The sample opposite the most dissimilar pair has the smallest sum of
    distances to the others. Ties prefer the anchor, then the middle sample.
    This prevents a one-frame speck in a tile's anchor from becoming durable.
    """
    first_middle = cv2.cvtColor(cv2.absdiff(first, middle), cv2.COLOR_BGR2GRAY)
    first_last = cv2.cvtColor(cv2.absdiff(first, last), cv2.COLOR_BGR2GRAY)
    middle_last = cv2.cvtColor(cv2.absdiff(middle, last), cv2.COLOR_BGR2GRAY)
    use_middle = (first_last > middle_last) & (first_last >= first_middle)
    use_last = (first_middle > middle_last) & (first_middle > first_last)
    result = first.copy()
    cv2.copyTo(middle, use_middle.view(np.uint8), result)
    cv2.copyTo(last, use_last.view(np.uint8), result)
    return result


class BoardEstimator:
    """Retain stable visible tiles. Unknown pixels have known == 0.

    Supply frames in capture order. A change's observed_at is the first clear
    sample of its confirmed appearance. If multiple tiles commit together it
    is the earliest such timestamp. Identical timestamps do not add evidence.
    glare is the fraction of rectified pixels near clipping in localized
    bright bands. held includes each tile intersecting such pixels.
    """

    def __init__(self, corners, width, height):
        width = _integer(width, "width", 3, 32766)
        height = _integer(height, "height", 3, 32766)
        self._input_shape = (height, width, 3)
        self._transform, self._size = _geometry(corners, width, height)
        w, h = self._size
        self.board = np.full((h, w, 3), 255, np.uint8)
        self.known = np.zeros((h, w), np.uint8)
        self._anchor = self.board.copy()
        self._middle = self.board.copy()
        self._board_level = np.full((h, w), _NO_EVIDENCE, np.uint8)
        self._anchor_level = self._board_level.copy()
        self._middle_level = self._board_level.copy()

        self._small_size = _analysis_size(w, h)
        short = min(self._small_size)
        self._thickness = _ellipse(max(7, short * _FOREGROUND_THICKNESS))
        self._no_glare = np.zeros((h, w), np.uint8)
        self._last_glare = self._no_glare

        # Clip edge tiles to the actual image. All occupancy thresholds use
        # their real area, including the narrow last row or column.
        self._tile = max(8, min(40, round(min(w, h) / 20)))
        self._x = np.r_[np.arange(0, w, self._tile), w]
        self._y = np.r_[np.arange(0, h, self._tile), h]
        area = np.diff(self._y)[:, None] * np.diff(self._x)
        self._min_pixels = np.minimum(area, np.maximum(2, np.ceil(area * 0.004)))
        self._min_line_pixels = np.maximum(3, self._min_pixels)
        self._samples = np.zeros(area.shape, np.uint8)
        self._since = np.zeros(area.shape, np.int64)
        self._settled = np.zeros(area.shape, bool)
        self._last_time = -1
        sw, sh = self._small_size
        self._reference = np.zeros((sh, sw), np.uint8)
        self._valid_reference = np.zeros((sh, sw), bool)
        self._small_tiles = np.ix_(((np.arange(sh) + 0.5) * h / sh).astype(int) // self._tile,
                                   ((np.arange(sw) + 0.5) * w / sw).astype(int) // self._tile)

    def _counts(self, mask, y=None, x=None):
        if y is None:
            y, x = self._y, self._x
        integral = cv2.integral(mask, sdepth=cv2.CV_32S)[np.ix_(y, x)]
        return np.diff(np.diff(integral, axis=0), axis=1)

    def _pixels(self, tiles):
        # Integer scale first: the final partial tile is clipped, not stretched.
        h, w = tiles.shape
        expanded = cv2.resize(tiles.view(np.uint8), (w * self._tile, h * self._tile),
                              interpolation=cv2.INTER_NEAREST)
        return expanded[:self.board.shape[0], :self.board.shape[1]].view(bool)

    def _appearance(self, frame):
        rect = cv2.warpPerspective(frame, self._transform, self._size,
                                   borderMode=cv2.BORDER_REPLICATE)
        small = cv2.resize(rect, self._small_size, interpolation=cv2.INTER_AREA)
        luma, cr, cb = cv2.split(cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb))
        bands = _glare_bands(small, luma)
        blocked = _obstructions(luma, cr, cb, self._thickness, self._reference,
                                self._valid_reference, bands)
        enhanced, level = _enhance(rect, luma, _INK_CONTRAST, measure_lines=True)
        glare = self._no_glare
        if bands.any():
            # Shoulders of a clipped stripe can contain a stable, thin edge.
            margin = cv2.dilate(bands, _ellipse(7))
            level[cv2.resize(margin, self._size, interpolation=cv2.INTER_NEAREST) != 0] = _NO_EVIDENCE
            # Check the raw pixels inside a band to retain any readable ink.
            glare = cv2.resize(bands, self._size, interpolation=cv2.INTER_NEAREST)
            glare &= cv2.inRange(rect, (_GLARE_CLIP - 2,) * 3, (255,) * 3)
        if blocked.any():
            blocked = cv2.resize(blocked, self._size, interpolation=cv2.INTER_NEAREST)
            blocked_tiles = self._counts(blocked) > 0
        else:
            blocked_tiles = np.zeros_like(self._settled)
        return enhanced, blocked_tiles, luma, bands, glare, level

    def update(self, frame_bgr, observed_at_ms):
        if (not isinstance(frame_bgr, np.ndarray) or frame_bgr.dtype != np.uint8
                or frame_bgr.shape != self._input_shape):
            raise ValueError(f"expected a uint8 BGR frame with shape {self._input_shape}")
        stamp = _integer(observed_at_ms, "observed_at_ms", 0, 2**63 - 1)
        if stamp < self._last_time:
            raise ValueError("frames must be supplied in capture order")
        if stamp == self._last_time:
            return dict(self._last_result, observed_at=stamp)
        current, blocked, luma, bands, glare, level = self._appearance(frame_bgr)
        different = self._counts(_different_pixels(current, self._anchor)) >= self._min_pixels
        # A sub-threshold line must also start a fresh observation window.
        # Compare its local structure to the pending anchor, not just to the
        # last saved board. A mark that vanishes or moves restarts the window.
        if blocked.any():
            border = cv2.dilate(blocked.astype(np.uint8), np.ones((3, 3), np.uint8))
            level[self._pixels(border != 0)] = _NO_EVIDENCE
        if self._settled.all():
            prior = self._board_level
        else:
            prior = np.where(self._pixels(self._settled), self._board_level, self._anchor_level)
        valid = (self.known != 0) & (level != _NO_EVIDENCE) & (prior != _NO_EVIDENCE)
        weak = _line_changes(cv2.subtract(level, prior), cv2.subtract(prior, level), valid)
        different |= self._counts(weak) >= self._min_line_pixels
        # Even a reveal to identical white paper needs fresh clear observations.
        if glare is not self._last_glare:
            different |= self._counts(glare ^ self._last_glare) > 0
        self._last_glare = glare
        start = ~blocked & ((self._samples == 0) | different)
        if start.any():
            pixels = self._pixels(start)
            cv2.copyTo(current, pixels.view(np.uint8), self._anchor)
            cv2.copyTo(level, pixels.view(np.uint8), self._anchor_level)
            self._since[start] = stamp
        np.minimum(self._samples + 1, _STABLE_FRAMES, out=self._samples)
        self._samples[start] = 1
        self._samples[blocked] = 0
        self._last_time = stamp
        age = stamp - self._since
        # Keep a distinct middle witness, even with sparse/irregular samples.
        halfway = (self._samples == 2) | ((self._samples > 2) & (age <= _STABLE_MS // 2))
        if halfway.any():
            pixels = self._pixels(halfway)
            cv2.copyTo(current, pixels.view(np.uint8), self._middle)
            cv2.copyTo(level, pixels.view(np.uint8), self._middle_level)
        ready = (self._samples >= _STABLE_FRAMES) & (age >= _STABLE_MS)
        newly_ready = ready & (~self._settled | start)
        self._settled = ready

        observed = stamp
        if ready.any():
            clear = ready[self._small_tiles] & (bands == 0)
            np.copyto(self._reference, luma, where=clear)
            self._valid_reference |= clear
        if newly_ready.any():
            observed = self._commit(current, level, glare, newly_ready, stamp)
        held = ~ready
        glare_fraction = 0.0
        if glare is not self._no_glare:
            held |= self._counts(glare) > 0
            glare_fraction = float(np.mean(glare))
        self._last_result = {"held": float(np.mean(held)), "observed_at": observed,
                             "glare": glare_fraction}
        return self._last_result.copy()

    def _commit(self, current, level, glare, ready, stamp):
        # Check the ready tiles and one surrounding tile for morphology/blur
        # context. A short stroke should not trigger a full-board medoid pass.
        x, y, w, h = cv2.boundingRect(ready.view(np.uint8))
        rows = self._y[max(0, y - 1):y + h + 2]
        cols = self._x[max(0, x - 1):x + w + 2]
        tiles = np.s_[max(0, y - 1):min(ready.shape[0], y + h + 1),
                      max(0, x - 1):min(ready.shape[1], x + w + 1)]
        region = np.s_[rows[0]:rows[-1], cols[0]:cols[-1]]
        rows, cols = rows - rows[0], cols - cols[0]
        ready = ready[tiles]
        known, board = self.known[region], self.board[region]
        visible = glare[region] == 0
        stable = _stable_observation(self._anchor[region], self._middle[region], current[region])
        unseen = self._counts(((known == 0) & visible).view(np.uint8), rows, cols) > 0
        delta = _different_pixels(stable, board) & visible
        commit = ready & (unseen | (self._counts(delta, rows, cols) >= self._min_pixels[tiles]))
        first, middle, last = self._anchor_level[region], self._middle_level[region], level[region]
        saved = self._board_level[region]
        hh, ww = visible.shape
        pixels = self._pixels(commit)[:hh, :ww].copy()
        # The same signed line must be present in all three distinct witnesses.
        # Update its antialias fringe while preserving other pixels in the tile.
        if np.any(ready & ~commit):
            low = cv2.min(cv2.min(first, middle), last)
            high = cv2.max(cv2.max(first, middle), last)
            valid = (high != _NO_EVIDENCE) & (saved != _NO_EVIDENCE)
            evidence = _line_changes(cv2.subtract(low, saved), cv2.subtract(saved, high), valid)
            thin = ready & ~commit & (self._counts(evidence, rows, cols) >= self._min_line_pixels[tiles])
            if thin.any():
                support = cv2.dilate(evidence, _ellipse(5)) != 0
                pixels |= self._pixels(thin)[:hh, :ww] & support & valid
                commit |= thin
        if not commit.any():
            return stamp
        pixels &= visible
        cv2.copyTo(stable, pixels.view(np.uint8), board)
        median = cv2.max(cv2.min(first, middle), cv2.min(cv2.max(first, middle), last))
        cv2.copyTo(median, pixels.view(np.uint8), saved)
        known[pixels] = 255
        return int(self._since[tiles][commit].min())
