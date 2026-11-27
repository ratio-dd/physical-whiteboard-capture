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


def _enhance(rect, small_luma, contrast):
    short = min(small_luma.shape)
    light = cv2.morphologyEx(small_luma, cv2.MORPH_CLOSE, _ellipse(max(9, short * 0.12)))
    light = cv2.GaussianBlur(light, (0, 0), max(1, short * 0.015))
    h, w = rect.shape[:2]
    light = cv2.resize(light, (w, h), interpolation=cv2.INTER_LINEAR)
    gain = 246.0 / np.maximum(light.astype(np.float32), 96)
    adjusted = rect.astype(np.float32)
    if contrast != 1:
        # A narrow per-channel closing estimates the paper across a stroke.
        # Use only deficits present both at this pixel and in a denoised copy.
        smooth = cv2.GaussianBlur(rect, (3, 3), 0.7)
        size = max(7, min(23, round(min(w, h) * 0.035))) | 1
        paper = cv2.morphologyEx(smooth, cv2.MORPH_CLOSE, np.ones((size, size), np.uint8))
        raw_detail = cv2.subtract(paper, rect)
        detail = cv2.min(cv2.subtract(paper, smooth), raw_detail)
        b, g, r = cv2.split(detail)
        strength = cv2.max(b, cv2.max(g, r)).astype(np.float32)
        weight = np.maximum(strength - _INK_NOISE_FLOOR, 0) / np.maximum(strength, 1)
        # Strong edges already have contrast. Taper across their immediate
        # neighborhood as well, to avoid magnifying JPEG ringing around ink.
        b, g, r = cv2.split(raw_detail)
        # JPEG 4:2:0 color blocks can ring several pixels beyond the luma edge.
        nearby_edge = cv2.dilate(cv2.max(b, cv2.max(g, r)), np.ones((17, 17), np.uint8))
        taper = np.clip((128 - nearby_edge.astype(np.float32)) / 64, 0, 1)
        weight *= (contrast - 1) * taper
        adjusted -= detail * weight[:, :, None]
    return np.clip(np.rint(adjusted * gain[:, :, None]), 0, 255).astype(np.uint8)


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
    return _enhance(rectified_bgr, cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), contrast)


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
    np.copyto(result, middle, where=use_middle[..., None])
    np.copyto(result, last, where=use_last[..., None])
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

        self._small_size = _analysis_size(w, h)
        short = min(self._small_size)
        self._thickness = _ellipse(max(7, short * _FOREGROUND_THICKNESS))
        self._last_glare = np.zeros((h, w), np.uint8)

        # Clip edge tiles to the actual image. All occupancy thresholds use
        # their real area, including the narrow last row or column.
        self._tile = max(8, min(40, round(min(w, h) / 20)))
        self._x = np.r_[np.arange(0, w, self._tile), w]
        self._y = np.r_[np.arange(0, h, self._tile), h]
        area = np.diff(self._y)[:, None] * np.diff(self._x)
        self._min_pixels = np.minimum(area, np.maximum(2, np.ceil(area * 0.004)))
        self._samples = np.zeros(area.shape, np.uint8)
        self._since = np.zeros(area.shape, np.int64)
        self._settled = np.zeros(area.shape, bool)
        self._last_time = -1
        sw, sh = self._small_size
        self._reference = np.zeros((sh, sw), np.uint8)
        self._valid_reference = np.zeros((sh, sw), bool)
        self._small_tiles = np.ix_(((np.arange(sh) + 0.5) * h / sh).astype(int) // self._tile,
                                   ((np.arange(sw) + 0.5) * w / sw).astype(int) // self._tile)

    def _counts(self, mask):
        integral = cv2.integral(mask, sdepth=cv2.CV_32S)[np.ix_(self._y, self._x)]
        return np.diff(np.diff(integral, axis=0), axis=1)

    def _pixels(self, tiles):
        return tiles.repeat(self._tile, axis=0).repeat(self._tile, axis=1)[
            :self.board.shape[0], :self.board.shape[1]]

    def _appearance(self, frame):
        rect = cv2.warpPerspective(frame, self._transform, self._size,
                                   borderMode=cv2.BORDER_REPLICATE)
        small = cv2.resize(rect, self._small_size, interpolation=cv2.INTER_AREA)
        luma, cr, cb = cv2.split(cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb))
        bands = _glare_bands(small, luma)
        blocked = _obstructions(luma, cr, cb, self._thickness, self._reference,
                                self._valid_reference, bands)
        enhanced = _enhance(rect, luma, _INK_CONTRAST)
        # The stripe boundary comes from spatial context. Test exposure at the
        # original rectified resolution to retain readable ink inside the stripe.
        glare = cv2.resize(bands, self._size, interpolation=cv2.INTER_NEAREST)
        glare &= cv2.inRange(rect, (_GLARE_CLIP - 2,) * 3, (255,) * 3)
        blocked = cv2.resize(blocked, self._size, interpolation=cv2.INTER_NEAREST)
        return enhanced, self._counts(blocked) > 0, luma, bands, glare

    def update(self, frame_bgr, observed_at_ms):
        if (not isinstance(frame_bgr, np.ndarray) or frame_bgr.dtype != np.uint8
                or frame_bgr.shape != self._input_shape):
            raise ValueError(f"expected a uint8 BGR frame with shape {self._input_shape}")
        stamp = _integer(observed_at_ms, "observed_at_ms", 0, 2**63 - 1)
        if stamp < self._last_time:
            raise ValueError("frames must be supplied in capture order")
        current, blocked, luma, bands, glare = self._appearance(frame_bgr)
        different = self._counts(_different_pixels(current, self._anchor)) >= self._min_pixels
        # Even a reveal to identical white paper needs fresh clear observations.
        different |= self._counts(glare ^ self._last_glare) > 0
        self._last_glare = glare
        start = ~blocked & ((self._samples == 0) | different)
        if start.any():
            np.copyto(self._anchor, current, where=self._pixels(start)[:, :, None])
            self._since[start] = stamp
        distinct = stamp != self._last_time
        if distinct:
            np.minimum(self._samples + 1, _STABLE_FRAMES, out=self._samples)
        self._samples[start] = 1
        self._samples[blocked] = 0
        self._last_time = stamp
        age = stamp - self._since
        if distinct:
            # Keep the nearest preceding sample to the middle of the window.
            # The second distinct frame also covers sparse/irregular sampling.
            halfway = (self._samples == 2) | ((self._samples > 2) & (age <= _STABLE_MS // 2))
            if halfway.any():
                np.copyto(self._middle, current, where=self._pixels(halfway)[..., None])
        ready = (self._samples >= _STABLE_FRAMES) & (age >= _STABLE_MS)
        newly_ready = ready & (~self._settled | start)
        self._settled = ready

        observed = stamp
        if ready.any():
            clear = ready[self._small_tiles] & (bands == 0)
            np.copyto(self._reference, luma, where=clear)
            self._valid_reference |= clear
        if newly_ready.any():
            stable = _stable_observation(self._anchor, self._middle, current)
            visible_pixels = glare == 0
            unseen = self._counts(((self.known == 0) & visible_pixels).astype(np.uint8)) > 0
            delta = _different_pixels(stable, self.board) & visible_pixels
            changed = self._counts(delta) >= self._min_pixels
            commit = newly_ready & (unseen | changed)
            if commit.any():
                visible = self._pixels(commit) & visible_pixels
                np.copyto(self.board, stable, where=visible[:, :, None])
                self.known[visible] = 255
                observed = int(self._since[commit].min())
        held = ~ready | (self._counts(glare) > 0)
        return {"held": float(np.mean(held)), "observed_at": observed, "glare": float(np.mean(glare))}
