"""Optional real-time FCN -> native electronic-ink runtime.

The default HTTP capture path does not import this module.  It is an explicit
opt-in engine for callers that provide a session id, rectified BGR/known
frames, the existing pipeline, and the repository's SVG serializer.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
import queue
import threading
import time
from typing import Any, Callable, Protocol

import cv2
import numpy as np


class ScoreBackend(Protocol):
    def predict(self, bgr: np.ndarray) -> np.ndarray: ...


class InkPipeline(Protocol):
    def step(self, bgr: np.ndarray, known: np.ndarray,
             score: np.ndarray) -> Any: ...


class RuntimeErrorState(RuntimeError):
    """Raised when a runtime is stopped, failed, or given a bad frame order."""


def _official_force_binary(probability: np.ndarray) -> np.ndarray:
    """Return ink=1 from LectureNet's paper-white binary output.

    The official image uses 255 for paper and 0 for black ink.  The runtime
    contract is the opposite: one means ink support for the temporal pipeline.
    """
    byte_mask = np.clip(probability * 255.0, 0.0, 255.0).astype(np.uint8)
    return (byte_mask < 128).astype(np.float32, copy=False)


@dataclass(frozen=True)
class RuntimeResult:
    session_id: str
    seq: int
    observed_at: int
    svg: str
    pipeline_result: Any
    arrival_to_svg_ms: float


@dataclass(frozen=True)
class RuntimeStats:
    submitted: int
    completed: int
    dropped: int
    max_inflight: int
    queue_depth: int
    state: str
    error: str | None


@dataclass
class _Frame:
    session_id: str
    seq: int
    observed_at: int
    bgr: np.ndarray
    known: np.ndarray
    due: float


class OrderedInkRuntime:
    """Two-stage bounded runtime with one temporal CPU worker per session.

    ``backpressure='block'`` is the lossless policy used for recordings and
    for camera capture when preserving every board state matters.  The
    explicit ``drop_oldest`` policy is available for live preview callers;
    dropped frames are counted and never silently reused as states.
    """

    def __init__(self, backend: ScoreBackend,
                 pipeline_factory: Callable[[], InkPipeline],
                 svg_serializer: Callable[[Any, int, int], str], *,
                 max_queue: int = 2,
                 backpressure: str = "block",
                 result_sink: Callable[[RuntimeResult], None] | None = None) -> None:
        if max_queue < 1:
            raise ValueError("max_queue must be positive")
        if backpressure not in {"block", "drop_oldest"}:
            raise ValueError("backpressure must be block or drop_oldest")
        self.backend = backend
        self.pipeline_factory = pipeline_factory
        self.svg_serializer = svg_serializer
        self.max_queue = max_queue
        self.backpressure = backpressure
        self.result_sink = result_sink
        self._lock = threading.RLock()
        self._in: queue.Queue[_Frame | object] = queue.Queue(maxsize=max_queue)
        self._out: queue.Queue[tuple[_Frame, np.ndarray] | object] = queue.Queue(maxsize=max_queue)
        self._stop = object()
        self._cancel = threading.Event()
        self._threads: tuple[threading.Thread, threading.Thread] | None = None
        self._session_id: str | None = None
        self._state = "idle"
        self._error: BaseException | None = None
        self._last_seq = 0
        self._submitted = self._completed = self._dropped = self._max_inflight = 0
        # Sink-backed production does not enqueue a second copy.  The
        # sink-less convenience API is a bounded latest-result preview.
        self._results: queue.Queue[RuntimeResult] = queue.Queue(maxsize=max_queue)

    def start(self, session_id: str) -> None:
        with self._lock:
            if self._state not in {"idle", "stopped", "failed"}:
                raise RuntimeErrorState("runtime is already running")
            if not session_id:
                raise ValueError("session_id is required")
            self._in = queue.Queue(maxsize=self.max_queue)
            self._out = queue.Queue(maxsize=self.max_queue)
            # Results are session-scoped too.  Do not let an unconsumed result
            # from a stopped session appear after a restart.
            self._results = queue.Queue(maxsize=self.max_queue)
            self._stop = object()
            self._cancel.clear()
            self._session_id, self._state, self._error = session_id, "running", None
            self._last_seq = self._submitted = self._completed = self._dropped = self._max_inflight = 0
            self._threads = (threading.Thread(target=self._score_loop, name=f"ink-score-{session_id[:8]}"),
                             threading.Thread(target=self._ink_loop, name=f"ink-native-{session_id[:8]}"))
            for thread in self._threads:
                thread.start()

    def submit(self, seq: int, observed_at: int, bgr: np.ndarray, known: np.ndarray) -> bool:
        with self._lock:
            if self._state != "running" or self._session_id is None:
                raise RuntimeErrorState("runtime is not running")
            if seq <= self._last_seq:
                raise ValueError("seq must increase strictly")
            if (not isinstance(bgr, np.ndarray) or bgr.dtype != np.uint8 or
                    bgr.ndim != 3 or bgr.shape[2] != 3):
                raise ValueError("bgr must be HxWx3 uint8")
            if (not isinstance(known, np.ndarray) or known.dtype != np.uint8 or
                    known.shape != bgr.shape[:2]):
                raise ValueError("known must match bgr height and width")
            if np.any((known != 0) & (known != 255)):
                raise ValueError("known must contain only 0 or 255")
            frame = _Frame(self._session_id, seq, int(observed_at),
                           bgr.copy(), known.copy(), time.perf_counter())
            cancel = self._cancel
        if self.backpressure == "drop_oldest" and self._in.full():
            try:
                self._in.get_nowait()
                with self._lock:
                    self._dropped += 1
            except queue.Empty:
                pass
        # Do not hold the runtime lock while applying block backpressure:
        # stop() must be able to set the cancellation event and unblock this
        # producer even when the model is slower than the camera.
        while True:
            if cancel.is_set():
                raise RuntimeErrorState("runtime was stopped while submitting")
            try:
                self._in.put(frame, timeout=0.1)
                break
            except queue.Full:
                continue
        with self._lock:
            if self._state != "running" or self._session_id != frame.session_id:
                raise RuntimeErrorState("runtime stopped while submitting")
            self._last_seq = seq
            self._submitted += 1
            self._max_inflight = max(self._max_inflight, self._submitted - self._completed)
            return True

    def get_result(self, timeout: float | None = None) -> RuntimeResult:
        try:
            return self._results.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError("no completed ink result") from exc

    def stop(self, *, drain: bool = True, timeout: float = 10.0) -> None:
        with self._lock:
            if self._state == "idle":
                return
            if self._state not in {"running", "failed"}:
                return
            self._cancel.set()
            if not drain:
                self._state = "stopped"
                while True:
                    try: self._in.get_nowait()
                    except queue.Empty: break
            self._in.put(self._stop)
        for thread in self._threads or ():
            thread.join(timeout)
        with self._lock:
            if any(thread.is_alive() for thread in self._threads or ()):
                raise TimeoutError("runtime workers did not stop")
            if self._state != "failed":
                self._state = "stopped"

    def stats(self) -> RuntimeStats:
        with self._lock:
            return RuntimeStats(self._submitted, self._completed, self._dropped,
                                self._max_inflight, self._in.qsize(), self._state,
                                repr(self._error) if self._error else None)

    def _fail(self, exc: BaseException) -> None:
        with self._lock:
            self._error, self._state = exc, "failed"
            self._cancel.set()
        try: self._out.put_nowait(self._stop)
        except queue.Full: pass

    def _score_loop(self) -> None:
        try:
            while True:
                item = self._in.get()
                if item is self._stop:
                    self._out.put(self._stop); return
                score = self.backend.predict(item.bgr)
                if not isinstance(score, np.ndarray) or score.shape != item.known.shape:
                    raise ValueError("backend score must match known shape")
                self._out.put((item, score.astype(np.float32, copy=False)))
        except BaseException as exc:
            self._fail(exc)

    def _ink_loop(self) -> None:
        try:
            pipeline = self.pipeline_factory(); expected = 0
            while True:
                item = self._out.get()
                if item is self._stop: return
                frame, score = item
                if expected and frame.seq <= expected:
                    raise ValueError("runtime frame order regressed")
                expected = frame.seq
                result = pipeline.step(frame.bgr, frame.known, score)
                svg = self.svg_serializer(result, frame.bgr.shape[1], frame.bgr.shape[0])
                done = time.perf_counter()
                output = RuntimeResult(frame.session_id, frame.seq, frame.observed_at,
                                       svg, result, (done-frame.due)*1000)
                if self.result_sink is None:
                    try:
                        self._results.put_nowait(output)
                    except queue.Full:
                        try:
                            self._results.get_nowait()
                        except queue.Empty:
                            pass
                        self._results.put_nowait(output)
                else:
                    self.result_sink(output)
                with self._lock:
                    self._completed += 1
        except BaseException as exc:
            self._fail(exc)


class LectureNetMPSBackend:
    """Official LectureNet V34 loaded from caller-supplied local resources.

    ``output_mode='proposal_text'`` preserves the previously validated
    real-time experiment (text-mask candidate support).  Use
    ``output_mode='official_ink'`` for the author's black-ink binary branch.
    Both modes return float32 0/1 masks, with their polarity made explicit.
    """

    def __init__(self, model_root: str | Path, weights: str | Path, *, fp16: bool = True,
                 output_mode: str = "proposal_text"):
        import torch
        if output_mode not in {"proposal_text", "official_ink"}:
            raise ValueError("output_mode must be proposal_text or official_ink")
        if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable; refusing CPU fallback")
        root = Path(model_root).resolve()
        # setup-lecturenet exposes the cache root; the official Python package
        # and config live one level below it. Accept either form.
        release = root / "ACCESS2021_release"
        if (release / "configs" / "FCN_LectureNet.conf").is_file():
            root = release
        sys_path = str(root)
        import sys
        if sys_path not in sys.path: sys.path.insert(0, sys_path)
        config = import_module("AM_CommonTools.configuration.configuration").Configuration.from_file(
            str(root / "configs/FCN_LectureNet.conf"))
        config.set("FCN_BINARIZER_USE_CUDA", "False")
        cls = import_module("AccessMath.lecturenet_v1.FCN_lecturenet").FCN_LectureNet
        model = cls.CreateFromConfig(config, 3, False)
        # The released file is a state_dict; avoid pickle object execution.
        model.load_state_dict(torch.load(str(weights), map_location="cpu", weights_only=True))
        self._torch, self._cls, self._model = torch, cls, model.eval().to("mps")
        self._fp16 = bool(fp16)
        self.output_mode = output_mode
        if self._fp16: self._model.half()

    def predict(self, bgr: np.ndarray) -> np.ndarray:
        from PIL import Image
        image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        ow, oh = image.size; w, h = ow, oh
        while w*h > 2_500_000:
            image = image.resize((w//2, h//2), Image.Resampling.LANCZOS); w, h = image.size
        batch = self._cls.prepare_image(image).to("mps")
        if self._fp16: batch = batch.half()
        with self._torch.inference_mode():
            binary, text_mask, _ = self._model.forward(batch)
            # The historical real-time proposal used the text-mask branch as
            # a broad candidate support.  Keep it as the default for replay
            # compatibility; ``official_ink`` is the author's black-ink
            # binary branch and is available as an explicit comparison mode.
            raw = text_mask if self.output_mode == "proposal_text" else binary
            binary = raw.sigmoid().float().cpu().numpy()[0, 0]
        if (w, h) != (ow, oh):
            binary = cv2.resize(binary, (ow, oh), interpolation=cv2.INTER_NEAREST)
        if self.output_mode == "proposal_text":
            byte_mask = np.clip(binary * 255.0, 0.0, 255.0).astype(np.uint8)
            return (byte_mask >= 128).astype(np.float32, copy=False)
        return _official_force_binary(binary)
