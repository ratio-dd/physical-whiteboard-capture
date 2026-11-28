"""Build one local YOLO ink replay package from an existing session.

The generator reads only the SQLite history, runs an explicitly supplied local
predictor once per stored state, and delegates timeline identity matching to
the product ink builder. It never opens a camera or downloads weights.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
from history import History
from ink import build_ink

_PIPELINE_PATH = Path(__file__).parents[1] / "lecturemath-ink" / "vector_pipeline.py"
_SPEC = importlib.util.spec_from_file_location("lecturemath_vector_pipeline_replay", _PIPELINE_PATH)
_PIPELINE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = _PIPELINE
_SPEC.loader.exec_module(_PIPELINE)
extract_hybrid = _PIPELINE.extract_hybrid


class _ProbabilityModel:
    def __init__(self, predictor, version: str):
        self.predictor = predictor
        self.model_version = version

    def predict_proba(self, board, known=None):
        raw = self.predictor.predict(board)
        if isinstance(raw, (tuple, list)):
            raw = raw[0]
        if isinstance(raw, dict):
            raw = raw.get("probability", raw.get("score_map"))
        probability = np.asarray(raw, dtype=np.float32)
        if probability.shape != board.shape[:2] or not np.isfinite(probability).all():
            raise ValueError("predictor must return a finite full-frame probability map")
        if probability.min() < 0 or probability.max() > 1:
            raise ValueError("predictor probability map must be in [0, 1]")
        if known is not None:
            probability = np.where(known > 0, probability, 0).astype(np.float32)
        return probability


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_factory(spec: str, weights: Path):
    module_name, function_name = spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), function_name)
    return factory(str(weights))


def build_package(data_dir: Path, session_id: str, output_dir: Path, predictor,
                  model_version: str, weights_sha256: str, threshold: float = .5,
                  recognition_mode: str = "yolo-experiment") -> Path:
    if recognition_mode not in {"yolo-experiment", "yolo-mps-experiment"}:
        raise ValueError("recognition_mode must be yolo-experiment or yolo-mps-experiment")
    history = History(data_dir / "whiteboard.sqlite3")
    try:
        session = next((row for row in history.sessions() if row["id"] == session_id), None)
        if session is None:
            raise KeyError(session_id)
        head = int(session["seq"])
        model = _ProbabilityModel(predictor, model_version)

        def extractor(board, known):
            return extract_hybrid(model, board, known, threshold=threshold).vector_paths

        package = build_ink(history, session_id, mode=recognition_mode,
                            extractor=extractor, model_version=model_version)
        package.update({"session_version": head, "weights_sha256": weights_sha256,
                        "recognition_threshold": threshold})
        if package["through_seq"] != head:
            raise RuntimeError("session advanced while building replay package")
        filename = f"{head}.json" if recognition_mode == "yolo-experiment" else f"{head}.{recognition_mode}.json"
        destination = output_dir / session_id / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + f".tmp-{os.getpid()}")
        temporary.write_text(json.dumps(package, separators=(",", ":")), encoding="utf-8")
        os.replace(temporary, destination)
        return destination
    finally:
        history.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_id")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--loader", required=True, help="module:function factory accepting a local weights path")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="frozen local deployment profile JSON")
    parser.add_argument("--mode-id", choices=("cpu", "metal"), default="cpu",
                        help="prediction provenance; CPU keeps the legacy yolo-experiment package name")
    args = parser.parse_args()
    if not args.weights.is_file():
        raise SystemExit("weights must be an existing local file; refusing automatic download")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.loader != "yolo_loader:load_tiled_v2":
        raise SystemExit("the frozen v2 profile requires yolo_loader:load_tiled_v2")
    required = ("mode", "conf_floor", "imgsz", "iou", "tile", "overlap", "max_det",
                "mask_threshold", "checkpoint_sha256", "validation_threshold")
    missing = [key for key in required if key not in config]
    if missing:
        raise SystemExit("deployment profile is missing: " + ", ".join(missing))
    if config["mode"] != "tiled" or int(config["imgsz"]) != 512:
        raise SystemExit("deployment profile must use tiled 512 inference")
    if not (0 < float(config["conf_floor"]) < 1 and 0 < float(config["iou"]) <= 1 and
            int(config["tile"]) >= int(config["imgsz"]) and int(config["overlap"]) < int(config["tile"]) and
            int(config["max_det"]) > 0 and 0 < float(config["mask_threshold"]) < 1):
        raise SystemExit("deployment profile has invalid tiled inference parameters")
    digest = _sha256(args.weights)
    if digest != config.get("checkpoint_sha256"):
        raise SystemExit("weights SHA-256 does not match frozen profile")
    threshold = float(config.get("validation_threshold"))
    if not 0 < threshold < 1:
        raise SystemExit("frozen profile has an invalid validation threshold")
    predictor = _load_factory(args.loader, args.weights)
    recognition_mode = "yolo-mps-experiment" if args.mode_id == "metal" else "yolo-experiment"
    output = build_package(args.data_dir, args.session_id, args.output, predictor,
                           config.get("model_version", getattr(predictor, "model_version", "yolo-experiment")),
                           digest, threshold, recognition_mode)
    print(output)


if __name__ == "__main__":
    main()
