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
                  model_version: str, weights_sha256: str, threshold: float = .5) -> Path:
    history = History(data_dir / "whiteboard.sqlite3")
    try:
        session = next((row for row in history.sessions() if row["id"] == session_id), None)
        if session is None:
            raise KeyError(session_id)
        head = int(session["seq"])
        model = _ProbabilityModel(predictor, model_version)

        def extractor(board, known):
            return extract_hybrid(model, board, known, threshold=threshold).vector_paths

        package = build_ink(history, session_id, mode="yolo-experiment",
                            extractor=extractor, model_version=model_version)
        package.update({"session_version": head, "weights_sha256": weights_sha256,
                        "recognition_threshold": threshold})
        if package["through_seq"] != head:
            raise RuntimeError("session advanced while building replay package")
        destination = output_dir / session_id / f"{head}.json"
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
    args = parser.parse_args()
    if not args.weights.is_file():
        raise SystemExit("weights must be an existing local file; refusing automatic download")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.loader != "yolo_loader:load_tiled_v2":
        raise SystemExit("the frozen v2 profile requires yolo_loader:load_tiled_v2")
    required = {"mode": "tiled", "conf_floor": .05, "imgsz": 512, "iou": .7,
                "tile": 640, "overlap": 96, "max_det": 300, "mask_threshold": .5}
    for key, value in required.items():
        if config.get(key) != value:
            raise SystemExit(f"frozen profile mismatch for {key}: expected {value}")
    digest = _sha256(args.weights)
    if digest != config.get("checkpoint_sha256"):
        raise SystemExit("weights SHA-256 does not match frozen profile")
    threshold = float(config.get("validation_threshold"))
    if not 0 < threshold < 1:
        raise SystemExit("frozen profile has an invalid validation threshold")
    predictor = _load_factory(args.loader, args.weights)
    output = build_package(args.data_dir, args.session_id, args.output, predictor,
                           config.get("model_version", getattr(predictor, "model_version", "yolo-experiment")),
                           digest, threshold)
    print(output)


if __name__ == "__main__":
    main()
