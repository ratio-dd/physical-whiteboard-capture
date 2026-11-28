"""Build a lecture-isolated JSON manifest from an extracted annotation archive.

The archive itself stays outside the repository.  Each lecture directory is
expected to contain matching ``keyframes/`` and ``binary/`` files; paths in the
manifest are relative to the manifest location.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def build(root: Path, images_dir: str, masks_dir: str, validation_fraction: float, seed: int) -> dict:
    lectures = sorted(p for p in root.iterdir() if p.is_dir())
    if not lectures:
        raise ValueError(f"no lecture directories found below {root}")
    rng = random.Random(seed)
    shuffled = lectures[:]
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * validation_fraction)) if len(shuffled) > 1 else 0
    validation = {p.name for p in shuffled[:n_val]}
    pairs = []
    for lecture in lectures:
        image_root = lecture / images_dir
        mask_root = lecture / masks_dir
        if not image_root.is_dir() or not mask_root.is_dir():
            continue
        masks = {p.stem: p for p in mask_root.iterdir() if p.is_file()}
        for image in sorted(p for p in image_root.iterdir() if p.is_file()):
            mask = masks.get(image.stem)
            if mask is None:
                continue
            pairs.append({
                "lecture_id": lecture.name,
                "image": image.relative_to(root.parent).as_posix(),
                "mask": mask.relative_to(root.parent).as_posix(),
                "split": "validation" if lecture.name in validation else "train",
            })
    if not pairs:
        raise ValueError("no matching image/mask pairs found")
    return {"mask": {"positive_values": [0]}, "pairs": pairs, "seed": seed,
            "validation_lectures": sorted(validation),
            "source_layout": {"images": images_dir, "masks": masks_dir}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("archive_root", type=Path, help="extracted LectureMath annotation root")
    ap.add_argument("output", type=Path, help="manifest JSON path")
    ap.add_argument("--images-dir", default="keyframes")
    ap.add_argument("--masks-dir", default="binary")
    ap.add_argument("--validation-fraction", type=float, default=.2)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()
    if not 0 <= args.validation_fraction < 1:
        ap.error("validation fraction must be in [0, 1)")
    spec = build(args.archive_root, args.images_dir, args.masks_dir,
                 args.validation_fraction, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(spec['pairs'])} pairs across "
          f"{len({p['lecture_id'] for p in spec['pairs']})} lectures to {args.output}")


if __name__ == "__main__":
    main()
