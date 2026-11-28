#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "未找到 .venv，请先运行：python3.13 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt" >&2
  exit 1
fi

# Classic-only remains the default and therefore does not require torch. To
# opt into the local MPS LectureNet runtime after tools/setup-lecturenet, set
# LECTURENET_LIVE=1. Paths remain configurable and never point at research data.
if [ "${LECTURENET_LIVE:-0}" = "1" ]; then
  CACHE_DIR=${LECTURENET_CACHE:-"$ROOT_DIR/.cache/lecturenet-v34"}
  set -- --live-model-root "$CACHE_DIR/lecturemath/ACCESS2021_release" \
         --live-weights "$CACHE_DIR/LectureNet_model_BIN_V34_final.dat" "$@"
fi

exec "$PYTHON_BIN" "$ROOT_DIR/server.py" "$@"
