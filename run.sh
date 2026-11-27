#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "未找到 .venv，请先运行：python3.13 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt" >&2
  exit 1
fi

exec "$PYTHON_BIN" "$ROOT_DIR/server.py" "$@"
