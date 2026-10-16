#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
uv venv .venv --python 3.13 --allow-existing
uv pip install --python .venv/bin/python -r requirements.lock
