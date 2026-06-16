#!/usr/bin/env bash
set -euo pipefail
FORMULA="${1:-main}"
python -m epiresist.counter assay --formula "$FORMULA"
