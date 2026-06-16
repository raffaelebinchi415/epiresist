#!/usr/bin/env bash
set -euo pipefail
FORMULA="${1:-main}"
# single node, single A100 reproduces the reported configuration
torchrun --standalone --nproc_per_node=1 -m epiresist.counter compound --formula "$FORMULA"
