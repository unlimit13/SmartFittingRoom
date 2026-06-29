#!/usr/bin/env bash
# Recreate the project virtualenv on a fresh Raspberry Pi 5 clone of this repo.
# Reproduces the exact environment used for low-memory CPU inference and the
# multi-Pi parallelization PoC (same torch version on every node -> gloo works).
#
# Usage (from the repo root):
#   bash setup_env.sh
#   source .venv/bin/activate
#   python download_ckpt.py          # fetch model checkpoints (gitignored)
#
# Requirements: Python 3.11 on aarch64 (Pi 5 OS). Override the interpreter with
#   PY=python3.11 bash setup_env.sh
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-python3}
echo "[setup] using $($PY --version)"
case "$($PY --version 2>&1)" in
  *"3.11"*) ;;
  *) echo "[setup] WARNING: this env was pinned on Python 3.11; other versions may"
     echo "        fail to resolve the exact wheels (esp. torch 2.12.1+cpu)." ;;
esac

$PY -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate
pip install --upgrade pip wheel

# torch / torchvision are +cpu builds -> served by the PyTorch CPU index, not PyPI.
# Deps live in the repo-root requirements_vton.txt (single source of truth).
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r ../../requirements_vton.txt

echo
echo "[setup] done. Next:"
echo "  source .venv/bin/activate"
echo "  python download_ckpt.py     # download checkpoints (not in git)"
echo "  bash run_lowmem.sh          # single-Pi inference"
