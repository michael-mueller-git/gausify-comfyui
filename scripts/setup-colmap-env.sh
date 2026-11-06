#!/usr/bin/env bash
# Build the isolated COLMAP venv ahead of time and check that it sees the GPU.
#
# Usage (with ComfyUI's venv active, or PYTHON=/path/to/ComfyUI/venv/bin/python):
#   scripts/setup-colmap-env.sh [VENV_DIR]
#
# VENV_DIR defaults to $GAUSIFY_COLMAP_VENV, else ComfyUI/models/gausify/colmap-venv
# when this repo sits in ComfyUI/custom_nodes. Use ComfyUI's Python: the venv is
# stamped with its version, and the node rebuilds it if that differs.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python="${PYTHON:-python}"

if [[ $# -ge 1 ]]; then
    venv="$1"
elif [[ -n "${GAUSIFY_COLMAP_VENV:-}" ]]; then
    venv="$GAUSIFY_COLMAP_VENV"
elif [[ "$(basename "$(dirname "$repo")")" == custom_nodes ]]; then
    venv="$(cd "$repo/../.." && pwd)/models/gausify/colmap-venv"
else
    echo "error: not inside ComfyUI/custom_nodes; pass VENV_DIR or set GAUSIFY_COLMAP_VENV" >&2
    exit 2
fi

if [[ -z "${VIRTUAL_ENV:-}" && -z "${PYTHON:-}" ]]; then
    echo "warning: no venv active; using $(command -v "$python"). Activate ComfyUI's venv or set PYTHON." >&2
fi

echo "ComfyUI python: $("$python" -c 'import sys; print(sys.executable, sys.version.split()[0])')"
echo "COLMAP venv:    $venv"

cd "$repo"
GAUSIFY_COLMAP_VENV="$venv" "$python" -c "
import logging; logging.basicConfig(level=logging.INFO, format='%(message)s')
from gausify.colmap_env import ensure_colmap_env
print('ready:', ensure_colmap_env())
"

"$venv/bin/python" -I -c "
import sys; sys.path.insert(0, sys.argv[1])
from gausify.cuda_libs import import_pycolmap
p = import_pycolmap()
n = p.get_num_cuda_devices() if p.has_cuda else 0
print('pycolmap', p.__version__, '| cuda build:', p.has_cuda, '| devices:', n)
sys.exit(0 if n > 0 else 3)
" "$repo" || { echo "warning: COLMAP sees no CUDA device; it will run on CPU" >&2; exit 3; }

default="$(cd "$repo/../.." 2>/dev/null && pwd)/models/gausify/colmap-venv"
if [[ "$venv" != "$default" ]]; then
    echo "note: export GAUSIFY_COLMAP_VENV=$venv for ComfyUI too"
fi
