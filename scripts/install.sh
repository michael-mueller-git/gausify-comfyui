#!/usr/bin/env bash
# Install everything gausify-comfyui needs into ComfyUI's Python env, then check it.
#
# Usage (with ComfyUI's venv active, or PYTHON=/path/to/ComfyUI/venv/bin/python; without
# either it looks for venv/ or .venv/ next to ComfyUI, and /comfy/mnt/venv):
#   scripts/install.sh [--models] [--colmap]
#
#   --models   also download the ONNX models (SAM 2.1 fp32, RMBG-1.4, ALIKED-LightGlue k3584; ~415 MB)
#   --colmap   also build the isolated COLMAP venv (scripts/setup-colmap-env.sh)
#
# Steps:
#   1. Core environment: a private venv (models/gausify/core-venv) with OpenCV 4, ONNX Runtime GPU,
#      scipy, ... that the compute code runs in; torch is taken from ComfyUI. Other nodes cannot break it.
#   4. Download brush-cli (and with --models the ONNX models) into ComfyUI's
#      models/gausify, where the nodes look for them.
#   5. Check: cv2.AKAZE_create, ONNX Runtime CUDA, brush-cli.
#
# pip arguments: PIP_* variables and your pip config apply (e.g. PIP_INDEX_URL).
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
want_models=0
want_colmap=0
for arg in "$@"; do
    case "$arg" in
        --models) want_models=1 ;;
        --colmap) want_colmap=1 ;;
        -h|--help) sed -n '2,21p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "error: unknown argument $arg (see --help)" >&2; exit 2 ;;
    esac
done

if [[ -n "${PYTHON:-}" ]]; then
    python="$PYTHON"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    python="$VIRTUAL_ENV/bin/python"
else
    # No venv active: look for ComfyUI's venv in the usual places (next to
    # ComfyUI or the base directory, and /comfy/mnt/venv of the Docker images).
    base="$(cd "$repo/../.." && pwd)"
    for v in "$base/venv" "$base/.venv" "$base/../venv" "$base/../.venv" /comfy/mnt/venv; do
        if [[ -x "$v/bin/python" ]]; then
            python="$(cd "$v" && pwd)/bin/python"
            echo "using ComfyUI venv $(dirname "$(dirname "$python")") (activate another venv or set PYTHON to override)"
            break
        fi
    done
    if [[ -z "${python:-}" ]]; then
        echo "error: no ComfyUI venv found; activate it or set PYTHON=/path/to/ComfyUI/venv/bin/python" >&2
        exit 2
    fi
fi

# Caches where the nodes look inside ComfyUI (models/gausify/...), when this
# repo sits in <ComfyUI or --base-directory>/custom_nodes.
if [[ "$(basename "$(dirname "$repo")")" == custom_nodes ]]; then
    models="$(cd "$repo/../.." && pwd)/models/gausify"
    export GAUSIFY_BIN_DIR="${GAUSIFY_BIN_DIR:-$models/bin}"
    export GAUSIFY_MODEL_DIR="${GAUSIFY_MODEL_DIR:-$models/models}"
else
    echo "warning: not inside custom_nodes; downloads go to \$GAUSIFY_BIN_DIR/\$GAUSIFY_MODEL_DIR or ~/.cache/gausify" >&2
fi

step() { printf '\n== %s\n' "$*"; }

step "Python: $("$python" -c 'import sys; print(sys.executable, sys.version.split()[0])')"

step "Core environment (own venv: OpenCV 4, ONNX Runtime GPU, scipy, ...)"
# The compute code runs in this venv, so other nodes can't break it (OpenCV 5, plain onnxruntime, ...).
# torch comes from ComfyUI's site-packages through a .pth file.
cd "$repo"
core_python="$("$python" -c 'from gausify.core_env import ensure_core_env; print(ensure_core_env())')"
echo "  $core_python"

step "Downloads"
cd "$repo"
downloads=0
"$python" - "$want_models" <<'EOF' || downloads=$?
import logging, sys
logging.basicConfig(level=logging.INFO, format="  %(message)s")
from gausify.downloader import ensure_binary, ensure_model
ok = True
try:
    print("  brush-cli:", ensure_binary("brush-cli"))
except Exception as e:  # noqa: BLE001
    ok = False
    print(f"  brush-cli: FAILED ({e}); Train Splat will retry on first use", file=sys.stderr)
if sys.argv[1] == "1":
    for name in ("sam2-fp32-vision_encoder", "sam2-fp32-mask_decoder", "sam2-fp32-memory_attention",
                 "sam2-fp32-memory_encoder", "rmbg", "aliked-lightglue-k3584"):
        try:
            print(f"  {name}:", ensure_model(name))
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  {name}: FAILED ({e})", file=sys.stderr)
sys.exit(0 if ok else 1)
EOF

step "Checks"
set +e
"$core_python" - <<'EOF'
import glob, os, sys
status = 0
try:
    import cv2
except Exception as e:  # noqa: BLE001
    sys.exit(f"  OpenCV: import FAILED: {str(e).strip().splitlines()[-1]}")
ak = hasattr(cv2, "AKAZE_create")
print(f"  OpenCV {cv2.__version__}: AKAZE {'ok' if ak else 'MISSING'}")
status |= 0 if ak else 1

try:
    import onnxruntime as ort
except Exception as e:  # noqa: BLE001
    sys.exit(f"  ONNX Runtime: import FAILED: {str(e).strip().splitlines()[-1]}")
prov = ort.get_available_providers()
cuda = "CUDAExecutionProvider" in prov
used = "-"
models = sorted(glob.glob(os.path.join(os.environ.get("GAUSIFY_MODEL_DIR", ""), "*.onnx")), key=os.path.getsize)
if cuda and models:
    # Creating a session is what actually loads the CUDA/cuDNN libraries.
    getattr(ort, "preload_dlls", lambda: None)()
    o = ort.SessionOptions(); o.log_severity_level = 3
    try:
        used = ort.InferenceSession(models[0], sess_options=o,
                                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]).get_providers()[0]
    except Exception as e:  # noqa: BLE001
        used = f"error: {e}"
print(f"  ONNX Runtime {ort.__version__}: providers {', '.join(prov)}")
if not cuda:
    print("  ONNX Runtime: NO CUDA provider, mask nodes run on CPU"); status |= 2
elif used == "-":
    print("  ONNX Runtime: CUDA provider present (run with --models to test that it initialises)")
elif used == "CUDAExecutionProvider":
    print("  ONNX Runtime: CUDA initialises ok")
else:
    print(f"  ONNX Runtime: CUDA provider does NOT initialise ({used}); mask nodes run on CPU."
          " Usually the CUDA/cuDNN libraries of this onnxruntime-gpu build are missing."); status |= 2
sys.exit(status)
EOF
checks=$?
set -e

if [[ $want_colmap -eq 1 ]]; then
    step "COLMAP venv"
    PYTHON="$python" "$repo/scripts/setup-colmap-env.sh"
fi

echo
if [[ $checks -eq 0 && $downloads -eq 0 ]]; then
    echo "done: gausify-comfyui is ready; restart ComfyUI."
else
    echo "done with problems (see above); restart ComfyUI after fixing them."
    exit 1
fi
