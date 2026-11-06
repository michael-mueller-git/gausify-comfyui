#!/usr/bin/env bash
# Install everything gausify-comfyui needs into ComfyUI's Python env, then check it.
#
# Usage (with ComfyUI's venv active, or PYTHON=/path/to/ComfyUI/venv/bin/python):
#   scripts/install.sh [--models] [--colmap]
#
#   --models   also download the ONNX mask models (SAM 2.1 fp32, RMBG-1.4; ~350 MB)
#   --colmap   also build the isolated COLMAP venv (scripts/setup-colmap-env.sh)
#
# Steps:
#   1. OpenCV: keep exactly one OpenCV distribution, version >= 4.12 and < 5
#      (OpenCV 5 has no cv2.AKAZE_create). An installed contrib build is kept
#      as contrib, since other nodes may need its extra modules.
#   2. ONNX Runtime: onnxruntime-gpu only (plain onnxruntime hides the CUDA provider).
#   3. pip install the rest of requirements.txt.
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
        -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "error: unknown argument $arg (see --help)" >&2; exit 2 ;;
    esac
done

if [[ -n "${PYTHON:-}" ]]; then
    python="$PYTHON"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    python="$VIRTUAL_ENV/bin/python"
else
    echo "error: activate ComfyUI's venv or set PYTHON=/path/to/ComfyUI/venv/bin/python" >&2
    exit 2
fi
pip=("$python" -m pip)

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

# Installed distributions whose name matches a pattern, as "name==version" lines.
dists() {
    "$python" - "$1" <<'EOF'
import re, sys
from importlib.metadata import distributions
pat = re.compile(sys.argv[1])
seen = set()
for d in distributions():
    n = (d.metadata["Name"] or "").lower()
    if pat.fullmatch(n) and n not in seen:
        seen.add(n)
        print(f"{n}=={d.version}")
EOF
}

step "OpenCV"
mapfile -t cv < <(dists 'opencv-(contrib-)?python(-headless)?')
printf '  installed: %s\n' "${cv[@]:-none}"
contrib=0
bad=0
for d in "${cv[@]}"; do
    [[ "$d" == opencv-contrib-* ]] && contrib=1
    major="${d##*==}"; major="${major%%.*}"
    [[ "$major" -ge 5 ]] && bad=1
done
[[ ${#cv[@]} -gt 1 ]] && bad=1
if [[ $contrib -eq 1 ]]; then want_cv="opencv-contrib-python-headless>=4.12,<5"; else want_cv="opencv-python-headless>=4.12,<5"; fi
if [[ $bad -eq 1 ]]; then
    echo "  fixing: removing ${cv[*]} (more than one build, or OpenCV 5), installing $want_cv"
    "${pip[@]}" uninstall -y "${cv[@]%%==*}"
fi
"${pip[@]}" install "$want_cv"

step "ONNX Runtime"
mapfile -t ort < <(dists 'onnxruntime(-gpu)?')
printf '  installed: %s\n' "${ort[@]:-none}"
for d in "${ort[@]}"; do
    if [[ "$d" == onnxruntime==* ]]; then
        echo "  fixing: plain onnxruntime hides the CUDA provider; reinstalling onnxruntime-gpu"
        "${pip[@]}" uninstall -y onnxruntime onnxruntime-gpu
        break
    fi
done

step "requirements.txt"
# OpenCV is handled above (a contrib build satisfies cv2 but not the
# opencv-python-headless name, and installing both makes them overwrite cv2/).
"${pip[@]}" install -r <(grep -v '^opencv' "$repo/requirements.txt")

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
                 "sam2-fp32-memory_encoder", "rmbg"):
        try:
            print(f"  {name}:", ensure_model(name))
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  {name}: FAILED ({e})", file=sys.stderr)
sys.exit(0 if ok else 1)
EOF

step "Checks"
set +e
"$python" - <<'EOF'
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
"${pip[@]}" check >/dev/null 2>&1 || echo "  note: 'pip check' reports conflicts (may come from other nodes): run '$python -m pip check'"
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
