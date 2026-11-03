#!/usr/bin/env bash
# Builds brush-cli from the pinned Brush commit with gausify-orbit's patches
# (same order and `git apply` as $REF/scripts/build-brush-web.mjs: the
# *.patch files of gausify-orbit sorted by name) plus gausify-cli.patch.
#
#   brush/build.sh                      release build for the host
#   brush/build.sh --zigbuild --target x86_64-unknown-linux-gnu.2.28
#   brush/build.sh --check              cargo check only
#   brush/build.sh --prepare            clone + patch only
#   brush/build.sh --out DIR            copy the binary into DIR
#
# Env: BRUSH_SRC (clone/cache dir, default ~/.cache/gausify/brush-src),
#      CARGO_TARGET_DIR (default $BRUSH_SRC/target), CARGO (default cargo).
# The clone is only a build cache: it is reset to the pinned commit and the
# patches are reapplied every run. Prints the binary path as the last line.
set -euo pipefail

REPO=https://github.com/ArthurBrussee/brush.git
REV=38185dedaf8143988a31a64c1f27c2323b1febdf
# gausify-orbit's patches, in its build order (sorted file names), then ours.
PATCHES=(brush-js.patch deg.patch geom.patch render.patch gausify-cli.patch)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${BRUSH_SRC:-${XDG_CACHE_HOME:-$HOME/.cache}/gausify/brush-src}"
CARGO="${CARGO:-cargo}"

target="" zig=0 mode=build out=""
while [ $# -gt 0 ]; do
  case "$1" in
    --target) target="$2"; shift 2 ;;
    --zigbuild) zig=1; shift ;;
    --check) mode=check; shift ;;
    --prepare) mode=prepare; shift ;;
    --out) mkdir -p "$2"; out="$(cd "$2" && pwd)"; shift 2 ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

run() { echo "\$ $*" >&2; "$@"; }

mkdir -p "$SRC"
if [ ! -d "$SRC/.git" ]; then
  run git init -q "$SRC"
  run git -C "$SRC" remote add origin "$REPO"
fi
head="$(git -C "$SRC" rev-parse -q --verify HEAD 2>/dev/null || true)"
if [ "$head" != "$REV" ] && ! git -C "$SRC" cat-file -e "$REV^{commit}" 2>/dev/null; then
  run git -C "$SRC" fetch -q --depth 1 origin "$REV"
fi
run git -C "$SRC" checkout -q --force "$REV"
run git -C "$SRC" clean -q -fd apps crates
for p in "${PATCHES[@]}"; do
  run git -C "$SRC" apply "$HERE/patches/$p"
done
[ "$mode" = prepare ] && { echo "$SRC"; exit 0; }

export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$SRC/target}"
args=(-p brush-cli --release --locked)
[ -n "$target" ] && args+=(--target "$target")
cd "$SRC"
if [ "$mode" = check ]; then
  run "$CARGO" check "${args[@]}"
  exit 0
fi
if [ "$zig" = 1 ]; then
  run "$CARGO" zigbuild "${args[@]}"
else
  run "$CARGO" build "${args[@]}"
fi

exe=brush-cli
# zigbuild's glibc suffix (x86_64-unknown-linux-gnu.2.28) is not part of the dir name.
dir="$CARGO_TARGET_DIR/${target%%.*}/release"
[ -n "$target" ] || dir="$CARGO_TARGET_DIR/release"
bin="$dir/$exe"
[ -f "$bin" ] || { echo "build output not found: $bin" >&2; exit 1; }
if [ -n "$out" ]; then
  mkdir -p "$out"
  cp "$bin" "$out/"
  bin="$out/$exe"
fi
echo "$bin"
