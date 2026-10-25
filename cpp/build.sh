#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${CUDA_ROOT:=/usr/local/cuda}"
if [[ -z "${TRT_INCLUDE_DIR:-}" ]]; then
  echo 'ERROR: Set TRT_INCLUDE_DIR to the TensorRT 11.4 C++ include directory (containing NvInfer.h).' >&2
  exit 2
fi
if [[ -z "${TRT_LIB_DIR:-}" ]]; then
  TRT_LIB_DIR="$(python - <<'PY'
import site
from pathlib import Path
for base in site.getsitepackages():
    p=Path(base)/'tensorrt_libs'
    if (p/'libnvinfer.so.11').is_file():
        print(p);break
PY
)"
fi
if [[ ! -f "$TRT_INCLUDE_DIR/NvInfer.h" || ! -f "$TRT_LIB_DIR/libnvinfer.so.11" ]]; then
  echo 'ERROR: TensorRT 11 headers or shared library missing.' >&2
  exit 2
fi
pkg-config --exists opencv4 || { echo 'ERROR: install libopencv-dev' >&2; exit 2; }
g++ -O3 -std=c++17 -Wno-deprecated-declarations main.cpp -o clip_classifier \
  -I"$TRT_INCLUDE_DIR" -I"$CUDA_ROOT/include" -L"$TRT_LIB_DIR" -L"$CUDA_ROOT/lib64" \
  -Wl,-rpath,"$TRT_LIB_DIR" -Wl,-rpath,"$CUDA_ROOT/lib64" \
  -l:libnvinfer.so.11 -lcudart $(pkg-config --cflags --libs opencv4)
echo 'Built cpp/clip_classifier'
