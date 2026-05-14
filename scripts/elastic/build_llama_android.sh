#!/usr/bin/env bash
# scripts/elastic/build_llama_android.sh
#
# NDK 交叉编译 llama.cpp 主程序（含 ggml-opencl 后端）到 arm64-v8a，用于
# 回归测试我们对 ggml-opencl.cpp 的改动。需要先跑 setup_opencl_android.sh。
#
# 用法：
#   ./scripts/elastic/build_llama_android.sh [Release|Debug]
#
# 输出：build-android-llama/bin/llama-cli 等

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NDK="${ANDROID_NDK:-/home/hz85760/android-ndk-r28b}"
ABI="${ANDROID_ABI:-arm64-v8a}"
PLATFORM="${ANDROID_PLATFORM:-android-31}"
BUILD_TYPE="${1:-Release}"
OCL="$ROOT/third_party/opencl-android"
BUILD_DIR="$ROOT/build-android-llama"

if [ ! -d "$NDK" ]; then
    echo "错误：找不到 NDK：$NDK" >&2
    exit 1
fi
if [ ! -f "$OCL/lib/libOpenCL.so" ] || [ ! -d "$OCL/include/CL" ]; then
    echo "错误：缺 third_party/opencl-android/。先跑 setup_opencl_android.sh" >&2
    exit 1
fi

GEN=Ninja
if ! command -v ninja >/dev/null 2>&1; then
    GEN="Unix Makefiles"
fi

echo "NDK        = $NDK"
echo "ABI        = $ABI"
echo "PLATFORM   = $PLATFORM"
echo "BUILD_TYPE = $BUILD_TYPE"
echo "OUTPUT     = $BUILD_DIR"

# 不 rm -rf，允许增量；首次会全编一遍
mkdir -p "$BUILD_DIR"

cmake -S "$ROOT" -B "$BUILD_DIR" -G "$GEN" \
    -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI="$ABI" \
    -DANDROID_PLATFORM="$PLATFORM" \
    -DANDROID_STL=c++_shared \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_OPENCL=ON \
    -DGGML_OPENCL_EMBED_KERNELS=ON \
    -DGGML_OPENCL_USE_ADRENO_KERNELS=ON \
    -DOpenCL_INCLUDE_DIR="$OCL/include" \
    -DOpenCL_LIBRARY="$OCL/lib/libOpenCL.so" \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_CURL=OFF

cmake --build "$BUILD_DIR" --target llama-cli -j

echo
echo "构建完成: $BUILD_DIR/bin/llama-cli"
echo
echo "部署示例:"
echo "  DEV=/data/local/tmp/elastic"
echo "  adb push $BUILD_DIR/bin/llama-cli \$DEV/"
echo "  adb push <your.gguf> \$DEV/"
echo "  # 基线（elastic=0，默认）："
echo "  adb shell 'cd '\$DEV' && LD_LIBRARY_PATH=.:/system/vendor/lib64:/vendor/lib64 ./llama-cli -m <your.gguf> -p \"hello\" -n 32 --seed 42 --temp 0'"
echo "  # 我们的改动（elastic=1）："
echo "  adb shell 'cd '\$DEV' && LD_LIBRARY_PATH=.:/system/vendor/lib64:/vendor/lib64 GGML_OPENCL_ELASTIC=1 ./llama-cli -m <your.gguf> -p \"hello\" -n 32 --seed 42 --temp 0'"
echo "  # 期望：两个输出 token 完全一致"
