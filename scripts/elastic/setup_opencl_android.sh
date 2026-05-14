#!/usr/bin/env bash
# scripts/elastic/setup_opencl_android.sh
#
# 一次性准备 Android 端的 OpenCL headers + ICD loader，输出到
# third_party/opencl-android/{include,lib}/。**不修改 NDK 安装目录**。
#
# 用法（env 可覆盖）：
#   ANDROID_NDK=/path/to/ndk ./setup_opencl_android.sh
#
# 默认：
#   ANDROID_NDK     = /home/hz85760/android-ndk-r28b
#   ANDROID_ABI     = arm64-v8a
#   ANDROID_PLATFORM= 24            （ICD loader 链接的 API level）
#
# 完成后产物：
#   third_party/opencl-android/include/CL/        <- 来自 Khronos OpenCL-Headers
#   third_party/opencl-android/lib/libOpenCL.so   <- arm64-v8a 的 ICD loader
#
# 来源：docs/backend/OPENCL.md（上游推荐路径），只是把 cp 到 NDK sysroot
# 改成 cp 到本仓库 third_party/，让仓库自包含、可重现。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NDK="${ANDROID_NDK:-/home/hz85760/android-ndk-r28b}"
ABI="${ANDROID_ABI:-arm64-v8a}"
PLATFORM="${ANDROID_PLATFORM:-24}"
PREFIX="$ROOT/third_party/opencl-android"
SRC="$PREFIX/src"

if [ ! -d "$NDK" ]; then
    echo "错误：找不到 NDK：$NDK" >&2
    echo "export ANDROID_NDK=/your/ndk/path 后重试" >&2
    exit 1
fi
if [ ! -f "$NDK/build/cmake/android.toolchain.cmake" ]; then
    echo "错误：$NDK 看上去不是有效 NDK（缺 toolchain）" >&2
    exit 1
fi

echo "NDK      = $NDK"
echo "ABI      = $ABI"
echo "PLATFORM = android-$PLATFORM"
echo "PREFIX   = $PREFIX"

mkdir -p "$PREFIX/include" "$PREFIX/lib" "$SRC"

# 1) OpenCL-Headers（纯头）
cd "$SRC"
if [ ! -d OpenCL-Headers ]; then
    echo "[1/3] clone OpenCL-Headers"
    git clone --depth 1 https://github.com/KhronosGroup/OpenCL-Headers
else
    echo "[1/3] OpenCL-Headers 已存在，跳过 clone"
fi
rm -rf "$PREFIX/include/CL"
cp -r OpenCL-Headers/CL "$PREFIX/include/"

# 2) OpenCL-ICD-Loader：交叉编译出 libOpenCL.so（arm64-v8a）
cd "$SRC"
if [ ! -d OpenCL-ICD-Loader ]; then
    echo "[2/3] clone OpenCL-ICD-Loader"
    git clone --depth 1 https://github.com/KhronosGroup/OpenCL-ICD-Loader
else
    echo "[2/3] OpenCL-ICD-Loader 已存在，跳过 clone"
fi
cd OpenCL-ICD-Loader
rm -rf build_android
mkdir build_android
cd build_android

GEN=Ninja
if ! command -v ninja >/dev/null 2>&1; then
    echo "警告：没装 ninja，退回 Make"
    GEN="Unix Makefiles"
fi

cmake .. -G "$GEN" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
    -DOPENCL_ICD_LOADER_HEADERS_DIR="$PREFIX/include" \
    -DANDROID_ABI="$ABI" \
    -DANDROID_PLATFORM="$PLATFORM" \
    -DANDROID_STL=c++_shared

cmake --build .

cp libOpenCL.so "$PREFIX/lib/"

# 3) 完成提示
echo
echo "[3/3] 完成。第三方 OpenCL 包已就绪："
echo "  $PREFIX/include/CL/cl.h"
echo "  $PREFIX/lib/libOpenCL.so"
echo
echo "接下来跑 scripts/elastic/build_android.sh 交叉编译 elastic_runtime + probe。"
