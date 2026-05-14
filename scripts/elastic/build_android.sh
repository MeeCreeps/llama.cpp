#!/usr/bin/env bash
# scripts/elastic/build_android.sh
#
# NDK 交叉编译 runtime/ 子工程到 arm64-v8a。先跑 setup_opencl_android.sh
# 准备好 third_party/opencl-android/ 再用本脚本。
#
# 用法（env 可覆盖）：
#   ANDROID_NDK=/path/to/ndk ./build_android.sh [Release|Debug]
#
# 默认：
#   ANDROID_NDK      = /home/hz85760/android-ndk-r28b
#   ANDROID_ABI      = arm64-v8a
#   ANDROID_PLATFORM = android-31    （Snapdragon 8 Gen 3 / Adreno 750 需要 31+）
#   BUILD_TYPE       = Release
#
# 输出：
#   runtime/build-android/libelastic_runtime.a
#   runtime/build-android/tests_elastic/test_budget_watcher    （Android 可执行）
#   runtime/build-android/tests_elastic/probe_cl_release        （Android 可执行）

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NDK="${ANDROID_NDK:-/home/hz85760/android-ndk-r28b}"
ABI="${ANDROID_ABI:-arm64-v8a}"
PLATFORM="${ANDROID_PLATFORM:-android-31}"
BUILD_TYPE="${1:-Release}"
OCL="$ROOT/third_party/opencl-android"
BUILD_DIR="$ROOT/runtime/build-android"

if [ ! -d "$NDK" ]; then
    echo "错误：找不到 NDK：$NDK" >&2
    exit 1
fi
if [ ! -f "$OCL/lib/libOpenCL.so" ] || [ ! -d "$OCL/include/CL" ]; then
    echo "错误：缺 third_party/opencl-android/ 内容" >&2
    echo "先跑：scripts/elastic/setup_opencl_android.sh" >&2
    exit 1
fi

echo "NDK        = $NDK"
echo "ABI        = $ABI"
echo "PLATFORM   = $PLATFORM"
echo "BUILD_TYPE = $BUILD_TYPE"
echo "OUTPUT     = $BUILD_DIR"

GEN=Ninja
if ! command -v ninja >/dev/null 2>&1; then
    GEN="Unix Makefiles"
fi

rm -rf "$BUILD_DIR"
cmake -S "$ROOT/runtime" -B "$BUILD_DIR" -G "$GEN" \
    -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI="$ABI" \
    -DANDROID_PLATFORM="$PLATFORM" \
    -DANDROID_STL=c++_shared \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DOpenCL_INCLUDE_DIR="$OCL/include" \
    -DOpenCL_LIBRARY="$OCL/lib/libOpenCL.so" \
    -DELASTIC_BUILD_TESTS=ON

cmake --build "$BUILD_DIR" -j

echo
echo "构建完成。adb 部署示例（按需调整）："
echo
echo "  DEV=/data/local/tmp/elastic"
echo "  adb shell mkdir -p \$DEV"
echo "  # 一次性：push libc++_shared.so（NDK c++_shared 链接产物需要）"
echo "  LIBCXX=\$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
echo "  adb push \$LIBCXX \$DEV/"
echo "  # 二进制"
echo "  adb push $BUILD_DIR/tests_elastic/probe_cl_release \$DEV/"
echo "  adb push $BUILD_DIR/tests_elastic/test_budget_watcher \$DEV/"
echo "  adb push $BUILD_DIR/tests_elastic/test_weight_buffer_manager \$DEV/"
echo "  # 运行：用系统 vendor 路径的 libOpenCL.so，本地路径取 libc++_shared.so"
echo "  adb shell 'cd '\$DEV' && LD_LIBRARY_PATH=.:/system/vendor/lib64:/vendor/lib64 ./probe_cl_release --size-mb 64 --iters 4 --mode interleaved'"
echo
echo "注意：**不要把我们 third_party/opencl-android/lib/libOpenCL.so push 到 \$DEV/**。"
echo "      Adreno 设备上没有 /etc/OpenCL/vendors/*.icd，系统直接用"
echo "      /system/vendor/lib64/libOpenCL.so 作 OpenCL 实现。push 我们那份 ICD loader"
echo "      并以 LD_LIBRARY_PATH=. 优先加载，会得到 CL_PLATFORM_NOT_FOUND_KHR(-1001)。"
echo "      我们那份 libOpenCL.so 只在编译时给链接器用。"
