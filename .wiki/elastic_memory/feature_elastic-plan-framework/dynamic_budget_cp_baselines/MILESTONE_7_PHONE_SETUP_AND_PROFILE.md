# Milestone 7: Phone Setup And Profile Run

## Target

Requested target was `op13`. The authorized devices visible from this server were:

```text
5ae7a43d               OnePlus 12 / CPH2583 / OP595DL1 / Android 15 / pineapple
172.20.115.151:5555    OnePlus 15 / CPH2749 / OP611FL1 / Android 16 / canoe
```

No OnePlus 13 device was visible in `adb devices -l`. The real phone run below uses:

```text
ADB_SERIAL=172.20.115.151:5555
market name: OnePlus 15
platform: canoe
kernel: 6.12.23-android16-5-gb3b66ace21e0-ab14672634-4k
```

## Branch

```sh
git remote add myfork https://github.com/MeeCreeps/llama.cpp.git
git fetch myfork
git checkout feature/elastic-plan-framework
git pull
git rev-parse --short HEAD
```

Result:

```text
a10bd13ed
a10bd13ed elastic: add dynamic budget planning baselines
```

## Android Build

Configured with OpenCL and CPU_Elastic enabled:

```sh
cmake -S . -B build-android-llama -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=/home/myid/hz85760/ndk/android-ndk-r28b/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-31 \
  -DANDROID_STL=c++_shared \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_OPENCL=ON \
  -DGGML_CPU_ELASTIC=ON \
  -DGGML_OPENCL_EMBED_KERNELS=ON \
  -DGGML_OPENCL_USE_ADRENO_KERNELS=ON \
  -DOpenCL_INCLUDE_DIR=$PWD/third_party/opencl-android/include \
  -DOpenCL_LIBRARY=$PWD/third_party/opencl-android/lib/libOpenCL.so \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_CURL=OFF

cmake --build build-android-llama --target llama-cli -j2
```

Result:

```text
PASS
CMake included OpenCL backend and CPU_Elastic backend.
Output: build-android-llama/bin/llama-cli
```

## Phone Files

Planned destination:

```text
/data/local/tmp/hyzheng/elastic
```

Files to push:

```text
llama-cli
libc++_shared.so
dynamic_budget_solver.py
build_cost_model.py
build_model_meta_from_profile.py
build_offline_budget_table.py
Llama-3.2-3B-Instruct-q4_0.gguf
const_1500_budget.csv
budget_trace.csv
```

## Budget Trace Source

The budget CSVs used in this run are synthetic test inputs created locally for BudgetWatcher:

```text
const_1500_budget.csv: fixed 1500 MiB profile budget
budget_trace.csv: hand-written 1500 -> 3000 -> 1500 MiB switching trace
```

They are not real phone memory-availability traces. The phone-real measurements in this milestone series are the profile CSVs and runtime logs produced by `llama-cli` on the target device.
