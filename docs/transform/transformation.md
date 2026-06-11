# CPU Repack (量化权重 layout 转换) Latency 实测

> 测试日期: 2026-06-10
> 设备: OnePlus CPH2583 (SM8650 / Snapdragon 8 Gen 3 / "pineapple"), Cortex-X4 prime core
> 目的: 量化 CPU 加载量化权重时, repack (row-interleave layout 转换) 这一步本身的延迟

---

## 1. 背景: 什么是 repack

CPU backend 加载权重的两种情况:

- **F16 / F32**: 原样 memcpy (甚至 mmap 零拷贝), **不做任何 layout 转换**。
- **量化类型 (Q4_0/Q8_0/Q5_K/Q6_K...)**: 若编译开启了对应 SIMD feature, 会走 **CPU_REPACK** buffer type,
  把磁盘上 row-major 的 quant block 重排成 **interleaved (xN) layout**, 以喂满 ARM i8mm/dotprod 整型点积指令。

repack 调用链:
```
model load → weight 分到 CPU_REPACK extra-buffer
           → init_tensor 选 optimal repack trait (写入 tensor->extra)
           → set_tensor 调 tensor_traits->repack()   ← 这一步重排, 不是 memcpy
```
- 触发与选型: `ggml/src/ggml-cpu/repack.cpp` `ggml_repack_get_optimal_repack_type()` (repack.cpp:4528)
- 重排实现: `repack_q4_0_to_q4_0_4_bl()` (repack.cpp:3200), `make_block_q4_0x4()` (repack.cpp:2742)
- set_tensor: `ggml_backend_cpu_repack_buffer_set_tensor()` (repack.cpp:4733)

---

## 2. ⚠️ 关键前提: repack 是否启用取决于编译期 SIMD feature

`ggml_cpu_has_matmul_int8()` / `ggml_cpu_has_dotprod()` 是 **编译期** 常量
(`#if defined(__ARM_FEATURE_MATMUL_INT8 / __ARM_FEATURE_DOTPROD)`, ggml-cpu.c:3723/3739),
**不是运行时 getauxval 检测**。

因此即使 CPU 硬件支持 (本机 `/proc/cpuinfo` 明确有 `i8mm` `asimddp`),
如果编译时 `-march` 没带 `+i8mm` `+dotprod`, 这两个函数恒返回 0,
`get_optimal_repack_type` 对所有量化类型返回 nullptr → **repack 全程禁用, 全走 generic 标量 mul_mat**。

### 本仓库各 build 的实测状态 (用 llvm-objdump 数 smmla 指令)

| build | march | `GGML_CPU_REPACK` | smmla 数 | repack |
|---|---|---|---|---|
| `build-android-llama` | `armv8-a` (默认) | ON | 0 | ❌ 禁用 |
| `build-android-norepack` | `armv8-a` | ON | 0 | ❌ 禁用 |
| `build-snapdragon` | `armv8.7a+fp16+dotprod+i8mm` | ON | 404 | ✅ 生效 |
| `build-android-repack` (本次新编) | `armv8.2-a+dotprod+i8mm+fp16` | ON | 244 | ✅ 生效 |

> **结论**: `GGML_CPU_REPACK=ON` 不代表 repack 生效, 还要编译 `-march` 带 `+i8mm` / `+dotprod`。
> 判别方法: `llvm-objdump -d <libggml-cpu.so> | grep -c '\bsmmla\b'`, >0 才有 repack。

本次测试使用 **`build-android-repack`** (i8mm=1, dotprod=1, 无 SVE), 运行时确认:
```
cpu has: neon=1 i8mm=1 dotprod=1 sve=0
```

---

## 3. 测试方法

- 自写 microbench `/tmp/bench-repack.cpp`: 对每个 type × shape, 构造原始量化数据,
  `ggml_backend_tensor_set` 写入 CPU_REPACK buffer (触发 repack), 计时。
- 绑定单核 prime (`taskset 80` = cpu7 / Cortex-X4), 取中位数 (warmup 3 次)。
- **repack 本身是单线程串行** (repack.cpp 无任何 OpenMP/线程), 所以绑单核数据准确无失真。

运行时实测各类型选中的 layout:
| 类型 | 命中分支 | layout | activation 量化目标 |
|---|---|---|---|
| Q4_0 | neon && i8mm | **4x8** | Q8_0 |
| Q8_0 | neon && i8mm | **4x8** | Q8_0 |
| Q5_K | neon && i8mm | **8x8** | Q8_K |
| Q6_K | neon && i8mm | **8x8** | Q8_K |

---

## 4. 结果: repack latency (ms, 单核 Cortex-X4, 中位数)

| shape | Q8_0 (4x8) | Q4_0 (4x8) | Q5_K (8x8) | Q6_K (8x8) |
|---|---|---|---|---|
| 1024×1024 | 0.076 | 0.045 | 0.066 | 0.070 |
| 2048×2048 | 0.455 | 0.255 | 0.412 | 0.457 |
| 4096×4096 | 2.549 | 1.494 | 2.447 | 2.689 |
| 8192×8192 | 12.41 | 6.07 | 10.47 | 13.22 |
| 4096×1024 | 0.447 | 0.246 | 0.328 | 0.332 |
| 1024×4096 | 0.445 | 0.248 | 0.410 | 0.459 |
| 4096×14336 (FFN up) | 10.35 | 5.14 | 8.71 | 9.67 |
| 14336×4096 (FFN down) | 10.44 | 5.26 | 8.74 | 9.71 |

## 5. 结果: repack 吞吐 (GiB/s, 按各自字节数归一化)

| shape | Q8_0 | Q4_0 | Q5_K | Q6_K |
|---|---|---|---|---|
| 1024×1024 | 13.6 | 12.3 | 10.1 | 11.5 |
| 2048×2048 | 9.1 | 8.6 | 6.5 | 7.0 |
| 4096×4096 | 6.5 | 5.9 | 4.4 | 4.8 |
| 8192×8192 | 5.4 | 5.8 | 4.1 | 3.9 |
| 4096×14336 | 5.6 | 6.0 | 4.3 | 4.2 |
| 14336×4096 | 5.6 | 5.9 | 4.4 | 4.4 |

稳态吞吐 (大 tensor):
- **Q4_0 ~5.8 GiB/s** (172 ms/GiB)
- **Q8_0 ~5.4 GiB/s** (185 ms/GiB)
- **Q5_K ~4.3 GiB/s** (233 ms/GiB)
- **Q6_K ~4.2 GiB/s** (238 ms/GiB)

---

## 6. 分析

1. **吞吐随 tensor 增大收敛到稳态**: 小 tensor (<4MB) cache-hot 可达 10–13 GiB/s;
   大 tensor 超 L2/L3 后受内存带宽限制, 收敛到 Q4_0/Q8_0 ~5.5–6, K-quant ~4.0–4.4 GiB/s。
   真实 LLM 权重都是大 tensor, 取稳态列。

2. **K-quant (Q5_K/Q6_K) 明显比 Q4_0/Q8_0 慢**: 它们的 make_block 要拆
   `d/dmin/scales/qh/ql` 多个分段分别搬运重组 (见 repack.h 的 `block_q6_Kx8` 结构),
   访存碎、分支多; Q4_0 只是连续字节 memcpy + 一次 XOR。

3. **只看总字节数, 不挑长宽比**: `4096×14336` 和 `14336×4096` (同字节) latency 基本一致,
   说明 repack 是纯带宽型 streaming。

4. **repack 单线程**: repack 路径零 OpenMP/线程。单个 weight 串行重排。
   (model load 时多个 weight 能否并行 repack 取决于上层 loader 的加载线程, 与 repack 内部无关。)

---

## 7. 部署开销估算

model load 时所有量化权重串行各 repack 一次, 按稳态吞吐 + 单核估:

| 模型 | 量化 | 权重量 | 纯 repack 开销 (单核 X4) |
|---|---|---|---|
| 8B | Q4_0 | ~4.5 GB | ~0.78 s |
| 8B | Q6_K | ~6.6 GB | ~1.6 s |
| 3B | Q4_0 | ~1.8 GB | ~0.31 s |

代价不止时间, 还 **失去 mmap 零拷贝** (repack buffer 必须实写一份重排数据)。
换来的是 matmul 阶段 i8mm/dotprod kernel 的吞吐。

---

## 8. 复现

- bench 源码: `/tmp/bench-repack.cpp`
- build: `build-android-repack` (`cmake -DGGML_CPU_ARM_ARCH="armv8.2-a+dotprod+i8mm+fp16"`)
- 编译 (NDK r28b):
  ```bash
  NDK=/home/hz85760/android-ndk-r28b
  CXX=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android31-clang++
  BIN=build-android-repack/bin
  $CXX -O3 -std=c++17 /tmp/bench-repack.cpp -I ggml/include \
    -L $BIN -lggml-cpu -lggml-base -lggml -lllama \
    -static-libstdc++ -llog -o /tmp/bench-repack2
  ```
- 运行:
  ```bash
  adb push $BIN/libggml-cpu.so $BIN/libggml-base.so $BIN/libggml.so $BIN/libllama.so \
           /tmp/bench-repack2 /data/local/tmp/
  adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp taskset 80 ./bench-repack2"
  ```
