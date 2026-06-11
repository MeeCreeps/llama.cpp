# OpenCL 权重 Layout 转换

本文说明 llama.cpp 的 OpenCL backend 在模型权重从 GGUF/disk 读入之后, `ggml_backend_opencl_buffer_set_tensor()` 如何把磁盘里的 GGML block 格式转换成 OpenCL backend 实际使用的格式。

代码入口:

- `ggml/src/ggml-opencl/ggml-opencl.cpp`: `ggml_backend_opencl_buffer_set_tensor()`
- `ggml/src/ggml-opencl/kernels/cvt.cl`: 量化 block 的 convert/restore kernel
- `ggml/src/ggml-opencl/kernels/transpose.cl`: Adreno dense 路径使用的 buffer transpose kernel
- `ggml/src/ggml-common.h`: 磁盘/CPU 侧 block 结构定义

## 1. 总体模型

GGUF 里保存的量化权重基本是 AOS(Array of Structs) block:

```c
block_0 = { scale/min/high_bits/quants... }
block_1 = { scale/min/high_bits/quants... }
...
```

OpenCL 量化 matmul 更希望同类字段连续存放, 所以在 `GGML_OPENCL_SOA_Q` 打开时, `set_tensor()` 对支持的量化类型做 AOS -> SOA:

```text
disk / host block array
    -> clCreateBuffer 临时整块上传
    -> 在原 tensor device buffer 上 clCreateSubBuffer 切出 d/q/s/qh/... 段
    -> kernel_convert_block_* 拆字段, 必要时重排 nibble/bit
    -> Adreno dense: 对部分段做 buffer transpose
    -> Adreno MoE: 部分类型把 quant 段绑定成 image1d_buffer
```

如果 `GGML_OPENCL_SOA_Q` 没开, 或者类型没有专门分支, 则最终走默认 `clEnqueueWriteBuffer()`, 按磁盘 layout 原样写进 OpenCL buffer。

## 2. FP16 / FP32 / BF16

FP16 和 FP32 权重在 OpenCL load 阶段不做 layout transform:

```text
host data -> clEnqueueWriteBuffer(extra->data_device)
```

也就是说 FP16 从 disk 读出来是什么连续数组, OpenCL buffer 里就是什么连续数组。

BF16 是例外: OpenCL buffer 中按 FP16 存, `set_tensor()` 先创建临时输入 buffer, 再跑 `kernel_convert_bf16_to_f16`, 每个 BF16 元素左移到 FP32 高 16 位后 cast 成 half:

```text
BF16 host data -> temp cl_mem -> kernel_convert_bf16_to_f16 -> FP16 device buffer
```

源码中的 `kernel_adreno_xmem_prepack_weight_f16` 不是 load 阶段的通用 FP16 转换。它只在 `GGML_OPENCL_ADRENO_XMEM_GEMM` 相关 GEMM 路径中临时 prepack 权重, 不改变普通 `set_tensor()` 的 FP16 存储格式。

## 3. Q8_0

磁盘结构:

```c
struct block_q8_0 {
    half d;
    int8_t qs[32];
};
```

OpenCL SOA:

```text
d[]  : 每个 block 一个 fp16 scale
q[]  : 每个 block 32 个 int8 quant
```

转换:

```text
block_q8_0[] -> kernel_convert_block_q8_0 -> d[] + q[]
```

标准路径只拆字段, 不改 quant 值。Adreno dense 大矩阵路径还会 transpose:

```text
q: 作为 32-bit 元素看待, logical shape = K/4  x M
d: 作为 16-bit 元素看待, logical shape = K/32 x M
```

这里用的是 `transpose_2d_as_32b()` 和 `transpose_2d_as_16b()`。Q8_0 没有 `_noshuffle` 变体, 因为没有 nibble shuffle 问题。

## 4. Q4_0 / Q4_1 / IQ4_NL

磁盘结构:

```c
block_q4_0  = { half d;          uint8_t qs[16]; }
block_q4_1  = { half d; half m;  uint8_t qs[16]; }
block_iq4_nl = { half d;         uint8_t qs[16]; }
```

OpenCL SOA:

```text
Q4_0   : d[] -> q[]
Q4_1   : d[] -> m[] -> q[]
IQ4_NL : d[] -> q[]
```

普通 convert kernel 只做 AOS -> SOA, 不拆 nibble:

```text
kernel_convert_block_q4_0
kernel_convert_block_q4_1
kernel_convert_block_iq4_nl
```

`cvt.cl` 注释里明确写了 Q4 标准 kernel "does not deshuffle the bits"。也就是说磁盘里一个 byte 的低 4 bit / 高 4 bit 顺序原样进入 `q[]`, 后续 matmul kernel 读取时再解释。

Adreno dense 路径使用 `_noshuffle`:

```text
kernel_convert_block_q4_0_noshuffle
kernel_convert_block_q4_1_noshuffle
kernel_convert_block_iq4_nl_noshuffle
```

`_noshuffle` 这个名字容易误解。它实际是在 load 时把 nibble 解成自然顺序, 让计算 kernel 端不再 shuffle。以 Q4_0 为例, 两个原始字节 `x0, x1` 会变成:

```text
q[i]        = low(x0), low(x1)
q[i + 8]    = high(x0), high(x1)
```

随后 Adreno dense 对字段做 buffer transpose:

```text
q: 16-bit view, shape = K/4  x M
d: 16-bit view, shape = K/32 x M
m: 16-bit view, shape = K/32 x M  (仅 Q4_1)
```

Adreno MoE 专家权重使用 `_trans4_ns` 变体, 例如 `kernel_convert_block_q4_0_trans4_ns`。它把 nibble 解扰和转置写出合并在一个 kernel 里, quant 输出按 `uint` 流写入, 然后创建 `CL_MEM_OBJECT_IMAGE1D_BUFFER`:

```text
Q4_0/Q4_1: q_img = image1d_buffer(q, CL_R, CL_UNSIGNED_INT32)
```

## 5. Q5_0 / Q5_1

磁盘结构:

```c
block_q5_0 = { half d;          uint32_t qh; uint8_t qs[16]; }
block_q5_1 = { half d; half m;  uint32_t qh; uint8_t qs[16]; }
```

OpenCL SOA:

```text
Q5_0 : d[] -> qh[] -> qs[]
Q5_1 : d[] -> m[]  -> qh[] -> qs[]
```

普通路径只拆字段:

```text
kernel_convert_block_q5_0
kernel_convert_block_q5_1
```

当前源码中 Q5_0/Q5_1 的 Adreno dense 分支没有 `_noshuffle` kernel, 也没有后续 buffer transpose。它们只有 MoE 专家权重的 `_trans4_ns` 路径:

```text
kernel_convert_block_q5_0_trans4_ns
kernel_convert_block_q5_1_trans4_ns
```

MoE 路径会把低 4-bit `qs` 解成自然顺序并转置写成 `uint` 流, 同时处理 `qh`, 最后创建:

```text
Q5_0/Q5_1: qs_img = image1d_buffer(qs, CL_R, CL_UNSIGNED_INT32)
```

## 6. Q4_K / Q5_K / Q6_K

K-quant 的 super-block 大小是 `QK_K = 256`。

磁盘结构要点:

```text
Q4_K: d + dm + scales[12] + qs[128]
Q5_K: d + dm + scales[12] + qs[128] + qh[32]
Q6_K: ql[128] + qh[64] + scales[16] + d
```

OpenCL 普通 SOA:

```text
Q4_K: d[]  -> dm[] -> s[] -> q[]
Q5_K: d[]  -> dm[] -> s[] -> q[] -> qh[]
Q6_K: ql[] -> qh[] -> s[] -> d[]
```

普通 convert:

```text
kernel_convert_block_q4_K
kernel_convert_block_q5_K
kernel_convert_block_q6_K
```

这些普通 kernel 主要拆字段, 不做 Adreno 需要的自然顺序重排。

Adreno dense 对 Q4_K/Q5_K/Q6_K 使用 `_noshuffle` 变体, 再做 buffer transpose:

```text
Q4_K:
  q  transpose as 16-bit, shape K/4   x M
  d  transpose as 16-bit, shape K/256 x M
  dm transpose as 16-bit, shape K/256 x M

Q5_K:
  q  transpose as 16-bit, shape K/4   x M
  qh transpose as 8-bit,  shape K/8   x M
  d  transpose as 16-bit, shape K/256 x M
  dm transpose as 16-bit, shape K/256 x M

Q6_K:
  ql transpose as 16-bit, shape K/4     x M
  qh transpose as 8-bit,  shape K/4     x M
  s  transpose as 16-bit, shape K/16/2  x M
  d  transpose as 16-bit, shape K/256   x M
```

有两个例外:

- Q4_K 如果 `ne1 >= 32768` 且是普通 2D 权重, 会走 flat gemv 例外, 不使用 `_noshuffle`/transpose。
- Q6_K 如果 `ne1 >= 32768 && ne0 >= 2048` 且是普通 2D 权重, 也会走 flat gemv 例外。

Adreno MoE 路径使用 `_trans4_ns`:

```text
kernel_convert_block_q4_k_trans4_ns
kernel_convert_block_q5_k_trans4_ns
kernel_convert_block_q6_k_trans4_ns
```

MoE 的 quant 主段写成 `uint` 流并绑定 image:

```text
Q4_K: q_img  = image1d_buffer(q,  CL_R, CL_UNSIGNED_INT32)
Q5_K: q_img  = image1d_buffer(q,  CL_R, CL_UNSIGNED_INT32)
Q6_K: ql_img = image1d_buffer(ql, CL_R, CL_UNSIGNED_INT32)
```

注意 Q6_K MoE 的 `ql/qh` subbuffer 尺寸和普通路径不同: 源码按 `uint32_t` 输出打包后的 4-bit/2-bit 流, 即 `ql = nelements/8 * 4` 字节, `qh = nelements/16 * 4` 字节。

## 7. MXFP4

MXFP4 磁盘 block 包含一个 8-bit exponent/scale 和 4-bit quant:

```text
block_mxfp4 = { e; qs[] }
```

OpenCL SOA:

```text
e[] -> q[]
```

普通路径:

```text
kernel_convert_block_mxfp4 -> e[] + q[]
q_img = image1d_buffer(q, CL_RG, CL_UNSIGNED_INT32)
```

与其他普通量化类型不同, MXFP4 标准路径也会给 `q` 创建 image, 格式是 `CL_RG + CL_UNSIGNED_INT32`, image width 是 `nelements / 32 * 2`。

Adreno MoE 路径使用:

```text
kernel_convert_block_mxfp4_trans4_ns
q_img = image1d_buffer(q, CL_R, CL_UNSIGNED_INT32)
```

## 8. Q2_K / Q3_K

当前 OpenCL `set_tensor()` 没有 Q2_K / Q3_K 的 SOA convert 分支。它们不会在 load 阶段被拆成 OpenCL SOA, 而是走默认写入:

```text
Q2_K/Q3_K host block bytes -> clEnqueueWriteBuffer -> device buffer 原样 block layout
```

因此 Q2_K/Q3_K 的 layout 仍然是 `ggml-common.h` 里的磁盘 block 格式, OpenCL 计算 kernel 如果支持这些类型, 需要直接按原 block 结构读取。

## 9. Adreno 路径选择

相关判断函数在 `ggml-opencl.cpp`:

```text
use_adreno_kernels(tensor):
  ne0 >= threshold_ne0
  ne1 >= threshold_ne1
  ne2 == 1
  ne3 == 1

threshold 默认 512 x 512;
部分旧 Adreno compiler 版本降到 128 x 128。
```

```text
use_adreno_moe_kernels(tensor):
  tensor name 包含 ("ffn" 且 "exps") 或包含 "as"
  且 ne1 % 32 == 0
```

```text
enable_adreno_trans_weight(tensor):
  use_adreno_kernels(tensor)
  且 nelements < 128 * 1024 * 1024
```

Adreno dense 和 MoE 是两条不同路径:

- dense 大矩阵: `_noshuffle` 后用 `transpose_2d_as_*b()` 做 buffer-to-buffer transpose, 通常不创建权重 image。
- MoE 专家权重: `_trans4_ns` 在 convert kernel 内完成转置式写出, 然后给 quant 主段创建 image1d_buffer, MoE kernel 通过 image 读取权重。

## 10. 类型汇总

| 类型 | OpenCL load 后格式 | Adreno dense | Adreno MoE |
|---|---|---|---|
| F32 | 原样 buffer | 原样 | 原样 |
| F16 | 原样 buffer | 原样 | 原样 |
| BF16 | 转 FP16 buffer | 同左 | 同左 |
| Q8_0 | `d[] + q[]` | `q/d` transpose | 无专用 MoE image 分支 |
| Q4_0 | `d[] + q[]` | `_noshuffle + q/d transpose` | `_trans4_ns + q_img` |
| Q4_1 | `d[] + m[] + q[]` | `_noshuffle + q/d/m transpose` | `_trans4_ns + q_img` |
| IQ4_NL | `d[] + q[]` | `_noshuffle + q/d transpose` | 无专用 MoE image 分支 |
| Q5_0 | `d[] + qh[] + qs[]` | 只 split | `_trans4_ns + qs_img` |
| Q5_1 | `d[] + m[] + qh[] + qs[]` | 只 split | `_trans4_ns + qs_img` |
| Q4_K | `d[] + dm[] + s[] + q[]` | `_noshuffle + transpose`, 大 M flat 例外 | `_trans4_ns + q_img` |
| Q5_K | `d[] + dm[] + s[] + q[] + qh[]` | `_noshuffle + transpose` | `_trans4_ns + q_img` |
| Q6_K | `ql[] + qh[] + s[] + d[]` | `_noshuffle + transpose`, 大 M flat 例外 | `_trans4_ns + ql_img` |
| MXFP4 | `e[] + q[] + q_img(CL_RG)` | 标准路径也建 image | `_trans4_ns + q_img(CL_R)` |
| Q2_K | 原样 block buffer | 原样 | 原样 |
| Q3_K | 原样 block buffer | 原样 | 原样 |

## 11. Hexagon/NPU load 后 layout transform

Qualcomm NPU backend 在本仓库里对应 `ggml/src/ggml-hexagon`。它和 OpenCL 的 load-time transform 不一样:

- OpenCL: 先把权重写到 OpenCL buffer, 再用 OpenCL kernel 做 AOS -> SOA / transpose。
- Hexagon/NPU: `ggml_backend_hexagon_buffer_set_tensor()` 在 host 侧直接把 GGUF/disk block repack 到 Hexagon buffer 里, 没有单独的 GPU/NPU transform kernel。

Hexagon 有两个 buffer type:

- 普通 `buffer_type`: `is_host = opt_hostbuf`, 用于一般 tensor。
- `repack_buffer_type`: 通过 `get_extra_buffers_type()` 暴露给 scheduler, 名字是 `*-REPACK`, `is_host = false`。量化 matmul 要求 src0 权重必须在这个 repack buffer 里, 否则 `supports_op()` 会拒绝该 `MUL_MAT` / `MUL_MAT_ID`。

load 入口在 `ggml_backend_hexagon_buffer_set_tensor()`。从 disk/GGUF 读出来的数据传进来之后, 分支如下:

| GGML type | load 后 Hexagon buffer 格式 | 处理函数 |
|---|---|---|
| Q4_0 | `q4x4x2` | `repack_q4_0_q4x4x2()` |
| Q4_1 | `q4_1x4x2` | `repack_q4_1_q4x4x2()` |
| Q8_0 | `q8x4x2` | `repack_q8_0_q8x4x2()` |
| IQ4_NL | `q4x4x2` | 复用 `repack_q4_0_q4x4x2()`; block layout 与 Q4_0 相同 |
| MXFP4 | `mxfp4x4x2` | `repack_mxfp4_mxfp4x4x2()` |
| F16/F32/其它类型 | 原样 memcpy | 无 layout transform |

`x4x2` 的核心含义是: 每行以 256 个 K 元素为一个 logical block, 也就是 8 个 32-element GGML quant block。repack 后把 8 个 block 的 quant 主数据合并在前面, scale/min/e8m0 元数据放在后面。这个 layout 更适合 HVX/HMX 连续加载和按 32x32 tile dequant。

每行 layout:

```text
Q4_0 / IQ4_NL x4x2:
  [quants: nb * 128 bytes] [scales d: nb * 8 * fp16]
  row stride = nb * (128 + 16) = 144 * nb

Q4_1 x4x2:
  [quants: nb * 128 bytes] [d,m: nb * 8 * 2 * fp16]
  row stride = nb * (128 + 32) = 160 * nb

Q8_0 x4x2:
  [quants: nb * 256 bytes] [scales d: nb * 8 * fp16]
  row stride = nb * (256 + 16) = 272 * nb

MXFP4 x4x2:
  [quants: nb * 128 bytes] [e8m0 scale: nb * 8 bytes]
  row stride = nb * (128 + 8) = 136 * nb

nb = ceil(K / 256)
```

注意这里的 transform 是行内 repack, 不是矩阵 transpose。权重仍按 GGML 的行方向保存, 只是每行内部从 AOS block layout 变成 `quants first, metadata later` 的 x4x2 layout。

runtime matmul 时还有第二阶段临时转换, 但它不属于 disk load:

1. `hmx_matmul_2d_f32()` 按 N chunk 把 repacked x4x2 权重 DMA 到 VTCM。
2. `dequantize_x4x2_weight_chunk_to_fp16_tiles()` 用 HVX 把 x4x2 权重 dequant/convert 成 HMX 需要的 32x32 tile-major FP16。
3. HMX 对 tile-major FP16 做矩阵乘。

因此 Hexagon/NPU 的权重生命周期可以概括为:

```text
disk GGUF block
  -> host set_tensor repack
  -> Hexagon shared buffer x4x2 row layout
  -> runtime DMA to VTCM
  -> HVX dequant/convert to 32x32 tile-major FP16
  -> HMX matmul
```

和 OpenCL 对比:

- OpenCL dense 权重 load 后可能已经是 SOA + transposed buffer, 后续 kernel 直接按转置后的 OpenCL layout 读。
- Hexagon load 后只做 x4x2 row repack; 真正给 HMX 用的 tile-major FP16 是每次 matmul/chunk 在 VTCM 里临时生成的。
- Hexagon 对 Q5_K/Q6_K/Q4_K 等 K-quant 没有 load-time repack 分支; 当前只覆盖 `Q4_0/Q4_1/Q8_0/IQ4_NL/MXFP4`。

### Hexagon/NPU transform benchmark

新增 `llama-hexagon-transform-bench`, 用真实 Hexagon backend `ggml_backend_tensor_set()` 测 load-time transform。测试口径:

- 设备: `CPH2583 / SM8650`, Hexagon backend `HTP`, buffer type `HTP0-REPACK`。
- source buffer、ggml tensor、Hexagon backend buffer 都在计时前创建。
- backend x4x2 repack 的 scratch buffer 改成 thread-local cache; warmup 后正式计时不包含 scratch malloc/free。
- 计时循环只包含 `ggml_backend_tensor_set(tensor, src, 0, nbytes)`。
- 完整原始 CSV: `/tmp/npu_transform_common_nomalloc.log`; 手机端同一份在 `/data/local/tmp/hexagon-transform-bench/npu_transform_common_nomalloc.log`。

完整测试覆盖 `1B/3B/4B/7B/8B` 的 hidden、ffn_up、ffn_down、lm_head shape。按 type 汇总如下:

| type | rows | median ms | median GiB/s | min GiB/s | max GiB/s | 说明 |
|---|---:|---:|---:|---:|---:|---|
| Q4_0 | 20 | 1.372 | 9.28 | 3.31 | 10.76 | x4x2 repack |
| Q4_1 | 20 | 1.586 | 9.50 | 7.31 | 10.01 | x4x2 repack |
| Q8_0 | 20 | 1.846 | 15.05 | 9.58 | 16.45 | x4x2 repack |
| IQ4_NL | 20 | 1.362 | 9.45 | 6.99 | 10.42 | 复用 Q4_0 x4x2 repack |
| MXFP4 | 20 | 1.403 | 8.69 | 6.64 | 9.85 | mxfp4x4x2 repack |
| Q4_K | 20 | 0.412 | 29.75 | 24.44 | 44.98 | 无 NPU repack, memcpy 对照 |
| Q5_K | 20 | 0.477 | 30.98 | 24.61 | 36.94 | 无 NPU repack, memcpy 对照 |
| Q6_K | 20 | 0.632 | 28.52 | 24.36 | 35.75 | 无 NPU repack, memcpy 对照 |
| F16 | 20 | 1.861 | 22.02 | 16.48 | 30.44 | memcpy 对照 |
| F32 | 20 | 4.466 | 21.56 | 20.04 | 22.65 | memcpy 对照 |

结论:

1. 真正有 NPU load-time layout transform 的是 `Q4_0/Q4_1/Q8_0/IQ4_NL/MXFP4`; 速度大致是 Q4/MXFP4 约 9-10 GiB/s, Q8_0 约 15 GiB/s。
2. `Q4_K/Q5_K/Q6_K` 当前没有 Hexagon x4x2 repack 分支, `set_tensor()` 只是 memcpy, 所以 25-31 GiB/s 的带宽不能视为 NPU quant transform 性能。
3. 这个 transform 仍发生在 CPU/host 侧, 只是目标 buffer 是 Hexagon shared buffer; HMX tile-major FP16 仍在 runtime VTCM 中生成。

### Q4_0/Q8_0 CPU vs OpenCL vs NPU

下面只比较当前三条实际有意义的 Q4/Q8 transform 路径:

- CPU: `CPU_REPACK`, `GGML_CPU_REPACK_THREADS=4`
- OpenCL: `dense_adaptive`
- NPU: Hexagon `HTP0-REPACK`, `ggml_backend_tensor_set()`, thread-local scratch warmup 后计时

winner 只按 transform wall/event 时间最小判断。Q4_0: OpenCL 赢 4 组, CPU4T 赢 8 组, NPU 赢 8 组。Q8_0: NPU 赢 11 组, CPU4T 赢 9 组, OpenCL 没有赢的 shape。

| type | shape | KxM | CPU4T ms | CPU GiB/s | OpenCL ms | OpenCL GiB/s | NPU ms | NPU GiB/s | winner |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Q4_0 | 1B_hidden | 2048x2048 | 0.704 | 3.12 | 0.240 | 9.14 | 0.663 | 3.31 | OpenCL |
| Q4_0 | 1B_ffn_up | 2048x5632 | 1.037 | 5.83 | 0.751 | 8.04 | 1.368 | 4.42 | OpenCL |
| Q4_0 | 1B_ffn_down | 5632x2048 | 1.197 | 5.05 | 0.705 | 8.57 | 1.204 | 5.02 | OpenCL |
| Q4_0 | 1B_lm_head_32k | 2048x32000 | 2.785 | 12.33 | 3.421 | 10.03 | 4.613 | 7.44 | CPU4T |
| Q4_0 | 3B_hidden | 3072x3072 | 0.781 | 6.33 | 0.561 | 8.81 | 0.460 | 10.76 | NPU |
| Q4_0 | 3B_ffn_up | 3072x8192 | 1.799 | 7.33 | 1.575 | 8.37 | 1.376 | 9.58 | NPU |
| Q4_0 | 3B_ffn_down | 8192x3072 | 1.533 | 8.60 | 1.801 | 7.32 | 1.252 | 10.53 | NPU |
| Q4_0 | 3B_lm_head_32k | 3072x32000 | 4.198 | 12.27 | 5.296 | 9.72 | 5.934 | 8.68 | CPU4T |
| Q4_0 | 4B_hidden | 2560x2560 | 0.871 | 3.94 | 0.370 | 9.28 | 0.335 | 10.24 | NPU |
| Q4_0 | 4B_ffn_up | 2560x6912 | 1.676 | 5.53 | 0.940 | 9.86 | 1.020 | 9.09 | OpenCL |
| Q4_0 | 4B_ffn_down | 6912x2560 | 1.572 | 5.90 | 2.122 | 4.37 | 0.879 | 10.55 | NPU |
| Q4_0 | 4B_lm_head_32k | 2560x32000 | 3.254 | 13.19 | 4.285 | 10.01 | 6.025 | 7.12 | CPU4T |
| Q4_0 | 7B_hidden | 4096x4096 | 1.045 | 8.41 | 1.012 | 8.68 | 0.827 | 10.63 | NPU |
| Q4_0 | 7B_ffn_up | 4096x11008 | 3.116 | 7.58 | 2.522 | 9.37 | 2.459 | 9.61 | NPU |
| Q4_0 | 7B_ffn_down | 11008x4096 | 2.172 | 10.87 | 3.463 | 6.82 | 2.686 | 8.79 | CPU4T |
| Q4_0 | 7B_lm_head_32k | 4096x32000 | 5.756 | 11.93 | 7.395 | 9.29 | 7.434 | 9.24 | CPU4T |
| Q4_0 | 8B_hidden | 4096x4096 | 1.125 | 7.81 | 1.010 | 8.70 | 0.861 | 10.21 | NPU |
| Q4_0 | 8B_ffn_up | 4096x14336 | 2.190 | 14.05 | 3.535 | 8.70 | 3.302 | 9.32 | CPU4T |
| Q4_0 | 8B_ffn_down | 14336x4096 | 3.099 | 9.93 | 4.244 | 7.25 | 3.868 | 7.95 | CPU4T |
| Q4_0 | 8B_lm_head_32k | 4096x32000 | 5.437 | 12.63 | 7.388 | 9.29 | 7.378 | 9.31 | CPU4T |
| Q8_0 | 1B_hidden | 2048x2048 | 0.840 | 4.94 | 0.362 | 11.47 | 0.252 | 16.45 | NPU |
| Q8_0 | 1B_ffn_up | 2048x5632 | 1.595 | 7.16 | 1.307 | 8.73 | 0.709 | 16.10 | NPU |
| Q8_0 | 1B_ffn_down | 5632x2048 | 1.571 | 7.27 | 1.257 | 9.08 | 0.817 | 13.97 | NPU |
| Q8_0 | 1B_lm_head_32k | 2048x32000 | 4.718 | 13.75 | 7.487 | 8.66 | 4.312 | 15.04 | NPU |
| Q8_0 | 3B_hidden | 3072x3072 | 1.148 | 8.13 | 0.874 | 10.68 | 0.583 | 16.03 | NPU |
| Q8_0 | 3B_ffn_up | 3072x8192 | 2.331 | 10.68 | 2.425 | 10.27 | 1.682 | 14.80 | NPU |
| Q8_0 | 3B_ffn_down | 8192x3072 | 1.649 | 15.10 | 3.065 | 8.12 | 2.009 | 12.40 | CPU4T |
| Q8_0 | 3B_lm_head_32k | 3072x32000 | 5.179 | 18.78 | 9.490 | 10.25 | 6.642 | 14.65 | CPU4T |
| Q8_0 | 4B_hidden | 2560x2560 | 0.721 | 9.00 | 0.613 | 10.59 | 0.417 | 15.55 | NPU |
| Q8_0 | 4B_ffn_up | 2560x6912 | 2.247 | 7.79 | 1.782 | 9.83 | 1.303 | 13.44 | NPU |
| Q8_0 | 4B_ffn_down | 6912x2560 | 2.140 | 8.18 | 2.195 | 7.98 | 1.525 | 11.48 | NPU |
| Q8_0 | 4B_lm_head_32k | 2560x32000 | 5.740 | 14.12 | 8.483 | 9.56 | 6.501 | 12.47 | CPU4T |
| Q8_0 | 7B_hidden | 4096x4096 | 1.466 | 11.33 | 1.569 | 10.58 | 1.073 | 15.48 | NPU |
| Q8_0 | 7B_ffn_up | 4096x11008 | 2.816 | 15.85 | 4.321 | 10.33 | 2.911 | 15.33 | CPU4T |
| Q8_0 | 7B_ffn_down | 11008x4096 | 2.720 | 16.40 | 5.880 | 7.59 | 4.659 | 9.58 | CPU4T |
| Q8_0 | 7B_lm_head_32k | 4096x32000 | 6.240 | 20.79 | 14.876 | 8.72 | 8.528 | 15.21 | CPU4T |
| Q8_0 | 8B_hidden | 4096x4096 | 1.224 | 13.57 | 1.570 | 10.57 | 1.096 | 15.15 | NPU |
| Q8_0 | 8B_ffn_up | 4096x14336 | 3.078 | 18.88 | 5.746 | 10.11 | 3.820 | 15.21 | CPU4T |
| Q8_0 | 8B_ffn_down | 14336x4096 | 2.770 | 20.98 | 8.088 | 7.18 | 4.990 | 11.64 | CPU4T |
| Q8_0 | 8B_lm_head_32k | 4096x32000 | 6.306 | 20.57 | 14.633 | 8.86 | 8.606 | 15.07 | CPU4T |

## 12. 关键结论

1. FP16/F32 在 OpenCL load 阶段没有 prepack, 只是写入 buffer。
2. BF16 在 load 阶段转换成 FP16 存。
3. Q8/Q4/Q5/Q6/MXFP4/IQ4_NL/K-quant 的主要转换是 AOS -> SOA, 把 scale/min/high_bits/quants 拆成连续数组。
4. `_noshuffle` 实际表示“load 时已经把 bit/nibble 解到自然顺序, 计算 kernel 不需要再 shuffle”。
5. Adreno dense 优化主要是 `_noshuffle + buffer transpose`; Adreno MoE 优化主要是 `_trans4_ns + image1d_buffer`。
6. Q2_K/Q3_K 当前没有 OpenCL load-time SOA transform, 仍按磁盘 block layout 存在 device buffer 中。

## 13. GPU transform benchmark

本仓库新增了 `llama-opencl-transform-bench`, 用来单独测 OpenCL 权重 transform 的 GPU 时间。

构建:

```bash
cmake --build build-snapdragon --target llama-opencl-transform-bench -j
```

Android/Adreno 上运行:

```bash
adb shell 'rm -rf /data/local/tmp/opencl-transform-bench && mkdir -p /data/local/tmp/opencl-transform-bench'
adb push build-snapdragon/bin/llama-opencl-transform-bench \
    ggml/src/ggml-opencl/kernels/cvt.cl \
    ggml/src/ggml-opencl/kernels/transpose.cl \
    /data/local/tmp/opencl-transform-bench/
adb shell 'cd /data/local/tmp/opencl-transform-bench && chmod +x llama-opencl-transform-bench'
adb shell 'cd /data/local/tmp/opencl-transform-bench && ./llama-opencl-transform-bench --kernel-dir . --type q4_0 --mode all --k 4096 --m 4096'
```

输出里会同时打印:

- `GiB/s`: 按输入 tensor 的原始字节数计算。
- `Gelem/s`: 按 `K*M` 量化元素数计算。
- `GFLOP/s`: 等效 FLOPS, 计算公式是 `K*M*flops_per_elem/time`; 默认 `--flops-per-elem 1.0`。transform 本身不是 matmul, 所以这个字段只用于统一吞吐口径, 不是硬件真实矩阵算力。

计时口径:

- 使用 OpenCL event profiling。
- 不计 `clCreateBuffer`, host upload, subbuffer 创建, kernel 编译。
- `standard` 只计 AOS -> SOA convert kernel。
- `dense` 计 convert/noshuffle kernel + transpose kernel + transpose copy; 不计 transpose 临时 buffer 创建。
- `moe` 只计 `_trans4_ns` kernel; 不计 `clCreateImage`。

OnePlus 12 / Adreno 750, `K=M=4096`, `iters=5` 的一次短跑结果:

```text
q4_0 dense  total_med=11.307 ms GiB/s=0.77 Gelem/s=1.48 GFLOP/s=1.48  convert=0.003 ms  transpose_kernel=10.975 ms  transpose_copy=0.329 ms
q8_0 dense  total_med= 6.734 ms GiB/s=2.32 Gelem/s=2.49 GFLOP/s=2.49  convert=0.003 ms  transpose_kernel= 6.076 ms  transpose_copy=0.657 ms
q5_k dense  total_med=16.871 ms GiB/s=0.68 Gelem/s=0.99 GFLOP/s=0.99  convert=0.003 ms  transpose_kernel=16.490 ms  transpose_copy=0.381 ms
q6_k dense  total_med=23.121 ms GiB/s=0.62 Gelem/s=0.73 GFLOP/s=0.73  convert=0.003 ms  transpose_kernel=22.635 ms  transpose_copy=0.482 ms

q4_0 moe    total_med= 0.691 ms ...
q4_1 moe    total_med= 1.029 ms ...
q5_0 moe    total_med= 0.843 ms ...
q5_1 moe    total_med= 2.683 ms ...
q4_k moe    total_med=13.647 ms ...
q5_k moe    total_med=10.254 ms ...
q6_k moe    total_med=10.485 ms ...
mxfp4 moe   total_med= 0.777 ms ...
```

这组数据说明: 对 Adreno dense 路径, 纯 convert/split kernel 很小, 主要成本来自 buffer transpose; MoE 路径则主要由 `_trans4_ns` kernel 自身承担。

## 14. CPU repack vs GPU dense transform

下面是同一台 adb 连接设备上的 CPU/GPU transform 对比。注意本次设备实际回报为:

```text
model=CPH2583
soc=SM8650
gpu=Adreno 750
```

也就是 Snapdragon 8 Gen 3 设备, 不是 OP13 / Snapdragon 8 Elite。若要 OP13 数据, 需要接入 OP13 后用相同命令重跑。

计时口径:

- CPU: `bench-repack-op`, `taskset 80`, 只包住 `ggml_backend_tensor_set()` 触发的 CPU_REPACK。
- GPU: `llama-opencl-transform-bench --mode dense`, 只统计 OpenCL event profiling 下的 convert/noshuffle + transpose kernel + transpose copy; 不计 buffer 创建、host upload、kernel 编译。
- GPU dense 是普通 Adreno 大矩阵权重会用的路径, 不是 MoE image 路径。
- `G/s` 按矩阵元素数 `K*M` 计算, 即每秒处理多少十亿个量化元素。
- benchmark 输出中的 `GFLOP/s` 默认等于 `Gelem/s`, 因为默认 `--flops-per-elem 1.0`; 如果要按其他等效 FLOP 口径, 运行时传入对应的 `--flops-per-elem`。

### Q4_0

| shape | CPU repack ms | CPU G/s | GPU dense ms | GPU G/s | GPU/CPU |
|---|---:|---:|---:|---:|---:|
| 1024x1024 | 0.115 | 9.12 | 0.157 | 6.68 | 1.37x |
| 2048x2048 | 0.474 | 8.84 | 1.549 | 2.71 | 3.26x |
| 4096x4096 | 1.832 | 9.16 | 11.330 | 1.48 | 6.18x |
| 8192x8192 | 7.504 | 8.94 | 58.159 | 1.15 | 7.75x |
| 4096x1024 | 0.473 | 8.86 | 1.727 | 2.43 | 3.65x |
| 1024x4096 | 0.463 | 9.07 | 0.721 | 5.82 | 1.56x |
| 4096x14336 | 6.806 | 8.63 | 28.060 | 2.09 | 4.12x |
| 14336x4096 | 6.314 | 9.30 | 64.799 | 0.91 | 10.26x |

### Q8_0

| shape | CPU repack ms | CPU G/s | GPU dense ms | GPU G/s | GPU/CPU |
|---|---:|---:|---:|---:|---:|
| 1024x1024 | 0.328 | 3.20 | 0.335 | 3.13 | 1.02x |
| 2048x2048 | 1.295 | 3.24 | 2.206 | 1.90 | 1.70x |
| 4096x4096 | 5.358 | 3.13 | 6.742 | 2.49 | 1.26x |
| 8192x8192 | 21.961 | 3.06 | 62.638 | 1.07 | 2.85x |
| 4096x1024 | 1.340 | 3.13 | 3.190 | 1.31 | 2.38x |
| 1024x4096 | 1.349 | 3.11 | 1.031 | 4.07 | 0.76x |
| 4096x14336 | 20.195 | 2.91 | 27.682 | 2.12 | 1.37x |
| 14336x4096 | 19.422 | 3.02 | 70.798 | 0.83 | 3.65x |

### Q5_K

| shape | CPU repack ms | CPU G/s | GPU dense ms | GPU G/s | GPU/CPU |
|---|---:|---:|---:|---:|---:|
| 1024x1024 | 0.257 | 4.09 | 0.259 | 4.05 | 1.01x |
| 2048x2048 | 1.093 | 3.84 | 3.219 | 1.30 | 2.94x |
| 4096x4096 | 4.350 | 3.86 | 16.867 | 0.99 | 3.88x |
| 8192x8192 | 19.668 | 3.41 | 78.899 | 0.85 | 4.01x |
| 4096x1024 | 1.100 | 3.81 | 3.444 | 1.22 | 3.13x |
| 1024x4096 | 1.095 | 3.83 | 1.473 | 2.85 | 1.35x |
| 4096x14336 | 17.438 | 3.37 | 35.704 | 1.64 | 2.05x |
| 14336x4096 | 16.980 | 3.46 | 87.605 | 0.67 | 5.16x |

### Q6_K

| shape | CPU repack ms | CPU G/s | GPU dense ms | GPU G/s | GPU/CPU |
|---|---:|---:|---:|---:|---:|
| 1024x1024 | 0.608 | 1.73 | 0.339 | 3.09 | 0.56x |
| 2048x2048 | 2.495 | 1.68 | 3.242 | 1.29 | 1.30x |
| 4096x4096 | 11.246 | 1.49 | 23.089 | 0.73 | 2.05x |
| 8192x8192 | 44.232 | 1.52 | 113.925 | 0.59 | 2.58x |
| 4096x1024 | 2.490 | 1.68 | 3.460 | 1.21 | 1.39x |
| 1024x4096 | 2.495 | 1.68 | 2.051 | 2.05 | 0.82x |
| 4096x14336 | 39.089 | 1.50 | 48.149 | 1.22 | 1.23x |
| 14336x4096 | 38.469 | 1.53 | 126.516 | 0.46 | 3.29x |

结论:

1. Adreno GPU dense transform 对 shape 很敏感, 尤其 `K` 很大、`M` 较小时非常慢, 例如 `14336x4096`。
2. GPU dense transform 的瓶颈基本是 transpose, 不是 AOS->SOA convert。实测 4096x4096 Q4_0 中, convert 约 0.003 ms, transpose kernel 约 11 ms。
3. CPU repack 更像线性 streaming, 对 `4096x14336` 和 `14336x4096` 这类同字节量 shape 更稳定。
4. 对普通 dense 权重加载, CPU repack 在这台设备上大多数 shape 明显快于 OpenCL dense transform; GPU 只有少数小 K / 大 M 或 Q6_K 小 shape 场景接近或更快。

### Tiled transpose retest on LLM weight shapes

原始 `kernel_transpose_*_buf` 使用 `global={stride, rows}`, `local={64,1}`。同一 workgroup 内读是连续的, 但写是 `output[x*rows + y]`, 相邻 work-item 写地址相隔 `rows` 个元素。对 `Q4_0 4096x4096` 的 `q[]` 来说, 相邻写地址相隔 `4096*sizeof(uint16_t)=8192` 字节, 在 Adreno 上非常慢。

OpenCL backend 默认 dense transform 已切到 `16x16` local-memory tiled transpose。benchmark 中 `dense` 默认也是 tiled; 旧的 `{64,1}` scalar-strided transpose 保留为 `dense_scalar` 只用于对照。下面仍是在同一台 `CPH2583 / SM8650 / Adreno 750` 上的结果。`GPU/CPU < 1` 表示 GPU 更快。

| type | shape | KxM | CPU ms | CPU GiB/s | GPU dense ms | GPU GiB/s | GPU/CPU |
|---|---|---:|---:|---:|---:|---:|---:|
| Q4_0 | hidden 4k | 4096x4096 | 1.829 | 4.80 | 1.590 | 5.53 | 0.87x |
| Q4_0 | ffn up 7B | 4096x11008 | 5.146 | 4.59 | 2.468 | 9.57 | 0.48x |
| Q4_0 | ffn down 7B | 11008x4096 | 4.816 | 4.90 | 6.621 | 3.57 | 1.37x |
| Q4_0 | ffn up 8B | 4096x14336 | 6.615 | 4.65 | 5.204 | 5.91 | 0.79x |
| Q4_0 | ffn down 8B | 14336x4096 | 6.353 | 4.84 | 9.283 | 3.31 | 1.46x |
| Q4_0 | lm head 32k | 4096x32000 | 15.552 | 4.42 | 7.275 | 9.44 | 0.47x |
| Q4_0 | hidden 8k | 8192x8192 | 7.415 | 4.74 | 8.870 | 3.96 | 1.20x |
| Q8_0 | hidden 4k | 4096x4096 | 5.358 | 3.10 | 1.584 | 10.48 | 0.30x |
| Q8_0 | ffn up 7B | 4096x11008 | 14.917 | 2.99 | 4.395 | 10.15 | 0.29x |
| Q8_0 | ffn down 7B | 11008x4096 | 14.867 | 3.00 | 6.468 | 6.90 | 0.44x |
| Q8_0 | ffn up 8B | 4096x14336 | 20.223 | 2.87 | 5.940 | 9.78 | 0.29x |
| Q8_0 | ffn down 8B | 14336x4096 | 19.475 | 2.98 | 8.881 | 6.54 | 0.46x |
| Q8_0 | lm head 32k | 4096x32000 | 45.229 | 2.87 | 15.308 | 8.47 | 0.34x |
| Q8_0 | hidden 8k | 8192x8192 | 22.483 | 2.95 | 8.890 | 7.47 | 0.40x |
| Q5_K | hidden 4k | 4096x4096 | 4.337 | 2.48 | 2.129 | 5.05 | 0.49x |
| Q5_K | ffn up 7B | 4096x11008 | 13.220 | 2.18 | 2.989 | 9.66 | 0.23x |
| Q5_K | ffn down 7B | 11008x4096 | 12.897 | 2.24 | 8.669 | 3.33 | 0.67x |
| Q5_K | ffn up 8B | 4096x14336 | 17.478 | 2.15 | 6.365 | 5.91 | 0.36x |
| Q5_K | ffn down 8B | 14336x4096 | 17.078 | 2.20 | 12.555 | 2.99 | 0.74x |
| Q5_K | lm head 32k | 4096x32000 | 39.582 | 2.12 | 8.495 | 9.88 | 0.21x |
| Q5_K | hidden 8k | 8192x8192 | 19.672 | 2.18 | 10.641 | 4.04 | 0.54x |
| Q6_K | hidden 4k | 4096x4096 | 11.193 | 1.15 | 2.854 | 4.49 | 0.25x |
| Q6_K | ffn up 7B | 4096x11008 | 29.895 | 1.15 | 3.808 | 9.05 | 0.13x |
| Q6_K | ffn down 7B | 11008x4096 | 29.447 | 1.17 | 11.466 | 3.00 | 0.39x |
| Q6_K | ffn up 8B | 4096x14336 | 39.044 | 1.15 | 8.565 | 5.24 | 0.22x |
| Q6_K | ffn down 8B | 14336x4096 | 38.445 | 1.17 | 15.829 | 2.83 | 0.41x |
| Q6_K | lm head 32k | 4096x32000 | 87.661 | 1.14 | 10.872 | 9.21 | 0.12x |
| Q6_K | hidden 8k | 8192x8192 | 44.184 | 1.16 | 14.532 | 3.53 | 0.33x |

新的结论:

1. tiled transpose 后, GPU dense transform 对 7B/8B 常见 shape 大多快于 CPU repack。
2. Q4_0 仍然对方向很敏感: `4096x11008` / `4096x14336` 这类 FFN up/gate 较快, 但 `11008x4096` / `14336x4096` 这类 FFN down 仍慢于 CPU。
3. Q6_K 在这组 shape 下全部快于 CPU, 因为 CPU repack 的 Q6_K 代价本身较高。
4. tiled kernel 已用 `transpose_check` 在 `4096x14336` 和 `14336x4096` 上对 Q4_0/Q8_0/Q5_K/Q6_K 做过逐字节对比, 输出与旧 scalar transpose 一致。

### CPU repack multi-thread experiment

CPU_REPACK 的 `set_tensor()` 路径默认仍保持原来的单线程 fast path。为了测 CPU 侧上限, 热路径 Q4_0/Q8_0/Q5_K/Q6_K 增加了可选 row-group 并行 repack, 通过环境变量打开:

```bash
GGML_CPU_REPACK_THREADS=4 ./bench-repack-llm-op
```

这个实现不依赖 OpenMP; 每个线程处理独立的 row group, 输出区间不重叠。`4096x14336` 和 `14336x4096` 上已验证 `GGML_CPU_REPACK_THREADS=8` 与单线程输出逐字节一致。

下面是同一台 `CPH2583 / SM8650` 上 `1T` 和 `4T` 的 CPU_REPACK 对比。`4T min` 比 median 更接近多核理论上限, 因为当前实验实现每次 tensor repack 都创建线程, median 会受调度抖动影响。

| type | shape | 1T med ms | 4T med ms | speedup | 4T min ms | min speedup |
|---|---|---:|---:|---:|---:|---:|
| Q4_0 | hidden 4k | 1.869 | 1.937 | 0.96x | 1.180 | 1.58x |
| Q4_0 | ffn up 7B | 5.209 | 4.394 | 1.19x | 3.926 | 1.33x |
| Q4_0 | ffn down 7B | 4.860 | 5.049 | 0.96x | 3.914 | 1.24x |
| Q4_0 | ffn up 8B | 6.823 | 5.191 | 1.31x | 3.842 | 1.78x |
| Q4_0 | ffn down 8B | 6.387 | 7.614 | 0.84x | 4.742 | 1.35x |
| Q4_0 | lm head 32k | 16.283 | 11.520 | 1.41x | 8.777 | 1.86x |
| Q4_0 | hidden 8k | 7.693 | 6.675 | 1.15x | 5.720 | 1.34x |
| Q8_0 | hidden 4k | 5.428 | 5.335 | 1.02x | 4.681 | 1.16x |
| Q8_0 | ffn up 7B | 14.882 | 12.172 | 1.22x | 8.901 | 1.67x |
| Q8_0 | ffn down 7B | 14.658 | 11.092 | 1.32x | 8.192 | 1.79x |
| Q8_0 | ffn up 8B | 19.981 | 16.357 | 1.22x | 12.577 | 1.59x |
| Q8_0 | ffn down 8B | 19.092 | 12.939 | 1.48x | 10.462 | 1.82x |
| Q8_0 | lm head 32k | 44.486 | 34.352 | 1.29x | 30.645 | 1.45x |
| Q8_0 | hidden 8k | 22.062 | 16.018 | 1.38x | 12.872 | 1.71x |
| Q5_K | hidden 4k | 4.128 | 3.656 | 1.13x | 1.711 | 2.41x |
| Q5_K | ffn up 7B | 11.854 | 8.299 | 1.43x | 8.098 | 1.46x |
| Q5_K | ffn down 7B | 11.584 | 8.103 | 1.43x | 4.820 | 2.40x |
| Q5_K | ffn up 8B | 15.465 | 11.752 | 1.32x | 8.101 | 1.91x |
| Q5_K | ffn down 8B | 14.987 | 8.023 | 1.87x | 6.872 | 2.18x |
| Q5_K | lm head 32k | 34.682 | 13.676 | 2.54x | 12.081 | 2.87x |
| Q5_K | hidden 8k | 17.328 | 6.212 | 2.79x | 6.129 | 2.83x |
| Q6_K | hidden 4k | 11.232 | 3.294 | 3.41x | 3.145 | 3.57x |
| Q6_K | ffn up 7B | 30.058 | 8.143 | 3.69x | 8.057 | 3.73x |
| Q6_K | ffn down 7B | 29.913 | 8.347 | 3.58x | 8.075 | 3.70x |
| Q6_K | ffn up 8B | 39.431 | 19.151 | 2.06x | 11.270 | 3.50x |
| Q6_K | ffn down 8B | 38.893 | 10.473 | 3.71x | 10.374 | 3.75x |
| Q6_K | lm head 32k | 88.293 | 23.734 | 3.72x | 22.899 | 3.86x |
| Q6_K | hidden 8k | 44.735 | 12.941 | 3.46x | 12.037 | 3.72x |

观察:

1. Q6_K 最适合 CPU 多线程 repack, 4T median 基本接近 3.5x-3.7x。
2. Q5_K 在大 shape 上有明显收益, 但小 shape 受线程创建和调度开销影响。
3. Q4_0/Q8_0 更接近内存搬运, 4T median 提升有限, 有些 shape 反而变慢; 这里需要 persistent threadpool 才能更接近 `min` 对应的上限。
4. 如果要把 CPU 多线程 repack 做成默认优化, 下一步应该复用 ggml threadpool 或实现持久 worker, 不应每个 tensor 创建一次线程。

### CPU Q8_0 direct-pack optimization

Q8_0 原来的 CPU repack 路径会先把 4 个 `block_q8_0` 复制到 `dst_tmp[4]`, 再调用 `make_block_q8_0x4()` 返回一个完整结构体。优化后改成直接从 4 行源 block 打包到目标 `block_q8_0x4`, 去掉临时 block 和结构体返回。

逐字节检查: `4096x14336` 和 `14336x4096` 上, Q8_0 `GGML_CPU_REPACK_THREADS=8` 与 `1` 线程输出一致。

| shape | old CPU 1T ms | new CPU 1T ms | speedup | new CPU 1T GiB/s | GPU dense ms | GPU GiB/s | GPU/CPU1T |
|---|---:|---:|---:|---:|---:|---:|---:|
| hidden 4k | 5.428 | 0.977 | 5.56x | 16.99 | 1.649 | 10.07 | 1.69x |
| ffn up 7B | 14.882 | 3.041 | 4.89x | 14.67 | 4.530 | 9.85 | 1.49x |
| ffn down 7B | 14.658 | 2.714 | 5.40x | 16.44 | 6.934 | 6.43 | 2.55x |
| ffn up 8B | 19.981 | 3.888 | 5.14x | 14.95 | 5.868 | 9.90 | 1.51x |
| ffn down 8B | 19.092 | 3.505 | 5.45x | 16.58 | 9.095 | 6.39 | 2.60x |
| lm head 32k | 44.486 | 9.327 | 4.77x | 13.91 | 14.366 | 9.03 | 1.54x |
| hidden 8k | 22.062 | 4.538 | 4.86x | 14.63 | 9.085 | 7.31 | 2.00x |

新的 Q8_0 结论: CPU direct-pack 已经比 GPU tiled dense transform 快。4T 对 Q8_0 不总是更好, 因为单线程 direct-pack 已接近内存带宽, 临时创建线程会带来调度和 cache 干扰。

### GPU fused convert-transpose experiment

为了验证 GPU 版本是否还能继续优化, 增加了实验模式 `dense_fused` / `dense_fused_wall`:

- `dense_fused` 在 `cvt.cl` 中为 Q4_0/Q8_0 直接把 GGUF block 写成 dense-transposed OpenCL layout, 省掉后续 transpose kernel 和 copy-back。
- `dense_fused_check` 用旧的 `dense_tiled` 输出做基准, 对 q/d buffer 逐字节比较。`4096x4096` 和 `14336x4096` 上 Q4_0/Q8_0 都通过。
- Adreno 750 上 convert 类 kernel 的 OpenCL event profiling 明显低估, 会出现 0.001-0.003 ms 这种不符合写入带宽的结果。因此 fused 性能以 `dense_fused_wall` 的 CPU wall average 为准; kernel 和 args 预先创建, 计时循环只 enqueue kernel, 最后统一 `clFinish`。

同一台 `CPH2583 / SM8650 / Adreno 750` 上, 常用 LLM shape 的结果:

| type | shape | KxM | dense tiled ms | dense tiled GiB/s | fused wall ms | fused wall GiB/s | fused/dense |
|---|---|---:|---:|---:|---:|---:|---:|
| Q4_0 | hidden 4k | 4096x4096 | 1.621 | 5.42 | 1.119 | 7.85 | 0.69x |
| Q4_0 | ffn up 7B | 4096x11008 | 2.527 | 9.35 | 5.094 | 4.64 | 2.02x |
| Q4_0 | ffn down 7B | 11008x4096 | 7.328 | 3.22 | 1.988 | 11.88 | 0.27x |
| Q4_0 | ffn up 8B | 4096x14336 | 5.196 | 5.92 | 6.735 | 4.57 | 1.30x |
| Q4_0 | ffn down 8B | 14336x4096 | 9.599 | 3.20 | 4.066 | 7.56 | 0.42x |
| Q4_0 | lm head 32k | 4096x32000 | 7.365 | 9.32 | 14.840 | 4.63 | 2.01x |
| Q4_0 | hidden 8k | 8192x8192 | 8.796 | 4.00 | 19.264 | 1.82 | 2.19x |
| Q8_0 | hidden 4k | 4096x4096 | 1.618 | 10.26 | 2.803 | 5.92 | 1.73x |
| Q8_0 | ffn up 7B | 4096x11008 | 4.347 | 10.26 | 11.054 | 4.04 | 2.54x |
| Q8_0 | ffn down 7B | 11008x4096 | 6.351 | 7.03 | 4.201 | 10.62 | 0.66x |
| Q8_0 | ffn up 8B | 4096x14336 | 5.949 | 9.77 | 14.520 | 4.00 | 2.44x |
| Q8_0 | ffn down 8B | 14336x4096 | 8.789 | 6.61 | 11.022 | 5.27 | 1.25x |
| Q8_0 | lm head 32k | 4096x32000 | 14.449 | 8.98 | 32.905 | 3.94 | 2.28x |
| Q8_0 | hidden 8k | 8192x8192 | 9.028 | 7.36 | 31.467 | 2.11 | 3.49x |

结论:

1. naive fused 不是默认替换 tiled transpose 的好方案。它在 `K > M` 的 Q4_0 down-proj shape 上很快, 但在更常见的 `K=4096, M` 很大的 up/gate/lm_head 上明显更慢。
2. 原因是 fused kernel 当前是一个 GGUF block 一个 work-item, 单 work-item 内做 unpack 并写多个输出列; 并行度和内存合并不如 tiled transpose 稳定。
3. GPU 仍有可优化空间, 但更合理的方向不是这个 naive fused, 而是:
   - 按输出 tile 切分 fused kernel, 让 workgroup 协作 unpack 并 coalesced store;
   - 对 Q4_0/Q8_0 的 `K > M` shape 可选择性使用 fused;
   - 对 Q5_K/Q6_K 先分析字段占比, 再决定是否为 q/qh/scale 分别 fused, 避免把复杂 bit repack 串行化到单 work-item。

### GPU optimization follow-up: copy-back and tile size

继续测试了 3 个方向:

1. `dense_nocopy`: convert 后只跑 transpose kernel, 不把 temp copy 回原 buffer。这个模式衡量“后端改成最终 transposed buffer ownership”能达到的上限。
2. `dense_fused`: Q4_0/Q8_0 直接 convert 到 transposed layout。正确但 naive block-level fused 不稳定, 上面已经说明不适合默认替换。
3. `dense_adaptive`: 保留 convert + transpose + copy-back, 但 transpose tile 在 `16x16` 和 `32x32` 之间按 shape 选择。

#### no-copy upper bound

`dense_nocopy` 的收益稳定, 因为现有路径里 copy-back 对大字段占比很高。例子:

| type | shape | dense ms | dense_nocopy ms | speedup |
|---|---:|---:|---:|---:|
| Q4_0 | 4096x32000 | 7.370 | 4.599 | 1.60x |
| Q8_0 | 4096x32000 | 14.396 | 8.997 | 1.60x |
| Q5_K | 4096x32000 | 8.610 | 5.448 | 1.58x |
| Q6_K | 4096x32000 | 11.423 | 7.293 | 1.57x |
| Q8_0 | 4096x14336 | 5.965 | 3.567 | 1.67x |
| Q6_K | 4096x14336 | 8.703 | 6.988 | 1.25x |

这个优化不能只删掉 `clEnqueueCopyBuffer`, 因为当前 `extra->q/d/...` 指向最终 buffer 的 subbuffer, convert 先写进去, transpose 再通过 temp copy 回同一个 subbuffer。真正落地需要改成:

1. convert 写入临时未转置 buffer;
2. transpose kernel 直接写入 `extra->q/d/...` 最终 buffer;
3. 后续 matmul 继续使用 `extra->q/d/...`。

这会改变 set_tensor 阶段的 buffer ownership, 风险高于单纯换 kernel tile, 但收益上限最大。

#### 32x32 tiled transpose

新增了 `kernel_transpose_{8,16,32}_buf_tiled32`, 并用 `transpose_check32` 对 Q4_0/Q8_0/Q5_K/Q6_K 的 `4096x14336`, `14336x4096`, `8192x8192` 做逐字节检查, 输出与旧 scalar transpose 一致。

`8x8` tile 明显慢于 `16x16`, 不保留为后端默认。`32x32` 不是全局更快, 但对 rows 为 1024 倍数的常见 LLM shape 和 down-proj 很有效。后端现在使用这个规则:

```text
use 32x32 if rows % 1024 == 0 or stride >= 2048
otherwise keep 16x16
```

其中 `rows` 是被 transpose 字段的行数, 对权重 transform 来说通常等于 `M`; `stride` 是字段的行内 element 数, 例如 Q4_0 q 字段是 `K/4`。

同一台 `CPH2583 / SM8650 / Adreno 750` 上, `dense_adaptive` 对比原 `dense`:

| type | shape | dense ms | adaptive ms | speedup |
|---|---:|---:|---:|---:|
| Q4_0 | 4096x11008 | 2.523 | 2.520 | 1.00x |
| Q4_0 | 4096x14336 | 5.242 | 3.529 | 1.49x |
| Q4_0 | 4096x32000 | 7.388 | 7.414 | 1.00x |
| Q4_0 | 14336x4096 | 9.621 | 4.243 | 2.27x |
| Q4_0 | 8192x8192 | 8.816 | 4.504 | 1.96x |
| Q8_0 | 4096x11008 | 4.346 | 4.328 | 1.00x |
| Q8_0 | 4096x14336 | 5.963 | 5.777 | 1.03x |
| Q8_0 | 4096x32000 | 14.617 | 14.624 | 1.00x |
| Q8_0 | 14336x4096 | 8.823 | 8.028 | 1.10x |
| Q8_0 | 8192x8192 | 9.115 | 8.202 | 1.11x |
| Q5_K | 4096x11008 | 2.919 | 2.915 | 1.00x |
| Q5_K | 4096x14336 | 6.322 | 4.567 | 1.38x |
| Q5_K | 4096x32000 | 8.621 | 8.635 | 1.00x |
| Q5_K | 14336x4096 | 12.578 | 5.672 | 2.22x |
| Q5_K | 8192x8192 | 10.733 | 5.723 | 1.88x |
| Q6_K | 4096x11008 | 3.743 | 3.734 | 1.00x |
| Q6_K | 4096x14336 | 8.785 | 6.358 | 1.38x |
| Q6_K | 4096x32000 | 11.440 | 11.444 | 1.00x |
| Q6_K | 14336x4096 | 16.720 | 8.975 | 1.86x |
| Q6_K | 8192x8192 | 14.771 | 8.669 | 1.70x |

结论:

1. 32x32 adaptive 是目前最适合直接落后端的 GPU 优化: 正确性已验证, 对不适合的 shape 基本不变, 对 square/down-proj/8B FFN 有明显收益。
2. no-copy 是下一步最值得做的结构性优化, 但需要重排 set_tensor 的临时 buffer 和最终 buffer 写入方向。
3. naive fused 不作为默认路径; 如果后续要做 fused, 应该做 workgroup tiled fused, 而不是一个 block 一个 work-item。

### CPU 4T vs OpenCL on 1B/3B/4B/7B/8B shapes

这组测试把 CPU 默认设为 `4T` (`GGML_CPU_REPACK_THREADS=4`), 对比 OpenCL 默认 `dense_adaptive` transform。测试设备仍是 `CPH2583 / SM8650 / Adreno 750`。时间单位是 ms, 带宽是 GiB/s; `OpenCL/CPU < 1` 表示 OpenCL 更快。

shape 选择覆盖常见 LLM 权重:

- 1B: hidden `2048`, FFN `5632`, lm_head `32000`
- 3B: hidden `3072`, FFN `8192`, lm_head `32000`
- 4B: hidden `2560`, FFN `6912`, lm_head `32000`
- 7B: hidden `4096`, FFN `11008`, lm_head `32000`
- 8B: hidden `4096`, FFN `14336`, lm_head `32000`

完整结果如下。总计 80 组, CPU4T 赢 25 组, OpenCL 赢 55 组; 按 type 看: Q4_0 是 10:10, Q8_0 是 CPU4T 14:6, Q5_K 是 OpenCL 19:1, Q6_K 是 OpenCL 20:0。

| type | shape | KxM | CPU 4T ms | CPU GiB/s | OpenCL ms | OpenCL GiB/s | OpenCL/CPU | winner |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Q4_0 | 1B_hidden | 2048x2048 | 0.704 | 3.12 | 0.240 | 9.14 | 0.34x | OpenCL |
| Q4_0 | 1B_ffn_up | 2048x5632 | 1.037 | 5.83 | 0.751 | 8.04 | 0.72x | OpenCL |
| Q4_0 | 1B_ffn_down | 5632x2048 | 1.197 | 5.05 | 0.705 | 8.57 | 0.59x | OpenCL |
| Q4_0 | 1B_lm_head_32k | 2048x32000 | 2.785 | 12.33 | 3.421 | 10.03 | 1.23x | CPU4T |
| Q4_0 | 3B_hidden | 3072x3072 | 0.781 | 6.33 | 0.561 | 8.81 | 0.72x | OpenCL |
| Q4_0 | 3B_ffn_up | 3072x8192 | 1.799 | 7.33 | 1.575 | 8.37 | 0.88x | OpenCL |
| Q4_0 | 3B_ffn_down | 8192x3072 | 1.533 | 8.60 | 1.801 | 7.32 | 1.17x | CPU4T |
| Q4_0 | 3B_lm_head_32k | 3072x32000 | 4.198 | 12.27 | 5.296 | 9.72 | 1.26x | CPU4T |
| Q4_0 | 4B_hidden | 2560x2560 | 0.871 | 3.94 | 0.370 | 9.28 | 0.42x | OpenCL |
| Q4_0 | 4B_ffn_up | 2560x6912 | 1.676 | 5.53 | 0.940 | 9.86 | 0.56x | OpenCL |
| Q4_0 | 4B_ffn_down | 6912x2560 | 1.572 | 5.90 | 2.122 | 4.37 | 1.35x | CPU4T |
| Q4_0 | 4B_lm_head_32k | 2560x32000 | 3.254 | 13.19 | 4.285 | 10.01 | 1.32x | CPU4T |
| Q4_0 | 7B_hidden | 4096x4096 | 1.045 | 8.41 | 1.012 | 8.68 | 0.97x | OpenCL |
| Q4_0 | 7B_ffn_up | 4096x11008 | 3.116 | 7.58 | 2.522 | 9.37 | 0.81x | OpenCL |
| Q4_0 | 7B_ffn_down | 11008x4096 | 2.172 | 10.87 | 3.463 | 6.82 | 1.59x | CPU4T |
| Q4_0 | 7B_lm_head_32k | 4096x32000 | 5.756 | 11.93 | 7.395 | 9.29 | 1.28x | CPU4T |
| Q4_0 | 8B_hidden | 4096x4096 | 1.125 | 7.81 | 1.010 | 8.70 | 0.90x | OpenCL |
| Q4_0 | 8B_ffn_up | 4096x14336 | 2.190 | 14.05 | 3.535 | 8.70 | 1.61x | CPU4T |
| Q4_0 | 8B_ffn_down | 14336x4096 | 3.099 | 9.93 | 4.244 | 7.25 | 1.37x | CPU4T |
| Q4_0 | 8B_lm_head_32k | 4096x32000 | 5.437 | 12.63 | 7.388 | 9.29 | 1.36x | CPU4T |
| Q8_0 | 1B_hidden | 2048x2048 | 0.840 | 4.94 | 0.362 | 11.47 | 0.43x | OpenCL |
| Q8_0 | 1B_ffn_up | 2048x5632 | 1.595 | 7.16 | 1.307 | 8.73 | 0.82x | OpenCL |
| Q8_0 | 1B_ffn_down | 5632x2048 | 1.571 | 7.27 | 1.257 | 9.08 | 0.80x | OpenCL |
| Q8_0 | 1B_lm_head_32k | 2048x32000 | 4.718 | 13.75 | 7.487 | 8.66 | 1.59x | CPU4T |
| Q8_0 | 3B_hidden | 3072x3072 | 1.148 | 8.13 | 0.874 | 10.68 | 0.76x | OpenCL |
| Q8_0 | 3B_ffn_up | 3072x8192 | 2.331 | 10.68 | 2.425 | 10.27 | 1.04x | CPU4T |
| Q8_0 | 3B_ffn_down | 8192x3072 | 1.649 | 15.10 | 3.065 | 8.12 | 1.86x | CPU4T |
| Q8_0 | 3B_lm_head_32k | 3072x32000 | 5.179 | 18.78 | 9.490 | 10.25 | 1.83x | CPU4T |
| Q8_0 | 4B_hidden | 2560x2560 | 0.721 | 9.00 | 0.613 | 10.59 | 0.85x | OpenCL |
| Q8_0 | 4B_ffn_up | 2560x6912 | 2.247 | 7.79 | 1.782 | 9.83 | 0.79x | OpenCL |
| Q8_0 | 4B_ffn_down | 6912x2560 | 2.140 | 8.18 | 2.195 | 7.98 | 1.03x | CPU4T |
| Q8_0 | 4B_lm_head_32k | 2560x32000 | 5.740 | 14.12 | 8.483 | 9.56 | 1.48x | CPU4T |
| Q8_0 | 7B_hidden | 4096x4096 | 1.466 | 11.33 | 1.569 | 10.58 | 1.07x | CPU4T |
| Q8_0 | 7B_ffn_up | 4096x11008 | 2.816 | 15.85 | 4.321 | 10.33 | 1.53x | CPU4T |
| Q8_0 | 7B_ffn_down | 11008x4096 | 2.720 | 16.40 | 5.880 | 7.59 | 2.16x | CPU4T |
| Q8_0 | 7B_lm_head_32k | 4096x32000 | 6.240 | 20.79 | 14.876 | 8.72 | 2.38x | CPU4T |
| Q8_0 | 8B_hidden | 4096x4096 | 1.224 | 13.57 | 1.570 | 10.57 | 1.28x | CPU4T |
| Q8_0 | 8B_ffn_up | 4096x14336 | 3.078 | 18.88 | 5.746 | 10.11 | 1.87x | CPU4T |
| Q8_0 | 8B_ffn_down | 14336x4096 | 2.770 | 20.98 | 8.088 | 7.18 | 2.92x | CPU4T |
| Q8_0 | 8B_lm_head_32k | 4096x32000 | 6.306 | 20.57 | 14.633 | 8.86 | 2.32x | CPU4T |
| Q5_K | 1B_hidden | 2048x2048 | 1.020 | 2.63 | 0.314 | 8.56 | 0.31x | OpenCL |
| Q5_K | 1B_ffn_up | 2048x5632 | 2.299 | 3.21 | 0.918 | 8.05 | 0.40x | OpenCL |
| Q5_K | 1B_ffn_down | 5632x2048 | 3.462 | 2.13 | 0.924 | 7.99 | 0.27x | OpenCL |
| Q5_K | 1B_lm_head_32k | 2048x32000 | 9.633 | 4.36 | 4.117 | 10.19 | 0.43x | OpenCL |
| Q5_K | 3B_hidden | 3072x3072 | 2.409 | 2.51 | 0.720 | 8.39 | 0.30x | OpenCL |
| Q5_K | 3B_ffn_up | 3072x8192 | 5.096 | 3.16 | 1.971 | 8.17 | 0.39x | OpenCL |
| Q5_K | 3B_ffn_down | 8192x3072 | 5.255 | 3.07 | 2.237 | 7.20 | 0.43x | OpenCL |
| Q5_K | 3B_lm_head_32k | 3072x32000 | 9.534 | 6.60 | 6.297 | 10.00 | 0.66x | OpenCL |
| Q5_K | 4B_hidden | 2560x2560 | 1.625 | 2.58 | 0.456 | 9.19 | 0.28x | OpenCL |
| Q5_K | 4B_ffn_up | 2560x6912 | 2.021 | 5.61 | 1.152 | 9.84 | 0.57x | OpenCL |
| Q5_K | 4B_ffn_down | 6912x2560 | 2.964 | 3.82 | 2.363 | 4.79 | 0.80x | OpenCL |
| Q5_K | 4B_lm_head_32k | 2560x32000 | 11.982 | 4.38 | 5.155 | 10.18 | 0.43x | OpenCL |
| Q5_K | 7B_hidden | 4096x4096 | 4.531 | 2.37 | 1.336 | 8.04 | 0.29x | OpenCL |
| Q5_K | 7B_ffn_up | 4096x11008 | 4.300 | 6.71 | 2.922 | 9.88 | 0.68x | OpenCL |
| Q5_K | 7B_ffn_down | 11008x4096 | 4.280 | 6.74 | 4.296 | 6.72 | 1.00x | CPU4T |
| Q5_K | 7B_lm_head_32k | 4096x32000 | 13.647 | 6.15 | 8.611 | 9.75 | 0.63x | OpenCL |
| Q5_K | 8B_hidden | 4096x4096 | 4.751 | 2.26 | 1.340 | 8.02 | 0.28x | OpenCL |
| Q5_K | 8B_ffn_up | 4096x14336 | 7.000 | 5.37 | 4.573 | 8.22 | 0.65x | OpenCL |
| Q5_K | 8B_ffn_down | 14336x4096 | 10.784 | 3.49 | 5.628 | 6.68 | 0.52x | OpenCL |
| Q5_K | 8B_lm_head_32k | 4096x32000 | 11.809 | 7.11 | 8.614 | 9.74 | 0.73x | OpenCL |
| Q6_K | 1B_hidden | 2048x2048 | 2.402 | 1.33 | 0.405 | 7.91 | 0.17x | OpenCL |
| Q6_K | 1B_ffn_up | 2048x5632 | 3.272 | 2.69 | 1.180 | 7.47 | 0.36x | OpenCL |
| Q6_K | 1B_ffn_down | 5632x2048 | 6.028 | 1.46 | 1.338 | 6.59 | 0.22x | OpenCL |
| Q6_K | 1B_lm_head_32k | 2048x32000 | 11.756 | 4.26 | 5.335 | 9.39 | 0.45x | OpenCL |
| Q6_K | 3B_hidden | 3072x3072 | 2.621 | 2.75 | 0.944 | 7.64 | 0.36x | OpenCL |
| Q6_K | 3B_ffn_up | 3072x8192 | 4.625 | 4.16 | 2.635 | 7.30 | 0.57x | OpenCL |
| Q6_K | 3B_ffn_down | 8192x3072 | 4.634 | 4.15 | 3.230 | 5.95 | 0.70x | OpenCL |
| Q6_K | 3B_lm_head_32k | 3072x32000 | 17.573 | 4.27 | 8.075 | 9.30 | 0.46x | OpenCL |
| Q6_K | 4B_hidden | 2560x2560 | 3.862 | 1.30 | 0.582 | 8.60 | 0.15x | OpenCL |
| Q6_K | 4B_ffn_up | 2560x6912 | 3.366 | 4.02 | 1.472 | 9.18 | 0.44x | OpenCL |
| Q6_K | 4B_ffn_down | 6912x2560 | 3.510 | 3.85 | 2.829 | 4.78 | 0.81x | OpenCL |
| Q6_K | 4B_lm_head_32k | 2560x32000 | 14.322 | 4.37 | 6.777 | 9.23 | 0.47x | OpenCL |
| Q6_K | 7B_hidden | 4096x4096 | 3.156 | 4.06 | 1.864 | 6.88 | 0.59x | OpenCL |
| Q6_K | 7B_ffn_up | 4096x11008 | 7.942 | 4.34 | 3.748 | 9.19 | 0.47x | OpenCL |
| Q6_K | 7B_ffn_down | 11008x4096 | 7.868 | 4.38 | 6.657 | 5.17 | 0.85x | OpenCL |
| Q6_K | 7B_lm_head_32k | 4096x32000 | 22.688 | 4.41 | 11.446 | 8.75 | 0.50x | OpenCL |
| Q6_K | 8B_hidden | 4096x4096 | 3.179 | 4.03 | 1.858 | 6.90 | 0.58x | OpenCL |
| Q6_K | 8B_ffn_up | 4096x14336 | 10.224 | 4.39 | 6.349 | 7.07 | 0.62x | OpenCL |
| Q6_K | 8B_ffn_down | 14336x4096 | 10.173 | 4.41 | 8.906 | 5.04 | 0.88x | OpenCL |
| Q6_K | 8B_lm_head_32k | 4096x32000 | 26.811 | 3.73 | 11.444 | 8.75 | 0.43x | OpenCL |

结论:

1. Q6_K 是最稳定的 OpenCL 胜场: 20/20 都比 CPU4T 快, 主要因为 CPU bit/scale 重排成本高。
2. Q5_K 也基本是 OpenCL 更快, 只有 `7B_ffn_down 11008x4096` 与 CPU4T 几乎打平。
3. Q8_0 更偏向 CPU4T: 大多数 7B/8B shape CPU4T 显著更快, 特别是 `ffn_down` 和 `lm_head`。
4. Q4_0 是 shape 相关: 小 hidden/ffn_up 更容易 OpenCL 胜, 大 vocab/lm_head 和部分 down-proj 更适合 CPU4T。
5. 因此如果 load-time transform 要按 type/shape 选择路径, 一个保守策略是: Q5_K/Q6_K 走 OpenCL; Q8_0 默认 CPU4T; Q4_0 对 `lm_head` 和大 down-proj 走 CPU4T, 其余可走 OpenCL。
