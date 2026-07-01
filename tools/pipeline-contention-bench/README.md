# Pipeline Contention Bench 代码说明

`pipeline-contention-bench.cpp` 是一个用于研究 elastic memory pipeline
资源冲突的 microbenchmark。它不跑完整 llama graph，而是把运行时里几个关键阶段拆成
可控 stage，分别测：

```text
单个 stage 的耗时
两个 stage 同时启动时能不能 overlap
多个 item 通过 pipeline 时是否真的提升吞吐
```

这个工具主要服务于 planner / runtime 的资源建模：

```text
哪些 stage 可以当成独立 resource
哪些 stage 需要加 overlap penalty
哪些 stage 应该当成同一个 scarce resource
```

当前主要目标设备是手机端 Android / OP12 / Adreno OpenCL。本机运行只适合检查编译，
不应该把本机 OpenCL 数字混进手机结论。

## 代码结构

`pipeline-contention-bench.cpp` 大致分成这些部分：

| 代码块 | 作用 |
|---|---|
| `params`, `parse`, `usage` | 命令行参数 |
| `cl_env` | OpenCL platform / device / context / queue / program |
| `cl_buffer`, `cl_kernel_wrap` | OpenCL 资源的 RAII 包装 |
| `disk_load_stage` | `O_DIRECT pread`，模拟 disk -> CPU staging |
| `disk_async_prefetch_stage` | 后台 IO thread + ring buffer，模拟提前 prefetch |
| `cpu_mem_load_stage` | 不走文件，只做 CPU memory read/write baseline |
| `cpu_xform_stage` | CPU 上做 q4_0 AOS -> SOA / q,d split |
| `cpu_compute_stage` | CPU FMA loop，制造 CPU compute 压力 |
| `gpu_compute_stage` | OpenCL FMA kernel，制造 GPU compute 窗口 |
| `gpu_write_stage` | 一个 host buffer 写到 GPU |
| `gpu_write_raw_stage` | raw q4_0 buffer 写到 GPU |
| `gpu_write_two_stage` | q/d 两个 buffer 写到 GPU |
| `gpu_xform_stage` | full GPU prepare：write + convert + transpose + copy-back |
| `gpu_xform_kernels_stage` | GPU transform kernels only |
| `gpu_convert_only_stage` | raw q4_0 已在 GPU 上，计时只跑 convert kernel |
| `gpu_transpose_only_stage` | q/d 已在 GPU 上，计时只跑 transpose kernels |
| `bench_single`, `bench_overlap` | 单 stage 和 pairwise overlap 测量 |
| `run_pipeline_bench` | 多 item pipeline 测量 |
| `main` | 构造 stage、执行实验、打印结果 |

## Stage 语义

### `disk_load`

对应：

```text
GGUF file / disk region -> CPU staging buffer
```

实现是 `disk_load_stage`：

```text
open(file, O_DIRECT)
aligned buffer
random aligned pread
```

它用来近似真实运行时里的 foreground/background direct read。pipeline 模式下，
读入的 raw bytes 会进入后续 `cpu_xform` stage。

### `disk_async_prefetch`

对应：

```text
GGUF file / disk region -> background prefetch ring -> CPU staging buffer
```

实现是 `disk_async_prefetch_stage`：

```text
open(file, O_DIRECT)
N 个 aligned ring buffers
后台 thread 持续 pread 填满 ring
前台 stage 只消费一个 prefetched buffer
```

它用来区分同步前台 `pread` 的调用成本，和真实 runtime 里提前 prefetch 能隐藏多少
load 时间。

### `cpu_mem_load`

对应：

```text
CPU memory buffer -> CPU memory buffer
```

它不读文件，只做 `memcpy` + touch，用来测一个纯 DRAM read/write 型 load 和
`cpu_xform` 是否抢 LPDDR / cache / memory controller。

### `cpu_xform`

对应：

```text
CPU prepare
raw q4_0 block -> q buffer + d buffer
```

q4_0 block 在 benchmark 里抽象成：

```text
uint16 d
uint8 qs[16]
```

`cpu_xform_stage` 做的事是：

```text
读取 raw block
拷贝 d scale
对 qs 做 nibble reorder
生成 q/d 两个 CPU buffer
```

注意：这不是 OpenCL kernel。它是 CPU loop，用来模拟 runtime 里的 CPU-side
pretransform / repack 路径。

### `cpu_compute`

对应：

```text
CPU backend compute pressure
```

它不是某个具体 GGML op，而是一个 float working set 上的 FMA loop。它的作用是制造
可调的 CPU core / cache / LPDDR 压力，用来观察 `disk_load` 或 `cpu_xform` 是否会和
CPU compute 抢资源。

### `gpu_compute`

对应：

```text
GPU decode compute window
```

它是一个 OpenCL FMA kernel，运行在单独的 `compute_q` queue 上。它不模拟某个精确的
matmul kernel，而是提供一个可调长度的 GPU critical section，观察 prepare stage 能否
藏在这个窗口里。

### `gpu_write`

对应：

```text
generic host -> GPU upload
```

它只做一个 `clEnqueueWriteBuffer + clFinish`。

### `gpu_write_raw`

对应：

```text
raw q4_0 payload upload
```

大小按当前 shape 算：

```text
raw_bytes = K * M / 32 * 18
```

### `gpu_write_two`

对应：

```text
CPU pretransform 后的 q/d upload
```

大小按当前 shape 算：

```text
q_bytes = K * M / 32 * 16
d_bytes = K * M / 32 * 2
```

这条路径最接近：

```text
CPU 已经完成 q/d split
然后把 q 和 d 分别写到 GPU SOA buffer
```

### `gpu_xform`

对应：

```text
完整 GPU-side prepare
```

它包括：

```text
host raw q4_0 -> GPU raw buffer
kernel_convert_block_q4_0_noshuffle
kernel_transpose_16_buf_tiled for q
copy tmp_q back
kernel_transpose_16_buf_tiled for d
copy tmp_d back
clFinish
```

这个 stage 用来测完整 GPU materialization 和 GPU compute 的冲突。

### `gpu_xform_kernels`

对应：

```text
GPU transform kernels only
```

构造时先把 raw q4_0 buffer 写到 GPU；计时阶段只跑：

```text
convert kernel
transpose q kernel
transpose d kernel
clFinish
```

它排除了 host upload 和 copy-back，用来单独看 GPU transform kernel 本身是否会和
`gpu_compute` 抢 Adreno / OpenCL queue / GPU memory path。

### `gpu_convert_only`

对应：

```text
raw q4_0 GPU buffer -> q/d GPU buffers
```

构造阶段先把 raw q4_0 写到 GPU；计时阶段只跑：

```text
kernel_convert_block_q4_0_noshuffle
clFinish
```

这个 stage 用来回答 convert kernel 本身能不能和 `gpu_compute` overlap。

### `gpu_transpose_only`

对应：

```text
q/d GPU buffers -> transposed q/d GPU buffers
```

构造阶段先把 q/d 写到 GPU；计时阶段只跑：

```text
transpose q kernel
transpose d kernel
clFinish
```

这个 stage 用来回答 transpose kernel 本身能不能和 `gpu_compute` overlap。

## Pairwise Overlap 指标

每个 stage 先单独测：

```text
single_A
single_B
```

然后两个 stage 同时启动，得到：

```text
overlap
```

输出里的指标：

```text
ideal    = max(single_A, single_B)
sum      = single_A + single_B

competition       = overlap / ideal
speedup_vs_serial = sum / overlap
extra_delay       = overlap - ideal
```

解读方式：

| 指标现象 | 含义 |
|---|---|
| `competition ~= 1` | 接近理想 overlap |
| `competition > 1` | 有资源竞争、排队、driver 串行化或内存带宽压力 |
| `speedup_vs_serial > 1` | 相比串行有收益 |
| `speedup_vs_serial ~= 1` | 基本等价串行 |
| `extra_delay` 很小 | 即使 speedup 不大，也可能适合提前调度 |

注意：小 stage 和长 compute overlap 时，`speedup_vs_serial` 可能看起来很小，因为
stage 本身只占 compute window 的一小段。这种情况下应同时看 `extra_delay`。

## Full Pipeline 模式

full pipeline 把多个 item 通过四个 worker thread：

```text
disk_load -> cpu_xform -> gpu_xform -> gpu_compute
```

每个 item 占一个 ring slot。`pipeline_slots` 控制允许多少 item 同时在 pipeline 中。

常见解读：

| 现象 | 含义 |
|---|---|
| `slots=1` | 近似串行 baseline |
| `slots=2/4` 明显快 | pipeline 有实际吞吐收益 |
| `slots=8` 不再变快 | lead 太深没有额外收益 |
| `gpu_xform busy/item` 暴涨 | GPU prepare 和 GPU compute 强冲突 |

pipeline 输出中的 per-stage counter 很重要：

```text
busy      该 stage 真正执行/等待底层设备完成的时间
wait      等上游/下游 slot 的时间
busy/item 每个 item 平均 stage busy 时间
```

如果 `gpu_xform` standalone 只有几毫秒，但 pipeline 里 `busy/item` 变成几百毫秒，
说明它不是独立在跑，而是在 GPU queue / driver / compute path 上和 `gpu_compute`
互相挤压。

## Pipeline GPU Mode

`run_pipeline_bench` 支持两种 GPU stage 模式：

```text
full
kernels-only
```

### `full`

默认模式。GPU stage 包含：

```text
write q buffer
write d buffer
transpose q kernel
copy q back
transpose d kernel
copy d back
clFinish
```

它模拟完整的 GPU-side materialization。

### `kernels-only`

构造阶段预加载 q/d GPU buffer；计时阶段只跑：

```text
transpose q kernel
transpose d kernel
clFinish
```

它用来隔离：

```text
GPU transform kernel 本身
```

也就是说，如果 `kernels-only` 仍然和 `gpu_compute` 冲突，那么问题不只是 upload 或
copy-back，而是 transform kernel 本身也占用了同一套 GPU/driver 资源。

OP12 上的结果正是这样：FFN shape 下 standalone `gpu_xform_kernels` 大约 8 ms，但
pipeline 中和 `gpu_compute` 并发后，`gpu_xform busy/item` 会膨胀到百毫秒级。

## 常见 LLM Shape

q4_0 近似大小：

```text
bytes = K * M / 32 * 18
MiB   = bytes / 1024 / 1024
```

其中 `K` 必须是 32 的倍数。下面这些 shape 可以作为更贴近 LLM 的 benchmark preset。

### Attention Projection

attention 的 q/k/v/o projection 常见是 hidden 到 hidden：

| 模型规模/hidden | shape | q4_0 大小 | 说明 |
|---|---:|---:|---|
| 1B-ish | `2048 x 2048` | ~2.25 MiB | 小模型 attention proj |
| 3B-ish | `3072 x 3072` | ~5.06 MiB | 中小模型 attention proj |
| 7B/8B-ish | `4096 x 4096` | ~9.00 MiB | 常见 7B/8B attention proj |
| 13B-ish | `5120 x 5120` | ~14.06 MiB | 更大 hidden projection |
| 70B-ish | `8192 x 8192` | ~36.00 MiB | 大模型 attention proj |

### FFN Projection

FFN 的 gate/up/down 通常更大，是 low-memory streaming 下更值得关注的 tensor。

| 模型规模/hidden | shape | q4_0 大小 | 说明 |
|---|---:|---:|---|
| 1B-ish | `2048 x 8192` | ~9.00 MiB | 小模型 FFN |
| 3B-ish | `3072 x 8192` | ~13.50 MiB | 中小模型 FFN |
| 7B Llama2-ish | `4096 x 11008` | ~24.19 MiB | 传统 7B FFN |
| 8B Llama3-ish | `4096 x 14336` | ~31.50 MiB | 当前 OP12 主测 FFN shape |
| 13B-ish | `5120 x 13824` | ~37.97 MiB | 更大 FFN |
| 70B-ish | `8192 x 28672` | ~126.00 MiB | 大模型 FFN，手机上可能过重 |

### MoE / Expert Projection

MoE 模型里单个 expert 的矩阵可能比 dense FFN 小，但每 token 会选多个 expert。可以考虑：

| 场景 | shape | q4_0 大小 | 说明 |
|---|---:|---:|---|
| small expert | `2048 x 4096` | ~4.50 MiB | 小 expert gate/up/down |
| medium expert | `3072 x 8192` | ~13.50 MiB | 中等 expert |
| large expert | `4096 x 11008` | ~24.19 MiB | 接近 7B dense FFN |

### Embedding / LM Head

embedding / lm_head 通常 shape 很大，但 decode 中访问模式和 projection 不完全一样。
如果要测，可用这些作为压力点：

| 场景 | shape | q4_0 大小 | 说明 |
|---|---:|---:|---|
| vocab 32k, hidden 4096 | `32000 x 4096` | ~70.31 MiB | 常见 vocab projection |
| vocab 128k, hidden 4096 | `128000 x 4096` | ~281.25 MiB | 大词表，手机上很重 |

这类 shape 更适合做极限 memory / IO 压力测试，不一定适合作为普通 per-layer pipeline
case。

## 推荐 Shape 组合

如果只想跑一组最有代表性的手机实验：

```text
attention:
  K=4096
  M=4096
  cpu_xform ~= 9 MiB
  load ~= 16 MiB

ffn:
  K=4096
  M=14336
  cpu_xform ~= 32 MiB
  load ~= 32 MiB
```

如果要覆盖 size sweep：

```text
small:
  2048 x 2048

medium:
  3072 x 8192

main:
  4096 x 4096
  4096 x 14336

large:
  5120 x 13824
  8192 x 8192
```

## 和 Planner 的关系

这个 benchmark 的结果应转成 planner 里的 resource policy：

```text
cpu_transform vs gpu_compute:
  best pipeline pair.  OP12 5-run average reaches about 96% of ideal overlap.

gpu_transform vs cpu_compute:
  good pipeline pair.  OP12 5-run average reaches about 98% of ideal overlap.

cpu_transform vs cpu_compute:
  same CPU-side resource class.  OP12 5-run average reaches only about 66% of
  ideal overlap, so charge a real CPU/cache/LPDDR contention penalty.

gpu_transform vs gpu_compute:
  shared GPU/driver/memory-path resource class.  OP12 5-run average reaches
  only about 52% of ideal overlap, so avoid this pair on the critical path.

async disk prefetch vs cpu_transform:
  can overlap if load has enough lead/ring depth.

foreground synchronous disk_load vs cpu_transform:
  avoid on the critical path; synchronous pread can serialize with transform.
```

核心结论：

```text
Cross-backend pairs are pipeline-friendly:
  cpu_transform + gpu_compute
  gpu_transform + cpu_compute

Same-backend pairs are contention-heavy:
  cpu_transform + cpu_compute
  gpu_transform + gpu_compute

Load should be modeled as async prefetch with lead, not foreground synchronous
pread inside the transform critical window.
```

## OP12 Balanced Pair 结论

为了避免一个 stage 特别短、另一个 stage 特别长导致结论被稀释，OP12 上还需要看
time-balanced pair。这里的 compute rounds 被调低，让 `transform` 和 `compute` 落在同一
量级。

这里主表不用 `speedup_vs_serial`，而用理论达成率：

```text
ideal_achieved = min(ideal / overlap, 1.0)
```

`ideal` 是两个 stage 完美 overlap 时的理论 wall time，也就是
`max(single_A, single_B)`。如果实测 `overlap < ideal`，通常是调度/频率噪声，不解释成
超过理论，只记为约 `100%`。

### `4096 x 4096`, 5-run average

| pair | single A avg | single B avg | overlap avg | ideal avg | 理论达成率 avg | 结论 |
|---|---:|---:|---:|---:|---:|---|
| `cpu_transform + cpu_compute` | 3.136 ms | 2.932 ms | 4.760 ms | 3.136 ms | 66% | CPU 侧竞争明显 |
| `cpu_transform + gpu_compute` | 3.136 ms | 6.053 ms | 6.223 ms | 6.053 ms | 97% | overlap 很好 |
| `gpu_transform + cpu_compute` | 5.259 ms | 2.932 ms | 4.929 ms | 5.259 ms | ~100% | overlap 很好 |
| `gpu_transform + gpu_compute` | 5.259 ms | 6.053 ms | 11.497 ms | 6.053 ms | 53% | 明显冲突，接近串行 |

所以 planner policy 应该按 backend/resource 分级：

```text
cpu_transform + gpu_compute:
  allow and prefer.

gpu_transform + cpu_compute:
  allow and prefer.

cpu_transform + cpu_compute:
  allow only with CPU-side contention penalty.

gpu_transform + gpu_compute:
  avoid unless the chunk is very small or the GPU compute window is idle.
```

### CPU Transform vs GPU Transform

这组用来回答 CPU backend prepare 和 GPU backend prepare 能不能并发。这里的
`gpu_transform` 是 full GPU prepare，包括 host upload、convert、transpose 和
copy-back。

| shape | `cpu_transform` | `gpu_transform` | overlap | ideal | 理论达成率 |
|---|---:|---:|---:|---:|---:|
| `4096 x 4096`, 5-run avg | 3.136 ms | 5.259 ms | 5.546 ms | 5.259 ms | 95% |

解释：

```text
4096 x 4096 attention-like shape:
  CPU transform 和 GPU transform 可以很好 overlap，5-run average 达到约 95%
  theoretical ideal。
```

Planner policy:

```text
cpu_transform + gpu_transform:
  allow overlap for small/medium tensors.
  for large tensors, add memory-bandwidth penalty or limit concurrent chunk size.
```

### 完整 balanced planner matrix：`4096 x 4096`, 5-run average

| pair | single A avg | single B avg | overlap avg | ideal avg | 理论达成率 avg |
|---|---:|---:|---:|---:|---:|
| `disk_load + cpu_transform` | 3.255 ms | 3.136 ms | 5.586 ms | 3.349 ms | 60% |
| `disk_async_prefetch + cpu_transform` | 3.331 ms | 3.136 ms | 4.309 ms | 3.331 ms | 78% |
| `cpu_mem_load + cpu_transform` | 1.294 ms | 3.136 ms | 3.191 ms | 3.136 ms | 98% |
| `disk_load + gpu_compute` | 3.255 ms | 6.053 ms | 6.216 ms | 6.053 ms | 97% |
| `cpu_transform + gpu_transform` | 3.136 ms | 5.259 ms | 5.546 ms | 5.259 ms | 95% |
| `cpu_transform + cpu_compute` | 3.136 ms | 2.932 ms | 4.760 ms | 3.136 ms | 66% |
| `cpu_transform + gpu_compute` | 3.136 ms | 6.053 ms | 6.223 ms | 6.053 ms | 97% |
| `gpu_transform + cpu_compute` | 5.259 ms | 2.932 ms | 4.929 ms | 5.259 ms | ~100% |
| `cpu_compute + gpu_compute` | 2.932 ms | 6.053 ms | 6.002 ms | 6.053 ms | ~100% |
| `gpu_transform + gpu_compute` | 5.259 ms | 6.053 ms | 11.497 ms | 6.053 ms | 53% |

### 完整 balanced planner matrix：`4096 x 14336`

Standalone stage:

| stage | backend | time |
|---|---|---:|
| `disk_load` | disk -> CPU | 7.030 ms |
| `cpu_transform` | CPU | 11.343 ms |
| `gpu_transform` | GPU | 14.305 ms |
| `cpu_compute` | CPU | 14.598 ms |
| `gpu_compute` | GPU | 21.383 ms |

Pairwise overlap:

| pair | single A | single B | overlap | ideal | sum | competition | speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| `disk_load + cpu_transform` | 7.030 ms | 11.343 ms | 14.683 ms | 11.343 ms | 18.373 ms | 1.29 | 1.25 |
| `disk_load + gpu_transform` | 7.030 ms | 14.305 ms | 15.964 ms | 14.305 ms | 21.335 ms | 1.12 | 1.34 |
| `disk_load + cpu_compute` | 7.030 ms | 14.598 ms | 13.312 ms | 14.598 ms | 21.628 ms | 0.91 | 1.62 |
| `disk_load + gpu_compute` | 7.030 ms | 21.383 ms | 21.925 ms | 21.383 ms | 28.413 ms | 1.03 | 1.30 |
| `cpu_transform + gpu_transform` | 11.343 ms | 14.305 ms | 16.513 ms | 14.305 ms | 25.648 ms | 1.15 | 1.55 |
| `cpu_transform + cpu_compute` | 11.343 ms | 14.598 ms | 22.274 ms | 14.598 ms | 25.941 ms | 1.53 | 1.16 |
| `cpu_transform + gpu_compute` | 11.343 ms | 21.383 ms | 22.023 ms | 21.383 ms | 32.725 ms | 1.03 | 1.49 |
| `cpu_compute + gpu_transform` | 14.598 ms | 14.305 ms | 16.618 ms | 14.598 ms | 28.903 ms | 1.14 | 1.74 |
| `cpu_compute + gpu_compute` | 14.598 ms | 21.383 ms | 21.645 ms | 21.383 ms | 35.981 ms | 1.01 | 1.66 |
| `gpu_transform + gpu_compute` | 14.305 ms | 21.383 ms | 36.348 ms | 21.383 ms | 35.688 ms | 1.70 | 0.98 |

### Load Variants vs CPU Transform

`disk_load` 用的是同步 `O_DIRECT pread`，它测到的不是纯粹的 storage device DMA，
而是：

```text
calling thread
kernel / filesystem / block layer path
device read
write into CPU DRAM buffer
```

为了确认同步前台 `pread` 是否低估了 overlap，加入两个对照：

```text
disk_async_prefetch:
  后台 IO thread 提前把 ring buffer 填好，前台只消费 prefetched chunk。

cpu_mem_load:
  不走 storage，只做 CPU memory read/write。
```

OP12 lazy-start 结果：

#### `4096 x 4096`

| pair | single A | single B | overlap | competition | speedup |
|---|---:|---:|---:|---:|---:|
| `disk_load + cpu_transform` | 3.255 ms | 3.136 ms | 5.586 ms | 1.67 | 1.14 |
| `disk_async_prefetch + cpu_transform` | 3.331 ms | 3.136 ms | 4.309 ms | 1.29 | 1.50 |
| `cpu_mem_load + cpu_transform` | 1.294 ms | 3.136 ms | 3.191 ms | 1.02 | 1.39 |

#### `4096 x 14336`

| pair | single A | single B | overlap | competition | speedup |
|---|---:|---:|---:|---:|---:|
| `disk_load + cpu_transform` | 6.931 ms | 11.319 ms | 18.179 ms | 1.61 | 1.00 |
| `disk_async_prefetch + cpu_transform` | 6.678 ms | 11.319 ms | 11.070 ms | 0.98 | 1.63 |
| `cpu_mem_load + cpu_transform` | 2.375 ms | 11.319 ms | 11.093 ms | 0.98 | 1.23 |

解释：

```text
synchronous foreground disk_load + cpu_transform:
  overlap 差，尤其大 FFN shape 基本串行。

disk_async_prefetch + cpu_transform:
  overlap 明显更好，说明 storage prefetch 本身可以被隐藏。

cpu_mem_load + cpu_transform:
  也能 overlap，但大 shape 下仍会受到 DRAM/cache 带宽影响。
```

因此不能把同步 `disk_load + cpu_transform` 解释成：

```text
storage prefetch can never overlap CPU transform.
```

更合理的 planner policy 是：

```text
foreground synchronous load + cpu_transform:
  avoid on critical path.

async prefetch + cpu_transform:
  allow overlap, but keep enough lead/ring depth.

cpu_mem_load + cpu_transform:
  allow overlap with memory bandwidth penalty for large tensors.
```
