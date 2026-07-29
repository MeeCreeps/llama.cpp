# Elastic working-unit granularity：Motivation 实验结果

> **2026-07-29 OP12 v21 revalidation in progress.** 下文保留此前
> motivation 数据以便追溯，不删除历史表；但旧 8B dynamic-baseline matrix
> 存在 pinned-weight budget、streaming headroom 或 worker isolation 口径不一致，
> 旧 GPU v19 fixed sweep 还漏掉了 `output` 的 inside-budget pin。它们不能与
> 新的 Diff-tree 结果混用。当前最终替换实验固定在 OP12，要求
> `token_embd`/`output` 在所有模式采用完全相同的 inside-budget policy，
> LOAD/PREPARE/COPY workers 隔离、相同 physical budget、真实 pipeline、
> common-token-prefix、完整 600 秒原始 trace，并审计 mixed-frontier、
> stage demand、pipeline wait 和 transition cost。v21 的 CPU/GPU 表完成前，
> 本文既有数字只作为带 provenance 的历史 motivation/diagnostic evidence，
> 不作为最新 Diff-now 结论。

## 1. 结论

真实 Elastic model inference 路径上的 pipeline-on 实验表明：只改变 weight
working-unit granularity，就会显著改变 decode latency，而且最优颗粒度会随
memory budget、budget 变化历史和 backend 改变。2026-07-26 的最终 OP13
公平实现进一步消除了 GPU 逐 unit `clFlush` 对 Tensor 的额外惩罚，并让 Cut
实际使用双 half-tensor fused kernel。

- 最新 CPU fixed budget（OP13）：30%/40% 是 Cut 最快；50%/60% 是
  Tensor 最快；70%--90% 又是 Cut/Tensor 接近。CPU Multi-fused 是
  Multi 内优化，不是第四种 granularity。
- 最新 GPU fixed budget（OP13）：30% 是 Cut 明显最快，40%--60% 是
  Tensor/Multi 接近，70% 是 Tensor，80% 是 Multi，90% 又是 Tensor。
  三种 granularity 都在至少一个真实模型预算点上成为最优。
- CPU dynamic nodes（OP13）：在相同 instantaneous budget bin 下，最慢模式
  比最快模式高 32%--43%；dynamic 60% 的排名与 fixed 60% 不同。
- GPU/OpenCL（OP12）：大多数 budget 下 Multi/Tensor 领先或接近，但 80%
  的 token-latency p50 由 Cut 最低；dynamic 60% 则由 Tensor 领先。
- 配套 8B 原始 10 分钟 trace（OP13）：不映射 budget、不压缩时间。CPU 的
  70% bin 是 Cut 最快，80%/90% 是 Multi 最快；GPU 的 70%/80% 是 Tensor
  最快，90% 是 Multi 最快。

因此，granularity 不是只影响 microbenchmark 的实现参数，而是
dynamic-memory Elastic inference 的独立性能维度。本实验只验证 motivation，
不进行在线 granularity 优化，也不包含 CPU+GPU heterogeneous placement。

## 2. 实验边界

- Branch：`feature/elastic-working-unit-granularity`
- Model：主要 fixed/dynamic-node 实验使用 Llama-3.2-3B-Instruct Q4_0；
  原始 trace 使用 Meta-Llama-3-8B-Instruct Q4_0 及其配套 trace。
- Backend：CPU Elastic-only 和 OpenCL Elastic-only 分开测试。
- Pipeline：本文所有真实模型结果都开启 pipeline。
- Multi：两个完整 tensor 组成一个 LOAD/PREPARE/residency/retire unit。
- Tensor：一个完整 tensor 是一个 unit。
- Cut：eligible 2-D MUL_MAT weight 沿输出 rows 切成两个独立 unit。
- Multi-fused：与普通 Multi 保持相同 unit 边界，只在已有 byte window 内
  opportunistically 融合 compatible layout/kernel pair，不扩大 pipeline
  range，也不为等待 partner 增加额外阻塞。
- Fixed 表值为三个 counterbalanced 独立进程的 steady-median 之中位数。
- `Gap` 定义为 `(slowest - fastest) / fastest`。

## 2.1 2026-07-26 最终公平实现：OP13 fixed budget

本节是当前应优先用于 granularity motivation 的 fixed-budget 结果。设备为
OnePlus 13 `3C15AU002CL00000`，模型为 Llama-3.2-3B-Instruct Q4_0。
weight budget 扫描 30%--90%，每个点三个 counterbalanced 独立进程；
`n_predict=25`，丢弃前 8 个 decode tokens，对后 16 个 token 取 steady
median。CPU 和 OpenCL 均为单 backend，pipeline 开启，不包含 heterogeneous
placement。最终二进制 SHA-256 为
`e00fac541e7894cd240b4146dcf078330bfe890e7ca961788ae383f4784b87c1`。

### 2.1.1 CPU Elastic-only

单位为 ms/token。Multi、Multi-fused、Tensor 使用统一
`graph-lookahead=4`。Cut 的统一配置在 40% 出现可复现的 speculative
cross-graph churn：三轮中位数为 1900.262；将 Cut 的 graph lookahead 调为
2 后，保留 pipeline overlap；设备互斥后的干净三轮为
620.688、632.686、623.842，中位数 623.842。下表因此报告每种实现的已验证
优化配置：Cut 只在 40% 使用
graph=2，其余点使用更快的 graph=4。

| Weight budget | Multi | Multi-fused | Tensor | Cut | 最快 | Gap |
|---:|---:|---:|---:|---:|:---|---:|
| 30% | 971.293 | 970.386 | 900.792 | **752.104** | Cut | 29.1% |
| 40% | 727.915 | 765.853 | 681.017 | **623.842** | Cut | 22.8% |
| 50% | 622.470 | 624.385 | **552.041** | 560.727 | Tensor | 13.1% |
| 60% | 508.978 | 515.324 | **458.978** | 487.015 | Tensor | 12.3% |
| 70% | 408.024 | 412.068 | 361.473 | **356.602** | Cut | 15.6% |
| 80% | 290.198 | 280.818 | 241.057 | **233.548** | Cut | 24.3% |
| 90% | 159.833 | 164.513 | 140.485 | **139.797** | Cut | 17.7% |

基础 sweep 84/84 runs、Cut 40% graph=2 干净补测 3/3 runs 均正常退出，
`runtime`、device-idle、token sequence 和 budget enforcement 检查通过，
budget violation 总数为 0。Multi-fused 每个 run 实际执行 46--337 次 fused
layout pair 和 1069--1346 次 fused kernel pair，kernel error 为 0；
它相对 Multi 的收益很小且不稳定，因此 fusion 不能替代 granularity 选择。

30% 的阶段计数进一步排除了“Cut 少处理 weight”的解释。三种模式每个
decode 实际读取和准备的 weight 均约为 1.50 GiB；下表中的阶段时间是累计
work demand 除以 24 个 decode token。由于 pipeline overlap，这些列不能相加
得到端到端 latency。

| 30% mode | Runtime read (ms / MiB) | Layout preparation (ms / MiB) | Compute (ms) | Exposed pipeline wait (ms) |
|:---|---:|---:|---:|---:|
| Multi | 752.307 / 1496.487 | 614.322 / 1496.487 | 79.600 | 672.529 |
| Tensor | 551.070 / 1500.717 | 611.327 / 1500.717 | 81.435 | 620.199 |
| Cut | 649.685 / 1503.667 | 603.670 / 1503.667 | 81.479 | **469.952** |

因此这组真实 CPU pipeline 中，30% Cut 的主要收益不是减少 bytes，也不是
更低的纯 I/O demand，而是更细 unit 将暴露在关键路径上的 pipeline wait
减少了 202.6 ms/token（相对 Multi）。作为对照，90% 时 non-resident work
降到约 0.23 GiB/token，Multi/Tensor/Cut 的 pipeline wait 分别为
89.142/71.462/59.854 ms/token；Cut 与 Tensor 的总 latency 只差
0.688 ms（0.49%），应视为持平而不是显著 Cut 胜出。完整阶段表见
`cpu_fixed_30pct_pipeline_breakdown_op13.csv` 和
`cpu_fixed_90pct_pipeline_breakdown_op13.csv`。

CPU raw results：

- 统一 graph=4：
  `exp-results/Llama-3.2-3B-Instruct-Q4_0/granularity_cpu_pipeline_fixed_30_90_optimized_op13_20260726`
- Cut 40% graph=2 干净复测：
  `exp-results/Llama-3.2-3B-Instruct-Q4_0/granularity_cpu_cut40_graph2_mutex_clean_op13_20260726`

### 2.1.2 GPU/OpenCL Elastic-only

旧 GPU 实现按 working unit 执行 `clFlush`，因此 Tensor 的 flush 次数是
Multi 的两倍，会把 unit 数量差异人为放大为 driver submission 开销。40%
三轮 A/B 在去除逐-unit flush 后得到：Multi 306.580、Tensor 297.542
ms/token；三轮输出一致。最终 GPU sweep 因此使用
`GGML_ELASTIC_GPU_UNIT_SYNC=none`，graph completion 仍保留必要同步。

| Weight budget | Multi | Tensor | Cut-fused | 最快 | Gap |
|---:|---:|---:|---:|:---|---:|
| 30% | 557.781 | 558.172 | **388.501** | Cut | 43.7% |
| 40% | 304.969 | **299.160** | 347.439 | Tensor | 16.1% |
| 50% | 256.651 | **254.057** | 288.978 | Tensor | 13.7% |
| 60% | 208.542 | **206.505** | 236.916 | Tensor | 14.7% |
| 70% | 167.131 | **162.004** | 180.279 | Tensor | 11.3% |
| 80% | **124.983** | 138.047 | 139.339 | Multi | 11.5% |
| 90% | 106.025 | **85.170** | 92.333 | Tensor | 24.5% |

三种 granularity 共 63/63 runs 正常退出并通过 runtime、device mutex、
token sequence 和 budget enforcement 检查，budget violation 总数为 0。
每个 Cut run 有 4825 个 dual candidates，其中 4635 次实际执行一个 dispatch
计算两个独立 half-tensor，190 次 shape fallback，budget fallback 和 queue
error 均为 0；21 个 Cut runs 合计执行 97335 次 dual dispatch。

这组结果给出了最直接的 crossover：

- 30% 时，Cut 的 median direct-read 为 3843.0 MiB，而 Multi/Tensor 均为
  6308.8 MiB。相同 memory budget 下，half-tensor residency 减少
  budget-fitting waste 和 reload bytes；该收益覆盖了更多 units 的开销。
- 40%--70% 时，Cut 虽然通常仍少读一部分 bytes，但约两倍的 Cut units 和
  dispatch/bookkeeping 开销开始占主导；Tensor 是更好的折中。
- 80% 时，Multi 以 2425 个 units 对比 Tensor 的 4850 和 Cut 的 9675，
  较少的 unit/dispatch 开销使其领先约 9.5%。
- 90% 时出现离散 residency threshold：Tensor 的 median direct-read 为
  1334.2 MiB，低于 Multi 的 1433.8 MiB；Tensor 的 pipeline wait 也只有
  624.0 ms，低于 Cut 的 664.4 ms 和 Multi 的 750.2 ms，因此 Tensor 再次
  领先。真实模型上不能用“高 budget 永远选 Multi”的单阈值规则。

GPU raw results：

- 最终 fixed sweep：
  `exp-results/Llama-3.2-3B-Instruct-Q4_0/granularity_gpu_pipeline_none_fixed_30_90_op13_20260726`
- unit-sync A/B：
  `exp-results/Llama-3.2-3B-Instruct-Q4_0/granularity_gpu_sync_none_ab_40_op13_20260726`

### 2.1.3 实现与实验有效性修复

- pin policy 原来使用 substring matching，`norm` 中的 `o` 会意外匹配并 pin
  所有 `attn_output`。现改为逗号分隔 token 的 exact matching；30% GPU
  Multi/Tensor 的 budget violation 均降为 0。
- runner 在 normal residency 和 pipeline staging 后都重新执行
  `evict_to_current_budget()`，并要求所有 budget samples 零 violation。
- GPU Tensor 去除逐-unit `clFlush`；该同步不是 graph correctness barrier。
- Cut fused kernel 在同一个 OpenCL dispatch 中读取两个独立 resident
  half-tensor layout，共享 activation，并写入各自 output row range。
- runner 新增按设备序列号的 host advisory mutex；pre-run idle gate 不再把
  使用相同 binary/model 的另一个 runner 误认为自身。早期并发运行目录
  `granularity_gpu_pipeline_fixed_30_90_optimized_op13_20260726` 和中断的
  `granularity_gpu_multi_tensor_fixed_30_90_op13_20260726`，以及与旧 GPU
  runner 重叠的 `granularity_cpu_cut_graph2_fixed_30_90_op13_20260726`
  已判为受污染，不用于任何结论。

## 3. CPU fixed budget：OP13，Llama-3.2-3B

单位为 ms/token；四种模式共 84/84 runs 通过 runtime、device-idle 和 token
correctness 检查。

| Weight budget | Multi | Multi-fused | Tensor | Cut | 最快 | Gap |
|---:|---:|---:|---:|---:|:---|---:|
| 30% | 1277.632 | 1279.306 | 981.703 | **959.453** | Cut | 33.34% |
| 40% | 962.635 | 963.981 | 744.371 | **695.743** | Cut | 38.55% |
| 50% | 695.564 | 698.932 | **514.264** | 593.851 | Tensor | 35.91% |
| 60% | 555.330 | 570.171 | **431.784** | 520.538 | Tensor | 32.05% |
| 70% | 428.006 | 436.854 | **339.024** | 426.807 | Tensor | 28.86% |
| 80% | 318.426 | 322.701 | **246.221** | **246.231** | Tensor/Cut | 31.06% |
| 90% | 197.674 | 200.144 | 155.983 | **149.909** | Cut | 33.51% |

30% Cut 的三轮值为 2133.307、888.668、959.453 ms/token，存在一次明显长尾；
表中按统一的 median-of-three 规则报告 959.453。这个点可以用于展示
granularity 差异，但作方差或尾延迟结论时需要更多重复。

Multi-fused 相对普通 Multi 的差异为 0.1%--2.7%，没有再出现旧实现因扩大
pipeline range 而大幅变慢的问题。融合本身不保证加速；这里的重要公平性条件是
两者 unit 边界相同且 fused path 确实执行。

Raw results：
`exp-results/Llama-3.2-3B-Instruct-Q4_0/granularity_cpu_pipeline_final_fixed_30_90_op13_20260723`

## 4. CPU dynamic nodes：OP13，Llama-3.2-3B

同一进程内执行 `90% -> 30% -> 60% -> 90%`，按实际 budget bin 汇总。
括号内为该 bin 的 decode-token 数。

| Weight budget | Multi | Multi-fused | Tensor | Cut | 最快 | Gap |
|---:|---:|---:|---:|---:|:---|---:|
| 30% | 1181.231 (4) | 1209.047 (4) | 921.753 (5) | **891.961 (6)** | Cut | 35.55% |
| 60% | 567.217 (9) | 568.160 (9) | 423.565 (12) | **397.900 (11)** | Cut | 42.79% |
| 90% | 200.061 (36) | 196.664 (36) | 154.862 (43) | **151.006 (43)** | Cut | 32.48% |

Fixed 60% 是 Tensor 最快，而动态切换后的 60% 是 Cut 最快。这说明性能不仅依赖
当前 budget，还受到切换过程中的 residency、reload 和 pipeline history 影响。
30%/60% plateau 的 token 数较少，因此它们是动态节点 evidence，不替代完整
trace 的多轮统计。

Raw results：
`exp-results/Llama-3.2-3B-Instruct-Q4_0/granularity_cpu_pipeline_final_dynamic_points_op13_20260723`

## 5. 原始 real trace：OP13，Meta-Llama-3-8B

### 5.1 先前的 74 秒 CPU 窗口

使用已有 10 分钟 model-matched trace01 的原始 176--250 秒窗口。窗口内
absolute total budget 为 3752.1--4746.2 MiB，直接传给 runtime，没有线性映射
或时间压缩。当前每种模式完成一个 74 秒窗口进程，因此本节是 real-trace case
study，证据强度低于三重复 fixed sweep。

| Scope | Multi | Multi-fused | Tensor | Cut | 最快 |
|:---|---:|---:|---:|---:|:---|
| Whole window | 779.695 (96) | 798.144 (86) | 754.323 (100) | **672.568 (117)** | Cut |
| Effective 70% bin | 1240.996 (4) | 1665.536 (3) | 1139.617 (5) | **975.789 (6)** | Cut |
| Effective 80% bin | 817.911 (61) | 860.038 (57) | 778.765 (65) | **699.818 (75)** | Cut |
| Effective 90% bin | 459.491 (27) | 557.140 (22) | 410.362 (26) | **370.148 (32)** | Cut |

表值为 p50 ms/token，括号内为 token 数。四种模式均正常退出并通过 runtime、
device-idle、Awake 和 common-token-prefix 校验。Multi-fused 实际执行了 337
次 fused layout-pair 和 2420 次 fused kernel-pair，kernel error 为 0，因此
不是 no-op fused mode。

Raw results：
`exp-results/Meta-Llama-3-8B-Instruct-Q4_0/granularity_cpu_pipeline_final_native_trace01_176_250_la128_op13_20260723`

### 5.2 完整 10 分钟 CPU/GPU trace

使用同一条 model-matched trace01 的原始 0--600 秒。absolute total budget
为 3752.1--6198.6 MiB，原值复制到 runtime；没有 budget rescaling，也没有
时间压缩。扣除 512 MiB KV 和 256 MiB misc 后，这对应约 67% weight
residency 到 full-residency 以上，因此原生数据只覆盖有效 70%/80%/90% bins，
不能把它表述为 30%--90% trace。

CPU 整段结果：

| Mode | p50 (ms/token) | p95 (ms/token) | Tokens/s | Decode tokens |
|:---|---:|---:|---:|---:|
| Multi | 215.212 | **851.738** | **2.893** | 1735 |
| Multi-fused | **214.142** | 1092.409 | 2.560 | 1536 |
| Tensor | 232.795 | 1021.368 | 2.602 | 1561 |
| Cut | 338.378 | 1888.977 | 2.113 | 1268 |

CPU 按 instantaneous effective weight-ratio bin 的 p50：

| Weight-ratio bin | Multi | Multi-fused | Tensor | Cut | 最快 |
|---:|---:|---:|---:|---:|:---|
| 70% | 1047.130 (16) | 1302.283 (13) | 1134.290 (17) | **1027.729 (19)** | Cut |
| 80% | **809.200 (95)** | 1066.057 (75) | 879.042 (85) | 851.444 (90) | Multi |
| 90% | **202.009 (1611)** | 213.753 (1435) | 231.655 (1446) | 278.194 (1146) | Multi |

GPU/OpenCL 整段结果：

| Mode | p50 (ms/token) | p95 (ms/token) | Tokens/s | Decode tokens |
|:---|---:|---:|---:|---:|
| Multi | **188.019** | 845.334 | 3.333 | 1995 |
| Tensor | 188.859 | **830.806** | **3.375** | 2020 |
| Cut | 199.751 | 867.687 | 3.248 | 1944 |

GPU/OpenCL 按 instantaneous effective weight-ratio bin 的 p50：

| Weight-ratio bin | Multi | Tensor | Cut | 最快 |
|---:|---:|---:|---:|:---|
| 70% | 1041.125 (18) | **1009.685 (18)** | 1108.171 (17) | Tensor |
| 80% | 880.414 (90) | **868.321 (92)** | 919.526 (88) | Tensor |
| 90% | **187.690 (1874)** | 188.287 (1897) | 199.440 (1826) | Multi |

表中括号为该 bin 的 decode-token 数。所有 CPU 4/4 和 GPU 3/3 runs 均覆盖
至少 600 秒，正常退出，通过 device-idle、Awake、runtime、pipeline 和
common-token-prefix 检查。GPU 每种模式均满足 unit-boundary count 等于
`clFlush` count、`clFinish=0`、sync error 为 0，并实际复用了 SOA pool。
CPU Multi-fused 执行 1517 次 layout-pair fusion 和 47567 次 kernel-pair
fusion，kernel error 为 0；其较高 p95 说明 fusion 是正交优化，不保证在动态
trace 中胜过普通 Multi。

70% bin 只有 13--19 个 token，因此 crossover 是完整 trace 中的 case-study
evidence，不能替代多重复统计。90% bin 占 trace 的大部分，整段 throughput
主要由高 budget 区间决定。相较 74 秒窗口中 Cut 在所有 bins 最快，完整
10 分钟结果出现 CPU `Cut -> Multi` 和 GPU `Tensor -> Multi` 的预算相关切换，
更直接支持“固定 granularity 不能覆盖所有 dynamic budget 状态”的 motivation。

Raw results：

- CPU：
  `exp-results/Meta-Llama-3-8B-Instruct-Q4_0/granularity_cpu_pipeline_final_native_trace01_full10min_op13_20260723`
- GPU：
  `exp-results/Meta-Llama-3-8B-Instruct-Q4_0/granularity_gpu_pipeline_final_native_trace01_full10min_op13_20260723`

### 5.3 原始 10 分钟 trace：placement + mixed-granularity plan

> **Historical diagnostic table; pending OP12 v21 replacement.** 本节保留
> 2026-07-28 的 OP13 数字以追踪此前实现。它不是当前七 baseline 的最终结果，
> 不能用来宣称 v21 Diff-now 的性能或排序。新的结果必须通过本文顶部所列的
> inside-budget pin、streaming reserve、worker affinity 和 plan-residency
> contract 后才会替换本节。

本节比较完整 Elastic plan，而不是上一节的全局固定
Multi/Tensor/Cut。模型为 `Meta-Llama-3-8B-Instruct.Q4_0.gguf`，设备为
OP13；CPU Elastic-only 与 GPU/OpenCL Elastic-only 分开运行，不包含
heterogeneous placement。两种 backend 均使用原始 0--600 秒 absolute
budget trace、相同 `token_embd,output` 保留策略和
`I/O -> Layout -> Compute` working-unit pipeline。

CPU：

| Method | ms/token | Direct read MiB/fwd | Direct read ms/fwd | Pipeline wait (s) | Tokens |
|:--|--:|--:|--:|--:|--:|
| Static-Min | 1676.287 | 2363.428 | 941.052 | 153.708 | 359 |
| Static-Max | **296.691** | 0.000 | 0.000 | 0.000 | 2022 |
| MRU | 427.586 | 261.831 | 101.547 | 104.087 | 1400 |
| Offline | 542.121 | 344.240 | 139.241 | 59.746 | 1107 |
| Online | 575.259 | 376.915 | 151.405 | 63.414 | 1044 |
| Diff-before | 638.514 | 384.860 | 165.474 | 76.303 | 941 |
| Diff-now | **575.298** | 383.694 | 154.313 | 75.217 | 1044 |

CPU Diff-now 比 Diff-before 快 9.9%，但与固定 Tensor 的 Online
基本相同（差异小于 0.01%）。这说明当前 CPU mixed-granularity update 已消除
旧 Diff-Tree 的额外代价，但尚未在整段 trace 上超过 Online。

GPU/OpenCL：

| Method | ms/token | Direct read MiB/fwd | Direct read ms/fwd | Pipeline wait (s) | Tokens |
|:--|--:|--:|--:|--:|--:|
| Static-Min | 1500.059 | 215.071 | 132.309 | 4.890 | 401 |
| Static-Max | **388.743** | 0.000 | 0.000 | 0.000 | 1541 |
| MRU | 843.610 | 180.380 | 94.723 | 9.364 | 707 |
| Offline | 760.417 | 30.526 | 39.990 | 3.089 | 788 |
| Online | 804.248 | 25.133 | 36.307 | 5.450 | 746 |
| Diff-before | 788.938 | 24.971 | 39.441 | 23.221 | 761 |
| Diff-now | **637.170** | 21.318 | 30.220 | 11.706 | 942 |

GPU Diff-now 比 Diff-before、Offline、Online 和 MRU 分别快 19.2%、
16.2%、20.8% 和 24.5%。Static-Max 仍是全驻留性能上界，符合实验预期。

GPU cost model 的关键修正是：pipeline-efficiency residual 只修正
non-resident I/O/Layout work 在 pipeline 上的暴露程度，不再二次缩放
resident GEMV compute。旧实现把 residual 同时乘到 Compute，使高预算 Multi
的实测 kernel 优势被错误抹去，并产生过度 split。修正后，Diff-now 的
frontier 在高预算约为 147 个 coarse units，预算降低时逐步增加到约 188--207
个 units，从而同时保留高预算的 overhead amortization 和低预算的 pipeline
flexibility。

所有 14 行均满足：

- 覆盖原始 600 秒 trace，未映射或压缩时间；
- 正常退出；每次运行前 thermal status 均为 0。GPU 七组结束时仍为 0；
  CPU 七组长时间运行后为 3，但每组均从相同冷却门槛重新开始；
- 共享同一 32-token prefix hash；
- continuous device-isolation check 通过；
- GPU 每行物理 WBM budget violation 为 0；
- GPU `resident_peak` 不超过当前 target 与 64 KiB alignment slack；
- 未出现 `CL_INVALID_MEM_OBJECT`、OpenCL assertion 或 mandatory-reserve
  failure。

早期 GPU dynamic runs 曾在 `NO_AUTO_EVICT=1` 时只更新 planner residency，
而 streamed weights 会在物理 WBM 中累积。这些 runs 已判为 invalid，不进入
上述表。最终 runtime 会在 ensure current unit 前预留空间，保护当前算子的
全部 weight sources，并对未完全释放的 eviction batch 做有界重试。

汇总：

- `exp-results/important_results/motivation/granularity/dynamic_plan/cpu_full10min_baselines.md`
- `exp-results/important_results/motivation/granularity/dynamic_plan/gpu_full10min_baselines.md`

最终 CPU raw results：

- Main baselines：
  `exp-results/Meta-Llama-3-8B-Instruct-Q4_0/difftree_cpu_native_full10min_all_baselines_op13_20260728_graph1`
- Diff-before：
  `exp-results/Meta-Llama-3-8B-Instruct-Q4_0/difftree_cpu_native_full10min_diff_before_phaseonly_op13_20260728_graph1`

最终 GPU raw results：

- Main baselines：
  `exp-results/Meta-Llama-3-8B-Instruct-Q4_0/difftree_gpu_native_full10min_all_baselines_op13_20260728_graph1_budgetsafe_v3`
- Diff-before：
  `exp-results/Meta-Llama-3-8B-Instruct-Q4_0/difftree_gpu_native_full10min_diff_before_phaseonly_op13_20260728_graph1_budgetsafe_v3`
- Diff-now：
  `exp-results/Meta-Llama-3-8B-Instruct-Q4_0/difftree_gpu_native_full10min_diff_now_nonresident_residual_v4_op13_20260728`

## 6. GPU/OpenCL fixed budget：OP12，Llama-3.2-3B

本节保留 2026-07-23 的 OP12 历史结果。它使用逐 unit `clFlush`，不能与
2.1.2 节 OP13 的最终 `unit-sync=none` 数据直接做设备或实现优劣比较。

修复了每个 logical unit 都执行 `clFinish` 的不公平同步后，使用 unit-boundary
`clFlush`。三种模式共 63/63 runs 正常退出并通过 runtime/token 检查。

| Weight budget | Multi | Tensor | Cut | 最快 |
|---:|---:|---:|---:|:---|
| 30% | **1247.833** | 1271.049 | 1590.780 | Multi |
| 40% | 1102.507 | **1093.907** | 1367.640 | Tensor |
| 50% | **924.498** | 942.298 | 1137.951 | Multi |
| 60% | 784.224 | **783.029** | 884.166 | Multi/Tensor |
| 70% | 655.090 | **653.855** | 746.199 | Multi/Tensor |
| 80% | 542.783 | 547.024 | **502.078** | Cut by p50 |
| 90% | **427.437** | 428.814 | 466.573 | Multi/Tensor |

80% 的 median-of-run-means 为 Multi 440.61、Tensor 457.63、Cut
464.20 ms/token，因此 Cut 的优势只出现在 token p50，不能表述为 throughput
也同时获胜。

Raw results：
`exp-results/Llama-3.2-3B-Instruct-Q4_0/granularity_gpu_pipeline_unified_pool_fixed_30_90_op12_20260723`

## 7. GPU/OpenCL dynamic nodes：OP12，Llama-3.2-3B

表值为三轮 plateau p50 的中位数，单位 ms/token。

| Phase | Weight budget | Multi | Tensor | Cut | 最快 |
|:---|---:|---:|---:|---:|:---|
| Initial | 90% | **442.646** | 444.785 | 468.032 | Multi/Tensor |
| Low | 30% | **1182.592** | 1251.788 | 1573.288 | Multi |
| Mid | 60% | 817.399 | **804.517** | 908.247 | Tensor |
| Recovered | 90% | **443.076** | 456.585 | 479.459 | Multi |

Raw results：
`exp-results/Llama-3.2-3B-Instruct-Q4_0/granularity_gpu_pipeline_unified_pool_dynamic_points_op12_20260723`

## 8. 独立设备/配置复现：CPU OP12

这组旧实验使用不同的 CPU thread/lookahead 配置，不能与 OP13 数值直接合并，
但可作为独立 crossover replication。

### Fixed budget

| Weight budget | Multi | Tensor | Cut | 最快 |
|---:|---:|---:|---:|:---|
| 30% | 916.343 | 889.692 | **860.745** | Cut |
| 40% | **643.747** | 645.766 | 702.687 | Multi/Tensor |
| 50% | **531.582** | 531.918 | 621.381 | Multi/Tensor |
| 60% | 450.979 | **438.836** | 537.272 | Tensor |
| 70% | 357.443 | **339.374** | 438.836 | Tensor |
| 80% | 225.347 | **225.076** | 246.811 | Multi/Tensor |
| 90% | 141.498 | **138.107** | 144.251 | Tensor |

### Dynamic nodes

| Phase | Weight budget | Multi | Multi-fused | Tensor | Cut | 最快 |
|:---|---:|---:|---:|---:|---:|:---|
| Initial | 90% | 154.074 | 156.640 | **143.379** | 150.857 | Tensor |
| Low | 30% | 991.081 | 1037.273 | 912.766 | **884.009** | Cut |
| Mid | 60% | 494.024 | 495.878 | **413.675** | 550.525 | Tensor |
| Recovered | 90% | 164.159 | 166.899 | **150.476** | 156.781 | Tensor |

## 9. 机制实验

### 9.1 独立等工作量阶段

三种模式处理相同的 1530 MiB Q4_0 weights 和 5.704253 GFLOP。

| Granularity | Units | I/O | Layout Preparation | Compute |
|:---|---:|---:|---:|---:|
| Multi | 680 | **0.577 s** | 9.138 s | **0.697 s** |
| Tensor | 1360 | 0.663 s | 9.118 s | 1.207 s |
| Cut | 2720 | 0.712 s | **6.477 s** | 2.204 s |

该结果说明不同阶段对颗粒度的偏好相反：

- I/O 偏好较粗 unit，因为相同 bytes 下请求数量更少。
- Compute 偏好较粗 unit，因为 submission/synchronization 更容易摊薄。
- Cut 的局部 transformation 使 Layout Preparation 更快，但付出更高计算和
  submission 开销。

这里的 Layout Preparation 包含 host-to-backend write、layout
conversion/reorganization 和必要同步，不等同于纯 layout kernel 时间。

### 9.2 Controlled operator-pipeline crossover

| Weight budget | Resident ratio | Multi | Tensor | Cut | 最快 |
|---:|---:|---:|---:|---:|:---|
| 216 MiB | 25% | 4.357 s | 4.748 s | **4.319 s** | Cut |
| 432 MiB | 50% | **3.061 s** | 3.389 s | 3.314 s | Multi |
| 648 MiB | 75% | **1.718 s** | 2.063 s | 2.245 s | Multi |
| 864 MiB | 100% | **0.401 s** | 0.676 s | 1.217 s | Multi |

这只能作为 mechanism evidence，不能当作真实 model inference 结果。

## 10. Dynamic memory budget 下的 granularity 选择分析

### 10.1 为什么不能只根据当前 budget 选择

现有结果说明，相同 budget ratio 并不总是对应相同的最优 granularity：

- 最新 CPU fixed 60%：Tensor 为 458.978 ms/token，Cut 为 487.015 ms/token。
- CPU dynamic 60%：Tensor 为 423.565 ms/token，Cut 为 397.900 ms/token。
- 最新 OP13 GPU fixed 30%：Cut 为 388.501 ms/token，Multi 为
  557.781 ms/token；80% 则由 Multi 最快，90% 又切换为 Tensor。
- 历史 OP12 GPU fixed 30% 使用逐-unit flush 时由 Multi 领先。这一对比说明
  backend implementation policy 也必须纳入选择模型，不能把旧同步开销误当成
  granularity 的固有代价。
- CPU Cut 在相同 40% budget 下，graph-lookahead=4 为 1900.262 ms/token，
  graph-lookahead=2 为 623.842 ms/token，说明 granularity 与 pipeline depth
  需要联合设计。
- 8B native full trace 的 CPU 70% bin 是 Cut 最快，80%/90% bins 则是
  Multi 最快。
- 同一 8B native full trace 的 GPU 70%/80% bins 是 Tensor 最快，90% bin
  则是 Multi 最快。

因此，`budget -> granularity` 的静态阈值不足以处理 dynamic 场景。选择还取决于
backend、当前 resident weights、已经完成的 layout preparation、pending/in-flight
units、budget 变化方向以及新 budget 的持续时间。更合适的问题定义是：

> 在当前 memory 和 pipeline state 下，哪种 working-unit granularity 能够使
> 未来若干 token 的 pipeline critical path 最短？

### 10.2 三种 granularity 的优劣

| Granularity | 优势 | 代价 | 更可能占优的条件 |
|:---|:---|:---|:---|
| Multi | I/O 请求较少；submission 和 synchronization 开销容易摊薄；计算效率高；compatible pair 可进一步 fuse | residency/eviction 粒度粗；budget-fitting waste 和 head-of-line blocking 较大；可供 pipeline 重叠的独立 units 较少 | budget 稳定；I/O 或 compute 是 critical stage；unit launch/sync 开销较高；有足够 headroom 容纳完整 unit |
| Tensor | 与原始 operator/kernel 自然对应；不需要切分、合并；memory flexibility 和 execution efficiency 较平衡 | 请求和 submission 数高于 Multi；没有 Cut 的局部 layout 优势；不能利用 partial-tensor residency | pipeline 各阶段接近平衡；budget 波动中等；选择置信度不足时作为稳健模式 |
| Cut | 局部 Layout Preparation 更快；更容易填满有限 budget；eviction/reload 反应快；head-of-line blocking 小；能够提供更多 overlap 机会 | I/O 请求、kernel launch、同步和 bookkeeping 更多；计算几何和 cache locality 可能变差；GPU submission/sub-buffer 代价尤其明显 | preparation、pipeline wait 或 budget fragmentation 是瓶颈；budget 快速变化；细粒度 overlap 收益能够覆盖执行开销 |

`multi_fused` 不作为第四种 granularity。它是 Multi unit 内部的实现优化；否则会把
granularity effect 和 kernel/layout fusion effect 混在同一个选择维度中。

### 10.3 在线状态指标

选择器至少需要下列四类信息。

1. Memory state

   - 当前 weight budget、resident bytes 和 non-resident bytes；
   - lookahead working set 中的 future non-resident bytes；
   - 下一候选 unit 是否能够放入 budget；
   - eviction/reload bytes 和已经 prepared 的 bytes。

   可定义：

   \[
   R_{\mathrm{nr}} =
   \frac{\text{future non-resident bytes}}
        {\text{lookahead working-set bytes}}
   \]

   `R_nr` 比单纯 resident ratio 更接近后续 token 真正需要执行的非驻留工作量。

2. Budget dynamics

   - `dB/dt`：budget 是上升、下降还是稳定；
   - 近期 budget variance；
   - 距离上次 budget change 的时间；
   - 预测 dwell time；
   - change amplitude 与每种 unit size 的比值。

3. Pipeline state

   - ready、pending、in-flight 和 missing unit 数量/bytes；
   - LOAD、PREPARE、COMPUTE queue depth；
   - pipeline wait、fill 和 drain 时间；
   - prepare-cap decline、unissued bytes 和 head-of-line blocking。

   关键指标为：

   \[
   S_g =
   \frac{T_{\mathrm{pipeline\ wait},g}}
        {T_{\mathrm{token},g}}
   \]

   较高的 stall ratio 表示更细的 units 可能通过增加 ready work 和 overlap 获益，
   但仍需与新增的 submission 开销比较。

4. Backend-specific service rate

   对每个 backend 和 granularity 分别维护在线 EMA：

   \[
   T_{\mathrm{io},g} =
   \frac{D_{\mathrm{io}}}{BW_{\mathrm{io},g}},\quad
   T_{\mathrm{prep},g} =
   \frac{D_{\mathrm{prep}}}{BW_{\mathrm{prep},g}},\quad
   T_{\mathrm{compute},g} =
   \frac{Work}{Perf_{\mathrm{compute},g}}
   \]

   还需要统计每个 unit 的 queue、launch、sync 和 bookkeeping cost。CPU 和 GPU
   必须使用独立参数；现有 30% 结果已经显示两个 backend 的选择方向可以相反。

### 10.4 候选代价模型

pipeline-on 时不能简单相加 I/O、preparation 和 compute。理想稳态主要受最慢
stage 限制，同时还存在 fill/drain、stall 和 unit overhead。最新 Cut 40%
结果还表明 pipeline depth `p` 与 granularity `g` 存在交互，因此代价模型应写成：

\[
\hat T_{g,p} =
\max
\left(
T_{\mathrm{io},g,p},
T_{\mathrm{prep},g,p},
T_{\mathrm{compute},g,p}
\right)
+
T_{\mathrm{fill/drain},g,p}
+
T_{\mathrm{stall},g,p}
+
N_{\mathrm{unit},g} h_{\mathrm{unit},g,p}
+
T_{\mathrm{churn},g,p}
+
T_{\mathrm{switch},g,p}
\]

最终选择可以写为：

\[
(g^*,p^*) =
\arg\min_{g,p}
\left[
\hat T_{g,p}
+
\lambda_m P_{\mathrm{overbudget},g,p}
+
\lambda_t T_{\mathrm{tail},g,p}
\right]
\]

其中 `T_churn` 表示 speculative staging 过深导致的提前 prepare、evict/reload
和 memory-bandwidth contention；`T_switch` 包括切换 unit metadata、已有
prepared/resident state 无法复用以及 pipeline drain 的代价。只有预测收益持续
若干 token 且超过 switching cost 时才切换，从而避免 budget 小幅抖动导致
granularity/pipeline-depth oscillation。

### 10.5 下一步需要补充的测量

- 将 cumulative counters 改为 per-token delta，记录每个 token 的 budget、
  resident/prepared bytes、stage time、wait time 和 queue state。
- CPU 分离 direct read、repack/materialize、compute 和 pipeline wait。
- GPU 使用 device event profiling 分离 backend write、layout kernel、
  launch/synchronization；不能把它们统一解释成纯 layout kernel。
- 设计不同 drop/rise amplitude、变化速度和 dwell time 的 step experiments，
  观察同一 instantaneous budget 下 history 对最优 granularity 的影响。
- 使用 fixed-budget 和 step/trace 数据建立 offline oracle label：
  `argmin_g latency(g)`，再判断解析模型、lookup table 或可解释 decision tree
  哪一种能够用最少指标逼近 oracle。
- 对完整 10 分钟 trace 增加 counterbalanced 重复，并补充其他 model-matched
  traces；当前单次 full trace 只作为真实路径 case study，不用于方差结论。

## 11. 可用于 Motivation 的表述

推荐使用以下保守表述：

> 在真实 CPU-only 和 GPU-only Elastic inference 路径中，working-unit
> granularity 会使相同 memory budget 下的 decode latency 出现显著差异；
> 最优选择随 budget、backend 和动态切换历史变化。独立阶段实验进一步表明，
> I/O/compute 偏好粗粒度，而局部 layout preparation 可偏好 sub-tensor
> 粒度，因此不存在对所有运行条件都占优的固定 granularity。

不要把以下内容作为正式证据：

- 旧 `multi_fused` 扩大 pipeline range 的错误实现；
- Cut 90% 约 2.3 s/token 的异常运行；
- pipeline-off pilots；
- 混入 pipeline waiting 的旧 “load time”；
- 将 OLMoE trace 线性映射到 Llama 30%--90% 的 derived trace；
- 旧 128 MiB mixed-stage 表对纯 I/O 的解释。
