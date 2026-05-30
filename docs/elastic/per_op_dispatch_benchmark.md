# Per-Op Runtime Dispatch — Benchmark Results

测试: Llama-3.2-3B FP16, cpu-elastic build (Linux x86), n=30 token decode, 3 runs each, ms/tok min/mean/max.

跑测 bench 脚本: `/tmp/bench_per_op_dispatch.sh`

## 主结果

| Policy | overrides | ms/tok (min / mean / max) | vs baseline | per-override |
|---|---|---|---|---|
| **baseline** (无 hook) | 0 | **281 / 300 / 319** | — | — |
| memory-driven (无触发) | 0 | 302 / 311 / 327 | +3.6% | (噪音) |
| alternate (50% mul_mat → CPU) | 3921 | 412 / 419 / 427 | **+39.7%** | 0.91 ms |
| layer-half (后 14 层 → CPU) | 3906 | 408 / 444 / 469 | **+47.8%** | 1.10 ms |
| ffn-cpu (所有 ffn → CPU) | 2604 | 403 / 431 / 455 | **+43.7%** | 1.51 ms |
| attn-cpu (所有 attn → CPU) | 5208 | 452 / 486 / 541 | **+61.8%** | 1.07 ms |
| memory-driven-FORCED (100% → CPU) | 7843 | 548 / 572 / 606 | **+90.7%** | 1.04 ms |

(per-override = (mean - baseline_mean) / (overrides / n_tokens) — extra ms per overridden op)

## 关键发现

### 1. 每 override 切换开销 ≈ 1 ms

无论 policy 怎么切, 每个被 hook 重新路由的 op 大约花 1 ms 额外时间:
- 跨 backend split 创建
- per-op `ggml_backend_graph_compute_async` 启动开销
- 跨 backend `ggml_backend_synchronize`

cpu-elastic 跟 cpu 都在 host memory 上, sync 是 mutex/atomic 级别 (微秒), 但 per-op
graph_compute 启动 + thread pool dispatch 是毫秒级.

### 2. Policy 选择影响巨大

| Override 数 / token | 慢多少 |
|---|---|
| 130 (alternate / layer-half) | +40-48% |
| 87 (ffn-cpu) | +44% |
| 174 (attn-cpu) | +62% |
| 261 (forced CPU) | +91% |

**每 token override 数跟 slowdown 近线性** — 每个 override ~1 ms × override 数 = slowdown.

### 3. cpu-elastic build 上 CPU 本身没明显更慢

baseline = elastic-CPU + ngl 默认 = 300 ms/tok
forced CPU (override 100%) = 572 ms/tok = 300 + 261 * 1 ms

= 100% override 的 272 ms slowdown 几乎全是 **per-op dispatch overhead**, 而不是 CPU
compute 本身慢. 因为这个 build 没有 GPU, "default backend" 也是 CPU.

## 对真 Mobile (CPU + Adreno OpenCL) 的外推

假设 Adreno GPU 比 CPU F16 GEMM 快 5-10×:
- baseline ngl=99 大概 ~180 ms/tok (实测 3B F16 -ngl 99 elastic OFF = 142 ms/tok)
- 移 1 个 mul_mat 到 CPU: GPU 该 op ~0.5 ms → CPU ~3-5 ms = +3-5 ms 真 compute loss + 1 ms dispatch overhead
- 移 100 个 mul_mat: ~300-500 ms 真 loss + 100 ms dispatch overhead = 总 +400-600 ms

dispatch overhead 是 **次要因素 (~20-30%)**, 真正大头是 CPU compute 本身慢于 GPU.

## Override 多少有意义

按 100 overrides/token 算:
- 真 CPU vs GPU 速度差 (e.g. 4× slower CPU): 每 override 多 3-5 ms compute → +300-500 ms/tok
- per-op dispatch overhead: +100 ms/tok

**用 per-op dispatch 是 OK 的**, 因为 overhead < 真 compute 速度差. 部署时按需要切几个
ops 决定整体损失.

## 何时不该用

- 整图都走 GPU 就行的场景: 无意义 (overhead 白付)
- 整图都走 CPU: 直接 -ngl 0 即可, 不需要 hook
- 只有少数 op (< 10/token) 需要切: 可以接受
- 大量 ops 切 (> 100/token): 每个都付 1 ms overhead, 累计明显

## Bench 复现

```bash
# 脚本: /tmp/bench_per_op_dispatch.sh
MODEL=/tmp/3bf16/Llama-3.2-3B-Instruct-f16.gguf
N_PRED=30
N_RUNS=3
# 每个 policy 跑 3 次, 取 min/mean/max ms/tok
```

3 runs 之间 variance 大概 ±15-30 ms/tok (= 5-10%), 所以 < 5% policy 差别在噪音内.
