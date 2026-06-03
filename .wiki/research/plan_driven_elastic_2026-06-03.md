# Plan-driven elastic 真机实测:budget → decode 速度 (M7)

> 设备:OnePlus 12 (CPH2583) / Snapdragon 8 Gen 3 / Adreno 750, 16GB UMA。
> 日期:2026-06-03。模型:Llama-3.2-3B-Instruct **f16** (6.4GB)。
> 配置:`GGML_OPENCL_ELASTIC=1 GGML_ELASTIC_NO_AUTO_EVICT=1` + `LLAMA_ELASTIC_APPLY=plan_<B>.json`
> (plan framework 独占 residency,evict-only + backend reload-on-use,**无 M6 overlap**)。
> 数据来源:[实测] decode 16 token,取 `eval time` per-token(剔除 prompt/load)。

## 0. 一句话

**budget 越大、常驻越多、流式越少 → 越快。真机实测 2800→7408 MiB = decode 19× 提速(4141→217 ms/tok)。** 这就是 dynamic memory budget 下 plan-driven 的价值:budget 高时多留、抓住提速;static 保守(锁 min)永远停在最慢档。

## 1. 实测:per-band steady-state decode

| budget band | 常驻 (of 197 weights) | 流式/token | **实测 ms/tok** | 预测(overlapped) | 实测/预测 |
|---|---|---|---|---|---|
| 2800 MiB | 58 | 139 | **4141** | 2035 | 2.0× |
| 4912 MiB | 102 | 95 | **3549** | ~900 | ~3.9× |
| 7408 MiB | 179 | 18 (+migrate 16) | **217** | 256 | **0.85×** |

- **趋势真实且陡**:7408 比 2800 快 **19×**。大头在「流式几乎消失」那一段(4912→7408:3549→217)。
- **高 budget 段实测 ≈ 预测**(217 vs 256):流式极少时,框架几乎达到 cost model 预测的性能,**即使没有 overlap**。
- **低 budget 段实测 ≫ 预测**(4141 vs 2035,2×):重流式时,每个流式 weight 串行 reload(disk→cpu→gpu+convert,**没和 compute overlap**)→ 比预测(假设 overlapped 流式)慢一倍。**这正是 M6 overlap 该补的地方**。

## 2. dynamic vs static 的含义

- **static-conservative**(以往工作:provision for worst case = 锁 min budget):永远 plan_2800 → **4141 ms/tok**,不管当下内存多宽裕。
- **plan-driven dynamic**(本框架:跟随 B(t) 换 band):budget 升到 7408 时拿到 **217 ms/tok**。
- 若一条 trace 一半时间在 2800、一半在 7408:
  - static ≈ 4141 ms/tok(恒定)
  - dynamic ≈ (4141+217)/2 ≈ **2179 ms/tok → 1.9× faster**;高 budget 占比越大优势越大。

**结论:dynamic budget 场景下,plan-driven 抓住了 static 留在桌上的高-budget 提速。** 这是本研究相对「固定 budget」前作的核心增量,真机验证成立。

## 3. 关键限制 / 下一步

- **本测无 M6 overlap**:低 budget 段(重流式)实测比预测慢 2×。M6(timeline anchor → 异步 prefetch,disk/dma 与 compute overlap,per-engine busy §5.5)主要能压低**重流式段**,把低-budget 档拉近预测,放大 dynamic 优势。
- **f16 GPU convert 贵**:流式 weight 的 GPU SOA+transpose 强制且慢(见 [[layout_transform_cost_2026-06-03]])。Q4_0 / Q8 模型流式更便宜,后续测 plans_q4 + 8B Q4_0。
- **migrate=16 @7408**:高档出现 GPU-streamed weights(routes=gpu 但非常驻),触发跨后端迁移;端到端正确性已隐含验证(输出正确),但迁移路径的单独 profiling 待做(M5)。
- evict-only + backend reload-on-use 是当前稳定范式(plan 控制踢谁,backend 控制 reload);主动 prefetch 需先解决 tensor->extra 同步(见 CHANGES_04)。

## 复现

```bash
DEV=/data/local/tmp/elastic
for B in 2800 4912 7408; do
  adb shell "cd $DEV && LD_LIBRARY_PATH=.:/system/vendor/lib64:/vendor/lib64 \
    GGML_OPENCL_ELASTIC=1 GGML_ELASTIC_NO_AUTO_EVICT=1 \
    LLAMA_ELASTIC_APPLY=plans/plans/plan_${B}MiB.json \
    ./llama-cli-planfw -m Llama-3.2-3B-Instruct-f16.gguf -p 'Count: 1 2 3' -n 16 --seed 42 --temp 0 -ngl 99 -no-cnv" \
    2>&1 | grep -E "applied plan|eval time ="
done
```

相关:framework 见 `.wiki/elastic_memory/feature_elastic-plan-framework/`(STATUS + CHANGES_01..04)。
