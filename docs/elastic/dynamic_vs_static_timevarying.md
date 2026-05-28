# Dynamic vs Static — time-varying budget 实测

## 测试设定

3B F16, OnePlus 12. mmap + chunk=256 + worker thread overlap (两边相同).
n=30 token decode.

### Schedule (time-varying)

| 时段 | B(t) | 模拟 |
|------|-----:|:-----|
| 0-3s | 5000 | 空闲 |
| 3-8s | 2000 | 其它 app 抢内存 |
| 8-15s | 3500 | 部分释放 |
| 15-25s | 5000 | 空闲 |
| 25-40s | 2500 | 再次 pressure |
| 40s+ | 5000 | 恢复 |

Static 必须按 min(B(t)) = 2000 MB 全程配置. Dynamic 跟随 B(t).

## 结果

| | r1 | r2 | r3 | min | avg |
|---|---:|---:|---:|----:|----:|
| static-safe (常 2000) | 4958 | 4390 | 7698 | 4390 | 5682 |
| **dynamic** | 2307 | 1367 | 1338 | 1338 | **1671** |

**Dynamic 比 static-safe 快 3.4× (avg 5682 → 1671 ms/tok)**.

## 分析

- Static 全程 ~75% miss → reload 量大
- Dynamic 在 B(t)=5000 (~50% 时段) 只 ~8% miss → 接近 ceiling
- pressure 时段退化到 static 水平
- 加权平均大幅领先

## 适用场景

mobile OS memory pressure 变化 (后台 app 启停). Static worst-case 配置浪费
90% 时段可用内存. Dynamic 让推理"随波逐流".

CSV: `b_dynamic.csv`, `b_static_safe.csv`. 格式 `t_ms,B_MB`.
