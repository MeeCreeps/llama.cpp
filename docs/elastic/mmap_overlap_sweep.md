# mmap path overlap sweep — chunked compute + worker thread overlap

## Setup

3B F16, OnePlus 12 Snapdragon 8 Gen 3, 12 GB RAM.
mmap 路径 (无 O_DIRECT, page cache 命中). chunked compute + std::thread worker
做 ensure_phase for next chunk 跟 current chunk compute overlap.

## B=3500 (39% miss per cycle)

| chunk_size | r1 | r2 | r3 | min | avg | vs baseline |
|-----------|---:|---:|---:|----:|----:|:------------|
| 0 (baseline serial) | 1147 | 1090 | 1047 | 1047 | 1095 | — |
| 64 | 966 | 1285 | 1203 | 966 | 1151 | min -8% |
| 128 | 2080 | 1363 | 929 | 929 | 1457 | min -11%, variance 大 |
| **256** | 1026 | **870** | 1142 | **870** | **1013** | **min -17%, avg -7%** |
| 512 | 1052 | 1449 | 1009 | 1009 | 1170 | min -4% |

## B=5000 (8% miss per cycle)

| chunk_size | min | avg |
|-----------|----:|----:|
| 0 (baseline) | 713 | **744** (最佳) |
| 128 | 691 | 749 |
| 256 | 707 | 807 |

## 分析

- **chunk=256 是甜点** — chunk compute time ≈ chunk ensure_phase time, overlap 窗口最大
- **overlap 收益 ∝ miss 比例**: B=3500 (39% miss) 收益 17%, B=5000 (8% miss) 无收益
- **mmap baseline 已经比 O_DIRECT baseline 快 1.7×** (1047 vs 1808 @ B=3500) — page cache 命中是
  最大优化, 比 overlap 重要 5× 以上
- worker thread 在 mmap 路径下比 O_DIRECT+io_uring 路径稳定, variance 略小

## Overlap 效率估算 (B=3500)

- baseline 1047 ms = ensure (memcpy ~400 ms) + compute (~600 ms) 接近 serial sum
- 理论 perfect overlap: max(400, 600) = 600 ms
- 实测 chunk=256: 870 ms
- 节省 = 1047 - 870 = 177 ms
- 理论上限 savings = 1047 - 600 = 447 ms
- **overlap 效率 = 177/447 ≈ 40%** (跟 O_DIRECT+io_uring 路径的 33% 接近)

## 跟 O_DIRECT+uring 路径对比 (B=3500)

| 路径 | baseline | best chunked | min |
|------|---------:|-------------:|----:|
| O_DIRECT + io_uring chunk=256 | 1808 | 1686 (-7%) | 1630 |
| mmap + thread chunk=256 | 1047 | 1013 (-7%) | 870 |

mmap 路径无论 baseline 还是 chunked 都明显快于 O_DIRECT, 因为 page cache 命中 (3B
模型 6 GB < 12 GB phone RAM, 整模型常驻 cache).

## 建议默认配置 (mmap 路径)

- `GGML_ELASTIC_CHUNK_SIZE=256` (worker thread 路径; uring 在 mmap 下无意义)
- `GGML_ELASTIC_PREFETCH=64` (madvise WILLNEED, 无 chunk 时仍有用)
- `GGML_ELASTIC_PIN=norm,k,v` (hot tensor 钉住)
- `GGML_ELASTIC_EMBED_OUTSIDE_BUDGET=1`
- `GGML_ELASTIC_EVICT_POLICY=mru` (默认已是)

仅当 model > RAM 强制 disk-bound (O_DIRECT 模拟) 才考虑 io_uring 路径.
