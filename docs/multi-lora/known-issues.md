# Multi-LoRA Bench: Known Issues / Open Questions

记下 spec 走读时识别的潜在问题，留到对应 milestone 实施时回看。
**这份文件不是 todo list**——是"做到那一步要回头看"的索引。

---

## I-1 (M2): "no starvation" 验收标准 vs "第一版不解决 starvation" 的张力

**位置**：`IMPLEMENTATION_GUIDE.md` §4 M2

**矛盾**：
- 文中明说"第一版**不解决 starvation**——如果某 adapter 永远是少数，就让它等"。
- 但同节验收标准 #3 写着"没有 slot 卡死：任何 active slot 在 30s 内必定推进 ≥ 1 token"。

**解读**：
两条不是直接冲突，因为 `pick_largest_adapter_group` 在 active slot 持续到达的情况下，
最大组会随时间变化——不会**永远**饿死单个 slot。但如果 trace 偏 skewed
（一个 adapter 持续来一大堆请求，其他 adapter 各 1 个，又持续被新请求挤掉
"最大组"位置），少数派 slot 的等待时间确实可能 > 30s。

**实施时怎么判**：
1. M2 跑验收 trace 时，metric 里看每个 slot 的 `admit_time → first_token_time` 间隔。
2. 如果有 slot > 30s——记录哪条 trace、哪个 adapter、组分布——
   先把数据贴到本文件，再决定要么加最简 priority age（N 步内必选一次最老 slot），
   要么放宽验收标准。
3. **不要在 M2 提前加 priority age**。先观测，再决定。

---

## I-2 (M3): `evict_until` 里 ref_count > 0 的 splice 逻辑会污染 LRU 语义

**位置**：`IMPLEMENTATION_GUIDE.md` §5.1，伪代码内 `evict_until`：

```cpp
if (victim.ref_count > 0) {
  lru_.splice(lru_.begin(), lru_, --lru_.end());
  continue;
}
```

**问题**：
把"暂时不能 evict 因为正在用"的 entry 从 LRU 尾部挪到**头部**（MRU 端）——
这等于谎报它"刚被访问过"。下一次 evict 也会再次错误地把它视为热门，
即使它的 `ref_count` 已经归零、且实际很久没被使用。

**正确做法（M3 实施时改 spec + 改伪代码）**：
- 维护**两个**列表：`lru_free_`（ref==0，可驱逐）+ `pinned_`（ref>0，不可驱逐）。
  `acquire` ref++ 时移到 pinned，`release` ref--==0 时移回 lru_free 的 MRU 端。
- `evict_until` 只扫 `lru_free_` 尾部。
- 这样 LRU 顺序只反映真实的最近访问，不被 in-use 状态污染。

**实施时**：M3 起手第一步，先在本文件写明改法，
更新 §5.1 的伪代码，再写代码。**不要照抄 spec 现有伪代码**。

---

## I-3 (M1+): llama.cpp LoRA / model API 在新版本可能已重命名

**位置**：`IMPLEMENTATION_GUIDE.md` §3.1, §1.1.2 伪代码

**怀疑名字已变**（第一次走读发现，未 grep 验证）：
- `llama_lora_adapter_init / _set / _clear / _free`
  → 可能 `llama_adapter_lora_init / ...`
- `llama_load_model_from_file` → `llama_model_load_from_file`
- `llama_new_context_with_model` → `llama_init_from_model`
- `llama_token_eos(model)` → `llama_vocab_eos(vocab)` 之类

**实施时（M1 起手第一步）**：
1. `grep` `include/llama.h` 与 `src/llama-adapter.*` 找实际 symbol。
2. 把对应表写到 `docs/multi-lora/api-versions.md`（commit hash + 实际签名）。
3. 如果跟 spec 不一致——**先改 spec**，再写代码。

---

*创建：2026-05-02。M2/M3 实施时回看本文件。*
