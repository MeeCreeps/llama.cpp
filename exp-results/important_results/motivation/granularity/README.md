# Elastic working-unit granularity motivation

本目录同时保留：

- [result.md](result.md)：面向论文 motivation 的完整汇总与解释。
- [results.d/results.md](results.d/results.md)：之前整理的机器可读 CSV 数据及索引。
- 根目录旧表和 `mechanism/`：历史结果与 provenance，保留但不与最终结果混用。

当前主要结论是：真实 CPU-only 和 GPU-only Elastic inference 中，仅改变
working-unit granularity 就会改变 decode latency；最优模式会随 memory budget、
backend 和动态切换历史改变。所有正式真实模型结果都开启 pipeline。

当前 OP12 v21 正在用统一的 inside-budget `token_embd,output` policy、
worker affinity、streaming headroom 和原始 600 秒 trace 重做 mixed-plan 七
baseline。v21 完成前，`result.md` 的 5.3 节只保留为历史诊断表，不能作为最新
Diff-now 排序。

旧 `multi_fused` 扩大 pipeline range 的错误数据、异常 Cut-90%、pipeline-off
pilot 和线性映射 OLMoE trace 只作为历史诊断文件保留，不进入正式结论。
