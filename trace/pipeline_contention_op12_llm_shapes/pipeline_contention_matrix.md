# Pipeline Contention Matrix Summary

| run | pair | med ms | ideal ms | serial ms | competition | speedup | class |
|---|---|---:|---:|---:|---:|---:|---|
| llm-8b-attn-q4_0 | `disk_load+cpu_xform` | 4.964 | 3.348 | 6.456 | 1.48 | 1.30 | poor |
| llm-8b-attn-q4_0 | `disk_load+gpu_xform` | 4.877 | 4.926 | 8.274 | 0.99 | 1.70 | good |
| llm-8b-attn-q4_0 | `disk_load+cpu_compute` | 45.284 | 45.510 | 48.857 | 1.00 | 1.08 | partial |
| llm-8b-attn-q4_0 | `disk_load+gpu_compute` | 79.734 | 78.953 | 82.301 | 1.01 | 1.03 | poor |
| llm-8b-attn-q4_0 | `cpu_xform+gpu_xform` | 5.031 | 4.926 | 8.035 | 1.02 | 1.60 | good |
| llm-8b-attn-q4_0 | `cpu_xform+cpu_compute` | 46.660 | 45.510 | 48.618 | 1.03 | 1.04 | poor |
| llm-8b-attn-q4_0 | `cpu_xform+gpu_compute` | 80.285 | 78.953 | 82.062 | 1.02 | 1.02 | poor |
| llm-8b-attn-q4_0 | `gpu_write+gpu_compute` | 82.795 | 78.953 | 79.757 | 1.05 | 0.96 | poor |
| llm-8b-attn-q4_0 | `gpu_write_raw+gpu_compute` | 81.752 | 78.953 | 79.362 | 1.04 | 0.97 | poor |
| llm-8b-attn-q4_0 | `gpu_write_two+gpu_compute` | 82.487 | 78.953 | 80.940 | 1.04 | 0.98 | poor |
| llm-8b-attn-q4_0 | `gpu_xform+gpu_compute` | 85.455 | 78.953 | 83.879 | 1.08 | 0.98 | poor |
| llm-8b-attn-q4_0 | `gpu_xform_kernels+gpu_compute` | 80.929 | 78.953 | 82.143 | 1.03 | 1.01 | poor |
| llm-8b-attn-q4_0 | `cpu_compute+gpu_compute` | 80.201 | 78.953 | 124.463 | 1.02 | 1.55 | good |
| llm-8b-attn-q4_0 | `disk_load+cpu_xform+gpu_xform+gpu_compute` | 85.679 | 78.953 | 90.335 | 1.09 | 1.05 | partial |
| llm-8b-ffn-q4_0 | `disk_load+cpu_xform` | 13.085 | 11.333 | 20.348 | 1.15 | 1.56 | partial |
| llm-8b-ffn-q4_0 | `disk_load+gpu_xform` | 12.719 | 14.702 | 23.716 | 0.87 | 1.86 | good |
| llm-8b-ffn-q4_0 | `disk_load+cpu_compute` | 188.279 | 186.823 | 195.838 | 1.01 | 1.04 | poor |
| llm-8b-ffn-q4_0 | `disk_load+gpu_compute` | 240.037 | 238.686 | 247.701 | 1.01 | 1.03 | poor |
| llm-8b-ffn-q4_0 | `cpu_xform+gpu_xform` | 15.102 | 14.702 | 26.035 | 1.03 | 1.72 | good |
| llm-8b-ffn-q4_0 | `cpu_xform+cpu_compute` | 188.332 | 186.823 | 198.157 | 1.01 | 1.05 | partial |
| llm-8b-ffn-q4_0 | `cpu_xform+gpu_compute` | 240.399 | 238.686 | 250.020 | 1.01 | 1.04 | poor |
| llm-8b-ffn-q4_0 | `gpu_write+gpu_compute` | 245.517 | 238.686 | 240.320 | 1.03 | 0.98 | poor |
| llm-8b-ffn-q4_0 | `gpu_write_raw+gpu_compute` | 246.234 | 238.686 | 240.286 | 1.03 | 0.98 | poor |
| llm-8b-ffn-q4_0 | `gpu_write_two+gpu_compute` | 246.084 | 238.686 | 241.482 | 1.03 | 0.98 | poor |
| llm-8b-ffn-q4_0 | `gpu_xform+gpu_compute` | 254.406 | 238.686 | 253.388 | 1.07 | 1.00 | poor |
| llm-8b-ffn-q4_0 | `gpu_xform_kernels+gpu_compute` | 244.796 | 238.686 | 246.943 | 1.03 | 1.01 | poor |
| llm-8b-ffn-q4_0 | `cpu_compute+gpu_compute` | 244.170 | 238.686 | 425.510 | 1.02 | 1.74 | good |
| llm-8b-ffn-q4_0 | `disk_load+cpu_xform+gpu_xform+gpu_compute` | 255.915 | 238.686 | 273.736 | 1.07 | 1.07 | partial |
