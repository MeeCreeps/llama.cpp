# Pipeline Contention Matrix Summary

| run | pair | med ms | ideal ms | serial ms | competition | speedup | class |
|---|---|---:|---:|---:|---:|---:|---|
| balanced-gpu-compute | `disk_load+cpu_xform` | 11.084 | 11.324 | 18.213 | 0.98 | 1.64 | good |
| balanced-gpu-compute | `disk_load+gpu_xform` | 7.039 | 6.889 | 11.856 | 1.02 | 1.68 | good |
| balanced-gpu-compute | `disk_load+cpu_compute` | 188.359 | 186.854 | 193.743 | 1.01 | 1.03 | poor |
| balanced-gpu-compute | `disk_load+gpu_compute` | 239.558 | 238.263 | 245.151 | 1.01 | 1.02 | poor |
| balanced-gpu-compute | `cpu_xform+gpu_xform` | 11.237 | 11.324 | 16.291 | 0.99 | 1.45 | good |
| balanced-gpu-compute | `cpu_xform+cpu_compute` | 188.388 | 186.854 | 198.178 | 1.01 | 1.05 | partial |
| balanced-gpu-compute | `cpu_xform+gpu_compute` | 239.415 | 238.263 | 249.587 | 1.00 | 1.04 | poor |
| balanced-gpu-compute | `gpu_write+gpu_compute` | 243.961 | 238.263 | 239.879 | 1.02 | 0.98 | poor |
| balanced-gpu-compute | `gpu_write_raw+gpu_compute` | 240.224 | 238.263 | 238.637 | 1.01 | 0.99 | poor |
| balanced-gpu-compute | `gpu_write_two+gpu_compute` | 240.820 | 238.263 | 240.038 | 1.01 | 1.00 | poor |
| balanced-gpu-compute | `gpu_xform+gpu_compute` | 243.904 | 238.263 | 243.229 | 1.02 | 1.00 | poor |
| balanced-gpu-compute | `gpu_xform_kernels+gpu_compute` | 240.244 | 238.263 | 241.438 | 1.01 | 1.00 | poor |
| balanced-gpu-compute | `cpu_compute+gpu_compute` | 243.745 | 238.263 | 425.116 | 1.02 | 1.74 | good |
| balanced-gpu-compute | `disk_load+cpu_xform+gpu_xform+gpu_compute` | 244.834 | 238.263 | 261.443 | 1.03 | 1.07 | partial |
| long-gpu-compute | `disk_load+cpu_xform` | 14.218 | 11.354 | 18.115 | 1.25 | 1.27 | partial |
| long-gpu-compute | `disk_load+gpu_xform` | 6.657 | 6.761 | 12.093 | 0.98 | 1.82 | good |
| long-gpu-compute | `disk_load+cpu_compute` | 375.114 | 373.613 | 380.374 | 1.00 | 1.01 | poor |
| long-gpu-compute | `disk_load+gpu_compute` | 290.630 | 288.909 | 295.670 | 1.01 | 1.02 | poor |
| long-gpu-compute | `cpu_xform+gpu_xform` | 11.236 | 11.354 | 16.686 | 0.99 | 1.49 | good |
| long-gpu-compute | `cpu_xform+cpu_compute` | 375.168 | 373.613 | 384.967 | 1.00 | 1.03 | poor |
| long-gpu-compute | `cpu_xform+gpu_compute` | 290.259 | 288.909 | 300.263 | 1.00 | 1.03 | poor |
| long-gpu-compute | `gpu_write+gpu_compute` | 296.177 | 288.909 | 290.522 | 1.03 | 0.98 | poor |
| long-gpu-compute | `gpu_write_raw+gpu_compute` | 291.750 | 288.909 | 289.278 | 1.01 | 0.99 | poor |
| long-gpu-compute | `gpu_write_two+gpu_compute` | 291.322 | 288.909 | 291.584 | 1.01 | 1.00 | poor |
| long-gpu-compute | `gpu_xform+gpu_compute` | 294.496 | 288.909 | 294.241 | 1.02 | 1.00 | poor |
| long-gpu-compute | `gpu_xform_kernels+gpu_compute` | 291.301 | 288.909 | 292.155 | 1.01 | 1.00 | poor |
| long-gpu-compute | `cpu_compute+gpu_compute` | 374.024 | 373.613 | 662.522 | 1.00 | 1.77 | good |
| long-gpu-compute | `disk_load+cpu_xform+gpu_xform+gpu_compute` | 295.905 | 288.909 | 312.356 | 1.02 | 1.06 | partial |
| short-gpu-compute | `disk_load+cpu_xform` | 11.106 | 11.331 | 14.518 | 0.98 | 1.31 | good |
| short-gpu-compute | `disk_load+gpu_xform` | 5.301 | 5.015 | 8.202 | 1.06 | 1.55 | good |
| short-gpu-compute | `disk_load+cpu_compute` | 93.565 | 93.442 | 96.629 | 1.00 | 1.03 | poor |
| short-gpu-compute | `disk_load+gpu_compute` | 79.830 | 78.672 | 81.859 | 1.01 | 1.03 | poor |
| short-gpu-compute | `cpu_xform+gpu_xform` | 11.157 | 11.331 | 16.346 | 0.98 | 1.47 | good |
| short-gpu-compute | `cpu_xform+cpu_compute` | 94.948 | 93.442 | 104.773 | 1.02 | 1.10 | partial |
| short-gpu-compute | `cpu_xform+gpu_compute` | 80.165 | 78.672 | 90.003 | 1.02 | 1.12 | partial |
| short-gpu-compute | `gpu_write+gpu_compute` | 82.466 | 78.672 | 79.520 | 1.05 | 0.96 | poor |
| short-gpu-compute | `gpu_write_raw+gpu_compute` | 81.340 | 78.672 | 79.239 | 1.03 | 0.97 | poor |
| short-gpu-compute | `gpu_write_two+gpu_compute` | 81.938 | 78.672 | 80.996 | 1.04 | 0.99 | poor |
| short-gpu-compute | `gpu_xform+gpu_compute` | 84.933 | 78.672 | 83.688 | 1.08 | 0.99 | poor |
| short-gpu-compute | `gpu_xform_kernels+gpu_compute` | 80.660 | 78.672 | 81.907 | 1.03 | 1.02 | poor |
| short-gpu-compute | `cpu_compute+gpu_compute` | 93.645 | 93.442 | 172.114 | 1.00 | 1.84 | good |
| short-gpu-compute | `disk_load+cpu_xform+gpu_xform+gpu_compute` | 85.261 | 78.672 | 98.205 | 1.08 | 1.15 | partial |

## Pipeline Runs

| run | wall ms | serial ms | speedup | steady items/s |
|---|---:|---:|---:|---:|
| pipeline-slots-1 | 6331.668 | 6312.577 | 1.00 | 3.79 |
| pipeline-slots-2 | 5880.422 | 11238.816 | 1.91 | 4.08 |
| pipeline-slots-4 | 5861.340 | 11586.574 | 1.98 | 4.09 |
| pipeline-slots-8 | 5891.281 | 11547.683 | 1.96 | 4.07 |
