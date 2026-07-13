# GPU Transform vs CPU Compute Summary

| case | shape | q4_0 MiB | q/d write + CPU compute extra | full GPU xform + CPU compute extra | GPU kernels + CPU compute extra | full GPU xform + GPU compute extra | GPU kernels + GPU compute extra |
|---|---:|---:|---:|---:|---:|---:|---:|
| `attn-4096x4096-q4_0` | `4096x4096` | 9.00 | 1.663 | 1.751 | 1.775 | 3.538 | 1.386 |
| `ffn-4096x14336-q4_0` | `4096x14336` | 31.50 | 0.185 | 0.528 | 0.363 | 13.182 | 4.707 |
