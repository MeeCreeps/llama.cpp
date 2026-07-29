# CPU original 10-minute trace

| Method | ms/token | Direct read MiB/fwd | Direct read ms/fwd | Pipeline wait (s) | Tokens |
|:--|--:|--:|--:|--:|--:|
| Static-Min | 1676.287 | 2363.428 | 941.052 | 153.708 | 359 |
| Static-Max | 296.691 | 0.000 | 0.000 | 0.000 | 2022 |
| MRU | 427.586 | 261.831 | 101.547 | 104.087 | 1400 |
| Offline | 542.121 | 344.240 | 139.241 | 59.746 | 1107 |
| Online | 575.259 | 376.915 | 151.405 | 63.414 | 1044 |
| Diff-before | 638.514 | 384.860 | 165.474 | 76.303 | 941 |
| Diff-now | 575.298 | 383.694 | 154.313 | 75.217 | 1044 |

Fastest: **Static-Max** at 296.691 ms/token.

All rows cover the original 600-second absolute-budget trace, share one token-prefix hash, and pass continuous device-isolation checks.
