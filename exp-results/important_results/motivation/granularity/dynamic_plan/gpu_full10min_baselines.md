# GPU original 10-minute trace

| Method | ms/token | Direct read MiB/fwd | Direct read ms/fwd | Pipeline wait (s) | Tokens |
|:--|--:|--:|--:|--:|--:|
| Static-Min | 1500.059 | 215.071 | 132.309 | 4.890 | 401 |
| Static-Max | 388.743 | 0.000 | 0.000 | 0.000 | 1541 |
| MRU | 843.610 | 180.380 | 94.723 | 9.364 | 707 |
| Offline | 760.417 | 30.526 | 39.990 | 3.089 | 788 |
| Online | 804.248 | 25.133 | 36.307 | 5.450 | 746 |
| Diff-before | 788.938 | 24.971 | 39.441 | 23.221 | 761 |
| Diff-now | 637.170 | 21.318 | 30.220 | 11.706 | 942 |

Fastest: **Static-Max** at 388.743 ms/token.

All rows cover the original 600-second absolute-budget trace, share one token-prefix hash, and pass continuous device-isolation checks.
