## Sandbox security

| Attack category | n | blocked by guard | blocked by sandbox (guard bypassed) | blocked by either |
|---|---|---|---|---|
| connection | 10 | 10/10 | 10/10 | 10/10 |
| ddl | 8 | 8/8 | 8/8 | 8/8 |
| dml | 5 | 5/5 | 5/5 | 5/5 |
| function | 5 | 5/5 | 5/5 | 5/5 |
| hidden_write | 3 | 3/3 | 3/3 | 3/3 |
| obfuscated | 4 | 4/4 | 4/4 | 4/4 |
| stacked | 3 | 3/3 | 3/3 | 3/3 |
| **all attacks** | 38 | 100% | 100% | **100%** |

Benign queries wrongly blocked (58, including every gold query): guard 0%, sandbox 0%. Database file unchanged after all attacks: **yes**.
