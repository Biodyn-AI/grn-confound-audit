# Runtime and Memory Benchmarks

## Machine

- **platform**: macOS-26.3-arm64-arm-64bit
- **machine**: arm64
- **processor**: arm
- **python**: 3.9.6
- **cpu_brand**: Apple M2 Pro
- **ram_gb**: 32.0
- **n_cores**: 10

## Per-scale wall-clock and peak RSS

| scale | n_edges | n_cells | sim_time_s | class1_time_s | class2_time_s | class3_time_s | total_audit_time_s | peak_rss_mb | rss_delta_mb |
|---|---|---|---|---|---|---|---|---|---|
| small | 390 | 200 | 0.02 | 2.22 | 2.66 | 2.27 | 7.15 | 164.1 | 11.2 |
| medium | 1526 | 500 | 0.06 | 3.72 | 15.79 | 9.32 | 28.83 | 207.0 | 42.9 |
| large | 4052 | 1000 | 0.18 | 4.9 | 42.73 | 26.26 | 73.89 | 401.6 | 194.6 |

**Notes.** Times above use ``n_null_replicates = 500`` and ``n_permutations = 300``, which are sufficient for the tool-validation analyses in this paper.  For real-data publication use we recommend ``n_null_replicates >= 2000`` with ``null_tail_method='gpd'`` (B = 2000 multiplies the Class 3 time by roughly 4x, but only Class 3 scales with B). All other modules are linear in n_edges and approximately constant in n_cells past a few thousand.
