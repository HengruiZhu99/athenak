# Complete route table

Times are milliseconds: median of each sample's maximum across twelve ranks. Device and staged include their required completion/copy costs; host excludes copies.

| n | Workload | Barrier mode | Device | Host only | Staged | Paired staged/device median | Staged faster rounds |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 16 | volume | barrier | 7.63872 | 8.16913 | 9.13317 | 1.19567 | 1/24 |
| 16 | volume | no_barrier | 8.4545 | 8.66383 | 9.66035 | 1.14715 | 0/24 |
| 16 | surface | barrier | 0.189845 | 0.14048 | 0.213159 | 1.12037 | 2/24 |
| 16 | surface | no_barrier | 0.196977 | 0.150899 | 0.221121 | 1.1215 | 0/24 |
| 16 | rk_mix | barrier | 29.7497 | 33.5923 | 39.0061 | 1.31018 | 0/6 |
| 16 | rk_mix | no_barrier | 30.8528 | 34.7263 | 39.2952 | 1.27609 | 0/6 |
| 24 | volume | barrier | 113.945 | 116.955 | 119.929 | 1.0519 | 0/24 |
| 24 | volume | no_barrier | 115.468 | 118.834 | 120.854 | 1.04633 | 9/24 |
| 24 | surface | barrier | 0.417428 | 0.246117 | 0.403189 | 0.998174 | 13/24 |
| 24 | surface | no_barrier | 0.438585 | 0.293598 | 0.417897 | 0.95528 | 21/24 |
| 24 | rk_mix | barrier | 455.479 | 469.082 | 480.785 | 1.05803 | 0/6 |
| 24 | rk_mix | no_barrier | 458.132 | 472.061 | 482.552 | 1.05338 | 0/6 |
| 32 | volume | barrier | 271.499 | 280.713 | 284.367 | 1.0475 | 0/24 |
| 32 | volume | no_barrier | 274.476 | 283.44 | 287.137 | 1.04578 | 0/24 |
| 32 | surface | barrier | 0.642599 | 0.477579 | 0.63126 | 1.02064 | 11/24 |
| 32 | surface | no_barrier | 0.705649 | 0.539191 | 0.724499 | 1.01299 | 8/24 |
| 32 | rk_mix | barrier | 1094.39 | 1124.96 | 1146.62 | 1.04767 | 0/6 |
| 32 | rk_mix | no_barrier | 1097.82 | 1128.04 | 1148.2 | 1.04605 | 0/6 |
