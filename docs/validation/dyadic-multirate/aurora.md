# Aurora GPU qualification

Aurora debug job 9100106 used Intel Data Center GPU Max devices with one MPI rank per tile and two tiles per physical GPU. Twelve GPUs occupy two nodes. All runs started from fresh data, used double precision, and used the same spatial operator. Each fixed-size pair covered the same physical duration. Uniform EX5 uses the smallest local timestep throughout. The elapsed column includes initialization, migration, diagnostics and checkpoint output. The steady column is the median driver time between common endpoints after two warmup intervals.

| Initial blocks | Regridded blocks | Physical GPUs | Method and placement | Common intervals | Elapsed (s) | Steady (s/interval) |
| ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 88 | 102 | 1 | Multirate, stage-aware METIS | 8 | 132.15 | 6.990 |
| 88 | 102 | 1 | Uniform EX5 | 8 | 110.18 | 8.579 |
| 88 | 102 | 12 | Multirate, stage-aware METIS | 8 | 93.50 | 7.366 |
| 88 | 102 | 12 | Uniform EX5 | 8 | 83.88 | 8.587 |
| 754 | fixed | 12 | Multirate, stage-aware METIS | 6 | 558.43 | 70.108 |
| 754 | fixed | 12 | Multirate, scalar weighting | 6 | 738.97 | 114.158 |
| 754 | fixed | 12 | Uniform EX5 | 6 | 994.95 | 155.984 |

The 754-block binary atlas has 250 agreement participants. The timestep assignment promotes 176 Cartesian leaves. The level-1 through level-4 RHS counts per interval are 368, 5,632, 10,624 and 1,536. On this mesh the initial normalized stability monitor reached 0.229 at scri and 0.084 in the Cartesian region. Scri set the larger monitored bound. Its coefficients and closure were held fixed.

On the 754-block mesh, stage-aware METIS achieved 2.22× the throughput of uniform EX5 after warmup and 1.78× over the complete short run. The stage-aware placement used 19.04 GB of MPI payload per interval compared with 41.51 GB for scalar weighting and completed steady intervals 1.63× faster. Both schedules evaluated the same 18,160 block RHS stages per interval. The final field comparison covered 139,683,024 field scalars and gave a maximum scaled difference of 7.90e-9 and an RMS of 4.84e-11. Scaling a difference uses the larger of unity and the absolute values of the two operands.

The smaller adaptive problem had a steady strong-scaling speedup of 0.949× from one to twelve GPUs, equivalent to 7.9% parallel efficiency. The accepted native checkpoints, including persistent diagnostics, were bitwise identical across these GPU counts. Each multirate run differed from its fine uniform control by at most 9.19e-15 over 18,896,112 field scalars.

Separate adaptive restart controls on one and twelve GPUs split a Cartesian parent at the first common endpoint, verified coarsening of eight siblings, and compared continuous trajectories with runs restarted before and after refinement. All 7,458,048 payload scalars agreed bitwise for each restart check. The stationary adaptive control differed from fine uniform EX5 by at most 1.78e-15.

The job writes per-run receipts containing executable and input SHA-256, physical-device binding, post-warmup samples, time per simulated mass, MPI bytes, batch counts, GPU memory and spatial regrid events. The benchmark comparison checks native topology and atlas metadata before computing normalized field differences. The tested interval is short. Long-time binary evolution and puncture-centred spatial transfer remain separate qualification gates.

## Receive scheduling and finest-level placement

Job 9100147 used two nodes and 24 tile ranks. The first two controls below replay the same block owners from the original METIS run. The spread trial builds a new weighted partition. Each mode evaluates the same 18,160 block RHS stages per interval.

| Partition and transport | Physical GPUs with finest blocks | Elapsed over six intervals (s) | Steady (s/interval) | MPI GB/interval | Poll wait at interval 5 (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Saved METIS, polled receives | 8 | 444.12 | 62.260 | 19.04 | 46.28 |
| Saved METIS, preposted receives | 8 | 287.32 | 40.216 | 15.03 | 18.00 |
| Weighted spread, preposted receives | 12 | 323.17 | 33.499 | 14.79 | 20.69 |

All 24 tile ranks received one finest-level leaf in the spread trial. Preposting reduced the steady interval time by 35.4% at fixed placement. It increased peak per-rank packet storage in interval 5 from 88.6 MB to 172.0 MB. On the spread trial, polls immediately following independent interior batches observed 11,451 record completions during that interval. This count measures observable communication progress during the interior phase. It does not measure concurrent hardware execution time. Replaying an existing partition avoids the fresh partitioning and migration cost in the spread trial.

The uniform EX5 figures above use the multirate scheduler in single-rate mode. The production N12 classical RK4 control follows its usual native task lists. These comparisons answer different performance questions.
