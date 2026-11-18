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

## Deep adaptive GPU scaling

On the binary mesh, the same off-puncture location criterion created 49 blocks at coarse endpoint 1 and 70 more at endpoint 2. The two decompositions reached 873 blocks and the same endpoint after five coarse intervals. All accepted native fields and persistent diagnostics agreed bitwise. The field-only comparison covered 161,728,488 scalar slots.

| Physical GPUs | Tile ranks | Total time, including both regrids (s) | Steady time per interval on 873 blocks (s) |
| ---: | ---: | ---: | ---: |
| 1 | 2 | 1,342.88 | 66.991 |
| 12 | 24 | 519.91 | 41.767 |

The weighted spread placement achieved 2.58× end-to-end speedup and 1.60× steady speedup from one GPU to two nodes. Steady parallel efficiency was 13.4%. On the last post-refinement mesh the schedule required 21,232 block RHS evaluations per interval. The experiment verifies the deep adaptive GPU trajectory and exposes a strong-scaling limitation at this workload.

Three archived native classical RK4 segments on the same 754-block atlas and two-node allocation covered 0–1.408, 1.408–3.828 and 3.828–6.213 M0. Their respective measured elapsed costs were 1,281, 1,255 and 1,285 s/M0. At 5,947 steady s/M0, the short spread multirate benchmark is slower than that historical native execution path. Executable, run length and output cadence differ. The same-build native control and the isolated 10 M0 trajectories provide the matched comparisons.

## Matched native uniform RK4 control and transport regression

Each pair below used one executable, 754 blocks, the same N12 initial data, six common intervals and 12 physical GPUs. Native uniform RK4 took 16 fine steps per common interval. The multirate cohort replayed the same block placement with all 24 fastest leaves distributed across the 24 tiles.

| Aurora job | Multirate implementation | Multirate steady s/M0 | Native uniform RK4 steady s/M0 | Multirate/native time |
| --- | --- | ---: | ---: | ---: |
| 9100244 | Per-peer GPU sends | 6,135 | 1,241 | 4.94 |
| 9100343 | Fused GPU publication and batched MPI completion polling | 20,052 | 1,222 | 16.41 |

Both multirate cohorts issued 5,920 local batches, evaluated 18,160 block RHS stages and sent 14.79 GB per common interval. Native uniform RK4 evaluated 48,256 block stages. At interval 4 the maximum time spent polling when no local work was ready rose from 25.47 to 96.39 seconds. Completions observed immediately following interior batches fell from 7,258 to 1,741. Poll time includes waiting for work on other ranks. These timings cannot distinguish driver registration, device copies, network transfers and delayed producing ranks. The two allocations used different nodes. Both all-engine and compute-engine counters returned zero across the fused run, including the native RK4 control, so those counters give no usable utilization estimate.

On the earlier executable, the contiguous placement took 5,042 steady s/M0, 18% less time than the fully spread cohort, while sending 13.66 GB per interval. The contiguous run also used eager batches, so placement and batching effects are coupled in that comparison. The optimized uniform-timestep baseline is materially faster on this mesh.

Job 9100791 replayed both archived executables and a corrected build on the same two nodes. Over five matched coarse intervals, the fully spread layout took 6,033 steady s/M0 on the original per-peer build, 20,161 on the fused build and 6,036 on the corrected build. The correction restores per-peer device buffers and MPI polling, limits halo and donor reconstruction to active recipients, and checks the characteristic CFL in the existing RHS validity reduction. Endpoints across all three builds agreed bitwise, including the persistent diagnostic ledgers. The corrected build recovered 3.34× throughput over the regression and matched the original throughput to 0.1%. Its own native uniform RK4 control took 1,212 steady s/M0, leaving a 4.98× elapsed-time gap with the fully spread layout.

On a two-rank CPU nonlinear pulse with 25 blocks refining to 32, the corrected complete-batch path, including bounded send-buffer reuse, reproduced all 7,458,048 checkpoint scalars bitwise. With floors inactive and the time interval held fixed, successive timestep halvings gave observed orders of 4.14–4.16 in the bulk, 4.01–4.03 at the radial interface, 4.19–4.21 at scri, 4.22–4.28 for the full regularity tensor and 4.19–4.21 for the clock-source mismatch.

## Immutable slope traffic and buffer reuse

Job 9100793 compared the same 754 blocks and spread owners on 12 GPUs. All methods covered six coarse intervals. Steady cost is the median of four post-warmup driver intervals and includes endpoint work. The native classical RK4 control advanced 16 uniform steps per coarse interval.

| Method | Block RHS evaluations per interval | Steady seconds per simulated M0 | Allocated GPU-hours per M0 |
| --- | ---: | ---: | ---: |
| Multirate, 26 independent slopes | 18,160 | 5,571 | 18.57 |
| Same multirate schedule, 96 MiB send cache per tile rank | 18,160 | 3,398 | 11.33 |
| Native uniform RK4 | 48,256 | 1,219 | 4.06 |

Both multirate cases sent 12.98 GB in 81,522 packets per interval. With reuse enabled, 81,404 allocations came from the cache in interval five. Median time per rank with no ready batch fell from 18.72 to 11.89 seconds. Median dependent RHS time stayed at 1.41 seconds. Buffer reuse therefore improved steady throughput by 1.64× at fixed work and traffic. The resulting elapsed cost was still 2.79× native uniform. Allocated GPU-hours include waiting and idle time.

A separate two-rank CPU test used alternating A/B/B/A trials to isolate the mapped-halo stencil change. Traversing the stencil once for the physical and residual interpolants reduced multirate halo cost from 1.767 to 1.201 summed kernel-seconds per coarse interval and multirate total kernel cost from 5.658 to 5.085. On the same code change, native RK4 went from 8.955 to 8.108 summed kernel-seconds over the matching interval. These are CPU measurements on the small fixture.

For the complete local batch path, host-to-device donor coupling tables now contain only the distinct donor blocks actually read by the batch. Across two CPU ranks the logical transfer fell from 96,000 to 59,520 bytes per interval before refinement and from 122,880 to 69,120 after refinement. Coalescing records produced by the same ready stage batch reduced MPI messages from 194 to 54 before refinement and from 250 to 54 after refinement. It preserved the byte volume and all 7,458,048 checkpoint scalars, including diagnostics. These message counts are from the adaptive pulse. N12 GPU timings select the transport for the isolated science run.

### Measured GPU execution

Level Zero device timing on all 24 MPI tile ranks covered two coarse intervals after two warmups. The collection window surrounded integrator tasks. Throughput above comes from separate uninstrumented runs. The profile sums active command durations across ranks. GPU-hours above use the number of allocated physical GPUs and elapsed wall time.

| Summed tile-seconds over two intervals | Multirate with send reuse | Native uniform RK4 |
| --- | ---: | ---: |
| Compute kernels | 147.76 | 132.38 |
| Driver copy and fill commands | 10.21 | 12.20 |
| All timed device commands | 157.97 | 144.58 |
| Scheme A RHS kernels, included in compute | 97.44 | 108.74 |
| Multirate owned stage reconstruction | 18.46 | — |
| Multirate recipient-specific donor reconstruction | 16.93 | — |

The multirate schedule reduced block RHS evaluations by 62.4% and GPU time in the RHS function by 10.4%. Each complete multirate stage batch averaged 6.14 blocks, compared with 31.42 for a native uniform rank-stage. Both methods split interior and halo-dependent points. Multirate issued 35,520 RHS-related kernel calls against 18,432 in native RK4 over the trace window. The two multirate reconstruction functions together consumed 35.39 tile-seconds. Total compute kernel time was 11.6% higher. Level Zero also recorded 2,452,192 device-to-host copies and the same number of host-to-device copies during multirate. Neither command category appeared in the native trace. The trace identifies the commands and durations but does not attribute each copy to an MPI or application call site.


The scalar stage specialization keeps the full jet path for accepted geometry checks and scri donor compatibility. On the two-rank CPU fixture, alternating A/B/B/A measurements reduced donor reconstruction from 0.2495 to 0.0954 summed kernel-seconds per coarse interval and owned-stage reconstruction from 0.1679 to 0.0985. The 25-to-32-block nonlinear pulse again matched all 7,458,048 stored values bitwise.
