# StellarAI SMR transport timings

On November 21, 2026, matched restarts near two initial ADM masses measured the cost of the 76-block reflected 288-unit photon-ring mesh. Each block has 36 × 36 active cells, each cell has 24 × 48 angular bins, and each SSPRK(10,4) timestep has ten stages. At this epoch the timestep is about 0.00126 code units for an ADM mass near 5. This gives roughly 4,000 timesteps per ADM mass.

The baseline was the centered-search executable, SHA-256 `058e4bbfc216a11677776d451733deb2e9a2f887d01b5a5eec0b9e62c6d48f12`. The candidate was `3edc7d06cfe5eccedabffa61b20a35b14d042fcc53e01c420fceeedc998d0103`. Both used the same CUDA/Kokkos/MPI build and precision. The candidate tiles the angular-characteristic and flux-budget kernels for the actual array layout and accumulates escaped energy, escaped angular momentum, and metric work in one Kokkos reduction.

## Kernel measurements

GPU event timings on rank zero of eight B200s covered ten evolution steps from the same 0.9508 restart in jobs 58945 and 58960. Percentages use the sum of named kernel timings on the sampled rank. The production benchmarks ran without the timer.

| Operation | Baseline share | Baseline ms | Optimized ms |
|---|---:|---:|---:|
| Angular characteristics | 33.0% | 926 | 228 |
| Phase-space fluxes | 13.8% | 386 | 386 |
| Outgoing flux budgets | 13.4% | 375 | 285 |
| Integrated sources and boundary fluxes | 10.6% | 297 | 101 |
| Cell moments | 5.5% | 155 | 156 |

In the original 0.9508 production segment from 1 to 2 M, the node was allocated for about 1,499 seconds through publication of the reduced checkpoint. The driver reported 1,376 seconds of evolution, 21.7 seconds of output timers and 6.85 seconds in the online horizon finder. These timers have different scopes and should not be added. The independent saved-slice analysis uses CPU time inside the GPU allocation.

## Measured scaling

One MPI rank per B200. The 42-step runs below used terminal output and disabled the online search during the measured interval. Seconds per step use the uninstrumented driver clock between the sixth and last logged cycles. Within each job both executables ran on the same physical GPUs; the eight-GPU comparison ran in baseline/candidate/candidate/baseline order.

| Job | B200s | Baseline s/step | Optimized s/step | Optimized GPU-s/step |
|---|---:|---:|---:|---:|
| 58958 | 1 | 1.801 | 1.146 | 1.146 |
| 58954 | 2 | 0.995 | 0.637 | 1.273 |
| 58953 | 4 | 0.577 | 0.364 | 1.458 |
| 58948 | 8 | 0.338 | 0.236 | 1.889 |

At production cadence on eight GPUs, job 58960 advanced 0.9508 for 240 steps with shared-filesystem output, passing two scheduled horizon searches. Step time fell from 0.3701 to 0.2703 seconds (27.0%). Job 58967 advanced 1.0054 for 60 steps, passing one scheduled search, and measured 0.3411 versus 0.2387 seconds (30.0%).

For both production-cadence comparisons, SHA-256 matched for the complete native double-precision checkpoint payload, every saved ADM, radiation and constraint field, and both the marginal and interior horizon logs. The final integrated photon history values also matched exactly. These searches occurred before verified trapping. The five local Boltzmann regression tests passed.

An indicative eight-GPU budget is 72–75 GPU-hours for 30 M if early-time costs persist. This extrapolates the measured 1–2 M baseline allocation at each spin, scales the evolution portion by its production-cadence speed ratio, and carries the remaining allocation overhead unchanged. Late-time timestep changes and offline verification cost remain unmeasured for a full 30 M run of the candidate.

Benchmark inputs, build scripts, timings and digests are in the StellarAI directory `~/photon-ring/smr288-performance-20261121/`. The independent benchmark executable is `athena-fast`. At the end of this first benchmark, production still used the original executable.

## Second round of eight-GPU restart benchmarks

Two redundant operations were removed. Z4c uses the stage stress for its RHS and the final accepted stress for constraints and output. Rebuilding accepted stress at the end of an intermediate RK stage discarded the result before the next RHS. The radiation timestep also uses the accepted metric. Intermediate stages now compute the inverse donor budget for every phase cell and evaluate the volume source directly. Only the accepted metric requires the global maximum for the next timestep. The fixed-metric path retains its cache and source-free shortcut.

The following three comparisons use the same GPUs for each executable pair, shared-filesystem output, and the production horizon-search cadence. The mean includes scheduled diagnostics and the terminal output, measured between the sixth and last logged cycles. The median is the typical elapsed time between individual cycles on that interval. Output timing varies between allocations.

| Initial ADM spin | Grid and reference executable | Steps | Mean seconds/step, reference → revised | Median seconds/step, reference → revised | Mean time saved | Slurm job |
|---|---|---:|---:|---:|---:|---:|
| 0.9508 | SMR, first optimized build | 240 | 0.2551 → 0.2155 | 0.2401 → 0.2002 | 15.5% | 59028 |
| 1.0054 | SMR, first optimized build | 120 | 0.2946 → 0.2298 | 0.2405 → 0.2005 | 22.0% | 59031 |
| 1.0600 | AMR, original wide-run build | 150 | 0.3950 → 0.2490 | 0.3433 → 0.2023 | 37.0% | 59032 |

For each pair the native checkpoint payload, every saved radiation, ADM and constraint field, the terminal photon history, the marginal and interior horizon logs and the sequence of logged timesteps matched exactly. The SMR replays started near 2 initial ADM masses. The AMR replay started at cycle 36000, near 11.6 initial ADM masses, and preserved its three-branch horizon search. These are finite-interval comparisons. The five local Boltzmann regression tests passed, including the equatorial AMR test.

Over ten timed steps on rank zero, radiation moment evaluations fell from 200 to 110 and from 156 to 86 ms. The donor-budget and CFL kernels together fell from 285 to 69 ms. The flux reconstruction remained the largest named kernel at 385 ms. Specializing the five flux directions reduced this to about 250 ms, but altered native double-precision rounding. In the 240-step 0.9508 replay, the maximum absolute radiation difference was 4.44 × 10⁻¹⁶ and the relative radiation L1 difference was 9.0 × 10⁻¹⁹. The campaign uses the original reconstruction arithmetic.

Paired logs, digests, GPU timers, build commands and run scripts are under `~/photon-ring/smr288-performance-20261121/followup/` on StellarAI. The comparisons above use `trial-c/`. Run `python3 compare.py trial-c/job-59028 trial-c/job-59031 trial-c/job-59032` from that directory to reproduce the table and verify the hashes.

### Production executable and continuation

The final executable also avoids an integer remainder for the common one- to four-bin shift on the periodic angular grid. Against the revised build above, the median step fell by another 1.2–1.5% across jobs 59036, 59039 and 59042. Over ten profiled steps, phase-space fluxes took 361 ms. All three native replays again matched exactly. The shortcut includes a general periodic wrap, tested on grids of 1, 6 and 48 sectors.

A final direct comparison ran the production candidate first in each allocation, reversing the earlier launch order. The reference for the SMR cases was the first optimized executable. The AMR reference was its running production binary. All three pairs matched on every diagnostic and the complete native checkpoint payload.

| Initial spin | Steps | Mean seconds/step, reference → production | Median seconds/step, reference → production | Mean time saved | Slurm job |
|---|---:|---:|---:|---:|---:|
| 0.9508 | 240 | 0.2562 → 0.2126 | 0.2405 → 0.1974 | 17.0% | 59051 |
| 1.0054 | 120 | 0.2679 → 0.2252 | 0.2397 → 0.1969 | 15.9% | 59052 |
| 1.0600 | 150 | 0.3674 → 0.2228 | 0.3428 → 0.1979 | 39.4% | 59053 |

The [paired benchmark timings](smr-followup-timings.csv) include all fifteen eight-GPU allocations and distinguish native checkpoint equality from agreement in the reduced outputs.

The production SHA-256 is `cd1e96485827bd4f80803ce662e41deb605bbe8f8896e33734c02e2af82602d3`. SMR workers 58731 and 58749 resolve the executable afresh on each segment. The switch was set while both were integrating segment 4, for their restart at 4 initial ADM masses. For adaptive 1.0600 the eight-GPU chain is 58150 to 20 M, 59045 to 40 M, and CPU handoff 58173 to submit the final eight-GPU segment to 60 M. The release record pins the previous and target executable digests and the continuation checks the parent ring hash. The centered-only SMR search and the AMR three-branch search retain their respective input settings.

For budgeting, the original 2–3 M SMR segments took 1,505 and 1,516 seconds between checkpoint publications, with 3,961 and 3,965 steps. Subtracting the measured saving in median solver time per step from these allocations gives approximately **62–66 GPU-hours per 30 M** on eight B200s, if the timestep, mesh, diagnostics and offline analysis costs remain comparable. A complete 30 M segment with the new binary has yet to be timed.

## Boundary exchange and transport on later checkpoints

The reference for all three comparisons was the preceding eight-GPU production release, SHA-256 `cd1e96485827bd4f80803ce662e41deb605bbe8f8896e33734c02e2af82602d3`. The new release is `6b8a73bcd4bff4687cfdbff28221cf95733e38b27e14ba39a10113f87576f610`. Each pair used the same restart, node, eight B200s, and input settings. Each executable advanced 240 steps with the campaign output and horizon cadence. The candidate ran first for 0.9508 and 1.0600, and second for 1.0054. Means use the elapsed driver clock from the sixth logged cycle through terminal output. Short intervals include shared-filesystem timing variation.

| Initial spin | Grid | Restart cycle | Mean s/step, previous → new | Mean time saved | Median s/step, previous → new | Slurm job |
|---|---|---:|---:|---:|---:|---:|
| 0.9508 | SMR | 15427 | 0.2275 → 0.2170 | 4.62% | 0.1973 → 0.1885 | 59119 |
| 1.0054 | SMR | 11737 | 0.2264 → 0.2197 | 2.96% | 0.1980 → 0.1905 | 59120 |
| 1.0600 | AMR | 38400 | 0.2305 → 0.2215 | 3.91% | 0.1987 → 0.1903 | 59121 |

The cell-centered exchange skips two launches that only append coarse data for Z4c. The primary halo pack and unpack distribute contiguous buffer entries across Kokkos team threads. Photon prolongation bypasses the Z4c-only counter allocations, and the outflow kernel visits the three strips that can contain outflow ghosts. Transport selects the speed field once per flux launch and uses the triangular structure of the photon frame in characteristic calculations. On the sampled rank of the 4 M replay, twelve steps reduced named send/receive kernel time from 139.8 to 90.7 ms and outflow-boundary time from 41.2 to 20.7 ms. Phase-space flux reconstruction remained the largest named kernel at 425.7 ms. Flux tiling and fusing the RK update with the diagnostic reduction both increased step time in separate trials. The [paired measurements](later-checkpoint-timings.csv) include those trials.

For every production pair, the native double-precision checkpoint payload, all saved radiation, ADM and constraint slices, terminal photon history, horizon and interior logs, and logged timesteps matched. Nine local tests passed, including equatorial AMR, dynamical Cartoon AMR on two MPI ranks, and Z4c interpolation and coarse-cache tests. A separate two-rank 3D hydrodynamic linear wave on a static mesh compared the previous and revised cell-centered exchange after six steps. The complete 13,830,825-byte restart matched. Inputs, build and replay scripts, timers, and output digests for the GPU comparisons are in `~/photon-ring/smr288-performance-20261121/later/` on StellarAI. From there, `python3 compare.py release/job-59119 release/job-59120 release/job-59121` reproduces the production comparisons.

The 0.9508 worker loaded this executable at 5 M in allocation 58731. The 1.0054 worker 58749 selects the same binary on its next restart. Job 59045 has the new release pinned for the 1.0600 continuation at 20 M, with the dependent handoff to 60 M intact.

## Lean Z4c synchronization and surface diagnostics

The production binary with SHA-256 `6b8a73bcd4bff4687cfdbff28221cf95733e38b27e14ba39a10113f87576f610` supports sparse device-side shared-vertex synchronization. The 288-unit campaign now selects it for both static and adaptive grids. The [runtime input](../../../inputs/boltzmann/photon_ring_lean.in) also selects admissibility checks when the solver consumes or accepts a state, disables intermediate axis audit logging and the shared-vertex postcondition scan, and retains the timestep contract diagnostic. Explicit settings override the values recorded in older restarts.

| Checkpoint | Eight-B200 Slurm job | Ordinary seconds/step, baseline → lean | Driver seconds/step including terminal output, baseline → lean |
|---|---:|---:|---:|
| SMR 0.9508, cycle 15427 | 59194 | 0.18954 → 0.13605 (28.2% saved) | 0.21131 → 0.15864 (24.9% saved) |
| AMR 1.0600, cycle 38400 | 59195 | 0.19084 → 0.13808 (27.6% saved) | 0.20709 → 0.15462 (25.3% saved) |

Each row averages two 240-step replays per mode in baseline/lean/lean/baseline order on one node. Ordinary timings omit the first five intervals and intervals over 1.15 times the median. The native checkpoint payloads, all saved radiation, geometry and constraint arrays, complete history files, horizon and interior logs, timestep contract and logged timesteps matched. Lean mode omits the axis regularity audit CSV. In the review's single-selector tests, switching the host-staged global shared-vertex exchange gave the largest reduction. Raw ABBA logs, digests and the comparison script are in `~/photon-ring/smr288-performance-20261121/review-20261121/` on StellarAI.

A separate two-B200 test (59289) used coupled reflected data with deterministic refinement. The two modes agreed on all six checkpoints starting before refinement and all five starting from the refined mesh. The first pair created three leaves, transferred blocks between ranks and deleted three leaves. The second pair resumed on five leaves and derefined. Both histories and all compared native geometry and photon payloads matched exactly. The local `boltzmann_equatorial_amr` regression now includes a paired restart across this derefinement.

The offline surface fitter batches the five displaced normal evaluations on the small least-squares grids and reuses their pole values for the regular axis limit. On four saved slices from 0.9508 and 1.0054, both the order-40 and order-48 fits produced byte-identical JSON including the coefficients, three interpolation checks and trapped classifications. In CPU job 59296 the combined wall time for the two serial fit commands was 7.05 and 6.72 seconds with the old code and 5.42 and 5.44 seconds with the new code. These timings include Python startup. The worker still finishes the whole-surface checks before publishing its checkpoint and classifying a segment.

Use [the static production deck](../../../inputs/boltzmann/smr288_refined_ring.athinput) or [the adaptive production deck](../../../inputs/boltzmann/refined_ring_288.athinput) for fresh starts. On SMR restarts the worker's `finder-restart.in` combines the centered-only finder input with the seven Z4c selectors. For a release-controlled adaptive restart, place a copy of `photon_ring_lean.in` as `runtime.in` alongside `executable-release.txt`. The rank launcher reads that input when the next segment starts, including segments whose batch script Slurm has already spooled. A direct AthenaK restart can use `-r CHECKPOINT -i photon_ring_lean.in`. The adaptive finder retains the branches in its checkpoint.

In SMR production job 58749 the 9–10 M segment took 608.92 s for 2,649 steps (1.353 GPU-hours), followed by 461.45 s for 2,683 steps over 10–11 M (1.025 GPU-hours) with the lean settings. The elapsed interval is between successive published state files, including launch, checkpoint writes and witness checks. This is a 24.2% reduction in GPU-hours per M across adjacent physical intervals. The same-checkpoint ABBA comparisons above isolate the solver change.

Job 59311 exercised the adaptive campaign release launcher with its deployed input overlay. The same two restart/regrid comparisons again matched on all 11 native checkpoint payloads. Both SMR workers independently recorded all seven selectors in their binary slice headers, starting at 10 M for 1.0054 and 2 M for 1.0600.

An experimental Kokkos kernel distributed the ten moment sums over separate threads while preserving the per-angle accumulation order and compensated energy sum. On the 1.0054 checkpoint (cycle 11737), four-B200 job 59308 ran 60-step ABBA replays with lean settings. All saved state and diagnostic digests and all logged timesteps matched. Mean ordinary step time increased from 0.8370 to 0.8511 s (1.7%) in this four-rank configuration. Production continues to use the existing moment kernel. Its eight-rank performance with the experimental layout remains unmeasured.
