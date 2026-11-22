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

Benchmark inputs, build scripts, timings and digests are in the StellarAI directory `~/photon-ring/smr288-performance-20261121/`. The independent benchmark executable is `athena-fast`. The campaign executables were not replaced.
