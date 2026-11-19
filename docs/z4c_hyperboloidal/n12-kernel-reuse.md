# Stationary geometry and N12 kernel performance

One Aurora node advanced the fixed 33,467,904 bulk point-stages in **1.887 s/interval** with the selected code. The same-job incumbent took 2.547 s, a 1.35× speedup. Against the earlier 2.47446 s qualification, the speedup is 1.31×. Throughput reached 17.73 million bulk point-stages/s and 335.2 s/M₀. Regular single-puncture RK4 previously measured 103.2 million/s, a 5.82× gap. The 2× target relative to the qualified 2.47446 s interval is 1.23723 s, and 3× would require 0.82482 s.

## Kernel changes

The grid already stored the Cartesian reference values, gradients and Hessians, the map Jacobian and its derivatives, and stationary gauge sources. The selected solver also caches the residual-weight derivatives and the prescribed spatial compactification. The normal derivative still uses the current lapse and shift. The Cartesian map has a diagonal Jacobian and a vanishing Hessian. Its finite differences use that structure. The time-dependent spatial metric still determines its own curvature at each stage. Direct contractions avoid forming an intermediate derivative of every Christoffel symbol.

Bulk kernels form the physical tensors from restored independent fields. Ordinary and radial finite differences run field by field, with the original endpoint reconstruction and shift correction on scri-adjacent rows. Dissipation uses the preassembled coefficients of the same weighted fourth-difference operator. Surface cases reuse the restored chart state from their own RK stage and the conformal rescaling of that stage’s metric. Topology changes rebuild the grid-owned caches. All kernels use portable Kokkos and the existing MPI exchange.

## One-node measurements

Each allocation used 12 MPI ranks on 12 bound GPU tiles and the same stable block ownership. The timing runs had output off, eight warm intervals and twelve timed intervals (cycles 9–20). Separate instrumented runs recorded device durations. The qualified incumbent used the same fixed-work RK hierarchy. Repeat and checkpoint runs have separate directories.

| Experiment | Job | Control s/interval | Trial s/interval | Speedup |
|---|---:|---:|---:|---:|
| Cache radial weight and spatial compactification | 9104368 | 2.4547 | 2.3894 | 1.027× |
| Diagonal Cartesian map | 9104400 | 2.4475 | 2.3989 | 1.020× |
| Direct curvature contractions | 9104400 | 2.4475 | 2.3225 | 1.054× |
| Sparse stationary reference components | 9104423 | 2.4602 | 2.3675 | 1.039× |
| Stream fields into physical tensors | 9104423 | 2.4602 | 2.2746 | 1.082× |
| Assemble the stationary dissipation stencil | 9104431 | 2.4712 | 2.4177 | 1.022× |
| Analytic scri compatibility matrix | 9104431 | 2.4712 | 2.4433 | 1.011× |
| Use evolved-connection Ricci on the inner plateau | 9104438 | 2.5830 | 2.5012 | 1.033× |
| Force inline point-equation helpers | 9104455 | 2.4447 | 2.4354 | 1.004× |
| Pass geometry by reference | 9104472 | 2.4442 | 2.4188 | 1.011× |
| Combine cache, affine map and contractions | 9104486 | 2.4551 | 2.2167 | 1.108× |
| Field-parallel radial derivatives | 9104486 | 2.4551 | 2.2509 | 1.091× |
| Parallelize bulk donor reconstruction over fields | 9104512 | 2.4442 | 2.4378 | 1.003× |
| Interpolate stationary donor references once | 9104512 | 2.4442 | 2.4334 | 1.004× |
| Cache the six-by-six Hessian transformation | 9104535 | 2.4384 | 2.4132 | 1.010× |
| Add streamed physical tensors | 9104556 | 2.4600 | 2.1176 | 1.162× |
| Add field-parallel radial rows and the KO stencil | 9104586 | 2.4644 | 1.9924 | 1.237× |
| Reuse the restored scri state and rescale its geometry | 9104586 | 2.4644 | 1.9312 | 1.276× |
| Split long overlap transpose sums | 9104610 | 2.0371 | 1.9846 | 1.026× |
| Reuse the restored scri state and rescale its geometry | 9104653 | 2.5472 | 2.0010 | 1.273× |
| Combine the segmented overlap and scri changes | 9104653 | 2.5472 | 1.8872 | 1.350× |

Job 9104610 used the combined bulk changes as its control. Other rows with the qualified incumbent as control include the effects of all changes present in the trial. The combined rows measure interactions between changes.

## Where the remaining time goes

In the final profiled interval (job 9104653), the busiest of the 12 tiles accumulated 1.826 s of device activity. The average was 1.437 s. On the busiest tile the ordinary point kernel took 0.533 s, the radial-capable point kernel 0.099 s, and fieldwise finite differences 0.132 s. Mapped halo interpolation took 0.167 s, recipient-specific donor reconstruction 0.157 s, and the scri solve and scri RHS took 0.134 s and 0.107 s. The accepted-state reconstruction took 0.108 s. These durations come from a separate instrumented run. Host time spent waiting for device work overlaps these durations. The separate contributions of memory traffic and arithmetic within each kernel remain unresolved.

The original production trace (9103299) attributed 0.574 s to ordinary point evaluation and 0.421 s to the radial-capable point kernel on the busiest tile. Field-parallel radial differentiation moves some work into the finite-difference kernel. In job 9104610, splitting long rows of the overlap transpose reduced the rank-2 transpose/apply time from about 0.181 to 0.045 s relative to the earlier combined-bulk profile. The full wall-time savings were smaller as other tiles entered the critical path.

The selected code still needs a further 0.650 s per interval (34.4% of its one-node wall time) to reach the 2× goal. Even tripling both remaining point-kernel rates would save about 0.42 s on the busiest tile under the same assignment and otherwise unchanged device service. An improvement of that size also requires savings outside point evaluation.

An offline enumeration of the exact production RK coupling rows found zero repeated combinations of base record, slope records and coefficients within any fixed donor/recipient level pairing. Across neighboring recipient levels, donor levels 0 through 4 had 11/12, 25/28, 50/56, 100/112 and 88/96 distinct combinations. Actual reuse would additionally require the same requested donor point. Caching evolved stage values on the basis of stage time alone would change the calculation.

## Correctness and two-node verification

One-node job 9104653 matched the baseline time and timestep at coarse cycle 21. Across 139,683,024 accepted field values, the maximum scaled difference was 3.215e-12 and the RMS was 1.090e-14. The maximum scaled difference in the eight persistent diagnostics per point was 1.443e-15. The acceptance bound was 1e-11.

Two-node job 9104673 used the established 24-rank ownership and measured 1.5413 s/interval, 21.71 million bulk point-stages/s and 273.7 s/M₀. The archived two-node measurements were 2.13944 s with the immediate incumbent and 3.035 s in the earlier scaling campaign. The common-endpoint maximum scaled field and diagnostic differences from the two-node incumbent were 3.215e-12 and 1.443e-15.

The selected one-node and two-node checkpoint payloads were bitwise identical at the common endpoint.

## Validation and provenance

The final native CPU and Aurora SYCL targets rebuilt successfully. On two CPU MPI ranks, the selected smoke and restart cases passed. A real Cartesian AMR refinement to 32 blocks followed by coarsening to 25 rebuilt the plans, and the restarted checkpoint matched uninterrupted evolution bit for bit. The targeted temporal convergence orders were 4.09 and 4.09 in bulk, 4.03 and 4.01 at the radial interface, and 4.21 and 4.19 at scri.

The direct curvature implementation agreed with the original full contraction to 2.23×10⁻¹⁶ in an independent 1,000-metric test. For the retained dissipation code, the CPU oracle checked the assembled support and coefficients from 2,661,120 legacy lines and 7,111,152 rates with a maximum scaled error of 4.21×10⁻¹⁵. The two-scenario surface oracle for the retained scri changes passed, with a maximum scaled surface-RHS error of 7.14×10⁻¹³. The final native stage smoke also checked the overlap energy identity. The historical long science trajectory, Bondi waveforms and clock qualification remain archived evidence.

The repository style command was run once on the final source. The C++ and Python suites still reported existing repository-wide violations. The added and replaced C++ lines since d4ab8ad45 had zero diagnostics, the Python lint diagnostic count matched the preceding campaign, and `git diff --check` passed.

The starting commit was d4ab8ad45. Commits 69d5b05e4, 7a7999688 and 601cbe8ea contain the selected solver. The Aurora binary SHA-256 is b940f1b3a345fd7211c52d9ca4567cf2a98f22ad08d15aec568317929d60ac34. Every changed production source file matched the Aurora build tree by SHA-256. Full timing receipts, inputs, binding records, solver clocks, profile summaries and checkpoint comparisons are archived under research/results/n12-kernel-reuse-20261119 alongside the earlier n12-single-node-deficit-20261118 campaign.
