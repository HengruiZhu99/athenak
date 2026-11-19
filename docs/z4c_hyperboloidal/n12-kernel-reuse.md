# Stationary geometry and N12 kernel performance

One Aurora node with 12 bound GPU tiles measured **1.757 s/interval**, **19.05 million bulk point-stages/s** and **312.1 s/M₀** for solver commit 40d0aac1a. This is a 1.41× speedup over the 2.47446 s qualification. Regular single-puncture RK4 measured 103.2 million/s, leaving a 5.42× throughput gap. At the production workload of 33,467,904 bulk point-stages, the 2× target requires 1.23723 s/interval and the 3× target requires 0.82482 s/interval.

## Curvature contractions

Job 9104811 tested the candidates on the same node with eight warm and twelve timed intervals. Raising one index of the trace-free curvature once supplies both quadratic contractions. In the inner adjustment, expanding the covariant derivative of the connection constraint cancels the derivatives of its contravariant components. This identity also holds off the constraint surface.

| One-node trial | Seconds/interval |
|---|---:|
| Accepted-state projection at 7dd9c5559 | 1.77855 |
| Share four halo donor windows | 1.77437 |
| Factor the curvature contractions and inner adjustment | **1.75724** |
| Restore reference fields in the derivative kernel | 1.76864 |
| Restore there and split geometric and gauge rates | 1.82976 |

The selected change saved 0.02132 s/interval, against a 0.01360 s range of four-interval means. The checkpoint run repeated at 1.76100 s/interval. At cycle 21, the maximum scaled difference from the original qualification was 3.216×10⁻¹² over 139,683,024 accepted field values and 1.555×10⁻¹⁵ for the eight persistent diagnostics per point. Both comparisons used the same time and timestep, with a tolerance of 10⁻¹¹. An independent CPU comparison of 2,000 off-constraint states gave a maximum scaled RHS difference of 2.23×10⁻¹⁵.

In separate device profiles, the mean ordinary and scri-adjacent bulk point kernels fell from 0.57281 to 0.55562 s/interval per tile. Rank 5 had the largest total device time for the selected solver, at 1.6816 s/interval. On that tile the point kernels used 0.6165 s, surface exchange kernels 0.3682 s, mapped halo interpolation 0.1565 s, stage donor construction 0.1472 s, and field derivative preparation 0.1294 s. These durations are device service on one tile. The timing table measures full uninstrumented intervals. Full receipts are in the n12-halo-kernels-20261119 results archive.

## Gap to regular Z4c at the same bulk work

The archived regular RK4 run (9102783) delivered 103.23 million bulk point-stages/s. At that throughput 33,467,904 point-stages take 0.32419 s, leaving a 1.43304 s gap to the measured 1.75724 s hyperboloidal interval. To compare device work, scale the regular profile (9102879) by 33,467,904 / 5,038,848 and compare it with the hyperboloidal profile (9104811). Both used 12 tiles.

| Contribution | Additional seconds per fixed workload | Share of wall-time gap |
|---|---:|---:|
| Mean bulk RHS device time, including differentiation and dissipation | 0.6062 | 42.3% |
| Difference between busiest and mean tile service | 0.3140 | 21.9% |
| Mean scri exchange and closure | 0.1838 | 12.8% |
| Mean stage donor construction | 0.0944 | 6.6% |
| Mean RK state reconstruction and update | 0.0888 | 6.2% |
| Wall time minus the busiest tile's recorded device service | 0.0739 | 5.2% |
| Mean overlap agreement | 0.0493 | 3.4% |
| Mean halo, device copies and remaining device work | 0.0226 | 1.6% |

At equal bulk work the mean bulk RHS device service was 0.6847 s for the hyperboloidal solver and 0.0785 s for regular Z4c. The 0.3140 s spread term measures uneven total device service across tiles. It includes the concentration of scri and donor work on particular ranks. The 0.0739 s residual compares wall and device clocks from separate runs and cannot distinguish exposed MPI, host synchronization and profiling variation. Data and the plot are archived as athena-accepted-algebra-device-vs-regular under job 9104811.

## Constant compactification in the production mesh

An inventory using the native atlas maps and bulk activity policy reproduced the archived bulk count for each of the 754 blocks. Weighting each block by its actual number of RK stages gives **29,818,368 of 33,467,904 point-stages (89.1%)** on the constant Ω plateau. The share is 93.2% on rank 5, the busiest tile in the factored curvature profile. All 700 plateau blocks lie completely inside the plateau, and the remaining 54 lie completely outside. The production inner gauge weight is one throughout the plateau.

In these cells the prescribed spatial gradient and Hessian of Ω vanish. Stationarity then makes the normal derivative vanish for any evolved shift. In the inner geometric source the derivatives of the connection constraint combine with Ricci, and the inner damping coefficients reduce to their plateau values. A CPU trial evaluated the resulting point equations at 6,000 states, including both sides of the radial cut, general compactification profiles, invalid metrics and chi, nonzero gauge sources and transport. The general and specialized routines accepted the same 5,458 states and rejected the same 542. Across all 31 rates, coordinate speeds and the speed gap, the maximum scaled difference was 1.42×10⁻¹⁴. This check establishes the algebraic comparison. The 1.757 s timing above is for solver 40d0aac1a. The per-block inventory and oracle are in the n12-halo-kernels-20261119 results archive.

The trial that specializes a full local batch also passed the two-rank CPU adaptive fixture. Across 5,928,192 accepted field values after refinement to 32 blocks, its maximum scaled difference from the archived accepted-projection solver was 2.02×10⁻¹⁶. After coarsening to 25 blocks the maximum was 1.70×10⁻¹⁶ across 4,631,400 field values. The eight persistent diagnostics matched exactly at both endpoints. Continuous and restarted runs of the trial produced identical checkpoint payloads.

## Accepted-state projection

An accepted Runge–Kutta record contains scalar field values. The solver now reconstructs those values directly. It checks that the implicit metric component has zero first and second derivatives for constant inputs and that the metric inverse is finite. The chi floor, its persistent history, and the physical lapse check still run at acceptance.

Job 9104779 used 12 bound GPU tiles, 33,467,904 bulk point-stages per interval, eight warm intervals and twelve timed intervals. Each trial started from the same executable source revision, 7cc9d1169. The cycle-21 checkpoint run repeated the selected projection at 1.7664 s/interval.

| One-node trial | Seconds/interval |
|---|---:|
| Previously selected solver | 1.84874 |
| Share donor loads among four halo targets | 1.83677 |
| Store metric geometry between bulk kernels | 1.95648 |
| Reuse projected bulk donors owned by the same stage | 1.84163 |
| Project accepted records from scalar values | **1.76795** |

The accepted projection saved 0.08080 s/interval. The larger range of four-interval means in that pair was 0.00787 s. In separate device profiles, this kernel fell from 0.09760 to 0.02529 s/interval averaged over the 12 tiles. At the same cycle and time, all 139,683,024 accepted field scalars and all eight persistent diagnostics per point matched the previous solver bit for bit. A CPU oracle reproduced the original acceptance decisions for 1,151 valid and 2,849 rejected charts, including poorly conditioned metrics. The two-rank CPU smoke and fixed-partition AMR tests passed. Refinement to 32 blocks and coarsening to 25 both preserved bitwise restart agreement.

Job 9104748 separately tested compact derivative scratch storage. Its 1.86144 s interval versus 1.86396 s for the control fell within the 0.00949 s variation of the timing windows. The checkpoint matched bit for bit. Full receipts and device summaries for both jobs are in the n12-halo-kernels-20261119 results archive.

The following measurements describe the earlier selection through solver commit 601cbe8ea. Its verified two-node result was 1.541 s/interval.

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
