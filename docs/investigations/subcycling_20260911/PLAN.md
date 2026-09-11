# VC Cartoon time-subcycling: review plan

Status: proposal only. No evolution source changes, builds, test jobs, or campaign changes made. Read-only analysis scripts and this document were created locally.

## Evidence and preliminary savings

Target: N128 A=-0.048875, provisionally classified collapse by user override, not a verified physical collapse. Remote lineage: /pscratch/sd/h/hzhu/n128-chite-bisection-20260910/cycle_03_recovery_24000. Executable provenance from previous campaign audit: source c802bcaf77f143cf3e83ce267cc93d5f59f2daa7; executable SHA256 fd859ee500d8c20437af81aa4b25be4957714c3d9c134ad5eb4fcbd39f983a2b. Reauthenticate before future tests.

Read actual AMR event trees and reconstruct all unique covered ancestors. Physical level = recorded logical level minus root level 3. All leaf blocks have 16x16 intervals. estimate.json records selected events and every level count; estimate.py reproduces the read-only remote calculation.

For maximum coarse/fine timestep ratio R, weight each block at level l by 2^(-min(L-l,log2 R)). Divide synchronous leaf-block work by weighted leaf-plus-parent work. Assumes equal cost per block, spatial factor-two allowed timesteps, unchanged hierarchy, and one predictor evolution per parent step. Excludes extra temporal-boundary work, stage-storage costs, occupancy loss, global gauge iterations, AMR, diagnostics and I/O. Additional stencil-support parents may be necessary. These are model work ratios, not benchmarked speedups or guaranteed bounds.

| Actual AMR event time | Leaf blocks | Finest physical level | R=16 | R=32 | Unbounded ratio |
|---:|---:|---:|---:|---:|---:|
| 50.556569 | 500 | 9 | 3.69 | 3.94 | 4.13 |
| 60.007922 | 1358 | 12 | 4.03 | 4.30 | 4.47 |
| 62.034569 | 3434 | 14 | 4.41 | 4.68 | 4.84 |
| 62.177782 | 6392 | 17 | 6.43 | 7.45 | 8.21 |
| 62.200215 | 7682 | 18 | 8.56 | 10.65 | 12.55 |
| 62.205996 | 9170 | 18 | 6.17 | 7.01 | 7.65 |

Last tree has 24 finest leaves, 730 on level17, 936 on level16, and 2416 on level15. All-ancestor predictor count is3046, approximately33% more blocks before stage-array and boundary-support memory. The final tree is slightly later than the last saved history sample; it is a recorded event, not asserted to be the last complete checkpoint.

The last timestep-contract row gives dt_spatial=9.69306e-6, dt_source=6.88970e-4, dt_final=2.42327e-6, limiter=z4c_spatial_or_other. CFL=.25. Thus spatial restriction presently dominates; even conservatively applying CFL to the reported source limit leaves about71 finest steps of headroom. This supports investigating ratios16/32 at this snapshot, but does not establish stability on every parent level or throughout evolution. At t62.0346 analogous headroom is only about16.45; source limits can cap useful ratios earlier.

Wall-time scenarios: S_wall=1/[f+(1-f)/S_work+h], where f is unchanged fraction of old wall time and h is new overhead relative to old wall time. At S_work=7 and h=.05: f=.2 gives3.06x, f=.4 gives1.87x, f=.6 gives1.41x. These fractions are hypothetical, not profiling results. Use2-4x only as an engineering target to test, not a prediction; speedup below1 remains possible. A representative8 GPU-hour segment would become2-4 GPU-hours at2-4x on the same single GPU. Do not extrapolate this to the entire run, queue wait, or reaching t200. Refinement continues to shrink the finest timestep.

## Proposed execution sequence, after approval

1. Measure the actual bottleneck before scheduler implementation.
   - Isolated single-GPU short restarts at existing t~62.0346 and latest complete t~62.20 checkpoint; immutable production files.
   - Separate RHS/RK, ghost exchange, shared-node synchronization, AMR tagging/topology/recording, gauge reductions, constraints/curvature, I/O and idle time. GPU events or profiler instrumentation must account for asynchronous execution.
   - Measure active batches by level, parent-memory budget, common-time gauge variability, and spatial/source timestep limits by level. Model wall-time weighting over recorded intervals, not averages of snapshot speedup ratios.
   - Deliver timing breakdown and revised benefit estimate. Reconsider full implementation if unaccelerated work dominates.

2. Establish level-local stepping and a synchronous reference.
   - Explicit block list, level, time, timestep, RK stage and stage time; batch device kernels by level/group.
   - Keep scheduler separate from Cartoon tensor/parity rules and spatial sampling.
   - First run all levels synchronously with the existing integrator; verify field agreement, symmetry and diagnostic ownership.
   - Introduce separately named classical RK4, preserving the current Ketcheson low-storage RK4. Validate synchronous temporal convergence before enabling subcycling.

3. Implement a two-level fixed-hierarchy prototype with prescribed gauge.
   - Evolve real auxiliary parent states; retain coarse RK information for stage-consistent fine-boundary construction. Do not reuse coarse_u0 as an evolved predictor.
   - Allocate parent/stage storage for the populated hierarchy, not the24000 reserved capacity. Count boundary support explicitly.
   - Use qualified VC restriction/injection and deterministic shared-vertex ownership at common times; retain axis parity and algebraic projection consistency.
   - Test waves crossing interfaces, translated smooth geometry, axis and corners. Require fourth-order temporal convergence in the resolved temporal-error regime, without growing interface artifacts. Use scalar tests to isolate temporal order from Z4c spatial-transfer limitations.

4. Generalize to bounded ratios2,4,8,16, then32.
   - Parent prediction, recursive child steps, then synchronization/restriction.
   - Respect every level's propagation and source stability limits; synchronization-interval rollback on an unexpected violation.
   - Group coarse levels to maintain GPU batch size. Record per-level efficiency and memory, not just block-step reductions.

5. Qualify the actual global telegraph gauge before campaign use.
   - max_domain_abs_K requires common-time data, including covered-region ownership rules. Never reduce across asynchronous physical times.
   - Evaluate a common-time coefficient predictor with corrected interval histories and bounded iteration/rollback. Verify convergence against a synchronous reference as interval and coefficient tolerance shrink; do not assume predictor-corrector automatically gives fourth order at max-location switches.
   - Freezing the coefficient is an explicitly different experimental approximation only. If coupling or explicit damping makes coarse steps too small, report loss of benefit rather than change the production gauge silently.

6. Add synchronized dynamic AMR, checkpoints and diagnostics.
   - Initially regrid at synchronization boundaries with sufficient padding. Separately assess cadence effects while preserving tagging thresholds and hysteresis.
   - Full-domain extrema, constraints, curvature and restart files use common-time fields, excluding covered parent points. Never synchronize coincident vertices at different physical times.
   - Introduce explicit synchronization and level-step counters. Match physical output times to the reference for comparisons; the existing120-cycle output setting must not silently become120 coarse steps. Fine-level stopping events request an earlier common-time check.
   - Test symmetry, restart reproducibility, refine/derefine transfers and early-stop decisions.

7. Compare Brill restarts and decide on deployment.
   - Three fixed-hierarchy comparisons: old synchronous RK, classical synchronous RK, classical subcycled RK. Then independently compare live AMR with matched physical regrid/output cadence.
   - Compare fields, global minimum lapse, global max absolute Kretschmann, central and whole-domain constraints, symmetry, mesh histories, peak memory and wall/GPU time at matched coordinate times. Include a milder dispersing case.
   - Accept production use only if differences converge with timestep/synchronization interval, remain consistent with measured baseline numerical error, and performance improves after all overheads. A short late checkpoint benchmark cannot establish long-time stability or validate provisional collapse.
   - Proposed practical performance target: at least2x end-to-end on the late workload without exhausting GPU memory. Report smaller gains honestly and decide whether they justify maintenance.

## Workflow policy that must accompany any eventual deployment

The eight-hour classification is a computational-budget heuristic. A faster integrator would let a run reach a different physical time before the cutoff, so it changes which cases receive provisional labels even with identical physics. Keep existing provisional labels explicit; do not interpret resulting bracket changes as physical threshold convergence. Decide whether to retain the wall-time budget or replace it with a different resource policy before using subcycling in automated bisection.

## Sources reviewed

- Mongwane2015: https://arxiv.org/abs/1504.07609 — high-order refinement-boundary consistency and order reduction.
- Ji et al.2025: https://arxiv.org/abs/2503.09629 — GPU subcycling, coarse dense output, fourth-order scalar tests and BBH comparisons.
- User-provided audit; local driver confirms the current rk4 option is Ketcheson's four-stage low-storage method.

Recommended first approval scope: profiling and synchronous level-local qualification, followed by a two-level prescribed-gauge prototype. Production-gauge support and live campaign deployment are subsequent gates. No implementation or jobs are authorized by this document alone.
