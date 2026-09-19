# Residual trumpet stability isolation

Two opt-in operator changes are being evaluated on `project/tde`. Both default off. Longer small-box controls still grow with either option or their combination: these changes address identified discrete operators, but are **not a demonstrated stability cure**. Current TDE jobs and their immutable executables are unchanged. Dedicated Aurora controls use `debug` only.

The completed [control comparison](CONTROL_RESULTS.md) includes different gauges, ghost extrapolation, inner/outer sponges, constraint damping, dissipation, timestep and resolution. The [full-phase upstream audit](UPSTREAM.md) localizes late positiveTheta work to coupled conformal-factor/shear terms in the nearest puncture cells. This is distinct from both the initial seed and later invalid outer ghosts. No tested variant qualifies for production promotion.

`z4c/residual_hamiltonian_balance=true` uses alpha_full*(Ht_full-Ht_background)/2 in the Theta equation. For a known vacuum background Ht_background=0 analytically; this removes the artificial delta_alpha*Ht_background_discrete/2 gauge source. The explicit -8*pi*alpha_full*E matter term is retained. The problem generator currently restricts this option to its direct Schwarzschild trumpet background. It does not reset residuals or enforce a zero physical Hamiltonian perturbation.

`z4c/user_rhs_after_boundary=true` changes the task dependencies to volume RHS, characteristic boundary RHS, additive user RHS, then RK. The old order projects out incoming components of an earlier sponge source. The new order retains the full discrete source including derivatives of its spatial profile. This changes the boundary datum and does not automatically preserve source terms internal to the volume calculation. It also does not prove the boundary operator is stable.

## Stage-level regression commands

Build with PROBLEM=z4c_tov_ks. With NumPy available:

```
python3 tst/regression/z4c_user_source_order.py --exe /path/to/athena --output /tmp/source-order-test
python3 tst/regression/z4c_hamiltonian_balance.py --exe /path/to/athena --output /tmp/hamiltonian-test
```

The source-order test verifies default behavior, no-sponge equivalence, and actual pre/post-source RK arrays. The Hamiltonian test checks an exactly zero background, pure lapse pulse with identical geometry including ghosts, a genuine nonzero constraint perturbation, and matter sourcing. These are operator regressions; long-time tests and mesh-transfer coverage are separate.

## Located mechanisms

In the dx0.25M, box +/-2M compact lapse pulse, identical spatial geometry including ghost cells still produces initial volumeThetaRHS3.573465265e-11. The pulse's delta_alpha multiplies a discrete vacuum Hamiltonian defect. At the peak (.625,.125,-.125)M, Ht_bg=-.00900205. An independent fixed-point finite-difference check shows this defect tends to zero at sixth-order as spacing decreases.

The characteristic update then creates an exterior Theta RHS peak of 5.94775e-10 at (1.875,-0.125,-0.125)M on the first RK stage. Cubic extrapolation multiplies the resulting Theta state by 35 in the farthest ghost. The boundary differentiates an already differentiated RHS, so its effective stencil reaches six cells inward and overlaps the initial pulse.

Doubling the box at fixed spacing removes that immediate boundary injection while retaining the interior truncation seed. Its core amplitude initially declines by 50M, but longer evolution reverses that decline: successive interior-norm peaks at approximately 37M and 81M grow by a factor of 8.3. This is oscillatory amplification, not stable saturation. The relative roles of the puncture and later boundary feedback still need separation. A single 16³ block and eight 8³ blocks give closely agreeing growing modes in the small box, so interblock exchanges are not necessary for it. None of these findings proves that the production run has exactly the same eigenmode.

The original production's first recorded invalid metric was in outer ghosts after 914M, with negative determinant, followed by widespread invalid-state abort at 939M. That is distinct from early near-horizon Hamiltonian maxima and later outer-boundary Theta growth.

Matched eight-rank GPU vacuum controls give the following abort times. These are later than the first recorded invalid ghost metrics, whose messages only have bracketing time records.

| Gauge | Cubic ghosts | Linear ghosts |
|---|---:|---:|
| Background adapted | 650.025M | 694.050M |
| Standard subtraction | 725.025M | 811.050M |
| Adapted plus lapse damping 0.1 | 693.000M | 774.000M |

Linear extrapolation delays failure but does not remove exponential growth. The PBS wrapper exits successfully after running all cases even when every application aborts; inspect application exits separately. A matched local single-block test of kappa1=0.5 also fails (706.05M versus 675M for kappa1=0.1). First invalid corner ghosts precede these aborts by tens of M while active-cell histories still report zero bad metrics.

Current detailed records are under review/stability-isolation-20260919 and review/boundary-audit-20260919. Do not infer stability from exact zero preservation, finite histories before an abort, or smaller short-time amplitudes.

## Validation and limitations

- The source-order regression checks five configurations over all three RK stages. Defaults agree bitwise with explicit legacy ordering; without a user source, the orderings agree bitwise. The retained discrete characteristic source agrees within 1.1e-22 for a source of magnitude 1.9e-7.
- The Hamiltonian regression checks seven configurations. The initial pure-lapse volume Theta source becomes exactly zero; a genuine constraint perturbation and explicit matter source remain. The signed diagnostic term decomposition is checked against the actual volume RHS. Identical Hamiltonians cancel exactly for finite identical floating-point evaluations; this does not make the entire perturbed RHS or the boundary update zero.
- Sixteen MPI controls cover zero and pulsed vacuum, one and four ranks, with each option separately and together. Active volume RHS and post-recast arrays agree bitwise across the rank decompositions through three stages. Combined four-rank controls reach 20M with zero residual preserved exactly and the pulse remaining finite.
- Aurora operator validation8840315 completed seven of eight20M cases. All four zero controls are bitwise zero on all eight ranks through saved first/final RK stages. The combined pulse's initial volumeTheta source is exactly zero; the retained source agrees within3.31e-24. CPU/GPU first-stage differences are at most4.67e-15 in RHS and2.22e-16 in state. The isolated Hamiltonian-pulse case stopped at time0 because initialization exceeded its75-second application cap; it is incomplete, not a passed20M test. Dedicated follow-up8840368 uses the identical input/executable with a larger application time cap and is tracked separately in the local gpu-operators records. [Compact original results](evidence/gpu-operator-regression.json) preserve the incomplete case.
- These checks do not establish stability on GPUs, across refinement interfaces, or with an evolved star. They do not justify changing production physics. Long-time controls, resolved boundary layers, physical matter, and refinement remain distinct validation steps.

## Reproducing the wider pulse control

`inputs/vacuum_pulse.athinput` has a [-4,4]M box, dx=0.25M, eight 16³ blocks, sixth-order spatial discretization, the original compact lapse pulse, and per-rank binary/restart output. The pulse is initially outside the effective boundary stencil. Its default outer sponge is disabled. To compare a resolved layer, set `problem/outer_sponge_enabled=true`; the input defines a width of 2M (eight cells), rate 1/M, and zero damping for max(|x|,|y|,|z|)<=2M. Compare `z4c/user_rhs_after_boundary=false` and `true` without changing anything else. Compare the Hamiltonian option separately with the sponge disabled.

`summarize.py --run 'label=/path/to/run' --output /path/to/report` accepts repeated runs and produces a white-background comparison and JSON growth fits. It reports actual final times and stopping reasons. A low-quality straight exponential fit to an oscillating norm is not evidence of saturation; inspect successive peak amplitudes as well.

The `evidence/` directory contains operator/MPI regression summaries and comparison figures. The operator figure's three-cell small-box sponge remains unstable with either ordering. A wider, resolved layer is a separate experiment, not a validated replacement.

## First invalid metric before fluid recovery

The optional `<mhd>/debug_metric_before_c2p=true` diagnostic scans the exact active/ghost range about to enter the main primitive-recovery call, and the slabs used by the legacy boundary-recovery path. It checks all ADM components, positive lapse/conformal factor, and positive definiteness of the spatial metric. It does not change fields or error handling. The disabled path launches no kernel. Other direct/FOFC EOS calls are not independently instrumented.

The first event per rank records cycle-start time, cycle, RK stage, rank, global block, level, coordinates, ghost depths, scanned range, bad-cell count, and metric components. The reported cell is the first invalid flattened index in that call; the log does not claim a globally first event or invent an RK substage timestamp.

A replay from the single-block baseline checkpoint at 500.025M records the first invalid input at cycle-start 567.15M, cycle 7562, stage 1, rank/block 0, at (-2.875,2.875,2.875)M. It is the fourth ghost in all three directions. Only one cell is invalid: all ADM components remain finite, but det(g)=-0.0144142. This precedes the recovery warning and the eventual active-cell failure around 675M. The fluid recovery's use of sqrt(det(g)) explains why its first visible symptom can be a fluid NaN even when the incoming metric is already invalid. It does not locate the much earlier growing-mode seed.

The diagnostic OFF/ON controls have byte-identical histories and complete restart payloads, including ghosts, through 5M. Reproduce with:

```
python3 tst/regression/z4c_metric_input_diagnostic.py --exe /path/to/athena --output /tmp/metric-input-test
```

This logger was validated on OpenMP; MPI/GPU execution remains untested. `evidence/metric-input-event.json` and `metric-input-regression.json` preserve the observed event and equality checks.
