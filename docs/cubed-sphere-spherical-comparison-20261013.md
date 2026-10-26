# Fresh disturbed-Minkowski test on the spherical outer boundary

The wider-collar spherical run completes `t=2`; the matched-collar runs time
out near `t=0.04226`. **The original near-scri failure has not been reproduced
or excluded on the new grid.** This comparison runs the old failing wave-map test on
the six-block cubed-sphere domain. Evidence and plots are in
[the comparison capsule](validation/cubed-sphere-spherical-comparison-20261013/README.md).
The previous [independent-refinement assessment](cubed-sphere-refinement-20261012.md)
and its frozen evidence are unchanged.

## Parameters and the comparison's limits

The fresh comparison restores the old large pulse, width `0.35`, reference
`R=1, a=0.5, r0=0.05, r1=0.95`, original wave-map gauge, physical
`P=Kphysical-2Theta`, dissipation `0.1`, **constraint damping `kappa1=10`**,
and pole timestep factor `0.03`. The earlier mapped smoke controls used
`kappa1=0`; they were not matched-parameter reproductions of the old test.
The wave-map helper remains byte-identical to the archived native recipe.
No exact arithmetic, manufactured forcing, floors or constraint projection
are added. The original final-stage algebraic metric/curvature projection
is retained.

The Cartesian core has half-width `0.25`, with `16^3` bulk cells. Each of six
shell patches has 272 radial cells and `12^2` angular cells. The smallest
active Omega is `0.0021704751841302764`, within **0.016%** of the old N24
failed cell's `0.0021701388888885099`. The near-scri comparison is deliberately
not credited with the much wider collar of the earlier S12 controls.

![Computational domain](validation/cubed-sphere-spherical-comparison-20261013/computational-domain.png)

This is a continuum-equation/initial-data comparison, not a boundary-only
change to the old discrete operator. The core spacing is `0.03125`, versus
`2.2/24=0.0916667` in the old N24 mesh. Mapped derivatives, interpolation at
interfaces/seams, logical KO coefficients, the distribution of Omega samples
and spatial CFL estimate also differ. Matching the minimum Omega does not
match those errors. The linear radial map with 272 radial and 12 angular
cells is strongly anisotropic, including near the inner interface. A failure
on this grid would need localization and refinement controls before it could
be assigned to the continuum equations.

The outer face is the smooth sphere `r=1`, with Omega zero on that face.
Evolution degrees of freedom remain at positive-Omega cell centers.
Quadratic and quartic extrapolation of reference deviations supply outer
ghosts in the mapped atlas. The ordinary regular-Z4c Sommerfeld operator is
not used by this probe. A spherical face simplifies the topology, but does
not by itself supply a constraint-preserving characteristic scri condition.

## Validation and execution

The mapped research adapter now executes its jet loading, geometric/gauge
RHS, donor fills, CFL, RK updates, projection and constraint samples through
Kokkos kernels, with host setup/reporting. The CPU early disturbed control
matches the previous frozen CPU probe exactly in all nine compared final
metrics in both regions. All five regular-grid/derivative/failure-capture
CPU regressions pass in 120.15 seconds.

Aurora job **8924109** completes the GPU stationary control at `t=0.001`
and early pulse at `t=0.005`. Maximum stationary field deviation is
`1.32e-15`; Hamiltonian maxima are below `3.41e-12`. The largest CPU/GPU
relative difference among the eight compared early-pulse metrics is
`4.93e-10`, below the declared `1e-7` tolerance. Both reference snapshots
are byte-identical before and after. All 377 actual source/configuration/input
files match the delivered source and preserved runner; ten noncompiled
AppleDouble sidecars in the transfer are separately identified. Diagnostic
radial counts, volumes and squared constraint integrals recount correctly.

The same job's wider-collar original-parameter control reaches
`t=1.104626887733561`, then hits its 180-second case cap. It has finite
diagnostics and no numerical exception, but no final reference snapshot or
completion at `t=2`. PBS reports exit zero and walltime `00:03:16` for the
whole job; that status admits the port, not the capped science control.

A one-step CPU admission at the matched collar completes for both ghost
degrees, with finite diagnostics and byte-identical immutable-reference caches
before and after. Those binary snapshots serialize the actual consumed device
cell cache and reference-value views, including ghost values; each payload is
883,132,538 bytes. Large binaries/reference payloads remain in research
storage, with hashes in the retained receipts.

The initial pre-port CPU wider-collar run with restored `kappa1=10` reached
`t=1.80029` without a loss of admissibility, then hit its 1200-second wall cap.
This is a timeout, not completion at `t=2`. Its smallest Omega is `0.0486184`,
so it cannot decide the matched-collar question.

All new Aurora submissions use **capacity, one node at a time, 30-minute
allocations** under `CompactBinaryMerger`, as requested. Workers retain short
runtime caps. The executable is Release SYCL with precise floating-point
settings; this is not a bounds-checked Debug build. Production hyperboloidal
guards remain unchanged. The mapped probe still requires uniform bulk storage;
these controls do not validate hyperboloidal AMR.

## Evolution outcome

Aurora job **8924120** finishes in **7 minutes 33 seconds**, PBS exit zero.
The two matched-collar cases use separate tiles of one physical GPU; an
independent physical GPU runs the wider-collar baseline. All per-case
diagnostics are finite, with radial counts/volumes/squared integrals checked.
All 387 manifest entries, including the ten transfer sidecars, are bound to
the retained files. No numerical exception or bad-stage stencil occurs.

| Fresh spherical case | Minimum Omega | Last diagnostic time | Outcome | Shell H L2 / maximum |
| --- | ---: | ---: | --- | ---: |
| S12, quadratic ghosts | 0.0486184 | 2 | Completes; reference unchanged | 0.0231057 / 0.207486 |
| S272, quadratic ghosts | 0.00217048 | 0.0422560619074 | 450-second cap, 800 recorded cycles | 0.00466087 / 0.204876 |
| S272, quartic ghosts | 0.00217048 | 0.0422560619084 | 450-second cap, 800 recorded cycles | 0.00466087 / 0.204876 |

The S12 case completes in 337.57 seconds and 2724 cycles, with global minimum
lapse `0.9816886` and chi `0.7602895`. Its consumed reference snapshots are
identical. The Cartesian N24/N32 runs fail at `0.79277` / `1.58373`, but S12's
minimum Omega is 22.4 times the old N24 failed cell's. Its completion cannot
be credited to boundary topology alone. At `t=2`, its shell Hamiltonian peak
is at radius about `0.828`; the fixed `r>0.9` collar also has a maximum
`0.195477`. Completion does not establish accuracy or convergence.

The matched-collar initial timestep is `5.28201e-5`. At the measured rate,
reaching the old N24 failure time would take roughly **2.3 hours**, beyond the
requested short allocations. This estimate is not a stability prediction.
The capped cases have no final reference snapshot; const ownership is enforced
in source, but before/after equality is observed only in completed controls.

At their last recorded time, both S272 cases have their largest shell H peak
at `(0.0236793,0.2841511,0.0236793)`, near the inner cube/shell interface.
The fixed inner-shell H L2 is `0.0148838`, versus only `7.31e-9` in the
fixed outer collar. Changing the outer extrapolation degree barely changes
this early peak. This points to interior/interface discretization error in
this early interval; it does not identify the origin of the old later failure.

![Spherical evolution controls](validation/cubed-sphere-spherical-comparison-20261013/spherical-evolution-controls.png)

## Next decisive controls

1. **Make the matched-collar retest affordable.** Validate a graded radial grid
   with a nonsingular map and balanced angular/interface resolution, and reduce
   probe overhead without changing its equations. Retest map derivatives,
   Cartesian jets and CPU/GPU parity before physics. The decisive result is
   fresh quadratic/quartic evolution through both old failure times and `t=2`
   at comparable outer Omega, within the requested allocation; a timeout is
   inconclusive.
2. **Converge the coupled spatial operator.** Hold the outer collar fixed and
   independently refine the bulk, shell angles and inner radial region. Require
   decreasing H/M errors in fixed physical subsets at three adequate grids,
   without moving a larger peak onto an interface. Use true half timesteps;
   their differences should be below 10% of the finest spatial difference.
   If this fails, repair the interpolation/dissipation/closure together and
   derive characteristic null/shear/Z4 scri conditions. Ghost degree alone is
   not a demonstrated repair.
3. **Keep arithmetic and oracle work tied to native evidence.** Compare exact
   binary64 failed-stage operands with higher precision before assigning a
   failure to arithmetic. A manufactured oracle needs an admissible slicing
   and a raw native RHS residual; stationary subtraction and finite readbacks
   do not establish evolution stability.
4. **Apply the original scientific gates.** Measure a substantial nonspherical
   gauge amplitude separately from the largely monopole 0.2 coefficient and
   demonstrate convergent constraints/curvature through several crossings.
   Then test wormhole-to-trumpet evolution with the same immutable Minkowski
   reference, resolved interior and convergent horizon/mass diagnostics.

No stable substantial angular disturbance or wormhole-to-trumpet transition
is accepted by these bounded controls. The requested puncture comparison is
tracked separately so its distinct initial data, gauge and layer are explicit.
