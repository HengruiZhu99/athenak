# Actual intrinsic RK/exchange stage evidence

Set `pc_gh/intrinsic_stage_dump=true` for bounded diagnostic runs. The default is
false. This adds no term, projection, reset, or transfer to the evolution.
Each RK stage writes three files, before RK, after RK and after ordinary
exchange. The latter hook is at the intrinsic algebraic task, which validates
the state and performs no projection. Initialization (stage zero) is excluded.
The free intrinsic path has neither GH reset nor auxiliary/core projection;
those operation budgets are not inferred from absent operations.

Each file has one JSON header line followed by native-endian float64 values in
explicit block/field/k/j/i order, independent of Kokkos storage layout. The
header records block IDs, physical origins/spacings, active indices, cycle,
stage, step-start time, dt, RK coefficients, FD order, damping/KO settings, and
ghost validity. Step-start time is not labelled as physical stage time.
All state cells are saved, including ghosts. Only pre-RK files additionally
contain the active RHS and active RK accumulator, in that order; neither is
assumed to have synchronized ghosts. Files include rank and first-block IDs.
Exclusive creation refuses to overwrite a previous run/restart epoch. Use a
new run directory when restarting a diagnostic run at an overlapping cycle.

`check_intrinsic_stage_dump.py` reconstructs complete periodic global arrays
from the active block cells. It checks unique ownership and uses that complete
array for all repeated derivatives; four stored ghosts alone are insufficient
for the independent physical Ricci operator. Post-RK stored ghosts are marked
invalid and are never used for derivatives. Pre-RK and post-exchange ghosts are
checked against the reconstructed periodic global array, including corners.

The checker retains full signed arrays and per-component coordinate-volume
L1/L2/RMS, signed extrema and grid locations for differences in physical H/M,
alpha-weighted M, C/Z, all reductions and intrinsic/raw Q curls. It uses the
independent primary physical diagnostic from `intrinsic_diagnostics.py`.
These are offline measurements of actual production states, not in-process
physical history output. No excised region is introduced in these periodic
operator fixtures. The state increment and norm of each diagnostic difference
are retained rather than inferred from differences of scalar maxima.

The pre-RK dump supplies the actual compiled mesh RHS for semidiscrete defects

    I = Gdot - D(wdot, rho*wdot+w*rhodot, sdot, betadot)
        - Lie_beta E + lambda E - KO(E),
    J = d(Gdot) - Lie_beta Omega + lambda Omega
        + d(lambda) wedge E - KO(Omega).

The actual family order is w,alpha,s[5],beta[3]; the first expression denotes
the tangent derivative of those potentials, with alpha=rho*w. All 30 components
of I and the 30 independent curl components of J are saved. The RHS already
includes primary and auxiliary KO; subtracting KO(E/Omega) identifies the
remaining defect against that specified discrete target. Nonzero values on
these underresolved seeded states are measurements, not a failed zero-injection
criterion or a claim of instability. Temporal and forced-evolution convergence
are still required.

## Executed evidence

`qualification-runs-20260907/pcgh-clean-reduction/intrinsic-stage-001/` contains
compact results and source/raw-artifact manifests. Large stage files, signed
NPZ arrays and the exact binary remain under
`/Users/hz0693/research/pcgh-clean-reduction-tests-20260907/`.

Six serial CPU FD2/4/6, 2D/3D one-step fixtures cover 18 actual RK stages with
four/eight blocks and independently seeded oblique 50-field states. The final
run is `intrinsic-stage-002`; `intrinsic-stage-001` preserves the earlier
successful instrumentation check before adding rate/KO header fields and
actual-RHS analysis. Results:

* Dump enabled/disabled final float64 payloads are bitwise identical.
* RK update reconstruction error is at most 1.110e-16 (tolerance 2e-12).
* Ordinary periodic exchange changes active cells by exactly zero.
* Every claimed-valid ghost matches the periodic global array exactly.
* All six duplicate-write controls fail clearly and leave existing dumps intact.
* The shared defect analysis matches 18 previously saved signed-vector snapshot
  results to 3.768e-18, including both KO values and all three resolutions/orders.
* All 19 legacy restart/layout controls pass after the new hooks.

Reproduce with the manifest's binary and independently seeded fixture root:

```sh
python3 -W error analysis/pc_gh_clean_reduction/check_intrinsic_stage_dump.py \
  --binary BUILD/src/athena --fixtures SEEDED_DECOMPOSITION_RUN \
  --output NEW_STAGE_RUN
```

The dump is opt-in and writes substantial data; it is intended for bounded
operator tests, not an unrestricted long-run output policy. MPI/CUDA stage-dump
validation, production physical history/norm reduction, nonconforming transfer
budgets and spatial/temporal physical qualification remain unfinished. Earlier
CUDA evolution results apply to the preceding source, not this new dump code.
