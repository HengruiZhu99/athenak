# Ref-GH stationary-trumpet pause and review handoff

## Status

This campaign is paused.  Slurm allocation `57261202` (four A100s) was
explicitly cancelled while the `dx=1/32` stationary run was at
`t=4.001847771734566`; restart checkpoint `.00002` had been written.  The
`dx=1/16` and `dx=1/24` cases completed `t=20`.  No simulation output larger
than compact histories is committed.

The three-resolution `t=20` stationary gate is incomplete and has not passed.
Do not start the time-dependent wormhole-to-trumpet stage from this evidence.

## Reproducible evidence

All cases use the reference-covariant source at the committed code state
`cace9133` plus the shared clean Perlmutter clone at `acd7188b` (identical
solver source; `cace9133` only adds the local history summarizer).  They used
the CUDA-aware-MPI, four-rank A100 launch established by the prior mapping and
restart evidence.  The final rows are transcribed exactly in
`stationary_t20_pause_summary.tsv`; histories are the authoritative data.

| active cells per block | final time | GH sqrt-integral | reduction sqrt-integral | bad state |
| --- | ---: | ---: | ---: | ---: |
| 64^3 | 20.0 | 4.19e-12 | 3.27e-10 | 0 |
| 96^3 | 20.0 | 8.68e-12 | 6.03e-10 | 0 |
| 128^3 | 4.00185 (paused) | 8.29e-11 | 1.81e-10 | 0 |

The characteristic speed stayed about 0.61 and effective CFL 0.05.  Regular
field maximum and metric condition diagnostics remained approximately one.
These facts establish finite execution over the listed intervals only.

## Scientific observation and boundary

At matched early times, common GH/reduction histories grow rather than
decrease with the active-cell resolution.  The fine run is not a completed
comparison and this pattern alone does not identify a cause.  It is therefore
incorrect to call the run convergent, stable through `t=20`, or a successful
formulation repair.  No thresholds were weakened and no forbidden numerical
intervention was applied.

The outer boundary is at `r=32M`; a characteristic-speed bound of about 0.61
gives a best-case one-way boundary-arrival time near `32/0.61 = 52M`, outside
the stopped interval.  This is a kinematic bound, not evidence that the
observed early resolution trend is or is not boundary-related.

## Requested code and formulation audit

1. Trace each covariant source term in `src/ref_gh/` from the reference-frame
   connection-difference equation to the Kokkos implementation.  Check all
   index placement, sign, symmetrization, and raised/lowered reference-frame
   contractions against the independent oracle.
2. Audit the regular-to-physical reconstruction and its first/second jets.
   Confirm that interpolation uses one compatible quintic-Hermite jet for
   value, derivative, and Hessian in every source path.
3. Audit the norm definitions and reductions.  Verify that the histories are
   squared volume integrals and that mesh-volume weights, MPI reduction, and
   near-puncture membership are formulation-independent.
4. Compare the source terms and RHS component contributions at the same
   physical points for 64^3, 96^3, and 128^3.  Separate roundoff amplification
   from a resolution-dependent discretization/formulation defect before
   changing any source or numerical parameter.
5. Review restart and output serialization only as a continuation safeguard;
   the separate restart regression already passed and should not be used to
   explain away the stationary trend.

The intended next action is an evidence-backed formulation fix or a documented
reason why this norm behavior is expected.  Only then should a new clean
three-resolution `t=20` ladder be authorized.

## Follow-up (commit `b5594291`)

The requested formulation audit found a concrete curvature-conditioning
defect: the generic Cartan Riemann reconstruction carries a growing Ricci trace
near the puncture even though `Q=Delta=0`.  The full analytic Schwarzschild
Weyl tensor now replaces that Riemann only for the exact stationary trumpet
provider.  See `../binary64_stationary_source_audit.json` and the regression
script of the same name.  This removes the stationary curvature-sector forcing
in the table-faithful audit.  A new production ladder is pending; this note is
not an updated stability qualification.
