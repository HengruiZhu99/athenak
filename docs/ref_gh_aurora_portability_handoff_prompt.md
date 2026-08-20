# Remote-agent prompt: audit and debug Aurora Ref-GH PVC failure

Work in `HengruiZhu99/athenak` on branch
`codex/ref-gh-covariant-source-repair-20260818`.  Fetch the branch tip and begin
with a read-only audit of the committed code and evidence.  Scope is vacuum
reference-frame first-order GH only.  Do not work on fluid coupling,
Kerr-Schild data, or horizon finding.  You may refactor aggressively for
performance portability, but do not change the mathematical formulation,
finite-difference stencils, RK algorithm, dissipation, gauge/source equations,
or diagnostic definitions.

The current Aurora qualification gate is failed.  On Intel Data Center GPU Max
1550/PVC, the one-rank, one-block `64^3`, `dx=1/16`, full-output stationary
trumpet initializes correctly.  The initial RHS Linf is `8.55241e-17`; all
diagnostics and explicit Kokkos fences complete.  At cycle 0 the next evolved
stage passes `InitRecv`, `CopyU`, and `CalcRHS zero`, then Level Zero reports a
GPU `NotPresent` write page fault and rank 0 aborts with exit 134.  The most
recent decisive run is PBS job `8769672` on `x4117c5s0b0n0`.

Read first:

- `docs/fo_gh_artifacts/reference_covariant_repair_20260818/aurora_portability_20260820/README.md`
- `docs/fo_gh_artifacts/reference_covariant_repair_20260818/aurora_portability_20260820/job_8769672/`
- `docs/ref_gh_covariant_source_repair.md`
- `docs/ref_gh_puncture_validation.md`
- `src/outputs/history.cpp`
- `src/ref_gh/ref_gh_calcrhs.cpp`
- `src/ref_gh/ref_gh_diagnostics.cpp`
- `src/ref_gh/ref_gh_tasks.cpp`

Important negative and positive controls are already recorded; do not repeat
them blindly.  No-output evolution passed on PVC.  The full native-history path
reproduces the fault.  Moving the metric condition calculation into diagnostics
did not fix it.  A truncated cached-condition stage-4/stage-5 pair passed one
cycle (`8769596`), but both a custom batched maxima reducer (`8769628`) and the
current built-in combined `Kokkos::Max<Real>` pattern (`8769672`) still failed
after their fences.  On CPU, the current histories are byte-identical to the
pre-refactor scalar-reduction histories.  Therefore a passing fence does not
exclude earlier latent device-memory corruption.

Perform a detailed code review against the mature main-branch Z4c execution
and history paths.  In particular, audit allocation extents and lifetime of
`u0`, `u1`, `u_rhs`, `u_con`, ADM scratch/state, and host reduction results;
the `ncon` diagnostic-slot expansion; every flattened `(m,n,k,j,i)` index;
history-data bounds and output ordering; task dependencies between diagnostics,
history output, and the next RK stage; SYCL USM captures; MPI reduction buffers;
and whether any device view or reducer result is destroyed/reused while work can
still reference it.  Compare the complete call/lifetime pattern, not only the
equations inside a kernel.  Treat the consistent delayed fault at the next RHS
write as possible latent corruption, not proof that `CalcRHS zero` is the
writer at fault.

Use Aurora's known-good environment and existing isolated campaign root as
documented in provenance, but create your own unique work/run directory and
do not touch other users' jobs or directories.  Keep at most one new PBS
request.  Preserve unrelated dirty files.  Never compile substantial work on
the login node.

Make one coherent equation-preserving correction based on the audit, validate
it locally, and run exactly one focused full-output one-cycle PVC gate using
`scripts/ref_gh/aurora_stationary_portability_repro.pbs`.  Do not claim the bug
fixed unless the evolved cycle exits cleanly on PVC.  If and only if that gate
passes, immediately run the committed three-resolution `t=1M` stationary-
trumpet ladder with `scripts/ref_gh/aurora_stationary_portability_t1.pbs` and
report actual errors/orders.  Do not begin a long evolution first and do not
claim convergence or stability from the one-cycle gate.

Return a code-review report that separates observations, inferences, and open
questions; identifies the exact root cause or remaining blocker; lists every
source change; and links each scientific/runtime conclusion to committed
evidence.
