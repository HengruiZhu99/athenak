# Remote review prompt: half-plane Kerr default-gauge failure

You are reviewing a failed numerical-relativity qualification experiment in
AthenaK.  Work read-only first.  Do not edit source, change thresholds, or run
new simulations until you have completed the evidence audit and proposed the
smallest discriminating next step.

## Repository and evidence

Clone/fetch `https://github.com/HengruiZhu99/athenak.git` and check out branch
`codex/cartoon-half-plane-rhs-diagnostic-20260813` at commit
`7ee7ecf327ed755392c27eb4eea4190257438c83`.  Its parent implementation is
`f95a05802621b76cff2894d562c38df4b0d09661`; Kokkos is
`6739bc623081648af9e752b616d9671527922cbf`.

Read, in order:

1. `docs/investigations/cartoon_half_plane_kerr_default_gauge_20260813/investigation_report.pdf`
2. `docs/investigations/cartoon_half_plane_kerr_default_gauge_20260813/data/diagnostic_summary.json`
3. the CSVs and plots in the same bundle
4. the default-off instrumentation in `src/z4c/z4c_calcrhs.cpp`, with its
   input options in `src/z4c/z4c.cpp` and `src/z4c/z4c.hpp`
5. the production half-plane operators, parity boundary, Kerr pgen, gauge RHS,
   and central diagnostics used by commit `f95a0580`

## Fixed experiment

- One Kerr black hole: M=1, dimensionless spin=0.5, axis aligned.
- Half-plane rho>=0 Cartoon grid with exact parity ghosts, centered finite
  differences, and analytic SO(2) reconstruction; no near-axis fitting.
- Initial lapse `alpha=psi^-2`, initial shift zero.
- Gauge: default advective 1+log plus direct Gamma-driver shift:
  `lapse_oplog=2`, `lapse_harmonicf=1`, `lapse_advect=1`,
  `shift_Gamma=1`, `shift_advect=1`, `shift_eta=2`.
- Telegraph, slow-start, scale-selective damping, and chi flooring disabled.
- O6 spatial order, RK4, `dchi_max=0.02`, target 5M.
- M/32 reached 5M but developed large late constraints and lost accepted
  horizons.  M/48 failed at cycle 2178, t approximately 4.560417M.  M/64 was
  intentionally not run because convergence had already failed.
- A CPU MPI4 replay from an authenticated M/48 restart reproduced the failure.

## Established observations

- The growing mode is localized near rho=0.0520833M=2.5h and
  z approximately plus/minus 0.28M, inside the apparent horizon.
- Equatorial parity remains approximately 1e-6 or better until terminal
  blow-up.  The failure is not an outer-boundary arrival and reproduces on CPU.
- Chi and lapse remain finite/smooth until after Gamma and conformal A grow.
- The dominant conformal-A RHS contribution is the trace-Ricci scalar term,
  peaking near 7.33e5 at cycle 2170, RK stage 2.
- Gamma is dominated by shift-derivative contraction, expansion, and second
  derivatives (up to about 1.40e6), while explicit Gamma damping peaks near
  1.06e2.  Thus existing evidence does not support explicit damping stiffness
  as the immediate trigger.
- The algebraic det(gtilde)=1 and tr(Atilde)=0 residuals stay small until the
  nonlinear event is already underway.
- Existing evidence does not distinguish a continuum puncture-interior
  moving-puncture mode from a nonlinear SO(2) derivative stability defect.

## Hard constraints for your recommendation

Do not recommend hiding the failure with a chi floor, clipping, relaxed finite
gate, lower order, arbitrary KO dissipation, arbitrary damping, or retrospective
gauge tuning.  Do not describe M/32 reaching 5M as convergence.  Do not infer a
physical instability solely from data inside the horizon.  Preserve the exact
default-gauge run as failed evidence.

## Questions to answer

1. Audit whether the term split and causal interpretation are correct.  Point
   to exact source expressions and identify any missing or double-counted term.
2. Rank the plausible causes: continuum puncture/gauge mode, nonlinear SO(2)
   derivative defect, AMR/interface coupling, central diagnostic pathology, or
   another mechanism supported by source and data.
3. Propose the smallest bounded diagnostic that distinguishes the top two
   hypotheses.  Prefer an operator-level or short restart comparison over a
   new full campaign.  State exact fields, locations, stages, and pass/fail
   observables.
4. State whether a matched Cartesian-vs-Cartoon short restart, a frozen-gauge
   RHS comparison, or a manufactured nonlinear Kerr oracle is the most useful
   next step, and why.
5. If you identify a concrete source defect, specify the narrowest repair and
   the regression tests required before any M/48 replay.  Otherwise explicitly
   recommend stopping without a production change.
6. Give strict stop conditions for any proposed follow-up.  M/64 must remain
   blocked until M/48 is stable and two-grid convergence is restored.

## Required response format

Return:

1. **Evidence audit** — what is proven, unproven, or internally inconsistent.
2. **Ranked failure hypotheses** — each with source and artifact evidence.
3. **Minimum discriminating next step** — exact bounded procedure and outputs.
4. **Repair decision** — justified narrow seam, or `NO PRODUCTION REPAIR YET`.
5. **Qualification boundary** — what may and may not be claimed afterward.

Quote no more data than necessary; cite repository paths, line numbers, commit
identities, and artifact filenames precisely.
