# Ref-GH analytic device-view performance handoff

## Review status

This is a deliberately partial checkpoint for external review.  The analytic
radial-q coefficient and geometry oracle is implemented and passes the focused
local CPU gate.  The production Ref-GH task graph still uses the generic
1171-Real-per-cell reference-cache pipeline.  No Aurora PVC run or performance
claim belongs to this checkpoint.

Branch and source provenance:

- branch: `codex/ref-gh-analytic-device-view-performance-20260828`
- required parent: `3dc7e622ab920e98cb6becd325dbd328f5e8009b`
- implementation checkpoint before this report: `e5dfd5a1b0963798960a260ed35c402ceeb693df`
- Kokkos submodule: `6739bc623081648af9e752b616d9671527922cbf`

The unrelated dirty checkout at `/home/hzhu/Desktop/research/gr/athenak` was
not modified.  Work was performed in the clean continuation worktree
`/home/hzhu/Desktop/research/gr/athenak-ref-gh-feedback-continuation-20260823`.

## Implemented milestone

`src/ref_gh/reference_analytic_radial_q.hpp` introduces independent compact
radial data layouts:

- 12 static Reals per point: trumpet alpha, L_T, B and their first and second
  radial derivatives, plus u, u_r, and u_rr;
- 8 stage Reals per point: L and its t/r derivatives through L_ttr and L_trr.

Both layouts have compile-time upper bounds of 16 Reals.  The implementation
uses the exact nonperturbative law

`L = L_T exp[-(q-1) W log(r/M)]`

and reconstructs Cartesian derivatives from radial data.  Lightweight analytic
accessors provide the same geometry interface as the generic reference object.

`scripts/ref_gh/generate_analytic_radial_q_geometry.py` is a deterministic
SymPy generator that starts from the analytic coframe and derives the metric,
inverse metric, coordinate derivatives, frame/coframe, Christoffels,
spin connection and derivative, frame Riemann tensor, and Ricci tensor.  Its
generated header is committed at
`src/ref_gh/generated/analytic_radial_q_geometry.hpp`.

Generator provenance:

- pinned SymPy version: 1.14.0
- generator SHA-256:
  `eaaf013057806d4c40dd67da7d83f902cace7585cefaba78a1342e58d252ea02`
- generated-header SHA-256:
  `bd09c1ca507f682b2694c25c355726d8b38dcff47daffb77da95b33ced85aae5`
- regeneration was checked byte-for-byte deterministic.

`src/pgen/ref_gh/source_unit.cpp` adds two device-side oracle gates over the
required parameter matrix

- q = 0.75, 0.9, 1.0, 1.1, 1.25, 2.0;
- qdot M = -0.1, 0, 0.1;
- qddot M^2 = -0.05, 0, 0.05;
- four deterministic off-axis points, for 216 samples total.

The compact coefficients are compared against the independent generic provider.
The generated geometry is compared against the generic geometry using the
existing conditioned category scaling and the unchanged `256 epsilon`
tolerance.

## Verified local evidence

A fresh Release build used GCC 13.3.0, Kokkos 4.7.2 Serial, MPI off, OpenMP
off, CUDA off, and SYCL off.  This command exited zero:

```text
athena -i tst/inputs/ref_gh_q_controlled_reference.athinput
```

Relevant output from the fresh rerun:

```text
reference-GH q-controlled trumpet provider passed: q=1 identity=4.44089e-16 profile=1.11022e-16
reference-GH analytic radial-q coefficient oracle passed: samples=216 max error=8.88178e-15
reference-GH generated analytic radial-q geometry oracle passed: samples=216 conditioned error=7.10543e-15 category=20
reference-GH q-controlled trumpet reprojection passed: metric=4.44089e-16 derivative=9.85933e-16 min-nontrivial-Pi=0.125187
reference-GH q-controlled gauge reprojection passed: Hhat=1.11022e-16 theta=2.22045e-16 subtraction=2.22045e-16
```

The generator passes `py_compile`, a second generation has the same SHA-256,
and `git diff --check` passes.

An initial raw absolute comparison of every geometry component reached
`4.54747e-13`.  It was replaced by the pre-existing production oracle's
conditioned category norm, not by loosening that norm or its `256 epsilon`
tolerance.  The conditioned maximum above is the controlling result.

## Not implemented and not established

The following controlling-goal items remain open:

- Production backend dispatch and replacement of the four generic arrays of
  100 + 410 + 313 + 348 = 1171 Reals per cell.
- A compact production implementation of contracted GH reference-source terms.
  The current generated full-geometry routine and direct curvature accessors
  are correctness scaffolding and are not suitable hot-kernel code.
- Analytic gauge-reference and `dtTheta` integration.
- Generic-versus-analytic comparison of all ten source components and all 61
  evolved RHS components.
- Z4c-like task-graph integration, compact q-estimator sample lists, analytic
  boundary population, restart coverage, and scientific regressions.
- Aurora PVC execution, memory evidence, matched timings, or performance uplift.

Therefore this branch does not yet fix the prior Aurora Level Zero failure and
does not establish trumpet convergence, stability, production readiness, or
the requested memory/performance target.

## External-review request

Please review the changes from parent `3dc7e622` through the branch tip, with
special attention to:

1. exact differentiation of the radial q ansatz, especially `L_ttr`, `L_trr`,
   and the `qdot`/`qddot` cross terms;
2. tensor index ordering, sign conventions, and frame-versus-coordinate bases
   in the independent generated geometry;
3. whether the conditioned oracle compares every generated component with an
   appropriate category scale and cannot silently skip a component;
4. Kokkos device portability, stack/register pressure, and accidental hot-path
   use of the monolithic generated geometry;
5. a production design that preserves the generic cache as an independent
   oracle while selecting the analytic backend once on the host and allocating
   no large generic arrays in analytic production mode.

Do not infer Aurora qualification or scientific convergence from this local
oracle checkpoint.  The next implementation milestone should first add compact
contracted-source generation and an all-61-RHS local oracle before requesting a
bounded PVC discriminator.
