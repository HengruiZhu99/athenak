# Remote review request: Ref-GH covariant-source delta-upper repair

## Decision requested

Audit this narrow formulation fix before any paused Perlmutter stationary or
puncture campaign is resumed. The only requested outcome is one of:

1. approve the two-pass `Delta` construction and its regression coverage;
2. identify a specific remaining index/order/sign defect with a reproducer; or
3. identify an inadequacy in the source oracle or in the interpretation below.

Do not broaden the review to fluid coupling, Kerr--Schild data, horizon
finding, AMR, or a new production evolution campaign.

## Defect and repair

`CovariantGhScalarWaveSource` forms
`Delta_ABC = (Q_BAC + Q_CAB - Q_ABC)/2` and then
`Delta^A_BC = psi^AD Delta_DBC`.

The old single nested loop filled `Delta_lower[A][B][C]` and immediately
formed `Delta_upper[A][B][C]`. Raising at fixed `A` reads every lower first
index `D`; rows with `D > A` had not yet been initialized. This is an
order-of-initialization error, not an approximation choice. The repair first
fills all lower-index components, then performs a separate raising pass.

The exact stationary state has `Q=Delta=0`, so its original `t=0` ladder did
not exercise the defect. Generic nonzero source states do.

## Required code audit

Review:

- `src/ref_gh/covariant_gh_source.hpp`: index placement, use of the physical
  inverse `psi^AD`, and the two-pass lifetime/order guarantee;
- `src/pgen/ref_gh/source_unit.cpp`: construction of Lorentzian metric jets,
  conversion between `(d_psi)` and `(Pi,Phi)`, and comparison of the production
  covariant source to the retained coordinate source plus transform;
- dispatcher/CMake/test wrapper plumbing;
- whether another fill-and-raise loop in Ref-GH has the same dependency bug.

The source unit deliberately uses 1,000 deterministic generic flat-reference
Lorentzian jets with nonzero shift, `Pi`, `Phi`, and `Delta`; it is not a
stationary-state-only regression.

It also contains 128 compiled nonflat manufactured references with nonzero
time-space coframe, spin connection, and curvature. Verify especially that the
coordinate derivatives of `Psi_AB` are completed over all frame-index pairs
before the coordinate metric jet is assembled; initializing them inside a
coordinate-index loop silently overwrites earlier frame components.

## Evidence and current limits

Before this repair the compiled test observed maximum mismatch `1.14718e-4`
(and `2.25991e-5` with zero shift). After it, the compiled OpenMP test reports
`6.93889e-17`. Exact Minkowski remains zero through `t=0.02`; linear-wave L1
orders are 3.9134 and 3.9456; robust Minkowski decays through `t=0.2` at 8, 16,
and 32 cells. The replacement exact-stationary `t=0` 64/96/128 ladder has RHS
Linf at roundoff scale.

The C++ nonflat source/coordinate comparison now gives `3.33067e-16` over 128
samples. It verifies the compiled source on a nonflat reference but still does
not substitute for a restarted or long-time stationary evolution.

The artifacts are in
`docs/fo_gh_artifacts/reference_covariant_repair_20260818/delta_upper_source_repair/`.
These local checks do **not** qualify a stationary `t=0.1`, `t=1`, or `t=20`
evolution, the transition, or any puncture result. The Perlmutter campaign is
paused and must not be resumed merely because this source-unit regression
passes.
