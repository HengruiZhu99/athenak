# Proposed N12 comparison protocol, declared before native evaluation

This protocol is a proposed engineering acceptance gate for a migration that preserves the reference discretization. It is not a measured native result or a truncation-error estimate. The parent should explicitly adopt or revise it before examining native comparison results, and record any revision and its independent reason.

## Exact identities

Require exact agreement on the initial-data model, all configuration values, KO=.1, stored chi floor=1e-10, constant shift policy, float64 precision, nominal dt=0.00046944203735617445, stage/time accounting, topology, and sampling coordinates. The target is 3551 full RK4 steps from the fresh initial state, not a clipped step ending at exactly 5M. Tracker remains disabled for the N12 reference. Require all snapshots to be finite, every expected point and independent field to be present, and the same complete scri closure to run at each stage.

Native storage may have different field offsets or block order. Match values by block identity and physical sample coordinate, converting both to the same physical field convention first. Match all bulk and surface fields, not only independent fields or global norms; dependent algebraic components should also reconstruct consistently. Do not compare a native restart file byte-for-byte to an SP1 research checkpoint.

## Arithmetic and short-step gate

Where arithmetic order is preserved, require bitwise equality. For changes that only alter floating-point evaluation order, compare fresh initial data, one complete RHS stage, one RK4 step, and a bounded 8-step evolution before the long run. For every field and point use

```
abs(candidate-reference) <= 1e-12 + 5e-10*abs(reference)
```

and additionally compare the maximum scaled difference `abs(delta)/(1+abs(reference))`, volume-weighted error RMS, location of the worst difference, and per-field maxima. The absolute tolerance is not permission to change chi near its floor: require matching floor policy and finite floor candidates, with chi differences separately bounded by `1e-14 + 5e-10*abs(chi_reference)`. Pointwise residual and constraint comparisons need consistent derivatives, halos, and sampling, so record those conventions with the result.

The source's normal-compatibility acceptance bound is 1e-7. Preserve that check and record the actual maximum at every stage. The scri clock diagnostic is not identically zero in this campaign (the recorded first-segment maximum is approximately .37684); compare it to the reference instead of imposing a new zero-clock requirement.

## Through-step-3551 gate

Proposed accumulated roundoff guard: require every full-state value to satisfy

```
abs(candidate-reference) <= 1e-10 + 1e-8*abs(reference)
```

with the tighter separate chi guard `1e-13 + 1e-8*abs(chi_reference)`. These tolerances are conservative engineering thresholds for preserving the numerical scheme through a few thousand steps; they are not mathematically derived stability bounds. Report both maximum and volume-weighted differences, the maximum's coordinate/field, and complete coverage. Require normal compatibility <=1e-7 and matching successful finite/geometry checks; compare the recorded clock/CFL values, chi minima, and floor diagnostics. Floor activations near exact thresholds can depend on roundoff, so report candidate and reference counts and maximum corrections rather than hiding their differences.

At every common stored time through 5M, require the full and full-horizon-exterior H, M, Theta, and Z **RMS** histories to satisfy `abs(delta) <= 5e-10 + 1e-6*abs(reference)`. Initial roundoff-size diagnostics should be reported separately rather than used to infer convergence. Require the same successful horizon-found/complete-surface status; propose horizon mass/radius agreement within relative 1e-7 (absolute floor 1e-10 in M units). Preserve FastFlow tolerances and centers; iteration counts and expansion values are diagnostics, not substitutes for complete-state agreement. Do not compare a tracked N24 horizon mask to the fixed-origin N12 mask.

The existing replay's `5e-11*(1+abs(expected))` comparison applies only to recomputing saved full/fixed-outer RMS values from exactly the same checkpoint. Keep that tighter gate for reference decoding/replay. Its zero recorded difference establishes decoding/diagnostic reproduction and does not establish a tolerance for a different numerical scheme.

## Changes to the discretization

If native SMR/AMR introduces different prolongation/restriction, coarse/fine derivative closures, physical sample locations, evolution overlap, or a different scri closure, the strict migration gates above no longer establish numerical equivalence. Do not relax them merely because a candidate failed. Document the changed discretization first and establish a separate error budget from independent resolution/consistency controls before evaluating the new result. A horizon mass close to one and a stable 5M evolution alone are insufficient to call it the reproduced N12 run.

Before any performance comparison, complete the adopted correctness gate. Use equal native block weights as requested, the same work at both node counts, maximum-rank fenced elapsed time, and complete RK4-step throughput; startup, diagnostics, and checkpoint costs belong in separate timings.
