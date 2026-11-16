# Wider-pulse diagnostic follow-up

This records the pre-run design; completed measurements and later controls are
documented in [report.md](report.md), with execution details in [status.md](status.md).

Parent campaign: `54ecbcf5865a1b29ed41eba9245db838148b779c`.
The user requested autonomous diagnosis and minimal edits, retaining the bulk
scheme, and specifically a much wider and better resolved lapse pulse.

## What the prior evidence cannot distinguish

The full regularity tensor grows over N12/N16/N20, while time-derivative paths
agree. This rules against a dominant time-record error at those cuts; shared
spatial derivatives mean it does not certify the spatial operator. Neither
nonmonotonic errors nor a nonzero discarded lapse equation uniquely identify a
continuum boundary inconsistency. The original bump had only 3/5/6 primary
radial nodes in its support. Width alone is also insufficient: the tails of
that particular C-infinity profile can dominate differentiated errors.

## Controlled experiment

Keep the original evolution executable, physical Schwarzschild seed, mesh,
full-overlap policy, gauge parameters, penalties, projections and normal lifts.
Only the test seed main is rebuilt. Change only lapse by

```
delta alpha = .001 exp[-6 z^2/(1-z^2)] [1+.5(nx^2-ny^2)]
z=(r-.86)/.08,  |z|<1;  delta alpha=0 otherwise.
```

Support (.78,.94) is 8.89 times wider than before; FWHM is approximately 4.5
times larger. There are 27/36/45 primary radial centers in the support and
8/11/13 above half radial peak. A centered radial-stencil survey with exact
ghost values measures first/second derivative errors about .38/.43 percent at
N12, decreasing at N16/N20. This survey does not substitute for the native
interpolation, compatible lapse-Hessian, transport, or angular operators.
All endpoint donors lie outside the support. Verify initial native endpoint
jets, tensor, lapse compatibility, and all persisted physical constraint data.

The wider pulse overlaps the configured gauge blend (.76619,.87410). B=0 and
unchanged D are free dynamical gauge initial data, not lapse-gradient constraints;
there is no claim that B is decoupled or stationary in that region. No trace or
extrinsic-curvature pulse is added to manufacture an outgoing eigenmode.

Frozen-background outgoing lapse travel times are 1.868 M from the inner edge,
.426 M from the center, and .093 M from the outer edge. Evolve through about
2.4 M (1704/2272/2840 RK4 steps). This extends beyond the slowest estimated
outgoing transit but does not certify that inward components or the background
have settled. Include separate matched Schwarzschild controls. Never subtract
control news and label the remainder physical radiation.

Use three actual spatial resolutions. N12 gauge records every accepted step;
other principal cases record every N/4 steps, giving matched physical cadence.
Measure N12 dense versus stride-three extraction to check that cadence choice.
If its error competes with the observed residual, obtain denser records before
interpreting spatial convergence. Reuse the previous timestep study only as
context; it is not automatically a timestep bound for the changed pulse.

## Failure attribution and decisions

At saved early/late/full-tensor extrema, independently form T_ab by the
contracted Koszul formula using only first metric derivatives. Split the
instantaneous result into chi-penalty time-rate contribution, shift-normal-lift
spatial-derivative contribution, and the remainder, at fixed metric values.
Verify exact additivity and agreement with the archived full tensor. This is
not an evolved penalty/lift ablation: changing a closure changes subsequent
states and compatibility solves. It cannot justify deleting a large term.

A substantial improvement with smooth spatial convergence would identify the
old pulse as a poor accuracy benchmark, without proving A for all data. A
persistent or growing full tensor requires further localization, including
bulk constraints and gauge-transition propagation. No arbitrary pass tolerance,
new source, background subtraction, or bulk-equation change is authorized by a
successful instantaneous cancellation alone.

## Execution boundaries

Two Aurora debug nodes, one task allocation at a time. Inspect current queue
and account; the first N12 job is 9093867 with a one-hour bound. Prior measured
per-step and extraction costs imply roughly 40–50 minutes for this allocation.
Inspect this result before deciding on finer runs. Preserve old run snapshots
and unrelated jobs/files. All larger analysis runs belong on allocated nodes.
