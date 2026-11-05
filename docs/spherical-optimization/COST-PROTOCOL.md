# Final controlled cost protocol

This protocol fixes the comparisons and screening targets before collecting
the final serial timing batch. It supplements the historical exterior-only
[accuracy–cost comparison](ACCURACY-COST.md); it does not turn the failed
areal target screen into a finite speedup ratio.

The central scale is 20M, compact slicing ends at r=.97, outer lapse damping
is .25, chi derivative representation is r2, the algebraic chart is exact,
and the compatible-lapse fraction is one. Reference derivatives, outgoing
shift-normal closure, width five, KO .5, reference step .000125 and exact
scri reference tail remain fixed. All cases evolve the same initial gauge
transient for 10M, with no floors, no perturbation and no extra probe output.
The already completed pulse runs qualify finite-duration robustness separately.

Compare N128, N256 and N512 for these step/penalty policies:

1. Penalty .6510416666660156 with the existing conservative `cfl=.2` policy,
   no fixed-step override. The scri policy limits its effective initial
   characteristic CFL to approximately .15451.
2. The same penalty with the qualified fixed initial characteristic CFL .55.
3. Penalty .5 with fixed initial characteristic CFL .68, **only if its
   refinement, identity, timestep and paired perturbation controls pass**.
   Otherwise retain its failures/pilots and omit it from performance admission.

The first two isolate step relaxation. The third measures the additional
penalty/step tradeoff at comparable frozen-spectrum margin; separate matched-
step controls establish its accuracy effect. This is a small measured family,
not a global optimum over all possible discretizations or gauges.

Use one v26 Release binary and identical execution-source hashes, with
OMP/OpenBLAS/Accelerate/MKL thread environment variables set to one. Run every
case serially three times, rotating the case order between repeats. Do not
run this laboratory's other evolutions, spectral analyses or plotting jobs
concurrently with the timing batch. Unrelated OS activity is not excluded.
Keep every log/result. A failed case remains in the record and is ineligible
for a successful fixed-error comparison.

Report median and min–max wall time for integration including its diagnostic
and output cadence, setup separately, and the complete subprocess duration
including imports, source archiving and setup. Record RHS calls/cost, actual
steps, dt/M, spacing/M, steps/M, initial and worst-evolved speeds and actual
accepted characteristic CFL. Require bitwise equal final states across
repeats of each configuration. Do not substitute nominal step-count ratios
for measured wall-time gains.

The exterior indicator E is the maximum dimensionless physical H/M/Theta/Z
RMS at 301 common reference R/M samples from 25 to 100, uniform in log R.
Record PCHIP and linear interpolation results and their disagreement. A target
requires both interpolations to pass. E is a constraint indicator, not a
certified solution-error bound; the independent field, temporal and spatial
convergence evidence remains necessary.

Report exterior-only targets .003, .001 and .0001 to connect with the existing
study. Also report the following separate, explicit boundary-aware target
boxes rather than mixing residuals of different units into one norm:

| Target | E | Absolute clock residual | Absolute differential D defect |
|---|---:|---:|---:|
| Loose | .003 | .1 | .1 |
| Medium | .001 | .01 | .01 |
| Tight | .0001 | .001 | .001 |

Clock and differential-D tolerances are in the fixed laboratory coordinate
units (M=1/3), not asserted to be invariant physical tolerances. The boxes
are illustrative engineering screens and are declared before final timing;
retain all raw residuals and plots so other tolerances can be applied. The D
screen uses the maximum absolute defect across the three established probe
increments. Recompute boundary diagnostics after timing, not inside selected
cases only. Plot exterior, clock and D residual versus cost separately.

For each family and target, choose the cheapest measured case passing all
specified conditions. If no case passes, report no measured solution and no
finite cost ratio. Do not extrapolate convergence or predict unseen grid
costs. Keep the conditional exterior frontier distinct from the boundary-
aware screen. The final report must state the tested durations and grids,
all failures and the separate 3D transfer requirements; no CPU result is a
GPU-hour or boosted-stability result.

`run_cost_plan.py` accepts a JSON list of explicitly named cases, each
referencing an immutable run manifest plus recorded overrides. It freezes
the expanded plan, source/binary identities, environment and command lines
before starting. The final analyzer must enforce the conditions above.
