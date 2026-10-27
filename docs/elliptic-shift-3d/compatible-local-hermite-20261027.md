# Local Hermite boundary coefficients: all-node solves and remaining failures

**Stable 3D wormhole-to-trumpet evolution remains unverified.** This report
tests a local boundary representation following the
[layer-affine study](compatible-layer-affine-20261027.md). No candidate here
has been adopted into native or research time evolution. The physical source,
full variable-metric connection variation, and independently evolved Z4
connection are retained. No puncture cells are removed.

## Trial space and precise scope

The old endpoint-collocated shift solves replace the outer gauge rows by exact
physical shift values. Their independent actual-scri metric rates remain
nonzero. This experiment supplies extra normal-derivative freedom while
retaining the same prescribed shift trace. Boundary slots store coefficients
of a piecewise cubic Hermite bubble B(q), whose value is zero at every metric
node. A separate lift carries the future-null boundary shift. The physical
metric-rate equations are evaluated at every metric node, including scri.

The endpoint-only control has B'=1 at the endpoint and zero at all other nodes.
The main candidate chooses slopes on the last four nodes satisfying

    sum_i H_i q_i^k B'_i = 0,  k=0,1,2,    B'_(N-1)=1.

Piecewise Hermite interpolation has zero nodal values and these slopes; its
support lies in the last four radial cells. The coefficients alter only the
shift trial representation. They do not subtract a physical source, filter
the metric rate, or project a constraint. Constant/linear physical shift modes
remain available through the full nodal part of the representation.

The complete nodal tensor rate is the existing full variable-metric rate plus
the bubble's symmetric gradient and determinant-convention trace term. The
same metric-built connection evaluator is differentiated independently. The
degree-two endpoint SBP pair and layer-confined affine coefficient update are
unchanged. These statements concern the declared collocation operator;
they do **not** establish a compatible energy norm for the enriched physical
space or accurate coupling of its off-node angular traces.

## Independent checks before the physical solves

The basis has zero nodal value to roundoff and satisfies its selected moment
conditions below 1e-12. A scalar interval control solves the full derivative
equations with fixed endpoint values and one extra outer coefficient. At
N=10 the balanced candidate's weighted rectangular condition estimate is 23.23,
versus 92.10 for the endpoint-only control. The balanced nodal error decreases
from 4.864e-3 at N=10 to 3.521e-5 at N=40; all directly recomputed scalar
residuals are below 5e-13. This is a scalar representation check, not a 3D
conditioning bound.

On the complete 19,000-node three-shell grid, both the flat and nonspherical
variable-metric tests pass independent cached/uncached and finite-metric-
variation checks. The finite-variation scaled RMS is below 8e-11 and the
weighted adjoint defects are below 2e-15. Boundary values remain exact.

Two adverse observations remain explicit. Pure-bubble vectors have negative
energy in the old nodal coefficient norm, so that norm supplies no positive
energy identity for this enlarged space; PCG is not used. Also, arbitrary
coefficients produce physical off-node angular trace jumps although all nodal
bubble values vanish. These jumps are measured, not identified away by an
angular projection. Evidence:
`evidence/compatible-local-hermite-qualification-20261027`.

## Initial full tensor solves

The first physical run uses the frozen successful spherical M=1/3 constructor
and the same three blended shells, s=(0,.45,.8,1), at block N=10. Every source
dependency is frozen and hashed. The original block-triangular preconditioner
is only an approximate inverse; all acceptance residuals are recomputed using
the full uncached metric connection.

At its 800-iteration cap, the balanced physical run has all-node RMS 1.496e-3
and scri maximum 0.02682. The boundary shift value and independent finite
variation pass, but the original all-node residual gate fails. Radial shift
RMS versus 1D is 7.382e-3. The maximum sampled off-node angular shift jump is
2.993e-4, from a coefficient jump of 0.01039. A small nodal boundary value error
would not reveal this defect.

The unrelated full variable-metric manufactured case also fails at 800
iterations: all-node RMS 7.556e-3, maximum 0.2588, and coefficient RMS error
2.337e-3. Its exact smooth boundary coefficients have matching seam traces;
the approximate solution develops an off-node jump of 6.782e-3. The full
nonunit determinant, nonspherical metric and random interior content are
retained. Evidence: `compatible-local-hermite-wormhole-n10-20261027` and
`compatible-local-hermite-mms-n10-20261027`.

The endpoint-only physical control also fails at 800 iterations: all-node RMS
2.464e-3, scri maximum 0.1581 and off-node seam jump 3.107e-4. Balancing the low
moments improves this particular capped residual, but neither control passes.

A full-action balanced coarse correction includes both the bulk modes and
degree-one angular bubble modes. Its 282-dimensional coarse matrix has a
condition estimate of 1037.5 and does not trigger the singularity guard. The
physical solve nevertheless fails at its 400-iteration cap, with all-node RMS
7.150e-3, scri maximum 0.2993 and off-node seam jump 1.387e-4. It costs about
196 seconds. Evidence: `compatible-local-hermite-endpoint-n10-20261027` and
`compatible-local-hermite-coarse-n10-20261027`.

## User-requested pause

The user requested wrapping up and pausing on 2026-10-27. A further
preconditioner assembles the exact 300-by-300 bubble block on each angular
face, retaining the full cross-face coupling in the fine action. It evaluates
eight columns together and compares selected columns with the ordinary full
operator. This is a preconditioner only, not an interface modification.

Two runs using it were deliberately interrupted after their process commands
and owned process groups were verified. The physical run completed setup and
280 iterations; its last checked RMS was 0.10474 and maximum 2.983. The unrelated
variable-metric run was still in setup and has no completed residual record.
Both manifests record SIGTERM exit code -15 and unchanged frozen sources, with
a separate `user-pause.json`. **These are user-interrupted experiments, not
completed numerical failures, successful solves, or live jobs.** No final
solution checkpoint was written by these two runs. Evidence:
`compatible-local-hermite-faces-n10-20261027` and
`compatible-local-hermite-faces-mms-n10-20261027`.

No native or research RK implementation changed. All newly started 3D numerical
processes are terminal. The [pause handoff](PAUSE-20261027.md) records the broader
status. Even a future converged collocation solve would still leave the energy,
off-node interface, puncture convergence and nonlinear evolution requirements
open. Further work requires the user to resume the paused goal.
