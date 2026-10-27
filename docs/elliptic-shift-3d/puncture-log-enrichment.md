# Local logarithmic derivative enrichment at a tracked puncture

This is a host-only Cartesian derivative prototype. It is not wired into the
pilot, elliptic operator, native evolution task graph, mapped patch transfers,
AMR or restart. It changes no physical field, gauge source, shift boundary data,
reference connection or continuum equation. The current coupled wormhole failure
is **not resolved** by these local tests.

For a positive field U, let q=log U and rho=|x−X| for the Cartesian puncture
position X. At each query x0, fit the sampled q values to

    q(x) = sum_{|a|≤5} c_a ((x−x0)/h)^a
           + c_log [log rho(x) − log rho(x0)].

The 56 total-degree-five Cartesian monomials retain every nonradial polynomial
mode. A fitted coefficient c_log is a local interpolation coefficient, discarded
when the derivative plan is rebuilt. No regularity exponent is supplied,
advected, dynamically tracked or assumed constant in time. For q=p log rho+H
with polynomial H of degree at most five, this representation is exact for any
constant spatial p. Smoothness beyond that family is tested rather than assumed.

`EnrichedLogDerivative` uses normalized coordinates, strictly positive Gaussian
least-squares weights, column scaling, column-pivoted modified Gram–Schmidt with
two orthogonalization passes, and explicit rank/condition rejection. A triangular
inverse gives an infinity-norm condition estimate for the normalized R matrix.
The plan computes reusable weights for value, all three first derivatives and
all six independent second derivatives. Its diagnostic value fit must **not**
replace an evolved value. Tests evaluate the physical constraint with the actual
analytic field value and only the approximate derivatives.

There are 57 basis columns. A literal 5×5×5 Cartesian stencil has only five
coordinate values per axis; x^5, y^5 and z^5 alias lower-degree polynomials. That
stencil is therefore rejected as rank deficient. The tested alternative selects
the nearest 125 points from a 7×7×7 Cartesian candidate neighborhood, with
reproducible tie breaking. The header accepts supplied points and does not search
or extrapolate a live mesh. Applying it near AMR or patch interfaces requires a
separate active-point selection/transfer contract; a valid local fit does not
establish interface accuracy.

Default rejection limits are relative QR pivot 1e−11, normalized condition 1e9,
scaled weight L1 norm 1e9, and absolute normalized basis reproduction error 1e−6.
These are algebraic guards, not scientific acceptance tolerances. Conditioning
is not a proof of physical discretization accuracy. A log basis far from the
puncture becomes almost polynomial and is explicitly rejected in the tested
case. No automatic fallback is implemented. A practical caller would need to
choose an independently validated ordinary stencil where enrichment is not
identifiable. Near-coincident queries can also amplify weights and fail the
guards; no distance floor is applied.

The puncture itself is an explicit unsupported endpoint: log U and log rho at
rho=0 need independent endpoint/solution-space data. Query or sample collocation
there is rejected. There is no parity, excision, beta=0 rule or manufactured
endpoint value. The supplied point is stamped using position, velocity, time,
RK stage and revision; changed stamps reject evaluation. Only instantaneous
spatial derivatives are computed. A time-dependent physical/chart field requires
its own correct time transport; no extra time term belongs in this spatial fit.
The operator remains direct Cartesian in beta and is unaffected by this prototype.

## Bounded controls and physical residual

Reproduce from the repository root after the existing CPU CMake build is present:

    python3 docs/elliptic-shift-3d/evidence/puncture-log-enrichment-20261027/reproduce.py

This compiles only the new standalone test using the existing build's compiler
flags/Kokkos libraries, runs it with a 30-second bound, and writes full case data,
source hashes, commands, timing, summary and plot. It changes no CMake files.
The test builds 64 local plans at h=.04,.02,.01,.005 for the eight Cartesian cells
bracketing each of X=(0,0,0) and X=(.037,.013,−.021) on fixed global grids. All
queries remain in the report; there is no core exclusion. This is a collection of
local derivative tests, **not** a full-domain physical-volume norm.

The 384 exact-family checks use p=4,2,1.09,.5 and p(t)=2+.4 sin t at t=.2,.8,
including nonradial xy, yz, cubic, quartic and fifth-degree terms. The temporal
examples are independent spatial snapshots. Maximum h-scaled first/second
error is 4.66e−14; maximum normalized R condition is 248.52 and maximum basis
reproduction defect is 7.77e−14. The literal 5³ stencil, query/sample endpoints,
and far-field log degeneracy reject explicitly. All these algebraic controls
pass; they do not validate collapse dynamics.

For a flat conformal metric, vanishing physical K and A, and stationary
Omega=(1−|x|²)/2, the original physical Hamiltonian used here is

    Hphys = chi [−6 + 2 Omega x·grad q
                 + Omega² (2 Delta q − |grad q|²/2)],  q=log chi.

It follows from the physical metric gamma_ij=delta_ij/(Omega² chi), without
subtracting any physical RHS or constraint residual. Manufactured powers need
not be vacuum: compare Hphys against its analytic value, which can be nonzero.
Ordinary fourth-order centered log finite differences are the comparison, with
full mixed derivative products. They use 125 points on 5³; the enriched fit uses
125 points with a wider nearest-neighbor footprint. Error improvement is not a
same-footprint performance comparison.

Maximum physical H errors in the exact power/polynomial family are respectively
1.56e−17, 3.77e−15, 2.84e−13, 4.09e−12 for p=4,2,1.09,.5; ordinary logFD errors
reach .00397, 1.125, 72.98, 730.82. Absolute H errors increase for less regular
powers as physical derivatives become large. Reproducing this family does not
establish uniform ellipticity, a well-posed puncture endpoint or arbitrary
angular regularity.

The actual initial isotropic inner Schwarzschild branch uses M=1/3,
k=.37754694621653295, a=Mk/2, and

    chi = k² rho⁴ / [Omega² (rho+a)⁴].

All stencil points in this control lie inside its isotropic inner branch. Its
analytic physical H is at roundoff (at most 2.29e−15). The all-eight-query maximum
errors are:

| h | enriched H error | ordinary logFD H error |
|---|---:|---:|
| .04 | 2.0728 | 12.2656 |
| .02 | .7790 | 8.1828 |
| .01 | .1780 | 3.6283 |
| .005 | .03027 | 1.2401 |

The enrichment removes the log-rho term but leaves −4 log(rho+a), which has a
Cartesian radial cusp at the missing point. Thus the improvement is substantial
and **does not make the coarse wormhole constraint accurate or prove fourth
order at the core**. These are local initial-slice controls, not a stationary
black-hole evolution or a replacement reference/gauge.

Additional q=2 log rho+H+.1 rho^s n_x n_y controls use s=.3,1.1,2.4. Their physical
H errors decrease at approximately orders .30,1.10,2.40 respectively, and all fail
a fourth-order gate. The new basis does not remove unspecified angular
singularities. Arbitrary time-changing or angle-dependent powers can generate
log-rho angular terms outside this basis; they require independent evidence.

No evolution, solver convergence, MPI/GPU portability, cost scaling, full-domain
norm, puncture motion, safe wormhole-to-trumpet collapse, or radial benchmark
reproduction claim follows from these tests. Next integration would need careful
stencil selection on the actual mesh, original-vector residual and physical
constraint checks, and matched radial evolution controls without weakening their
acceptance criteria.
