# Isolated odd-radial logarithmic derivative falsifier

The existing pilot, `EnrichedLogDerivative` and `PunctureLogPlans` are unchanged.
The new `OddRadialLogDerivative` is a separate host-only prototype, copied from
the tested log enrichment and augmented with fixed rho, rho³ and rho⁵ columns.
It is **not adopted into the current pilot**. The local results strongly improve
the initial radial wormhole constraint, but worsen fractional angular errors and
noise amplification. They do not justify an unconditional generic replacement.

For positive U, fit q=log U using all 56 Cartesian monomials through total degree
five plus four radial columns. With t=(x−x0)/h, Y=(x0−X)/h and R=|Y|, these are

    log rho(x) − log rho(x0),
    (rho(x)/h)^k − R^k,  k=1,3,5.

The fitted coefficients are local interpolation coefficients computed anew from
each stencil. No exponent is prescribed or tracked, no field is divided by a
fixed radial power, and no odd-power asymptotic law is imposed on the solution.
The extra columns are an approximation space; they cannot make unspecified
angular/fractional behavior smooth. All nonradial polynomial columns remain.
There are 60 columns and 125 nearest Cartesian sample points from a 7³ candidate
set. Both enriched variants use identical donors and Gaussian weights here.

For each odd column, its first and second normalized derivatives at the query are

    D_a = k R^(k−2) Y_a,
    D_ab = k R^(k−2) delta_ab + k(k−2) R^(k−4) Y_a Y_b.

Physical derivatives divide by h and h². Column-scaled, pivoted, two-pass MGS QR
and the original rank, condition, weight and basis-reproduction guards remain.
There is no floor, endpoint manufacture or silent fallback. Query/sample rho=0
rejects explicitly; literal 5³ remains rank deficient for total degree five.
Puncture position/velocity/time/stage/revision are stamped as in the original
prototype, and changed metadata rejects evaluation. Only instantaneous spatial
fits are involved. Time-varying exponent examples are independent spatial
snapshots, not an exponent evolution or transport model.

The diagnostic fitted value is never used as the physical value in these tests.
The original physical Hamiltonian uses the actual analytic q(x0) and approximate
q derivatives, for flat conformal metric, K=A=0 and Omega=(1−|x|²)/2:

    Hphys = exp(q) [−6 + 2 Omega x·grad q
                   + Omega² (2 Delta q − |grad q|²/2)].

No physical RHS/constraint residual is subtracted. Manufactured data need not be
vacuum: each numerical Hphys is compared with its own analytic Hphys.

## Results and limitations

The bounded test builds 64 near-core plans for each variant, plus 16 radial
conditioning probes. It uses eight cells bracketing each of X=0 and
X=(.037,.013,−.021) on fixed global Cartesian grids at h=.04,.02,.01,.005.
All locations remain reported. This is a local collection of core tests, not a
full-domain physical-volume norm or a composite-mesh interface test.

All 384 exact-family checks for

    q = p log rho − .7 rho + .4 rho³ − .2 rho⁵ + H(x−X)

pass, with p=4,2,1.09,.5 and p(t)=2+.4 sin t at t=.2,.8. H includes nonradial
xy, yz, cubic, quartic and fifth-degree terms. Maximum h-scaled first/second
error is 2.54e−12; maximum basis reproduction defect is 2.44e−10. These are
algebraic checks, not collapse regularity evidence.

For the actual initial inner isotropic Schwarzschild branch with M=1/3,
k=.37754694621653295 and a=Mk/2,

    chi = k² rho⁴ / [Omega² (rho+a)⁴],

all donor points remain inside its inner branch. Analytic Hphys is at roundoff.
The maximum physical H errors across all eight core queries are:

| h | odd radial + log | log only | ordinary logFD |
|---|---:|---:|---:|
| .04 | .069030 | 2.07281 | 12.2656 |
| .02 | .0061635 | .778981 | 8.18279 |
| .01 | .00018699 | .177996 | 3.62826 |
| .005 | 2.4931e−6 | .0302674 | 1.24012 |

Observed successive orders are 3.49, 5.04 and 6.23. This is consistent with better
representation of the initial radial cusp as resolution improves, not a proof
of uniform sixth order for evolving fields. The true −4 log(rho+a) remainder has
infinitely many radial powers. The finite basis does not encode a complete
wormhole or trumpet endpoint closure. The comparison with log-only enrichment
uses the same footprint; ordinary fourth-order logFD uses a different 5³
footprint and is only a separate control.

The numerical tradeoff is material:

- Maximum normalized QR condition rises from about 249 to **9.12e5** in the core
  controls. The maximum L1 weight norm over all reported functionals, including
  the fitted log coefficient, rises from 35.2 to about **10397**. This is not a
  norm solely of physical derivative weights.
- A deterministic sample perturbation bounded by 1e−12 yields maximum h-scaled
  first/second derivative change **1.14e−10**, versus **3.84e−12** for log only,
  about 30 times larger. This is one reproducible perturbation, not a worst-case
  error bound or a stability proof.
- Angular fractional data q=2 log rho+H+.1 rho^s n_x n_y, s=.3,1.1,2.4, have
  **6–13 times larger** physical H errors with the extra odd columns than with
  log only. Their observed orders remain approximately .30,1.10,2.40. All fail
  fourth-order acceptance.
- Radial fractional data q=2 log rho+H+.1 rho^s improve error constants, but their
  orders still approach s=.3,1.1,2.4 and all fail fourth-order acceptance. The
  fixed basis does not absorb an arbitrary noninteger correction.

The scalar radial columns can absorb unresolved angular content through the
finite Cartesian stencil fit. Better radial accuracy therefore does not imply
more robust generic 3D behavior. Further independent angular and physical-source
controls would be required before selecting this operator for any scientific
run. A conservative next use would be an explicitly qualified radial pilot,
with unchanged original-vector and physical-constraint acceptance criteria;
there is no automatic selection rule in this prototype.

Conditioning probes are ready at rho/h=.87,1.66,2.60,4.56,6.54,8.53, with growing
condition/weight norms. At rho/h=12.52 and 16.52, the basis-reproduction guard
rejects rather than supplying dubious weights. A far-field case is rank
 deficient. No fallback is implemented, and no claim is made that every grid
 offset or AMR donor pattern inside those radii will pass. Additional guards or
 a validated fallback would need to be explicit in an eventual integration.

Reproduce from the repository root after the existing CPU build is present:

    python3 docs/elliptic-shift-3d/evidence/odd-radial-log-20261027/reproduce.py

Evidence includes complete cases, conditioning probes, fractional convergence
failures, source hashes, source HEAD, commands, timing and a visually checked
plot. The executable has a 30-second bound; the measured run is under one second.
There is no evolution, AMR transfer, mapped interface, MPI/GPU, restart, lapse-front
or safe wormhole-to-trumpet conclusion from this isolated falsifier.
