# Low-frequency damping and outer-boundary study

The best next **discriminators** are zero exterior constraint damping, first with the original gauge damping, then with weaker gauge damping. None of the tested smaller **nonzero** source tuples removes the demonstrated continuum boundary roots. These are frozen weak-field results, not clearance for a black-hole/star evolution. The production `shift_Gamma=1` bulk gauge is unchanged throughout.

## Quantitative screen

The constant coefficients are `alpha=.999665896620774`, `chi=.9993319048666159`, outward normal shift `.0003157233701617295`, and `G=1`. Original `zero_rate` closure is used. The two tangential wavelengths are 512M and 2048M, respectively 16 and 64 cells at dx=32M. The largest **found and verified** positive real eigenvalue is:

| (kappa1, eta, lapse damping) | wavelength 512M | wavelength 2048M |
|---|---:|---:|
| (.1, 2, .1) | .00725779/M | .00219757/M |
| (.03, 2, .1) | .00644763/M | .00216909/M |
| (.01, 2, .1) | .00478928/M | .00193665/M |
| (0, 2, .1) | none found | none found |
| (.1, .2, .01) | .00378120/M | .00137564/M |
| (.01, .2, .01) | .00241984/M | .00144274/M |
| (0, 0, 0) | none found | none found |

Search: 260 real samples in `[1e-7,.08]/M`, local minimum refinement, then signed canonical-trace root refinement. All accepted roots satisfy the full 20-field volume equation and all 10 boundary rows; maximum relative volume/boundary defect is 1.45e-11. Their signed profiles and physical constraint amplitudes are recorded. Multiple branches occur, so a single mode is not tracked monotonically across kappa. “None found” is a bounded search outcome, never a proof of stability. Tangential resolution does not establish resolution of the normal mode profile in a production mesh.

For kappa=0, both `(eta,lapse damping)=(2,.1)` and `(.02,.01)` also have **no verified complex RHP root found** at wavelengths 512M, 2048M and 16384M. The last matches the widest strip's `k=pi/(64*128)`. Each scan uses 32 real points from 1e-8 to .05/M and 115 nonnegative imaginary frequencies from zero to 5k; local minima and normal-shift surface-mode seeds are refined down to Re(lambda)=1e-10/M. Complex conjugation covers negative imaginary frequencies. All minima drift to the real-axis lower bound, rather than a verified positive root. The grid row-scaled minima are:

| gauge damping | 512M | 2048M | 16384M |
|---|---:|---:|---:|
| eta=2, lapse=.1 | .00514 | .000362 | .00000504 |
| eta=.02, lapse=.01 | .11285 | .07027 | .00460 |

These singular values are search diagnostics with a fixed row normalization, not a universal stability margin. No exhaustive complex-plane, zero-frequency, variable-coefficient, multidimensional discrete or nonlinear theorem follows. All grid evaluations classified normally. Twenty-one refinement evaluations for the weak-gauge 512M case at Im(lambda)=5k and Re(lambda)=6.3–9.8e-10 hit the existing normal-mode cutoff `Re(s)>1e-9`; these are explicitly recorded as unresolved, not successful checks.

## Why weaker constraint damping can help low frequencies

For this code's actual frozen constraint subsystem, define `D0=dt-beta·grad`, `c=alpha sqrt(chi)`, `sigma=alpha kappa1`, and `Q=Gamma-Gamma_metric`. It implies

```
(D0^2 - c^2 Laplacian + 2 sigma D0) Q = 0,
(D0^2 - c^2 Laplacian + 2 sigma D0) Theta = -alpha chi sigma div Q.
```

The homogeneous Q roots in the comoving frame are `lambda=-sigma +/- sqrt(sigma^2-c^2 k^2)`. For `c k << sigma`, one branch decays only as `-c^2 k^2/(2 sigma)`. Increasing kappa then slows that branch. For wavelength 2048M, its e-folding decay times are approximately 21265M, 6364M, and 2076M for kappa=.1,.03,.01. At kappa=0 it is undamped, not intrinsically growing. Removal must then come from outward propagation and a valid boundary or tested absorbing layer. These bulk dispersion values do not predict the boundary eigenvalues above.

The general low-frequency/constant-mode limitation is consistent with the primary Z4 damping analysis; our specific factor `2 sigma` is derived from the present Gamma damping, and is not substituted from the paper's different Z convention. [Gundlach et al., Eqs. 20–23](https://arxiv.org/pdf/gr-qc/0504114)

## Ranked next tests and safeguards

1. **Original zero_rate, kappa=0, eta=2, lapse=.1, wide smooth sponge.** This isolates removal of the boundary source coupling implicated by both continuum and discrete controls. It loses bulk constraint friction; measure constraints returning from the outer region, not just their maximum inside the layer.
2. **Same with eta=.02, lapse=.01.** This is the leading efficiency candidate for the long coarse weak-field control. It passed the same bounded continuum searches and reduces explicit source stiffness. A larger timestep requires matched short/long timestep checks of the actual discrete system; an RK3 stability estimate for scalar friction alone is insufficient.
3. **Do not promote a radial kappa taper from these uniform results.** The subsequent independent [resolved discrete screen](../discrete/README.md) tested kappa=.1 in the core decreasing to zero across512–1792M, with a wide rate.005 sponge and dx64M. It retained positive growth at every tested tangential frequency (gamma=.00025834, .00047619 and .00079592/M at kh=pi/64, pi/32 and pi/8). Theta peaked in the transition, around radius992M. This supersedes the earlier suggestion to try a taper as a preferred follow-up; it requires a separate explanation or redesign before an expensive evolution. These flat frozen-coefficient tests do not prove the gradient coupling below is the sole cause.

A kappa taper is **not** a collection of uniform tests. Even with frozen alpha and beta, a stationary spatially varying sigma yields

```
(D0^2 - c^2 Laplacian + 2 sigma D0) Q = -2 (D0 sigma) Q,
(D0^2 - c^2 Laplacian + 2 sigma D0) Theta
    = -alpha chi div(sigma Q) - 2 (D0 sigma) Theta.
```

Thus the scalar transition contains `-alpha chi grad(sigma)·Q`, and an outward normal shift samples the taper via `D0 sigma=-beta·grad(sigma)`. These terms vanish on the physical constraint surface but can scatter or mix perturbations. Curved-background coefficient terms add further effects. No taper has been evolved in this theory subtask; the separate discrete taper screen above is already negative. The independent boundary-audit agent checked these transition identities algebraically and confirmed both cancellations and the nonzero `D0 sigma` term.

For a 50000M test, `gamma <= ln(2)/50000 = 1.3863e-5/M` is a useful practical growth budget, **not stability**. Fit successive windows, monitor absolute norms as well as normalized growth, and retain inner/transition/sponge/face diagnostics separately. Pulse escape or roundoff-floor fits can create false saturation. Prior source-order changes, momentum-only damping, simple kappa taper over four cells, p-only SAT, and larger extrapolation order already had negative results; they are not new cures here.

## A principled boundary alternative beyond a sponge

The existing local physical-constraint condition with damping removes the fast real roots but still admits its known constraint-free, physical surface mode, `lambda=beta_n k/sqrt(2) + i c k/sqrt(2)`, at every source tuple checked. Replacing four rows while retaining the other gauge/radiation rows is therefore not a sufficient closure. Published Z4c CPBC work proves useful subsystem results but does not provide a ready-made fully damped, general-3D discrete closure for this code. [Ruiz, Hilditch and Bernuzzi, Secs. IV–V](https://arxiv.org/pdf/1010.0523)

A concrete next **reference operator** is the exact exterior Dirichlet-to-Neumann map of the entire frozen 20-field system, including kappa, eta, lapse damping and the normal shift. For Laplace frequency lambda and tangential Fourier k, let `q` denote the 10 configuration variables. Form the normal companion system for `(q,q_n)` from the existing full volume equations. Select its exterior subspace with `Re(s)<0` in `x>0`; if its trace basis is `(q_-,d_-)`, impose

```
q_n = N(lambda,k) q,      N = d_- inverse(q_-).
```

This selects the actual decaying exterior solution jointly for gauge, constraints and geometry. It does not guess independent scalar/vector row allocations or drop the damping/tangential terms. The included `dtn_reference.py` constructs that map and checks 40 points, including all six baseline real roots at the two resolved wavelengths and both physical-only surface roots. Those eight old roots are excluded (minimum row-scaled singular value .00330). Exterior invariant-subspace defects are <=4.68e-15. Other near-zero-frequency samples are poorly conditioned (as small as 1.53e-11); there is no uniform stability proof and no time-domain evolution here.

The map is nonlocal in time and tangential position. A feasible development sequence is: use it as a frozen-symbol target; fit a causal auxiliary-memory/rational approximation on the required long-time frequency band; retain the coupled full-state structure; then derive a compatible discrete ghost/SAT closure and verify roots, transient growth, constraints and convergence. Exact/nonlocal and efficient rational boundary kernels have a primary scalar-wave precedent. Extending that construction to this coupled damped system, corners, varying coefficients and AMR is new work. [Alpert, Greengard and Hagstrom, Secs. 2–4](https://math.nist.gov/~BAlpert/nrbc2.pdf)

A compatible SBP/SAT or normal-mode-stable ghost closure is an alternative discretization framework, not a replacement for correct continuum incoming data. The earlier coarse p-only SAT result failed on refinement. Published shifted-wave work explicitly couples ghost equations to the actual derivative operator; modern energy-based scalar schemes also evolve compatible boundary variables. Neither theorem transfers automatically to this Z4c system. [Calabrese and Gundlach](https://arxiv.org/pdf/gr-qc/0509119), [Erickson, Kozdon and Harvey](https://arxiv.org/pdf/2106.00706)

A complex-coordinate PML is lower priority: one must stretch the complete shifted coupled operator, carry its auxiliary variables and constraints, and prove continuum/discrete stability. Applying a scalar stretch to selected Z4c components is not a supported cure. The full-state DtN benchmark is a safer discriminator before a PML implementation.

## Reproduce

Requires NumPy and SciPy. All model imports are vendored unmodified in `model/`; provenance and source hashes are recorded. No C++, production inputs, jobs or automations changed.

```
OPENBLAS_NUM_THREADS=1 python3 screen.py --model model --output screen-results.json
OPENBLAS_NUM_THREADS=1 python3 complex_screen.py --model model --output complex-results.json
OPENBLAS_NUM_THREADS=1 python3 complex_screen.py --model model --k .0003834951969714103 --output complex-longwave-results.json
OPENBLAS_NUM_THREADS=1 python3 dtn_reference.py --model model --output dtn-reference-results.json
```

Real screening took 2.7 seconds; complex screening approximately 11–13 seconds plus 4–5 seconds for the matched longest wavelength on this host. These are matrix diagnostics, not AthenaK run throughput.
