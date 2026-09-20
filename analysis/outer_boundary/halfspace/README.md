# Frozen production boundary symbol

This portable package reproduces growing modes of the **original production `zero_rate` boundary**, with `shift_Gamma=1` unchanged. It also records why the tested physical-constraint replacement is incomplete. It does not change AthenaK, submit jobs, or establish a stable production configuration.

Run the quick checks from this directory with Python, NumPy and SciPy:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 validate.py
```

Optionally use `--output checked.json`. To repeat the longer bounded practical-control search, use `python3 scan_controls.py --output control-scan.json` (tens of seconds). The checked reference output is `verified.json`. The quick test takes approximately 0.3 seconds on the local runtime. It asserts the independently reduced normal principal matrix, five original-boundary roots at two tangential frequencies, full volume/boundary defects, and two rejected replacement modes. It also checks that the all-q controls contain exactly the intended ten rows. `scan_resolved.py --output resolved-scan.json` repeats the bounded real-frequency search at the tangentially resolved frequency below.

## Model and provenance

The source is the immutable `domain2048-20260919/source` snapshot, nominal `8b694211 + static-floor.patch`; source hashes are in `provenance.json`. `characteristics.py` vendors three characteristic functions verbatim, with attribution and the source hash. All other dependencies are local files plus NumPy/SciPy. No old checkout, binary, checkpoint, or absolute filesystem path is required.

The later implementation audit also confirmed an independent unavailable-RHS-ghost read at physical boundaries intersecting internal tangential block edges when the metric is off diagonal. `SOURCE_AUDIT.md` corrects the initial audit and explains the active-only stencil repair; `BOUNDARY_STENCIL_REVIEW.md` records its focused review and proposed integration checks. The continuum model below has no blocks or ghost storage, so it neither contains that bug nor proves that repairing it stabilizes production.

The background is frozen at radius sqrt(2000²+2000²+976²)M, with a local normal along the two positive incident faces. Thus alpha=0.999665896620774, chi=0.999331904866616, beta_n=0.000315723370162. This is a planar weak-field model, **not a corner or AMR discretization**. Kappa1=0.1, kappa2=0, shift damping=2, lapse-residual damping=0.1, and background-adapted G1 gauge are retained. Background K/A, coefficient gradients, atmosphere and finite differences are omitted. Tangential background advection is removed by a frequency translation that leaves growth rates unchanged.

The 20 evolved variables are ordered:

```
chi, hxx, hyy, hxy, hxz, hyz,
Khat, Theta, Axx, Ayy, Axy, Axz, Ayz,
Gamma_x, Gamma_y, Gamma_z, alpha, beta_x, beta_y, beta_z
```

Metric/A perturbations are algebraically trace free. The ten q fields are chi, five metric fields, lapse and shift. The remaining ten are p fields. The q_t-to-p matrix is invertible; elimination never divides by lambda-beta_n*s. The normal quadratic operator has twenty roots, ten decaying into x<0. Independent first-order reduction constraints and frozen auxiliary B fields are not introduced.

For exp(lambda*t+s*x+i*k*y), select Re(s)>0 in the interior x<0. An ordered Schur basis avoids ill-conditioned individual eigenvectors. QR factors out the trace-basis conditioning before row-scaled boundary singular values are evaluated.

## Demonstrated original-boundary failure

For Re(lambda)>0, original zero-rate incoming characteristic evolution implies C_in=0. At k=0.1/M:

| Mode | Re(lambda), /M | Boundary defect | Full volume defect |
|---|---:|---:|---:|
| Scalar constraint | 0.0245537680007263 | 7.3e-16 | 1.7e-14 |
| Vector constraint | 0.0153808239020071 | 1.3e-15 | 4.4e-14 |

At unit maximum evolved-state amplitude, the scalar mode has Theta=0.01072, max Q=0.04958, H=0.00607 and max M=0.00664. The vector mode has Theta/H at roundoff but Q=0.05627 and M=0.00606. Q is evolved Gamma minus the metric-defined conformal connection. These are constraint-violating modes, not mere gauge relabeling.

The benchmark k=0.1/M is slightly above the Nyquist frequency pi/32=0.09817/M for 32M cells, so it must not be identified with a resolved production mode. A separate bounded search at k=pi/(8*32)=0.0122718463/M, or 16 cells per tangential wavelength, found three further positive real roots with the same frozen coefficients:

| Mode | Re(lambda), /M | Boundary defect | Full volume defect |
|---|---:|---:|---:|
| Scalar constraint | 0.00335801999846489 | 4.6e-14 | 4.4e-14 |
| Vector constraint | 0.00553795899218515 | 1.1e-14 | 6.9e-14 |
| Scalar constraint | 0.00725779440391440 | 1.6e-14 | 2.2e-14 |

At unit state amplitude the scalar modes have Theta=0.00236/0.00173 and H=0.000931/0.000605; the vector has Q=0.00317 and M=0.000326 with Theta/H at roundoff. These establish continuum boundary growth at a tangentially resolved frequency, while neither normal-profile resolution nor agreement with the full production discretization is established. The search was confined to positive real lambda in [1e-7,0.05]/M; it is not a complete complex-spectrum census.

An independent implementation reproduced the flat-limit root. The periodic volume symbol has no positive growth at the sampled frequencies. In bounded scans, removing kappa eliminates these boundary roots, while kappa alone without shift/lapse damping also has none. This supports a lower-order damping/boundary incompatibility; it does **not** justify removing volume damping. These frozen growth rates are not measured production rates, and the calculation does not identify the first production injection.

## Tested physical-constraint replacement and its obstruction

With D0=partial_t-beta dot grad, c=alpha sqrt(chi), sigma=alpha kappa1, the production frozen constraint equations imply

```
D0² Q - c² ΔQ + 2 sigma D0 Q = 0,
D0² Theta - c² ΔTheta + 2 sigma D0 Theta
  = -alpha chi sigma div Q.
```

The local target tested is

```
FQ = D0 Q + c Dn Q + sigma Q,
FTheta = D0 Theta + c Dn Theta + sigma Theta
         - sigma sqrt(chi) Qn/2.
```

The scalar coupling is the leading normal approximation. The exact zero-normal-shift exterior solution also contains kappa1 div_T Q_T/(2 rho), with rho²=(lambda²+2 sigma lambda)/c²+k_T²; that oblique term is nonlocal. No exact nonlinear absorbing-boundary claim is made.

The fast real constraint roots disappear in the bounded scan, but keeping the old gauge/TT rows admits an exact physical surface mode

```
lambda = beta_n*k/sqrt(2) + i*c*k/sqrt(2).
```

At k=.1 this is 2.2325e-5+i*0.0706634/M. Boundary/volume defects are 7.8e-16/2.7e-15. Physical constraints and lapse/shift perturbations are at roundoff; nonzero electric/magnetic curvature identifies physical radiation. Replacing the TT rows with frozen physical-Weyl conditions removes this specific mode, but the attempted gauge completions retain a curvature-free coordinate mode at lambda=beta_n*k. The tested completions include lapse/shift Dirichlet, outgoing gauge waves, mixed reference-frame h_nA anchoring, and mixed anchoring with normal-shift Dirichlet. None is a complete cure.

For the last variant an analytic counterexample is supplied by `coordinate_mode`: a coupled harmonic spatial/time coordinate displacement gives h_nA=alpha_delta=beta_delta=0 at the boundary, but an x*exp(kx) contribution to the metric and nonzero A. Q/H/M and linear vacuum Weyl curvature vanish; the full PDE defect is 1.25e-18. Thus the selected gauge rows are blind to an incoming coordinate freedom, rather than ten additional physical constraint conditions being needed.

The frozen physical-Weyl TT equation tested is

```
A_AB,t^TF = beta·∂ A_AB^TF - c Dn A_AB^TF
           + c [D_(A A_B)n]^TF - chi [D_A D_B alpha_delta]^TF
           + alpha chi [D_(A Q_B)]^TF.
```

The Q term removes the evolved-Gamma contribution to the Ricci tensor. This is a frozen linear condition, not an implemented nonlinear AthenaK boundary. The stored electric-curvature diagnostic is chi*delta Ricci_ij; it equals the vacuum electric Weyl tensor on the constraint-free modes where that interpretation is used, not on arbitrary constraint-violating states.

## Practical reflecting/outgoing comparators

The final bounded comparison imposed either all ten q fields Dirichlet or a common outgoing-wave condition `(D0+c Dn)q=0`, with p inherited from the unchanged q_t equation. Neither is a CPBC or a production fix. No right-half-plane root was found in the sampled weak/strong G1 bands. Dirichlet stayed well separated (minimum sampled singular value about .970 weak/.205 strong); common outgoing approached a small singular value near lambda=0, so no uniform stability claim follows. `comparisons.json` records the bands and minima. These need compatible discrete evolution tests; resetting fields is not an equivalent implementation.

Raw exploratory scans remain outside the repository. No further gauge variants are included in this package. The useful conclusion is the demonstrated original boundary failure and explicit requirements a replacement must satisfy—not a passed vacuum or matter stability gate.
