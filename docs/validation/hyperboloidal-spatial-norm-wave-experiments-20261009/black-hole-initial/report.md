# Detached black-hole initial compatibility: live spatial-norm shift source

Scratch audit only, 2026-10-09. The production runtime remains unchanged. The
geometry and beta-only proofs frozen in the preceding 69-file index are
unchanged. This report audits necessary conditions at the initial instant;
it establishes neither a preserved nonlinear scri manifold nor a native black-hole
evolution. The finite Minkowski pulse gate and later wormhole-to-trumpet
acceptance remain separate.

The actual Minkowski reference is S=1, a=.5, width .05–.95. The detached
M=.5 black-hole data retain that same compactification and use their own
.30–.95 height cutoff. The metric, A and Lambda are the black hole's own
derived ADM fields. The outer initial lapse equals the geometric static
lapse; the positive pre-collapsed interior lapse is retained. No BH RHS or
BH gauge-source subtraction is introduced.

## Source and conventions

The candidate retains the complete coupled production shift and physical-P
lapse principal parts, with preferred source off and xi=1/a. It adds only
the shift pole numerator

```text
S_beta^i = -eta W [ beta^i-beta_ref^i + C n^i (G/Ghat-1) ],
G = chi*gtilde_inverse^ij*Omega_i*Omega_j,
Ghat = chi_ref*gtilde_ref_inverse^ij*Omega_i*Omega_j,
n^i = -delta^ij*Omega_j / sqrt(delta^kl*Omega_k*Omega_l),
eta = rho*S/a^2,    C = (S/a)*(1-1/rho),    rho=3/2.
```

W is the existing production gauge cutoff (.45–.85), not the geometric
height cutoff. The normal n is the outward Euclidean radial unit vector
for this monotone radial Omega. The feedback responds to the live spatial
geometry. Its exact difference is evaluated as

```text
G-Ghat = [(chi-chi_ref)*gtilde_ref_inverse^ij
            + chi*(gtilde_inverse^ij-gtilde_ref_inverse^ij)] Omega_i Omega_j.
```

This avoids subtracting two complete spatial norms on the reference. The
source uses no Omega floor, projection, imposed evolved-field falloff or
mass-dependent target. The scratch header returns the original gauge before
forming n when W=0. The 101-point oblique Minkowski reference check and the
exact Cauchy-core checks pass.

Here alpha is the conformal lapse and beta_rad is the contravariant Cartesian
shift contracted with n. With Omega fixed in time,

```text
omega_n = -beta^i Omega_i/alpha,
Q = (P-3*omega_n)/Omega,
N_raw = G-omega_n^2,
dt N_raw = chi_dot*gtilde_inverse^ij*Omega_i*Omega_j
             -chi*v^i*gtilde_dot_ij*v^j
             +2*omega_n/alpha*(beta_dot^i Omega_i+omega_n*alpha_dot),
v^i = gtilde_inverse^ij Omega_j.
```

S, a and M have length units; xi and eta have inverse length units; alpha,
beta, Omega, C, a1 and b1 are dimensionless; Q has inverse length units and
N_raw has inverse length squared units. At scri omega_n=-1/a. The geometric
term in dt N_raw is retained throughout.

## Initial leading conditions

For the fixed BH ADM jets, write alpha-alpha_ref=a1*Omega+... and
beta_rad-beta_ref_rad=b1*Omega+..., with chi=1-2M*Omega/S+.... The initial
quadratic null falloff requires a1+b1=M/a; its conformal-trace limit is

```text
Q_scri = -3/S + 4M/(aS) - 3(a1+b1)/S.
```

The new source changes the beta-only leading formula to

```text
alpha_dot0 = S/a^2 * [b1-2a1-4M/a],
beta_dot0  = S/a^2 * [a1+b1+M/a] - eta*[b1-2CM/S],
chi_dot0   = 2M/(3a^2)-2a1/a-4b1/(3a),
g_rr_dot0  = 8M/(3a^2)-4b1/(3a),
g_tt_dot0  = -g_rr_dot0/2,
P_dot0     = 3/a^2*(a1+M/a),
N_raw_dot0 = (chi_dot0-g_rr_dot0)/a^2
                +2/(aS)*(alpha_dot0+beta_dot0).
```

The original geometric gauge jets a1=-M/a, b1=2M/a make every displayed
rate zero for any eta>0 when C=(S/a)*(1-S/(eta*a^2)) and xi=1/a. The physical
Theta rate is also zero. No individual live chi/g/A/Lambda boundary value
is pinned by this argument. For the BH initial data the vanishing geometric
rates follow from the equations and chosen jets.

Both gauge pole numerators have zero initial time derivative:

```text
dt [delta beta_rad+C*(G/Ghat-1)]_scri
       = beta_dot0+C*(chi_dot0-g_rr_dot0) = 0,
dt S_alpha|scri
       = -alpha0^2 P_dot0-alpha0*(2xi+1/a)*alpha_dot0
             +alpha0/a*beta_dot0 = 0.
```

An independent rho=3/2 necessary value-branch elimination confirms the
positive null, finite-Q, Theta=0, finite-gauge-RHS branch is alpha0=alpha_ref,
beta_rad0=beta_ref and G0=Ghat. In normalized variables x=alpha/alpha_ref,
y=beta_rad/alpha_ref and z=sqrt(G/Ghat), the shift/null/lapse relations give
y=-(z^2+2)/3=-xz and x^2=1/(2-z). Elimination yields

```text
(z^2+2)^2*(2-z)-9z^2
  = -(z-1)*(z^4-z^3+3z^2+4z+8).
```

The final factor equals z^2*[(z-1/2)^2+11/4]+4z+8>0 for z>0; hence z=x=1
and y=-1. Tangential beta components are separately fixed by their pole.
This is an algebraic necessary value condition, not its evolution proof.

## Next null coefficient and the corrected initial gauge

The unmodified complete BH jets do not preserve quadratic null falloff at
the initial instant. At S=1,a=M=.5 and xi=2 the exact rational result is

```text
dt N_raw = [7eta/2-50+(64-4eta)*delta_b2]*Omega + O(Omega^2),
```

where delta_b2 is an additional radial beta coefficient multiplying Omega^2.
An additional alpha coefficient delta_a2*Omega^2 does not change this term.
The independent 100-digit oracle gives:

| eta | C | Original Omega coefficient | Cancelling delta_b2 | Corrected Omega^2 coefficient |
| ---: | ---: | ---: | ---: | ---: |
| 5 | .4 | -32.5 | 65/88 | -95.25 |
| 6 | 2/3 | -29 | 29/40 | -92.7 |
| 10 | 1.2 | -15 | 5/8 | -76.5 |

For the scaled rho=3/2 candidate, add

```text
beta^i_initial += SmoothCutoff(r,.97,.99)*(29/40)*Omega^2*n^i.
```

This preserves the first gauge jets, the positive lapse, the BH physical
spatial metric, K/P/A, Lambda and mass. It changes the time-coordinate
threading from the exact static outer Schwarzschild gauge. The resulting
initial expansions are N_raw=3.9*Omega^2+... and
dt N_raw=-92.7*Omega^2+..., while Q_scri=-2 is unchanged. It is not a
stationary black-hole solution. At eta=16 this particular second-jet-only
correction cannot cancel the remaining +6*Omega coefficient because its
linear coefficient vanishes. The audit does not advocate that parameter.

## Connection, shear and all exposed geometric poles

The added beta Hessian gives a finite nonzero initial boundary rate

```text
Lambda_dot^r = Gamma_tilde_dot^r = 8*delta_b2/(3a^2),
```

which is 116/15 for eta6. It is required by the evolving metric connection;
clipping or pinning Lambda to zero would create Z. The actual Lambda equation
and an independent one-sided derivative of the complete metric RHS agree
with this value. Thus Z_dot0=0.

The complete geometric first time jets from this correction are

```text
partial_r chi_dot  =  2*delta_b2/(3a^2),
partial_r g_rr_dot =  8*delta_b2/(3a^2),
partial_r g_tt_dot = -4*delta_b2/(3a^2),
partial_r gamma_bar_rr_dot =  2*delta_b2/a^2,
partial_r gamma_bar_tt_dot = -2*delta_b2/a^2.
```

Consequently the radial and tangential physical-spatial connection time
derivatives both equal delta_b2/a^2. Their contribution to the physical
Hess(Omega) time derivative is isotropic, so its tracefree part vanishes.
The initial nonzero mass shear is retained: A_rr=-4M/(3aS),
A_tt=2M/(3aS), and Hess(Omega)_rr^TF=-4M/(3a^2S),
Hess(Omega)_tt^TF=2M/(3a^2S). These cancel in the actual A pole
2alpha*Hess(Omega)^TF+alpha*A*(P-omega_n).

The correction changes P advection by delta_beta*P_r=O(Omega^2), leaves
Theta=0, and changes A Lie/advection terms by O(Omega). Thus P_dot0,
P_dot_r0, Theta_dot0, Theta_dot_r0 and A_dot0 are zero. Along with
omega_n_dot0=G_dot0=Z_dot0=0 and the isotropic Hess(Omega) time derivative,
this makes the initial time derivatives of the exposed chi/P/Theta/A/Lambda
pole numerators zero. These statements use the audited exact outer initial
ADM data. They impose no arbitrary live-field vanishing assumption and do
not establish higher-time compatibility.

## Executable evidence and limitations

`run_audit.py` writes `receipt.json`, exact compiler invocations, source
hashes, binary hashes, stdout/stderr and durations. It executes the actual
kernel in Release and Debug with Address/UndefinedBehavior sanitizers, plus
three independent exact/100-digit proof scripts. Both kernel runs contain
48 outer rows, 101 reference points, oblique first/second derivative checks,
leading Lambda/connection/time-jet checks and exact scri pole checks.
The final receipt has all five checks passing in 2.31647 seconds of test
execution, 4.64032 seconds including both builds, at source HEAD
`f615acf4356206eceddc59fa929fcc15a671fe09`. The final Release/Debug binary
SHA256 values are `429881be3f994279a5bfd4fb778b16bed2d42ccb02d8808b6d682cf993769020`
and `a9dead648ca446b7b60e013ac77dfdddd7454070437ab7b1e9acfd4a8433427f`.

Both actual-kernel audits pass: H maximum 1.954e-14, M maximum 8.882e-16;
raw/factored gauge/null-rate agreement 9.809e-11 at Omega>=1e-4; exact scri
pole residual maximum 3.553e-15. Consumed beta first/second derivative relative
errors are 3.959e-10/2.092e-4 at FD step 1e-6. The initialized lapse remains
positive; its minimum over the tested .96–1 collar is 1.844704614.

An intermediate rerun after adding the rho1.5 value-branch proof failed a
structural SymPy equality of equivalent polynomial factorizations. Both
actual-kernel modes passed in that rerun. `failed-structural-proof/` preserves
the failed receipt, exact proof source and diagnostic; the assertion now
checks that the simplified algebraic difference is zero. The final receipt
reruns all five checks against the corrected proof bytes.

Raw double-precision assembly is retained as negative evidence, not used
as a limiting proof. At Omega≈1.9984e-15 the eta6 full null rate is about
17.4815 for both original and corrected jets, whereas the 100-digit factored
limits vanish. Raw P/Theta rates are about 9.77778 there. At this scale the
complete pole numerators lose cancellation before division by Omega; no
floor or artificial zeroing was added. Factored double precision also loses
the corrected Omega^2 coefficient once it falls below roundoff; the
100-digit oracle supplies that coefficient independently.

The live source in `spatial_norm.hpp` was read-only compared with
`build-layer-research/continuum/preferred/native-overlay/spatial-norm-family/spatial_norm_control.hpp`;
the mathematical source, exact reference difference, outward normal, xi2,
eta6 and C2/3 agree. The latter's Fourier/native work is independently owned
by the continuum agent. This audit ran no native evolution and makes no
stability claim. A future implementation still needs an actual boundary
closure preserving the simultaneous gauge/null/trace/Z/shear conditions
under general evolved and angular perturbations. The already documented
finite-Q counterexample is not eliminated by these initial-data tests.

The new `frozen-index.json` records the completed current sources/reports,
receipts/logs and binaries by hash and byte count. The old 69-file index is
verified before and after the new run. Production source hashes are also
unchanged throughout the run.
`value_branch_review.md` separately records the read-only independent
review of root's monotonicity proof for the full 1<=rho<=5/2 interval;
its domain and endpoint argument need no correction.
