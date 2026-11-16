# Scheme A: bulk–boundary analysis, 2026-11-16

This investigation derives a constraint-compatible radial regularity branch and gives
a principal-level energy argument for the incoming spatial-gauge sector.
It does **not** certify the full nonlinear, regular-singular, three-dimensional
initial-boundary-value problem or its native discretization.

The main new result is a constraint-compatible formal power-law branch whose
smoothness depends on **both** lapse damping and shift restoration. Increasing
lapse damping alone admits less-smooth branches. Applying the linearized
production normal closure to these modes reproduces the observed slow error
orders. The previous coarse-grid damping gains must not be interpreted as
earlier entry into high-order convergence.

No production source or default has been changed. Source kernels are those of
`f90ab38472405907f21347afc19a7c9e90a1d665`; the analysis was developed after local
test commit `5919dbe47`. Raw matrices, commands, executable hashes, exact symbolic
checks and plots are in:

`/Users/hz0693/research/results/bulk-boundary-20261116`

## 1. What the principal boundary calculation establishes

The full 26-field normal principal limit is reconstructed from the actual
`EvaluateBulkPoint` kernel, with algebraic dependent fields expanded for every
perturbation. The reduction uses normal derivatives of the twice-differentiated
fields, and P/Omega and Theta/Omega for the two trace variables. For
`q_t = P_n d_n q`, the eigenvalues of the reconstructed exact rational limit are

| Eigenvalue | Algebraic multiplicity | Eigenspace dimension |
|---|---:|---:|
| -2 | 7 | 7 |
| 0 | 7 | 7 |
| -1 | 6 | 6 |
| -1-sqrt(2), -1+sqrt(2) | 1 each | 1 each |
| -1-sqrt(3/2), -1+sqrt(3/2) | 2 each | 2 each |

Thus this reference normal limit has a complete basis, including the
coincident light/lapse families. This does not remove the separately reported
defective crossing inside the gauge blend near r=0.8197.

The three incoming shift eigenvectors are controlled by the principal
time-differentiated shift-value conditions. In the eigenvector normalization
used by the exact calculation, their boundary matrix is diagonal with entries
`9/4-3 sqrt(2)/2`, `3/2-sqrt(6)/2`, `3/2-sqrt(6)/2`, all nonzero.
This is a normal-incidence rank result, not by itself a boundary-stability proof.

There is a stronger statement for the homogeneous incoming spatial-gauge
sector. Let x be the outward normal coordinate, with the domain x<0, and xi
an infinitesimal spatial coordinate displacement. At the normalized reference
scri limit, its principal fields satisfy

    delta chi = -(2/3) div xi,
    delta g_ij = d_i xi_j + d_j xi_i -(2/3) delta_ij div xi,
    delta beta = (d_t+d_x) xi,
    delta Lambda = delta D = Delta xi +(1/3) grad div xi.

The lapse, trace, Theta, A and B principal amplitudes vanish in this sector.
Substitution into the full coupled kernel symbol yields

    (d_t+d_x)^2 xi = mu Delta xi +(mu/3) grad div xi,   mu=3/2.

The actual projected values give

    delta beta_x +(delta chi-delta g_xx)/2 = d_t xi_x = 0,
    delta beta_T = (d_t+d_x) xi_T = 0.

With homogeneous initial normal boundary displacement, xi_x=0 at the boundary.
After expanding the convective square, a positive energy is

    E = (1/2) integral [ |xi_t|^2 +(mu-1)|xi_x|^2
                        +mu|grad_T xi|^2 +(mu/3)|div xi|^2 ].

Here `xi_x` in the energy denotes the vector normal derivative; in the boundary
conditions it denotes the normal component. The outward flux is

    F_x = |xi_t|^2 -(mu-1) xi_t . d_x xi
          -(mu/3) (xi_t)_normal div xi.

At the homogeneous boundary it reduces to `mu |(xi_t)_T|^2 >= 0`, so this
principal gauge subsystem has a dissipative boundary condition for arbitrary
tangential dependence. `analytic_checks.py` checks the algebraic flux identity.
`principal_gauge.py` independently verifies the coordinate-mode embedding
against all 26 kernel equations, then checks 164 Laplace/Fourier samples.

This argument assumes the homogeneous outgoing/constraint forcing in this
sector. It is **not** a uniform estimate for the complete singular-source
system, arbitrary evolved backgrounds, or the discrete lift.

## 2. The singular sources change the regularity question

An ordinary principal symbol freezes Omega>0 and discards lower-order terms.
That does not control the simultaneous approach to scri, because the damping
and restoration terms contain 1/Omega. We retain those terms in a radial
indicial calculation.

Set rho=1-r, so Omega=rho+O(rho^2). At the normalized flat/reference endpoint,
take the constraint-compatible leading ansatz

    delta chi = C rho^s,       delta g_xx = G rho^s,
    delta alpha = a rho^s,     delta beta_x = b rho^s,
    delta P = P rho^s,         delta A_xx = A rho^(s-1),
    delta Lambda_x = delta D_x = -s G rho^(s-1),
    delta Theta = delta B_x = 0.

The other diagonal metric and A amplitudes are -G/2 and -A/2. Equal D and
Lambda here means equal **leading coefficients**, not an imposed identity
throughout the evolution. The lower-order auxiliary relaxation can enter the
subleading coefficients.

Write k for `outer_lapse_damping` and R for the outer shift `restoring`
coefficient. The current R is 4. The two metric evolution equations, the
Hamiltonian and radial momentum constraints, and the two gauge equations give
the following six exact leading equations:

    s C +(2/3)P -2a +(2s/3-2)b = 0,
    s G -2A -(4s/3)b = 0,
    (3-s)A +(2s/3)P = 0,
    (s-2)(s-3)G +2(s-3)(s+1)C -4P = 0,
    (s-R)b -(3s/2)G = 0,
    -P +(s-1-2k)a +b = 0.

The first two follow directly from `ConformalRHS`. For example, the leading
normal-Omega perturbation is delta w=(a+b)rho^s, which supplies the -2a-2b
terms in the chi equation. The constraints follow by linearizing
`EvolvedConstraints`: their leading coefficients are respectively
`(s-2)(s-3)G+2(s-3)(s+1)C-4P` and `(3-s)A+(2s/3)P`.
The last two follow from the outer gauge's `-R(beta+x)/Omega` and lapse pole.
These six equations are independent of the numerical matrix reconstruction.

Their determinant is

    -8 s (s-3) [ k s^2 + k(R+1)s - R(k+2) ].

Consequently, for R=4, the positive gauge exponent is

    s_+(k) = (-5 + sqrt(41+32/k))/2.

This is an exact leading-order algebraic result. It identifies a **formal
Frobenius branch**, not a constructed global nonlinear solution. The full
10-field scalar pencil extracted independently from the point kernels has
the same factor:

    (8/15) s (s-3) (s-1)^2 (31s-300)
           [ k s^2 +5k s -4k -8 ].

Its branch vector satisfies leading Theta=0, the connection constraint,
Hamiltonian and momentum constraints, and leading D-Lambda=0. Thus this
mechanism cannot be dismissed as merely a leading constraint-violation mode.

## 3. Why stronger damping can reduce the observed order

For the noninteger branch, normal differentiation of rho^s at the endpoint
has a generic finite-difference error proportional to h^(s-1). The lapse clock
is sensitive to these normal derivatives, even though alpha=1 is projected
exactly at the endpoint.

| k, with R=4 | s_+ | Predicted noninteger clock order s_+-1 | Manufactured closure order, finest pair |
|---:|---:|---:|---:|
| 0.25 | 4.000000 | integer branch; generic power estimate cancels | leading polynomial cancellation |
| 0.375 | 3.119905 | 2.119905 | 2.119994 |
| 0.5 | 2.623475 | 1.623475 | 1.623582 |
| 0.75 | 2.073474 | 1.073474 | 1.073672 |
| 1.0 | 1.772002 | 0.772002 | 0.772753 |

The earlier actual control evolutions give orders approximately 1.61 for
k=0.5 on N64 to N128, 1.08 for k=0.75 on N32 to N64, and 0.77 for k=1 on
N32 to N64. These agree closely with the branch prediction. Agreement is
evidence for the mechanism, not a unique decomposition of the evolved errors.
The shorter k=0.375 sweep had a steeper coarse trend; it does not exclude a
later crossover to the noninteger branch.

For s>2, all jets required by the boundary point kernel for this leading
perturbation vanish at scri. Its values obey the projection, its leading
auxiliary difference vanishes, and its normal compatibility conditions are
homogeneous. Scheme A therefore does not exclude this formal branch through
those pointwise conditions. For k>0.8, s_+<2: a nonzero branch amplitude would
instead fail the classical finite-second-jet regularity needed at the endpoint.

At k=0.25 the leading exponent is the integer 4, and the endpoint interpolant
differentiates the corresponding polynomial exactly. One must not assign it
order 3 simply by substituting into the noninteger estimate. However, the separate static recurrence below does require a logarithm when
a particular mass-like branch is present; polynomial cancellation alone is
not a smoothness certificate.

`mode_test.py` applies the manufactured leading vector to the 26-field
linearized projection, seven-node normal rows, three-variable normal solve,
and polynomial lift, including its last-two-bulk-layer footprint. It uses
the production origin-r^2 chi convention. It records 30 combinations of
damping and spacing. The data are local frozen-face tests; their tangential
derivatives are continuous local Fourier derivatives, not native angular
interpolation or cubed-sphere seam exchange.

### A static resonance at the original parameters

At (k,R)=(1/4,4), the scalar roots s=3 and s=4 differ by an integer.
The first spherical recurrence has been checked both with exact linearized
radial equations and with the independent full point-kernel Jacobian. In the
scalar ordering (chi,g_rr,A_rr,P,Theta,Lambda_r,alpha,B_r,beta_r,D_r), the
s=3 leading vector, normalized to unit g_rr amplitude, is

    v3 = (2,1,21/2,0,0,-3,3,0,-9/2,-3).

The s=4 gauge vector, normalized to unit lapse amplitude, is

    v4 = (1/2,0,10/3,5/4,0,0,1,0,-5/4,0).

Component powers have the same weights as above: A, Lambda, B and D have
one fewer power of rho. The static first recurrence requires

    q = rho^(3+w) v3 + rho^(4+w) [v_sub + (120/13) log(rho) v4] + ... .

Thus a pure power series with this nonzero leading branch is inconsistent at
the next order. The auxiliary equation matters: D-Lambda=-rho^3+... at
this normalization, rather than zero. For example, the lapse perturbation
contains `(120/13) rho^4 log(rho)`. This is a first formal static recurrence
about the flat reference, **not** a construction of a global solution or
proof that the control/pulse evolutions contain this branch. The s=3 branch
is called mass-like here because it is constraint-compatible at the mass
falloff order; its normalization is not a calibrated physical mass.

`resonance_exact.py` expands the two spherical metric equations, Hamiltonian
and momentum constraints, lapse and shift equations, retaining the auxiliary
relaxation correction. Its six next-order equations give exactly 120/13.
`resonance.py` retains spherical angular jet terms and reconstructs the full
10-field first recurrence from the production point kernel. Independent
radius/perturbation choices and the wormhole reference recover this static
coefficient within 7.2e-4 (absolute). The exact result is 9.23076923.
The time-dependent expression in the exploratory JSON is not an independently
validated result and is not used here.

This adds a concrete resonance obstruction to any blanket claim of smoothness
at the original parameters. Further recurrence/corner analysis is still
needed, including whether the tested initial data excite its amplitude.

## 4. Independent kernel and discretization checks

The point Jacobians include values, all first derivatives and all symmetric
second derivatives of every independent field. Signed perturbations and their
halves are combined with Richardson extrapolation. Independent radius sets
and noninteger test exponents check the inferred leading matrices. The maximum
independent extrapolated matrix discrepancy is 6.25e-7; the normal principal
matrix discrepancy is 3.33e-8. The actual wormhole reference gives the same
leading pencil within 4.57e-7 in the checked cases. Exact symbolic determinants,
nullvectors and the independent six-equation derivation are retained separately
from these numerical checks.

The principal coordinate-mode identity agrees with the reconstructed kernel
to 6.3e-17. The 164 Laplace/Fourier samples have residuals below 1.4e-15; no
sampled incoming boundary matrix loses rank. The smallest sampled boundary
singular value is 1.20e-5 as the Laplace real part approaches zero. Finite
sampling is not a uniform Lopatinski estimate.

A full-field semidiscrete outer strip was also assembled, with the actual
normal reconstruction, projection, upwind correction, penalties and lift.
It has an artificial fixed inner boundary and continuous Fourier angles.
It is **not a stability certificate**. At width 0.06, its leading real
eigenvalue changes from -0.54 at 12 intervals to +5.92 at 24 and +8.73 at 48.
The growing mode is predominantly away from scri. At matched spacing 0.0025,
moving the artificial inner edge from 0.94 to 0.88 moves its centroid from
0.961 to 0.910 and changes the growth rate from 5.92 to 4.60. This makes the
strip result inconclusive about an intrinsic outer-boundary instability.
Small eigenpair residuals do not cure this modeling limitation. No native
angular/seam stability conclusion is drawn from it.

## 5. A parameter relation, not a recommended default

The joint condition for keeping the positive exponent at 4 is

    R = 20 k / (2-3k),    0 < k < 2/3.

It recovers the current (k,R)=(0.25,4). The pair (0.5,20) is therefore a
principled experimental comparator: it changes lower-order gauge terms while
preserving this leading exponent and the principal shift speeds. It does not
by itself establish acceptable transients, explicit timestep limits, other
indicial branches, or global smoothness. The optional `--restoring` argument
was added to the radial test driver only, with its old default unchanged.
Bounded control/wide-pulse runs and a timestep check are recorded in
`paired-receipts.json`; see `paired-summary.json` for their results. This pair
has not been adopted in production.

For the tested pair (0.5,20), the control peaks at N16/N32/N64 are
1.784e-2 / 2.115e-3 / 1.576e-4. The last control doubling has order about 3.75,
consistent with removing the fractional-order limitation in this window.
The corresponding wide-pulse peaks are 1.781e-2 / 5.489e-3 / 1.709e-3: there is
no uniform improvement over the original (0.25,4) wide-pulse results. Preserving
one indicial exponent is not sufficient to optimize accuracy or transients.
The default-R parity run reproduces its earlier scri history byte for byte.

## 6. Remaining gap and implications

The completed analysis establishes the incoming principal spatial-gauge
boundary energy statement, the constraint-compatible leading radial exponent,
the response of the actual linearized normal closure to that branch, and
a static first-recurrence logarithmic obstruction at the original parameters.
It does **not** establish:

- A uniform weighted energy or resolvent estimate for the entire coupled
  regular-singular system, including all constraint/tensor/tangent branches
  and the joint high-frequency / Omega->0 limit.
- A nonlinear existence/uniqueness result or a full corner-compatibility and
  logarithmic-resonance analysis on evolving backgrounds.
- Stability of the actual native angular/seam/interpolation operator and its
  RK implementation. The finite-strip inner boundary must first be replaced
  by an appropriate characteristic/matched closure for such a spectral audit.
- Removal of the separate interior constant-policy characteristic crossing.

The next diagnostic should fit the radial s=3, s=s_+, and resonant-log
amplitudes in evolved data, then test whether reference/source matching or
a jointly chosen asymptotic gauge can suppress the offending branch. That
would identify a targeted change before modifying the bulk equations. For
a stability claim, the full weighted regular-singular estimate, the interior
characteristic crossing, and the native angular operator still need resolution;
another unconstrained increase in lapse damping does not address them. For high-order accuracy, retain the current
asymptotic pair until a joint change has been independently qualified. Coarse
error reductions from damping alone remain real observations, but are not
evidence for improved asymptotic smoothness.

## Reproduction

Use Python with NumPy/SciPy and SymPy. `--sympy-path` can point to an existing
separate SymPy installation. From the simulation checkout:

```sh
sh tst/z4c_hyperboloidal/radial1d/build.sh EXISTING_KOKKOS_REPO PROBE_BINARY \
  "$PWD/tst/z4c_hyperboloidal/bulk_boundary/kernel_jacobian.cpp"
python tst/z4c_hyperboloidal/bulk_boundary/analyze.py PROBE_BINARY OUTPUT
python tst/z4c_hyperboloidal/bulk_boundary/analytic_checks.py OUTPUT/analytic-checks.json
python tst/z4c_hyperboloidal/bulk_boundary/principal_gauge.py OUTPUT
python tst/z4c_hyperboloidal/bulk_boundary/resonance_exact.py OUTPUT/resonance-exact.json
python tst/z4c_hyperboloidal/bulk_boundary/resonance.py PROBE_BINARY OUTPUT
python tst/z4c_hyperboloidal/bulk_boundary/mode_test.py PROBE_BINARY OUTPUT/mode-test
python tst/z4c_hyperboloidal/bulk_boundary/strip.py PROBE_BINARY OUTPUT/strip-n24-e0 --n 24
```

The kernel probe records commands and hashes for every matrix set. Data are
stored outside the manuscript repository; the paper and its main branch have
not been changed by this investigation.
