# Advisory assessment of a common-rho full-ball control

The proposed solid-harmonic times polynomial-envelope trial space is a sensible regular-Minkowski full-ball representation. It is not yet a radial PDE discretization or an outer closure. It changes the native bulk derivatives as well as the embedded boundary, so even a successful later control would not isolate primitive ghost errors by itself. The present assessment constructs no quadrature nodes, differentiation/PDE operator, boundary condition, spectrum or propagator.

## Origin freedom and polynomial truncation

For every allowed (J,m,L,spin), let F(x)=solid_(L,m)(x) W_L(rho), rho=x.x. Independent smooth W_L supply the complete regular Cartesian amplitudes in that total-J sector. The frozen CG/tensor representation already includes the component relations; imposing additional equalities or homogeneous values between distinct W_L at rho0 would remove legitimate freedoms. In particular, J1 includes constant Cartesian vectors and J2 includes constant STF tensors. A radial component parity rule alone would miss these channels.

With N interior nodes, polynomial interpolation gives N independent envelope coefficients per channel, through degree N−1. It retains that many regular origin Taylor coefficients, not every smooth function exactly at finite N. The maximum Cartesian degree in one channel is L+2N−2, so equal envelope degree is not a uniform Cartesian total-degree cutoff. Polynomial sequences can approximate smooth nonanalytic envelopes and their consumed derivatives; a finite polynomial ansatz does not by itself prove analyticity of a converged solution. It excludes singular/logarithmic/fractional origin behavior, which is appropriate for this regular Minkowski control and is not an eventual puncture/trumpet basis admission.

Reference metric/chi/A conversion must use the frozen full coefficient-jet lift. Smooth invertible radial conversions preserve Cartesian regularity and total J, but mix orbital channels. They must not be replaced by independent chi plus all six metric amplitudes or by a zero-reference-trace A projection.

Cancellation-safe identities are

    u_L=r^L W(rho),
    u_L'=r^(L−1)[L W+2rho W_rho],
    u_L''=r^(L−2)[L(L−1)W+2(2L+1)rho W_rho+4rho²W_rhorho],
    Delta[solid_L W]=solid_L[4rho W_rhorho+(4L+6)W_rho].

The last identity avoids subtracting individually singular centrifugal terms near the origin. The exact Cauchy core has smooth constant background coefficients, so the true continuum operator maps arbitrary regular Cartesian inputs to regular outputs. Apparent singular W coefficients there must cancel analytically; they cannot justify extra relations between independent W channels. Positive-radius angular extraction alone will not establish this limit. At small Gauss nodes the r^L rescalings and separate 1/r terms can amplify roundoff even though the field is smooth. A globally polynomial output envelope can also conceal a wrong pointwise cancellation by interpolation; independent origin Taylor tests remain necessary.

The barycentric D_rho differentiates the unique degree-N−1 interpolant. Its composed square is the same interpolant's second derivative in exact arithmetic. This is useful algebraic compatibility, not a discrete Bianchi or stability certificate. Variable coefficient multiplication, interpolation/projection, reference conversion and differentiation do not obey an exact finite-dimensional product rule. Nonpolynomial C-infinity layer/gauge coefficients are not globally analytic at their flat cutoffs, so exponential convergence must not be presumed. Aliasing, high-degree cancellation and coefficient-resolution errors need separate measurement.

## Common Jacobi nodes and norms

Under t=2rho/R²−1 the rho^(1/2) quadrature weight corresponds to Jacobi alpha0,beta1/2. Standard Jacobi orthogonality uses (1−t)^alpha(1+t)^beta; Gaussian rules use interior polynomial zeros. These facts specify quadrature, not a physical boundary condition. See [NIST DLMF18.3](https://dlmf.nist.gov/18.3) and [DLMF3.5(v)](https://dlmf.nist.gov/3.5#v).

Common nodes are convenient for coupled channels and are mathematically permissible. They do not make a common unweighted envelope norm the spatial field norm. With unit angular normalization,

    integral_0^R r² |u_L|² dr = (1/2) integral_0^(R²) rho^(L+1/2)|W_L|² drho.

Thus an L channel adds rho^L to the base quadrature weight. For degree-N−1 trial polynomials, the highest mass-product degree is2N−2+L. An N-point base Gaussian rule's degree2N−1 exactness therefore does not cover every top-degree L>=2 mass product. This is not a ban on common nodes: an exact dense mass matrix, verified overintegration, or a recorded approximate norm can be used. A reference-dependent full-field symmetrizer, if derived, would add further coupled weights; these spatial L2 observations do not supply a full Z4c energy.

For this plain polynomial mass product alone, an M-point base rule is exact when 2M−1>=2N−2+L, hence M>=N+floor(L/2). The suggested N+ceil(L/2) is also sufficient, conservatively one point higher for odd L. Neither formula establishes exactness of nonpolynomial reference/normalization/coefficient products or of the eventual coupled derivative energy.

For the bare rho derivative, weighted integration by parts also contains the derivative of rho^(1/2). A diagonal Gaussian weight does not automatically give the unweighted SBP identity H D+D^T H=endpoint traces. Its exact continuum counterpart is the endpoint term minus the mass form with derivative weight (1/2)rho^(−1/2). Any intended discrete SBP/SAT claim must be derived and verified for the actual coupled operator/norm, not inferred from the node family. The existing production radial_sbp.hpp proves an energy identity for a scalar outgoing characteristic on a shell with an inner SAT; its own header explicitly excludes a whole-Z4c closure. That certificate cannot be transferred here.

## What the polynomial ansatz does at the outer endpoint

An unconstrained degree-N−1 interpolant imposes neither W(R²)=0 nor W_rho(R²)=0. Its endpoint value and derivatives are linear extrapolation functionals of the interior nodal values. Collocating a PDE at all N interior nodes without boundary rows or penalties still selects a finite-dimensional extrapolative closure for incoming data; it does not establish a consistent physical inflow/outflow prescription. Keeping endpoints off the grid is not a characteristic or stability argument.

The physical interval must first be specified. A fixed R<S gives a finite-radius problem, whose actual incoming characteristic and constraint data require derivation. A Gauss interval ending at R=S excludes the endpoint node but still approaches scri under refinement; the simple/double poles and all compatibility conditions remain. A moving R_N<S adds a second limiting parameter. Record N, R, closest Omega, endpoint traces and coefficient conditioning separately, and do not mix these limit procedures. Polynomial smoothness of the finite-N interpolant does not enforce the required coupled R0/null/Qtrace/constraint Taylor relations at scri. Nor does it establish preservation of them.

No radial boundary condition is proposed here. The local full tensor angular gate must first identify the actual characteristic/gauge/constraint structure, and a later closure needs its own justified data and compatibility analysis. The native final-only raw22 RK map and the continuous projected20 generator are also distinct comparison targets; a reduced harmonic ODE cannot silently equate them.

## Useful weak-form alternative and essential later gates

A Galerkin/weak form with the same regular trial space would expose mass and boundary flux terms more directly and permit exact or overintegrated coefficient products. It would not fix the outer physics or establish a full Z4c energy. For the scalar Laplacian envelope,

    4rho W''+(4L+6)W'=4rho^(−L−1/2) d_rho[rho^(L+3/2)W'],
    (1/2)int rho^(L+1/2) U Lap_L(V)
        = [2rho^(L+3/2) U V']_0^(R²)−2int rho^(L+3/2)U'V'.

Origin flux vanishes for smooth envelopes. Dropping the outer envelope flux would impose W'=0, equivalent to u_L'=(L/R)u_L, not the same as a physical homogeneous Neumann condition or an outgoing wave condition. A weak form must retain that distinction and the complete coupled principal boundary terms. Its costs are a mass solve and controlled coefficient quadrature; it is an alternative control, not automatic stabilization.

Before any interpretation of growth rates, the essential gates are:

1. Pass the independent actual-kernel angular closure, all-m/layout/lift, reference and algebraic-normal gates already assigned separately. Derive the regular-core W action and origin limits explicitly; do not extrapolate a rank-deficient value fit at r0.
2. Verify barycentric first/composed-second derivatives and endpoint trace functionals on resolvable envelope polynomials. Test solid-harmonic polynomials through the available degree, all orbital channels, cross-channel Cartesian TT/pure-gauge combinations and held-out radii including the exact origin. Compare factored Cartesian jets to an independent oracle, and retain floating point conditioning/error scales.
3. Use the same rho derivative representation for the actual physical H/M/Z/Theta functional. Verify reference constraints/stationarity and manufactured Einstein/pure-gauge constraint rates. Measure finite-grid constraint closure errors with their coefficient/product-rule terms and convergence on matched analytic fields; do not demand exact closure of a separately discretized continuum identity at finite N.
4. Establish the physical domain and derive the outer characteristic/constraint/gauge prescription before forming a global operator. Then test any boundary form, its constraint coupling and any semidiscrete norm claim directly; no assumed staggered outflow or inherited scalar-SBP certificate.
5. Refine N and the boundary gap separately, monitor coefficient/envelope tails and independent quadrature/precision sensitivity, and resolve the C-infinity transitions. Only later compare matched norms/constraints and genuinely small operator residuals. Changing all bulk stencils means a difference from the Cartesian mode is an attribution control, not proof that ghosts caused it.

The narrow next decision is therefore to finish local angular/core extraction and decide the outer mathematical problem. Common-rho polynomial envelopes are an admissible representation proposal; the missing outer closure remains decisive. The separate FINITE-RADIUS.md checks a possible finite-radius incoming-characteristic principal algebra without supplying that missing bulk or constraint closure.
