# Held total-J Cartesian-component control recipe

This is preparation only. No scientific source has been compiled, no radial coefficient action or operator has been generated, and no evolution has been launched from this directory. Admission requires both a byte-pinned immutable total-J basis and an explicit root review/release recorded in `admission.json`. A basis working file or successful basis-only checker is insufficient.

The eventual control would use the actual continuum C0 equations and the frozen spatial-norm gauge that produced the saved approximate N16/N20 modes. It would retain Cartesian scalar, vector and symmetric-tensor components on the full spherical ball. Radial discretization, the origin equations and the characteristic treatment at scri remain separate decisions after the angular closure gate. No inner boundary, annulus, generic-Theta falloff, floor, SPD repair, or source modification is implicit in this recipe.

## Fixed baseline and source binding

Use S=1, a=.5, geometry cutoffs (.05,.95), gauge cutoffs (.45,.85), physical_trace_lapse=true, preferred_source=false, xi=2, eta=6, C=2/3, C0 kappa_input=10 and kappa2=0. The physical-P storage/evolution is unchanged: the stored trace is P=Kphys-2Theta and Theta is physical. Production runtime source identity is `27c19d20696ea6dd4704032c51dfd026218f64f2`; the launch documentation HEAD is a separate provenance field. The exact source hashes are in `source-pins.json`.

The generic dual reference/gauge binding is `Reference` and `Gauge(...,norm=true)` from the frozen `covariant-z4-candidate/immutable-C1-stiffness-20261009/full20.cpp`. Only its C0 branch (`form=0`) is admissible. Do not include the C1 additions, reinterpret another candidate as the baseline, or use the C1 driver's Fourier matrix routine as an angular closure test. Its dual spatial-jet machinery comes from the frozen `discrete-bianchi/immutable-discrete-bulk-20261009/dual_helpers.hpp`.

The private native `native_injection.hpp` dispatches the spatial-norm helper only for `T=double`; a dual scalar silently takes its production-gauge fallback. This is a hard binding failure, not a harmless implementation detail. The new local bridge must call the exact generic dual spatial-norm formula, and independently compare against the actual double `spatial_norm::Gauge` and `spatial_norm::Assemble` helper. The negative control must show that the double-only wrapper would omit a nonzero directional spatial-norm response. Add beta pole/Omega exactly once; the production gauge assembler itself omits this pole.

Use actual `ConformalRHS`, `AssembleInterior`, live stationary Omega-normal construction, reference jets and the native damping normalization kappa_input/alpha. Preserve the actual fixed analytic Minkowski source subtraction when checking the reference. Its derivative is zero, but that does not authorize changing the live gauge or subtracting an arbitrary live residual. Capture all compiler arguments and transitive dependency hashes when compilation is eventually admitted.

## Independent fields and physical metric lift

Use four spin-0 sectors (alpha, P, Theta and the trace of delta bar-gamma), two spin-1 sectors (beta and Lambda), and two spin-2 sectors (the STF part of delta bar-gamma and independent STF curvature data S). There is no independent chi in addition to all six physical metric components. Here bar-gamma=gtilde/chi, and a scalar metric basis uses delta bar-gamma=phi*I/sqrt(3), with this normalization recorded explicitly.

For an arbitrary symmetric h=delta bar-gamma, the algebraic tangent lift is

    delta chi = -(chi/3) * bar-gamma^{-1}:h
    delta g   = chi*h + (delta chi/chi)*g
    bar-gamma^{-1} = chi*g^{-1}.

Apply these formulas as full spatial-jet algebra, retaining every first and second derivative of the analytic reference coefficients. A value-only conversion followed by differentiating only the harmonic seed is invalid in the layer.

For Euclidean STF input S, use

    Pi_g(S) = S - (g/3)*(g^{-1}:S)
    delta A = Pi_g(S) + (g/3)*(Aref^{ij}*delta g_ij)
    Aref^{ij} = g^{ik} Aref_kl g^{lj}.

This enforces g^{-1}:delta A=Aref^{ij}delta g_ij, including nonzero background A. Retain the first derivatives of this map, as consumed by the actual kernel. Metric/chi and lapse/shift jets require the consumed second derivatives. Lambda stays independent; setting Lambda to contracted Gamma would impose a differential constraint. The coordinate-zz native chart may be used as an independent algebraic-projector oracle, but is not the primary harmonic lift.

Check the full 22-component input/output tangent identities before reducing to 20 independent fields. Record both the raw algebraic-normal RHS residual and its native projector image. Do not silently replace a large normal residual by projection. Native final-only finite-RK projection and a projected continuous generator are distinct; no finite-RK map is being prepared at this phase.

## Basis contract and angular extraction

The planned external API is `totalj::EvaluateBasis<T>(J,spin,L,x,WJet{W,W_rho,W_rhorho})`, rho=x.x. It returns Cartesian value/first/second jets of W_L(rho) times a regular solid coupled harmonic. Spin-2 output is the full 3x3 STF tensor, with off-diagonal multiplicities retained in contractions. The external symbolic data must retain every m needed for an independent evaluator; the primary generated C++ API may be real m=0.

For J=0,1,2 the sector dimensions are 8,16,20. At each positive radius evaluate three independently prescribed local radial-envelope jets per amplitude: (W,W_rho,W_rhorho)=(1,0,0),(0,1,0),(0,0,1). J=2 therefore has **60 input radial-jet actions**, not 60 scalar matrix entries. A0,A1,A2 each contain 20x20 scalar coefficient functions. For J=0 and J=1 there are 24 and 48 actions respectively.

The first extracted matrices are explicitly B0,B1,B2 in `Wdot=B0*W+B1*W_rho+B2*W_rhorho`. If later named A0,A1,A2 for r derivatives, record the conversion at r>0: A0=B0, A1=B1/(2r)-B2/(4r^3), A2=B2/(4r^2). This conversion is for W itself; its solid-harmonic r^L factor has already been differentiated by the Cartesian basis. Never use these radial divisions at the origin.

Stack native 22-component value maps on deterministic oblique directions with all components nonzero. Fit each RHS action against the stacked tangent basis using rank-revealing QR or SVD; use neither axis-only fitting nor normal equations. Record the raw and column-scaled singular values, numerical rank, condition estimate, scaling convention, absolute residual and residual normalized by the actual action. Powers of r^L may be removed by an explicitly recorded exact column scaling. Reject ill-conditioned or rank-deficient extraction; do not hide it by increasing a least-squares tolerance.

Proposed radii include the Cauchy core, both transition endpoints, the interior layer and the outer CMC region. Use positive-radius angular extraction at .025,.05,.1,.2,.3,.45,.6,.8,.85,.9,.95,.98 and the original N16 closest-shell radius; record actual Omega and scales. Independently test unused oblique angles, rotated Cartesian components and m=1 (also J=2,m=2). The same radial coefficient matrices must predict these tests in the declared real/complex CG convention. A fit residual alone does not establish angular closure.

At r=0 the value matrix loses rank because solid harmonics of L>0 vanish. Do not run the positive-radius fit there and pretend it determines all channels. Test the analytic Cartesian polynomial/Taylor action at the origin and the r->0 action on regular envelopes. An eventual radial representation must explicitly include the origin and its regularity equations.

## Required local gates after release

1. Verify every immutable basis and baseline pin; capture source-only extraction differences and compiler/dependency provenance. Validate the generic-dual gauge against actual-double directional finite differences at the analytic reference and nontrivial SPD finite-Omega perturbations. Exercise all 20 free columns, nonzero metric/chi norm response, A trace coupling and xi=2. Use several amplitudes and expose reference cancellation rather than accepting a single epsilon.
2. Check the physical tangent lift, its spatial derivatives, the full native algebraic projector and the 22-component RHS normal residual. Check actual reference constraints and source-subtracted stationarity separately from the tangent calculation. No floors or metric repair.
3. Use an independent Cartesian derivative oracle for W(rho) solid harmonics. In particular, componentwise

       Laplacian[W(rho)*solidY_L] = [4*rho*W_rhorho+(4*L+6)*W_rho]*solidY_L.

   Test origin values and derivatives directly, not with radial divisions.
4. In the exact Minkowski Cauchy core, test the TT wave h_xy=f(z-t), delta A_xy=f'(z-t)/2. At t=0 with f=z^2, h_xy=z^2 and A_xy=z; the actual linear RHS must give hdot_xy=-2z and Adot_xy=-1, with all other fields and physical constraints zero. This full-kernel oracle need not be confined to the J<=2 truncated representation. Add manufactured pure lapse/shift gauge jets with initially zero constraints and coefficient-aware continuum Cdot=0; a primitive frozen-symbol QL check is not a replacement.
5. Extract the A0/A1/A2 actions only after the independent local gates pass. Require held-out-angle, independent-m, rotation and core manufactured tests; retain any failure and its conditioning/cancellation evidence. Release and ASan/UBSan local builds are separate receipts. Proposed numerical tolerances are to be chosen and recorded before extraction, with absolute/relative scales and conditioning explicitly accounted for.

## Scope and later decisions

The angular closure gate concerns the actual continuum full-tensor equations with analytic jets and spherical reference coefficients. Cartesian finite-h independent Dxx, mixed DxDx, Lx and KO stencils are anisotropic and do not preserve a fixed total-J subspace. They cannot silently be retained in a radial harmonic operator. A later spherical-ball discretization changes the bulk spatial discretization as well as the embedded boundary. It would need its own constraints, origin and scri compatibility tests and cannot by itself isolate a unique boundary cause for the saved Cartesian mode.

No radial operator, spectrum, propagator, crossing-time result, global stability, continuum instability or whole-Z4c boundary conclusion is claimed by this held preparation. No native evolution is authorized here.
