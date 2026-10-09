# Advisory finite-radius harmonic boundary algebra

A deliberately finite r_b<S inside the exact CMC/harmonic collar is a more explicit mathematical control than silently extrapolating incoming data at off-endpoint Gauss nodes. It defines a different finite-radius problem. The calculation below checks the frozen boundary principal algebra, including the proposed momentum-only penalty, and does not construct a radial differentiation/PDE operator, choose physical incoming data, prove a Gauss/SBP energy identity, or admit a solver.

## Which twenty variables the principal certificate describes

The actual tst/hyperboloidal/kernel_symbol.cpp extraction is a normalized second-order/pseudodifferential principal reduction. It does not act on the twenty raw stored values. Ten entries represent normal derivatives of the configuration fields (alpha, chi, gtilde5, beta3); the other ten represent momenta V=(P/Omega,Theta_phys/Omega,Atilde5,Lambda3), with the metric-frame and alpha/chi normalizations used in that source. For example the lapse derivative is D_s alpha/alpha, the chi derivative is D_s chi/chi, and the shift derivative is D_s beta/alpha in the orthonormal spatial frame. The determinant/trace algebraic constraints are imposed before extraction. At a nonzero reference A, its linear trace coupling to the metric variation still belongs in the tangent lift.

For a reference-linearized boundary control all these are perturbation entries with the normalizations frozen at that reference: in particular V contains deltaP/Omega and deltaTheta/Omega. The principal lapse entry is delta(D_s alpha)/alpha_ref. Choosing instead D_s(deltaalpha/alpha_ref) changes a lower-order term proportional to deltaalpha and the background gradient. Such choices give the same principal matrix but different full boundary operators; the complete reduction/data definition must be specified before an actual closure is claimed.

In source ordering the derivative entries are q_indices=[0,1,2,7,8,11,12,15,16,18], and V_indices=[3,4,5,6,9,10,13,14,17,19]. The scalar sector is (dlogalpha,dlogchi,dg_nn,P/Omega,Theta/Omega,A_nn,Lambda_n,dbeta_n/alpha); each vector sector is (dg_nA,A_nA,Lambda_A,dbeta_A/alpha), followed by two tensor derivative/A pairs. Frame conventions are precisely those of the pinned extractor; these abbreviations are not a replacement coordinate implementation.

At W_gauge=1 the constrained normalized matrix A has only eigenvalues +1 and −1 and obeys exactly

    A²=I, H=I+A^T A>0, HA=A+A^T,
    P±=(I±A)/2, rank(P±)=10,
    P±^T H=H P±, P+^T H P−=0.

Positivity follows directly from y^T H y=||y||²+||Ay||². The exact rational block matches all 288 retained harmonic extraction rows from the frozen local gate within8.8818e−16; this is a check of existing records, not a new kernel run. Physical-P, Q, preferred/null-feedback and spatial-norm lower-order sources do not change this principal block. No extension of A²=I to W<1 is claimed.

This supplies a symmetrizer for the normal principal block in the normalized variables. It is not automatically a local three-dimensional symmetrizer valid for every direction or a uniform total-J energy. For a fixed-J radial reduction angular derivatives become coupled lower-order terms; their coefficients, the varying frame and all reference/normalization derivatives must remain. In raw variables the P/Omega and Theta/Omega normalization is not uniformly equivalent to an unweighted norm as r_b approaches S. A finite-radius certificate therefore cannot be promoted to a uniform scri estimate by continuity.

## Speeds and incoming data at a finite CMC boundary

Use the RHS convention y_t=K y_r+lower_order, K=beta_r I+alpha A in the flat Penrose collar. Then

    k_in=beta_r+alpha=(S−r_b)²/(2aS)>0,
    k_out=beta_r−alpha=−(S+r_b)²/(2aS)<0.

The corresponding outward transport speeds are −k_in<0 and −k_out>0. Thus the +1 eigenspace is incoming at the outer boundary in this convention; projector signs are not outward-speed signs. There are ten incoming principal combinations in the full twenty-field block, and a justified fixed-J reduction must preserve the corresponding restricted count. At S1,a.5 the examples are:

| r_b | Omega | Incoming outward speed | Outgoing outward speed | kappa10/Omega |
|---|---:|---:|---:|---:|
| .98 | .0396 | −.0004 | 3.9204 | 252.5253 |
| .995 | .009975 | −.000025 | 3.980025 | 1002.5063 |

Small inward speeds do not remove incoming data, the source stiffness, or a need to test the r_b→S limit. Incoming data act on derivative/curvature/connection combinations; applying these projectors directly to raw lapse/metric/shift values would be a different and unjustified boundary prescription. Homogeneous incoming characteristic data relative to the exact reference could be explicitly named as a finite-boundary principal control. It is not automatically a constraint-preserving or physical no-incoming-radiation condition. The incoming constraint map and its differential coupling to main/gauge fields still require derivation.

## Proposed momentum-only penalty: frozen algebra passes

Let C± be H-orthonormal characteristic rows, so E=||y+||²/2+||y−||²/2, y±=C±y. Let E_V inject only the ten normalized V RHS entries, and G±=C±E_V. The actual harmonic block has det(A_qV)=128/3, hence a nonzero pure-V vector cannot lie in a characteristic eigenspace with zero q component. In particular G+ is invertible. Equivalently its positive Gram matrix is

    G+^T G+=E_V^T H P+ E_V,
    det(G+^T G+)=13573175/41472.

All ten leading Gram minors are positive in the exact rational calculation. This result depends on the actual derivative/momentum split; it would not follow merely from A²=I for an arbitrary choice of stored variables.

For zero incoming data the proposed normalized boundary source is

    tau=−G+^{-1} k_in y+,
    delta(y+)_t=−k_in y+,
    delta(y−)_t=−k_in R y+, R=G−G+^{-1}.

When its energy contribution really is y_b^T H E_V tau, adding it to the principal boundary flux gives

    −k_in ||y+||²/2−|k_out| ||y−||²/2−k_in y−^T R y+.

The sign and Schur-complement condition in the proposal are correct: this is nonpositive iff k_in ||R||²<=|k_out|. The characteristic-row orthogonal choices do not change the singular values of R. Their squared values are the roots of

    (z−1)^4 (11z²−23z+11)^2 (4487z²−13821z+4487)=0,
    ||R||²=(13821+sqrt(110487365))/8974=2.711421553150291.

Consequently k_in||R||²/|k_out| is2.7664744e−4 at r_b.98 and1.7031435e−5 at r_b.995, for the exact reference. The frozen boundary quadratic has a strong margin. These statements do not assume or prove that the same signs/margin persist for arbitrary live lapse, shift and metric states.

Momentum-only injection is attractive because q=D U can remain a derived quantity rather than an independently evolved auxiliary field: at the continuum principal level q_t=D(U_t) gives the same reduced block. The stored V sources must be converted back from the normalization, and the independent A5 injection must respect the actual metric-dependent trace constraint. The ten configuration values U still evolve by the original first-order-in-time equations and have not disappeared; differentiating them also produces coefficient, frame, angular and reference terms. An independent first-order reduction would additionally carry q−D U and tangential integrability constraints, require its own auxiliary data analysis, and cannot be identified with a direct penalty on twenty stored values.

If a first-order reduction replaces every U_r in U_t by an independent q, the ten U equations have zero principal speed; a formulation retaining an explicit beta U_r assigns an advective configuration block instead. These are reduction choices to match, not ten extra incoming wave data. The q/V energy alone also misses constant-in-space configuration modes. A full state estimate must add an appropriate U norm and retain its lower-order evolution. The regular solid-harmonic basis preserves the constant scalar/vector/STF configurations where permitted by J; it must not remove them to make a derivative-only norm positive.

## What remains before an actual Gauss/SBP-SAT claim

The previous quadratic assumes that the nodal boundary lift produces exactly y_b^T H E_V tau in the chosen discrete energy. That assumption requires a proof. H has nonzero q/V cross blocks in the present split. With variable normalization, different solid-L envelopes and exact/overintegrated coupled mass matrices, a scalar inverse quadrature weight times endpoint evaluation does not automatically realize the desired momentum-only lift. The lift and all mixed energy terms must be checked together. An exact common-node polynomial mass product may need overintegration as detailed in ASSESSMENT.md; nonpolynomial coefficient terms need independent quadrature convergence.

The Gauss endpoint evaluation is a polynomial trace functional, not a point node. A correct SAT uses the adjoint trace lift for an actually verified discrete integration-by-parts identity, with the r²/rho angular mass factors and variable H/coefficient production terms. The base Jacobi weight alone supplies no such coupled identity. Using D_rho² as a composed derivative does not prove a discrete Leibniz or Bianchi identity. The full-ball origin is not a second artificial radial boundary: regular solid-polynomial identities must cancel apparent 1/r terms and make its flux limit vanish. A normal frame/projector is undefined at r0, so finite-positive-radius characteristic formulas cannot replace that regularity gate.

The minimum later gates are therefore: actual fixed-J principal/reduction matching and origin limits; consistent q=D U with all angular/reference terms; verified mass/SBP/trace-adjoint/lift algebra; independent incoming constraint/gauge/physical boundary analysis; matched manufactured/reference and constraint-functional tests; and separate refinement in N and r_b with source stiffness/conditioning retained. This finite-radius option is a plausible explicit control to investigate after those gates. It changes the bulk radial representation and outer mathematical problem, so it cannot by itself isolate Cartesian embedded-boundary ghosts or establish exact-scri stability.
