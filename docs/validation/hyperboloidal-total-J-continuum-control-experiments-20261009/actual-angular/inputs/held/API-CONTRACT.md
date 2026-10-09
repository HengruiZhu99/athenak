# Planned local bridge contract (held)

This contract is descriptive. There is no compiled bridge or generated operator in this directory.

## External basis

The basis working API is expected to be:

    totalj::Field<T> EvaluateBasis(
        int J, int spin, int L,
        const std::array<T,3>& x,
        const totalj::WJet<T>& w);

`w.value`, `w.rho_d`, `w.rho_dd` refer to derivatives of W with respect to rho=x.x. `Field.components` is 1,3,9 for spin 0,1,2; tensor component 3*i+j is the Cartesian ij component, with both ij and ji present. Each component has value, d[i], dd[i][j]. Pin the final frozen API before implementing any adapter. The generated header supplies real m=0; an independent evaluator must use the frozen all-m exact data for m=1 and J=2,m=2 validation.

The allowed orbital channels are:

| J | spin 0 L | spin 1 L | spin 2 L | full independent amplitudes |
|---|---|---|---|---|
| 0 | 0 | 1 | 2 | 8 |
| 1 | 1 | 0,1,2 | 1,2,3 | 16 |
| 2 | 2 | 1,2,3 | 0,1,2,3,4 | 20 |

Each scalar channel is repeated for alpha, P, physical Theta and the Euclidean trace of h=delta bar-gamma. Each vector channel is repeated for beta and Lambda. Each tensor channel is repeated for the STF part of h and the Euclidean STF curvature seed S. The scalar h normalization is I/sqrt(3). STF tensor normalization, CG phases and all-m real/complex conventions must be inherited from the immutable basis, not reconstructed from component guesses.

## Actual field layouts

Native raw22 point values have order:

    chi, gxx, gxy, gxz, gyy, gyz, gzz,
    P, Axx, Axy, Axz, Ayy, Ayz, Azz,
    Lambdax, Lambday, Lambdaz, Theta, alpha, betax, betay, betaz.

The original native free20 chart selects raw indices:

    0,1,2,3,4,5,7,8,9,10,11,12,14,15,16,17,18,19,20,21.

The frozen generic-dual local helper's 20-row reporting order differs:

    alpha,chi,P,Theta,betax,betay,betaz,
    gxx,gxy,gxz,gyy,gyz,Axx,Axy,Axz,Ayy,Ayz,
    Lambdax,Lambday,Lambdaz.

Use explicit named layout adapters and cross-check them, not an unchecked memory reinterpretation. For the primary angular fit, assemble the raw22 tangent RHS including gzz and Azz before chart restriction. Inputs must satisfy det(g)=1 and g^{-1}:delta A-Aref^{ij}delta g_ij=0 to the declared numerical scale. Record algebraic-normal outputs before any projection.

The eight physical diagnostic components have order:

    Hphysical, Mcov_x, Mcov_y, Mcov_z,
    Zcov_x, Zcov_y, Zcov_z, Theta_physical.

Do not multiply M, Z or Theta by unrecorded powers of Omega. Reference-covector norm contractions use the actual Penrose inverse metric. This is a diagnostic convention, not an asserted positive physical energy.

## Planned phase separation

1. Admission/pin inspection (read-only and available now).
2. After explicit release: bridge binding, actual-double/dual directional checks, algebraic/spatial-jet and Cauchy-core independent oracles.
3. After those pass: local continuum angular coefficient action and held-out-angle/m closure only.
4. Later explicit decision: origin-inclusive radial representation and characteristic/scri closure, then its independent diagnostics.
5. Later explicit decision: operator/propagator and any native comparison. None is currently admitted.

For any eventual compilation, use a fresh source tree/prefix and capture exact source diffs, compiler/flags, transitive dependencies and executable hash. Do not alter the frozen basis, original tangent servers, archived mode histories or previous candidate directories.
