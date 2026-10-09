# Cartesian total-J solid-harmonic prototype

This is a mathematical basis and analytic-jet prototype for a later boundary-fitted control. It calls no AthenaK evolution/constraint kernel and imposes no radial boundary condition. It does not admit any solver, scri closure, stable pulse or black-hole evolution.

For orbital L and spin s, define the Cartesian coupled polynomial

    B_(J,m;L,s)(x) = i^(L+s-J) sum_(p+q=m) CG(L,p;s,q|J,m) [r^L Y_(L,p)(n)] e_(s,q).

The constant Cartesian spin basis uses e_(1,+1)=-(ex+i ey)/sqrt(2), e_(1,0)=ez, e_(1,-1)=(ex-i ey)/sqrt(2); spin2 is their CG(1,1|2) symmetric trace-free tensor product. Scalars have spin0. Condon--Shortley spherical harmonics have unit sphere norm. For p>=0 the regular solid harmonic is

    r^L Y_(L,p) = N_(L,p) (-1)^p (x+i y)^p r^(L-p) [d^p P_L(t)/dt^p]_(t=z/r).

Each derivative-Legendre monomial has an even nonnegative residual power of r, so replacing that power by a power of rho=x²+y²+z² gives an exact Cartesian polynomial. Negative p uses `Y_(L,-p)=(-1)^p conjugate(Y_(L,p))`. The chosen overall phase makes m0 real and gives `B_(J,-m)=(-1)^m conjugate(B_(J,m))` for the full coupled fields. All-m coefficients are preserved in basis-data.json, even though the initial standalone C++ evaluator returns real m0 fields.

The total rotation generator is `J_k=-i epsilon_(kij) x_i partial_j+S_k`. For a vector, `(S_k v)_i=-i epsilon_(kij)v_j`; for a tensor it acts on both Cartesian indices. The checker acts with this independent differential generator, not just the CG labels: every generated field obeys J²B=J(J+1)B and J_z B=mB exactly. It also checks unit sphere/Frobenius norm, STF symmetry/trace, m conjugacy, homogeneous degree L, and orbital parity(-1)^L. Under spatial inversion of a polar rank-s field the total parity is(-1)^(L+s).

Allowed L satisfy `|J-s|<=L<=J+s`. For J0,1,2 the full field channels are:

| Field family | Multiplicity | J0 L | J1 L | J2 L |
| --- | ---: | --- | --- | --- |
| alpha, metric trace, P, physicalTheta |4|0|1|2|
| beta, Lambda |2|1|0,1,2|1,2,3|
| metric STF, independent A |2|2|1,2,3|0,1,2,3,4|

This gives8/16/20 radial amplitudes. The metric trace basis tensor is I/sqrt(3); the STF tensors have unit Frobenius normalization. Lambda is not a general-coordinate vector, but under the constant orthogonal Cartesian rotations used here its inhomogeneous connection transformation vanishes, so its spin1 representation is appropriate. The chi/gtilde density weights also add no factor under these determinant-one rotations.

Each channel multiplies its solid polynomial by an independent smooth W_L(rho). Thus its angular radial amplitude is u_L(r)=r^L W_L(r²), and every Cartesian component is regular at the origin. This is a smooth Cartesian condition, not a parity-only extension of arbitrary u(r). The prototype evaluates W, W_rho and W_rhorho directly and never divides by r. Tests include the exact origin and W=1,rho,rho²,1+rho+rho². A later finite-r extraction using arbitrary u,u',u'' must convert to W jets only at r>0 and separately preserve these origin conditions.

The stacked-angle rank test removes each column's harmless nonzero normalization and computes exact rational ranks. It uses the x and z axes plus two independent rational oblique angles. For each J/spin the rank equals the number of allowed orbital L. This is a basis rank gate; a later actual operator extraction must use a well-conditioned angular fit, independent off-fit angles and m1/m2 checks. Finitely many value samples alone do not replace the representation-theoretic closure statement or a tested actual kernel.

## Physical metric and A tangent conversion

The metric channels parameterize an arbitrary symmetric perturbation H of the Penrose spatial metric bar-gamma, using one trace plus the STF channels. They do not independently prescribe chi and five gtilde components. With gref=chi_ref bar-gamma_ref, the tangent is

    delta chi = -(chi_ref/3) bar-gamma_ref^(-1):H,
    delta gtilde = chi_ref H+bar-gamma_ref delta chi.

The standalone reference_conversion.hpp propagates every supplied coefficient value, gradient and Hessian through these products and matrix inverse. It neither substitutes a constant reference metric nor drops derivatives of a radial conversion coefficient.

Let T be the independent Euclidean STF A input. The production A trace constraint varies as

    gref^(-1):delta A = Aref^(ij) delta gtilde_(ij).

Here Aref^(ij) raises both indices with gref inverse. Therefore

    delta A = T-(gref/3)(gref^(-1):T)
                +(gref/3)(Aref^(ij) delta gtilde_(ij)).

Projecting T to zero reference trace alone would omit the final term. The standalone helper includes it, with all supplied coefficient derivatives. The compiled conversion test uses a spatially varying anisotropic SPD bar-gamma, nonconstant chi, nonzero reference A and a nonzero required delta-A trace. Values/first/second derivatives are compared to separately differentiated symbolic formulas. A later actual consumer needs metric derivatives through second order and A derivatives through first order; it must supply every reference derivative that it actually consumes, rather than interpreting unavailable higher A derivatives as physical zeros.

## Scope of a later spherical control

The analytic continuum kernel with radial reference/gauge coefficients should commute with SO(3), so a total-J decomposition is the appropriate control. Cartesian finite-difference, one-sided-advection and KO operators do not have that exact rotational symmetry. Replacing them by a radial discretization is an explicit control change, not a reproduction of the embedded Cartesian operator. No angular closure of the actual kernel, radial characteristic closure, exact-scri regularity, energy estimate or eigenvalue is established by this basis prototype. The regular ball avoids an artificial inner annular boundary; the outer finite-radius boundary and radial discretization remain separate future gates.
