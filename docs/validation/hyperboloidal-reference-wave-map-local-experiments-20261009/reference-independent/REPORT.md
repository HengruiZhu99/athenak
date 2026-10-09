The independent fixed-reference radial oracle passes the declared local gates. All 2,052 exported field/ordinary-derivative comparisons have scaled error at most 4.210048918290138e-12 against the owner binary64 backend. The separately evaluated 100/130-digit functions agree to 4.876466421318419e-100 in the same scale. The endpoint-tail checks retain relative and nonzero requirements: 139 relative checks have maximum error 9.044529103015244e-14, and all 183 required nonzero values survive, including a native value 3.235e-320.

The current oracle source is `radial_oracle.py`, SHA b52d538658a73c373f8ac9a7757e5f9ab75a14e53218c723ca30f33190b22eaf. The fixed recipe is `recipe.json`, SHA c0212f3f6db2d2b09a1a7624e4efa5bc8d9258770241b4e608accde30120a3eb. `radial-attempt002/receipt.json` has SHA b435db2e1be942ed9eb75bd4340c41c728dc1b2728232efdf8510b4f29843ef9. Its run took 11.627837208 seconds and returned zero with empty stderr. No owner executable was invoked by this independent comparison.

The owner recipe is 3e7107cdb0b3904a8881a3ee8bb25b1e8b25e0dba2b266f50ac2238f664cad36, and its saved Release/ASAN-Debug reference output is byte-identical with SHA 887c6494c242925b171a1079d89aa20d6c34a7cc1b93cbc1637ac5f1e8865e00. The 38 prescribed binary64 radii and every source/executable/input pin are checked before the oracle runs. This is the fixed S=1, a=1/2, r0=.05, r1=.95 reference. The literals .05, .95 and .9, and each radius, are lifted to exact binary64 rationals; in particular .9 matches the explicit width literal in the captured backend. This does not establish a general-parameter backend.

The independent formulation uses o=1-r^2, b0=r/a, l0=1+r^2 and the original C-infinity logistic cutoff. Its exponent is g=-1/s+1/t, with s=(r-r0)/.9 and t=(r1-r)/.9. The small function exp(-|g|)/(1+exp(-|g|)) is evaluated directly, retaining weight and complement separately. The oracle imports no owner Taylor, Cartesian, or complete-reference implementation.

For the same Minkowski height and monotone compactification,

    Omega=(1-w)+w*o, b=r*w/a, L=Omega-r*Omega',
    alpha=sqrt(Omega^2+b^2), chi=(alpha/L)^(2/3),
    g_radial=(L/alpha)^(4/3), beta=-b*alpha/L.

Here beta is the radial contravariant shift amplitude, g_radial is the radial conformal metric eigenvalue, and the two tangential conformal metric eigenvalues equal chi. Their determinant is one. With the repository's negative extrinsic-curvature sign convention,

    k_radial=-b'/L, k_tangent=-b/(r*L), Kbar=k_radial+2*k_tangent,
    A_radial=-(2/(3*a))*g_radial*r*w'/L,
    A_tangent=(1/(3*a))*chi*r*w'/L,
    P=-(3*w+r*Omega*w'/L)/a,
    lambda=-(1/g_radial)'-2*(1/g_radial-1/chi)/r.

The A and P expressions follow exactly from the first line and retain their small transition values. Lambda is the radial contracted conformal connection amplitude for this determinant-one spherical tensor. It needs metric derivatives through three; L through three and A/P through three require cutoff/Omega through four. No higher-jet padding is introduced.

Each function is represented as a known exact core/outer background plus a separately differentiated small analytic defect. The alpha defect uses a rationalized square-root difference. The metric defects use log1p/expm1 of (alpha-L)/L. To differentiate lambda without subtracting rounded logarithmic derivatives, the oracle factors D=alpha^2-L^2 and uses

    lambda=(4/3)*(L'*D-L*D'/2)/(L*alpha^2*g_radial)
           +2*(delta_g_radial-delta_chi)/(r*chi*g_radial).

mpmath.diff differentiates the direct functions, including the defect functions, rather than the owner's Taylor coefficients. This is why 100/130 digits suffice even when a cutoff value is exponentially smaller than the working precision relative to its O(1) complement. It is not a claim that unseparated subtraction of 1-w at those precisions is accurate. All high-precision decimal results and per-comparison errors are retained.

The first attempt remains FAILED. Its source daa4574db985fb7cac6089f748a8ddb219d3c58c23a52bb220386aab4023e5cc and recipe c0d5d59849a41b33e60b5f744020de4bc51b7e3bd2ed732bd971b32711a96723 are copied verbatim under `failed-classifier-history001`; its complete original result directory and stdout/stderr are retained. That classifier mistook a small derivative for an endpoint tail and required relative accuracy through the midpoint's even-derivative cancellation zeros. Four r=.5 weight/complement rows failed, although every native scaled check already passed. The corrected, separately pinned recipe identifies the endpoint regime by min(weight,complement)<=1/4. Both derivative tails are checked there; at order zero only the small function is checked. Native scaled2e-10, tail-relative2e-8 for magnitude>=1e-300, and tail-nonzero for magnitude>=1e-320 are unchanged. This classification correction supplies no evidence of a C++ defect, and does not erase the original failed status.

The separate Cartesian consistency test also passes: 3,400 comparisons over both precisions, maximum algebraic-vs-direct error 2.0546082152506814e-98 and precision difference 2.0315434311307826e-98. It took 42.120712292 seconds, exit zero/empty stderr. At four oblique points of radii .1,.5,.84,.96 it compares explicit radial scalar/vector/tensor derivative identities against direct multivariate mpmath differentiation: Omega through four, alpha/chi/P through three, beta through three, Lambda through two, and all six metric/A components through three. The tensors use t(r)*delta_ij+(radial(r)-t(r))*x_i*x_j/r^2; the vector uses radial(r)*x_i/r. The check preserves every multiindex and both decimal evaluations. This is internal independent composition consistency, not a comparison with an exported C++ Cartesian backend or admission of the coordinate lift.

`branch-readback.json` additionally verifies 351 exact zero checks over eight exact core/outer rows, all 2,166 saved entries finite, 190 positive radial alpha/L/Omega/chi/g_radial values, and Release/Debug byte equality. At the origin the core profiles are constant and beta/A/Lambda vanish, so no radial division is consumed. These finite sampled checks do not prove arbitrary-lapse positivity, a uniform puncture or scri limit, a full coordinate-gauge lift, constraint propagation, PDE stability, or a black-hole evolution. The later black-hole requirement remains a wormhole-to-trumpet inner transition with the Minkowski hyperboloidal reference retained throughout.
