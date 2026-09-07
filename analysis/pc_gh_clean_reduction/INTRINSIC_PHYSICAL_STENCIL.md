# Primary physical constraint stencil

`src/pc_gh/intrinsic_physical_constraints.hpp` implements a diagnostic H/M
consumer for materialized primary fields `(w,rho,K,g[9],A[9])`. It reads no GH
fields or independent reductions. It is not yet called from the production
task graph. Its purpose is to supply the independent physical diagnostics
needed before serious intrinsic evolution.

The kernel uses the shared `Dx`, `Dxx` and `Dxy` operators to obtain metric
first/second derivatives, A first derivatives and w/K derivatives. It constructs
g inverse, its derivative `-g^-1 (Dg) g^-1`, the metric Christoffels and their
derivatives, and contracts the ordinary spatial Ricci tensor. It does not call
the evolution's Rstar. The constraints are

    H = 2 K^2/3 - tr((g^-1 A)^2) + w^2 R[g]
        + 4 w Laplacian_g(w) - 6 |dw|_g^2,
    M_i = nabla_j A^j_i - 2 partial_i K/3 - 3 A^j_i partial_j w/w.

The alpha-weighted momentum is computed separately as
`rho*w*m_i - 3*rho*A^j_i*partial_j w`. The diagnostic divisions by det(g) and w
require valid positive primary geometry; they do not enter the fundamental
evolution kernel, and no floor or clipping is introduced.

Direct Dxx has radius p/2 for order p. Dxy uses a tensor product reaching p/2
along each of its two distinct axes. Thus the FD6 diagnostic requires a valid
three-cell box stencil, including corner ghosts, and fits four available
ghosts. It does not differentiate an already differenced Christoffel field
over a six-cell reach. This is a different finite-difference realization from
the offline repeated-derivative diagnostic; equality at finite h is not claimed.
Their continuum physical target is the same.

## Analytic compiled checks

The standalone Kokkos harness evaluates two fixtures with oblique unit direction
n, orthogonal v, and t=n cross v. Coordinates are rotated only for the analytic
fixture; production derivative axes remain the mesh axes. Both fixtures use
theta=n dot x, w=1+0.1 cos(theta), rho=0.8+0.03 sin(theta), K=0.03 sin(theta).

1. Volume-preserving shear: b=0.15 cos(theta),
   `g=I+b(v n^T+n v^T)+b^2 n n^T` and
   `A=0.02[v v^T+b(v n^T+n v^T)+(b^2-1)n n^T]`.
   Here R[g]=0, tr((g^-1 A)^2)=2(0.02)^2, Laplacian(w)=w'',
   and `M=n[3(0.02)w'/w-2K'/3]`.
2. Curved unimodular metric: f=0.12 cos(theta),
   `g=exp(2f)v v^T+exp(-2f)t t^T+n n^T`, A=0.
   Here R[g]=-2(f')^2, Laplacian(w)=w'', and `M=-2nK'/3`.
   This prevents an identically zero conformal Ricci calculation from passing.

Each final fixture tests 306 points: 2D/3D, FD2/4/6, N=16/32/64, 17 phases,
with spacing ratios (1,1.3,0.7). Every H, M and alpha-M refinement ladder passes
the frozen minimum order p-0.6. Minimum observed orders across both fixtures
and dimensions are 1.953, 3.894 and 5.771 for FD2, FD4 and FD6 respectively.
Alpha-M agrees with alpha times M to the fixed 2e-12 tolerance. Each fixture
also restricts the FD6 accessor to radius two in a negative control and obtains
a nonfinite result, demonstrating that missing stencil support is detected.

Evidence is in `intrinsic-physical-stencil-001/` under the qualification root.
Final executable/source identity is in `source-manifest.json`; final tests are
`intrinsic-physical-stencil-002` (shear) and
`intrinsic-physical-stencil-curved-001`. The earlier five-column shear harness
result is retained as preliminary evidence; use the final six-column harness
and final executable for reproduction. The initial build attempt lacked a
generated target; reconfiguring the existing CMake build resolved that setup
error without source changes.

```sh
cmake -S analysis/pc_gh_clean_reduction/compiled -B ORACLE_BUILD
cmake --build ORACLE_BUILD --target intrinsic_constraints -j4
python3 -W error analysis/pc_gh_clean_reduction/check_intrinsic_physical_stencil.py \
  --binary ORACLE_BUILD/intrinsic_constraints --fixture shear --output NEW_SHEAR_RUN
python3 -W error analysis/pc_gh_clean_reduction/check_intrinsic_physical_stencil.py \
  --binary ORACLE_BUILD/intrinsic_constraints --fixture curved --output NEW_CURVED_RUN
```

The CMake build must retain its configured Kokkos and Athena config/source paths
as recorded in the existing oracle build configuration. CUDA verification of
this new physical stencil is not yet run. Next: materialize primary geometry
on valid mesh ghosts, invoke this diagnostic at synchronized states and reduce
component norms/locations across ranks. This checkpoint does not enable physical
history output or advance the puncture/binary gates.
