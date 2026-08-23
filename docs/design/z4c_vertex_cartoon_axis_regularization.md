# Vertex-centered SO(2) Cartoon axis regularization

Status: independent derivation and implementation contract. The evolved axis
is the active vertex `i == layout.is` on an axis-touching MeshBlock. Axis
identity comes from topology and index; production code never compares
floating-point `rho` with zero.

## Coordinates and parity

AthenaK component order is fixed:

```text
(rho, z, y_suppressed) = (x1, x2, x3).
```

Under signed-rho reflection through the axis, scalars are even. Vectors have
parity `(odd, even, odd)`. A symmetric tensor has parity equal to the product
of its component parities:

| component | parity |
|---|---|
| `rr`, `zz`, `yy`, `ry` | even |
| `rz`, `yz` | odd |

For a VC axis index `ia`, negative-rho ghost `ia-q` is filled from active
source `ia+q`, multiplied by component parity, for `q=1..ng`. This deliberately
differs from the existing CC face-axis reflection.

## Exact active-axis state identities

For every vector-like evolved state (`Gamma`, `beta`, `B`):

```text
V_rho = 0
V_y   = 0
```

For conformal metric, conformal `A`, native ADM metric/extrinsic curvature,
and applicable symmetric-tensor diagnostics:

```text
T_rr = T_yy = (old T_rr + old T_yy)/2
T_ry = 0
T_rz = 0
T_yz = 0
```

The corresponding linear identities apply to the RHS at the active axis.
Axis correction telemetry records maximum absolute and relative corrections,
component, `z`, cycle, and RK stage. Nonfinite or materially large correction
is a visible failure, never a silent repair.

## Suppressed derivatives away from the axis

The reduced plane is `y=0`. Killing symmetry supplies suppressed derivatives.
Representative identities in the local component order are:

```text
d_y f = 0
d_yy f = (d_rho f)/rho

d_y V_rho = -V_y/rho
d_y V_y   =  V_rho/rho
d_y V_z   = 0

div V = d_rho V_rho + d_z V_z + V_rho/rho
```

The complete vector/tensor first, second, and mixed table is the Cook et al.
SO(2) Appendix-C table mapped from its radial coordinate to `rho`, and is
cross-audited against `src/z4c/cartoon_derivatives.hpp`. Component order and
variance are explicit template inputs; mixed-index tensors do not reuse the
all-lower/all-upper table.

At the first four positive VC layers, direct quotient evaluation would lose
one formal order: a centered `O(h^p)` derivative error divided by
`rho=q h` is only `O(h^(p-1))` at fixed layer `q`. Production therefore fits
the regular coefficients in `s=rho^2` on local nodal samples. Even fields use
`F(s)`; odd fields use `V/rho`; and tensor isotropy differences and `T_rho y`
use their quadratic coefficients divided by `rho^2`. A degree `NGHOST-1`
functional gives O2/O4/O6-compatible coefficient derivatives for
`NGHOST=2/3/4`. This closure is selected only by the native-VC location tag;
the legacy CC path and VC points outside layers 1--4 retain the bulk identities.

## Analytic axis limits

No production axis branch evaluates `1/rho` or `1/rho^2`. Smooth parity and
l'Hopital limits give, among others,

```text
d_y f = 0
d_yy f = d_rhorho f
(d_rho f)/rho -> d_rhorho f

V_rho/rho -> d_rho V_rho
div V -> 2*d_rho V_rho + d_z V_z

d_y V_rho -> -d_rho V_y
d_y V_y   ->  d_rho V_rho
```

For tensor expressions, apparent `1/rho` and `1/rho^2` terms are replaced by
the regularized Cook et al. Appendix-C limits before evaluation. In
particular, differences such as `(T_rr-T_yy)/rho^2` use the appropriate radial
second derivative limit and odd radial tensor components use their centered
radial derivative limits. The implementation table must enumerate every
current `OnAxis` branch and map it to the cited analytic identity; unclassified
branches block qualification.

## Centered stencils through the axis

The active radial derivative at `rho=0` uses the ordinary centered O2/O4/O6
coefficients through parity-filled negative-rho ghosts. No one-sided radial
stencil is used when the parity stencil exists. Uniform VC and CC grids share
finite-difference coefficients; only coordinates, active bounds, and the
presence of the evolved axis differ.

## Stage contract

Before RHS, parity ghosts derive from the synchronized accepted/stage state.
After RK update, active-axis state regularity is imposed before the state is
used as communication input. Following shared-node synchronization and
hanging-node fill, physical/corner completion is performed and negative-rho
parity ghosts are rebuilt. At the accepted stage, algebraic projection is
followed by axis regularity and the complete exchange is repeated.

## Test oracle

Manufactured tests cover even scalar polynomials; radial and azimuthal vector
modes; every symmetric-tensor parity class; mixed rho-z derivatives; exact
axis limits; near-axis O2/O4/O6 convergence; divergence and contractions;
metric and `Atilde` regularity; and `beta/Gamma/B` state and RHS identities.
Tests execute the production kernels on host and available CUDA/SYCL backends.
Static tests are supplementary and cannot substitute for numerical kernels.

Primary references are Cook et al. [arXiv:1603.00362], especially Section 4
and Appendix C, and Pretorius [arXiv:gr-qc/0407110] for the modified-Cartoon
reduced-hyperplane method. Exact source provenance is recorded in
`z4c_vertex_centered_sources.md`.
