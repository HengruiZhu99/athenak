# Exact metric-connection shift equation and implemented boundaries

The equations below are continuum identities. Their expanded mapped discretization
does not exactly enforce the connection of the actual discrete metric evolution;
[the discrete gauge audit](discrete-gauge-qualification.md) records a large mismatch.
This qualification applies to every earlier reported original-vector residual.

This is a **partial CPU research implementation**, based on
`8bef4966e253e92d85761f46d8885e7cc80dc2ed`. The production AthenaK hyperboloidal
and cubed-sphere restrictions remain in force. See `status.md` for the distinction
between this backend, the native tensor kernel, and production task integration.

## Metric equation and connection variation

Write `g_ij = gamma_tilde_ij`, with the Lagrangian determinant convention used by
`ConformalRHS`. Its exact metric equation is

```
g_ij,t = H_ij(beta) + C_ij
H_ij = beta^k g_ij,k + g_kj beta^k_,i + g_ik beta^k_,j
       - (2/3) g_ij beta^k_,k
C_ij = -2 alpha A_ij.
```

Here `A_ij` has precisely the normalization of `Z4cJet::a`; it is not a physical
curvature substituted without the conformal transformation. Neither chi, the
physical trace nor the evolved Z4 connection is identified with a metric
constraint. All components are in one global Cartesian reference chart, where
the fixed reference connection is zero. Thus

```
V^i = g^jk Gamma^i_jk
h^ij = g^ia g^jb h_ab
DV_g[h]^i = -h^jk Gamma^i_jk
 + (1/2) g^jk {g^ia (h_ak,j + h_aj,k - h_jk,a)
              -h^ia (g_ak,j + g_aj,k - g_jk,a)}.
```

This is direct differentiation of the inverse metric and Christoffel formula.
It is also valid for small determinant errors in sampled metric jets. The code
does not drop those errors using the continuum identity `V^i=-partial_j g^ij`.
The derivatives of H include all of
`beta_,d g_,k`, `beta g_,dk`, `g_,d beta_,i`, `g beta_,di`, and the derivative
of the determinant-convention term. No metric coefficient is lagged within an
individual linear solve.

The **continuum vector equation discretized by the expanded operator** is

```
L_g beta := DV_g[H(beta)] = -eta V - DV_g[C] =: S.
```

The geometry, lapse, curvature and their spatial jets are known at each RK
stage. This is a linear solve for beta at that stage; the overall evolution is
nonlinear. `Lambda` is absent from both the operator and its source. It evolves
through its original independent tensor equation. The tests change Lambda by
order one million while checking bitwise unchanged elliptic source values.

The principal part is
`g^jk partial_j partial_k beta^i + (1/3) g^ij partial_j partial_k beta^k`.
For a nonzero covector xi, its symbol has two eigenvalues `g^jk xi_j xi_k` and
one eigenvalue `(4/3) g^jk xi_j xi_k`, positive for positive definite g.
Uniform ellipticity additionally needs uniform metric bounds; puncture limits
are not covered by that assumption.

For flat g this gives `Delta beta + grad div beta/3`. With flat trace-free A,
the source is `2 partial_j(alpha A^ij)-eta V^i`. The Shibata decomposition in
the approved specification is therefore an exact flat-space identity. On a
general metric its independent scalar equations omit variable-metric terms.
This backend instead solves the original three-component equation using right
flexible GMRES. Three mapped flat scalar Poisson V-cycles precondition the
solve. They are approximate inverses only; their residual is never the stopping
criterion for the original vector equation. No potential boundary assumptions
enter this implementation.

For the spherical determinant-one metric, the independent radial formula is

```
V = g'/g^2 + 2(sqrt(g)-1/g)/r
Q = beta g' + (4g/3)(beta'-beta/r) - 2 alpha A_rr
V_t = Q'/g^2 + [-2g'/g^3 + 1/(r sqrt(g)) + 2/(r g^2)] Q.
```

The pointwise native test compares the full tensor residual with the expanded
1D second-order equation divided by g squared, including nonzero damping and
nonflat g. Separately, the coupled radial pilot is compared with the existing
independent `g1_rhs.py` evolution and quadrature shift. That reference never
uses 3D angular averages as its evolution variables.

## Maps, global solve and discretization

`CompositeMesh` contains an arbitrary vector of shell radii, subdivided faces,
and Cartesian leaf bounds. The first layer uses the existing cube-to-sphere
map. Subsequent layers use `r=R_inner+s(R_outer-R_inner)` and the same cubed-sphere
angular chart. Each leaf has the same logical `N x N x N` active shape and two
ghost zones. Physical shell cells need not be cubes. Cartesian refinement and
coarsening change a revision number; geometry, halos and solver hierarchies are
rebuilt. Shell refinement is explicitly unsupported, while the descriptors
separate layer, face and logical bounds from block identity.

Cartesian components are never transformed into patch components. Derivatives
use both inverse Jacobians and inverse Hessians:

```
partial_i f = J[A,i] D_A f
partial_i partial_j f = H[A,i,j] D_A f
                     + J[A,i] J[B,j] D_A D_B f.
```

The final operator uses fourth-order first, diagonal-second and mixed-second
differences. Ghost interpolation uses six active points per donor direction
when available. This supplies the additional order needed before taking two
derivatives. Ghost rows never depend on other ghost rows. The physical owner
selects donors at seams, corners, Cartesian/shell and Cartesian coarse/fine
interfaces. Independent patches with frozen interface values are never solved.

The lower-order preconditioner has its own global donor plan at **every** level,
including the pivoted dense global coarse solve. Its second-order mapped
Laplacian retains inverse-Hessian and mixed-derivative terms. Weighted Jacobi
avoids incorrectly applying the scalar Cartesian red-black independence rule
to mixed stencils. Restriction is the AthenaK eight-child average with physical
map-volume weights; trilinear correction uses its 3:1 interpolation weights.
Flat preconditioner coefficients need no evolved-metric coarsening: the exact
map coefficients are rebuilt on every coarse level. All evolved-metric terms
remain in the fine original operator. The bounded dense coarse solve rejects
more than 1024 scalar unknowns; unsupported odd/noncoarsenable large blocks are
not silently solved independently.

The production `Multigrid` class assumes a Cartesian octree and cubic physical
cell spacing. It cannot be assigned fictitious shell neighbors. The new host
backend adapts its transfer stencils, but **does not yet reuse its native task
graph, buffers or MPI/GPU halo engine**. The MPI correctness oracle distributes
owned Krylov vectors and exchanges them with allgather, then redundantly applies
the global mapped MG hierarchy on each rank. This verifies global coupling and
rank consistency; it is not a scalable distributed coarse/fine implementation.

Residual norms use mapped Penrose-spatial proper volume
`|det(dx/dq)| sqrt(det(g))/chi^(3/2) d^3q`. No bulk, interface, scri-collar or
puncture samples are removed from reported point maxima or finite-domain norms.
Physical proper volume contains an additional `Omega^-3` and diverges at scri;
a normalized finite RMS on the entire infinite physical slice would require a
separately stated weighted function space. It is not silently substituted here.

## Actual scri and the three shift conditions

The outer **physical face is r=1 and Omega=0**. All cell-centered active points
lie inside it. No Omega floor or finite-radius outer boundary is used. For the
stationary compactification `Omega=(1-r^2)/2`, nullness is

```
(beta^i Omega_,i)^2 = alpha^2 chi g^ij Omega_,i Omega_,j.
```

Let n_i=x_i/r denote the outward Euclidean coordinate covector, and
`a_n=sqrt(chi g^ij n_i n_j)`. The outward unit Penrose-spatial covector is
`s_i=n_i/a_n`. Future scri has `beta^i s_i=-alpha`, hence
`beta^i n_i=-alpha a_n`, and `beta^i Omega_,i` is positive. In the spherical
restriction this is exactly `beta^r=-alpha sqrt(chi/g_rr)`.

The two additional gauge conditions are fixed reference angular coordinates:
`beta^i partial_i theta=beta^i partial_i phi=0`. Equivalently, beta is parallel
to the coordinate radial direction. Together with nullness they give

```
beta^i_scri = -alpha sqrt(chi g^jk n_j n_k) x^i/r.
```

These three independent conditions prescribe one complete vector Dirichlet
datum. For a general anisotropic metric they differ from setting beta equal to
the metric-normal vector `-alpha s^i`; the latter need not fix the reference
angles. The boundary test uses a nondiagonal metric specifically to distinguish
these choices. The lapse clock alpha=1 is supplied by the evolution gauge and
never changed by the elliptic solver. No V or Lambda boundary value is added.

Vector Dirichlet data are inserted through a physical-face polynomial ghost
lift. The inhomogeneous lift is assembled once per stage; Krylov corrections
have homogeneous data. The original affine vector residual is evaluated again
after convergence. Independent extrapolation from active values measures
consistency with the prescribed face data, separately from the exact imposed
datum. Coarse failures against the original nodal 1D targets remain recorded.

## Evolution closure: implemented candidate and unresolved requirement

`ConformalRHS` returns `R+S/Omega`. For smooth compatible data with `S|scri=0`,
the actual limit is

```
U_t|scri = R|scri + (partial_r S)/(partial_r Omega)|scri
         = R|scri - partial_r S|scri.
```

All metric and lapse pole numerators must be combined before this operation.
Nullness and the lapse clock do not by themselves prove propagation of the full
tensor corner-compatibility hierarchy. The CPU coupled pilot uses active-only
six-point outflow extrapolation for geometric deviations and the prescribed
lapse/shift face values. It does not project pole numerators, H, M, Theta, Z,
determinant or trace-free constraints. Exact reference jets are added to spatial
derivatives of deviations; **no numerical equilibrium RHS is subtracted**.

This is a disclosed **experimental cell-centered boundary closure**, not a
derived complete tensor characteristic/regular-limit closure. Raw pole values
are independently reconstructed on the actual Omega=0 faces. Passing a short
Minkowski or tiny-pulse test cannot admit it for the selected nonlinear wormhole,
late fronts or generic nonspherical radiation. That closure remains a scientific
and implementation prerequisite for completion.

## Vector telegrapher and lapse scope

Let B have global Cartesian reference-vector components, and let D denote
`partial_t-beta^j partial_j`, acting componentwise. The implemented extension is

```
delta D alpha = chi partial_i(w B^i)
tau D B^i = kappa w delta^ij partial_j(alpha-alpha_ref) - B^i.
```

This uses reference Euclidean divergence and gradient, not physical covariant
divergence, and no Lie-transport stretching term is silently added. For radial
`B^i=B(r) x^i/r` and radial beta it reduces to
`chi[w B'+(w'+2w/r)B]` and `beta B'+[kappa w (alpha-A)'-B]/tau`.
The kernel test checks both equations independently. The pilot uses the selected
tau=.02, kappa=.005, and AF-regular outer taper from the spherical study:
one for r<=.72, zero for r>=.8, and
`w=y^8/[y^8+(1-y)^8]`, `y=(.8-r)/.08`, in between. Its first seven derivatives
vanish at the joins. The stage CFL includes the combined lapse/trace/flux speed,
not only the telegrapher speed, and the explicit relaxation bound.

The regular-center pilot uses the verified symmetric-restoring lapse control
with xi=1, nu=1 and the disclosed Appendix-C Theta source. The selected **wormhole**
configuration instead uses multiplicative restoring, the physical-log source
profile and collapsed-inner background source, logarithmic positive fields and
the AF coefficient chart. Those controls have **not** been replaced by the
regular-center lapse or claimed as implemented in 3D. The complete spherical
telegrapher spatial study passed; temporal order, quadrature, KO-amplitude and
adjacent-parameter controls remain incomplete in its recorded evidence.

## Puncture limitation

The required first black hole remains the verified isotropic Schwarzschild
wormhole with M=1/3 and Minkowski reference. Its other end has alpha~r squared,
chi~r fourth power, and odd radial powers in their coefficients. These are not
regular-center parity conditions. The spherical connection coefficient can have
a direction-dependent Cartesian limit, and a general 3D coefficient chart must
retain its angular/tensor content. A shrinking unsampled cell does not establish
regularity or convergence. This backend has no approved 3D singular-power and
changing wormhole/trumpet chart, so it does not run that benchmark with artificial
center parity, an Omega/chi floor, excision or a substituted initial slice.

The elliptic kernel itself imposes no puncture beta value. Its interior storage
and boundary interfaces can accept translating fields; beta=0 at every puncture
is not built into the solver. Translating and binary tests remain later work.

## Restart scope

The research checkpoint has a version/endian contract, full leaf geometry and
shell radii, all 22 evolved bulk/shell fields including the vector flux, RK time
lattice step/dt, operator order, gauge flag, tolerance and shift warm start. A
64-character source binding and payload checksum reject mismatched or corrupt
files. Halos and MG hierarchies are rebuilt after reading; continuation is
compared bitwise. This does **not** change the production AthenaK checkpoint
format: its existing refusal to omit shell state remains intact.
