# Scri normal discretization: read-only audit

The 3D boundary prescription has the correct regular continuum limit, but its
normal discretization is **not the preserved 1D endpoint operator**. Copying the
coefficients `+4/h` and `-8/h` did not copy its radial layout, numerator
differentiation or coupling to the volume. These differences are a justified
control to investigate; this audit does not establish the cause of the observed
3D instability. No production source, evolved state, boundary condition or
numerical parameter was changed, and no RHS, eigensolve or evolution was run.

## Limit and off-regularity extension

Write the native interior equation as `u_t = R + N/Omega`. For the stationary
compactification, `Omega_n=-1` at the unit sphere. On a smooth regular solution
with `N=0` there, l'Hôpital gives `R - d_n N`. The implemented boundary equation
adds `4*N/h - 8*C/h`, where C denotes the chosen regularity traces:

```
boundary_rate = R - d_n N + 4*N/h - 8*C/h.
```

This follows [TensorScriLimit](source/three-d/tensor_scri.hpp#L211) and
[BoundaryRates](source/three-d/hyperbolic_composite.hpp#L145). When `N != 0`,
`N/Omega` has no finite boundary limit; the expression is an h-dependent
extension away from regularity, not an application of l'Hôpital to an existing
limit. Added terms vanish on an exactly regular solution. Discrete residuals
of order `h^p` are multiplied by `1/h`, however, and neither formal consistency
nor the explicit `dt*12/h <= .5` guard supplies a coupled boundary stability
proof. This is a limitation of the available argument, not a demonstrated sign
error.

Let `v=delta(chi*g^nn)`, `p=delta P`, `theta=delta Theta`, and let `n=x` at scri.
Linearize at Minkowski, with alpha=1, K=-3, g=I, chi=1, A=0 and beta=-n. Write
`C_Lambda=delta(Lambda-contracted_Gamma)` and let `C_A` be the linearized native
A regularity trace. The source's physical constraint definitions give

```
H   = -6*v - 4*p - 8*theta
M_i = -(2/3)*d_i(p+2*theta) + 2*A_in.

N_chi   = -H/6
N_P     = H/2 + 7.96*theta
N_Theta = H/2 - 4.04*theta
N_Lambda_i = 2*M_i + 2*d_i(theta) - 2*C_Lambda_i
N_A = -2*C_A - TF(C_Lambda tensor n + n tensor C_Lambda).
```

These identities use kappa1=4, kappa2=-.99 and the active
[HyperbolicDamping](source/three-d/hyperbolic_shift.hpp#L130) cancellation of
the `-3*w*Theta` pole term. They follow directly from
[EvolvedConstraints](source/three-d/conformal_rhs.hpp#L99),
[TensorPoleSources](source/three-d/tensor_scri.hpp#L123), and
[ScriRegularityTraces](source/three-d/tensor_scri.hpp#L16). TF is the Euclidean
tracefree part at this background; the symbols denote linear perturbations,
not nonlinear identities away from Minkowski.

Thus the Lambda penalty contains `8*(M+grad Theta)/h - 16*C_Lambda/h`; it is not
just a pointwise connection damping term. C_A also contains first metric/chi
derivatives, so its time derivative cannot be inferred from the A penalty in
isolation. Conversely, the isolated derivative-free scalar penalty block is
damping. Its H/Theta restriction is

```
h * d_t [H, theta]^T = [[-28, 1.92], [2, -24.16]] * [H, theta]^T.
```

The eigenvalues are approximately -23.33657 and -28.82343; the remaining scalar
penalty direction has eigenvalue -8. These facts rule out a simple local scalar
anti-damping sign. They do not settle the derivative-coupled constraint,
normal-gauge, geometry or interface subsystems. The held lapse's discarded
`R_alpha-d_n N_alpha` clock residual is measured at
[BoundaryRates](source/three-d/hyperbolic_composite.hpp#L177), not independently
enforced.

## Actual 1D and 3D outer operators

The preserved donor's [regular evolution](source/one-d/evolution.py#L24) uses
seven nodes and includes its outer endpoint in the radial grid. For the
regular-center uniform grid, the preceding nodes are one, two, ..., six full
spacings inside the endpoint. Its
[endpoint_first](source/one-d/numerics.py#L140) differentiates **sampled complete
pole numerators**. [PunctureEvolution](source/one-d/puncture.py#L98) does the same
and uses physical endpoint scale `hs = self.h*self.jac[-1]` in both penalties.
There are no outer ghost cells: the positive-domain derivative and advection
rows clamp their seven-node stencils to available radial nodes.

The actual N513 gamma4 donor has `grid_stretch=0`, but still selects the
[`inner-exp-scri` map](source/one-d/radial_grid.py#L8). Its outer modification
remains active at that setting, with `J(1)=1/2`. Consequently `hs=1/1024`, while
the nearest physical gap is `1.07550623428*hs`, and subsequent gaps are
nonuniform. The [saved grid](original-n513-r.npy) was extracted from the retained
original profile archive only after verifying that archive's SHA256 against its
[original manifest](original-1d-manifest.json). No field/state arrays were
extracted.

The 3D [surface sampler](source/three-d/scri_surface.hpp#L139) instead uses five
interior cell-center nodes at normal offsets `-4.5h,-3.5h,-2.5h,-1.5h,-.5h`,
plus a separate endpoint. Here `h=(1-.75)/n` for the present outer shell.
First/second jets are reconstructed directly, after which directional AD
differentiates the nonlinear pole algebra. A term in N containing `d_n p`
therefore leads to a direct D2 in this 3D closure, whereas the donor generally
forms `D_endpoint * D_node`. The latter spans ten outer nodes because the
interior derivative rows used in the sampled numerators have different
supports. The two operations are not identical even in a scalar linearization.

| Derivative diagonal, scaled by the stated h or hs | 1D uniform seven-node example | Actual mapped N513 donor | 3D half-cell six-node sampler |
|---|---:|---:|---:|
| First derivative | 49/20 = 2.45 | 2.10847087194 | 1126/315 = 3.57460317460 |
| Composed endpoint-first × nodal-first | 16709/3600 = 4.64138888889 | 3.31508786590 | Not the implemented pole differentiation |
| Direct second derivative | 203/45 = 4.51111111111 | 3.23938387760 | 10640/1323 = 8.04232804233 |

These entries compare coordinate derivative matrices, not full Jacobians;
physical reference-variable conversions and the nonlinear pole coefficients
add further terms. The h values also differ. For a boundary/interior mismatch
in an algebraic portion of N, the 3D combination `-d_n N+4*N/h` has boundary
diagonal `(4-1126/315)/h = (134/315)/h`. This is one reason an isolated penalty
coefficient is not a stability analysis. Cross-face angular jets and the V4
surface angular operator do not change this normal sampler.

The volume introduces another distinct coupling. Its
[outer ghost lift](source/three-d/composite_difference.hpp#L251) passes through
the boundary and six interior cell-center samples, forming a degree-six
polynomial. Holding the interior samples fixed, an endpoint mismatch enters the
first and second radial ghost values with factors `1024/231 = 4.43290043290`
and `1024/33 = 31.0303030303`. The surface jet uses five interior samples, while
this lift uses six. This is not the donor's ghost-free endpoint stencil. The
factors describe the radial lift before any angular donor interpolation; they
are not claims about an unstable eigenvalue or the full ghost operator norm.

## Smallest controlled comparison, not implemented here

First isolate numerator differentiation, retaining the **existing six 3D
radial nodes, all angular operators, penalty coefficients, equations, Q rates,
null closure and final algebraic projection**. Evaluate the complete native
geometric pole numerator at the five interior nodes and actual endpoint, then
replace only `d_n N` in the geometric limit by the derivative of that sampled
numerator. Use the native damping modification at every sample. This is a
donor-like factorization control, not an exact reproduction of the donor's
seven-node/mapped operator. Changing radial layout/width would be a separate
control. Compare actions on smooth low-angular-degree radial profiles and on
surface/interior mismatch profiles before any duration test.

Do not zero the interior numerator as a shortcut. Even exact Minkowski has
`N_chi=r^2-1=-2*Omega` and `N_P=N_Theta=1.5*(1-r^4)=6*alpha*Omega` in the collar.
At scri their normal derivatives are 2, -6 and -6, which cancel the corresponding
regular rates. Both radial polynomial stencils represent these background
polynomials exactly in exact arithmetic. A Minkowski identity check therefore
cannot distinguish their perturbative feedback.

The ordering must be explicit. Existing
[EvaluateRHS](source/three-d/hyperbolic_composite.hpp#L232) computes surface rates
and dependent D before filling volume ghosts. Sampled interior poles instead
need geometric first jets near the boundary, and those jets may need ghost
values. Static dataflow confirms geometric poles are independent of auxiliary
D: `Z4cJet` contains g/A/chi/alpha/P/Theta/beta/Lambda, `ConformalRHS` and
`HyperbolicDamping` use those fields, and `TensorPoleSources` has no D, B or J
argument. Geometric ghost filling depends on the current boundary geometry and
null alpha/beta values; its nonlinear null closure does not read D.

A prospective control can therefore:

1. Restore/materialize the current stage values and enforce the current null
   boundary values.
2. Fill geometric ghosts from those actual values; compute the sampled native
   geometric numerators and replace only their normal derivative contribution.
3. Add the unchanged weak regularity and optional angular rates, then calculate
   qdot, speed_t and dependent D from the **final** geometric rate.
4. Refresh D ghosts (or refill all ghosts) before evaluating any bulk shift or
   auxiliary rate. Do not retain pre-closure D in the bulk calculation.

Before interpretation, verify the ordering with a D-only perturbation at fixed
geometry: pole samples and the preliminary geometric ghosts must be unchanged,
while the finalized bulk gauge must receive the new D trace. Also require the
existing null/dependent-D compatibility and CPU/device parity checks. These are
proposed controls, not tests performed in this audit. They must preserve
independently evolved Lambda, Theta and all outgoing fields. No Lambda=Gamma,
Theta=0 or strong geometric-regularity reset is justified by this audit.

An h-scaled unstable normal closure is compatible with the earlier resolution
controls, but angular discretization scales with n too. The unconverged full
low-degree Ritz candidate and the failed trajectories do not uniquely establish
normal causality. A characteristic or discrete energy analysis is required
before calling any new boundary penalty incoming-only or constraint preserving.
Use separate original-J actions of the regular rate, pole derivative, numerator
penalty, trace penalty and Q to inspect H/M/Theta/C_Lambda/C_A and clock defects;
do not interpret a component action as an eigenvalue shift.

## Reproduction and evidence

From the repository root:

```sh
python3 docs/hyperbolic-shift-3d/reproduction/scri-normal-discretization-audit-v1/audit.py \
  --out /tmp/scri-normal-stencil-replay.json
```

The output must be new. The calculation uses small rational polynomials and
seven-by-seven derivative-weight systems only. [results.json](results.json)
records all coefficients and supports. Exact rational polynomial-moment checks
verify the uniform examples, composed support and ghost weights. The mapped
calculation uses the preserved donor `weights()` and checks the retained grid
and source hashes first. The five 1D source snapshots match the original
manifest; the six 3D snapshots match frozen V4 commit
`d98e2a99bbf1d5546956be3d724c82715ca704ec`. Their complete identities, original
archive/grid hashes and actual grid settings are in
[source-binding.json](source-binding.json). Full original state archives are
not duplicated in this package.

The actual run manifest is authoritative for the 1D source. Its preserved
`puncture.py` includes speed-gap diagnostics/capture beyond recorded scientific
commit `ef3e9ba0d345854861beaa9675e66de6b8b62db5`; it matches the actual manifest
and archival donor `c0ae7472a7aecc92cc72312c33d4cdc417a26291`. The other four
preserved 1D files also match the recorded scientific commit. This distinction
does not alter the endpoint formulas examined here.
