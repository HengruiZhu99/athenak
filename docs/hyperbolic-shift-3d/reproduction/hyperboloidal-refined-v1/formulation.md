# RF1: flat conformal frame and radial placement

RF1 is a separate research formulation. BN1 and AMR1 remain selectable controls.
The flat experiment is not promoted as a scientifically superior formulation.
Its default radius is the minimum distance from the origin to the six actual
Cartesian domain bounds, found from the Cartesian blocks. For the current box
this is 0.18; its corners have radius sqrt(3)*0.18 and are outside the plateau.

## Parameters and coordinate meanings

The strict `key=value` files are in `inputs/hyperbolic_shift_3d/refined`.
Unknown, duplicate, empty and nonfinite values are rejected.

| Parameter | Default / experiment | Meaning |
|---|---|---|
| `omega_profile` | `flat`; explicit `legacy` control | Conformal frame selector |
| `omega_flat_radius` | `auto` from domain | Compact Cartesian radius of exact plateau |
| `omega_outer_radius` | flat radius + 0.04 | Compact radius where old Omega resumes exactly |
| `omega_plateau` | 0.5 | Constant interior Omega |
| `gauge_begin` | outer radius + 0.06 | Compact radius where inner RHS starts blending |
| `gauge_end` | 0.56 | Compact radius where inner RHS has vanished |
| `buffer_cells` | 4 | Minimum separation using the largest core/radial spacing in the layer |
| `cfl_target` | 0.2 | Initial mapped characteristic CFL, capped by requested dt |
| `shell_radii` | .305,.45,.6,.75,.875,1 | Six outer shell interfaces in compact r |
| Boost localization | rest isotropic rho=rho0/4 to rho0, C6 | Native rest-coordinate radius; unchanged |
| Overlap quadrature | r=0.16 to 0.18 | Compact radial partition; Cartesian corners remain outside |
| Chi-floor support | r<=0.03 | Compact radius; unchanged |
| Tracker support | r<=0.16 | Compact Cartesian support; unchanged |
| Native slicing | 4M to 12M, C6 | **Areal R**, unchanged physical initial slice |
| Shift restoration | coefficient 4, r=0.6 to 0.9 | **Compact r**, unchanged |
| Donor lapse/gauge weight | r=0.56 to 0.8 | **Compact r**, unchanged outside inner blend |

`flat.cfg` changes the conformal frame and adapted inner equations/gauge together.
`redistributed.cfg` then changes only the shell radii to
.235,.30,.56,.75,.875,1. The final shell and scri support are unchanged.
The first compact shell starts at 0.16. Interpolation/KO/FD orders and evolved
overlap are retained. Actual halo-plan construction rejects unsupported donors.
The transition width must span at least three of the largest radial/core cells;
this is an admission floor, not evidence that a steep C6 profile is well resolved.
Unsmoothed profile CSVs expose the derivatives rather than hiding them by filtering.

## Conformal frame and spatial map

For z=(r-r_flat)/(r_outer-r_flat), let S be the normalized integral of
12012 z^6(1-z)^6. Inside the transition,
Omega=plateau+S(z)*((1-r^2)/2-plateau). Interior and exterior branches are exact.
Analytic gradients and Hessians use the radial chain rule. Admission requires
0<r_flat<r_outer<1 and plateau >= (1-r_flat^2)/2. This sufficient condition
makes Omega positive and nonincreasing, and Omega-r*Omega' positive, away from
scri. Gauge onset is outside the entire Omega transition with the resolved
buffer, lies in the native affine map region, and precedes gauge end <=0.56.

The native physical lab map is retained; physical radius is **not** redefined
as r/Omega. This is essential for retaining the exact boosted initial slice.
With f=Omega_new/Omega_old, the stored initial fields transform as
chi_new=chi_old/f^2, Atilde_new=Atilde_old/f, alpha_new=f*alpha_old.
The determinant-one metric, Lambda, physical Khat/Theta and physical spatial
geometry are unchanged. The freely chosen initial shift uses the adapted
compact gauge transition. Static storage offsets remain representational only.

Native inversion retains its positive-Jacobian and spacelike-slice checks.
The mesh-map oracle checks forward/inverse consistency and nonzero Jacobians
on every Cartesian/shell block. Native initialization samples both hemispheres
through actual production initialization. The initializer computes the lab
normalization k=0.30614228336307869 from the 4M--12M native map; RF1 uses
k^2=0.093723097662759575, not the BN1 literal. A conformal-frame change does not
change that physical coordinate normalization. Eta remains 6.

## Off-constraint inner equations

Constant Omega alone does not make the tensor conformal equations equal to
AthenaK Cartesian Z4c. The first nontrivial reference test failed, including
Atilde and Lambda components; its development log is retained in the evidence.
RF1 explicitly blends the Cartesian off-constraint prescription into the inner
region, with the same weight as the adapted inner gauge. BN1 is unchanged.

Write deltaGamma=Lambda-Gamma(metric), Z_i=gtilde_ij deltaGamma^j/2 and
E_ij=(gtilde_ki partial_j deltaGamma^k + gtilde_kj partial_i deltaGamma^k)/2.
The adjustment replaces the covariant Z derivatives in the Atilde equation
with E, removes its Z_i partial_j Omega pole terms, and replaces the Theta
curvature trace consistently. In Lambda it replaces Lambda*div(beta) with
Gamma(metric)*div(beta) and removes the K*deltaGamma term. The existing inner
Z4c damping is retained. Complete local RHS terms are blended; there are no
invented derivatives of the blending weight. Pole numerators remain explicit.

The oracle calls the actual `Z4c::CalcRHS<3>` on nontrivial polynomial fields:
nonzero Theta and connection constraint, transport, lapse/chi Hessians,
tracefree A and B gradients, and both constant and varying unit-determinant
metrics. A separate oracle checks conformal-frame invariance of physical
Hamiltonian/momentum constraints through the transition. These are equation
and implementation checks, not an evolution convergence result.

## Execution and identities

The same POD parameters enter CPU and GPU bulk, diagnostic and sampled-normal
kernels, halo consumers and moving-horizon geometry. The unchanged outer
Omega and final shell retain scri nullness and original scri operators.
Physical lapse flooring uses actual Omega. The conformal-chi floor remains
1e-8 inside compact r=0.03 with existing sign/event accounting; its physical
interpretation depends on the conformal frame. No new floor or clipping of
Omega/source singularities is introduced.

A fresh run chooses dt from the globally largest mapped characteristic rate,
including lapse, shift and light speeds, plus existing relaxation/angular
bounds, capped by the supplied baseline dt. Stage CFL <=0.4 is enforced.
A healthy restart retains its bound timestep; regrid admission checks the new
spacing and rejects an incompatible retained dt. This is a CFL guard, not a
proof of nonlinear stability. Fixed effective scri penalty remains
0.6510416666660156; original KO8 and scri operators are unchanged.

RF1 fixed and AMR checkpoints have separate magic and exact resolved parameter
identities. AMR includes its existing whole-payload checksum and topology/RK/
event ledger. Different parameters cannot resume each other. Numerical failures
are terminal. Campaign continuations may use only verified healthy walltime
checkpoints with the same source/binary/configuration identity.

Aurora storage for this task is capped at 50,000,000,000 bytes following the
user's updated allowance. Accounting includes the prior performance root and
the dedicated CompactBinaryMerger/hzhu scratch root, logical/allocated bytes,
planned outputs and an atomic-checkpoint reserve. It never deletes files.
BN1 and its monitor are independent and untouched.
