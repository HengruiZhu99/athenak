# Related full-3D coupled scri controls, 2026-10-27

All three requested N6/N8/N10 evolutions completed at `t=.0002` with bitwise
midpoint restart agreement. **Every case fails the scientific gates.** The
original vector equation, actual scri clock/null/tangent conditions, and restart
controls pass. The tensor pole and Hamiltonian constraint targets fail. These are
short classical-RK4 research-engine evolutions, not native Driver task results or
late-time wormhole/lapse-front qualification.

## Fixed experiment and provenance

The driver was compiled privately from source at isolated-branch HEAD
`210b46036787d832ab2b8da3ec3bc2049a8306a0`. Compiler-reported project dependencies,
Kokkos headers, generated configuration, and static libraries were copied before
compilation. A second compiler dependency pass reports **zero live project paths**.
External SDK dependencies and the private capsule hashes remain unchanged after
all runs. Shared CMake, binaries, source headers, and original 1D experiments were
untouched.

The binary SHA256 is
`296298c50efffaa8488dcdda12feb8bcd09c9f48bdb55f9ada42c72cf34e1edf`;
the compiled-source archive SHA256 is
`b3583bc06031c92f7d4e7b14684d8c3c40fd16030ba204210f3359ca1c929895`.
The archive includes the actual compiler inputs/static libraries and exact
initializer sources. The executable is retained losslessly as `pilot.gz`.

Every run uses eight equal Cartesian blocks and seven shell layers with radii
`.45,.55,.60,.65,.70,.75,1`; each block has the same N cubed active logical shape
and two ghosts. All other controls are fixed:

- classical RK4, `dt=1e-4`, two steps;
- exact isotropic M=1/3 wormhole initial slice and Minkowski reference;
- positive storage2, selected telegraph taper `.24`, no lapse/chi floor;
- fourth-order operator, radial interface stencils, global radial-line smoothing
  and topology coarsening;
- supported positive-jet scheme1, Hermite boundary scheme1, seven surface nodes;
- vector tolerance `1e-10`, no Cartesian AMR, no log-halo cache.

The frozen pilot exported the actual cell coordinates before initialization.
Every x/y/z/r tuple, in its exact active-cell ordering, was bitwise equal to the
corresponding existing N6/N8/N10 native accuracy initializer coordinates. Reused
initial data were copied, never modified; their radius columns, positivity,
source paths, coordinate tuple hashes, and file hashes were checked and recorded.
Only the specified initial slice is radial. The evolved Cartesian metric,
curvature, independent Lambda, lapse/chi, telegraph flux, and globally solved
shift remain full 3D fields without spherical averaging/projection.

Each command ran sequentially on local CPU with its own owned process group and
180-second cap. No timeout, unsupported-constructor fallback, changed tolerance,
or repeated coupled case was needed. Exits1 below are explicit scientific-gate
failures after completed evolution/output/restart, not execution failures.

## Actual evolution results

| N | Cells | Process seconds | Restart | Vector iterations, primary path | Tensor pole max | H RMS | H max |
|---|---:|---:|---|---:|---:|---:|---:|
| 6 | 10,800 | 21.70 | bitwise | 110 | 1.07350145e-6 | 0.16354050 | 0.79429244 |
| 8 | 25,600 | 49.71 | bitwise | 125 | 2.47959812e-7 | 0.09257602 | 0.28090495 |
| 10 | 50,000 | 102.72 | bitwise | 148 | 1.35403992e-7 | 0.06142992 | 0.16473958 |

The pole is the original tensor/gauge numerator maximum tracked through RK
stages and accepted-state evaluation, using the requested Hermite source path.
The unchanged target is `1e-7`; all three fail. The separate legacy-sampler final
`scri_pole` diagnostic in stdout is retained but is not substituted for the
Hermite stage/source maximum. The original spherical H RMS target is `1e-5`;
all three fail by large factors. No near-pass description is justified.

| N | M RMS | Theta RMS | Z RMS | Original vector RMS max | Original vector point max |
|---|---:|---:|---:|---:|---:|
| 6 | 7.10982961e-2 | 5.82861723e-6 | 2.17977264e-4 | 7.04151392e-11 | 2.04177419e-9 |
| 8 | 2.83854624e-2 | 1.68935944e-6 | 1.18209143e-4 | 7.77034149e-11 | 2.54859106e-9 |
| 10 | 1.37609127e-2 | 8.74549881e-7 | 6.60346119e-5 | 5.13223290e-11 | 5.26150196e-9 |

All original-vector RMS values satisfy `1e-10`, and point values satisfy `1e-7`.
Actual nodal and reconstructed-ghost lapse clocks, future-null normal condition,
and fixed-angle tangential condition meet the original `1e-13/1e-12` limits;
N10 ghost values are respectively 0,4.44e-16,4.13e-16. No potential/preconditioner
residual is used to replace the original vector equation.

The stderr cost receipt includes 13 solves per case, including the extra restart
path: iteration sums 154/176/208 and solve-wall sums 16.11/38.77/82.74 seconds.
These differ from the primary-path iteration totals in the table because
restart reproduction repeats one step. No distributed or asymptotic scalability
claim follows from these three Serial pilots.

## Matched physical radial comparison, all cells retained

The comparison uses the exact selected `.24`, `t=.0002`, M=1/3 Minkowski-reference
N2049 radial profile and independent N1025 control under
`selected-taper-pilot-20261027`. These reference parameters, library revision,
and file hashes are retained. The matched radial runs also use classical RK4
with the same timestep; no native Ketcheson/classical RK equivalence is assumed.

Weights are exactly

```
w_p = mapped_cell_volume_p * sqrt(det(g_tilde_p)) / chi_p^(3/2)
RMS = sqrt(sum_p w_p*error_p^2 / sum_p w_p).
```

This is the proper-volume quadrature of the Penrose spatial metric
`gamma_bar=g_tilde/chi`. There is no additional Omega factor, floor, clipping,
inner exclusion, omitted interface, or scri masking. Unnormalized L2, bias,
point maxima, every raw field/direction, and overlapping regional diagnostics
are retained alongside RMS.

| Physical-field volume RMS error | N6 | N8 | N10 |
|---|---:|---:|---:|
| alpha | 9.2168033e-7 | 4.2894290e-7 | 1.9564881e-7 |
| chi | 1.8907318e-6 | 7.9747896e-7 | 3.6427644e-7 |
| g_rr | 3.5747836e-6 | 1.4695971e-6 | 6.8008984e-7 |
| A_rr | 1.9581293e-4 | 7.1688645e-5 | 3.4085308e-5 |
| Khat | 7.9297745e-5 | 2.8933706e-5 | 1.3849445e-5 |
| Theta | 5.8286172e-6 | 1.6893593e-6 | 8.7454972e-7 |
| Lambda_r | 2.8580220e-5 | 1.1006701e-5 | 5.4326861e-6 |
| beta_r | 1.0222524e-3 | 5.1687456e-4 | 2.3882199e-4 |

The N1025-minus-N2049 controls are much smaller here: alpha RMS about 3e-11,
beta about 1–2e-11, and A_rr about1.2–3.6e-8. These are resolution controls,
not rigorous bounds on all radial reference errors. They do not explain away
the observed 3D discrepancy.

The three grids are related, but the smallest sampled puncture radii shrink and
the total quadrature volumes change: 7.19289,7.50800,7.89138. Thus apparent
all-cell volume-RMS exponents are diagnostics rather than a proof of convergence
on a common regular domain. H RMS exponents are 1.98 and1.84; the pole reduction
exponents change from 5.09 to2.71. No stable scientific convergence order has
been established. The original bulk/constraint order target 3.5, endpoint field
order target 3, duration and independent control requirements remain unmet.

Both common-radius 64-direction samples and actual-node angular data are retained.
The former include interpolation error: their maximum alpha scatter is at
`r=.05`, with 0.00480074/0.00156184/0.000208666, where the coarsest core is poorly
resolved. Actual-node maximum alpha scatter is instead
1.64217e-6/3.84104e-7/1.59696e-7. Neither source is filtered or substituted for
the other. Raw direction-resolved data distinguish sampling/transfer error from
physical evolved angular variation; no averaging feeds back into evolution.

N6's global H maximum lies on the first transition-shell cell next to the cube,
`r=.272946`, not an AMR interface. N8/N10 maxima lie at the nearest Cartesian
puncture cells, `r=.0270633/.0216506`, with `rho/h=.8660254` inside the supported
positive-jet region. Regional H RMS at `r<.1` remains
0.406553/0.230119/0.136377. At `r>=.9` it is
4.05877e-7/1.69559e-7/7.57975e-8. These regional improvements do not override
full-domain failure. CSV lacks derivative-complete local Z and original shift
residuals; global values are retained honestly instead of reconstructing them
from independent Lambda or algebraic shift data.

## Deliverables and next diagnostic

Evidence is in `evidence/related-scri-family-20261027/` and
`evidence/related-scri-family-comparison-20261027/`: full source/binary capsules,
all commands/stdout/stderr, lossless coordinate/initializer/full-state/shell-surface
checkpoint CSV.gz/binary.gz, original residual cost receipts, all-cell arrays,
raw angular arrays, localization, comparison JSON, and plots. Fresh executions:

```
PYTHONDONTWRITEBYTECODE=1 python3 tst/elliptic_shift_3d/run_related_scri_family.py \
  --output docs/elliptic-shift-3d/evidence/NEW-related-family
PYTHONDONTWRITEBYTECODE=1 python3 tst/elliptic_shift_3d/analyze_related_scri_family.py \
  --family docs/elliptic-shift-3d/evidence/related-scri-family-20261027 \
  --output docs/elliptic-shift-3d/evidence/NEW-related-comparison
```

No further coupled case is justified by the current result alone. The next
bounded task is a same-data N10 midpoint comparison of Hermite 7/8 normal nodes,
with component/location and metric-normal/angular decomposition. A midpoint
local pole must not be mislabeled as the maximum over the completed evolution.
Any node-count change is a discretization control, not a fourth spatial grid or
permission to weaken the original boundary/constraint targets.
