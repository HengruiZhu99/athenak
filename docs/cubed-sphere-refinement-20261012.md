# Independent refinement and native-failure controls, 2026-10-12

The old wave-map failure is reproducible and boundary sensitive. Independent
bulk refinement also exposes errors at the new cube/shell interface. These
results argue for resolving the coupled discretization before replacing the
conformal equations or investing further in exact arithmetic. They do not
establish that boundary treatment explains every previous failure.

This work continues the [assessment handoff](hyperboloidal-assessment-handoff-20261010.md)
at `24616eb8d52060414f234192548abbfc7758efb0` and the
[regular cubed-sphere implementation](cubed-sphere-extension.md). The evidence
is collected separately in [this capsule](validation/cubed-sphere-refinement-20261012/README.md);
the earlier frozen capsules are unchanged.

## Reproduce before changing the formulation

The executable used for the old N24 reference-wave-map run was copied and
rehashed against its original receipt. Its SHA256 is
`829d590ee02e7865744ee9ea8e2bb00e036ee0dd9b7c2609da60740c15138b8c`.
The original final saved checkpoint, at `t=0.775`, has SHA256
`36cab8b80bb86c81f1e17d2f04fff627831b0496b20eb43853e2a430ba4a2288`.
No frozen source, input or output was edited.

The replay aborted after 33.01 seconds with a terminal error byte-identical
to the original: `t=0.79277343749968521`, cycle 12177, at
`(0.1375,-0.9625,-0.22916666666666674)`, `Omega=0.0021701388888885099`,
`alpha=3984.8591153407006`, `chi=-0.0071212456996532277`.
This is a checkpoint replay, not a fresh evolution from zero. Whole-stderr
comparison initially failed because restart emits unused initial-data parameter
warnings; the separate terminal-line comparison passes. Both records survive.

Boundary-only counterfactuals use that same binary and checkpoint, with the
same equations, gauge, damping, dissipation and timestep policy. Only the
ghost degree changes; `tlim=0.82` bounds the work.

| Ghost degree | Result |
| --- | --- |
| 2, original | Same failure at `0.79277343749968521` |
| 3 | Same cell fails at `0.78964843749968805`; negative lapse and chi |
| 4 | Reaches `0.82`, exit zero, 692 additional cycles |

A linear trial was rejected at setup: the existing policy accepts degrees
2–5. It is not a scientific evolution result. Quartic completion shows a
boundary response, not long-time stability. All restarts inherit accumulated
errors from the quadratic run; a fresh-from-zero closure comparison remains
necessary. The result neither validates quartic ghosts nor convicts the
continuum formulation.

The separate N32 failure was also replayed from its last saved checkpoint,
`t=1.5750509765619771`. Its terminal error is byte-identical to the original:
`t=1.5837303635638276`, cycle 13621, at
`(0.92812500000000009,0.30937500000000007,-0.171875)`, with
`Omega=0.013330078124999695` and `alpha=-1823717415.8052011`.
Thus reproduction is not limited to the anomalously small-Omega N24 case.
Quartic ghosts on this N32 checkpoint still abort, after 45.17 seconds, with
`invalid metric in conformal projection`. The last logged time is `1.581518`;
the exception records neither terminal time nor cell, so those remain unknown.
The wrapper's original fatal-line extractor looked only for the ADM message
and returned an empty list; the preserved stderr and separate comparison
identify the projection exception. This is an unsuccessful boundary-only
counterfactual, and higher-degree extrapolation is not a general repair.

## Independent regular-Z4c refinement

Aurora job **8923901**, one node in `debug`, first reproduces the original
`B32/S16`, amplitude-0.2, `t=2` pulse norms with a relative tolerance of
`1e-10`. It then holds the shell fixed while changing bulk resolution and
holds the bulk fixed while changing shell resolution. Independent cases use
the two tiles of one physical GPU. The worker cap is nine minutes, inside a
ten-minute PBS limit. The source and executable are frozen in its receipt.

Regional diagnostics are read-only. They retain fixed radial bins, maxima
coordinates, volume, minima of lapse/chi, and grid-width interface/seam/outer
collars. Collar widths shrink with resolution; their norms are not a
comparison over an identical physical subset. Fixed radial bins provide that
comparison. Their squared integrals and cell counts are checked against the
global norms.

Job 8923901 completed with exit zero in **8 minutes 18 seconds**. The baseline
reproduced every checked metric to the printed precision, and diagnostic
partitions passed their independent accounting checks.

| Bulk/shell | Bulk H L2 | Bulk M L2 | Bulk Weyl L2 | Shell H L2 | Shell M L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32/16 | 0.483708 | 0.174428 | 0.111886 | 0.0804734 | 0.0477955 |
| 48/16 | 0.657369 | 0.354479 | 0.228484 | 0.160221 | 0.0911229 |
| 64/16 | 1.10582 | 0.527028 | 0.364750 | 0.245099 | 0.139183 |
| 32/24 | 0.404870 | 0.0203690 | 0.0283758 | 0.0121509 | 0.00883272 |
| 32/32 | 0.402447 | 0.0150600 | 0.0271618 | 0.00439623 | 0.00330535 |

![Independent regular-solver refinement responses](validation/cubed-sphere-refinement-20261012/independent-refinement.png)

With the shell fixed at 16, the largest bulk Hamiltonian peak moves from the
central region to `(-0.015625,0.984375,-0.984375)` at bulk 64, beside a cube
edge. With bulk fixed at 32 and the shell refined, the largest bulk peak stays
at `(-0.28125,0.03125,0.03125)`. This separates an interface contribution from
an underresolved central contribution. Shell 16 to 32 reduces shell H L2 by
94.5%, bulk M L2 by 91.4%, and bulk Weyl L2 by 75.7%.

The targeted central-refinement control uses native static octree refinement:
root bulk 32, effective 64 in `[-0.5,0.5]^3`, root spacing at the cube/shell
interface, and shell 24. It has 120 bulk blocks and six shell blocks. Aurora
job **8923936** completed at `t=2`, exit zero, in **4 minutes 37 seconds** on
one debug node under `CompactBinaryMerger`. Its numerical executable matches
the independent matrix byte for byte. The first attempt, 8923918, hit a
270-second worker cap with last logged time `1.972575`; its failed attempt
is retained. The repeat changes only wrapper time budgets and uses the same
numerical input, with an eight-minute PBS allocation and seven-minute worker
cap. These are Release SYCL binaries in the debug queue, not bounds-checked
Debug binaries.

| Same-shell comparison at t=2 | Bulk H L2 | Bulk H max | Bulk M L2 | Bulk Weyl L2 | Shell H L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uniform bulk 32 / shell 24 | 0.404870 | 16.5801 | 0.0203690 | 0.0283758 | 0.0121509 |
| Central effective 64 / shell 24 | 0.0703503 | 3.12477 | 0.0140520 | 0.00947795 | 0.0119141 |

This reduces bulk H L2 by **82.6%**, its maximum by **81.2%**, bulk M by
**31.0%** and bulk Weyl by **66.6%**. The largest H peak remains central at
`(-0.328125,0.015625,0.015625)`; the bulk outer-interface collar's H maximum
is `0.0515031`. Shell H changes only 1.95%. Regional accounting and host/device
norm agreement pass. This is a static refinement control, not new nonzero-pulse
dynamic-AMR acceptance or a measured spatial convergence order: native
timesteps also change. The earlier true-half-step baseline argues against a
large time error there, but does not replace a matched-step control here.

Shell refinement removes a large interface contribution, while central bulk
refinement reduces the narrow central Hamiltonian peak. Refining the entire
bulk against a fixed coarse shell can worsen constraint norms. Coupled
refinement alone had concealed this response. The regular pulse uses a
Gaussian without the old hyperboloidal pulse's taper; its shift still advects
(`shift_advect=1`, despite `shift_Gamma=0`). Its central peak is not evidence
of interpolation error alone, and possible gauge steepening needs a separate
characteristic/admissibility check. The remaining constraint maxima are too
large to accept accuracy from completion alone.

## Mapped immutable-reference probe

`Athena_ENABLE_CUBED_SPHERE_CONFORMAL_PROBE=ON` enables an explicitly separate
CPU Serial research probe. Production hyperboloidal guards are unchanged.
It requires uniform bulk storage, no AMR, `R=1`, and `time/nlim=0`; it restores
the regular state before the ordinary driver runs. Its own conformal CSVs
contain the research evolution. The ordinary final regular CSV measures the
restored Minkowski storage and must not be mistaken for its constraints.

The probe uses the original `ConformalRHS`, physical `P=K-2Theta` and Theta
conventions, wave-map gauge, live stationary-Omega normal, native upwind
correction, damping normalization and final-stage algebraic projection.
The wave-map helper is copied byte-for-byte from the archived native recipe.
Only the analytic Minkowski geometric RHS's roundoff residual is subtracted,
as in the old code. No exact-arithmetic backend, manufactured forcing,
lapse/chi floor or differential-constraint projection is introduced.

One `const LayerReference`, with `a=0.5`, `r0=0.05`, `r1=0.95`, supplies
Minkowski values and analytic jets at actual mapped coordinates. Cached jets
have an immutable owner, and cached value views expose const elements.
Ghost interpolation acts on deviations and adds the exact target reference.
Complete logical jets transform with both the inverse Jacobian and Hessian;
an independent forward-map Cartesian-quadratic check exercises every scalar,
vector, metric and curvature derivative slot on six faces at three radii.

The interface/seam interpolation remains six-point Lagrange. Outer ghosts
use configurable degree-2–5 extrapolation, initially quadratic. The ordinary
Sommerfeld RHS is **not** applied in this probe; its input entry is a storage
placeholder. KO retains the old four-node active-line closure style in logical
directions, with mapped scale factors. The old uniform-grid dissipation energy
identity does not transfer to this variable-coefficient interpolating atlas.
This is a mapped discretization of the old formalism, not the identical old
boundary operator or a proven stable new operator.

The stationary `B16/S12` control reaches `t=0.01`, 54 cycles. Maximum field
deviation remains below `1.6e-14`; raw Hamiltonian maxima remain below `4e-12`.
Before/after reference payloads are byte-identical. Stationarity is a plumbing
and reference-preservation test: analytic reconstruction and reference residual
subtraction help ensure it by construction. It is not an independent proof of
the equations or a native nonzero-disturbance convergence test.

The original tapered pulse is preserved exactly:

```
shape = (1-r*r)^4 exp(-r*r / 0.35^2)
delta_alpha = 0.2 shape (1 + 0.2 x + 0.3 y z)
delta_beta = 0.1 shape (1 + 0.3 y z, 0.2 x, 0.1 x y)
```

It reaches `t=0.02` on `B16/S12` with positive lapse/chi and SPD metric, but
nonzero constraints. At `t=0.005`, independent refinement gives:

| Bulk/shell | Bulk H L2 | Shell H L2 | Bulk fixed-interior H L2 | Bulk fixed-interface H L2 |
| --- | ---: | ---: | ---: | ---: |
| 16/12 | 0.00240321 | 0.000268888 | 0.000627597 | 0.00311488 |
| 32/12 | 0.0137031 | 0.000270966 | 0.0000532264 | 0.0180221 |
| 16/16 | 0.000505468 | 0.0000931169 | 0.000624667 | 0.000396479 |

The fixed bulk interior is `max(abs(x),abs(y),abs(z))<0.18`; its complement
within the cube is the fixed interface region. Bulk refinement improves the
interior by a factor 11.8 while making the interface norm 5.8 times larger.
Its maximum moves beside the cube edge, to `(0.2421875,0.2421875,-0.0078125)`.
Shell refinement reduces that interface norm by a factor 7.9 while leaving
the fixed interior essentially unchanged. This implicates the coupled
discretization as a dominant early error source, alongside interior truncation error.
It does not prove that interpolation alone is responsible, nor distinguish
every interpolation, mixed-derivative and logical-dissipation contribution.

These are short controls, not accepted stability. Their smallest active
Omega is `0.0486184` at S12 and `0.0362181` at S16, respectively 22.4 and
16.7 times the old failed cell's Omega. PDEs are evaluated only at positive
Omega cell centers; the boundary face lies at scri but has no evolved
Omega-zero degree of freedom. Reference continuation outside scri supplies
ghost values only. No exact-scri closure is validated. Norms use compactified
Cartesian coordinate volume, not divergent physical ADM volume.

A genuine half-step B16/S12 control uses an initial timestep
`9.4265187539803992e-05` versus `0.00018853037507960798`, and takes 54 versus
27 cycles to `t=0.005`. The largest relative change among H L2/max and M/Z L2
is `6.23e-6`. This supports spatial error dominance for this early comparison;
it does not certify longer-time integration.

To build and repeat a bounded control:

```
cmake -S . -B build-cubed-sphere -DAthena_ENABLE_CUBED_SPHERE_TESTS=ON \
  -DAthena_ENABLE_CUBED_SPHERE_CONFORMAL_PROBE=ON
cmake --build build-cubed-sphere -j6
python3 -B tst/cubed_sphere/check_conformal_probe.py \
  build-cubed-sphere/src/athena /path/to/a/new/attempt \
  --amplitude 0.2 --bulk 16 --shell 12 --tlim 0.005
```

The runner freezes the executable, records source hashes, retains the resolved
input and checks reference payload hashes. It refuses to overwrite an attempt.

For a matched-collar control, the existing algebraic map with `L=0.25`,
`nr=272`, `nangular=12` gives minimum active `Omega=0.0021704751841309417`,
within 0.016% of the old N24 failed cell. This is only a geometry calculation,
retained in the capsule. No evolution was run at that resolution. Radial
refinement must also preserve adequate angular/interface resolution and admit
the resulting timestep cost; matching a single minimum does not match an
entire discrete boundary operator.

## What the original evidence does and does not establish

The handoff's exact-product units and v10 finite readbacks establish their
specific arithmetic comparisons. They do not bind a consequential arithmetic
error to the original failed native stage. The larger exact-rational gauge
backend remains a source prototype; passing a primitive cannot recover atoms
rounded upstream or establish stability of the coupled PDE discretization.

The Gaussian third-jet units and timing establish bounded oracle checks. The
full registry and native raw `RHS-u_t` binding were unexecuted at handoff;
positive inverse Jacobian is insufficient for spacelike slices, as the saved
future-time counterexample already demonstrates. A manufactured solution
would help measure consistency once it has an admissible slicing and the
native boundary operator, but would not by itself certify finite gauge-pulse
stability. Paired subtraction cannot replace a raw residual test.

Finite matrix residuals and positive sampled energies do not establish a
forward propagation bound or exclude nonnormal transient growth. A stable
interior principal symbol also would not establish a stable discrete interface
and scri closure. The new tests give direct reasons to prioritize those
coupled operators. No native evidence presently rules out gauge/constraint
growth in the continuum formulation either.

The pulse's 0.2 lapse coefficient is largely a monopole. It must not be quoted
as a 20% angular harmonic amplitude. Future acceptance should report the
actual nonspherical component, not merely this input coefficient.

## Ranked next gates

1. **Converge the balanced central-refinement grid.** The first static control
   reduces the central peak without moving it onto the outer interface. Now
   hold a sufficiently resolved shell fixed while refining the central bulk
   with native mesh refinement. Require lower Hamiltonian and momentum errors in
   fixed physical regions without moving a larger peak onto a coarse/fine or
   patch interface. Compare against uniform refinement at the same finest
   spacing. Repeat dynamic AMR with nontrivial fields and GPU bounds checking;
   constant-state refine/coarsen smoke tests are necessary but insufficient.
2. **Test the old formalism against boundary and interface controls.** Run fresh
   from zero, with matched pulse, layer and actual minimum-Omega collar; compare
   quadratic/quartic closures, true half timesteps and joint bulk/shell
   refinement. Do not credit a wider untested scri collar with curing the old
   failure. The current mapped probe remains CPU/uniform until these controls
   justify an AMR/GPU port. Compare initial analytic-deviation derivatives with
   native loaded jets at the measured interface peaks.
3. **Replace an unstable coupled closure if required.** Analyze the complete
   linearized gauge, gravitational and constraint system with patch and outer
   operators included. Require a resolution-independent finite-time growth
   bound and convergent oblique wave packets through seams/corners. Use a
   compatible penalty/mortar or SBP treatment if interpolation cannot meet
   those criteria. Derive regular null/shear/Z4/Theta scri conditions; ordinary
   regular-Z4c Sommerfeld data do not supply them.
4. **Bind arithmetic and the manufactured oracle to actual native operands.**
   Save a first bad RK stencil and compare raw binary64 and multiprecision
   RHSs at those exact states, including reference, Omega and derivative atoms.
   Advance the exact backend only if its errors are consequential relative to
   measured discretization errors. Admit the manufactured slicing globally
   over the tested interval before using it as an evolution oracle.
5. **Apply scientific evolution acceptance.** Declare dimensional and
   pulse-normalized constraint/Weyl budgets and a measured angular amplitude.
   Require admissibility, decreasing errors at three jointly adequate
   resolutions, and true-half-step agreement through `t=2`, then `6`, then
   `12`, across several exterior crossings. Only then introduce a resolved
   wormhole interior and puncture gauge. Keep the same Minkowski descriptor
   throughout and require convergent horizon area/mass and exterior constraints
   during the wormhole-to-trumpet transition. Completion without those checks
   is not acceptance.

For the next flat-pulse benchmark, a concrete proposed accuracy gate is
`w^2 * L2(H,M,Weyl) / A <= 0.01` and corresponding maxima `<=0.1` in each
fixed physical region, where `w=0.35` and `A` is the measured initial total
gauge amplitude. Use `w * L2(Z,Theta) / A <=0.01` for quantities with one
inverse-length dimension, retaining unweighted conformal constraint norms
near scri as well. These are proposed engineering budgets to register before
new acceptance runs, not mathematical stability bounds or retroactive PASS
criteria. Require at least third-order observed convergence in smooth fixed
interiors, positive convergence at every interface/collar, and half-step
differences below 10% of the finest spatial-resolution difference. A plateau,
resolution-worsening peak, growing incoming constraint mode or failed budget
rejects the run even if it reaches its end time. Measure and report the
nonspherical lapse/shift amplitude separately; a later substantial-angular
target should explicitly reach, for example, 10% relative nonspherical lapse
amplitude while remaining geometrically admissible.

## Cartesian compactification and Pretorius

The recalled implementation is real: Pretorius uses componentwise
`x_physical=tan(pi*x_grid/2)` at unchanged time, regularizes the evolved
variables, and puts spatial-infinity data on six cube faces. He also discusses
singular coefficients and eventual outgoing-wave underresolution/reflection.
This is a different asymptotic problem from hyperboloidal evolution to future
null infinity. See [Pretorius, section IV B](https://arxiv.org/pdf/gr-qc/0407110)
and [Zenginoglu's hyperboloidal-layer construction](https://arxiv.org/abs/1008.3809).

The following is a geometric inference, not a claim made by Pretorius. A single
smooth defining function that vanishes on two intersecting planar cube faces
must have zero gradient at their edge: continuity requires its gradient there
to be normal to both distinct face planes. This conflicts with a regular
conformal boundary's `dOmega!=0`. A single Cartesian cube therefore cannot
represent a smooth scri cut through its sharp edges using one nonsingular
smooth chart and defining function. One can use singular coordinates, rounded
faces or multiple charts, but must validate their transformed equations and
edge regularity. Cubed-sphere patches provide six tensor-product logical outer
faces representing a smooth spherical physical boundary while retaining the
Cartesian AMR interior and global Cartesian field components. That is the
more direct path for this implementation.
