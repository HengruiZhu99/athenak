# Frozen-background massless test scalar

This is a standalone scalar project under `src/test_scalar`. Its CMake project
links Kokkos and reuses `src/mesh/cubed_sphere/map.hpp`; it does not build, link,
allocate, initialize, evolve, or restore a Z4c/Einstein state. The production
baseline, older experiments, and their scientific failures remain unchanged.
Scalar success does not complete the original metric-evolution objective.

The current corrected configuration uses `--normal-upwind 1 --outer-sbp 2
--ko 0 --reduction-damping 0 --smooth-layer 1 --radial-stretch -2`. It evolves
only `psi, Pi, q_i` on frozen coefficients. Its angular Minkowski CPU and
Aurora GPU validations pass. Schwarzschild validation remains in progress:
the fixed-cube `N24/36/48` matrix fails its coarse-pair late convergence checks,
and the completed `N36/48/60` matrix still fails late field L2 convergence
after the independent profile sampling and resolution audits.
On 2026-10-17 the user explicitly authorized proceeding to an exploratory
`a/M=0.5` Kerr case before full Schwarzschild admission, then restricted new
work to low-resolution local runs. The local `N12/16/20` exploration completed
`100M` at all three resolutions. Its evidence is in
`docs/validation/test-scalar-20261016/local-kerr-spin05-20261017-v1`.
This does not mark the earlier failed gates as passed or constitute Kerr
validation. Historical configurations and failed gates below remain evidence.

The complex finite-radius waveform differences from the exploratory `600x8`
independent solution decrease from `13.19%` to `7.74%` to `4.22%`. Dense exact-scri
traces, raw checkpoints, radial profiles and all free-frequency fit windows
are saved under its `analysis` directory. The coarse reduction residuals and
window-sensitive fits do not satisfy the original validation gates; the small
independent reference has not received a resolution certificate. Native
checkpoint fragments are stitched before the authoritative comparisons;
driver fits to a final fragment are separately qualified and retained.

Automatic Aurora follow-on controllers have been stopped. New work is local
only for now. For future Aurora allocations the user selected one node on
`capacity`, account `CompactBinaryMerger`, for `00:30:00`, with at most one task
allocation queued or running. Fresh packaging records these resources and
submission checks reject conflicting PBS settings. Workers reserve time for
native checkpoints before the shorter wall limit. Historical job files are
unchanged. The policy checks executed no scheduler commands.

The local exploration preserves the scalar evolution operator and background
provider. An isolated copy adds a passive exact-scri waveform observer every
`0.5M`; removing that observer reproduces the original source exactly. An
initialization-only check gives identical raw scalar checkpoints and all four
frozen-cache fingerprints. It records both raw angular projections and
real/imaginary harmonic Gram-normalized radiation modes `r Phi = S psi`.
The separately preflighted local `600x8` complex Kerr solver supplies an
exploratory comparison; no independent-reference resolution claim is made.
New runs are bounded by local wall checkpoints and continue the exact binary
and scalar settings, without repeating completed trajectories.

## Physical backgrounds and independent reference

Minkowski is exact. Schwarzschild and Kerr use the exact stationary
Dennison–Baumgarte–Montero trumpet with `R0=M`, spherical Cartesian coordinates
`x=r n`, and Boyer–Lindquist radial label `R=r+M`. Source provenance is
`origin/project/kerr-trumpet-spin09`, commit
`81c7d8030c49489577c54b977b8b7f7b42d9a96e`,
`src/coordinates/kerr_trumpet.hpp`. A saved copy is evidence, not an evolved
metric dependency. See [the original paper](https://arxiv.org/abs/1409.1887).
The scalar implementation uses only its physical four-metric, with independently
checked physical pullbacks. It does not import the provider's Z4c conformal
fields, constraints, or reference-subtracted gauge equations.

With `c=sqrt(M^2-a^2)`, `Sigma=R^2+a^2 n_z^2`, `A=R^2+a^2`,
`X=A^2-a^2 r^2(1-n_z^2)`, and `w=(-n_y,n_x,0)`,

```
alpha = r sqrt(Sigma/X)
beta = [c A r n - a(2Mr+M^2+a^2) r w]/X
gamma_ij = [Sigma delta_ij + a^2(1+2MR/Sigma) w_i w_j
            - a c(n_i w_j+w_i n_j)]/r^2
sqrt(-g) = Sigma/r^2.
```

This gives an exact horizon-penetrating four-metric through
`g_TT=-alpha^2+gamma_ij beta^i beta^j`, `g_Ti=gamma_ij beta^j`.
At zero spin, `alpha=r/(r+M)`, `beta^i=M x^i/(r+M)^2`, and
`gamma_ij=((r+M)/r)^2 delta_ij`. The horizon is `r=c`; the limiting trumpet
surface is `r=0`, with nonzero physical area and infinite proper distance.
These analytical trumpets are a different slicing from stationary 1+log.
The paper calls `r=R-R0` quasi-isotropic, and notes that `K` depends on
direction at the Kerr trumpet surface. It does not provide a stability
guarantee for puncture numerics; the Kerr scalar convergence and small-cube
sensitivity tests remain necessary. The added outer hyperboloidal height
below is derived for this physical chart, rather than imported from the paper.
No small-deviation expansion, lapse floor, interior metric regularization, or
backreaction is used. The even, cell-centred Cartesian grid omits the puncture.
Optional excision is rejected if any tested actual face has an incoming scalar
characteristic. At zero excision the bulk has no artificial inner boundary.

The separate Minkowski hyperboloidal reference is constructed with `M=a=0`
and `h0=sqrt(r^2+C^2)-C`. It never acquires the black hole's height or mass.
Both physical and reference coefficients are cached before stepping in one
read-only Kokkos geometry view; neither cache is written during evolution.
The maps and parameter structures are const throughout a run.
The physical RHS uses the full physical coefficients. Subtracting the separately
stored reference gives their full difference; no linearization of that
difference or replacement of the physical inverse by a reference inverse is
made near the horizon.
`--audit` exports both inverse metrics and their full difference, as well as
separate actual/reference curvature potentials and densities. Reconstruction
from reference plus difference agrees with the actual metric to `2.5e-17`
relative error in the saved audit; the Schwarzschild horizon difference in
the time-time inverse coefficient is about `-3.01`, not a small perturbation.

## Fixed hyperboloidal transformation

Let `rho=|y|`, `x=y/Omega(rho)`, `r=rho/Omega`, and
`lambda=Omega-rho Omega'`. Use `C=10`, `S=24`,

```
Omega = 1 - smoothstep((rho-8)/8) rho/S,
smoothstep(z) = 0 (z<=0), z^3(10-15z+6z^2) (0<z<1), 1 (z>=1).
```

For a black hole, `T_tr=t+h(r)` with the fixed derivative

```
h'(r) = smoothstep((r-8)/8)
        [r/sqrt(r^2+C^2) + (2M+c)/r].
```

Define `h(0)=0` and its integral; the coefficients use its exact derivative.
Thus `h'=0` in the trumpet interior. At infinity,
`h=r+(2M+c) ln(r)+C^2/(2r)+constant`. The logarithm is required:
`g_tr^{TT}=-1-2M/r+O(r^-2)`, `g_tr^{Tr}=c/r+O(r^-2)`,
`g_tr^{rr}=1-2M/r+O(r^-2)`. Cancellation in the transformed time-time
coefficient requires the `1/r` term in `h'` to be `2M+c`.
The discarded Kerr–Schild proposal had a different logarithmic coefficient;
no Kerr–Schild evolution was run. Geometry is checked to remain spacelike.

The original, admitted Minkowski configuration uses this quintic transition
and a linear shell map. The experimental `--smooth-layer 1` configuration
replaces the **physical** transition by the C-infinity function
`w(z)=exp(-1/z)/[exp(-1/z)+exp(-1/(1-z))]`, with constant endpoint extensions.
It also uses this physical function for the black-hole height switch and
initial inner taper. The reference always retains the original quintic
function and `h0`, including when the physical layer is changed. The optional
`--radial-stretch -2` changes grid sampling using the existing cubed-sphere
map; it does not change either continuous metric function. All selections
remain fixed throughout each run.

Use `bar_g=Omega^2 g`, with its actual physical pullback. `geometry.hpp`
algebraically factors the time-time numerator by `Omega^2` without dropping
higher terms. All first spatial coefficient derivatives use analytic jets.
The factorization makes exact scri evaluable without large-radius clipping:

```
bar_g^{tt}|scri = -[C^2+4M(M+c)-a^2(1-n_z^2)]/S^2
bar_g^{ti}|scri = -n_i
bar_g^{ij}|scri = delta_ij-n_i n_j
sqrt(-bar_g)|scri = 1.
```

For an outward unit radial covector, scalar speeds are
`b +/- sqrt(b^2-bar_g^{nn}/bar_g^{tt})`, where `b=bar_g^{tn}/bar_g^{tt}`.
At scri they are `0` and `2S^2/[C^2+4M(M+c)-a^2(1-n_z^2)]`.
The zero branch needs no prescribed incoming data. At finite truncations the
incoming branch is measured and an asymptotic scalar characteristic penalty
is available. Analytic finite-radius ghosts are only a Minkowski verification
control. Polynomial interface/outer extrapolation has no general SBP stability
proof; the declared convergence and sensitivity tests decide admission.
The outer entries of `characteristics.csv` use the logical radial covector
`d s`, so their magnitudes include the radial map Jacobian. Their signs are
the physical scalar characteristic signs. Excision entries use Cartesian
unit face covectors pointing into the removed cube; both speeds must leave
the evolved domain. Finite caps `.9` and `.95` have very large physical radii;
their early sensitivity tests do not establish immunity to later causally
returning reflections.
The current guard checks face centres, corners and edge midpoints. A dense
independent face audit finds incoming corner modes for some Kerr cubes even
inside the horizon. The initial spin `.5` Kerr stage therefore uses the full
puncture and, on the finest grid, a small verified outflow cube as an inner
sensitivity control. Larger cuts that fail characteristics are rejected before
stepping. The saved spin `.9` checks are static geometry checks only.
An additional exact all-face bound applies in the inner trumpet chart, where
`Omega=1` and `h'=0`. Writing `u=c n-a w`, its physical inverse spatial block
is `(r^2 delta_ij-u_i u_j)/Sigma`. A cube of halfwidth `d` has strictly
outgoing characteristics on every inward face if `d<(c-|a|)/3`: this makes
`g^{AA}<0` and the outward shift projection positive everywhere, including
edges and corners. The bound is sufficient. It proves outflow for the current
Schwarzschild `.25/.20` cubes and the planned spin `.5` small-cube controls;
it gives no such guarantee at spin `.9`. The original face probes and dense
independent numerical checks remain recorded separately.

## Equation and scalar reduction

Solve `Box_g Phi=0`. Set `Phi=Omega psi`. Since the *physical* backgrounds
are vacuum, the conformal transformation gives

```
(Box_bar_g - R[bar_g]/6) psi = 0,
V = R[bar_g]/6 = -Omega^-3 Box_g Omega.
```

`V` uses the actual conformal physical metric, never the reference curvature.
For the trumpet, `Delta=r^2-c^2` and a radial Omega,

```
Box_g Omega = (1/Sigma) d_r(Delta d_r Omega),
V = -Omega/(lambda Sigma_bar)
    d_rho[(rho^2-c^2 Omega^2) Omega'/lambda],
Sigma_bar=(rho+M Omega)^2+a^2 Omega^2 n_z^2.
```

The azimuthal radial flux has zero divergence. This exact expression is
regular at scri, where `V=0`. Independent value-only conformal-metric curvature
contractions check the potential, in addition to inverse and divergence checks.
The reference potential is stored separately and never enters the physical RHS.
An independent value-only 3+1 pullback also checks the stationary normal-scalar
source terms. Differentiating its spatial metric gives `K` and all three
contracted connections, including the Kerr azimuthal terms. Sixty probes at
spins `0,+.5,-.5,.9` agree with the cached-wave-operator source algebra to
`5.2e-16` relative error. This is a static geometry/source check.

The original coordinate-velocity variant evolves five scalar fields:
`psi`, `v=partial_t psi`, and
`q_i=partial_i psi`, all in the common compact Cartesian basis. Define
`B^mu=(1/D) partial_i(D bar_g^{i mu})`, `D=sqrt(-bar_g)`. Then

```
psi_t = v
v_t = -[2 bar_g^{ti} partial_i v + bar_g^{ij} partial_i q_j
         + B^t v + B^i q_i - V psi]/bar_g^{tt}
(q_i)_t = partial_i v + beta^j(partial_j q_i-partial_i q_j)
beta^i = -bar_g^{ti}/bar_g^{tt}.
```

The curl addition is the usual scalar first-order reduction: it vanishes
when `q=d psi` and transports reduction modes with the frozen shift. There
are no independently evolved constraint models. Measure `q_i-D_i psi` and
independently differentiate the evolved field for derivative convergence.
With nonzero scalar reduction damping `kappa`, the coordinate-velocity variant
uses `psi_t=v-beta.C` and adds `-kappa C_i` to the auxiliary equation and
`-kappa beta.C` to `v_t`, where `C=q-Dpsi`. Here `v` is an auxiliary equal to
the coordinate-time derivative on the reduction surface. The physical
coordinate-time derivative diagnostic is `v-beta.C`; older pilot histories
report the auxiliary `v`. Black-hole history columns named `psi_L2` and
`Dpsi_L2` are field norms, not errors against an exact black-hole solution.
The independent-profile validators provide those errors.

Experimental `--compatible-reduction 1` differentiates the discrete scalar
RHS when updating `q`; its temporary `C` array is derived, never evolved.
It damps the reduction residual to roundoff, but the finer matrix exposes
field growth at the Cartesian/shell interface and is **not admitted**.
Small reduction residuals alone do not establish a correct evolution.

Experimental `--normal-upwind 1 --ko 0 --outer-sbp 0` instead evolves exactly
`psi,Pi,q_i`, with `Pi=(beta.q-v)/alpha`. Its stationary normal scalar system is

```
psi_t = beta^i partial_i psi - alpha Pi
Pi_t = beta^i partial_i Pi - alpha gamma^ij partial_i q_j
       + alpha K Pi + (alpha Gamma^j-gamma^ij partial_i alpha) q_j
       + alpha V_actual psi
q_i,t = beta^j partial_j q_i - alpha partial_i Pi
        + (partial_i beta^j) q_j - (partial_i alpha) Pi - kappa C_i.
```

The lapse and shift are fixed algebraic functions of the cached actual
conformal inverse metric; their analytic spatial derivatives are cached.
`K` and contracted connections are stationary
coefficients, algebraically obtained from the same cached wave operator;
they are not evolved. In each logical direction the physical characteristic
projections use `-beta^A +/- alpha sqrt(gamma^AA)` and `-beta^A`, with
fourth-order upwind derivatives and one-sided outflow endpoint derivatives.
This endpoint variant passes the short Minkowski tests but fails long-time
Minkowski and Schwarzschild evolution; it is not admitted for ringdown.
The normal implementation requires KO=0 and excludes characteristic ghost
mixing. Output snapshots convert `Pi` to
the coordinate-time derivative `beta.Dpsi-alpha Pi`; raw restart checkpoints
retain `Pi`. `parameters.json` identifies the stored auxiliary explicitly.
The snapshot coordinate velocity is a reconstruction using the independently
centred spatial gradient; the semidiscrete scalar RHS uses its upwind gradient.
These are consistent approximations differing at truncation order. Derivative
convergence does not rely on treating an auxiliary gradient as an exact
derivative of the evolved field.
An additional saved-state audit evaluates the actual compiled scalar RHS
without taking another time step. The undamped `N24/32/40` semidiscrete
`psi_t` converges in both norms and every region, with minimum order `3.85`.
At `t12` it reads the original full checkpoint; the independently reconstructed
`Pi` differs by at most `2e-20`. All four frozen-cache fingerprints remain
unchanged. Thus the time-derivative check also covers the scalar RHS itself.

The earlier `N=24,32,40` Minkowski normal-momentum matrix passed its short CPU
tests: field/velocity-reconstruction/spatial-derivative/reduction convergence,
half/quarter timestep controls, finite caps `.9/.95`, actual characteristics and
frozen-cache checks. All regional diagnostics also pass. The minimum pair order
is `1.511`, the finest normalized reduction maximum `0.002459`, maximum timestep
fraction `1.35e-7`, and maximum quarter/half difference ratio `0.0851`. Finite-cap
waveform sensitivities are `2.20e-5` and `1.12e-5`. The coarser failed matrix
remains failed. See `minkowski-normal-cpu-admission.json` for exact arguments,
source/operator equivalence controls and audit hashes. The subsequent
`normal-admission-scope-update.json` limits this result to `t<=12`: the same
endpoint method grows at scri and fails at `51M` in the long Minkowski control
and `62M` in Schwarzschild. No finer black-hole run was advanced with it.

The replacement `--normal-upwind 1 --outer-sbp 2 --ko 0` uses the diagonal-norm
SBP6/3 radial derivative `Dc` and the positive upwind pair
`Dplus/minus = Dc +/- H^-1 D4^T D4/(128 h)`. The truncated fourth difference
`D4=[1,-4,6,-4,1]` annihilates cubics; its contribution has third-order endpoint
and sixth-order interior consistency. The compiled rows agree exactly with
an independent matrix construction, including the endpoint closure. The
SBP identity, positivity, polynomial and homogeneous-inflow energy checks
pass. Cartesian and angular derivatives retain the fourth-order characteristic
upwind method. This radial proof is not a proof of the nonconforming interfaces.
Only this variant needs four halo cells; the original default retains three.
Separate regression controls reproduce both original and replacement fields
bit for bit, without altering any frozen geometric coefficients.

The SBP pair reaches `100M` in the coarse Minkowski control and Schwarzschild
`N24` pilot. It still lacks scientific admission. The fixed-damping
`N24/32/40` matrix passes field and derivative checks but fails late maximum
reduction convergence (orders `1.343`, `1.420`). A predeclared grid-scaled
damping control also fails (`0.508`, `1.462`) and is rejected. Those residuals
are above roundoff; no roundoff exemption is applied. The next predeclared
fixed-damping matrix is `N40/48/56`, with `N48/56/64` conditional on failure,
reusing every completed matching run. All thresholds remain unchanged.

Logical fourth-order centred differences, sixth-order KO, RK4 and active-donor
six-point interpolation operate on the original cube-to-six-block map. The
standalone bulk is uniform; no scalar AMR claim is made.

The analytic Minkowski quadrupole is `partial_X partial_Y` of
`[F(T-r)-F(T+r)]/r`, with `F(u)=exp(-((u+4)/1.2)^2)`. Its scalar field,
time derivative, and compact Cartesian gradients are evaluated independently
of finite differences, including the exact scri limit. Norms are weighted by
compact coordinate volume, with both L2 and maximum errors reported for bulk,
shell and the full domain. They are not physical-volume energy estimates.

## Reproduction and admission

Build independently:

```sh
cmake -S src/test_scalar -B build-test-scalar -DKokkos_ENABLE_SERIAL=ON
cmake --build build-test-scalar -j 6
build-test-scalar/test_scalar --n 12 --tlim 12 --output /absolute/new-result
```

`--mass` selects the background black-hole mass, and `--spin` selects `a/M`.
The scalar mass and spin weight are always zero.

The evidence directory is `docs/validation/test-scalar-20261016`.
`criteria.json` was saved before evolution. `trumpet-amendment.json` records
the user's slicing correction, also before evolution. Existing failed audits,
compiler logs, run receipts, binary/source hashes, full field checkpoints, and
plots are retained. `tst/test_scalar/run_cpu.py` refuses to overwrite an existing
attempt and reuses a matching invocation. The CPU and GPU admission reports
must distinguish completed computations from passed scientific gates.
Scalar-only restarts require matching physical and reference fingerprints,
payload size, the preserved executable, and the checkpoint SHA256 receipt.
The helper carries saved numerical settings forward. A bounded wall-time
exit writes `checkpointed.txt`; only reaching the target writes `complete.txt`.
Both physical and reference caches and their added lapse/shift derivatives
are hashed before and after stepping. Independent derivative audits use
high-precision value-only pullbacks; native `long double` on this Mac does
not supply extra precision beyond `double`.

Spin-weight-zero scalar reference frequencies come from the independent
[qnm Leaver solver](https://qnm.readthedocs.io/en/latest/README.html):
`M omega_220=0.48364387221071264-0.09675877597828755i` at zero spin and
`0.5859896293050021-0.09349441899180953i` at `a/M=0.5`.
The real Kerr initial data can excite both frequency branches. Its planned
complex fit therefore leaves both frequencies and amplitudes free and checks
the independent `m=2` and `m=-2` scalar Leaver targets, including the declared
window-spread gates. The independent regular `m=2` 2+1 solver preserves
complex scalar checkpoints; its Cartesian field/gradient reconstruction
reduces to the separate Schwarzschild implementation to about `1e-16` in
the saved profile-validator control. No nonzero-spin evolution is implied.
The requested sequence remains Minkowski, Schwarzschild ringdown, then Kerr
ringdown. No GPU stage is admitted before its CPU scientific gates pass.

In the earlier admission history, the original angular Minkowski CPU matrix
passes at exact scri, including field/derivative convergence, time controls,
reduction checks and finest-grid finite-cap sensitivity. The medium-grid
`.9` cap failure is retained. The independent Schwarzschild normal-momentum
reference passes its resolution, timestep, inner-boundary and QNM checks.
At that stage the new SBP normal-momentum operator remained unadmitted while
its finer Minkowski matrix and controls were evaluated. The Schwarzschild
`N24` pilot reaches `100M` but has `2.321%` independent waveform error, above
the `2%` gate, and fails the declared free-frequency/window checks. Finer
Schwarzschild runs await the new Minkowski admission; no nonzero-spin
Kerr time evolution has been launched. The independent 2+1 reference matches
the Schwarzschild reference at zero spin to `9.2e-16` in relative waveform L2.

Aurora job `8936111` finished with scheduler exit 0 using `debug`,
`CompactBinaryMerger`, one node and 60 minutes, after the original Minkowski
CPU admission. Its output has now been collected and audited: all remote and
collected hashes agree, the three resolutions pass analytic convergence, and
the maximum scaled CPU/GPU field or derivative difference is below `2e-14`.
The saved backend output identifies SYCL and an Intel Data Center GPU Max 1550.
This GPU admission applies to the original Minkowski configuration only.
Earlier configuration and build failures are retained.
The task controller checks its exact job name and uses a persistent submission
lock to prevent overlapping or duplicate allocations.

A separately declared CPU-only Aurora allocation accelerates finer fixed-damping
convergence and numerical controls; it uses the same one-node debug queue,
account and 60-minute limit. It contains no GPU evolution. Native wall-time
guards preserve the full scalar checkpoint before the allocation deadline.
The new operator and black-hole GPU stages remain gated on their CPU admission.

The completed local `N48` fixed-damping run fails the `N40/N48` late maximum
reduction order (`0.557`), although its field and derivative checks pass.
An additional predeclared undamped (`kappa=0`) scalar control reaches `100M`
in the coarse Minkowski stability test and passes the full `N24/32/40`
field/velocity/derivative/reduction matrix in all regions (minimum order `3.65`,
finest normalized reduction maximum `0.00120`). The half/quarter-timestep
controls pass (maximum spatial fraction `5.16e-7`, quarter/half ratio `0.0678`),
as do the `.9/.95` finite caps (waveform sensitivities `1.25e-5`, `7.58e-6`).
This configuration has full source-bound **Minkowski CPU admission** in
`minkowski-undamped-sbp-cpu-admission.json`. The `.95` case ran on Aurora's CPU;
its matching-source backend control agrees within `1.92e-11` of the joint
initial scale, and its physical geometry/source probes pass independently.
The native local guard prevented duplicate `.95` evolution. Shared Cartesian
and cubed-sphere dependencies are bound by hash as well as the scalar sources.
The optional damping changes neither
the continuous equation on the scalar reduction surface nor any geometry.

GPU job `8937333` completes the three resolutions and passes direct analytic
convergence and frozen/backend checks. Its strict CPU agreement gate **fails**:
the `N40` maximum scaled difference reaches `1.85e-4`, above `1e-9`, despite
matching timesteps and coordinates. This configuration has no GPU admission.
Frozen interpolation/operator exports are being compared across GCC, icpx CPU
and SYCL; the tolerance is unchanged. A diagnostic audit-build failure is
preserved, and its corrected two-phase build has passed preflight.
The new undamped Schwarzschild CPU matrix follows the separate
`schwarzschild-undamped-sbp-plan.json`. Its `N24` case reaches `100M` but has
`2.183%` independent waveform error and `3.286%` damping window spread, so no
black-hole admission follows from that coarse run. The completed `N24/32/40`
matrix has finest waveform error `0.1657%` and passing scalar QNM/window checks,
but several late field/derivative/reduction convergence orders fail. The matrix
remains failed. A separately declared `N40/48/56` refinement reuses `N40`;
`N48` was started with bounded checkpoint continuations and later stopped at
`70M` after the endpoint defect below was identified. The required controls
remain outstanding. No nonzero-spin time evolution has begun.

The subsequent frozen-operator comparison identifies a **physical geometry
evaluation defect** in the smooth-layer configuration. Near
`rho=15.999872713026816`, forming `w*(1-w)` from a rounded `w` amplifies
roundoff in its spatial derivative. GCC and icpx shift-gradient coefficients
differ by up to `0.01338`. Replacing only transfer rows changes the initial
RHS by about `9.2e-15`; the full backend discrepancy is `2.62e-9` there.
An independent 80-digit physical-metric pullback confirms the defect: the
old evaluation fails the new 144-probe endpoint audit with scaled error
`0.01094`. Thus the smooth-layer v21 CPU admission above is **superseded**;
it is preserved as evidence and cannot authorize further evolution stages.
The immutable original quintic Minkowski admission remains valid.

The corrected scalar-only evaluation uses the two exponentials directly for
the quotient jet and computes `w*(1-w)=ab/(a+b)^2`. This changes no continuous
coordinate map, height function, conformal factor or background. The same
endpoint audit now passes at `2.11e-16`. The separate quintic reference cache
and its derivatives match the previous fingerprints bit for bit, and the
original default scalar fields also match bit for bit. All 642 protected
production baseline files match their archived hashes.
The correction and unchanged quantitative gates were declared before editing
in `stable-smooth-endpoint-plan.json`. The corrected configuration has full
source-bound **Minkowski CPU admission** in
`minkowski-stable-sbp-cpu-admission.json`: all global/regional field, velocity,
gradient and reduction orders exceed `3.64`; actual compiled `psi_t` orders
exceed `3.83`. The timestep fraction is below `5.16e-7`, quarter/half ratio
below `0.0677`, and the two finite-cap waveform sensitivities remain below
`1.26e-5`. Coarse `t100` late stability is below `0.000873` of its initial
peak. More than one million cached Minkowski points also match independent
closed physical coefficients to `4.33e-15`.
Aurora CPU job `8937562` supplied the reserved fine case, caps and stability
control; its 91 collected files match every saved hash. CPU-only job `8937629`
evaluated the saved fine-grid RHS without time stepping, checking the frozen
inputs and caches; all 12 collected files match. The next corrected GPU matrix
is job `8937736`, with the original strict agreement gate unchanged. It now
has **Minkowski GPU admission**: all 49 collected files match their hashes,
all direct analytic convergence checks pass, the actual backend is Intel SYCL,
and the maximum scaled CPU/GPU field or derivative difference is `4.35e-14`,
below the unchanged `1e-9` limit. Fresh physical geometry and normal-source
probes from that executable also pass independently. The differing donor
indices in the older static exports are exclusively unused corner ghosts for
this first-order scalar operator; no transfer policy was changed.
The old Schwarzschild `N48` checkpoint at `70M` is fully
preserved and its controller stopped before continuation. Corrected
Schwarzschild `N24` CPU evolution reaches `100M` with independent waveform error
`2.183%` and damping window spread `3.286%`, both above their gates; its
`N32/N40` cases are reserved for the next allocation after the Minkowski GPU
review. The completed corrected `N24/32/40` matrix still fails late field,
derivative and reduction convergence, despite finest waveform error `0.1657%`
and passing finest scalar QNM/window checks. The actual inner cube halfwidths
are `.125/.09375/.15`, a geometric change between resolutions that the next
experiment removes. The predeclared
`schwarzschild-commensurate-boundary-plan.json` uses `N24/36/48` and fixed actual
cube halfwidth `.25M`, with unchanged scientific tolerances. Exact inner
Schwarzschild characteristics give a positive lower bound `0.0304` over the
entire cube face. CPU job `8937936` runs this new matrix after the previous
allocation finishes. This is a new computational-boundary experiment; it
does not replace the failed matrix or alter the physical metric, height,
conformal factor or Minkowski reference. No corrected Schwarzschild or Kerr
admission is claimed.

An independent characteristic travel-time audit finds that the original `.9`
and `.95` caps first return a reflected pulse to the extraction point after
about `430M` and `913M` in Schwarzschild, beyond the planned `100M` runs. A
closer `.6` cap returns around `60M` and is planned as an additional causal
boundary diagnostic. Early far-cap agreement must not be described as a test
of late returning reflections. Every black-hole half-time report now includes
common-exterior norms and scalar reduction maxima, including times between
the full field snapshots. Added diagnostics were verified to leave fields
and cached geometry bit-for-bit unchanged.

The completed fixed-cube `N24/36/48` matrix (job `8937936`, 56 collected
files with matching hashes) still fails the late coarse-pair velocity, gradient
and reduction maximum-norm orders (`1.44`, `0.61`, `0.685`). These failures
remain recorded. The fine `N48` waveform error is `0.0542%`, its scalar QNM
checks pass, and its normalized reduction maximum is `0.00706`. The finer pair
passes every order check. Before new evolution,
`schwarzschild-commensurate-finer-plan.json` declares `N36/48/60`, reusing
completed `N36/N48` and adding only `N60`. The same physical geometry, fixed
inner cube, reference and tolerances apply. Full CPU admission still requires
timestep, boundary and actual compiled time-derivative checks.

The new `N60` run reaches `100M` through an exact same-binary checkpoint at
`74.5M` (jobs `8938093/8938174`). All 40 collected files match their hashes,
and all four physical/reference coefficient fingerprints match across the
restart. Its independent waveform error is `0.0153%`, every scalar QNM/window
check passes, and its normalized reduction maximum is `0.00270`. The
`N36/48/60` matrix nevertheless **fails** three original PCHIP error orders.
The failed review and plots are preserved; no CPU or GPU admission follows.

A preregistered reference audit isolates two early maximum-norm failures to
shape-preserving profile interpolation. A six-point value interpolator passes
degree-five real/complex polynomial checks to `1.99e-15` and independent
smooth-function refinement at order `6.00`. Reassessing the existing fields
with it removes those two early failures. Refining the independent radial
solution to 2400 points bounds its profile uncertainty by `2.08%` of the
finest primary errors in both norms, below the new `10%` diagnostic limit.
The remaining late field L2 order is still `1.4017 < 1.5`. This is a genuine
unresolved numerical gate, rather than the profile-sampling issue. The new
independent solver also saves every scalar auxiliary in a full checkpoint;
its output-only change matches the original short waveform/profile bit for bit.

`scalar-auxiliary-damping-diagnostic-plan.json` declares one CPU-only `N36`
comparison with the already implemented scalar auxiliary damping `kappa=2`.
The physical background, reference, coordinate functions, grid and initial
data are identical to the undamped run. Only the existing derived `q-Dpsi`
feedback changes; no constraint model or additional state is evolved. The
original undamped failure remains recorded. This diagnostic is not an
admission and cannot authorize a black-hole GPU run. Any candidate still needs
the complete source/argument-bound scalar validation sequence.

The damping diagnostic completed in job `8938287`; all 22 collected files
match their hashes. Its initial scalar state and all four frozen coefficient
fingerprints match the undamped comparison exactly. The late field L2 error
is `2.2484` times larger, and QNM window consistency fails, despite the smaller
normalized reduction residual (`0.01081`). It fails its predeclared promotion
screen and is not promoted. The current numerical configuration keeps
`kappa=0`.

`aurora/cpu-memory-rhs-audit-v1/scientific-plan.json` declares a read-only
audit of the saved `N36/48/60` states and a CPU DDR/HBM placement benchmark.
Job `8938386` uses the same debug/account/node/time limits as the validation
runs. No time steps are taken. Both placements must produce bitwise identical
values for all five scalar RHS components, leave every active scalar value
unchanged, and preserve all four geometry fingerprints. Performance alone
cannot admit any scientific stage. The actual compiled time derivative is
also assessed against the independently resolved 2400-point radial solution
using the verified six-point sampler.

This allocation completed successfully. All 53 collected files match their
hashes. The full five-component active RHS is bitwise identical between DDR
and HBM placement; no active scalar value or frozen coefficient changes.
Default placement takes `43.7ms` per fill/evaluate pair; HBM takes `63.8ms`.
Default placement is retained. The actual compiled `psi_t` and reconstructed
velocity converge in both norms at all four times, with minimum order `2.791`.
The unified resolved-oracle certificate also passes, including independent
profile resolution, timestep, inner-boundary and free scalar QNM/window
controls. These results do not cure the remaining late field convergence gate.

`schwarzschild-dyadic-plan.json` declares a separate `N24/48/96` family before
new evolution. Every logical grid direction doubles; the actual excision cube
stays at halfwidth `.25M`. It reuses the completed corrected-source `N24/N48`
trajectories and adds only `N96`, using default CPU memory placement and the
same preserved executable. The original `N36/48/60` failure remains recorded.
The new family retains the `1.5` field/derivative/reduction order gates, all
waveform/QNM tolerances, medium-grid half/quarter timestep controls and
finest-grid inner/outer controls. Its independent-reference uncertainty must
also pass at the actual new finest grid points. Native wall checkpoints are
resumed serially in one-node debug allocations of at most 60 minutes. Initial
job `8938435` starts the missing `N96` trajectory. No black-hole stage is
admitted by this submission or by trajectory completion alone.

A verified closed `N96` snapshot at `2M` passes all eight early field,
velocity, spatial derivative and reduction order checks in the dyadic family.
It is explicitly partial evidence, without end-of-run frozen integrity or
stage admission. At this finer grid, the 1200/2400 reference difference is
`13.80%` of the velocity maximum error, above the unchanged `10%` uncertainty
gate. The preregistered `schwarzschild-reference-n4800-plan.json` therefore
adds only a new 4800-point independent radial solution. It reaches `100M`,
preserves the full scalar checkpoint, passes all free scalar QNM/window
checks, and differs from the 2400-point waveform by `1.35e-8` relative L2.
Using the 2400/4800 difference lowers the early maximum uncertainty fraction
to `0.00860`; all early primary order checks still pass. The immutable
`cpu-schwarzschild-dyadic/reference-selection.json` selects this oracle for
the full assessment. No primary field, timestep, geometry, map or run is
changed or duplicated. All later times and sensitivity controls remain
outstanding.

The waiting Kerr validators now also expose the verified six-point smooth
profile sampler. `kerr-reference-sampling-amendment.json` retains the existing
`N36/48/60` family and every scientific threshold before any nonzero-spin
evolution. A non-QNM complex polynomial instrument compares the reconstructed
real field and Cartesian derivatives with independent 50-digit value
derivatives at 96 off-grid points. Its maximum errors are below `1.84e-15`,
and the default PCHIP diagnostic reproduces its previous errors bit for bit.
This is a measurement change; the primary scalar operator and physical
backgrounds are unchanged. Kerr remains gated on full Schwarzschild CPU/GPU
validation.

`kerr-reference-uncertainty-amendment.json` also predeclares the measured-error
uncertainty gate for Kerr before any Kerr time evolution: both norms of the
independent coarse/fine field, velocity and Cartesian-gradient difference
must be at most `10%` of the actual finest 3D errors at `2/6/12/100M`.
An unresolved certificate stops the CPU controls and GPU stage; completed 3D
states are preserved while only the independent oracle is refined. Exposing
the existing reconstruction for this comparison leaves the default errors
bit-for-bit unchanged and repeats the independent 50-digit polynomial
instrument with the same errors below `1.84e-15`.

The local `observe_scalar_evidence.py` controller saves field/convergence and
free-QNM plots when new CPU or GPU primary reports appear. It writes a final
scalar summary only after all three complete CPU admissions and their direct
GPU reviews pass, with protected-baseline and current-source checks. The
summary always records that metric evolution remains incomplete. Transport
now uses compressed rsync with the same complete native-file SHA256 checks;
a copied binary RHS instrument matches the original bytes exactly. None of
these collection or plotting changes modifies a native scalar run.

The first three `N96` allocations (`8938435/8938621/8938729`) close at `39M`.
All 57 collected files match their hashes; each same-binary restart preserves
all four physical/reference geometry and derivative fingerprints. A read-only
assessment of the closed data passes all early `2/6/12M` field, velocity,
gradient and reduction orders, with minimum `3.551`. The 2400/4800 independent
reference uncertainty is at most `0.860%` of the actual finest errors. The
closed waveform error through `39M` is `1.15e-5` relative L2, and every closed
half-time reduction report gives a normalized maximum at most `0.0004162`.
These are partial indicators, not a `100M` or stage admission. The next
same-trajectory continuation is job `8938817`; all late-field, timestep and
inner/outer-boundary gates remain required.

The fourth closed `N96` allocation reaches `52M`, preserving the same four
geometry fingerprints. The fifth, job `8938884`, last reports scheduler state
`E` with exit status zero after finite live diagnostics through `64.5M`.
SSH access then fails during collection; its final remote payload has not yet
been confirmed. `access-recovery-20261017` preserves the interruption, original
controller sources, and recovery checks. The recovered controller first
inspects and collects that existing submission, then continues its exact
native checkpoint. It does not start a replacement trajectory. Scheduler
reads now wait through SSH outages; partial collections still require explicit
inspection, and wrong-job journals are refused. A failure-injection instrument
checks these guards and collection exactly once without submitting any job.
All 642 protected baseline files and the admitted scalar operator remain
unchanged. Schwarzschild admission and all Kerr evolution remain pending.

The read-only closed-prefix assessment through `52M` also passes the early
order and oracle-uncertainty checks. Its waveform relative L2 error is
`1.153e-5`; the normalized reduction maximum remains `0.0004162`.
`schwarzschild-dyadic-closed-t52-v22` preserves the inspected plot and hashes.
This assessment adds no native trajectory and does not replace the outstanding
`100M`, timestep or boundary gates.
