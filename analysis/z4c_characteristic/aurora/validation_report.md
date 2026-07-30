# Residual Z4c characteristic CPBC: Aurora validation

## Scope and current verdict

This change implements an opt-in, background-consistent, all-sector \(L=1\)
residual characteristic boundary treatment for the supported
background-adapted Z4c gauge.  It is a fused active-cell
Bjørhus/method-of-lines compatibility correction.  Residual ghost cells
remain polynomially extrapolated for centered finite-difference and KO
stencils; analytic-background fields are reconstructed independently and are
never extrapolated into the residual state.

The algebra, exact-background, normal-incidence reflection, convergence,
orientation/sign symmetry, edge/corner determinism and convergence,
unsupported-gauge rejection, compact-launch parity, mesh-layout, radial
sponge, and far-boundary TDE-reference gates pass.  At the finest tested
spacing, far-control-subtracted normal-incidence reflection is at most
`5.11409524e-3`, versus `4.203e-2`--`7.724e-2` for the matched extrapolating
Sommerfeld path.  The CPBC median improvement is `8.21885`; the weakest
per-case improvement is `8.21880`.

This report is not yet a delivery acceptance.  The first small-boundary
cell-relaxation CPBC case was stable but failed the returning-signal gate; a
mathematically corrected zero-rate compatibility candidate has passed the
normal-pulse and exact-background gates and is in the small-boundary TDE
gate.  The production timing/smoke run also remains open.  The two strict
legacy-regression threshold misses have been classified by byte-identical
comparison with an untouched base build.  No commit or push is permitted
until the remaining gates are resolved.

## Mathematical claim and limitations

The symbolic and implemented projectors are derived directly from AthenaK's
background-adapted residual RHS.  No characteristic coefficient is copied
from a published Z4c decomposition.  The finite-residual normal symbol keeps
the distinct full-state and analytic-background shift advections.  In the
local conformal orthonormal frame its characteristic polynomials are

```text
scalar:
  [(lambda-BF)^2-C N^2]^2
  [(lambda-BF)(lambda-BB)-C L]
  [(lambda-BF)(lambda-BB)-4 G/3]
vector:
  [(lambda-BF)^2-C N^2]
  [(lambda-BF)(lambda-BB)-G]
tensor:
  [(lambda-BF)^2-C N^2]
```

Here `BF` and `BB` are the full and background normal shifts, `C=chi`,
`N=alpha`, `L` is the background-adapted lapse driver, and `G` is the
effective Gamma-driver coefficient.  The hybrid gauge roots are

```text
lambda = (BF + BB +/- sqrt((BF-BB)^2 + 4 Q))/2,
```

with `Q=C L`, `4 G/3`, or `G`.  With the code's sign convention,
positive `lambda` has negative outward coordinate velocity and is incoming.
The startup checks require one positive and one negative member of every
pair, giving ten incoming modes per open face.
The production `mu_S=1` transverse-shift/physical-speed coincidence is
supported and remains diagonalizable; only singular coincidences of the
longitudinal-shift row with the lapse or physical scalar rows are rejected.

The four light-speed constraint rows have also been checked directly against
the principal Z4 constraint subsystem.  Their normal derivatives equal the
incoming combinations of the Hamiltonian constraint, the three momentum
constraints, \(\partial_s\Theta\), and \(\partial_s Z_i\), up to fixed
nonzero normalizations.  The TT row is the incoming normal-wave derivative;
its homogeneous value is the \(L=1\) maximally dissipative condition
compatible with zero residual \(\Psi_0\).  These identities and their
executable SymPy assertions are included in the derivation.

The kernel evaluates incoming characteristic amplitudes from residual fields
and inward second-order normal derivatives.  It computes their complete
volume/source rate from the already assembled residual RHS, then corrects
only the momentum-like RHS fields
`(Khat, Theta, A_ij, Gamma^i)`.  The current candidate imposes
time-independent characteristic data,

```text
d_t w_in = 0.
```

Thus initially homogeneous data remain exactly homogeneous.  Unlike the
rejected cell-scale relaxation `d_t w_in=-lambda_in*w_in/h`, this condition
does not erase a stationary residual near field merely because its principal
decomposition has nonzero incoming and outgoing pieces.  This retains the
implemented nonlinear geometry, damping, source, and KO terms already present
in the volume RHS.  It does not hard-zero a field or a ghost cell.

The condition has complete incoming-sector coverage for the implemented
frozen local normal system, but it is not the auxiliary-field \(L=4\)
absorption hierarchy and is not claimed to be a separately derived full
nonlinear \(\Psi_0\)/constraint hierarchy.  Projector time derivatives and
tangential principal terms are not added as independent lower-order boundary
data.  The local frame and coefficients are frozen during each compatibility
operation.  Matter must remain absent from the boundary.  These limitations,
the ghost closure, and all sign conventions are stated in
`docs/z4c/residual_characteristic_cpbc.tex`.

## Source and build provenance

- Base commit: `4dacc5b98b078e09fd76e9f70ae7d9ba662986d5`
  (`origin/project/bg_subtract` when the clean worktree was created).
- Clean worktree:
  `/flare/MHDTidal/hzhu/tde_n3_validation/worktrees/athenak_characteristic_cpbc`
- Branch: `codex/z4c-characteristic-cpbc`
- Current staged implementation/test diff SHA-256 (excluding this
  self-referential results report):
  `afffe8b469a50d4d8249cd5663def40666bbd390f436ec28191df8414f12fde5`
- Production-pgen build:
  `/flare/MHDTidal/hzhu/tde_n3_validation/build/aurora-intel-gpu-characteristic-cpbc`
- Current zero-rate production-pgen executable SHA-256:
  `cd7cefef042ff075e85688dd3ec08dca243df479d50b91e0439dd8a20da46478`
- Archived rejected cell-relaxation executable SHA-256:
  `bca5f8e0837c6a85d36539d27baca6e08be913a5a21350a8164f904084cbd038`
- Built-in-pgens regression build:
  `/flare/MHDTidal/hzhu/tde_n3_validation/build/aurora-intel-gpu-builtins-characteristic-cpbc`
- Built-in-pgens regression executable SHA-256:
  `a32232b1c24cd1702c21fc2f244a8ab082c68f859b8016732487da0da482a096`
- Archived built-in-pgens executable:
  `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/executables/athena_builtins_cpbc_final_a32232b1c24cd170`
- Configuration: `mpic++`/icpx 2025.3.2, Release, MPI, SYCL, Intel PVC,
  Level Zero, double precision, and `-fp-model=precise`.
- Build command: `make -j64`.
- Production build log:
  `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/build_logs/cpbc_zero_rate_make_j64_20260730.log`
- Built-in-pgens build log:
  `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/build_logs/cpbc_builtins_final_make_j64_20260730.log`
- Final built-in-pgens relink log:
  `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/build_logs/cpbc_builtins_relink_make_j64_20260730.log`

The final `make -j64` relink completed successfully at 10:18 UTC on
2026-07-30.  It reproduced the executable SHA-256 above exactly; `cmp`
also verified that the build-tree executable and archived executable are
byte-identical.

The three fused CPBC orientation kernels in the current zero-rate build
compile at SIMD32 with 256 registers and approximately 4, 10, and 6 spill
slots.  (The three adjacent 6, 14, and 18-slot kernels are the retained
Sommerfeld path.)  The NGHOST=4 volume RHS kernel spills about 177 slots and
several existing production kernels spill more.
This static comparison is not the performance gate; the required measured
kernel fraction remains pending.

The CPBC adds no field-sized global temporary.  The production allocation is
11 diagnostic `Real` values plus `11 + 3*max_nmb_per_rank` integer
counter/list entries.  With `max_nmb_per_rank=16`, that is 88 + 236 = 324
bytes of device storage per rank (and 236 bytes for the integer host mirror),
apart from a few host scalar timers and counts.  The pre-existing `u_bg` and
`u_full` field arrays are present in base commit `4dacc5b9` and are not CPBC
memory overhead.

## Compact boundary-MeshBlock launch guard

The production performance optimization scans the authoritative host
boundary flags and maintains three compact, orientation-specific device
lists.  Interior MeshBlocks no longer enter a fused CPBC launch.  A
MeshBlock may appear in multiple orientation lists, but the existing
per-cell ownership rules remain disjoint.

Aurora job `8716575` compared the then-current cell-relaxation executable
(`bca5f8e...038`) with
the archived pre-compaction executable (`b6680435...463a`) using a
nonzero scalar-constraint pulse on four x-directed MeshBlocks.  It exited
zero and reported:

```text
boundary_blocks_max=2,0,0
CPBC compact launch is byte-identical across 3 outputs
PASS cases=103 algebra_error=1.11599467e-13
frame_error=1.88737914e-15
```

All three history files are byte-identical.  The same job reran the
Minkowski, Schwarzschild, and spinning Kerr exact-background suite with that
executable and reproduced the accepted roundoff-level results below.
Thus the earlier pulse, reflection, edge/corner, and far-reference evidence
transfers exactly across the launch compaction; measured performance remains
a separate gate.

## Algebra and derivation

The exact SymPy derivation checks the scalar, vector, and tensor
characteristic polynomials, all left and right eigenpairs, `LR-I`,
incoming-projector idempotence, outgoing annihilation, and the scalar
compatibility-map determinant.  The dependency-light numerical check covers
103 admissible randomized and physical states, including Minkowski,
Schwarzschild, spinning Kerr--Schild, finite residual shifts, and composite
corner normals.

Accepted Aurora result:

```text
PASS cases=103 algebra_error=6.30819793e-14
frame_error=1.88737914e-15
```

Both errors are below the required `1e-12`.  The eleven-page LaTeX derivation
compiles with `pdflatex`.

## Exact analytic backgrounds

Aurora job `8717175` repeated the causally connected exact-background suite
with the current zero-rate executable (SHA-256 `cd7cef...478`) and exited
zero.  The Schwarzschild and Kerr cases use a spin-aware interior excision
mask entirely inside the horizon; earlier unexcised/coarsely excised harness
attempts are retained as failed-test provenance, not accepted data.

| Background | max residual | max incoming | enforcement |
|---|---:|---:|---:|
| Minkowski | `0` | `0` | `0` |
| Schwarzschild | `8.21565038e-15` | `1.57202000e-16` | `2.95822800e-31` |
| Kerr--Schild, `a=0.9` | `8.43769499e-15` | `1.58043600e-16` | `2.71170900e-31` |

There is no analytic-background extrapolation signature.

## Normal-incidence pulse convergence

The measurement is independent of the boundary compatibility diagnostic.
Each run writes `variable=z4c_residual` directly, projects an interior
incoming characteristic field, and subtracts a causally disconnected
same-spacing far-boundary control.  Initial incoming contamination is required
to be below 0.5% of the outgoing norm.  All accepted cases are finite.

For each Cartesian orientation, all ten families and both signs were run at
spacings `h=0.25`, `0.125`, and `0.0625 M`.  Jobs:

- axis 1: `8715426`, `8715449`, `8715490`;
- axis 2: `8715561`, `8715581`, `8715606`;
- axis 3: `8715686`, `8715715`, `8715741`.

All 60 orientation/sign/family convergence cases pass.  The minimum observed
order is `2.040227`; the maximum fine-grid reflection is
`5.11409524e-3`.  Values below summarize one sign of axis 1; the opposite
sign and other orientations agree in the shown digits.  The raw convergence
logs retain every individual value.

| Family | coarse | medium | fine | min order |
|---|---:|---:|---:|---:|
| lapse | `8.81683e-2` | `2.10179e-2` | `5.10998e-3` | `2.04023` |
| longitudinal shift | `8.12860e-2` | `1.26079e-2` | `1.56038e-3` | `2.68867` |
| transverse shift 1 | `8.86502e-2` | `2.11485e-2` | `5.11409e-3` | `2.04801` |
| transverse shift 2 | `8.86502e-2` | `2.11485e-2` | `5.11409e-3` | `2.04801` |
| scalar constraint Theta | `8.63315e-2` | `1.30301e-2` | `1.65557e-3` | `2.72804` |
| scalar constraint Z | `8.86501e-2` | `2.11485e-2` | `5.11407e-3` | `2.04801` |
| transverse constraint 1 | `8.86502e-2` | `2.11485e-2` | `5.11409e-3` | `2.04801` |
| transverse constraint 2 | `8.86502e-2` | `2.11485e-2` | `5.11409e-3` | `2.04801` |
| TT plus | `8.86501e-2` | `2.11485e-2` | `5.11408e-3` | `2.04801` |
| TT cross | `8.86502e-2` | `2.11485e-2` | `5.11409e-3` | `2.04801` |

## Matched Sommerfeld comparison

Jobs `8715539` (axis 1) and `8715662` (axis 2) use identical initial data,
grid, timestep, output, extrapolation order, pulse amplitude, and analysis
window.  Only `<z4c>/boundary_rhs` changes.  Axis 3 was not redundantly
repeated after the three-axis CPBC convergence suite established rotational
symmetry.

| Family | CPBC | Sommerfeld | Sommerfeld/CPBC |
|---|---:|---:|---:|
| lapse | `5.10998e-3` | `4.22846e-2` | `8.2749` |
| longitudinal shift | `1.56038e-3` | `7.72353e-2` | `49.4976` |
| transverse shift 1 | `5.11409e-3` | `4.20322e-2` | `8.2189` |
| transverse shift 2 | `5.11409e-3` | `4.20322e-2` | `8.2189` |
| scalar constraint Theta | `1.65557e-3` | `4.59912e-2` | `27.7796` |
| scalar constraint Z | `5.11407e-3` | `4.20317e-2` | `8.2188` |
| transverse constraint 1 | `5.11409e-3` | `4.20317e-2` | `8.2188` |
| transverse constraint 2 | `5.11409e-3` | `4.20317e-2` | `8.2188` |
| TT plus | `5.11408e-3` | `4.20318e-2` | `8.2188` |
| TT cross | `5.11409e-3` | `4.20317e-2` | `8.2188` |

Aggregate axis-1 result:

```text
cases=20
cpbc_median=5.11409044e-3
sommerfeld_median=4.20319606e-2
median_improvement=8.218854
cpbc_worst=5.11409377e-3
sommerfeld_worst=7.72352881e-2
worst_improvement=15.102439
```

No matched family is worse under CPBC; the largest CPBC/Sommerfeld ratio is
`0.121672`.

## Edges and corners

The three orientation kernels have disjoint ownership.  At a physical edge or
corner one owner constructs a normalized composite normal from every incident
face, so no two device kernels write the same active cell.  Residual ghost
fills remain AthenaK's ordered tensor-product polynomial extrapolation.

Representative `tt_cross` oblique tests were selected after the full
orientation/sign normal suite established symmetry:

| Geometry | Job | response/control-subtracted outgoing | repeat checksum | bytes |
|---|---|---:|---:|---:|
| 2-D edge | `8715843` | `8.49418169e-2` | `1810358521` | `26229708` |
| 3-D corner | `8715855` | `8.75713133e-2` | `815485936` | `26229708` |

Both repeated outputs are bit-identical and finite.  The approximately 8.5%
value is an oblique-incidence response of a planar \(L=1\) face condition; it
is not substituted for the normal-incidence reflection gate and is not
expected to converge to zero at finite incidence angle.

Job `8716102` reanalyzed the untouched coarse, medium, and fine binaries
using a fixed physical exclusion width of \(8/3\,M\), corresponding to 4, 8,
and 16 cells.  This keeps the sampled physical region invariant under
refinement.  The earlier fixed-cell analysis changed the sampled region with
resolution and is superseded.

| Geometry | coarse | medium | fine | order | extrapolated limit |
|---|---:|---:|---:|---:|---:|
| 2-D edge | `6.68401749e-2` | `7.27377909e-2` | `7.39983217e-2` | `2.22610052` | `7.43409805e-2` |
| 3-D corner | `7.24770055e-2` | `7.50108649e-2` | `7.54861525e-2` | `2.41446381` | `7.55958879e-2` |

The finite-angle response converges at greater than second order to a
nonzero planar-\(L=1\) limit, while initial incoming contamination drops from
approximately `2e-4` on the coarse mesh to `2e-7` on the fine mesh.  Thus the
ordered tensor-product ghost closure does not reduce the measured edge or
corner convergence order.  No claim of exact multidimensional absorption at
a nonsmooth boundary point is made.

## Input validation and mesh preflights

- Job `8715873` confirms that an unsupported gauge is rejected at startup
  rather than silently falling back.
- Tight causal-boundary mesh job `8715880` contains 3424 MeshBlocks on
  physical levels 1--6 with counts `416, 656, 784, 848, 336, 384`.
- Production mesh job `8715912` passes with:

```text
Root grid = 2 x 2 x 2 MeshBlocks
Total number of MeshBlocks = 1184
Physical levels 1--13:
52, 84, 84, 84, 80, 108, 140, 148, 84, 84, 84, 88, 64
```

At 32 nodes/384 ranks this is 3--4 MeshBlocks per rank, below
`max_nmb_per_rank=16`.

## Legacy-regression audit

The current patch leaves the default `boundary_rhs=sommerfeld`, and its new
volume-RHS timer is inactive unless CPBC is selected.  Nevertheless, strict
regressions are being measured rather than inferred.

The first 12-rank 3-D AMR wave job (`8715925`) is finite and second-order
convergent:

```text
second_r32=1.23243500e-10
second_r64=2.56704800e-11
ratio=2.08290741e-1
sixth_r64=8.21939900e-12
```

The sixth-order value exceeds the upstream single-GPU threshold `6e-12`.
The first 12-rank and official-mode 1-rank boosted runs (`8715944`,
`8715950`) also reproduce four small upstream-limit misses:

```text
Mx-norm2=1.15623000e-3
My-norm2=6.55857000e-4
Mz-norm2=6.55857000e-4
Theta-norm=3.17222000e-5
```

These misses are not hidden or reclassified as upstream-threshold passes.  To
test the actual regression requirement ("default path unchanged"), an
untouched `4dacc5b9` executable was built with identical compiler flags
(SHA-256
`c7159aa03cea403e1bf3ee00d92f13086457a10f3871d38ea65df41234ab2fa0`).

Untouched-base boosted job `8715961` reproduces the same values.  Its complete
history file and the patched job `8715950` history are byte-identical:

```text
SHA-256 9b759837a8e5dd92c0812a24c49e4a8de4c99c760ce9b1e0be8100f38d8be8f1
```

Untouched-base sixth-order wave job `8715970` reproduces
`RMS-L1=8.219399e-12`.  Its error file and the patched job `8715925` error
file are also byte-identical:

```text
SHA-256 2ede4e9c4503197bba4597724774db25b97ef137a78f7dba9cebe58169f9beed
```

Thus this Aurora compiler/base branch does not meet two absolute upstream
reference tolerances, but the CPBC patch introduces no change whatsoever in
the tested default-periodic/Sommerfeld data.  The second-order AMR/MPI wave
convergence gate passes, and exact baseline parity proves the requested
backward compatibility.

## Radial-sponge isolation

Matched one-node, 12-rank Minkowski runs used the production radial profile
(`512--640M`, `tau=16M`), CPBC, and an exterior spherical Theta pulse:

- undamped control: job `8716020`;
- radial sponge: job `8716030`.

Both reached `t=24M` with exit status zero, finite histories,
`bad-metric=0`, and no causal interaction with the physical boundary.  This
coarse isolation deck sets `shift_eta=0`: its `dt=2.4M` would otherwise put
the unrelated local term `-2 beta` at `z=-4.8`, outside RK3's stability
interval.

At `t=24M`, the expected full-strength attenuation is
`exp(-24/16)=2.23130160e-1`:

| Diagnostic | Control | Sponge | Sponge/control |
| --- | ---: | ---: | ---: |
| `Theta-max` | `7.55407588e-9` | `1.68523739e-9` | `2.23089815e-1` |
| `beta-res` | `2.94269284e-8` | `6.56277645e-9` | `2.23019418e-1` |
| `Gam-res` | `2.03227268e-9` | `4.53737941e-10` | `2.23266270e-1` |
| `res-ramp` | `7.21747788e-8` | `1.61301463e-8` | `2.23487298e-1` |
| `res-outer` | `2.84133795e-7` | `6.34360475e-8` | `2.23261184e-1` |

The largest attenuation-curve error over every positive-time sample is
`1.2441%`, below the predeclared `2%` gate.  The maximum `res-inner`
control/sponge difference is `2.66454e-15`, confirming no direct interior
source.  The reproducible checker is
`analysis/z4c_characteristic/check_radial_sponge.py`.

Earlier evolved-gauge jobs had appeared to show a factor-2.9 sponge
amplification.  Their `shift_eta=2`, `dt=2.4M` combination gives RK3
amplification factors `10.712` without the sponge and `11.9133125` with
`sigma=1/16`; the predicted ten-step ratio is `2.89481`, matching the
observed `2.897`.  The corrected stable A/B demonstrates that this was
timestep stiffness in the test harness, not a sponge sign or coupling defect.
The production AMR timestep is approximately `5.86e-4M`.

## Far-boundary TDE reference

Aurora job `8716217` ran the causally disconnected far-boundary reference on
32 nodes and 384 ranks with the pre-compaction executable
`b6680435...463a`.  It exited zero after 32 minutes, reached `t=13.5M` in
3840 cycles, and wrote 28 direct-residual slices spanning `t=0`--`13.5M`.
The root layout was `5 x 4 x 4`; the AMR mesh contained 3664 MeshBlocks, with
9--10 blocks per rank and maximum normalized load `1.111`.

The star remained finite and tracked normally; at `t=13.5M` its x coordinate
was approximately `37.986M` and the maximum density was
`1.24833e-4`.  No MPI, SYCL, nonfinite, or invalid-characteristic diagnostic
was reported.  The comparator now reads binary residual slices from the
actual production `bin/` subdirectory and has been verified to discover all
28 files.  It obtains `R_star=0.47106M` from Athena's initialized isotropic
TOV radius, making the `0.1 R_star` trajectory gate `0.047106M`; it no longer
uses the larger `0.71M` AMR-refinement radius as a proxy.  It also rejects a
pointwise far/near subtraction if the local leaf AMR levels differ.  A
five-slice `t>=11.5M` far-reference self-subtraction sampled 9984 fixed
physical points per slice and returned exactly zero in every gauge and
constraint field.

Job `8716575` proves that launch compaction itself is byte-identical on a
nonzero CPBC evolution.  The later incoming-data correction does not
invalidate the far reference: no physical face is causally connected to the
sampled region by `t=13.5M`, and the CPBC is the only changed evolution
operation.

## Small-boundary TDE A/B

The near Sommerfeld control, job `8716635`, used the same 32-node/384-rank
spacing and initial data as the far reference, changing the `+x` boundary
from `72M` to `48M`.  It used executable SHA-256 `bca5f8e...038`, exited zero
in 23:26, reached `t=13.5M` in 3840 cycles, and wrote all 28 direct-residual
slices.  The startup mesh had 3424 MeshBlocks on levels 1--6; dynamic AMR
finished with 1352 blocks.  All three histories and 3840 valid star-track
rows are finite, `bad-metric=0`, and no MPI, SYCL, Level Zero, or
characteristic error was reported.

The fixed-point, matched-AMR-level residual-component subtraction against the
far reference uses five slices at `t>=11.5M` and measures the causal returning
signal:

| Sommerfeld return diagnostic | RMS | Linf |
|---|---:|---:|
| gauge group | `2.25194400e-8` | `1.45512860e-7` |
| constraint group | `2.11197068e-7` | `1.30728178e-6` |

Both signals are well above the predeclared `1e-12` observability floor.
The secondary far-reference differences are
`density_fraction=7.09497471e-9`, zero sampled trajectory difference, and
maximum selected global-history difference `1.37737390e-6`.

The first no-sponge CPBC case, job `8716873`, used the rejected
cell-relaxation target.  It exited zero in `27:10`, reached `t=13.5M` in 3840
cycles, wrote all 28 residual slices, and retained 3840 finite/valid
star-track rows.  Its raw residual-component result was only `0.804414x` in
the gauge group and `1.923341x` in the constraint group relative to
Sommerfeld, so the predeclared tenfold gate failed.

The corrected independent observable reconstructs the local full conformal
metric and analytic Kerr--Schild lapse/shift, builds the exact `+x`
face-normal frame, and projects all four incoming gauge and four incoming
constraint characteristics on matched far/near leaf cells.  It excludes two
cells at every MeshBlock edge rather than differentiating across block
boundaries.  A closed-form finite-residual synthetic check covers all eight
modes with maximum error `8.32667268e-17`; the Schwarzschild background
reconstruction error is zero.  There are at least 55,888 matched interior
cells in each of the five accepted time slices.

This stronger projection confirms the cell-relaxation failure:

| Incoming mode/group | Sommerfeld RMS | CPBC RMS | improvement |
|---|---:|---:|---:|
| gauge group | `5.40074273e-8` | `2.97327683e-8` | `1.816428` |
| constraint group | `3.83203235e-7` | `1.60210080e-7` | `2.391880` |
| lapse | `2.03003289e-8` | `9.12096759e-9` | `2.225677` |
| longitudinal shift | `2.89344877e-8` | `1.59434570e-8` | `1.814819` |
| transverse shift 1 | `5.40074273e-8` | `2.97327683e-8` | `1.816428` |
| transverse shift 2 | `7.89011893e-10` | `4.24936323e-10` | `1.856777` |
| scalar constraint Theta | `1.45792576e-7` | `7.64030518e-8` | `1.908204` |
| scalar constraint Z | `3.83203235e-7` | `1.60210080e-7` | `2.391880` |
| transverse constraint 1 | `9.03236601e-8` | `1.04957743e-7` | `0.860572` |
| transverse constraint 2 | `1.32001284e-9` | `1.50890267e-9` | `0.874816` |

At initialization, the star's stationary residual near field gives principal
incoming gauge/constraint amplitudes `1.95e-8` and `3.10e-8`, comparable to
their outgoing amplitudes.  The rejected `-lambda*w/h` target immediately
erased that legitimate time-independent content and launched a transient.

The zero-rate candidate was therefore tested first on all ten families and
both signs of axis 1 in job `8717147`.  It exited zero; all 20 independently
measured reflection ratios remain between `1.56037501e-3` and
`5.11409417e-3`.  The largest absolute change from the previously accepted
fine-grid values is `1.236e-8`.

Exact-background rerun `8717175` then exited zero with the same executable.
Minkowski remained identically zero.  Schwarzschild and \(a=0.9\) Kerr--Schild
had maximum residuals `8.21565038e-15` and `8.43769499e-15`, respectively;
their maximum incoming amplitudes were `1.57202000e-16` and
`1.58043600e-16`.  All are below the declared `1e-12` gate.

The no-sponge small-boundary TDE rerun was debug-scaling job `8717189`.
It used 32 nodes and the unchanged `cd7cef...478` executable, exited zero
after `31:49`, and completed all requested outputs through `t=13.5M`.
Against the same far reference, the independently projected incoming
characteristics were

| Incoming group | Sommerfeld RMS | Zero-rate CPBC RMS | improvement |
|---|---:|---:|---:|
| gauge | `5.40074273e-8` | `2.97278654e-8` | `1.816727` |
| constraint | `3.83203235e-7` | `1.60199933e-7` | `2.392031` |

The lapse and longitudinal-shift improvements were `2.385002` and
`1.957105`; transverse shift improvements were `1.816727` and `1.857332`.
The scalar-Theta and scalar-Z improvements were `1.932452` and `2.392031`.
The first transverse constraint family was worse by about 16% (improvement
`0.860530`).  The density fractions relative to the far case were
`7.09497471e-9` for Sommerfeld and `7.68416011e-9` for CPBC, and both
sampled trajectory differences were zero.  Thus this run is finite and
matched, but it fails the tenfold reflection gate and the per-family
non-regression gate.  More importantly, neither treatment produced a
material stellar artifact by `t=13.5M`, so this `+x=48M` case is not yet the
required unstable TDE baseline.

## Thin-z reproducer checkpoint

Historical `bcdebug_*` runs localize the strongest growth at a physical
z face in domains with `z=+-12.8M`, but those runs used the old default
face sponge and therefore are geometry evidence rather than admissible
no-sponge controls.  A cheaper one-close-face input was consequently added:

```text
x = [-48, 72], nx1 = 160
y = [-48, 48], nx2 = 128
z = [-6, 42],   nx3 = 64
root dx = 0.75
star center = (40, 0, 0)
tlim = 30
outer_sponge_enabled = false
boundary_rhs = sommerfeld
```

The lower-z face is more than `5.5M` of vacuum from the initial stellar
surface and is the only face expected to return a fastest-gauge signal
before the analysis interval.  Mesh-only job `8717946` exited zero and
reported a `5 x 4 x 2` root-block grid, 2756 total MeshBlocks, and this
physical-level histogram:

```text
level 1: 264
level 2: 364
level 3: 560
level 4: 848
level 5: 336
level 6: 384
```

The failed predecessor `8717932` was only a one-rank mesh-only capacity
limit (`max_nmb_per_rank=192` versus 2756); the corrected preflight used
4096 and did not change the evolution deck.  No thin-z evolution has yet
been submitted.  The next validation action is the immutable-executable
Sommerfeld control, followed by the matching zero-rate CPBC run only if the
control reproducibly shows a causally timed material boundary artifact.

The TDE comparator now reads x-y, x-z, or y-z residual slices and supports
boundary axes 1--3.  Its rotated synthetic x-z check passes for the existing
eight gauge/constraint projections.  The two independent TT/radiative
projections are not yet implemented in this comparator, so no future TDE
result is complete until those modes and their synthetic expectations are
added.

The current experimental tangential-principal implementation built
successfully with `make -j64`; its executable SHA-256 is
`483325ce59992648fc8efcb92fcc9f3bd437c35d01faad0291cf89a1c31e4b2e`.
This executable has not passed the pulse, exact-background, corner,
reflection, TDE, or performance gates and must not be described as a
validated CPBC.

## Pending acceptance gates

1. Reproduce a material, causally attributable Sommerfeld/extrapolation
   artifact in the smallest clean no-sponge TDE domain.
2. Run the immutable old Sommerfeld/zero-rate-CPBC A/B on that selected
   domain and report all ten independently projected incoming families.
3. Audit and validate the multidimensional normal/tangential principal
   split before using the experimental executable for a TDE claim.
4. Demonstrate at least tenfold returning gauge/constraint reduction, no
   family more than 10% worse than Sommerfeld, central-density difference
   below 1%, and trajectory difference below `0.1 R_star`.
5. Pass all controlled pulse reflections below 2%, exact backgrounds below
   `1e-12`, corner determinism, and second-order convergence.
6. Measure CPBC kernel cost below 3% of the Z4c volume RHS.
7. Complete the 32-node production-input smoke with the `512--640M`,
   `tau=16M` sponge and the final candidate.
8. Rerun default-path parity regressions with the final executable, then
   perform the final source/report audit before any release push.

## Failed or superseded harness attempts

Failed/cancelled jobs are retained in the run tree and will be listed in the
final report.  The important classes are:

- unexcised or spin-insensitive pure-background masks;
- full-state single-precision output that destroyed small residuals by
cancellation;
- missing runtime parameters in early harnesses;
- pulse measurements taken before a causal boundary return;
- the stable but insufficiently absorbing cell-relaxation TDE job `8716873`;
- an exploratory demand that finite-angle oblique reflection converge to
  zero;
- cancelled 8/16/32-node TDE sizing jobs used only to measure throughput or
  diagnose the active Aurora reservation.

None of these results is used as acceptance evidence.
