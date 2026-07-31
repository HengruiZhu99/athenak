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
- Resumed WIP checkpoint and current `HEAD`:
  `2325fc2de6efff07f9ddff846b19735243fb0a3b`.
- Clean worktree:
  `/flare/MHDTidal/hzhu/tde_n3_validation/worktrees/athenak_characteristic_cpbc`
- Branch: `codex/z4c-characteristic-cpbc`
- Exact recovered validated zero-rate source snapshot
  (`src/z4c/z4c_Sbc.cpp`) SHA-256:
  `5abb0ef3c22c8f989b5916d277676b0a2a7ef993540edfe917c3e74d1b76a2c8`
- Git blob for that recovered source snapshot:
  `437fe5be205f8a2130f85f68ec75f3500d50cb1b`
- Current uncommitted harness/test patch SHA-256 (including the new
  single-run analyzer and excluding this self-referential results report):
  `e428cb97322d0b47e37085c588a47b466f3d09541fc5ea85b8db2090c6ee8993`
- Production-pgen build:
  `/flare/MHDTidal/hzhu/tde_n3_validation/build/aurora-intel-gpu-characteristic-cpbc`
- Archived validated zero-rate production-pgen executable SHA-256:
  `cd7cefef042ff075e85688dd3ec08dca243df479d50b91e0439dd8a20da46478`
- Clean tangential-principal candidate SHA-256:
  `298646be8918a7778329510eefb617cf9458ffe11056688c4f5f881434f9999e`
- Immutable tangential-principal candidate:
  `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/executables/athena_cpbc_tangential_principal_298646be8918a777`
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
- Clean tangential-principal build log:
  `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/build_logs/cpbc_tangential_principal_clean_make_j64_20260730.log`
- Built-in-pgens build log:
  `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/build_logs/cpbc_builtins_final_make_j64_20260730.log`
- Final built-in-pgens relink log:
  `/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/build_logs/cpbc_builtins_relink_make_j64_20260730.log`

The final `make -j64` relink completed successfully at 10:18 UTC on
2026-07-30.  It reproduced the executable SHA-256 above exactly; `cmp`
also verified that the build-tree executable and archived executable are
byte-identical.

The recovered source hash and Git blob above identify the exact zero-rate
boundary source used to build the validated archived executable; they are
recorded separately from the later tangential-principal source.

### Runtime boundary and source selection

`boundary_rhs` now defaults to `characteristic_cpbc`.  The generic
`boundary_rhs=characteristic_cpbc` path retains the validated zero-rate
behavior.  Its compatibility target is selected separately through
`characteristic_bc_source`:

- `zero_rate` is the default and imposes the tested
  `d_t w_in = 0` target;
- `tangential_principal` retains the separate experimental principal-source
  target and must be requested explicitly.

The tangential-principal source has not been validated in a TDE run.  This
runtime split and default change do not themselves constitute a new
validation run, and none of the scientific comparisons below are attributed
to the tangential-principal source.

The source split was checked locally in a clean Release/Serial
`PROBLEM=z4c_tov_ks` build on 2026-07-30.  Both explicit source modes passed
the Minkowski, Schwarzschild, and `a=0.9` Kerr--Schild exact-background
smokes.  The maximum reported residual was `7.99360578e-15` for zero-rate and
`2.22044605e-14` for tangential-principal; all enforcement errors were below
`3.3e-31`.  A source-free input selected
`boundary_rhs=characteristic_cpbc` with `source=zero_rate`, an invalid source
name was rejected, the numeric characteristic check passed all 103 cases
(`algebra_error=7.51026332e-14`, `frame_error=1.66533454e-15`), and an
explicit legacy Sommerfeld mesh smoke completed.  These are build/parser and
exact-background checks, not a replacement for an Aurora GPU or TDE
validation of the new source-selection code.

The three fused CPBC orientation kernels in the archived validated zero-rate
build compile at SIMD32 with 256 registers and approximately 4, 10, and 6
spill slots.  (The three adjacent 6, 14, and 18-slot kernels are the retained
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

The historical regression audit below used an earlier candidate for which
`boundary_rhs` defaulted to `sommerfeld`; its volume-RHS timer was inactive
unless CPBC was selected.  The current runtime default is
`boundary_rhs=characteristic_cpbc`, selecting the validated zero-rate source.
The historical numbers below have not been rerun for that new default; they
establish byte-identical parity for the explicitly selected Sommerfeld path.

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
isolate whether the patch changed the explicitly selected Sommerfeld path, an
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
the tested explicit-periodic/Sommerfeld data.  The second-order AMR/MPI wave
convergence gate passes, and exact baseline parity proves the requested
Sommerfeld-path backward compatibility.  This statement is not a validation
of the newly selected zero-rate default.

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
been completed.  The immutable-executable Sommerfeld control was submitted
to `debug-scaling` as 32-node job `8718120` and was initially queued.  The
exact command was

```text
qsub -N z4c_somm_zm6 -l select=32 \
  -v REPO_DIR=/flare/MHDTidal/hzhu/tde_n3_validation/worktrees/athenak_characteristic_cpbc,BUILD_DIR=/flare/MHDTidal/hzhu/tde_n3_validation/build/aurora-intel-gpu-characteristic-cpbc,ATHENA_EXE=/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/executables/athena_cpbc_zero_rate_cd7cefef042ff075,CASE_NAME=zminus6_sommerfeld_archived_20260730,VALIDATION_KIND=athena,INPUT_DECK=/flare/MHDTidal/hzhu/tde_n3_validation/worktrees/athenak_characteristic_cpbc/inputs/tde/aurora/z4c_tov_ks_n3_schwarzschild_bgadapt_cpbc_zminus6_aurora.athinput,RANKS_PER_NODE=12,ATHENA_WALLTIME=00:53:00,ATHENA_EXTRA_ARGS=z4c/boundary_rhs=sommerfeld:z4c/extrap_order=4:problem/outer_sponge_enabled=false \
  analysis/z4c_characteristic/aurora/submit_z4c_cpbc_validation.pbs
```

Immediately before submission the immutable executable SHA-256 was verified
as `cd7cefef042ff075e85688dd3ec08dca243df479d50b91e0439dd8a20da46478`.
Job `8718120` waited 48 minutes, started at `2026-07-30 15:39:53 UTC`, and
verified the requested 32-node/384-rank mesh, exact executable hash, source
`HEAD`, order 4, and disabled sponge.  Aurora then raised a device page fault
immediately after cycle 0:

```text
Segmentation fault from GPU ... type: 0 (NotPresent) ... access: 1 (Write),
banned: 1, aborting.
... rank 362 died from signal 6
```

PBS recorded `Exit_status=143` and seven seconds of wall time.  Only the
initial history rows and `00000` slices exist, so this is a preserved
infrastructure/device failure and supplies no Sommerfeld physics evidence.
Its run directory is
`/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/runs/characteristic_cpbc/zminus6_sommerfeld_archived_20260730`.
The scheduler-joined output was copied into that directory as
`pbs_output.o8718120`; its SHA-256 is
`829688e62223ef4c05c65ed4b6ad4f619b19ec4487f34660806c438a33cd86bf`.

After the failed job reached terminal state, the identical sequential retry
was submitted as debug-scaling job `8718377` with a fresh run directory:

```text
qsub -N z4c_somm_zm6r1 -l select=32 \
  -v REPO_DIR=/flare/MHDTidal/hzhu/tde_n3_validation/worktrees/athenak_characteristic_cpbc,BUILD_DIR=/flare/MHDTidal/hzhu/tde_n3_validation/build/aurora-intel-gpu-characteristic-cpbc,ATHENA_EXE=/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/executables/athena_cpbc_zero_rate_cd7cefef042ff075,CASE_NAME=zminus6_sommerfeld_archived_retry1_20260730,VALIDATION_KIND=athena,INPUT_DECK=/flare/MHDTidal/hzhu/tde_n3_validation/worktrees/athenak_characteristic_cpbc/inputs/tde/aurora/z4c_tov_ks_n3_schwarzschild_bgadapt_cpbc_zminus6_aurora.athinput,RANKS_PER_NODE=12,ATHENA_WALLTIME=00:53:00,ATHENA_EXTRA_ARGS=z4c/boundary_rhs=sommerfeld:z4c/extrap_order=4:problem/outer_sponge_enabled=false \
  analysis/z4c_characteristic/aurora/submit_z4c_cpbc_validation.pbs
```

The retry ID is
`8718377.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`; it was initially
queued.  Job `8718120` was not cancelled or resized.

### Reproduced archived-executable Sommerfeld failure

Retry `8718377` started at `2026-07-30 15:55:27 UTC` and reproduced a
catastrophic, lower-z-localized Sommerfeld failure.  The finite onset is
visible in the independently located Hamiltonian maximum:

| time | H maximum | x | y | z |
|---:|---:|---:|---:|---:|
| `7.00313` | `31.8178` | `-0.0703125` | `0.445312` | `-5.97656` |
| `7.05234` | `39.3038` | `-0.0703125` | `0.445312` | `-5.97656` |
| `7.10156` | `44.4944` | `2.17969` | `-0.0703125` | `-5.97656` |
| `7.15078` | `69.0354` | `2.03906` | `0.0703125` | `-5.97656` |

Thus the growing maximum sits in the first cell layer at the only close
physical face.  The problem history is finite through `t=7.15078125`; the
first nonfinite history row is `t=7.2`, where the direct diagnostic also
reports `H=inf`.  STAR_TRACK first reaches the atmosphere floor at
approximately `t=7.270313`.  This destructive onset is close to the
predeclared `8--9M` fastest-gauge round-trip estimate and is much earlier
than any return from the other five faces.  It is therefore the required
material, causally timed boundary artifact rather than merely a measurable
reflected pulse.

The run produced at least 29 finite pre-failure x-z residual slices and more
than 100 total x-z slices.  After the failure was fully established, the
user explicitly authorized early termination rather than spending the
remaining allocation evolving nonfinite data.  `qdel 8718377` was issued;
PBS recorded `Exit_status=271` after approximately 53 minutes.  The run
directory and its histories, slices, copied input, metadata, and
`athena.stdout` are preserved at
`/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/runs/characteristic_cpbc/zminus6_sommerfeld_archived_retry1_20260730`.

The Sommerfeld-first gate is now open.  The previously proven all-ten pulse
suite for the same archived zero-rate executable is job `8717147`, so the
user directed the immediate same-executable/same-domain CPBC comparison.
An unstarted one-node pulse job for the newer experimental executable,
`8718570`, was cancelled while still queued and created no run directory.
The archived CPBC TDE comparison was then submitted as job `8718576`:

```text
qsub -N z4c_cpbc_zm6 -l select=32 \
  -v REPO_DIR=/flare/MHDTidal/hzhu/tde_n3_validation/worktrees/athenak_characteristic_cpbc,BUILD_DIR=/flare/MHDTidal/hzhu/tde_n3_validation/build/aurora-intel-gpu-characteristic-cpbc,ATHENA_EXE=/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/executables/athena_cpbc_zero_rate_cd7cefef042ff075,CASE_NAME=zminus6_cpbc_archived_20260730,VALIDATION_KIND=athena,INPUT_DECK=/flare/MHDTidal/hzhu/tde_n3_validation/worktrees/athenak_characteristic_cpbc/inputs/tde/aurora/z4c_tov_ks_n3_schwarzschild_bgadapt_cpbc_zminus6_aurora.athinput,RANKS_PER_NODE=12,ATHENA_WALLTIME=00:53:00,ATHENA_EXTRA_ARGS=z4c/boundary_rhs=characteristic_cpbc:z4c/extrap_order=4:problem/outer_sponge_enabled=false \
  analysis/z4c_characteristic/aurora/submit_z4c_cpbc_validation.pbs
```

The only intentional evolution difference from the successful Sommerfeld
retry is `boundary_rhs`; both use the byte-identical archived executable,
order 4, disabled sponge, input, ranks, placement policy, and wall limits.
Job `8718576` was initially queued and started at
`2026-07-30 16:56:31 UTC`.

At the user's request, subsequent segments use Athena's restart path with an
internal `-t 00:50:00` limit so final outputs and a new checkpoint are
written before the one-hour PBS limit.  The first segment had already
started fresh with its recorded `00:53:00` internal limit when this request
arrived, and no restart existed at that time, so it was neither killed nor
duplicated.  The submission helper now supports an explicit `RESTART_FILE`:
it requires the existing case directory and checkpoint, invokes `athena -r`,
appends rather than overwrites `athena.stdout`, and writes a job-specific
`run_metadata.restart.<jobid>.txt`.  Fresh runs retain the strict refusal to
reuse any existing case directory.  Local and Aurora `bash -n` checks pass.

Heartbeat automation `monitor-aurora-cpbc-tde` checks this job and run once
per hour.  It may submit exactly one sequential 32-node restart with the
same executable and physics arguments only after a clean below-`t=30`
termination, a valid latest checkpoint, and confirmation that no live job
is using the case.  It must stop rather than restart on nonfinite,
bad-metric, MPI, SYCL, or checkpoint errors.

The first live checkpoint at `2026-07-30 17:40 UTC` was healthy at
`t=6.2015625`: `H-norm2=1.56996140e-5`,
`M-norm2=4.10210206e-6`, `rho-max=1.23580692e-4`, and `bad-metric=0`.
The Hamiltonian maximum was `9.13556e-4` near the star rather than the
Sommerfeld lower-face value of order `10^1` by `t=7`.  No GPU, MPI, SYCL, or
nonfinite signature was present.  Twenty-four x-z slices had been written.
All 384 rank-local restart members were present while the live job was
finishing its output phase; they are not eligible for restart use until PBS
terminates cleanly and their completeness is rechecked.

The first archived-CPBC segment subsequently terminated cleanly on its
internal wall-clock limit.  PBS job `8718576` recorded `Exit_status=0`,
`resources_used.walltime=00:57:34`, and obit time
`2026-07-30 17:55:07 UTC`.  Its final history row is finite at
`t=6.500390625`: `H-norm2=1.65408346e-5`,
`M-norm2=3.43603718e-6`, and `bad-metric=0`.  The final reported Hamiltonian
maximum is `8.05063e-4` at `(36.2578,-3.72656,2.92969)`, near the star rather
than the close lower-z face, and the final `rho_max=1.235100e-4`.  The
complete stdout contains no nonfinite, bad-metric, MPI, SYCL, or GPU-fault
signature and ends with `Terminating on wall clock limit`.

After PBS termination, restart generation `00001` contained exactly one
nonempty member for each of the 384 ranks.  The rank-zero template passed to
Athena is
`rst/rank_00000000/z4c_tov_ks_n3_schwarzschild_bgadapt_cpbc_zminus6_aurora.00001.rst`;
its SHA-256 is
`2e2c2269a84e7cc8d6c345a43147a14f0cb86cedeafd3b2f570c866272852f15`.
Source inspection and the existing production restart harness confirm that
supplying the rank-zero path causes each MPI rank to substitute its own
`rank_XXXXXXXX` directory.

With no live job using the case and the archived executable reverified as
`cd7cefef042ff075e85688dd3ec08dca243df479d50b91e0439dd8a20da46478`,
the first 50-minute restart segment was submitted as job `8718778`:

```text
qsub -N z4c_cpbc_zm6r1 -l select=32 \
  -v REPO_DIR=/flare/MHDTidal/hzhu/tde_n3_validation/worktrees/athenak_characteristic_cpbc,BUILD_DIR=/flare/MHDTidal/hzhu/tde_n3_validation/build/aurora-intel-gpu-characteristic-cpbc,ATHENA_EXE=/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/executables/athena_cpbc_zero_rate_cd7cefef042ff075,CASE_NAME=zminus6_cpbc_archived_20260730,VALIDATION_KIND=athena,RESTART_FILE=/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/runs/characteristic_cpbc/zminus6_cpbc_archived_20260730/rst/rank_00000000/z4c_tov_ks_n3_schwarzschild_bgadapt_cpbc_zminus6_aurora.00001.rst,RANKS_PER_NODE=12,ATHENA_WALLTIME=00:50:00,ATHENA_EXTRA_ARGS=z4c/boundary_rhs=characteristic_cpbc:z4c/extrap_order=4:problem/outer_sponge_enabled=false \
  analysis/z4c_characteristic/aurora/submit_z4c_cpbc_validation.pbs
```

At `2026-07-30 17:58 UTC`, job `8718778` was queued in
`debug-scaling`.  It is sequential with the completed first segment and
reuses the identical run, executable, rank count, boundary condition,
extrapolation order, and disabled-sponge configuration.

Job `8718778` started at `2026-07-30 18:08:13 UTC`.  At
`2026-07-30 18:28 UTC` it was running at `t=9.499219`, beyond the
Sommerfeld control's first catastrophic lower-face growth at approximately
`t=7.0` and first nonfinite history near `t=7.2`.  The latest complete
history row at `t=9.45` remains finite with
`H-norm2=4.77603731e-6`, `M-norm2=9.94985107e-7`, and
`bad-metric=0`; `rho_max` remains approximately `1.241e-4`.  The appended
stdout contains no nonfinite, bad-metric, MPI, SYCL, or GPU-fault
signature.  This is already a qualitative survival improvement over the
matched archived-executable Sommerfeld control, but the run must continue
to `t=30` and the independent all-ten-family comparison remains pending.

The first restart segment then terminated cleanly on its internal
50-minute limit.  PBS job `8718778` recorded `Exit_status=0`,
`resources_used.walltime=00:52:00`, and obit time
`2026-07-30 19:01:14 UTC`.  Its final history row at `t=14.002734375`
remains finite with `H-norm2=8.10899445e-7`,
`M-norm2=1.68814050e-7`, `bad-metric=0`, and
`rho_max=1.245926e-4`.  The final Hamiltonian maximum,
`1.21269e-4` at `(44.3438,-6.28125,-5.90625)`, is on the close-face
layer but is approximately five orders of magnitude below the matched
Sommerfeld value during its failure.  No nonfinite, bad-metric, MPI, SYCL,
GPU, or restart error was found.

Restart generation `00002` contains exactly 384 nonempty rank members.  Its
rank-zero template has SHA-256
`9e0563086ed4a499b2748ad1d64c0e071e4213598d8b859ef1e02adaa63e2147`.
After PBS fully released job `8718778`, the executable hash was reverified,
no live job was using the case, and the next sequential segment was
submitted as job `8718933`:

```text
qsub -N z4c_cpbc_zm6r2 -l select=32 \
  -v REPO_DIR=/flare/MHDTidal/hzhu/tde_n3_validation/worktrees/athenak_characteristic_cpbc,BUILD_DIR=/flare/MHDTidal/hzhu/tde_n3_validation/build/aurora-intel-gpu-characteristic-cpbc,ATHENA_EXE=/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/executables/athena_cpbc_zero_rate_cd7cefef042ff075,CASE_NAME=zminus6_cpbc_archived_20260730,VALIDATION_KIND=athena,RESTART_FILE=/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/runs/characteristic_cpbc/zminus6_cpbc_archived_20260730/rst/rank_00000000/z4c_tov_ks_n3_schwarzschild_bgadapt_cpbc_zminus6_aurora.00002.rst,RANKS_PER_NODE=12,ATHENA_WALLTIME=00:50:00,ATHENA_EXTRA_ARGS=z4c/boundary_rhs=characteristic_cpbc:z4c/extrap_order=4:problem/outer_sponge_enabled=false \
  analysis/z4c_characteristic/aurora/submit_z4c_cpbc_validation.pbs
```

At `2026-07-30 19:02 UTC`, job `8718933` was queued in
`debug-scaling`.

Job `8718933` started at `2026-07-30 19:12:34 UTC`.  At
`2026-07-30 20:00 UTC` it was still running at `t=19.74727`.  The latest
complete history row at `t=19.7015625` remains finite with
`H-norm2=4.14081869e-7`, `M-norm2=8.25550121e-8`, and
`bad-metric=0`; `rho_max` is approximately `1.260e-4`.  No nonfinite,
bad-metric, MPI, SYCL, GPU, or restart error is present.  Because the job
was still live, no further restart was submitted at this checkpoint.

Segment `8718933` subsequently terminated cleanly.  PBS recorded
`Exit_status=0`, `resources_used.walltime=00:52:42`, and obit time
`2026-07-30 20:06:18 UTC`.  The final history row at
`t=19.75078125` remains finite with `H-norm2=4.11994481e-7`,
`M-norm2=8.20895725e-8`, `bad-metric=0`, and
`rho_max=1.260673e-4`.  The close-face Hamiltonian maximum is still bounded
at `1.22624e-4`; stdout ends on the internal wall-clock limit with no
nonfinite, bad-metric, MPI, SYCL, GPU, or restart error.

Restart generation `00003` contains exactly 384 nonempty rank members.  Its
rank-zero template has SHA-256
`fd4f5fd317c89f65e095cdd6f432fbef1751816851be8ab4455b3fd184c2610b`.
After revalidating the executable and confirming that no live CPBC job was
using the case, the next sequential segment was submitted as job `8719210`:

```text
qsub -N z4c_cpbc_zm6r3 -l select=32 \
  -v REPO_DIR=/flare/MHDTidal/hzhu/tde_n3_validation/worktrees/athenak_characteristic_cpbc,BUILD_DIR=/flare/MHDTidal/hzhu/tde_n3_validation/build/aurora-intel-gpu-characteristic-cpbc,ATHENA_EXE=/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/executables/athena_cpbc_zero_rate_cd7cefef042ff075,CASE_NAME=zminus6_cpbc_archived_20260730,VALIDATION_KIND=athena,RESTART_FILE=/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/runs/characteristic_cpbc/zminus6_cpbc_archived_20260730/rst/rank_00000000/z4c_tov_ks_n3_schwarzschild_bgadapt_cpbc_zminus6_aurora.00003.rst,RANKS_PER_NODE=12,ATHENA_WALLTIME=00:50:00,ATHENA_EXTRA_ARGS=z4c/boundary_rhs=characteristic_cpbc:z4c/extrap_order=4:problem/outer_sponge_enabled=false \
  analysis/z4c_characteristic/aurora/submit_z4c_cpbc_validation.pbs
```

At `2026-07-30 21:01 UTC`, job `8719210` was queued in
`debug-scaling`.

Job `8719210` started at `2026-07-30 21:11:49 UTC` and reached the requested
`t=30` without consuming its full internal wall-clock allowance.  PBS
recorded `Exit_status=0`, `resources_used.walltime=00:34:42`, and obit time
`2026-07-30 21:47:30 UTC`.  The final history row is finite with
`H-norm2=1.09007383e-7`, `M-norm2=2.56217870e-8`,
`bad-metric=0`, and `rho_max=1.28672199e-4`.  The star tracker remains
valid at `(35.37305,-0.005859,-0.005859)`.  The final Hamiltonian maximum is
`1.20496e-4` at `(44.3438,-6.28125,-5.90625)`; this close-face residual
remains bounded through more than three nominal gauge round trips.  Stdout
ends with `Terminating on time limit` and contains no nonfinite, bad-metric,
MPI, SYCL, GPU, or restart error.  No further segment was submitted.
Hourly automation `monitor-aurora-cpbc-tde` was paused after target
completion.

### Archived-executable thin-z CPBC comparison

The completed run establishes a strong qualitative difference from the
byte-identical archived-executable Sommerfeld control: Sommerfeld developed
order-10 to order-100 lower-face Hamiltonian maxima, nonfinite histories,
and a density-floor stellar state beginning at `t=7.0--7.27`, whereas the
zero-rate CPBC case remains finite with a close-face maximum near
`1.2e-4` through `t=30`.

The independent all-ten-family analyzer was run on the causal
`t=6.5--7.15` stellar/return window (`x=32--45`, `z=-6--6`).  This window
retains a resolvable face-normal metric frame and does not use boundary
enforcement diagnostics.  The logs are
`analyze_zminus6_sommerfeld_causal_6p5_7p15_20260730.log`
(SHA-256
`380c58c27faf947ff36d0f994ac18d503d245c1dab7bd2e380f477c29f99af49`)
and `analyze_zminus6_cpbc_causal_6p5_7p15_20260730.log`
(SHA-256
`f04b710b4fcf1bc466b680b8a6d1204b7b140ee5d533f1558548b203e14ae71d`)
under the project `build_logs` directory.

The maximum raw incoming RMS ratios in that pre-nonfinite slice window are:

| Incoming family | Sommerfeld/CPBC |
|---|---:|
| lapse | `1.000036` |
| longitudinal shift | `1.014123` |
| transverse shift 1 | `1.078244` |
| transverse shift 2 | `1.577494` |
| scalar constraint Theta | `1.020440` |
| scalar constraint Z | `1.114506` |
| transverse constraint 1 | `1.022379` |
| transverse constraint 2 | `1.147016` |
| TT plus | `1.752773` |
| TT cross | `1.000010` |

These raw values include the legitimate stationary stellar near field and
are not a far-reference-subtracted returning signal.  They must not be
reported as satisfying the tenfold reflection gate.  They show that the two
runs remain nearly identical in the star region through the last finite
Sommerfeld residual slice at `t=7.00313`; the catastrophic divergence
appears immediately afterward in the global histories and at the close
face:

| Causal-window diagnostic | Sommerfeld maximum | CPBC maximum | Sommerfeld/CPBC |
|---|---:|---:|---:|
| `Theta-max` | `3.85857e-2` | `2.01625e-5` | `1.914e3` |
| `alpha-res` | `1.08070e-2` | `9.49331e-6` | `1.138e3` |
| `beta-res` | `3.24569e-4` | `3.59848e-6` | `9.020e1` |
| `Gam-res` | `3.20782e-3` | `2.44045e-5` | `1.314e2` |
| `C-norm2` | `5.10348e1` | `1.65408e-5` | `3.085e6` |
| `H-norm2` | `4.76992e1` | `1.30876e-5` | `3.645e6` |
| `M-norm2` | `3.13449` | `3.43604e-6` | `9.122e5` |
| `Z-norm2` | `5.02851e-2` | `4.87999e-10` | `1.030e8` |
| `Theta-norm` | `4.32375e-5` | `1.51984e-8` | `2.845e3` |

A requested lower-face-only characteristic projection used
`x=-6--6`, `z=-6---3`.  The evolved Sommerfeld inverse-metric normal
developed an out-of-xz-plane component of `1.00845e-3` at the failure
slice, exceeding the analyzer's geometric resolvability guard.  The
projection was therefore rejected rather than silently assuming a
coordinate-aligned normal.  Raw coordinate-bearing histories and stdout
still localize the failure at the first lower-z cell layer.  A future
fully three-dimensional residual output would be required to project the
all-ten face modes after that metric tilt develops.

The late CPBC-only projection over `t=27--30` contains 13 slices and all ten
incoming/outgoing families.  Its log is
`analyze_zminus6_cpbc_late_27_30_20260730.log`, SHA-256
`24ddd2d0cdba5fe7e975d785e8c4d2a7ccd7ae6b0f1644e2dd1c5c4298105856`.
The maximum incoming RMS values remain bounded:

| Incoming family | Late CPBC maximum RMS |
|---|---:|
| lapse | `3.22520e-6` |
| longitudinal shift | `3.45682e-6` |
| transverse shift 1 | `1.83089e-6` |
| transverse shift 2 | `8.37305e-9` |
| scalar constraint Theta | `6.41182e-6` |
| scalar constraint Z | `1.07866e-6` |
| transverse constraint 1 | `1.69201e-6` |
| transverse constraint 2 | `1.08745e-8` |
| TT plus | `1.34214e-6` |
| TT cross | `8.72326e-8` |

Restarted runs emit the same state at both sides of a segment boundary.
The CPBC dataset contains duplicate slice times at `6.50039`, `14.0027`,
and `19.7508`; each pair has the same mesh layout and exactly zero field
difference, although its binary header and whole-file hash differ.  The
reader now collapses only field-identical duplicate-time slices and rejects
any conflicting pair.  Interval-restricted history analysis likewise
checks finiteness inside the requested interval, permitting finite
pre-failure Sommerfeld analysis while still preserving its later nonfinite
rows.  Python compilation, `git diff --check`, and the 20-row analytic
projection test pass after these reader changes; the projection error is
`9.02056208e-17`.

This archived zero-rate CPBC result is not a final acceptance result.  The
existing far reference has different thin-z geometry, only the earlier
slice orientation, and no executable parity at `t=30`; it cannot supply the
required matched returning-signal, density, or trajectory subtraction for
this case.  Consequently the tenfold group gate, per-family 10% gate,
central-density 1% gate, and `0.1 R_star` trajectory gate remain
unassessed.  A matched far-boundary reference with x-z residual outputs is
required if the archived candidate is to be evaluated against those
criteria.  The newer tangential-principal executable is a separate,
still-unvalidated candidate and is not represented by this TDE result.

The independent TDE comparator now reads x-y, x-z, or y-z residual slices
and supports boundary axes 1--3.  It projects all ten incoming families from
the same independently reconstructed full-conformal-metric face frame.  The
two radiative projections are

```text
tt_plus  = -2 A_plus/sqrt(chi) + d_s g_plus
tt_cross = -2 A_cross/sqrt(chi) + d_s g_cross
```

where the plus and cross components use the two deterministic metric-unit
tangents returned by that frame.  Both the x-y and rotated x-z synthetic
fixtures contain nonzero TT curvature and normal-metric-derivative
contributions.  On Aurora they pass all ten closed-form expectations with
maximum incoming error `8.32667268e-17`; the independent Schwarzschild
background reconstruction error remains exactly zero.  The projector also
implements the ten outgoing partners using the independently transcribed
negative-root rows from the boundary kernel.  Expanded closed-form xy/xz
fixtures exercise all 20 rows and pass on Aurora with maximum error
`9.02056208e-17`.  The comparator applies the tenfold improvement gate to the
gauge and constraint groups and separately rejects any individual family
whose RMS is more than 10% above its matched Sommerfeld value.  Its CPBC and
CPBC-plus-sponge positional inputs are now optional, so the required
Sommerfeld-first control can be analyzed and reported independently without
supplying duplicate placeholder cases.

The optional-case path was exercised end to end on the completed far and
Sommerfeld-only x-y datasets without rerunning either evolution.  Five
matched slices and at least 55,888 cells per slice reproduce the earlier
gauge RMS `5.40074273e-8` and constraint RMS `3.83203235e-7`, while adding
radiative RMS `1.15740417e-8`.  The separate TT values are
`tt_plus=1.15740417e-8` and `tt_cross=2.25161618e-10`; all ten modes are
printed independently.  The exact output is
`/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/build_logs/compare_tde_boundaries_ten_mode_sommerfeld_smoke_20260730.log`
with SHA-256
`fedf2d7d9567e960d3e7e41eac3227c08916c4af3e59b6109abbf97dd835d9c2`.

`analysis/z4c_characteristic/analyze_tde_boundary_series.py` is the
single-run companion for causal diagnosis.  On every selected x-z residual
slice it reports RMS and Linf for each incoming and outgoing lapse,
longitudinal-shift, two transverse-shift, two scalar-constraint, two
transverse-constraint, and two TT modes.  Each Linf includes its physical
`x,y,z` cell center, and the final summary retains the time and coordinates
of the run-wide maximum for every direction/family.  The same per-slice and
run-wide coordinate-bearing summaries cover residual Theta, Khat, all three
Gamma components, lapse, and all three shifts.  It also reports the selected
interval's central-density and bad-metric histories, Hamiltonian and
momentum norms (including components), STAR_TRACK motion, and a fatal/MPI/
SYCL error scan.  It uses the same full-metric frame, exact background
reconstruction, slice-normal resolvability check, and block-edge derivative
exclusion as the matched comparator; it does not use a boundary-kernel
diagnostic.  STAR_TRACK and error scans now stream stdout line by line rather
than loading the entire file; this was required once the intentionally
continued post-failure control log reached approximately 1.7 GB.  Optional
asymmetric row bounds allow a lower-face-only x-z window that excludes the
black-hole/excision region, where open-face sign-straddling assumptions are
not applicable.

As an end-to-end reader/projection smoke, the analyzer processed the
completed far-reference x-y slices over `t=13.0`--`13.5`, selected at least
74 leaf blocks and 55,888 post-margin cells per slice, emitted all 20
per-direction/family rows plus all summaries, found finite histories and
zero error signatures.  The 107-line machine-readable log is
`/lus/flare/projects/MHDTidal/hzhu/tde_n3_validation/build_logs/analyze_tde_boundary_series_full_smoke_20260730.log`
with SHA-256
`0b902b3f41aae7f9f57c83e0e1b88870c000345b014d334396d145a85301f6ff`.

The reduced-gate PBS path was statically preflighted but not submitted while
the control is queued.  This audit found that `PULSE_EXTRAP_ORDER` was
declared by the PBS harness but was not exported to the pulse-suite child
process, which would have silently selected that script's order-2 default.
The harness now passes it explicitly as `CPBC_EXTRAP_ORDER`, records the
resolution, center, end time, extrapolation order, axes, sides, and family
selection in `run_metadata.txt`, and records both staged and unstaged patch
hashes.  Local and Aurora `bash -n` checks pass.  The prepared reduced suite
will set `CPBC_AXES=1`, retain both default signs and all ten default
families, and use the requested order 4; no candidate job has been launched.

The experimental tangential-principal implementation was rebuilt from a
clean build tree with `source ~/athenak_env` and `make -j64`.  The build
completed successfully with no compiler error; the resulting executable
SHA-256 is
`298646be8918a7778329510eefb617cf9458ffe11056688c4f5f881434f9999e`
and the build-tree/archive copies are byte-identical.  The earlier
incremental checkpoint build had SHA-256
`483325ce59992648fc8efcb92fcc9f3bd437c35d01faad0291cf89a1c31e4b2e`
and is superseded as validation provenance.

The clean device link reports SIMD32/256-register spill counts of
approximately 78, 82, and 79 slots for the three tangential-principal CPBC
orientation kernels, compared with 6, 14, and 18 for the adjacent retained
Sommerfeld kernels.  This material regression is explicitly unresolved.
The clean executable has not yet passed the pulse, exact-background, corner,
reflection, TDE, or measured-performance gates and must not be described as
a validated CPBC.

### Tangential-principal static audit

A term-by-term source audit against `src/z4c/z4c_calcrhs.cpp` confirms that
the experimental target reconstructs the same NGHOST=4 centered first,
second, and mixed derivatives used by the volume RHS.  Its frozen principal
operator retains full-state shift advection for the geometric variables and
analytic-background shift advection for the background-adapted lapse and
shift.  It includes the conformal Ricci/lapse trace-free terms, the
Hamiltonian principal terms, the Gamma-driver Laplacian and grad-div terms,
and the normal derivatives of the configuration-variable principal RHSs.
For each incoming row the implemented target is

```text
full frozen principal characteristic rate
  - lambda_in * normal derivative of that characteristic
```

so the remainder is the tangential and mixed principal datum.  The correction
solves only for the momentum-like RHS variables `Khat`, `Theta`, `A_ij`, and
`Gamma^i`.  It does not separately assign an outgoing characteristic or
overwrite the lapse, shift, conformal metric, chi, or auxiliary-B RHS.  As is
unavoidable in this second-order formulation, changing a momentum-like RHS
also changes the algebraically evaluated outgoing rate; that is not an
independent outgoing prescription.

This audit does not establish a nonlinear constraint/Psi0 boundary hierarchy.
The target intentionally contains only the frozen principal tangential
terms, not separately derived nonlinear constraint or residual-Psi0
lower-order terms.  The implementation must therefore continue to be called
an experimental tangential-principal CPBC.  Static agreement does not
substitute for the pending pulse, exact-background, oblique determinism, and
TDE gates.

## Archived thin-z CPBC/Sommerfeld comparison artifacts

The matched archived-executable comparison is complete.  The Sommerfeld
reproducer (`8718377`) develops a lower-z-localized instability at
`t/M ~= 7`, has its first nonfinite history row at `t/M = 7.2`, and shortly
afterward loses the stellar density tracker to the atmosphere floor.  The
zero-rate characteristic-CPBC run (`8718576`, `8718778`, `8718933`, and
`8719210`) exits every segment cleanly and reaches `t/M = 30`.  At the exact
finite history sample `t/M = 7.1015625`, the Sommerfeld/CPBC ratios are
`5.296842e6` for `H-norm2`, `1.320728e6` for `M-norm2`, and `3.3223e3` for
`Theta-norm`.  The CPBC endpoint has `H-norm2 = 1.0900738e-7`,
`M-norm2 = 2.5621787e-8`, close-face `Hmax = 1.20496e-4`, and zero
bad-metric count.  The supported claim is specific: the archived zero-rate
CPBC cures the observed Sommerfeld instability in this thin-z configuration.

The orbital-plane morphology analysis reads all 61 CPBC `xy_mhd` density
slices.  The half-maximum effective core radius stays within `-0.63%` to
`+1.24%` of its initial value, the density-weighted principal-axis ratio
stays in `[1.0005, 1.0748]`, and the centered normalized shape difference
stays below `0.0738`.  At `t/M = 30`, these diagnostics are
`R_half = 0.0831065 M`, `a/b = 1.02819`, and shape difference `0.0442`.
The selected Sommerfeld slice is `t/M = 7.00313`, where
`rho_max = 1.236772e-4` is physical rather than floor-dominated.  No CPBC
density slice exists at that exact time; the nearest stored CPBC slice is
`t/M = 7.50234`, offset by `0.49921 M`, and is labeled as such.  The density
evidence establishes continuity of the two-dimensional orbital-plane core,
not the unobserved full three-dimensional stellar shape.

The committed report and reproducibility artifacts are:

- `docs/z4c/tde_boundary_comparison.tex` and the compiled six-page
  `docs/z4c/tde_boundary_comparison.pdf` (PDF SHA-256
  `88c90982e391c4d6ec5f31e2ea8041d10f6c02556e55ef00f35cdbbb97517d3d`);
- `docs/z4c/figures/tde_boundary_constraint_histories.pdf`,
  `tde_boundary_residual_histories.pdf`, and
  `tde_boundary_theta_slice_t7.pdf`;
- `docs/z4c/figures/tde_stellar_morphology_cpbc_vs_sommerfeld.pdf`
  (SHA-256
  `b1af7d25d1522f4e415c76a5816d9f206fc4124dfc6d6658edc7b03ff205c5bf`);
- `docs/z4c/figures/tde_stellar_morphology_metrics.csv` (SHA-256
  `c7e713617cccc37fed4972289763bf1176dd5392c42f99998609d67891787757`).

The final cleanup also hardens the formal comparator and launch harness.  The
comparator now requires history, STAR_TRACK, and residual-slice coverage
through a finite required end time (default `30 M`), rejects any nonzero
bad-metric history, and validates every numeric acceptance threshold against
NaN or invalid ranges.  The single-run analyzer recognizes explicit
nonfinite, bad-metric, GPU page-fault, and device-lost signatures.  The PBS
harness performs kind/repository/build/executable/input/restart preflight
before creating a run directory, requires a rank-0 base restart inside the
case `rst/` tree, verifies a complete nonempty rank-local restart cohort, and
records the effective pulse axes/families/center/end time.  The harness passes
`bash -n`; Python sources compile; the 20-row incoming/outgoing projection
test passes with maximum error `9.02056208e-17`; and an invalid-kind preflight
test exits with status 2 without creating a case directory.

## Pending acceptance gates

The material Sommerfeld failure and the immutable archived-executable
Sommerfeld/zero-rate-CPBC evolutions on the selected thin-z domain are now
complete.  Remaining gates are:

1. Run the matched thin-z far-boundary reference with x-z residual outputs
   and executable parity, then evaluate the all-ten returning-signal,
   density, and trajectory gates.
2. Audit and validate the multidimensional normal/tangential principal
   split before using the experimental executable for a TDE claim.
3. Demonstrate at least tenfold returning gauge/constraint reduction, no
   family more than 10% worse than Sommerfeld, central-density difference
   below 1%, and trajectory difference below `0.1 R_star`.
4. Pass all controlled pulse reflections below 2%, exact backgrounds below
   `1e-12`, corner determinism, and second-order convergence.
5. Measure CPBC kernel cost below 3% of the Z4c volume RHS.
6. Complete the 32-node production-input smoke with the `512--640M`,
   `tau=16M` sponge and the final candidate.
7. Rerun default-path parity regressions with the final executable, then
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
