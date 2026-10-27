# Puncture AMR: field differentiation and value-only halo prototype

This is a bounded, read-only diagnostic of the actual N6, seven-shell,
refinement-three, 47,088-cell initial state in
`evidence/amr-core-pilot-20261027`. It identifies the largest initial Hamiltonian
defect and qualifies a new optional scalar-log halo component. No existing
`EnrichedLogDerivative`, `PunctureLogPlans`, `CompositeDifference`, pilot, CMake,
physical source, reference gauge or constraint equation was changed here.
Original one-dimensional jobs were not signalled or modified.

## Concrete failure mechanism

The reported maximum is active point 34308 at
`(-.059895833333333336,-.059895833333333336,-.002604166666666668)`, radius
`.08474552132954678`, local spacing `.005208333333333333`, and rho/h=16.27114.
It is within the existing `.1` enrichment radius but is skipped by its
rho/h<=8 selection. Its fourth-order scalar stencils use 42 internal ghost
locations. The initial metric is conformally flat and curvature/trace vanish
in this isotropic inner branch, so the physical spatial-curvature expression
from q=log(chi),

    chi * [-6 + 2 Omega x·grad(q)
                 + Omega^2 (2 Laplacian(q) - |grad(q)|^2/2)],

equals the local Hamiltonian. Direct evaluation with the ordinary mapped
stencil gives `.4063052927419656`, reproducing the actual pilot result to
roundoff. Analytic radial scalar jets from the independent existing 1D
initializer give `7.48e-15`. They are diagnostic exact initial data, never an
evolution source or a background subtraction.

| Field control at34308 | H from chi | Maximum log-chi gradient error | Maximum log-chi Hessian error |
|---|---:|---:|---:|
| Ordinary mapped fourth-order FD | .4063053 | .0168386 | 6.69270 |
| Exact values at the 42 internal ghosts only | -.000381830 | .000281202 | .0436828 |
| Active-only enriched derivative, 125 nearest Cartesian donors | .000912239 | .0000855606 | .0396553 |
| Enriched **value** at coarse/fine ghosts, unchanged FD afterward | .000598882 | .000325447 | .0501884 |

The exact-ghost control leaves every active value and the mapped FD/Jacobian/
Hessian machinery unchanged. Its roughly 1,064-fold reduction in absolute H
localizes the dominant defect to coarse/fine ghost interpolation. This is
different from a failure of the elliptic solve or a missing physical source.
For log(alpha/alpha_reference), the ordinary Hessian error 3.34636 falls to
.0218414 with exact ghosts, .0198278 with direct enriched derivatives, and
.0250942 with value-only ghosts. The lapse reference remains the prescribed
Minkowski `(1+r^2)/2`.

The derivative diagnostic includes eight sign-related worst points and eight
additional radius/ratio controls. All tested 125/175/256-donor derivative fits
have full rank. At34308 the basis condition is 3.45e5. Its reported maximum
scaled-weight L1=151568 belongs to the **log-power diagnostic coefficient**;
the h-scaled gradient and h²-scaled Hessian L1 values are 3.38 and 3.47. Across
the sampled125-donor targets, these derivative values are at most 7.22 and 17.75.
The sample up to rho/h 18.835 does not establish that extending the existing
derivative wrapper's selection is safe on every mesh. No such extension was
made; the value-only approach below avoids that selection change.

## New isolated value component

`src/z4c/hyperboloidal/enriched_log_value.hpp` caches only scalar VALUE weights.
It reuses the audited rank-revealing 56 degree<=5 Cartesian polynomial columns
plus log(rho), recovers functional0 using impulse evaluations, checks value
reproduction and a separate VALUE L1 cap, and releases all derivative and
log-power coefficient rows. It does not cache an evolved power or fitted field.
Evaluation uses anchor differences to preserve constants bitwise. A ghost
query coincident with an existing active node uses exact identity evaluation;
puncture collocation is rejected. Active values are never replaced.

At34308, 32 of the 42 used ghosts cross Cartesian refinement levels. The other
10 coincide with same-level active nodes. Changing all 42 and changing only 32
coarse/fine values give **bitwise identical** target derivatives. The largest
VALUE L1 is 5.488, below the fixed8 cap; largest value reproduction error is
1.14e-13. Deterministic donor perturbations of amplitude1e-12 produce maximum
ghost-value perturbation 2.10e-12 and change H by 1.99e-9. The original physical
positive fields are reconstructed from the same active values; there is no
field projection. Position/time or topology changes are rejected before output
is written.

`src/z4c/hyperboloidal/puncture_log_value_plans.hpp` adds an optional host cache:

```cpp
PunctureLogValuePlans(grid, puncture, radius, maximum_unique_rows,
                     maximum_value_l1, maximum_donors);
cache.FillPositiveLogs(log_chi_padded, log_alpha_reference_padded, puncture);
```

It selects Cartesian receiving halos within rho<.12 whose active Cartesian
donors cross refinement levels. The existing first/second stencils never read
triple-logical-corner padding, so that unused padding is not cached. No active
cell is excluded. A KD tree finds deterministically ordered nearest **active**
Cartesian donors, with no ghost-to-ghost dependencies. Only the two scalar-log
buffers are accepted by this API; beta, metric, signed curvature and flux fields
are not passed to it.

## Complete cache qualification and explicit rank control

The full mesh has 19,248 selected receiving assignments. The initial one-plan-
per-assignment implementation correctly rejected its10,000-row engineering
budget before allocating the cache. Exact Cartesian logical-lattice keys,
anchored at `-half_width`, identify 9,616 distinct physical query/receiving-
spacing pairs. Their duplicate floating coordinates differ by at most 1.23e-17.
Sharing a field-independent value row retains **all 19,248 assignments**. No
puncture position or physical coordinate is snapped. A separate test evaluates
a degree-five polynomial plus 1.09log(rho) at every actual receiving target;
maximum reproduction error is 2.69e-14, including the floating coordinate aliases.

A125-only full cache then explicitly rejected halo 48768 at
`(-.11458333333333334,-.03125,-.01041666666666667)`. Its125 nearest donors have
only five distinct x planes. A degree-five x polynomial is therefore
unidentifiable: rank 56 of57. Counts 126,128 and135 still fail; 150 donors add the
sixth plane and recover rank 57, condition 4.87e4 and VALUE L1=2.762. This failure
is preserved; it is not masked by a lower polynomial degree or an ordinary-FD
fallback.

The helper therefore exposes **explicit bounded support enlargement**. Its
default maximum_donors=125 retains the strict125-only rejection. A caller may
request support counts 125→150→175→256, with every rejected support and selected
count recorded. The fixed full-cache qualification requested 256:

| Complete cache result | Measured value |
|---|---:|
| Unique cached value plans / receiving assignments | 9,616 /19,248 |
| Selected125 /150 /175 /256 donor supports | 9,070 /544 /2 /0 |
| Rejected rank-deficient attempts | 548 |
| Maximum full-basis condition | 9.68e5 |
| Maximum VALUE L1 | 6.807, with cap8 |
| Maximum internal degree-five/log reproduction error | 3.79e-13 |
| Maximum reproduction error at actual receiving targets | 2.69e-14 |
| Active-field change | exactly0 |
| Cache build / one two-field fill | 15.45seconds /7.93milliseconds |

The retained donor-index/weight payload is about 19.5MB, plus row/assignment and
stamp metadata. The KD tree and full basis factors are released after build.
The entire cache test took 21.84seconds including the unchanged existing 2,728-row
derivative reconstruction and diagnostics. Build and test processes each have a
180-second bound and use 47,088 active cells. A pilot with restart must still
measure its **total** time, including hierarchy/cache reconstruction; constructor
cost must not be omitted from the budget.

The cache checks mesh revision, active index map, logical shape/halo/order,
patch descriptors and bounds, and the puncture's full position/velocity/time/
stage/revision stamp. Regrid, map changes, restart rebuilds or changed puncture
kinematics require a fresh cache. A temporary grid is rejected at compile time.
Both field buffers are committed only after every row/input succeeds; stale
position, revision, changed map and nonfinite-data controls leave outputs
unchanged. The tests preserve constants bitwise and exercise these rejections.

## Raw core result and remaining scientific gate

With the current derivative wrapper still unchanged at rho/h<=8, the raw
Cartesian rho<.12 **diagnostic partition** contains 18,368 active cells:

| Raw chi-derived physical spatial curvature | Before | Value-only cache |
|---|---:|---:|
| Maximum | .4063053 | .0478225 |
| Physical-volume RMS | .0321722 | .0320288 |

The new maximum is point 34919,
`(-.0026041667,-.0026041667,-.0130208333)`, rho/h≈2.598. It already uses the
active-only derivative enrichment. That exact same enriched jet is used before
and after the value-cache comparison, so this remaining peak is inherited
near-puncture reconstruction error, not an introduced halo error. The small RMS
change also makes clear that removing the largest interface peak does not solve
the unresolved core regularity problem.

These partition diagnostics are **not** full-domain acceptance. No coupled
evolution, full H/M/Theta/Z norms, scri closure or restart pilot was rerun by this
subtask. The parent must retain every active cell in its existing acceptance
norms and test the optional cache in that actual coupled pipeline. No MPI,
device, native-task or translating-puncture qualification is claimed.

The intended insertion is after the ordinary positive-log ghost fill and
before existing mapped first/second derivatives in the evolution-stage sampler,
only for exponent-independent log storage. The existing core derivative override
can then run as before. Direct-vector shift boundary/operator evaluation and
all other fields keep their current paths. Reconstruct cache rows from geometry
after restart/regrid; no fitted exponent or cache data need become evolved state.
The constructor must continue to reject unsupported rank, weight, lattice,
Cartesian donor-domain or row-budget configurations explicitly.

## Reproduction and preserved evidence

```sh
python3 tst/elliptic_shift_3d/run_amr_core_derivatives.py \
  --build build-shift-3d \
  --initial docs/elliptic-shift-3d/evidence/amr-core-pilot-20261027/initial.dat \
  --output docs/elliptic-shift-3d/evidence/amr-core-derivatives-reproduction

python3 tst/elliptic_shift_3d/run_amr_core_values.py \
  --build build-shift-3d \
  --initial docs/elliptic-shift-3d/evidence/amr-core-pilot-20261027/initial.dat \
  --oracle docs/elliptic-shift-3d/evidence/amr-core-derivatives-weight-audit-20261027/exact-jets.dat \
  --output docs/elliptic-shift-3d/evidence/amr-core-values-reproduction

python3 tst/elliptic_shift_3d/run_amr_core_values.py \
  --build build-shift-3d \
  --initial docs/elliptic-shift-3d/evidence/amr-core-pilot-20261027/initial.dat \
  --cache --maximum-donors 256 \
  --output docs/elliptic-shift-3d/evidence/amr-core-cache-reproduction
```

The derivative runner samples only 16 diagnostic targets, 567 used internal ghost
locations and 2,392 active donor points. Initial/oracle values agree exactly.
Two oracle-only samples at r=.5 and r=1 satisfy the existing generator's
nonempty-transition and final-scri assumptions; neither is used in any core
derivative. Its first empty-transition generator failure is preserved without
editing that one-dimensional implementation.

Authoritative completed frozen-source manifests:

* `evidence/amr-core-derivatives-weight-audit-20261027/manifest.json`
* `evidence/amr-core-values-20261027/manifest.json`
* `evidence/amr-core-value-cache-enlarged-support-20261027/manifest.json`

Their source, static libraries and actual initial data remained unchanged during
the completed runs. Each records exact build/run commands, hashes, stdout,
stderr and durations. Earlier generator, row-budget and125-only rank failures
remain in the adjacent `amr-core-derivatives-20261027`,
`amr-core-value-cache-20261027`, `amr-core-value-cache-counts-20261027`, and
`amr-core-value-cache-deduplicated-20261027` directories. The separate
`amr-core-value-rank-20261027` control records the donor-plane/rank transition.


## Completed optional shared-engine pilot

The subsequent shared tensor engine now inserts this cache after ordinary
positive-log halo filling, only when `log_halo_enrichment` is explicitly enabled
with exponent-independent log storage and the derivative wrapper. The research
checkpoint v13 binds this flag and reconstructs cache geometry after restart.
No cached fit or exponent becomes an evolved variable.

The 47,088-cell selected wormhole run to t=.0002 completes in127.55s including
rebuild/restart and is bit-identical after restart. All-cell H RMS changes
.0298108→.0296799 and maximum .406127→.195554. The remaining maximum moves to
point25482 at r=.169491, outside the cache's .12 support. Its error was already
.195553 before the change. All-cell M RMS remains .0328553. The full vector
solves converge (maximum physical RMS2.73e-11, point1.25e-9), but the unchanged
scri tensor pole is1.44e-5 and fails its1e-7 target. This is a completed failed
scientific pilot, not a successful radial reproduction. Exact command, source
archive, outputs and maximum localization are in
`evidence/amr-log-halo-coupled-20261027`. The broader halo support and separate
scri defect still require their own qualification.
