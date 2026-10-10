# AthenaK hyperboloidal project: assessment handoff, 2026-10-10

The user requested that research stop here, the work be committed and pushed,
and another agent independently assess the best way forward. No scientific
process is running at handoff. The project objective remains unmet: a
reasonably large angular gauge disturbance must evolve stably on Minkowski,
followed by a single black hole that survives its inner wormhole-to-trumpet
transition **with a Minkowski hyperboloidal reference throughout**.

Remote: <https://github.com/HengruiZhu99/athenak>. Implementation branch:
`z4c_hyperboloidal_layer`. The previous pushed checkpoint was
`284b4c21e09077ab86f0d0cbbbb5b3a11503cf58`; this handoff is its successor.
The original `z4c_hyperboloidal` branch remains at
`ca77b353a60a939c66227e42b02318c5cd32be9d`.

Production `src/` and root `CMakeLists.txt` remain byte-identical to
`27c19d20696ea6dd4704032c51dfd026218f64f2`. This commit preserves private
research code and evidence; it does not integrate the new exact arithmetic
into AthenaK. The supported production path remains CPU Serial/double,
vacuum, one uniform 3D MeshBlock. Unsupported MPI, AMR, GPU, matter and
floor combinations remain rejected.

## Read these first

- [Formulation and initial limitations](hyperboloidal-layer.md),
  [stabilization and wormhole data](hyperboloidal-layer-stabilization.md), and
  [production validation](hyperboloidal-stability-validation.md).
- [Actual final native evolution failures](hyperboloidal-reference-wave-map-final-matrix.md).
- [Gauge derivative tests and original 63-comparison failure](hyperboloidal-derivative-pilots-inner-gauge.md).
- [Manufactured Gaussian slicing and its future-time counterexample](hyperboloidal-manufactured-Gaussian-slicing.md).
- [This checkpoint's evidence](validation/hyperboloidal-assessment-handoff-20261010/README.md)
  and [the next-agent prompt](hyperboloidal-next-agent-prompt-20261010.md).

The evidence capsule mirrors paths below `build-layer-research/`. In the rest
of this document, `CAP/` means
`docs/validation/hyperboloidal-assessment-handoff-20261010/`; `LOCAL/` means
`build-layer-research/` in the original checkout at
`/Users/hz0693/research/hyperboloidal`. CAP contains exact small text copies;
large arrays, maps, scientific JSONL and compiled files are hash/size metadata
only. Its catalog SHA256 is
`0767d909b4b55e17356c72fe9990cc5f8dc36bf8f8c7ae997eb43246009e86fc`.
It contains 1,384 files totaling 50,486,018 bytes, including 665 finite JSON
copies; nine payloads are metadata only. The collector rehashed 5,368 unique
protected inputs and changed no original. Collection source and its metadata
reader failure are preserved in the separate collector capsule.

## Completed new checks and their scope

| Stage | Actual result | What remains outside its scope |
| --- | --- | --- |
| v9 final mass-product diagnostic | PASS; 131,072 fixed products classified | Original v9 radial readback remains FAIL |
| v10 primary, radial-pair and angular-pair readbacks | All three actual root/child PASS | No generator spectrum, propagation or native stability |
| Gaussian third-jet oracle v3 units | 318 fixed units PASS | Full geometry registry and native RHS binding |
| Gaussian v3 timing | 20 records PASS at 180/220 digits | Full 5,010-record stage NOT RUN |
| Interval cache v7 | Independent source review PASS | Proposed 237 units, producer and replay NOT RUN |
| Exact rational backend and whole gauge | Source/math reviews only | No compilation, arithmetic units or native integration |

The v9 mass diagnostic identifies 28 tiny final `measure*mass_sum` products
at radial index 627: 20 round to zero and eight to nonzero subnormal. It
classifies saved operands without assembling E or rerunning a solver.
The original strict-underflow failure is preserved.

v10 changes exactly five final measure products, E/Ks/Kw/G/loads, to five
separate instances of the previously tested exact tiny-product fallback.
Inner operands, quadrature weights, accumulation order, BLAS/einsum,
SVD/solve and tolerances are unchanged. Root elapsed times were
350.105694125 / 687.94382875 / 385.948976167 seconds for primary/radial/angular;
all finished within their new, explicitly declared 900-second caps. Radial E
uses 28 fallbacks; the other four final products use none.

The saved-only independent review recounted 3,534 checks each in primary and
angular, including the 33 direct-forcing fields. Their energy minimum is
about 0.2608755 and condition about 51,776. The 1,276,457-byte radial result
was never decoded by that review or this collector. Its PASS is associated
through actual root/child receipts and exact output hashes, with ten compact
rounding audits; its numerical comparisons were not independently recounted.
This distinction must remain visible.

Read `CAP/boundary/reference-wave-map-v10-independent-all-three-saved-review-20261009/REPORT.md`.
Index SHA256: `6dda5fd46f74067adf37c671bc4fe1bcc9963a93cebdf02d153f6dc5cf8d235d`.
The earlier v8/v9 radial FAIL and v10 root001 relative-path preparation FAIL
remain intact. Fresh root002 corrects only that metadata path binding.

## Gaussian oracle and incomplete global slicing certificate

The analytic time map is `Y0=T+epsilon*F`, `Yi=Xi`, with
`F=partial_X1 partial_X2[(f(T-R)-f(T+R))/R]` and
`f(T)=sigma^4 exp[-T^2/(2 sigma^2)]`. The inverse Jacobian bound does not
imply that native slices are spacelike. The preserved `a=.5, sigma=.5,
epsilon=.75` future-native example has `D/D_reference=-.25785359...` despite
positive Jacobian. The private `a=2` screen and regional analytic bounds do
not prove the remaining compact box globally positive.

v3 implements independent third jets and geometric identities, with the
source protocol in
`CAP/continuum/manufactured-angular-Gaussian-third-jet-oracle-v3-held-20261009/PLAN.md`.
Its source index is
`41a050d288076d6b54ff396b09851835dac03dd3518f74dca65890f6fd025a8a`.
All 318 units passed in 5.424128584 seconds enclosing time. The 20-record
timing passed in 55.79448625 seconds, reporting 11,816 identity and 1,692
paired-precision checks with zero failures. Independent saved-only review
verified provenance and compact totals without decoding scientific JSONL.

v2's six near-scri conformal-connection failures at 110/150 digits remain
FAIL. v3 uses 180/220 digits with unchanged `1e-55` comparator, formulas and
fixed grids. Higher precision is evidence about oracle conditioning, not a
repair of native arithmetic. The full 5,010 records, 3,330,320 identity rows
and 470,752 precision comparisons have not run. No native queries, exact
binary64 coefficient protocol or raw native `RHS-u_t` binding exists yet.
The required future raw comparison cannot be rescued by paired subtraction.

Read
`CAP/continuum/manufactured-Gaussian-third-jet-v3-timing-independent-saved-review-20261009/REVIEW.md`
and `cost-assessment.json`. Nominal-bin measured runtime scenarios are
10,376.54–11,432.94 seconds; these are empirical estimates, not bounds.
A 25% inflated maximum scenario is 14,286.41 seconds against a 14,400-second
soft cap, with little margin. No nonorigin epsilon-zero event was timed.
Full-stage cost admission is explicitly false; no full launcher was released.

The interval producer v6 reached its 600-second cap with 8,677 visited nodes,
4,331 accepted leaves and 19 pending boxes. Its partial payload is metadata
only and cannot be resumed or treated as a certificate. v7 changes the
Taylor/separated dispatch boundary from `R=sigma` to `R=2 sigma`, coherently
in producer and replay. Source review and Taylor remainder bounds pass,
but its 237 proposed units and both certificate stages are unexecuted.
No v7 root-control prefix was created. Read
`CAP/continuum/manufactured-Gaussian-a2-cache-v7-overlap-held-20261009/PLAN.md`.

## Uncompiled exact arithmetic prototypes

The previously pushed 70-case exact signed-product primitive passed Release
and ASan/UBSan Debug. That bounded primitive cannot recover an inverse or
auxiliary atom rounded upstream. The original complete far-helper registry
still fails 63 metric-direction comparisons over 26 contexts, including eight
exact-zero targets. The diagnostic does not establish those failures as the
cause of the native evolution failures.

The new private backend is
`CAP/continuum/exact-rational-backend-WIP001/exact_dyadic_ratio.hpp`.
It implements exact dyadic operations and a final nearest-even rational
rounding with a 131,072-bit capacity. Its contract covers canonical objects
made by its own factories; public mutable/raw-invalid objects are outside
that proof. Strong exact-range overflow, exact +0, negative underflow -0,
subnormals and explicit capacity failures are deliberate contracts.

Source001's proposed driver has a source-only registry guard bug: decoded
JSON operation lists were compared with regenerated tuples. It never ran.
Fresh source002 changes only canonical JSON structural comparison and its
metadata; arithmetic sources and the proposed 82 controls are unchanged.
Index SHA256:
`ee41e069eee322e93d126699ec5a6251444fc843394c2c8f5d324de8f54ed5e9`.
Read `CAP/continuum/exact-rational-backend-source002-held-20261009/PLAN.md`.
Its independent driver review is capture-only, unfinished, with no PASS.
Root preparation and launcher scripts exist but are unexecuted. No backend
compilation or Fraction registry evaluation occurred. Dependency closures
must be admitted before any probe/oracle execution; unknown headers require
a new reviewed attempt, not automatic baseline expansion.

The whole-row header is
`CAP/continuum/exact-rational-gauge-WIP001/exact_gauge_rows.hpp`, SHA256
`dc5232d2ae8ff0768de98276c07c99b2ce8e82e430b4ebbe6813e9ebaba5fc1c`.
It accepts 106 binary64 value/direction pairs and forms general adjugates,
complete first directions, eight split parts and four independently
assembled rational RHS rows. It rounds each exact result once. It retains
zero-primal/nonzero-direction factors and permits arbitrary reference,
connection and Omega directions. Physical symmetry/SPD admission is a
separate caller requirement; positive Omega is required for this joint API.
Reference physical P must bind the actual consumed `p.k_physical`.

The independent complete-row pencil and source/math review found no blocking
formula issue. Read
`CAP/boundary/reference-wave-map-complete-rational-rows-independent-pencil-20261009/ASSESSMENT.md`
and
`CAP/boundary/exact-rational-gauge-WIP001-independent-source-review-20261009/REVIEW.md`.
The latter index is
`10a3a67f501584cc2a93eee919cbe02d5624faf0dfabff5ac7aafb8c389d6350`.
There is no probe, independent whole-gauge Fraction/Gauss–Jordan oracle,
native adapter, compilation or execution. Existing assembly from rounded
parts cannot supply the intended exact complete RHS; a future interface must
consume original inputs explicitly. Performance cost is unmeasured.

## Assessment priorities and reproduction limits

First audit whether continuing exact arithmetic, the Gaussian oracle and the
finite-matrix route is the best use of effort for the actual stability goal.
Do not assume these three strands are a validated path to that goal. Separate
arithmetic cancellation, continuum gauge/constraint growth and the masked
Cartesian boundary. The original nine native t2 processes had six failures;
the three completed controls do not supply candidate stability acceptance.
No t6/t12 angular run, stable finite disturbance or black-hole transition is
accepted. Exact-scri null/shear/Z4/Theta closure is still unresolved.

If the exact-arithmetic route is retained, complete backend002 driver review
and its fixed Release/Debug controls first. Then construct an independent
whole-row oracle using a different inverse algorithm and literal real gauge
equations, covering arbitrary directions, nonsymmetric algebraic controls,
range failures and exact cancellation. Only after that should it face the
original unchanged 2,373-record/4,900-call far gate and the separate 15,740
local/legacy controls. Their tolerances and seeds must stay fixed.

If the oracle/interval route is retained, run v7's units before a fresh bounded
producer, require a complete producer PASS before replay, and make an explicit
cost decision before the full Gaussian gate. A finite positive sample, an
unfinished interval tree or backward matrix residual is insufficient for a
global theorem or forward spectrum claim.

Original recipes contain absolute paths and platform/compiler/runtime hashes.
The local Python environment was Python 3.9 with NumPy 2, SymPy 1.14,
mpmath 1.3 and pinned SciPy 1.13.1; the candidate runner uses the CLT Python
and literal `clang++` invocation with separately guarded resolved binaries.
Use `-B`, `PYTHONOPTIMIZE=0`, bytecode disabled and single-thread BLAS/VECLIB/OMP
settings. Read each recipe before running anything. Omitted large artifacts
remain in the original local checkout; a remote-only assessor can inspect
source/evidence but must regenerate separately named inputs to reproduce
affected stages. Do not execute scripts from an archive assuming portable
paths, edit frozen records or overwrite previous attempt directories.

Production equations did not change, so production regressions were not
repeated for this checkpoint. Wrap-up verification covers collection hashes,
finite JSON, staged byte equality, binary/large-payload exclusion and the
unchanged production tree. No new scientific or evolution test was performed
solely to produce this handoff.
