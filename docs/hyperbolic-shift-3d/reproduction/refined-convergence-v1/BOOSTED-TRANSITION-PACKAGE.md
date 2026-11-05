# Prepared boosted transition campaign

This records package preparation, not an evolution result. Aurora access is
restored:9038758 was verified at a healthy2.017625M walltime checkpoint and
continued as9040850. The user now prioritizes the boosted family after the
currently allocated stationary segment finishes. See state.json for build and
submission status; never repeat a successful build or ambiguous submission.

The only upload candidate is
`output/refined-convergence/boosted-transition-package-v2`. The earlier local
`boosted-transition-package-v1` is superseded: it assumed the local configuration
identity would be identical on Aurora. Package v2 has now been uploaded and its hashes checked; v1 was never uploaded.
Actual build disposition is recorded separately in state.json.

## Scientific binding

Source commit `d98e8d9e7f6131a4af55afe96ef7589a73d07b9e` contains the already
qualified full400-leaf SMR tube, radial24/angular2 shell configuration, mixed
angular/radial donor repair and fixed-topology buffer retirement. The input is
`central20_boosted_smr_tube_radial24_angular2_reference.cfg`, SHA256
`57d0d28cdd7833aabaf2ef91445692c850cdc5ce0b3d1262f797c173d06fa753`.
It retains exact global v=0.1 boosted geometry, the same conformal factor and
coherent reference gauge. See FULL-BOOSTED-SMR.md for measured local evidence
and its two-step limitation. No new equation change is introduced by packaging.

Source manifest SHA256:
`50037c604c81ce37c5465da2e08c6082dc54d198eeb941654d366e12f75b554e`.
All2759 source/Kokkos manifest files and18 package files were hash/length checked;
archive membership is exact, with no AppleDouble metadata. Python syntax was
checked. Records are in `boosted-transition-preparation/`. These checks are not
an end-to-end launcher or GPU validation.

## Exact compiler-dependent identity

The previously measured Aurora compiler rounds the map scale k and k-squared
inner coefficient one unit differently from the local compiler. The new builder
therefore runs a host-only configuration utility after compiling the production
target, before freezing science and launcher records. It uses that target's
actual compile database and link flags. It performs no MPI/Kokkos initialization,
GPU allocation or evolution step.

The utility computes identities for N12/14/16, requires their exact agreement,
and accepts only the qualified local string or the explicitly recorded Aurora
roundoff form. Unexpected drift fails before freeze. The resulting identity is
frozen and exact checkpoint matching remains mandatory. The old calibration
is evidence of compiler roundoff, not admission of an old failed state.

Local validation matched all three native identities exactly. The existing
local build lacked a compile database, so this validation explicitly used its
flags.make and link recipe. Aurora requires its production compile database;
the local fallback is not enabled there. The first local invocation stopped
before compilation because that database was absent. No physics rerun occurred.

## Execution after access and scheduler reconciliation

Planned remote namespace:
`/lus/flare/projects/CompactBinaryMerger/hzhu/refined-convergence-20261104-v1/boosted-transition-v1`.
Check that it has no prior build/submission before upload; preserve all existing
scientific snapshots. Verify transferred package hashes against
`package-records.json`, then run its build.sh once. It uses make-j64 and freezes
actual binary hashes only after successful compilation and identity resolution.
Do not repeat a successful build, freeze or GPU physics smoke.

After preserving the completed build records and verifying every launcher
manifest entry, use explicit `/usr/bin/python3.11` on Aurora:

```sh
python3.11 launcher-boosted-transition-v1/plan.py ROOT --n 12 --segment 1
python3.11 launcher-boosted-transition-v1/submit_reviewed.py ROOT PLAN
python3.11 launcher-boosted-transition-v1/control.py collect ROOT PLAN
```

`submit_reviewed.py` is the sole submit entry point. It checks all user jobs and
both debug queues, writes intent before qsub and retains ambiguous intent for
manual reconciliation. Four debug-scaling nodes,48 ranks, CompactBinaryMerger;
fresh native data, dt=1/(2000N), target10/3, diagnostics every.025 coordinate time.
No plans or jobs have yet been generated for this package. The planner requires
actual completed build/science records, so no fictitious binary hash is needed.

Healthy admitted same-case walltime continuations use `--segment NEXT
--previous PLAN` after matching terminal lock release. The parent terminal
verification, plan, release and checkpoint are bound and reverified. The RF2
checkpoint carries topology, tracker and horizon seed. Failed or unadmitted
states never restart. Preserve all artifact hashes/lengths locally before any
retirement; inspect scheduler/native/worker disposition together.

## Storage and remaining evidence

The unchanged50GB guard accounts for all prior roots and counts each inode
once, using max(logical,allocated) bytes. Estimated checkpoint sizes include
64MB slack over cubic scaling of the measured N12 checkpoint; these estimates
are budgets, not asserted checkpoint schemas.

| N | Planned output bytes | Atomic reserve bytes | Sum at binding |
|---|---:|---:|---:|
|12|2,298,339,504|1,085,169,752|3,383,509,256|
|14|3,423,900,232|1,647,950,116|5,071,850,348|
|16|4,921,693,640|2,396,846,820|7,318,540,460|

The existing worker conservatively retains those reservations while scanning
growing output. Before starting, allow for actual new outputs in addition to
the reservations (up to2*planned+atomic relative to the pre-output footprint).
Retire only obsolete outputs after fresh local and remote verification and a
receipt; never weaken the guard or delete an active restart. A build adds its
own reserved space and must also fit.

The short local control gives no10M stability or evolved convergence result.
The user now prioritizes this boosted resolution family, preserving
fixed-domain constraints, horizon acceptance/mass gaps,
tracker and SMR support margins, scri residuals, floor/sign accounting, radial
and angular profiles and cost. Compare actual elapsed times before claiming a
performance gain. Fixed-topology memory retirement is qualified locally;
dynamic AMR behavior and the active stationary Aurora binary are unchanged.
