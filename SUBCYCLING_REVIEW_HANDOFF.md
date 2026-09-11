# VC Cartoon subcycling: independent-review handoff

PAUSED, 2026-09-11. **Production subcycling has not accepted its first interval.
No speedup has been established.** Do not launch jobs, change tolerances or
budgets, or resume implementation pending review. Production remains untouched.

## Revision identity

Repository: git@github.com:HengruiZhu99/athenak.git
Branch: codex/vc-cartoon-subcycling-20260911
Reviewed implementation full SHA: c5950ce2b19436f03a86edada43df747469fd8e2

The final handoff commit is the commit containing this document; resolve it with
git log -1 --format=%H -- SUBCYCLING_REVIEW_HANDOFF.md in the delivered checkout.
Its full SHA is supplied in the delivery message and verified against the remote.
A commit cannot embed its own hash without changing that hash. The implementation
SHA above is immutable; the final commit adds this document only.

Relevant baselines:
- Branch start: c930074d95d63b9345b097806effca2f2dc008b5.
- Historical production source: c802bcaf77f143cf3e83ce267cc93d5f59f2daa7.
- Native physical-corner correction: 0d18e162b722b9b2dd926e50089eabc5d54eca4d.
- Actual-stage global gauge fix: 884f51702e825a8b79c5cfb0367f25626338a830.
- Output cadence fix: a26f002f47a205b4e1a72323721aac6da3a08ede.
- Corrector tracing: fb3d341c7e49ecd8ba4dbbafcb3877c05a7c9f6f.

Evidence paths below are relative to
[docs/investigations/subcycling_20260911](docs/investigations/subcycling_20260911/).
WORK_LOG.md preserves the chronological development and earlier failures.

## Architecture and qualification gaps

The opt-in Driver path uses classical RK4 and a reusable
SynchronizedHierarchyEvolution owner. Populated covered parents and physical
leaves have independent storage. HierarchyRK4 predicts parent histories,
recursively advances finer levels, restricts children and iterates coupled
endpoint, retained RK histories and global-gauge history. Dense temporal boundary
data are stage consistent; shared vertices synchronize at common times.
Native fields are published only after acceptance. ADM reconstruction,
constraints, stopping, real AMR and output stay in the native driver.
Topology changes rebuild the hierarchy owner.

Native and hierarchy timestep physics share spatial/source ceilings, including
covered levels. Stage tightening causes rollback and interval halving.
Synchronous groups use the actual stage physical-leaf maximum of |K|;
asynchronous coupling uses common-time gauge prediction. Axis regularity,
physical boundaries and hanging vertices must remain consistent at every stage.

Main files: src/driver/driver.cpp, src/driver/hierarchy_rk4.hpp,
src/driver/corrector_control.hpp, src/z4c/live_hierarchy_evolution.hpp,
src/z4c/synchronized_hierarchy_evolution.hpp, src/z4c/hierarchy_physics.hpp,
src/z4c/checkpoint_subcycle_probe.hpp. Helpers and tests accompany these;
reproducible harnesses are under scripts/subcycling.

Qualification is limited to single-rank VC Cartoon fixtures. CC, generic3D,
multi-rank execution, long strong-field evolution and production speedup remain
unqualified. In the live subcycled path a cycle means a synchronization cycle,
so real AMR each cycle has a different physical cadence: an unresolved comparison
effect. Classical RK4 differs from production low-storage rk4. The corner fix
is an intentional native baseline numerical change.

Production trials: maximum ratio16, interval cap4e-5, corrector min3/max8 passes,
absolute tolerance1e-12, relative tolerance1e-10, at most8 retry halvings.
Normalized endpoint/history/gauge residuals must each be <=1.

**Unfinished last change:** the reviewed implementation preserves the already
written live subcycle_corrector_max_passes option (default8, bounds3..64)
and test-harness plumbing. CPU compilation passed, but this change has NO runtime
test and was NOT deployed remotely. No larger-budget trial ran. It is not a
demonstrated fix. No algorithm/tolerance/budget edits occurred during handoff.

## Exact production target

Campaign:
 /pscratch/sd/h/hzhu/n128-chite-bisection-20260910/cycle_03_recovery_24000

N128 A=-0.048875 is only provisional collapse under the resource-time rule,
not validated physical collapse. Settings: half-rho mesh64x128, blocks16x16,
spatial order4, prolongation6, nghost4, extrapolation2, CFL0.25, dissipation0.5,
chiTE0.001, reference_nx1=64, max_nmb_per_rank24000. Telegraph tau=kappa=0.01
with global max|K|, zero shift, no damping, full_constraint_bjorhus boundary.
Use the exact input rather than reconstructing it from this summary.

Checkpoint:
 /pscratch/sd/h/hzhu/n128-chite-bisection-20260910/cycle_03_recovery_24000/rst/lapse200.00066.rst

SHA256: 4b4dc1f576fe2b88d00c72de17bc413bf5bfa0ccbfce4ef36e2980a6cd979b4b
Time62.205560322980219, cycle133489, 8924 blocks,
dt2.422989058508578e-6, 1,115,701,237 bytes.

Campaign files and SHA256:
- input.athinput: e9a00f671ccdab33ef1ae042c4ae6e29b8365036f87d99d66283f2532f4fb12f
- initial.coefficients: 6e619d81f821c6d0bc0a8ab91b5cfe3700b79e28d3168c16736bfe97bf14e6fb
- Production executable, recorded provenance:
  fd859ee500d8c20437af81aa4b25be4957714c3d9c134ad5eb4fcbd39f983a2b.
  Bundle: /pscratch/sd/h/hzhu/n128-chite-bisection-20260910/bundle.
  This is archived provenance, not a fresh rebuild verification. Historical
  source SHA alone does not prove absence of applied patches.
- Required first90925809 bytes of amr_history.jsonl:
  1dfba16e1aefaeda6e17d613aedf58bd1d925f59bd8622da7452ee426dae2fd0.
  Restart embeds an absolute production history path: redirect to a private
  copied prefix and preserve source compatibility. Never append production.

Remote experiment root: /pscratch/sd/h/hzhu/vc-subcycling-20260911
- brill-endpoint-a26f002f/{legacy_sync,classical_sync,subcycled}, job58211022.
- corrector-probe-fb3d341c, job58211844.
- brill-endpoint-884f5170, job58210606: invalid output-disabled comparison.

Exact derived input SHA256 (case directory/input.athinput):
- legacy_sync: 46394fee4edb95ab150a488b57d75f7bcb8147c8f4b3f21c504165463de0b272
- classical_sync: 7119d41d38746c3c0d0ea58a97568f0b3d577678848f8128899e5cca7728c36a
- subcycled: d348b49ffd2304215a710da732d2401ff09550fd36b55752bd722c0840b3c8c1

- Instrumented input: 27fe7dc02809abc6b22f801e5a0514466fdee07f28cd1466ccd0accc58be8f8b.
  A fresh directory changes the absolute AMR path and therefore input hash;
  preparation records its new hash.

## Executable provenance

Test binaries embed base38c1fab49565794bb1ab6b7777b806e440eac07a.
**The embedded SHA alone does not identify either executable.**
Applied patches, in order:
1. vc-live-884f5170.patch: live/gauge/probe corrections and harness.
2. vc-output-cadence.patch: explicit cadence selection.
3. vc-corrector-trace.patch: instrumentation (traced binary only).

Exact patches, source diff/status and production provenance are committed under
review-provenance/. Patch SHA256:
- vc-corrector-trace.patch: 1e396bf000fdbd6c5d0ee9a6015fff234594a39294da42174ed525fe91997ad2
- vc-live-884f5170.patch: 16c9ab8bb261c3824603be3b1f7957c28583a7c51e671a381f21e272f071421a
- vc-output-cadence.patch: df0670a34b2974c29690297b0d0f86ba5d7855802296d4de4a6a8d96f6db2e14

- Controls and uninstrumented failure:
  /pscratch/sd/h/hzhu/vc-subcycling-20260911/pinned-a26f002f/athena
  SHA256: 034b7091bed84033d45119c48874fdfc7d14c537cecb29e0e77b9465db8d29e3,
  base plus patches1,2.
- Instrumented failure:
  /pscratch/sd/h/hzhu/vc-subcycling-20260911/pinned-corrector-trace/athena
  SHA256: d3e95c8a10aad08b70a26caaacf5f9534dd4216f40e5035b58f9c6d777a5bc2b,
  base plus patches1,2,3.
- Additional remote untracked files are analysis/test scripts recorded in
  remote-source-state.txt, not additional compiled numerical patches.
- Final reviewed source is newer than these pinned binaries: it includes the
  compile-only pass option. Do not equate the binaries with final branch HEAD.

## Tests: results and limits

| Evidence | Result and meaning |
|---|---|
| Manufactured/classical RK4 tests | Fourth-order temporal convergence on controlled fixtures, not strong-field qualification. |
| production-gauge-evidence / stage-gauge-evidence | Actual stage maximum of absolute K fix resolves larger-step mismatch. Fixed-duration error ratios approximately16.13 sync /16.09 subcycled on smooth small fixtures. |
| live-driver-evidence / live-amr-evidence | Live/frozen agreement at roundoff, exact split restart; real refine/coarsen and mixed topology changes including same-count changes tested. |
| Stage-limit/retry tests | Tightened source ceiling rolls back .01/.005, accepts .0025 matching direct result; not proof of nonlinear history convergence. |
| gpu-befad-evidence, jobs58207979/58208872 | Boundary, parent, RK, gauge and rollback helpers pass after stale oracle/CPU mirror-alias test fixes. |
| live-gpu-884-evidence, job58210242 | GPU live/restart and real/mixed AMR pass; production-gauge temporal ratio16.090291. |
| output-cadence-evidence / brill-cadence-rerun | Explicit time cadence writes four expected checkpoints despite inherited dcycle; legacy default preserved. |
| corrector-trace-evidence | Six CPU endpoint field hashes identical with instrumentation; production traced run FAILS below. |
| review-provenance/pass-budget-compile.log | Last option compiles only; no runtime qualification. |

Preserved negative evidence includes prior boundary oracle failures, tiny-dt
order-test failure at roundoff (~1e-16), resolved larger-step gauge mismatch,
cancelled output-disabled run, and both production corrector failures.
Do not represent the development history as an all-green suite.

Matched controls cover only t62.205560322980219 to62.205994062314062,
about179 original steps, NOT the full eight-hour run.
Legacy rk4 exactly matches stored archived endpoint:
minLapse0.017361930148278765, maxAbsKret259724.66624038288,
C-norm2 131.3371573622191, axisLapse0.510622988950541, blocks9155,
maximum physical refinement18.
Classical sync relative differences: minLapse3.84e-13,
maxAbsKret-9.43e-10, C-norm2 1.32e-12, axisLapse2.50e-13; mesh counts agree.
Wall times including initialization/output:351.324s legacy,349.835s classical.
These are controls, not a subcycling speedup. High constraints also prevent
equating reproduction with physical validity.

## Instrumented first-interval failure

It DID RUN: job58211844. Full stderr contains retries and per-pass diagnostics;
trace.json contains extracted residuals, and manifests record all hashes in
corrector-trace-evidence/brill. First attempt hits a stage-stability ceiling.
Subsequent attempts hit the corrector pass limit; eight halvings exhaust retries.
Residual columns are normalized endpoint / retained history / gauge feedback,
each requiring <=1.

| Attempt | dt | Last recorded outcome/residuals |
|---:|---:|---|
| 1 | 3.3088853306340469e-05 | stage-stability ceiling, no corrector residual |
| 2 | 1.6544426653170235e-05 | pass8: 4653.35781 / 352646390 / 0 |
| 3 | 8.2722133265851173e-06 | pass8: 27.3183758 / 16227621.8 / 0 |
| 4 | 4.1361066632925586e-06 | pass8: 0.186652819 / 775852.324 / 0 |
| 5 | 2.0680533316462793e-06 | pass8: 0.00324097498 / 53206.839 / 0 |
| 6 | 1.0340266658231397e-06 | pass8: 0.0023998089 / 3298.67872 / 0 |
| 7 | 5.1701333291156983e-07 | pass8: 0.000237506463 / 266.949646 / 0 |
| 8 | 2.5850666645578491e-07 | pass8: 6.70659502e-05 / 36.4283616 / 0 |
| 9 | 1.2925333322789246e-07 | pass8: 0.00011315852 / 2.1999596 / 0 |


Final pass8: endpoint0.00011315852023589828, history2.1999595957891334,
gauge0, worst history logical level18/tick8. Final-attempt history values at
passes2..8:18230.9,7659.86,2353.32,246.013,53.5257,18.9099,2.19996.
Endpoint/gauge alone would misleadingly pass. This identifies the immediate
blocker, not its ultimate mathematical cause. Extra passes are neither tested
nor justified as a fix. No accepted interval or subcycled endpoint exists.

## Reproduction commands for later review

NOT executed during this handoff. Use the pinned traced binary for the observed
failure, not an unqualified rebuild of HEAD. Execute on Perlmutter from the
delivered source checkout. The helper verifies the checkpoint and prior input,
copies only the required history prefix, redirects its path to a fresh directory
and limits nlim to one synchronization interval. It imports the campaign's
workflow_common.py setparam helper. Production is read only.

    root=/pscratch/sd/h/hzhu/vc-subcycling-20260911
    fresh="$root/review-first-interval-$(date -u +%Y%m%dT%H%M%SZ)"
    python3 scripts/subcycling/prepare_corrector_probe.py "$root/brill-endpoint-a26f002f/manifest.json" --exe "$root/pinned-corrector-trace/athena" --output "$fresh"
    # Inspect fresh/manifest.json and compare the checkpoint/binary/prefix hashes.
    export root fresh
    salloc -A m3328_g -q shared_interactive -C 'gpu&hbm80g' --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus=1 --time=00:10:00
    module load PrgEnv-gnu cudatoolkit cmake cray-hdf5
    export MPICH_GPU_SUPPORT_ENABLED=1 MPICH_GPU_IPC_ENABLED=0 OMP_NUM_THREADS=1 ATHENA_SUBCYCLE_TRACE=1
    cd "$fresh"
    srun --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus=1 --gpu-bind=single:1 --cpu-bind=cores --exact --kill-on-bad-exit=1 "$root/pinned-corrector-trace/athena" -r /pscratch/sd/h/hzhu/n128-chite-bisection-20260910/cycle_03_recovery_24000/rst/lapse200.00066.rst -i "$fresh/input.athinput" -t 00:08:00 > stdout.log 2> stderr.log
    rc=$?
    printf '%s\n' "$rc" > exit-status

Expected historical outcome:134, CorrectorFailure before acceptance. Preserve
stderr. Original run-corrector-probe.sh has the exact launcher/hash checks but
names the OLD directory: do not execute it unchanged. Fresh preparation refuses
existing directories. No large outputs/checkpoints are committed; reproduction
requires access to the Perlmutter artifacts.

## Paused state and review priorities

Task-owned remote build PIDs710534,880731,959300,1051167 absent; trace build status0.
Local CPU build completed and no build process remains.
Jobs58210242 COMPLETED0 (1m26s),58210606 CANCELLED (4m45s),
58211022 FAILED1 (13m44s),58211844 FAILED134 (1m57s).
No task-owned job needed cancellation. Unrelated58207074 corner-continue was
RUNNING and left untouched. No jobs or simulations launched during handoff;
production campaign/settings/monitors unchanged.

Review history fixed-point coupling, stage consistency, comparison of iterates
at identical physical times, norm scaling, covered parents/hanging vertices,
global gauge feedback and full state restoration on retry. Assess whether
smooth tests can reveal this failure, and what independent evidence could
justify any change. Only after correctness, examine AMR cadence and repeated
full-hierarchy passes for an honest performance comparison.

Longer/earlier intervals, strong-field convergence, multi-rank behavior and useful
speedup remain unfinished. **Stop pending independent review.**
