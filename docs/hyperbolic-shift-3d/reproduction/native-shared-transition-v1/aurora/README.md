# Frozen N16 GPU comparison

See ../EVOLUTION-MATRIX.md for scientific attribution and ../STATUS.md for the
measured consistency qualifications. These are five controlled N16 evolutions,
not a resolution convergence demonstration. No long N24/N32 runs are authorized.

The source package is exactly commit
`fd4652ff69d55ec973b61029b4e77609e49f0860`, including pinned Kokkos
`6739bc623081648af9e752b616d9671527922cbf`. The separate launcher comes from
`d446f75c6505b47f26f8b4beabada722da423552` (launcher v2). `provenance.json` binds every
launcher file and scientific manifest. The source manifest is
`source-native-transition-v1-manifest.json`, SHA256
`7bd61dc86efe172c73e6363e237b36cdb36160eddb986110b75f0c78a46f36f0`.
Production initialization is native C++ at t=0; no external initial-data files.

Remote root remains `/home/hzhu/hyperbolic-core-shell-20261028`. Preparation
packs the committed source and launcher with `native_transition_prepare.py stage`.
The explicit build wrapper loads oneAPI2025.3.1, configures SYCL/PVC+MPI in
`build-sycl-mpi-native-transition-v1`, and uses `make -j64` for all five evolution
targets and five matching checkpoint diagnostics. Source/launcher checks run
before and after compilation. Build logs preserve compiler/MPI identity and
command outcomes. Storage accounting is polled during compilation, reserving
500MB and rejecting a start without 1GB additional planned space under the5GB
task cap. A stopped/partial build is preserved for diagnosis, never blindly retried.

After successful compilation, freeze verifies CMake configuration, all ten
targets' native/direct-variable and variant definitions, and hashes binaries.
It creates `launcher-native-transition-v1/native-transition-science.json` and
final `provenance.json`. Never invent or reuse binary hashes from local builds.

| Case | Restoration | Dissipation | Scri first derivatives | Schema |
|---|---|---|---|---|
| A | Original collar | KO8 | Original | NI1 |
| B | Shared W once | KO8 | Original | NT1 |
| C | Original collar | KO10 | Original | NI2 |
| D | Original collar | KO10 | first6 | NC2 |
| E | Shared W once | KO10 | first6 | NS2 |

Each case first needs its two-step GPU smoke (Minkowski then puncture), followed
by its fresh puncture evolution to10M. Run cases sequentially A→B→C→D→E.
The helper only creates plans and verifies; it never submits. For each case:

```sh
python3.11 launcher-native-transition-v2/native_transition_campaign.py plan --root "$PWD" --variant A --purpose smoke
python3.11 launcher-native-transition-v2/control.py bind --root "$PWD" --plan campaigns/n16-native-transition-v1-a-smoke-segment01/plan.json --source-manifest source-native-transition-v1-manifest.json
qselect -u hzhu -q debug
# Require no other user debug job, then submit once with control.py submit.
```

Inspect saved submission/worker/native/scheduler state before every action. A
queue or running state is not permission to submit another copy. Use
CompactBinaryMerger/debug or debug-scaling, one node,12 MPI ranks/GPU tiles. Smoke verification
requires both cases completed and admitted. Only then prepare/bind/submit the
same variant with `--purpose evolution`. All start fresh at t=0, direct chi/lapse,
dt=.0005, physical lapse floor1e-10, positive chi floor1e-8 at r<.03, nonpositive
chi rescue everywhere, original damping/penalties, horizon lmax12/ntheta24,
tolerance.001 and diagnostics.05. Floor candidates/signs remain recorded.

On a terminal healthy walltime stop, collect with the frozen controller, then
run `native_transition_campaign.py verify --root ... --plan ...`. Require exact
schema/configuration, admitted finite physical payload, time/steps/progress,
source/binary/input hashes and matching scheduler/worker/native dispositions.
Healthy verification writes `checkpoint-verified.json`. Release only the matching
terminal root lock with `control.py collect --release-terminal-lock`. Generate
the next same-case plan with `--segment NEXT --restart CHECKPOINT --sha256 HASH`;
the helper includes any horizon-seed sidecar. Bind unchanged science and check
all user debug and debug-scaling jobs before submitting. Remote verification can precede local
collection, but no purge can precede local length/SHA256 preservation.

On numerical failure collect and localize it; never restart its failed state or
change dt/equations/floors automatically. Preserve the failed case's evidence and
proceed to the next independent authorized case after its smoke. Infrastructure,
storage or integrity failures need diagnosis before any retry. Keep each case's
scientific identity immutable. The two old SG1 N32 checkpoints are now fully
preserved locally and their remote retirement is recorded separately.

At disposition compare measured histories with SG1 and between controls;
reconstruct only admitted matched endpoints with the matching diagnostic target
and one common cutoff. Preserve missing horizon masses, sum segment-local sign
and floor counters, keep boundary-interpolation counts separate, report scri
null/regularity residuals and radial errors especially.775–.9. Long-run survival
does not override the previously measured initial underresolution or scri seam
pointwise-order qualifications.

Launcher validation:64 tests pass, including every schema, corrupted payloads,
changed frozen inputs, wrong compiler/backend/variant flags, unsuccessful build
records and terminal scheduler mismatches. Actual local NS2 smoke checkpoint
inspection also passes. These tests are not GPU evolution results.

Build attempt01 stopped at CMake generation: the package omitted the required
`tst/elliptic_shift_1d/spherical_api.cpp` support source. No target compiled or
ran. The corrected package includes that file from the SAME scientific commit;
local clean configuration now passes. The old package/build/logs remain under
`retired-native-transition-build-attempt01` on Aurora; attempt01 provenance and
configuration evidence are preserved here. This is a packaging repair, not a
scientific change.

## Analysis and backend controls

`../serial_smokes.py` records two-step SERIAL/MPI-one references against the
previously bound local binary hashes. Cases A–D each passed Minkowski and
puncture (eight executions); case E's prior combined-control smoke is preserved
in checks/scri-full. These are local results, not Aurora validation.
`../compare_smokes.py` compares admitted two-step physical fields against the
12-rank SYCL outputs, separately reporting compensated-RK bookkeeping. It reports
per-field differences normalized by1+maximum field magnitude; the1e-10 parity
threshold checks backend agreement, not physical accuracy.

`../compare_histories.py` accepts repeated `--case A=plan01,plan02` arguments,
`--root` for the collected native mirror, `--baseline-root` for preserved SG1,
and a fresh `--output`. It verifies artifact/science identities and case/segment
continuity, retains numerical-failure histories without admitting failed
checkpoints, preserves accepted-mass gaps and unavailable boundary diagnostics,
and sums segment-local sign/floor counters separately from boundary counters.
Its own-candidate history plots do not replace the required common-cutoff
endpoint radial reconstruction with each matching diagnostic executable. Four
synthetic validation tests cover changed scientific settings, wrong scheduler
state, out-of-range histories, missing smoke cases and physical disagreement.

## First GPU submission

Build attempt02 completed successfully; both configure/build commands returned0
and all ten targets passed freeze checks. See build-result.json, provenance.json
and native-transition-science.json for actual hashes. Footprint after build was
3,148,973,629bytes. CaseA smoke job9031714 is submitted, plan
`campaigns/n16-native-transition-v1-a-smoke-segment01/plan.json`, SHA256
`f8ce85176041faa30396348adff831f32e2d3951a57c0bc403b9aae87d82177e`.
No user debug/debug-scaling jobs were present before submission. Binding reserved
217,285,583bytes outputs plus89,725,224bytes atomic space, leaving1.544GB headroom.
Submission does not establish GPU correctness; native reports/checkpoints and
SERIAL comparison remain to collect. See state.json/current remote state before
any further action. Never submit this plan again.

## CaseA GPU smoke result

Job9031714 completed with PBS exit0 and worker passed. Both native NI1
checkpoints are admitted and passed exact configuration/finite-payload/hash
checks remotely and locally. All9 recorded artifacts are locally preserved.
The Minkowski/puncture physical-state comparisons with the bound SERIAL
references have largest scaled differences9.19e-14/1.34e-13, respectively;
RK-compensation differences are below4.45e-16. Null surface/ghost maxima are
at most1.12e-15. See a-smoke-verified/, a-smoke-local-verification.json and
a-smoke-parity.json. This validates the native-initializer control on12GPU
ranks for TWO steps, not the later repair variants or long-time accuracy.

Metadata qualification: frozen coupled_mpi.cpp retains legacy static
`scri_angular_stencil` and `scri_normal_bulk_derivative_order` strings/numbers.
For casesD/E, the explicit `scri_first6` flag, unique checkpoint schema and
bound compilation definitions identify the first6 repair. The legacy baseline
labels do not override the source-defined outer-shell/surface stencils. No
scientific executable is modified mid-campaign to change those labels.

CaseA fresh10M evolution is now submitted as job9031739, plan
`campaigns/n16-native-transition-v1-a-evolution-segment01/plan.json`, SHA256
`0926118fd24a73633d7d846db174bc46a6d9d43a3f058a1be8cc40dd1fe0f6fc`.
It starts from native wormhole data at t=0, not from the smoke checkpoint. The
smoke terminal lock was released after verification; both user debug queues
were checked before submission. No long-time result yet.


## Current replacement and automatic continuation

Job9031739 was cancelled while still queued; it ran no evolution. The old worker selected30seconds rather than the configured180seconds for the new phase. Cancellation and terminal lock release are recorded in a-evolution-cancelled-before-start/. Launcher v2 fixes this and permits one-node debug-scaling. Its provenance is launcher-v2-provenance.json; use the v2 controller/helper for all further work. The helper intentionally still binds the unchanged `launcher-native-transition-v1/native-transition-science.json`. No scientific rebuild or parameter change occurred.

Replacement job9031760 is submitted on debug-scaling with one node/12 ranks. Plan `campaigns/n16-native-transition-v2-a-evolution-segment01/plan.json`, SHA256 `c91f0ae67f647188f217002d43c9a8455493536ee7c5ce6abebb6245e6a5eb53`. It differs only in output prefix/queue; the corrected worker supplies2920seconds native time within the3100second command. Latest scheduler check: queued, node offline. Follow saved state/submissions before any action.

The existing monitor is ACTIVE every20minutes and automatically submits healthy same-case checkpoint continuations until10M, then advances the authorized case matrix. Numerical failures require evidence preservation/localization and cannot resume a failed checkpoint. No further routine launcher tests are needed: prioritize actual evolution histories, common-cutoff regional errors and the high-order convergence objective. These five N16 attribution controls alone cannot demonstrate resolution convergence.


## Native initializer control completed10M; shared-transition GPU trial submitted

CaseA job9031760 completed10M/6667steps, with native return0, worker passed and PBS exit0. Frozen endpoint verification passed (NI1,168-byte header,scheme3); all six worker artifacts were copied and hash/length-verified locally. Checkpoint SHA256 ab26ed9fce097b965cb19217f7599efb47c8c119df39ea49dc4d536d5e77cddb. No remote purge was needed. Terminal lock released.

Measured own-candidate exterior H/M/Mr histories nearly overlay SG1: maximum relative changes2.20e-5/4.52e-6/5.80e-6. At10M H=.0154149684,M=.0290076414; accepted horizon mass error+.3413204% with45/68rows accepted. Negative chi stage/final counts56/8 match SG1; no lapse, surface or boundary sign interventions. Surface/ghost null maxima remain below1.56e-15, while pole/clock maxima are.0281679363/.0220012275. Native initialization alone does not materially improve the evolved exterior errors. See a-versus-sg1 for inspected history plots and counters. Independent common-cutoff regional endpoint reconstruction remains pending; these history norms use each run’s candidate cutoff.

CaseB two-step GPU physics smoke is submitted as job9031861 on debug-scaling,one node/12ranks. Plan campaigns/n16-native-transition-v1-b-smoke-segment01/plan.json, SHA256 b6829754eafaac006479a41492305f5053ff212aef9d46a48a5aaf46d1de0502. Use launcher v2 despite folder spelling. After verified smoke/parity, start fresh shared-W/KO8/original-scri N16 to10M; no caseB long run submitted yet. The20-minute monitor was updated to this job and will automatically continue admitted healthy walltime states.


## Shared-transition fresh10M run submitted

B smoke job9031861 passed both two-step GPU cases. Exact NT1 checkpoint and artifact checks passed; all9 worker artifacts were locally hash/length verified. Maximum physical SERIAL/GPU scaled differences were9.19e-14 (Minkowski) and1.34e-13 (puncture). This is backend/partition agreement, not long evolution accuracy. Its terminal lock was released.

Fresh native shared-W/KO8/original-scri N16 evolution is submitted as job9031894 on debug-scaling,one node/12ranks. Plan campaigns/n16-native-transition-v1-b-evolution-segment01/plan.json, SHA2563ce975744577a0727794d852c9deacb4910295e3b395aa2b68b4d5667f632073. Use frozen launcher v2. Science,dt,floors and diagnostics remain matched; no smoke restart. The20-minute monitor now follows this job and automatically continues verified healthy checkpoints when necessary. C/D/E remain unsubmitted.

A-versus-SG1 common-cutoff reconstruction is now complete at r>.1099 using the matching NI1 diagnostic. Reconstructed full-domain norms agree with the native report. Exterior H=.0154149683973,M=.0290076414088, essentially unchanged from SG1. The outer.775–.9 still holds62.197609%/75.024155% of exterior H²/M²; .16–.30 holds23.442777%/8.975670%. This independently confirms the initializer alone does not remove the evolved outer error. Diagnostic command/binary/checkpoint hashes, radial data, comparison and reproducible plot script are in a-versus-sg1.


## Shared restoration with original numerics failed; KO10 control submitted

B job9031894 failed at last accepted9.391500000000638M/6261steps; last recorded diagnostics are9.3M. Device reason2 is invalid native geometry/source, at global active cell57352. All six worker artifacts, including the rejected-stage capture, are locally hash/length verified. The checkpoint is explicitly unadmitted; no restart is permitted. Terminal lock released.

The rejected volume contains eight finite but indefinite conformal metrics, all at r=.609375 in the shell, with minimum eigenvalue−1.9128613e-4. At the first point(x=.3748364114,y=z=−.3397319203), eigenvalues are approximately(−.000191286,.726199,439.502), determinant−.061052, chi9.77581 and conformal lapse.00578966. No volume values are nonfinite; the captured surface metric remains positive(minimum eigenvalue.28425). This is not a lapse/chi sign-trigger termination or failure to enforce the algebraic null condition. It localizes the immediate invalid geometry; it does not determine the cause of its preceding growth. The rejected RK stage abscissa was not recorded and is not inferred.

Scri surface/ghost null maxima remained below1.56e-15, but pole/clock maxima grew to.7126948/.1148995. Candidate-exterior errors dropped around6–8M and then grew; last9.3M H=.02819945,M=.06584202. These use evolving candidate cutoffs; no matched10M endpoint or convergence claim is available. Accepted mass gaps remain. Chi floor stage/final totals112/32 differ from negative-candidate counts104/32 (positive below-floor interventions also count). Events remain at the nearest eight puncture cells, spatially separate from the failed shell. No lapse/surface/boundary sign interventions. Comparison plots are in ab-versus-sg1; failure evidence/localization is in b-failure*.

C GPU physics smoke job9031971 is submitted on debug-scaling,one node/12ranks, plan campaigns/n16-native-transition-v1-c-smoke-segment01/plan.json SHA256980b9bdb1275cbcd5405ad5bd9c9d705eb32f048119e7a6392387c6e1a914f14. This independent native original-collar/KO10/original-scri control retains matched settings. Verify smoke/parity before fresh C evolution to10M; do not resume B. The20-minute monitor now follows C. No remote evidence was purged.


## KO10 distributed donor repair (2026-11-02)

Original C smoke job9031971 completed its two steps, but the required physical-state comparison failed: maximum per-field scaled SERIAL/GPU differences were 5.0841e-4 (Minkowski) and 2.1209e-4 (puncture). All nine artifacts are locally hash/length verified and preserved; the terminal lock is released. No old C long run was launched.

The cause is in selective MPI Cartesian donor requests: the communication plan requested offsets −4 through +4 even for the KO10 operator, whose B5-transpose W B5 composition reaches ±5. Consequently partitioning the Cartesian leaves across ranks omitted required fifth donors. Old local 12-rank Minkowski reproduces old GPU12 to 9.19e-14 while differing from one rank by 5.08e-4. This isolates a communication dependency defect rather than GPU arithmetic. The two-rank layout kept the core together and did not expose it.

Scientific fix e7d55b008c86774580386dbc3e13028fa72b7e24 derives request reach from CompositeDissipation::order. KO8 reach remains four. Repaired local 12-rank Minkowski and puncture checkpoints, including RK compensation, are each byte-identical to their frozen one-rank references. See checks/ko-dependencies/verification.json and its read-only reproducer. The old B failure used KO8; this newly found KO10 defect does not explain that failure.

A separate frozen Aurora SYCL/MPI build is underway from that commit in build-sycl-mpi-native-transition-ko-dependencies-v1, using launcher-native-transition-ko-dependencies-v1 and manifest source-native-transition-ko-dependencies-v1-manifest.json (SHA256 6d81b6803fe811d57a4789a8d10a60e920f7f56c48e4458f7ff50836c2dd611d). Existing source/build snapshots remain unchanged. C/D/E must use repaired binaries; require fresh corrected GPU parity before each long control. No claim of corrected GPU validation or long-time convergence is made yet. The twenty-minute monitor follows the replacement build and will advance the authorized cases.

To maintain the 5GB remote cap, nine obsolete large output copies (677,664,320 logical bytes) were locally hash/length verified and remotely rehashed before retirement; receipt aurora/retired-before-ko-dependency-build.json. All originals remain locally. No active restart was removed.

Analysis now accepts an explicit provenance-checked frozen helper: compare_smokes.py --campaign-helper PATH, and compare_histories.py --case-helper C=PATH (repeat D/E). Defaults preserve original A/B science; this permits old and repaired identities in one attributable comparison without relaxing binding checks.


## Corrected KO10 GPU test queued

The isolated donor-repair SYCL/MPI build completed successfully: configure and make returned zero, and all five evolution plus five matching diagnostic targets passed frozen configuration/provenance checks. Source is e7d55b008c86774580386dbc3e13028fa72b7e24. New science SHA256 is 2f503e9822b2c119b12cf49d92d7abbfb70d4d02c27848be945db7b50105c352; corrected C executable SHA256 is dfb97dc0865bbf9c0e56d376bbd0477926d25489394aa5757fb7436db71d559e. Full hashes and compiler/build logs are preserved in aurora/ko-dependency-build (relative to this reproduction directory).

Fresh corrected C smoke job9032033 is queued on debug-scaling, one node/12 ranks. Plan campaigns/n16-native-transition-ko-dependencies-v1-c-smoke-segment01/plan.json, SHA25653f52e2f8c7050a701d9f4b6b8b2855d39fee526f9d57691c0771ba1de87aaec. Both user debug queues were empty before submission. Binding measured3.516GB with1.177GB headroom after output/atomic reservations. The new frozen launcher-native-transition-ko-dependencies-v1 supplies the matching scientific identity; old C/D/E executables are not used. No corrected GPU result or C long evolution yet. The20-minute monitor follows this saved submission and will proceed to fresh C10M only after successful two-step Minkowski/puncture admission and SERIAL/GPU comparison. Subsequent D/E controls and healthy checkpoint continuations remain authorized.


## Corrected KO10 GPU partition parity passed

Corrected C smoke job9032033 passed both two-step cases with native return0, worker passed and PBS exit0. The frozen helper verified admitted NI2 schema/configuration, finite payload, exact time/steps and scientific/artifact identities. All nine artifacts were copied and locally length/SHA256 verified. Maximum per-field scaled physical differences from the frozen one-rank SERIAL references are9.1874e-14 (Minkowski) and1.3406e-13 (puncture); RK compensation differences are below4.45e-16. Null surface/ghost maxima are at most1.111e-15. This confirms the MPI donor repair restores short-run backend/partition agreement, without establishing long-time stability or convergence. Evidence is in aurora/c-corrected-smoke-parity.json, c-corrected-smoke-local-verification.json and c-corrected-smoke-verified/. Original C parity-failure evidence remains separate.


Fresh corrected C evolution is submitted as job9032055 on debug-scaling,one node/12ranks. Plan campaigns/n16-native-transition-ko-dependencies-v1-c-evolution-segment01/plan.json SHA2562b9ac185cddddaa50919f51dd7076f5465518da7fcab1f0849bd9a24ec51b57c. It starts native wormhole data at t=0 toward10M, NI2,dt=.0005,original collar/KO10/original scri, with the repaired MPI donor reach and unchanged other settings. Both user debug queues were empty before submission. Binding footprint3.676GB leaves1.097GB headroom after reservations. Smoke lock released; all smoke artifacts remain remotely and locally preserved. The20-minute monitor follows this long run and automatically continues healthy admitted walltime checkpoints; no long-time result yet. D/E remain pending.
