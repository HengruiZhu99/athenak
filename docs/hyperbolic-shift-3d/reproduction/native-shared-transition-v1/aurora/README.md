# Frozen N16 GPU comparison

See ../EVOLUTION-MATRIX.md for scientific attribution and ../STATUS.md for the
measured consistency qualifications. These are five controlled N16 evolutions,
not a resolution convergence demonstration. No long N24/N32 runs are authorized.

The source package is exactly commit
`fd4652ff69d55ec973b61029b4e77609e49f0860`, including pinned Kokkos
`6739bc623081648af9e752b616d9671527922cbf`. The separate launcher comes from
`6d1f582e4beab213ff1657646d1f553ba9edbe9c`. `package-provenance.json` binds every
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
python3.11 launcher-native-transition-v1/native_transition_campaign.py plan --root "$PWD" --variant A --purpose smoke
python3.11 launcher-native-transition-v1/control.py bind --root "$PWD" --plan campaigns/n16-native-transition-v1-a-smoke-segment01/plan.json --source-manifest source-native-transition-v1-manifest.json
qselect -u hzhu -q debug
# Require no other user debug job, then submit once with control.py submit.
```

Inspect saved submission/worker/native/scheduler state before every action. A
queue or running state is not permission to submit another copy. Use
CompactBinaryMerger/debug, one node,12 MPI ranks/GPU tiles. Smoke verification
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
all user debug jobs before submitting. Remote verification can precede local
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
