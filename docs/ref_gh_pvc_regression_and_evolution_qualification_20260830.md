# Ref-GH PVC regression and evolution qualification

Status: `PVC PORTABILITY REGRESSION ISOLATED` at `ab30fa96`.

This report tracks the source-regression and GPU-lifecycle audit required
before the fully-subtracted stationary-trumpet evolution campaign can resume.
It is deliberately claim-limited: local CPU success and cycle-zero PVC output
do not qualify positive-time evolution.

## Frozen source checkpoints

| Label | Commit | Tree | Role |
| --- | --- | --- | --- |
| W | `68bc1ee30c0ac64c3afe8d3961bead65efce205a` | `00d370276ed4b1377b2b9e46456f838e8e4cbbb0` | Last known Aurora-positive Ref-GH source |
| R | `3c9a34c8c3123c2570eb33e8ec77368feb1f1c61` | `677936216ba8f81d885563bcde41b6596f31cc08` | First repaired source associated with the new PVC failure |
| C | `5e36b641b1ff04272b59e9a66f35db4d1a4ac363` | `d8f0428f7d7196b34766b27435c76613f12603a6` | Starting repair tip with later portability discriminators |

All three source checkpoints pin Kokkos commit
`6739bc623081648af9e752b616d9671527922cbf`.

## Phase 1: identical-build W versus R

The controlling job is
`scripts/ref_gh/aurora_pvc_regression_wr_control.pbs`. It configures and builds
W and R sequentially on one Aurora node, with one module environment and
Kokkos bounds checking disabled, then runs the frozen 96^3, 216-MeshBlock,
12-rank STANDARD Case-D workload for exactly one RK cycle on that same node.

Aurora job `8791722` ran on node `x4216c4s7b0n0` with twelve distinct PVC
tile mappings. Both builds used Intel oneAPI DPC++ 2025.3.2, MPICH
5.0.0.aurora_test.3c70a61, Kokkos SYCL/PVC, GPU-aware MPI, Release mode, and
Kokkos bounds checking off. The controlling input SHA-256 was identical:
`6d483ded11b70d640f4a166fd21757f956802622d4e3994ceda71ae8649235eb`.

| Source | Executable SHA-256 | Result | Latest history time |
| --- | --- | --- | --- |
| W | `e0f2b1e1380b3bc4c3d296ef77e8a5af39bc426c392c8d625de5342b245f585a` | `PASS_ONE_CYCLE_PVC` | `0.003404497M` |
| R | `ae105aa20563ab3d7f9020bc4966cfec244ffcbc667d7caa064ecf42165c5132` | `FAIL_LEVEL_ZERO` | `0M` |

W completed the positive-time cycle with finite diagnostic output. R passed
through its first physical metric and gauge boundary fences, then multiple
tiles reported Level Zero PDE-level `NotPresent` writes and the MPI job ended
with status 143. Since W passed and R failed on the same node and contemporary
build environment, a source regression is established. The last reported
kernel remains only a localization boundary, not proof of the corrupting
operation.

Compact evidence is under
`artifacts/ref_gh_pvc_regression_repair_20260830/phase1_aurora_8791722_wr_control`.
The full Aurora campaign remains at
`/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_pvc_regression_20260830_wr_v2`.

## Phase 2: source bisect

The first midpoint was commit
`b290c2bda559d4db62213266b093458c020920b5`, which introduces the exact
matched q=1 initialization and boundary representation. Aurora job `8791730`
passed one positive-time cycle on twelve PVC tiles at history time
`0.003404497M`; its executable SHA-256 was
`7ee9449234a630edfca604045b519d2b8ec22ea502bef758407400871e3aaa51`.
Therefore `b290c2bd` is good and the boundary/exact-state commit is not the
first-bad source.

Commits `78383244`, `28eee9d8`, and `4ec57dd2` change only the disabled
source-unit target or its source-unit-only device-code-split property under
the controlling `Athena_ENABLE_REF_GH_SOURCE_UNIT=OFF` build. They cannot
change the production executable in this gate. The remaining
production-relevant candidates after `b290c2bd` are the fully-subtracted RHS
dispatch at `ab30fa96` and the stationary production diagnostic
instrumentation at `3c9a34c8`.

The compact midpoint evidence is under
`artifacts/ref_gh_pvc_regression_repair_20260830/phase2_aurora_8791730_mid_b290c2bd_pass`.

The next production candidate was
`ab30fa963f5d1d7ce54748ffb287c91c87705153`, whose exact parent is
`8e028cafe71df0a576ed125db820f9d983d7ae9e`. The parent has the same
production source as the passing `b290c2bd` checkpoint. Aurora job `8791745`
reproduced the PDE-level Level Zero `NotPresent` write fault on multiple PVC
tiles. All ranks had passed the evolved-stage physical metric and gauge
boundary fences; PBS then ended with status 143 before the script could write
its ordinary result file. No positive-time history was produced. The
executable SHA-256 was
`0c64e3cbb784590581af847a55ad993943f93c81bfaf6f6289bfc7dec7e1f475`.

Therefore `ab30fa96` is the first-bad source commit. Its source diff changes
the production RHS and its fully-subtracted analytic gauge helpers; it does
not change `ref_gh_tasks.cpp`, the stationary-trumpet problem generator, or
the physical-boundary implementation relative to its parent. The complete
source diff is preserved with the compact evidence under
`artifacts/ref_gh_pvc_regression_repair_20260830/phase2_aurora_8791745_first_bad_ab30fa96`.

## Phase 3: RHS versus task/boundary hybrid matrix

Hybrid A used W as the base, overlaid the fully-subtracted `ab30fa96`
`ref_gh_calcrhs.cpp` plus every required changed/added helper header, and
retained W's task and stationary problem-generator blobs exactly. Its overlay
manifest SHA-256 was
`3511e55316a7e0c2467b7d0dd7cf48d9d266f9885982d40a9800ce2db13fcf9f`.
Aurora job `8791763` reproduced fourteen Level Zero `NotPresent` writes on the
same node as the W/R control and ended with PBS status 143. Its executable
SHA-256 was
`4068d6e7f1574f938a11c96fde9b2cf3bd78b2c1100aa66fa1762bf5b48e340b`.

Hybrid B is exactly represented by the `b290c2bd` midpoint run: its production
`ref_gh_calcrhs.cpp` blob is byte-identical to W while its exact-state,
stationary initialization, task, and boundary path are the repaired versions.
That job passed one positive-time cycle.

| Hybrid | RHS/device helpers | Task/init/boundary path | Result |
| --- | --- | --- | --- |
| A | fully-subtracted `ab30fa96` | W | `FAIL_LEVEL_ZERO` |
| B | W | repaired `b290c2bd` | `PASS_ONE_CYCLE_PVC` |

The regression therefore follows the fully-subtracted RHS/helper device image
rather than the new exact-state/task/boundary semantics. The changing final
fence location remains consistent with asynchronous fault reporting and is
not treated as the corrupting operation.

## Phase 4: device-image/code-layout discriminator

Aurora job `8791775` rebuilt exact first-bad `ab30fa96` with the sole global
build override `-fsycl-device-code-split=per_kernel`. It passed one complete
positive-time cycle at `0.003404497M`; executable SHA-256 was
`e4f7d1aefd16b135dbf96912f1e72760144c59f5cf1c0fa438fd3158c1d6974e`.

The default and per-kernel binaries both contain 7119 unique `_ZTS` strings.
The default binary's `sycl-spir64_gen` bundle was `0x962eb68` bytes and the
per-kernel bundle was `0xbe78078` bytes. The corresponding text sizes were
180,980,804 and 224,227,460 bytes. Compiler pressure did not improve: the
compatible-Phi kernel remained near 1281 Reals of reported spill, while the
physical-geometry/gauge kernel rose from roughly 332 to 893 Reals under
per-kernel splitting. Thus the pass cannot be attributed to reduced private
spill allocation.

The evidence classifies the regression as a SYCL/IGC/Level Zero
device-image/code-layout portability issue. It does not yet establish an
upstream compiler bug. The smallest permanent build scope and the independent
MPI/message-contract discriminators remain to be completed.

## Phase 5: GPU-aware MPI and completion discriminators

Aurora job `8791789` ran exact first-bad `ab30fa96` with the default device
image on one rank and one PVC tile. The full 96^3 grid and all 216 MeshBlocks
were retained, preserving the same physical-boundary kernels. It completed all
four RK-stage RHS, update, communication, prolongation, and physical-boundary
fences, then reproduced two Level Zero `NotPresent` writes. PBS exit status
was 134 and executable SHA-256 was
`a3ec043b7c2228c1a2e92a72468a03d2e8db73c829a3db9022975c629aa20671`.

Off-rank GPU-aware MPI is therefore not required for the fault. The explicit
send-only and send-plus-receive completion variants are retained as separate
device-page-lifecycle discriminators and are not mathematical changes.

The send-only variant moved `ClearSend` immediately after `RecvU` and before
prolongation/physical BC while retaining `ClearRecv` at stage end. Aurora job
`8791807` reproduced 22 `NotPresent` writes and ended with PBS status 143.
Executable SHA-256 was
`2e6bde2a138f62ee1fe7e4c6744859d7f4f68c07f433027da562d2d4374c5dc0`.
Waiting for sends alone therefore does not repair the regression.

The send-plus-receive variant moved both `ClearSend` and `ClearRecv` ahead of
prolongation/physical BC. Aurora job `8791820` reproduced four `NotPresent`
writes and ended with PBS status 143. Executable SHA-256 was
`b6f79f9448b847bf9778393c33b4e0c5bf9f18b9ddb13a67e7897a912c1786ae`.
Explicit completion of both communication directions therefore does not
repair the fault. Together with the one-rank failure, this removes outstanding
GPU-aware MPI completion from the leading cause list; the temporary task-graph
variants are not candidates for retention.

## Phase 6: rank-packed uniform-grid message contract

The rank-packed size inconsistency is confirmed. Before correction,
`MeshBoundaryValues::GetVarDataSize()` selected `isame_z4c_ndat` for every
same-level Z4c-like message. In contrast, `PackAndSendCC()` and
`RecvAndUnpackCC()` append the coarse same-level array only under the exact
condition `is_z4c_ && multilevel`. On a uniform grid the metadata therefore
advertised `nvars * isame_z4c_ndat`, while the kernels transferred only
`nvars * isame_ndat` initialized elements.

The generic helper `RankPackedSameLevelVarDataSize()` now encodes the pack
truth table and is used by rank-packed metadata:

| Z4c-like | Multilevel | Same-level payload |
| --- | --- | --- |
| false | false | `nvars * isame_ndat` |
| false | true | `nvars * isame_ndat` |
| true | false | `nvars * isame_ndat` |
| true | true | `nvars * isame_z4c_ndat` |

A focused source-unit oracle covers all four cases. Fresh local validation
passed the MPI-enabled build and source unit, the expanded q-controlled and
all-61 RHS oracle, a two-rank uniform Z4c evolved cycle, a two-rank AMR Z4c
evolved cycle, and a two-rank fully-subtracted Ref-GH evolved cycle. A fresh
ASan+UBSan+Kokkos-bounds build also passed the complete q-controlled/all-61
source unit without weakening tolerances. Exact compact evidence is under
`artifacts/ref_gh_pvc_regression_repair_20260830/phase6_rank_packed_contract_local`.

This is an equation-preserving message-contract correction, and true
multilevel Z4c behavior is unchanged. It is not called the source-regression
root cause: W already survived positive-time evolution with the old behavior.
Aurora job `8791879` then built current C plus the correction once on node
`x4020c0s7b0n0` and ran the frozen one-rank and twelve-rank cases with one
executable (`58964e2b...`). The one-rank case passed a complete positive-time
cycle at `t=0.003404497M`, including all four RK stages and final diagnostics.
The twelve-rank case still produced two PDE-level `NotPresent` writes after
all ranks had reported their physical metric and gauge boundary fences. It
ended with status 143 before positive time. Thus the correction repairs the
real metadata/payload contract but does not by itself restore twelve-rank PVC
portability.

Job `8791868` was a preserved harness-only failure: `git archive` had created
an empty submodule directory and the pinned Kokkos symlink was nested one
level too deep. It consumed one second, did not compile, and has no numerical
classification. The corrected source snapshot retained the same commit/tree,
input hash, and Kokkos commit for job `8791879`.

Compact job evidence is under
`artifacts/ref_gh_pvc_regression_repair_20260830/phase6_aurora_8791879_rank_contract_gate`;
the setup-only failure is preserved separately under
`phase6_aurora_8791868_harness_failure`.

## Phase 7: smallest device-image correction

The first scope test applied `-fsycl-device-code-split=per_kernel` only while
compiling `src/ref_gh/ref_gh_calcrhs.cpp`. Aurora job `8791895` confirmed from
`compile_commands.json` that exactly that one source received the option. The
12-rank frozen cycle nevertheless failed at `t=0` with four Level Zero
`NotPresent` writes after all twelve ranks reported their final diagnostic
history fence. Its executable SHA-256 was `c93fef98...`.

Source-object splitting is therefore insufficient. This negative result is
important because the prior global-flag discriminator passed: the remaining
distinction is the option on the final SYCL device link. The next minimal
candidate leaves all translation-unit compile commands unchanged and applies
per-kernel splitting only as an `athena` target link option. This remains a
build/code-generation correction with no arithmetic, PDE, task, or numerical
method change. It is not qualified until three frozen twelve-rank cycles pass.

Compact evidence for the failed source-only scope is under
`artifacts/ref_gh_pvc_regression_repair_20260830/phase7_aurora_8791895_source_scoped_split_fail`.

## Remaining required evidence

The smallest build-scoped device-image correction must still be selected and
combined with the rank-packed contract correction. Required remaining gates
are the rank-contract one-rank/twelve-rank PVC tests, three repeated
twelve-rank one-cycle PVC passes of the final portability correction, and all
positive-time 3M/5M/resolution/20M/100M scientific runs. No stationary-trumpet
robustness or convergence claim is made.
