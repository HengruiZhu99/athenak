# Controlled N16 evolution matrix (GPU smoke validation started)

All new cases start from the same native wormhole at t=0 and target10M. They use N16, dt=.0005, direct chi/alpha RK and physical-variable KO, the existing lapse/chi floors and pre-floor sign logging, unchanged damping, angular epsilon2, scri penalty scale1, horizon lmax12/ntheta24/tolerance.001, and diagnostic interval.05. No evolved coarse/failing checkpoint or log-to-direct conversion is used. Healthy walltime continuations must use that case's own admitted checkpoint and exact configuration.

The preserved SG1 N16 run supplies the existing external-initialization reference. Five bounded N16 cases separate the effects requested by the user:

| Case | Native target suffix | Restoration | KO | Scri first derivatives | Checkpoint |
|---|---|---|---|---|---|
| A | control | Original collar | Original KO8 | Original | NI1 |
| B | shared | Shared W, once | Original KO8 | Original | NT1 |
| C | control_ko10 | Original collar | KO10 | Original | NI2 |
| D | control_ko10_scri6 | Original collar | KO10 | first6 repair | NC2 |
| E | shared_ko10_scri6 | Shared W, once | KO10 | first6 repair | NS2 |

Targets are `hyperbolic_shift_3d_mpi_native_SUFFIX`; diagnostic targets use `hyperbolic_shift_3d_checkpoint_constraints_native_SUFFIX`. Magic strings have prefix `HYP3D_MPI_` and the listed identity. Comparisons: A versus preserved SG1 isolates native initialization; B−A isolates gauge restoration with the original numerics; C−A isolates KO; D−C isolates the scri derivative repair; E−D isolates gauge restoration with repaired numerics. B versus E gives the combined numerical change with shared restoration. This is not a resolution convergence study. Retaining old numerical controls does not imply that their proven third-order scri defect is acceptable for the final method.

The current tested launcher runs cases sequentially under the one-node/12-rank CompactBinaryMerger debug policy. The user subsequently permitted debug-scaling with up to5nodes; that queue allows one running job per user. Parallel cases would therefore need one explicitly tested multi-node allocation with separate case outcomes and the same total5GB storage cap; current single-node plans must not be reinterpreted silently. Do not bundle cases in a way that a time limit or failure silently prevents later fresh trials. A numerical failure terminates that case; preserve it and continue to the next authorized fresh case after disposition. Do not change equations, floors or dt within a case. A missing or failed endpoint cannot supply a matched10M radial comparison; use measured histories and clearly label available earlier times.

Before submission: inspect all current user debug jobs and saved submission records; verify storage including planned output and atomic checkpoint reserve below5GB; build SYCL/MPI with make-j64 using a recent documented Aurora environment; bind scientific source and all selected binary hashes; freeze a separate launcher with case-aware schema verification. Source commit fd4652ff6 contains the tested science; subsequent launcher-only changes must preserve scientific content. GPU startup/smoke verification is required because all new audits so far used local SERIAL Kokkos. CaseA GPU smoke job9031714 passed and matched SERIAL physical fields within1.34e-13 scaled error. CaseA fresh long job9031760 is submitted toward10M; casesB–E remain pending. Aurora authentication was renewed; corrected SYCL/MPI build attempt02 completed and all ten binaries are frozen. Read aurora/state.json and verify the actual process/result before any action.

Initial-data and KO gates are documented in STATUS and checks. Scri manufactured RMS consistency improves markedly with first6, but seam pointwise maxima still have finite-grid orders below4. Carry this qualification into the report; do not label long runs a convergence demonstration. Reaching10M alone is not accuracy evidence.

Collect all artifact hashes/lengths before remote retirement; preserve matching admitted checkpoint inputs and horizon seeds until successors are verified. User-authorized healthy continuations may be submitted after full remote validation before local transfer, but no purge is allowed before local preservation. Keep the5GB cap even when SSH transfers fail.

Final comparisons must retain full/candidate-exterior H/M/Mr histories, accepted-only horizon masses and finder flags, null and regularity residuals, segment-summed floor/negative/zero counts with boundary counters separate, and common-cutoff endpoint regional/radial errors, especially .775–.9. Use matching numerical diagnostic binaries for each case. No new long N24/N32 runs are part of this matrix.

All five production variants compiled successfully in the fresh local SERIAL/MPI build; commands/log and binary identities are in checks/native-matrix-build.log/json. This does not substitute for Aurora SYCL verification.


Current submission uses corrected frozen launcher v2 (`d446f75c6505b47f26f8b4beabada722da423552`) on debug-scaling, one node/12 ranks. The original job9031739 was cancelled while queued, before any evolution: the worker omitted this phase from the configured180-second exit-reserve selection. V2 honors180seconds (native2920seconds inside command3100seconds); scientific executables/settings remain unchanged. Current plan is `campaigns/n16-native-transition-v2-a-evolution-segment01/plan.json`. The20-minute monitor is ACTIVE and will automatically continue healthy admitted checkpoints. Prioritize measured puncture errors and convergence controls; do not repeat launcher tests without a demonstrated blocker.
