# Controlled N16 evolution matrix (GPU smoke validation started)

Current handoff (2026-11-02): A, repaired C and D completed10M; B failed9.3915M. D native momentum norm is higher than C, concentrated outside; common diagnostic-operator control is pending before pure evolved-state attribution. E smoke9032174 is submitted with sharedW/KO10/first6. Historical entries below retain chronology.

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


## Native initializer control completed10M; shared-transition GPU trial submitted

CaseA job9031760 completed10M/6667steps, with native return0, worker passed and PBS exit0. Frozen endpoint verification passed (NI1,168-byte header,scheme3); all six worker artifacts were copied and hash/length-verified locally. Checkpoint SHA256 ab26ed9fce097b965cb19217f7599efb47c8c119df39ea49dc4d536d5e77cddb. No remote purge was needed. Terminal lock released.

Measured own-candidate exterior H/M/Mr histories nearly overlay SG1: maximum relative changes2.20e-5/4.52e-6/5.80e-6. At10M H=.0154149684,M=.0290076414; accepted horizon mass error+.3413204% with45/68rows accepted. Negative chi stage/final counts56/8 match SG1; no lapse, surface or boundary sign interventions. Surface/ghost null maxima remain below1.56e-15, while pole/clock maxima are.0281679363/.0220012275. Native initialization alone does not materially improve the evolved exterior errors. See aurora/a-versus-sg1 for inspected history plots and counters. Independent common-cutoff regional endpoint reconstruction remains pending; these history norms use each run’s candidate cutoff.

CaseB two-step GPU physics smoke is submitted as job9031861 on debug-scaling,one node/12ranks. Plan campaigns/n16-native-transition-v1-b-smoke-segment01/plan.json, SHA256 b6829754eafaac006479a41492305f5053ff212aef9d46a48a5aaf46d1de0502. Use launcher v2 despite folder spelling. After verified smoke/parity, start fresh shared-W/KO8/original-scri N16 to10M; no caseB long run submitted yet. The20-minute monitor was updated to this job and will automatically continue admitted healthy walltime states.


## Shared-transition fresh10M run submitted

B smoke job9031861 passed both two-step GPU cases. Exact NT1 checkpoint and artifact checks passed; all9 worker artifacts were locally hash/length verified. Maximum physical SERIAL/GPU scaled differences were9.19e-14 (Minkowski) and1.34e-13 (puncture). This is backend/partition agreement, not long evolution accuracy. Its terminal lock was released.

Fresh native shared-W/KO8/original-scri N16 evolution is submitted as job9031894 on debug-scaling,one node/12ranks. Plan campaigns/n16-native-transition-v1-b-evolution-segment01/plan.json, SHA2563ce975744577a0727794d852c9deacb4910295e3b395aa2b68b4d5667f632073. Use frozen launcher v2. Science,dt,floors and diagnostics remain matched; no smoke restart. The20-minute monitor now follows this job and automatically continues verified healthy checkpoints when necessary. C/D/E remain unsubmitted.

A-versus-SG1 common-cutoff reconstruction is now complete at r>.1099 using the matching NI1 diagnostic. Reconstructed full-domain norms agree with the native report. Exterior H=.0154149683973,M=.0290076414088, essentially unchanged from SG1. The outer.775–.9 still holds62.197609%/75.024155% of exterior H²/M²; .16–.30 holds23.442777%/8.975670%. This independently confirms the initializer alone does not remove the evolved outer error. Diagnostic command/binary/checkpoint hashes, radial data, comparison and reproducible plot script are in aurora/a-versus-sg1.


## Shared restoration with original numerics failed; KO10 control submitted

B job9031894 failed at last accepted9.391500000000638M/6261steps; last recorded diagnostics are9.3M. Device reason2 is invalid native geometry/source, at global active cell57352. All six worker artifacts, including the rejected-stage capture, are locally hash/length verified. The checkpoint is explicitly unadmitted; no restart is permitted. Terminal lock released.

The rejected volume contains eight finite but indefinite conformal metrics, all at r=.609375 in the shell, with minimum eigenvalue−1.9128613e-4. At the first point(x=.3748364114,y=z=−.3397319203), eigenvalues are approximately(−.000191286,.726199,439.502), determinant−.061052, chi9.77581 and conformal lapse.00578966. No volume values are nonfinite; the captured surface metric remains positive(minimum eigenvalue.28425). This is not a lapse/chi sign-trigger termination or failure to enforce the algebraic null condition. It localizes the immediate invalid geometry; it does not determine the cause of its preceding growth. The rejected RK stage abscissa was not recorded and is not inferred.

Scri surface/ghost null maxima remained below1.56e-15, but pole/clock maxima grew to.7126948/.1148995. Candidate-exterior errors dropped around6–8M and then grew; last9.3M H=.02819945,M=.06584202. These use evolving candidate cutoffs; no matched10M endpoint or convergence claim is available. Accepted mass gaps remain. Chi floor stage/final totals112/32 differ from negative-candidate counts104/32 (positive below-floor interventions also count). Events remain at the nearest eight puncture cells, spatially separate from the failed shell. No lapse/surface/boundary sign interventions. Comparison plots are in aurora/ab-versus-sg1; failure evidence/localization is in aurora/b-failure*.

C GPU physics smoke job9031971 is submitted on debug-scaling,one node/12ranks, plan campaigns/n16-native-transition-v1-c-smoke-segment01/plan.json SHA256980b9bdb1275cbcd5405ad5bd9c9d705eb32f048119e7a6392387c6e1a914f14. This independent native original-collar/KO10/original-scri control retains matched settings. Verify smoke/parity before fresh C evolution to10M; do not resume B. The20-minute monitor now follows C. No remote evidence was purged.
