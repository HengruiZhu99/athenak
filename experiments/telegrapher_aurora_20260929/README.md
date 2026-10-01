# Telegrapher lapse: one boosted puncture on Aurora

Scope: item 1 of the requested run plan. Establish the new-image GPU build, test both integrated apparent-horizon finders from a stationary hole through gamma=5, then evolve the successful high-boost setup to 20 M. Other initial-data problems and merger runs are excluded.

## Source

Branch: `HengruiZhu99/athenak:project/telegrapher_lapse`.

- PR 790 head: `22baa243970fa1880b2bbc48e88a590069d55e47`.
- PR 792 head: `08a181fc397f74df7e68975def40a95b5602b102`.
- Merge commit: `efacb13cbb3cbbef198b624c58a3dadb18758da8`.
- Both PR heads are ancestors of this branch; their branches were not pushed to.

The conflict in `Z4c::FindHorizon` was resolved using the new shared `HorizonFinder` interface. Run-support changes add an optional fastflow physical-time cadence (default remains every step), retain checkpoint fields on restart, and support an ellipsoidal initial guess plus the full previous surface. The latter are opt-in. An optional RMS expansion threshold guards against cancellation in the mean-expansion or mass-stall tests. Normalized Legendre recurrences replace the scalar harmonic factorial sum, which becomes inaccurate at high multipoles; addition-theorem and derivative checks cover modes through l=96.

## Input and units

The decks derive from `telegrapher_gauge/inputs/z4c_boosted_puncture.athinput`. The current code requires the built-in `z4c_boosted_puncture` generator: `z4c_one_puncture` is a stationary, separately compiled generator and does not read a velocity. Rest mass is 1; the x velocity is exactly sqrt(1-gamma^-2). Accordingly the requested final time is 20 in rest-mass units; the boosted ADM energy is gamma times that mass.

Telegrapher lapse is enabled with the sample's tau=0.1 and kappa=0.2. The mesh keeps the sample's [-64,64]^3 box, root 128^3 cells, 32^3 blocks, four ghost cells, RK4, and CFL=0.25. Tracker-centered AMR protects the horizon as the hole moves. The per-rank block capacity is 128 for levels 5/6 and 384 for level 7; retaining the sample's 600 eagerly allocates unnecessary GPU storage. Initial trees contain 344, 400, and 3200 blocks respectively. Levels 5, 6, and 7 have finest spacings 1/32, 1/64, and 1/128 respectively. `cases.json` records all exact inputs. Two level-7 gamma=5 decks provide a factor-two resolution repeat. Fastflow is sampled every 0.5 M; BHaHAHA cadence depends on the deck revision, described below.

The initial unboosted horizon has area 16*pi and irreducible mass 1. At high boosts, inspect expansion residuals and shapes as well as mass; a fastflow mass-stall flag alone is insufficient. Its `hrms` output column actually stores the mean square expansion, so `analyze.py` takes its square root. BHaHAHA's circumference-ratio spin estimates can be NaN for a nonspinning hole; its area, mass, and expansion columns must remain finite.

## Aurora build

Campaign: `/lus/flare/projects/CompactBinaryMerger/hzhu/telegrapher_lapse_20260929`.

A fresh clone lives in `source`; builds and simulation output remain in this campaign. The current login environment is oneAPI 2026.1, MPICH 5.0.0, and compute image `compute_aurora_prod_20260928T161726_a233edf0_69791cc`.

Run `build_aurora.sh` on the login node. It retains the pinned Kokkos 4.7.2 submodule and applies two recorded compatibility edits: replace the missing USM pointer aliases with `sycl::global_ptr`, and pass PVC backend tokens separately to avoid oneAPI 2026.1's escaped-quote failure. Distinct target-qualified flags prevent CMake from deduplicating them. Host OpenMP discovery is supplied explicitly because device compilation lacks `_OPENMP`. MPI is provided through CMake's imported target, using `icpx` directly. SYCL RDC is enabled. Subsequent links allow eight parallel AOT jobs; the initial link used the compiler default of one. The physics-registration translation unit additionally uses a host compiler pass with the same SYCL types and headers: oneAPI 2026.1 recursively checks its mesh/CCE pointer containers during device compilation. It launches no kernels; all kernel files remain at O3 with SYCL/RDC. The compiler launcher records this workaround in the recipe. Compute execution must validate the linked result. The script records the commit, compiler/modules, complete Kokkos diff, and executable SHA256.

## Debug runs

`run_debug.pbs` requests one node, one hour, and account `CompactBinaryMerger`, with 12 MPI ranks mapped to GPU tiles and eight host threads per rank. All files are created in a fresh job-specific directory, and each run changes working directory before constructing the horizon finder.

```bash
qsub -v PHASE=smoke run_debug.pbs
qsub -v PHASE=ladder run_debug.pbs
qsub -v PHASE=evolution,CASE=g5_fastflow_L6 run_debug.pbs
qsub -v PHASE=evolution,CASE=g5_bhahaha_L6 run_debug.pbs
```

Smoke and ladder jobs explicitly stop after four steps to qualify initial horizon finding. They do not count as 20 M evolutions. Evolution jobs retain tlim=20 and checkpoint before the batch limit. A continuation can set `RESTART_FILE` to the rank-0 checkpoint; the code selects matching rank files. Continuations read parameters from the checkpoint without loading the initial deck again. Do not count wall-clock or cycle-limit exits as completion. If more nodes are needed, override the PBS queue and selection (for example `-q debug-scaling -l select=8`); the launcher uses 12 ranks per allocated node and reduces per-rank AMR capacity.

```bash
python3 analyze.py /path/to/job/group > results.json
```

This reports actual final time, termination, successes, mass-stall convergence, expansion, and finite constraint history with peak times. Inspect failures and residuals before moving to the next stage. Constraint norms use the code's existing chi-based interior excision; a peak by itself does not identify its spatial cause.

## Local validation

The serial CPU build passed. Two reduced-mesh, two-step unboosted integration tests found horizons with fastflow mass 0.9999976338 and BHaHAHA mass 1.0000000962. BHaHAHA's dimensionless L2 expansion was 1.948740159e-5. A checkpoint reload at the saved final cycle took zero evolution steps and did not reinitialize the puncture. These are integration checks, not Aurora or high-boost qualifications. Details are in `local_smoke_results.json`.

The gamma=5 CPU initial-data probes (one Euler step at CFL=1e-8, physical time about 1.24e-10 M) recovered mass 0.999989969 with fastflow and 0.999735743 with BHaHAHA. Fastflow used a 1/gamma x-axis seed, lmax=64, ntheta=80, alpha=0.1; its dimensionless RMS expansion was 5.45e-4. BHaHAHA used Nr=256 and a single 32x64 angular grid; L2 expansion was 9.95e-4 and Linf 3.73e-3. Its tighter 2e-5 target did not converge in the earlier probes. The high-boost qualification decks explicitly use L2 tolerance 1e-3, while the low-boost decks keep 2e-5. These tolerances and angular resolution remain subjects of the higher-resolution comparison. An experimental change to BHaHAHA coarse-grid stopping was reverted. Details are in `local_high_boost_results.json`.

## Aurora results and current status

The compile/run milestone is achieved. Both original PR heads are merged on the separate branch, and the original branches and `telegrapher_gauge` checkout remain unchanged. The complete boosted tracking and resolution campaign is unfinished; new submissions stopped at the user’s wrap-up request; `results/campaign_status.json` records submissions and outcomes.

| Job | Queue / nodes | Actual outcome |
| --- | --- | --- |
| 8879276 | debug / 1 | Both unboosted finders pass four steps; PBS exit 0, wall 169 s. |
| 8879324 | debug / 1 | Both finders pass initial searches at gamma=1.5, 2, 3, 4, 5; four steps per case; PBS exit 0, wall 896 s. |
| 8879327 | debug-scaling / 2 | Both unboosted evolutions reach exactly 20 M, each 40 successful searches and zero failures; PBS exit 0, wall 1208 s. |
| 8879482 | debug / 2 | Fastflow gamma=1.5 and 2 reach 20 M, each 40/40 successes. BHaHAHA gamma=1.5 reaches 20 M with 19 failed searches; gamma=2 stops at 1.70818 M with 66 failures. Boosted BHaHAHA tracking remains unqualified. |
| 8879483 | debug-scaling / 8 | First gamma=5 fastflow evolution hits the 1000-iteration cap at 0.500744 M; canceled with partial outputs retained. |
| 8879523 | debug-scaling / 8 | v1 gamma=5 BHaHAHA reaches 20 M with 20 successful and 20 failed searches. Some reported successes exceed its configured L2 threshold, so tracking is unqualified. Fastflow with 4000 iterations passes six searches but stops at 2.512066 M on the batch wall limit; later gamma=3/4 cases did not start. |
| 8879643 | debug / 2 | Rejected cadence transition: gamma=5 has 27 successes within threshold, nine failures, saved history through 1.20054 M; canceled. Later cases did not start. |
| 8879644 | debug-scaling / 32 | Canceled before execution to separate finder budgets. |
| 8879662 | debug / 2 | Uniform 0.01 M BHaHAHA probe reaches 0.3 M: 18 successes, six failures, no above-threshold successes; rejected. |
| 8879663 | debug-scaling / 32 | v2 gamma=5 level-7 fastflow running, target 20 M. |
| 8879674 | debug-scaling / 8 | Canceled before execution; replacement 8879705 also canceled at wrap-up. |
| 8879681 | debug / 2 | Nr=2048 probe reaches 0.3 M, 18 successes and six failures; rejected. |

Jobs 8879325, 8879509, 8879539, and 8879561 were canceled before execution. Cancellation/rejected attempts are retained separately from successful qualifications. Only actual t=20 termination and successful exit count as a completed evolution. A completed evolution with failed horizon searches does not establish reliable horizon tracking.

### Build evidence

The first new-image GPU executable was built from `30cd2301`. One-node debug job 8879276 validated it on node `x4712c3s0b0n0`. Fastflow recovered mass 0.9999999127 with RMS expansion 9.99e-5; BHaHAHA recovered mass 1.0000000812 with L2 expansion 1.93e-5. Modules, compiler, complete build log, compatibility patch, source revision, and executable hash are in `results/compilation`.

The independent v2 build in `source/build/aurora-sycl-v2` completed from `1a91212fe7a68bdb1f423329babcab90da9df301`, with SHA256 `8b186cec17f03b2c1029e3c11165c085bc7a45180ec6e13cb010692cab8f3600`. Evidence is in `results/compilation_v2`. New jobs explicitly select this executable and `inputs_v2`.

### Finder revisions and inputs

The first moving BHaHAHA runs alternate successful cold searches with interpolation failures on warm starts. The short bootstrap interval builds position history before switching to the regular interval. Low-boost moving decks also use the single finest angular grid: the original coarse-grid warm-start indexes fine-grid history at the coarse resolution. Nr=256 provides more radial stencil room. These configuration choices still require actual evolution validation.

v2 requires freshly evaluated BHaHAHA residuals at the stopping test. Between diagnostic evaluations, over-relaxation can overwrite cached residuals with values for another trial surface; a cached norm alone must not establish convergence. This revision does not loosen the requested residual thresholds.

Moving BHaHAHA v2 decks use `bah_initial_dt=0.01`, `bah_dt=0.1`, Nr=256, and a single 32x64 angular grid, with a 20,000-iteration budget. Shape output is every five successful searches. L2 tolerance remains 2e-5 at gamma=1.5 and 2, and 1e-3 at gamma=3 through 5. Unboosted BHaHAHA retains its validated multigrid settings and 0.5 M cadence.

Fastflow has a 0.5 M interval. Moving v2 decks enable an optional bounded linear/quadratic predictor of the full surface; the existing flow and residual criteria validate that guess. High-boost decks permit 4000 iterations. RMS thresholds are 0.002 at levels 5/6 and 0.001 at level 7; alpha remains 0.1 for moving holes. Faster alpha values tested locally were rejected because the residual either increased or a mean-only stopping test hid a large RMS error. Failed fastflow rows repeat the previous successful surface properties with a new timestamp. The analyzer filters those stale rows using the verbose convergence log.

v2 serializes all full-surface coefficients and up to three successful samples into checkpoints. Parsing allows 4 MiB and limits long-value padding. The lmax=64 checkpoint probe restored all 4225 coefficients exactly from a 113542-byte header and took zero evolution steps; the eight-step moving-puncture predictor probe passed four searches and exercised both linear and quadratic prediction. See `spectral_checkpoint_probe.json` and `spectral_predictor_probe.json`. These are integration checks, not 20 M qualifications.

### Constraints and resolution comparison

`diagnostics.py results` produces scientific plots and `constraint_slice_peaks.json` from saved histories and x-axis slices (requires matplotlib). Slice tables are losslessly gzip-compressed; checkpoints remain on Aurora. Histories store physical-volume integrated squared norms, not square-root norms or volume averages. Both histories and slice comparisons exclude chi < 0.0625, which is a coordinate-dependent mask rather than an apparent-horizon excision.

In the unboosted 20 M baseline, both finder runs have identical finite histories and slices. Peak integrated squared Hamiltonian norm is 1.61735e-5 at 0.101478 M. The largest sampled non-excised x-axis |H| is 2.51144e-4 at t=9.005504, x=-0.359375, chi=0.0644641, 11 cells from a block edge. Outermost x-axis cells separately show persistent |H| around 3.27e-5 from t=1 through 20, versus about 2.44e-14 initially.

For completed gamma=1.5 and 2 fastflow runs, integrated H-squared peaks are 0.00041357 and 0.00594881, respectively. Their largest sampled non-excised x-axis |H| occurs close to the puncture and chi mask: gamma=1.5 has 0.145745 at t=17.00113, x minus tracker=0.341485, chi=0.0631898, at a block edge; gamma=2 has 0.526746 at t=13.00349, x minus tracker=0.298976, chi=0.0665413, 14 cells from an edge. Thus the sampled peaks are not uniformly at block interfaces. A one-dimensional slice cannot determine the cause of the three-dimensional integrated norm. Gamma=5 level-7 repeats will test spatial resolution while retaining the angular finder settings; this does not establish angular convergence.

The v2 cadence-transition test (8879643) establishes that the fresh-residual guard prevents above-threshold reported successes, but not reliable tracking: the first three 0.01 M bootstrap finds succeed, then the jump to 0.1 M fails. The interval change was a possible contribution. The uniform 0.01 M probe in `inputs_cadence_probe` still fails after each third successful search, ruling out cadence as the sole cause. At that point the interpolation shell changes from 20% to 5% radius margins. The radial spacing is set by the full maximum search radius rather than by the shell width; `inputs_radial_probe` tests Nr=2048 to isolate radial stencil support while retaining all solver tolerances.

The nearest successful fastflow samples place the largest x-axis slice peaks inside their sampled minimum horizon radii. At gamma=1, 1.5, and 2, peak distances from the nearest sampled horizon center are 0.359375, 0.308910, and 0.276650, while minimum horizon radii are 1.134640, 0.763498, and 0.526911. Time offsets are 0.048056, 0.097170, and 0.075810 M, respectively; this is a nearest-sample comparison, not synchronous horizon excision. These well-separated distances indicate that the chi mask retains interior constraint peaks. The current integrated norms should not be interpreted as exterior-horizon constraints. Exact records are in `results/constraint_peak_horizon_comparison.json`, reproducible with `diagnostics.py`.

Run-support revision v3 adds opt-in `bah_cold_start_each_find` for one independent horizon. It resets only finder history before each search, centers a full sphere on the live puncture tracker, retains the regular cadence, and still requires the configured fresh expansion tolerances. The option defaults to false; multi-horizon/BBH mode rejects it. Moving BHaHAHA decks in `inputs_v3` use this mode at 0.5 M intervals with Nr=256 and their existing angular resolution and tolerances. High-boost fastflow v3 decks use 0.1 M intervals; other decks match v2. This deliberately avoids the currently unreliable warm-start path; it does not claim that warm tracking is repaired.

The Nr=2048 radial probe (8879681) also reaches 0.3 M with 18 successes and six failures. Increased radial resolution alone does not repair warm tracking. The local v3 cold-start integration check passes all four searches through eight RK4 steps (0.04818137 M), with no above-threshold success. An initial interval of 0.001 M was deliberately supplied alongside regular interval 0.01 M; searches retain the regular interval in cold-start mode. See `results/cold_start_probe.json`.

High-boost fastflow at 0.5 M intervals spends thousands of iterations relaxing the change between consecutive surfaces, even with prediction. v3 tests a 0.1 M interval while retaining all flow coefficients and residual criteria. `FASTFLOW_FIND_DT` can override the existing cadence parameter on a checkpoint continuation; `actual_arguments.txt` and `restart_source.txt` record the effective invocation. Such continuations preserve the saved fields and spectral history.

The independent v3 GPU build succeeded from the cold-start support revision; see `results/compilation_v3` for its source and binary hash. Job 8879707 (two-node debug) requests cold gamma=5, 1.5, and 2 BHaHAHA evolutions to 20 M. Job 8879705 replaces queued job 8879674 before execution, using v2 fastflow with v3 0.1 M decks for gamma=3, 4, and 5. `RESTART_CASE` restricts a supplied checkpoint to the named case in a mixed batch; other cases start from their own decks.

## Wrap-up snapshot — 2026-09-29 23:18 UTC

New submissions stopped at the user’s request. Queued job 8879705 was canceled before execution. Jobs 8879663 (32-node level-7 gamma=5 fastflow) and 8879707 (two-node cold-search BHaHAHA batch) were already running and are left to terminate normally within their existing one-hour allocations so checkpoint writes can complete. No continuation is submitted or scheduled. Latest progress-log times are 2.944022 M and 1.568291 M, respectively; these are partial observations, not final results. The latter has successful early cold searches, but does not yet qualify 20 M tracking.

Completed qualifications: both finders at gamma=1, and fastflow at gamma=1.5 and 2, each through 20 M with 40 successful searches and zero failures. Both finders also passed initial searches at gamma=1.5, 2, 3, 4, and 5. Reliable boosted BHaHAHA tracking, the remaining higher-boost 20 M fastflow cases, and both gamma=5 level-7 20 M resolution comparisons remain unfinished. The v3 cold-search mode avoids unreliable warm-start history; warm tracking is not repaired.

All completed-run evidence and all three Aurora build provenance records are retained in results. A partial snapshot of the two running jobs is retained separately in results/wrap_snapshot_20260929; it must not be interpreted as their final output. Large restart files and eventual final outputs remain under /lus/flare/projects/CompactBinaryMerger/hzhu/telegrapher_lapse_20260929/runs on Aurora. Collect final outputs before judging these two jobs, then use the saved spectral-history checkpoints for any authorized future continuation.

## Resumed — 2026-10-01, debug queue only

Final outputs of 8879707 qualify cold-search BHaHAHA at gamma=1.5 and 5 through 20 M: each has 40 successful searches, zero failures, finite histories, and zero successful residuals above the configured L2 threshold. At gamma=5, max L2 is 0.00099941108 against 0.001, final irreducible mass is 1.0001584333, and maximum mass deviation is 0.00155655. At gamma=1.5 the threshold is 2e-5 and final mass 1.000015067. Gamma=2 has 26 successful searches, zero failures through 13.0604 M, then writes a wall-limit checkpoint. Final level-7 gamma=5 fastflow output (8879663) has ten successful searches and zero failures through 4.508697 M; it also terminates at the wall limit. Full outputs and parsed summaries are saved in results/evolution_8879707, results/evolution_8879663 and retrieved_20261001_*.json. Earlier wrap snapshots are partial and superseded by these final outputs.

Revision v4 adds opt-in bah_retry_cold_on_failure (default false, one independent horizon only). A rejected warm attempt triggers exactly one full-sphere retry at the same simulation time and existing residual tolerances. All MPI ranks broadcast the failed result, reset finder history, recenter on the tracker, and collectively interpolate the fresh metric grid before retrying. Logs distinguish rejected warm attempts, retries, and terminal failures; the analyzer reports recovery counts separately. The evolving center/radius/time history in the C solver now updates regardless of bah_verbosity; previously these state changes incorrectly depended on printing diagnostics. Fresh-search mode remains available and is the mode already qualified above. These changes do not repair the underlying coordinate-frame mismatch suspected in the warm surface predictor.

All new submission attempts explicitly specify -q debug and select=2. At resume the scheduler rejected both the gamma=2/3/4 BHaHAHA batch and gamma=5 fastflow batch with “would exceed queue generic's per-user limit of jobs in 'Q' state,” despite an empty account job listing. No new run has been accepted, and no debug-scaling submission is attempted. A separate v4 build and local integration checks are prepared while this scheduler restriction is investigated.

The user redirected new submissions to short capacity allocations. Jobs 8883677 (capacity, two nodes, one hour: restart gamma=2 BHaHAHA at 13.0604 M then fresh gamma=3/4) and 8883678 (capacity, sixteen nodes, one hour: gamma=5 level-7 cold-search BHaHAHA) were accepted. There are still no new debug/debug-scaling jobs from this campaign. The allocation is active through 2027-02-01 and has 301356.8 available node-hours as queried on 2026-10-01.

The full active-job audit shows no hzhu jobs before these submissions and no CompactBinaryMerger jobs in debug. Live debug limits are max_run=[u:PBS_GENERIC=1] and queued_jobs_threshold=[u:PBS_GENERIC=1], not a one-job-per-project rule. Direct qsub -f, another login node, explicit server, and held submission also failed; one explicit-server attempt returned an account_check hook exception. ALCF documents this exact no-active-job rejection and directs users to support: https://docs.alcf.anl.gov/running-jobs/known-issues/#error-would-exceed-queue-generics-per-user-limit. Capacity submission succeeds without overriding any limit or changing the charged project.

Job 8883681 (capacity, eight nodes, one hour) was accepted for gamma=5 fastflow followed by gamma=3/4, using the 0.1 M v3 decks and the v2 executable. Job 8883677 began on 2026-10-01 at 13:31 UTC and resumed gamma=2 at the saved checkpoint time.

The gamma=1.5 local warm/retry integration check reaches 0.1248997 M through twenty steps, ten successful searches, zero failures and zero retries needed (L2 below 2e-5). The deliberately coarse gamma=5 dx=1/32 probe exercises cold retry, but retry iteration limits remain and that probe is rejected. At dx=1/64 with quiet diagnostics, three gamma=5 searches succeed through twelve steps (0.03806819 M), but that probe crashes at cleanup and is not qualified. A mesh diagnostic counter was allocated with max_level entries although physical levels include max_level-root_level+1 entries; a single root block overruns its counters at the finest level and corrupts heap state. The counter allocation is corrected; a clean-exit regression check and post-fix finder checks are required. This mesh bug does not affect the Aurora decks, which have a nonzero logical root level.

The analyzer distinguishes warm attempt rejection, cold retry and terminal horizon failure. A restart segment alone must start at t=0 and retain late horizon samples to qualify a full evolution. analyze_chain.py validates chronological checkpoint links, matching restart/final history times, no terminal failures, finite histories, configured residual thresholds, and a bounded gap between successful horizon samples. The existing gamma=5 cold-search 20 M baseline passes with max sample gap 0.504 M.

The single-root mesh cleanup regression passes after the counter fix (root_level=0, highest physical level=2, zero evolution steps, clean exit 0). A fourteen-step post-fix gamma=5 quiet-diagnostics probe retains the original dx=1/64 grid and solver tolerances. Until that finishes, the earlier fine-grid finder probe remains unqualified because of its cleanup crash.

The gamma=2 cold-search checkpoint chain now qualifies through 20 M: forty distinct successful horizon times, no terminal failures, residuals below 2e-5, finite histories and max sample gap 0.506 M. Restart links point to the previous run’s final checkpoint, and the first restart horizon time (13.060) matches the previous final field time (13.0604) to the stored precision. The first history row is later (13.2055), as the saved output counters retain the next scheduled output rather than forcing a new initial history record. See results/g2_cold_chain_20M_20261001.json.

The v4 verified build succeeds from source 0cf934c4 with SHA256 ce8af50c2aa4aca77c11878c2a767f0e7cd5a75b6e80286702273ca4d852e13a. Two initial build launches overlapped in the same directory after an interrupted Git fetch; the redundant tree was stopped, and the finder, diagnostics and Z4c units were deliberately recompiled and relinked in a controlled pass. Both original and verified logs are saved; only verified_rebuild.exit_status=0 is the completion evidence. This binary predates the single-root mesh diagnostic counter fix; Aurora decks have nonzero root_level and do not hit that overrun. Job 8883700 (capacity, four nodes, one hour) uses this v4 executable and inputs_v4 to test gamma=5 warm searches with one cold retry on rejection.

Job 8883708 (capacity, sixteen nodes, one hour) resumes level-7 gamma=5 fastflow from the 4.508697 M spectral-history checkpoint using v2 with FASTFLOW_FIND_DT=0.1. Fields and stored surface history are retained. This continuation must be qualified jointly with 8879663 rather than counted as a complete 0–20 M run by itself. These five accepted jobs are all capacity allocations; no further jobs are submitted while the per-user active/queued limit is occupied.

The post-fix gamma=5 CPU integration probe passes fourteen steps at dx=1/64 with bah_verbosity=0, reaching 0.04462332 M and clean exit 0. Four searches succeed; the fourth rejects its warm attempt with error flag 13 and one same-time cold retry recovers it. Maximum dimensionless L2 expansion is 0.000995035 against the unchanged 0.001 target; all histories are finite. Raw deck, log, diagnostics and exit status are in results/g5_retry_cpu_evidence. This is short integration evidence, not a 20 M GPU qualification.

Changing resources with qalter alone triggered account_check exceptions, but explicitly repeating -A CompactBinaryMerger allowed the edit. Job 8883700 now requests two nodes with its existing one-hour walltime. No job was canceled or duplicated for this change. Other queued capacity jobs report normal resource/top-job conflicts.
