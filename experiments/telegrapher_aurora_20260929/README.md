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

Telegrapher lapse is enabled with the sample's tau=0.1 and kappa=0.2. The mesh keeps the sample's [-64,64]^3 box, root 128^3 cells, 32^3 blocks, four ghost cells, RK4, and CFL=0.25. Tracker-centered AMR protects the horizon as the hole moves. The per-rank block capacity is 128 for levels 5/6 and 384 for level 7; retaining the sample's 600 eagerly allocates unnecessary GPU storage. Initial trees contain 344, 400, and 3200 blocks respectively. Levels 5, 6, and 7 have finest spacings 1/32, 1/64, and 1/128 respectively. `cases.json` records all exact inputs. Two level-7 gamma=5 decks provide a factor-two resolution repeat. Both finders are sampled every 0.5 M.

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

## Aurora results

The fresh SYCL/RDC/PVC build succeeded on 2026-09-29 with source commit `30cd2301`. The link compiled all 144 GPU images. One-node debug smoke job `8879276` finished with PBS exit status 0 on node `x4712c3s0b0n0`, under `CompactBinaryMerger`, using the new default compute image. Wall time was 2 min 49 s. Both cases completed four RK4 cycles (time 0.02899383 M) with finite constraint histories. Fastflow recovered mass 0.9999999127 and dimensionless RMS expansion 9.99e-5; BHaHAHA recovered mass 1.0000000812 and L2 expansion 1.93e-5. Raw small outputs and `results.json` are saved in `results/smoke_8879276` (large checkpoints stay on Aurora). The first compile/run milestone is achieved.

Boost-ladder job `8879324` completed all ten four-step boosted cases on one-node debug (PBS exit 0, wall 14 min 56 s). Both finders succeeded at gamma=1.5, 2, 3, 4, and 5. At gamma=5, fastflow recovered mass 0.9999937002 with RMS expansion 0.0018643; BHaHAHA recovered mass 0.9997362410 with L2 expansion 0.0009922. These are initial qualification runs, not 20 M evolutions. See `results/ladder_8879324`.

Two-node debug-scaling job `8879327` completed both unboosted evolutions to exactly 20 M (PBS exit 0, wall 20 min 8 s). Each finder succeeded in all 40 searches. Final sampled masses were 0.9999292434 (fastflow, RMS 9.04e-5) and 1.0000142097 (BHaHAHA, L2 1.97e-5). Last horizon samples were near 19.627 M, reflecting the 0.5 M sampling cadence. Both constraint histories remained finite; peak integrated squared Hamiltonian norm was 1.61735e-5 at 0.101478 M. See `results/evolution_8879327`.

The SSH proxy was renewed on 2026-09-29, and the completed PBS outcomes were retrieved. Full boosted evolutions are submitted as job `8879482` (gamma=1.5–2, two-node debug) and `8879483` (gamma=3–5, eight-node debug-scaling). The latter starts with gamma=5. The launcher shares a 50-minute budget across cases and checkpoints unfinished evolutions; only actual t=20 termination counts as completion. Higher-resolution gamma=5 remains pending.

`diagnostics.py results` produces `evolution_diagnostics.png/.pdf` and `constraint_slice_peaks.json` from saved histories and x-axis slices (requires matplotlib). The history labels “norm2” are integrated squared norms using physical volume, not square-root norms or volume averages. The slice analysis applies the same chi >= 0.0625 threshold. In the unboosted baseline, both finder runs have identical constraint histories and slices. Their largest sampled, non-excised x-axis |H| is 2.51144e-4 near x=-0.359375 at t=9.005504, chi=0.0644641; it lies 11 cells from a block edge. This local maximum is near the excision threshold and does not by itself establish the cause of a three-dimensional constraint spike.

The unboosted x-axis outermost cells have |H| approximately 3.27e-5 from t=1 through 20 M, compared with approximately 2.44e-14 initially. This persistent boundary feature appears in both finder runs. The saved profiles distinguish it from the near-puncture peaks; a one-dimensional slice cannot attribute the full three-dimensional integrated norm to either region.

Gamma=5 level-7 resolution job `8879509` is queued on 16-node debug-scaling, with both finders and a 20 M target. This repeats the evolution at dx=1/128 while retaining the angular settings, allowing a spatial-resolution comparison rather than claiming angular convergence.

The first gamma=5 evolution attempt in job `8879483` reached the fastflow iteration cap at 0.500744 M. Its failed-search output repeats the previous successful surface; that repeated mass and residual are stale and must not count as convergence. The verbose log records the rejection. The run was stopped and preserved in `results/evolution_8879483`. The pending resolution job `8879509` was canceled before execution. High-boost fastflow decks now allow 4,000 iterations with the same alpha and expansion thresholds. A replacement batch starts with BHaHAHA.

Saved slice tables are gzip-compressed without losing numerical values; `diagnostics.py` reads the compressed files directly. The gamma=5 resolution replacement uses BHaHAHA first and the increased fastflow iteration cap.

The first gamma=5 BHaHAHA evolution alternates cold-search successes with Error Flag 14 (`INTERP1D_HORIZON_TOO_SMALL`) on the following warm search. The solver starts its shell around a zeroth-order center prediction when only one previous surface exists; the moving puncture can leave that shell before the next 0.5 M search. Boosted BHaHAHA decks now use a 0.01 M interval to bootstrap and maintain motion history, retaining the same expansion tolerance. Shape output is every 50 searches to limit storage. Runs already started with 0.5 M retain their original settings and are diagnostic attempts, not proof of uninterrupted tracking. The higher-resolution job reads the updated deck when it starts.

Run-support revision v2 requires freshly evaluated BHaHAHA norms at the stopping test. Optional `bah_initial_dt` supplies a shorter interval until three successful position samples exist. Boosted low-gamma decks now use the single finest 32x64 angular grid, with a 20,000-iteration budget, avoiding mismatched coarse-grid indexing of fine-grid history. A reduced CPU moving-puncture check found all four horizons through eight RK4 steps (0.04818137 M) with this single-grid setup. Fastflow now serializes its full surface and up to three successful samples, and offers an opt-in bounded linear/quadratic predictor as an initial guess. The flow algorithm and residual criteria still validate the predicted guess. Header parsing permits up to 4 MiB and caps padding of long values, accommodating these spectral checkpoints. v2 is built in a separate directory so running jobs retain their original executable.

Validation of v2 checkpoint support restored all 4,225 lmax=64 coefficients exactly from a 113,542-byte header, then took zero evolution steps. A reduced eight-step moving-puncture predictor test passed all four searches and exercised both linear and quadratic guesses. v2 decks live in `inputs_v2`: moving BHaHAHA uses initial interval 0.01 M, regular interval 0.1 M, a single 32x64 angular grid, and Nr=256. High-boost fastflow permits 4,000 iterations and enables the optional predictor. Select `INPUT_DIRECTORY` and `EXECUTABLE_BUILD` explicitly for v2 jobs; original inputs and executable remain available for comparison.
