# Extended-domain stellar revalidation

This campaign reruns the latest domain2048-20260919 configuration from fresh initial data to2500M, beyond its2120.025M failure. The old2005.05M checkpoint already contains a large growing mode, so it is not used. Old campaign data remain unchanged.

The code promotion is narrow: repaired active-only characteristic boundary derivatives (e3bf4add, cherry-picked from e39cc33c), plus the21-line static central AMR floor already used by the previous production executable. The floor defaults off. Pre-existing experimental operators on project/tde remain off; no coupled exterior-memory prototype is introduced.

## Experimental run settings

Same2e5-solar-mass BH/solar star, initial1.5 tidal radii, nominalperiapsis20M, ±2048M domain,32³blocks,10 levels, BH dx.125M and initialstar dx.25M. Static minimum level3 inside±256M and density-gradient refinement are preserved from the latest campaign settings. The density sensor is enabled from this fresh start, whereas the older campaign enabled it during a continuation. The initial876-block mesh_structure.dat is byte-identical to the original.38nodes456ranks gives1.921blocks/rank initially; AMR can increase this. Preserve per-rank binary/restart output and add a read-only x-z constraint slice every50M, MHDTidal/debug-scaling,1hourPBS,-t00:55:00.

Changed run parameters: kappa1 .1→0, shift eta2→.02, outer radial C2 sponge starts512M, ramps1280M, maxrate.001/M. Lapse damping remains.1 and kappa2 remains0. G1 background-adapted gauge, linear ghosts and sixth-order volume differences remain. The layer is outside the initial star and periapsis region. These are promising tested mitigations, NOT a proven cure for the stellar failure. Positive-kappa wide-sponge controls still failed. Three flat direct-Theta controls reached50000M, while the separate refined BH lapse control stopped cleanly at247.25M on walltime, below its1000M target.

## Verification

The poisoned-ghost stencil suite passes1536 derivative cases and98 normal configurations:72 old NaNs become0, zero remains exact and nonzero response remains. The production-source CPU build passes the dynamic floor regression on1/2MPI: floor-on keeps64 leaves, matched floor-off coarsens to8, all zero residual payloads/ghosts stay exactlyzero, and the perturbation remains finite/nonzero. Raw full/ghost metric checks pass. Independent checkpoint Theta maximum and exterior proper-volume RMS match application histories; this checks the stored Theta index17 and geometric weighting. Initial mesh geometry matches the previous run exactly.

The checkpoint reader separately rejects an indefinite fourth ghost, a missing rank and mixed-cycle headers. Seven controller tests use mocked scheduler calls to verify no duplicate running jobs, no uncertain-submission retry, failed-job pause, pending-queue deferral and missing-manifest rejection. No test submits a job.

Reproduce with Python/NumPy and a compiled CPU MPI executable:

```
python3 tst/unit/z4c_boundary_stencil/run.py --build-dir BUILD --output-dir NEW_UNIT_DIR
python3 tst/regression/z4c_static_floor.py --exe BUILD/src/athena --output NEW_RUN_DIR
python3 analysis/tde_revalidation/test_controller.py
```

## Continuations

The deployed advance.py verifies immutable executable/input/helper hashes, PBS and application exits, actual stopping reason, finite histories, all456checkpoint payloads and raw ghost metrics, matching time/cycle/layout, full dyadic coverage and the central refinement floor. It records separate interior/exterior/outer-face Theta norms and maxima; these regions overlap. Every clean continuation must advance from the preceding final checkpoint. The first launch is fresh; later launches cannot use another campaign's checkpoints. A lock and persistent submission_uncertain state prevent blind retries after ambiguous qsub. Finite completion is not a stability declaration. Growth must still be assessed against the old boundary-localized failure before continuing.

The two-hour monitor is responsible for inspection and invoking guarded resubmission, not a blind PBS dependency chain. No automatic changes of physics, mesh limits or target are allowed. Stop at2500M or evolution/checkpoint failure; report renewed exponential growth instead of calling it fixed. Live state, binary hash, submission receipt and diagnostics are stored separately under /lus/flare/projects/MHDTidal/hzhu/tde_1e4_solar_review/domain2048-revalidation-20260920 and local /Users/hz0693/research/TDE/tde-revalidation-20260920.
