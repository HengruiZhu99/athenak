# Read-only review prompt: Brill shift controls

Repository: https://github.com/HengruiZhu99/athenak

Branch: `codex/brill-zero-shift-advection-controls-20260818`

Final source: commit `1c95db8a2adc743672b49a525c21c4f762f35223`, tree
`5f343d0e19bc47fa5cfcf199c342885fde14154b`, Kokkos
`6739bc623081648af9e752b616d9671527922cbf`.

Please perform a strictly read-only source/evidence review. Start with
`docs/investigations/brill_zero_shift_advection_controls_20260818/REPORT.md`,
`evidence_manifest.json`, and the normalized history CSVs in that directory.
Do not run a long evolution, alter production numerics, weaken chi checks, or
claim convergence.

## Authenticated results

* **Z1 zero shift / N128-tree replay:** job `57251896`, terminal manifest
  `45f7fc4c8710db270bc1d1adcb55853ba9ca43f3a5a65b6ce664d6de7c83d15e`, run
  log `faba96361e1a676bfcb6e87764dede3ee73360552cef175c1acdad500a28db9`,
  executable `d1d17ca2ec96428ce67d597e3fcef4cf7d6026ceef4bff09243817983e280292`.
  Zero-shift invariant passed, but replay failed at `t=3.97265625M` before
  RK update completion with chi `-0.18994101508179184` at GID 44. The replay
  tree was under-resolved relative to native shadow requests.
* **U2-short Gamma/O2 transport:** job `57254459`, run log
  `67f635dea064332f05ed51f0d102f677c61c4da960c5ab897c80f6c2f806d8f9`, hst
  `e57f1c62fef022bd54134c25a184da73d25589f9396f181420c3c54acd100bf9`.
  It reached `t=10.2375M`, then failed with 240 invalid chi parent stencils
  (GID 33, level 5), before the known reference crossing at
  `10.5357421875M`. U2-full was not run.
* **Z2 native-AMR zero shift:** job `57255235`, run log
  `8364b6375dc8bf2d49a5de5a5b451d4ebb85161650bd638c6fec7e2e3f79f39c`, hst
  `a4207ad10d3fecfecb6fa99a1a988c7c714403e44ff8596b46bd109151261bc9`.
  It was bounded at `t=2.45273323M`, `dt=5.722046e-7M`, level 15 and 962
  MeshBlocks after repeated native AMR; no qualification gate was reached.

All three used separate one-GPU `gpu_shared_interactive` allocations. An
unrelated existing `gpu_interactive` job was observed later and was not
modified.

## Questions for review

1. Does the source and diagnostic ordering support the inference that native
   repeated AMR is a necessary contributor to Z2's runaway, while replay
   under-resolution limits what Z1 says about zero shift?
2. Why does U2 O2 transport still fail before the Gamma/O6 reference crossing?
   Is there a remaining shift-independent AMR/coarse-cache mechanism, or is
   the comparison confounded by the replay/restart state?
3. Is the first Z1 chi failure best interpreted as an active RK-stage failure,
   a transfer-generated parent failure, or unresolved because the current
   provenance probes are not stage-complete?
4. What is the smallest decisive next diagnostic? Prefer a short,
   stage-resolved census of chi immediately after RK update, after restriction,
   after receive/physical BC, after same-level coarse refresh, and immediately
   before the parent gate, with source stencil and shadow-average values.
5. If one source-level correction is justified, what exact invariant or cache
   contract should it enforce, and what focused test would falsify it?

Keep observation, inference, and hypothesis separate. Do not suggest floors,
clipping, weakened positivity gates, gauge/damping/KO/CFL sweeps, broad
parameter scans, unsupported convergence claims, or Figure-3 qualification.

