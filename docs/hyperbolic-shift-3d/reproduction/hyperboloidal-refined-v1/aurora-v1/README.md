# Aurora first staged comparison

Job 9037790, one `debug` node, CompactBinaryMerger, 12 ranks on 12 distinct GPU
tiles. All nine commands completed. GPU-aware MPI mapping, actual reference-Z4c
profile oracle, CPU/GPU parity and dynamic AMR checks precede the relevant
boosted evolutions. All three N16 evolutions reach t=0.05 (t/M=0.15), 200 steps,
dt=0.00025, without a restart. Scheduler exit status is zero.

**The `redistributed-n16` case here uses the first, Omega-focused placement**
(.235,.30,.56,.75,.875,1), now named `compactification_only.cfg`. It is not the
final both-layer placement. That follow-up is recorded in `../aurora-v2`.
The original names and evidence are preserved.

`raw-evidence.tar.gz` contains the histories, reports, logs, unsmoothed profiles,
launchers, source/build manifests and compile commands. `archive-receipt.json`
hashes every member and the archive; `verification.json` records independent
local verification. `job-state.json` records SHA256 hashes of the large original
checkpoints retained on Aurora. `checkpoint-framing.json` independently checks
RF1 magic, identity, admission and exact serialized length.

The exact scientific source is ddc5b48004420635f13d06cf56e0fbfea65d0d73;
`source-manifest.json` binds its archive and Kokkos revision. Later commits
isolate experimental algebra from legacy GPU kernels and correct the reported
checkpoint-header byte count. The frozen binary reports the 208-byte fixed
prefix; actual serialization already includes the identity length and string.
The collector checks the correct payload offset. These frozen runs do not
claim to execute a later binary. Local final-source regression receipts are
in `../local-gates`.

The flat formulation increases exterior constraints in this bounded comparison.
Omega-focused redistribution reduces final exterior H relative to flat, but
still exceeds legacy by 2.21 times. All eleven horizon samples per case pass
the 1e-2 postprocessing expansion threshold; no missing masses are filled.
Chi floors activate near the puncture in all cases; lapse floors remain zero.
The full-domain H norm grows to about 24, so exterior and total norms must not
be confused. Completion and accurate early horizon masses do not demonstrate
long-time robustness or convergence. See the final combined comparison in
`../aurora-comparison` for metrics, timing scope and both placements.
