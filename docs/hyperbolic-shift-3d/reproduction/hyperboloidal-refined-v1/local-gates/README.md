# Local qualification

Serial Kokkos and MPI on the local Apple workstation. These short tests qualify
implementation and interfaces; they do not establish evolution convergence.
See `completion.json`, the bytewise comparison receipts, `bounded-n16.json`,
and `raw-evidence.tar.gz`. Large checkpoints remain local with SHA256 receipts.

* RF1 legacy mode reproduces BN1's entire N16 two-step payload and fixed header
  exactly after removing its new magic and variable configuration identity.
* After the shared-code changes, the BN1 target itself still reproduces the
  frozen checkpoint SHA256 `8d78ce6cec0bab7c2edf5763b2d0e2300f5845747f5045dd88ba6f1fe15f44ee`.
* Flat AMR wave: 71→78→64 leaves. MPI2 uninterrupted six steps and an MPI2 save
  at step3 followed by MPI4 restart/coarsening are byte-identical, including
  all state, compensation, topology, counters and historical event arrays.
  SHA256 `6b5a81948fa559a4366a52919d7e7ccec6662060e4c86941c3e70f2734fd38eb`.
* A different radial/profile identity is rejected collectively before evolution.
* The redistributed AMR wave also completes both regrids. These coarse wave
  fixtures have large discretization constraints and are interface checks only.
* N16 boosted baseline, flat, and flat+redistributed runs all complete two steps
  to t=0.0005 (t/M=0.0015), dt=0.00025. No chi/lapse floors fire. The horizon
  remains found, with expansion RMS about 2.31e-4 and mass about 0.333333345.

| Profile | Initial exterior H L2 | Final exterior H L2 |
|---|---:|---:|
| Legacy | 0.01787656 | 0.0191242 |
| Flat | 0.111434 | 0.118615 |
| Flat + radial redistribution | 0.07017962 | 0.0745967 |

The flat profile worsens constraints at N16. Redistribution reduces the flat
case's final exterior H norm by about 37%, but it remains about 3.9 times the
legacy value. This is neither convergence nor demonstrated robustness gain.
The full-precision histories are authoritative for all rounded numbers here.

Reproduction (from the repository root, after an MPI/Serial Release build):

```sh
BIN=build-refined/local/tst/hyperbolic_shift_3d
CFG=inputs/hyperbolic_shift_3d/refined
mpiexec -n 1 "$BIN/hyperbolic_shift_3d_refined_profile" "$CFG/flat.cfg" flat-profiles.csv
"$BIN/hyperbolic_shift_3d_refined_device" "$CFG/flat.cfg"
mpiexec -n 2 "$BIN/hyperbolic_shift_3d_refined_amr" 12 wave 3 .00025 output/save - dynamic "$CFG/flat.cfg"
mpiexec -n 4 "$BIN/hyperbolic_shift_3d_refined_amr" 12 wave 6 .00025 output/resume output/save.checkpoint dynamic "$CFG/flat.cfg"
mpiexec -n 2 "$BIN/hyperbolic_shift_3d_refined_amr" 12 wave 6 .00025 output/full - dynamic "$CFG/flat.cfg"
cmp output/full.checkpoint output/resume.checkpoint
# Repeat with distinct output prefixes for legacy.cfg, flat.cfg, redistributed.cfg:
mpiexec -n 2 "$BIN/hyperbolic_shift_3d_refined" 16 output/flat puncture .0005 .00025 - - 300 .05 .001 12 24 2 "$CFG/flat.cfg"
```

Development failures are retained: before the explicit Cartesian inner source
adjustment the reference RHS comparison fails; after that adjustment and exact
parameter serialization it passes. Final oracles also cover nonconstant Omega
in the transition, rather than restricting the comparison to equilibrium.
