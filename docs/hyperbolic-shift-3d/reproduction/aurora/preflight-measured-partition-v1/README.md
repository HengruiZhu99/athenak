# Measured ownership GPU preflight — fidelity passed, no speedup

Frozen scientific source: `85b8ba91a45dca07496d941e765ea6929006c352`.
The source-v22 manifest contains 2,653 files (including 1,486 pinned Kokkos files),
with SHA256 `361cf418ffc284fe569208d3b2e1ec43d1fbdde5b514fa8b560d3b0813e452be`.
The SYCL/MPI build passed in 265.49 seconds using `make -j64`. Independent
verification matched every manifest entry to frozen main/Kokkos Git blobs.
Job **8987308** completed successfully; all ten cases and 425 independent checks
passed. The measured-cut candidate did not improve wall time or slowest-rank RHS
time. See [performance analysis](../../measured-partition-v1/README.md).

The prepared ten-case plan has 425 checks. It compares the frozen V21 sparse
executable with equal-cell ownership against V22 measured fixed ownership on
one node using all 12 GPU tiles. Both use the same physical equations, numerical
operators, initial state, timestep and diagnostic cadence. This isolates the
ownership change from the earlier sparse-exchange/boundary-reuse optimization.

The partition and its source profile are explicit bound inputs. They come from
the independently verified sparse run in job 8987177 and are retained under
`../../measured-partition-v1/inputs/`. Expected cuts and the profile digest are
checked in every candidate report. Lifecycle, CPU/device parity, Minkowski and
pulse restart checks precede the 100-step puncture pair and full-checkpoint
comparison. Independent build/plan verification and binary pins were recorded after build
completion and before submission. The verifier also reproduces the cuts from
the frozen cost planner and the previously verified source profile; its 21
offline tests passed.

The executed preparation/build scripts are retained here. As with the previous
campaign, the export helper runs from the local aurora-staging directory; the
other helpers run in `/home/hzhu/hyperbolic-core-shell-20261028`. Home storage is
bounded to 5 GB and no second debug job may overlap this campaign.

Independent verification checked 49 artifacts and complete byte equality of the
pulse restart and the 100-step puncture control/candidate checkpoints. The
allocation was released only after PBS reported terminal state F, exit status 0.
`verification.json`, `worker-status.json`, and `output/` retain the small evidence;
large checkpoints are in the verified local output archive. Archive SHA256:
`42984cbe0320872dd1c0841348d40aa068f3e300dbac7d26a1dafdb627effaf3`
(131,925,449 bytes). Source/build verification and the 21 verifier negative tests
preceded result verification.

After local hash verification, the six redundant remote checkpoints and evidence
archive were removed. `cleanup.json` records 3982790656 bytes remaining in the
Aurora campaign root, below the 5 GB limit. Parent checkpoints remain intact.
