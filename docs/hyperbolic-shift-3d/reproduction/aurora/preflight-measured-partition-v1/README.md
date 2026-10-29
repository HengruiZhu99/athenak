# Measured ownership GPU preflight — pending

Frozen scientific source: `85b8ba91a45dca07496d941e765ea6929006c352`.
The source-v22 manifest contains 2,653 files (including 1,486 pinned Kokkos files),
with SHA256 `361cf418ffc284fe569208d3b2e1ec43d1fbdde5b514fa8b560d3b0813e452be`.
Preparation passed; the SYCL/MPI build is running with `make -j64`. No preflight
job has been submitted for this candidate, and no speedup has been established.

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
comparison. Independent build/plan verification and binary pins remain to be
recorded after build completion and before submission.

The executed preparation/build scripts are retained here. As with the previous
campaign, the export helper runs from the local aurora-staging directory; the
other helpers run in `/home/hzhu/hyperbolic-core-shell-20261028`. Home storage is
bounded to 5 GB and no second debug job may overlap this campaign.
