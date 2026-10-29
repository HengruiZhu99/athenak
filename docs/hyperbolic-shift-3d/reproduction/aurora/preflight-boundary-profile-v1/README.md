# Boundary and error-synchronization profile — verified

Frozen source: `555ad41facd84c90a303f2f80f794ed3b917efaa` (source-v23).
Manifest SHA256: `d53c69a0e13581d58e55b333b6775c29d147c84845d2834e6d02ca2694cdd4c3`.
The source contains only the additional boundary/error-synchronization timing
relative to the measured-ownership implementation. This campaign returns to
equal-cell sparse ownership; the measured-cut experiment did not improve speed.

The separate SYCL/MPI build uses the retained environment and `make -j64`.
The build passed in 253.20 seconds; all 2,653 main/Kokkos source blobs, the
three named executable hashes, cache and launch bindings were independently
verified. The 21 offline evidence-reader tests also passed. The scheduler guard initially rejected submission before qsub while job
8987368 (`preflight-thin-core-v1`) occupied the allocation. After that job became
terminal and its owner released the allocation, this campaign was submitted as
**8987400**. All ten cases and 410 independently replayed checks passed; complete pulse
restart and puncture control/candidate checkpoints are byte-identical. See
[the measured breakdown](../../boundary-profile-v1/README.md).
The prepared ten-case plan has 410 checks and uses CompactBinaryMerger, one debug
node, all twelve ranks/GPU tiles. Source/build/binary verification and pins are
retained beside this plan and were completed before the attempted submission. Startup, 100-step puncture timing and checkpoint
equality are compared against frozen V21 sparse control on the same node.

The new timers separate projection, normal halo application, sampled-normal RHS,
angular work, scri RHS and boundary error/statistics collectives. All-error-check
MPI timing is also reported but overlaps other buckets and must not be summed
with exclusive phase totals. See [timer semantics and local byte-equivalence](../../boundary-profile-v1/README.md).
No performance gain is expected from the timers themselves. Their purpose is
to identify where further kernel batching or communication overlap is justified.

As in prior campaigns, export runs from the local `aurora-staging` directory;
remote preparation/build runs under `/home/hzhu/hyperbolic-core-shell-20261028`.
The build and campaign storage checks preserve the 5 GB limit. No physical
scheme, floor, horizon quality criterion, or long-time scientific gate changes.

The allocation was collected/released only after terminal PBS state F, exit 0.
The initial remote archive attempt stopped at the storage reserve guard before
creating an archive. Evidence was instead streamed directly over authenticated
SSH to local storage, with the stream SHA256/size computed remotely and verified
locally. No large remote archive was created. SHA256:
`e296d2d0c0366b5f57420493e93cc93a39bb8952adabe9a38e508df22bc90ee2`
(131,926,903 bytes). The independent verifier then checked all 49 worker-bound
artifacts, full checkpoint byte equality, and timer-accounting constraints.
`stream-evidence.py` and `archive.json` retain the export mechanism and receipt.

After verification, the six redundant remote checkpoints were deleted. The
cleanup receipt records 4284500030 bytes remaining, below the 5 GB cap.
No other campaign data or checkpoints were removed.
