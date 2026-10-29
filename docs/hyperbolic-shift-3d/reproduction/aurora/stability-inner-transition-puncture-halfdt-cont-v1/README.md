# Half-timestep continuation submitted

Job 8984609 continues the independently verified, scientifically accepted
checkpoint from job 8984574 (coordinate t=.5=1.5M) to t=1=3M. It uses the same
source-v14 binary, n16 mesh, dt=.00025 and all physical acceptance limits.
The checkpoint and horizon seed are hash-bound inputs; the original isotropic
input is not reloaded. Expected successful endpoint: 4002 total steps,
2001 segment steps. The added check requires the bound horizon seed to load.

This run tests whether timestep halving avoids the previous negative-lapse
failure near 2.9M. No outcome is claimed yet. Resources remain one debug node,
12 MPI ranks and 12 GPU tiles under CompactBinaryMerger. Footprint at binding:
3,863,696,082 bytes; planned outputs and checkpoint reserve fit below 5GB.
