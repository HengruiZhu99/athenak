# Measured fixed ownership candidate

`hyperbolic_shift_3d_mpi_continuous_positive_balanced` accepts a final partition
file argument after the angular dissipation coefficient. The cells, numerical
operators, stage arithmetic, and V11 checkpoint format stay unchanged. The
provided contiguous ownership ranges replace equal-cell rank splitting; all
pack/unpack and halo dependencies are built for those ranges. The report records
the actual cuts, minimum/maximum local sizes, and source profile SHA256.

`plan_cost_partition.py` reads a completed V11 report with exclusive per-rank
RHS timings. It estimates constant cost per cell within each measured rank's
ownership interval and proposes approximately equal-cost cuts. MPI waiting is
excluded. It uses explicit ownership cuts when available and the historical
equal-cell partition otherwise. Predictions omit communication, boundary work,
and within-interval variation: they must be tested, not treated as speedups.

The C++ reader rejects mismatched resolution, total cells or rank count,
non-covering/empty/reversed ranges, malformed digests, overflow, and trailing or
truncated input. Rank zero reads the file and synchronizes errors before
broadcasting cuts. The Aurora controller requires a bound partition input and,
for subsequent admitted phases, the exact preflight-admitted partition bytes.
A different file at the same path does not inherit admission.

Local poisoned-workspace regressions use deliberately unequal synthetic cuts
(61%/39%) on two ranks, plus a one-rank-to-two-rank restart. Pulse, Minkowski,
puncture, rejected-stage captures, and rollback checkpoints match the frozen
boundary-reuse control byte for byte. The synthetic profile digest is all zeros
and is explicitly a correctness fixture. It is not a measured performance plan.
Reproduce with `test_sparse_stage_mpi.py --measured-partition` and the balanced
executable, using the same required arguments as the sparse regression.

This is fixed ownership selected at startup, not in-flight repartitioning or
AMR field migration. Native geometry changes and AMR remain outside the current
runner. Twelve-tile performance validation of measured cuts is still pending;
no load-balancing speedup is claimed here.

Validation also includes four planner tests, the C++ descriptor-reader negative
controls, and all 103 offline Aurora controller/worker tests. GPU timing of
this balanced target has not yet been performed.

The next GPU candidate is frozen at `85b8ba91a45dca07496d941e765ea6929006c352`
(source-v22). `inputs/partition-n16-v21-r12.txt` is generated from the verified
sparse run in job 8987177; its sidecar records the cost estimate and source
report digest. The profile is also retained alongside the cuts. This input is
prepared for the twelve-tile comparison, not yet validated as a speedup.
