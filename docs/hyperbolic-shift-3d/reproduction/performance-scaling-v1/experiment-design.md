# Performance/scaling experiment design

Base: c16a7c212acb00e9f963cb9a9f42d5d886777a7e; Kokkos:
6739bc623081648af9e752b616d9671527922cbf. Isolated managed worktree and
codex/hyperboloidal-performance-scaling branch. The user's execution-request.txt
supersedes the copied review's no-remote language only within this task.

## Independent variants and acceptance

* base: frozen original AMR1, unchanged scientific compile definitions.
* m0: driver instrumentation; same arithmetic, diagnostics, topology and dt.
* m1a: exact-block persistent diagnostic transfer. CPU directly copies the
  host-accessible selected spans; GPU packs all selected blocks and fields in
  one kernel and performs one device-to-host copy. The full logical host padded
  vector stays intact for SampleJets. GPU staging adds persistent storage;
  this is a transfer experiment, not a compact-state memory claim.
* m1b: radial PreparePointJet computes first derivatives for fields 0..27 and
  seconds for fields 0..6,18,22..24. Generic Prepare remains all-field/all-second.
  Unexpected extra derivative requests use the direct evaluator rather than
  stale scratch. Every consumed summation order and stage refresh is unchanged.

All arithmetic-preserving same-backend/same-P A/B final checkpoints must be
byte-identical, including compensation, counters, surface arrays and horizon seed.
The preexisting native derivative test's 2e-12 scaled CPU-reference tolerance is
retained; direct-device comparisons must be exact. No CPU/SYCL whole-state
identity or newly relaxed tolerance is claimed. Known changed-rank FastFlow seed
variation is assessed separately using analyze_amr.py, not excused in same-P A/B.
A numerical failure stops that comparison and preserves raw data. Dynamic boosted
rejection is expected only with the established invalid-interpolated-geometry
message; its donor reconstruction must still match. It is never throughput data.

CPU Serial uses MPI1/2/4 on an Apple M5 Pro with 18 heterogeneous cores and 48 GiB
RAM. macOS affinity and background scheduling limit timing precision. Initial
correctness runs overlapped compilation; they are excluded from performance.
The dedicated CPU plan runs sequentially after correctness jobs/builds finish.
First instrumentation overhead comparison: three alternating pairs at P4, fixed
15-leaf wave, eight steps. Mask experiment: three pairs each at P1/2/4. Diagnostic
transfer: three pairs at P4, fixed boosted mesh and unchanged every-step FastFlow.
The first two steps are excluded from the CPU throughput window; whole-process
and after-setup wall times remain separate. Additional samples are required if
apparent gains cannot be distinguished from the observed spread.

Aurora uses one debug allocation first, with same-P sequential alternated A/B
pairs and no competing task runs in that allocation. GPU-aware MPI is validated
by an actual device-pointer MPI_Alltoall, alongside one visible GPU per rank,
rank/tile masks, CPU affinity and allocation node inventory. The historical
12-tile mapping is verified in the allocation. Setup/diagnostic samples and all
same-P checkpoint hashes accompany timing. Intrusive fenced kernel/allocation
profiling uses a separate Kokkos tool and separate runs; those runs are excluded
from ordinary speedup estimates and their overhead is reported.

At most two allocated compute nodes concurrently, with sequential job submission
and saved receipts before any retry. Uniform 64-root cases are a distinct scaling
problem. Mixed AMR retains the native eight-root rank ceiling. Same-P placement
comparisons distinguish inter-node costs from adding ranks. No weak-scaling claim
is made from a resolution or fixed-shell sweep.

## Observability interpretation

Each rank writes exclusive driver phases and individual phase samples. These host
wall timers insert no synchronization. Generation records retain setup, work,
communication and existing nested timers before destruction, so retire/final
records (not constructed/coexisting duplicates) accumulate old plus new stages.
The existing closure/halo/RHS timers include diagnostics where the corresponding
operator is called. Error synchronization overlaps other timers and must never
be added to exclusive totals. MPI intervals include waiting, not just network
transfer. Rank-zero wall time is reported separately from the largest rank value.

Named Kokkos allocation live/peak totals are sampled with the optional tool across
generations, including coexistence; they do not include ordinary std::vector/map
allocations. Process peak RSS supplements the census for host memory. Label-wise
peaks must not be summed as if simultaneously live; per-space peaks are measured
from the sum of live allocations. Driver memory counts are partial structure
counts and are explicitly not total device memory. Timing-tool fences are never
loaded into default evolution.

## Deferred gates

Native costs, rank participation, capacity and payload handling remain unchanged;
see native-contract-gates.md before any such change. A local-padded storage change
needs a new dependency proof for FD/mixed/advection/KO/scri and moving horizon
reads. Initial profiles point to host model construction as a large startup/regrid
cost; its internal searches need focused attribution before indexing changes.
Neither these experiments nor short fixed boosted survival establish robust
adaptive boosted evolution, convergence, a mixed-grid KO energy estimate, or
production readiness.
