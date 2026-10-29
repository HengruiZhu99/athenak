# Standalone MPI communication control

`hyperbolic_shift_3d_mpi_communication` isolates the buffer routes behind
the [original-driver timing audit](mpi-performance-audit.md). It constructs
no mesh, RHS or evolution state. The native equations, closure, timestep
and production MPI runner are unchanged. Local Serial/MPI checks establish
correctness of the benchmark, not Aurora performance. The retained
[local validation record](reproduction/communication-benchmark-local-v2/README.md)
covers one, two, three and twelve MPI ranks.

Build the MPI target from an existing configured build:

```sh
cmake -S . -B build-hyperbolic-mpi
cmake --build build-hyperbolic-mpi --target hyperbolic_shift_3d_mpi_communication -j64
```

For a bounded local correctness run:

```sh
python3 tst/hyperbolic_shift_3d/test_communication.py \
  --binary build-hyperbolic-mpi/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_mpi_communication \
  --mpiexec /opt/homebrew/bin/mpiexec --ranks 3 --n 16 --n 24 --n 32 \
  --output-dir output/hyperbolic-core-shell/communication-check
```

The MPI launcher path above is specific to the local machine. CTest also
provides `hyperbolic_shift_3d_communication_cli` and
`hyperbolic_shift_3d_communication_mpi`; the latter uses CMake's MPI launcher
and two ranks. The tests run communication and CLI checks only.

For Aurora, use the existing one-node, twelve-rank launcher and tile
binding. Supply these arguments to the new binary in a fresh, bound
campaign; this document does not submit an allocation:

```text
--output campaigns/COMMUNICATION/output/communication.report.json
--n 16 --n 24 --n 32 --repeats 24 --warmup 3 --mix-repeats 6
--mode both --expect-ranks 12 --expect-backend SYCL
```

The repeat count 24 is divisible by six, so every route appears equally in
each order position. CLI bounds are 1–200 measured single-buffer rounds,
0–20 warmup rounds, and 0–20 mixed rounds. An outer launcher timeout remains
necessary; bounded call counts do not make a hung collective interruptible.
`--layout-only` reports layouts/runtime identity without allocating payload
buffers. Successful native reports use `status=completed`, matching the
campaign worker contract. `full_array_validation_passed` is false for
layout-only reports and true only after the measured workloads' complete
receive arrays pass validation on every rank. Existing output or temporary
files are rejected. One shared-memory
node and the production double-precision datatype are required. Each rank
records its hostname, Kokkos configuration and selected GPU/MPI environment
variables; the launcher remains responsible for verifying twelve distinct
GPU tiles.

## Layouts, routes and validation

Volume ownership partitions `19*n^3` cells over the actual MPI size;
surface ownership partitions `6*n^2` rows. Each point has 31 doubles, stored
point-major. Counts/displacements use exactly the integer cell/row
partition in `coupled_mpi.cpp`; n16/n32 therefore have unequal volume
counts on twelve ranks. The JSON retains every count and displacement.
Every rank holds persistent separate local-send/global-receive device and
`Kokkos::HostSpace` buffers, reused for all groups at one resolution.
HostSpace is recorded explicitly; pinned allocation is not claimed.

Every round measures all three routes in a deterministic rotating/reversed
order covering all six permutations:

| Route | Timed operation |
| --- | --- |
| `device` | Device-pointer `MPI_Allgatherv`, then a separately timed Kokkos completion fence |
| `host` | HostSpace-pointer `MPI_Allgatherv`; no copy included |
| `staged` | Local device-to-host copy and fence, host collective, global host-to-device copy and fence |

Input values encode the global element index, owning rank and changing
epoch as exactly representable integers. Receive buffers are poisoned
before each workload. The staged host send is also poisoned, so a missing
device-to-host copy cannot pass. After every measured and warmup workload,
all elements of each final receive buffer are checked for exact equality;
staged validation checks both host and device copies. A mismatch or
nonfinite/negative timing aborts the job rather than writing a passed
report. Device validation is a consuming kernel followed by a fence; the
benchmark still relies on the MPI implementation's blocking device-buffer
completion contract. A Kokkos fence alone is not evidence about an MPI
implementation's internal queues.

Optional `rk_mix` uses the original accepted-step collective order,
with `0=volume` and `1=surface`:

```text
1 1 0 1  1 1 0 1  1 1 0 1  1 1  0 1  1 1
```

This is four volume and fourteen surface gathers. Intermediate gathers use
identical send payloads; the final volume/surface arrays are validated.
The individual-buffer groups validate the same routes/layouts separately.
No geometric, interpolation or RHS work is inserted between mixed calls,
so the sequence is a communication control, not a synthetic evolution
throughput measurement.

## Timing and interpretation

`barrier` mode places an MPI barrier **before every individually timed
operation**, outside that operation's clock. `no_barrier` adds none inside
the measured group. Both modes perform a group-start barrier and
correctness reductions outside their samples. Input preparation and local
full-array validation occur between workloads, outside operation timing;
they can affect the next sample's arrival pattern. Thus `no_barrier` does
not recreate production arrival imbalance.

Every sample retains every rank's timings:

- `mpi_seconds` measures the blocking MPI calls, excluding producer fences,
  explicit staging copies and post-call completion fences. This corresponds
  most closely to the production exchange bucket.
- `d2h_seconds` and `h2d_seconds` include their explicit copy and completion
  fence. `completion_fence_seconds` records the direct-device post-call
  fence separately.
- `operation_seconds` sums the complete individually timed operations.
  `producer_fence_seconds` and `barrier_seconds` are excluded from that sum.
  `sequence_seconds` is elapsed wall time across the call sequence,
  including those producer fences/barriers and loop/timer overhead.
- `validation_seconds`, `checked_values` and `mismatches` describe the
  untimed full-array checks.

No per-sample reporting collective is inserted into a group. Timing arrays
are gathered after all measurements. JSON summaries first take the
**maximum across ranks for each sample**, then compute minimum, median,
linearly interpolated p95 and maximum for each route. Independently
maximized component timings are not an additive decomposition of the
maximized operation time. Rank clocks are never assumed synchronized.

Compare paired route samples and their spread before interpreting a
discontinuity. The recorded payload is a receive-buffer size, not wire
traffic; no network bandwidth is inferred. Shared resources, allocation
type, input preparation and rank arrival remain possible contributors.
A Serial backend uses HostSpace for both named memory routes and cannot
establish GPU transfer behavior. Aurora measurements and source/binary/
launcher binding are still required before changing the production path.
