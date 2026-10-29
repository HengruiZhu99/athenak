# Original-driver MPI timing audit

This read-only audit finds a sharp increase in recorded collective time as
the blocks grow. The current evidence does **not** identify an MPI defect
or establish a transport optimization. The experiment proposed below is
now implemented as a [standalone communication control](mpi-communication-benchmark.md);
local Serial/MPI checks do not establish Aurora performance. It preserves
the evolution equations and boundary discretization.

The three Minkowski runs use the same original executable, SHA-256
`6e7553b556dd9c4f8b56f81eb0f43c15cf0517a0b81776320a9a6c6d3c29ccef`,
and numerical source manifest
`3f367265e2e627aa770d0ae78be4fac912f4c0980a09471382cf9b295c677d7f`.
The [source-equivalence record](reproduction/aurora/preflight-v2/source-commit-equivalence.json)
binds that manifest to numerical commit
[`113aaff783008204e1ff8cbf2fb8bb5f2af2204c`](https://github.com/HengruiZhu99/athenak/commit/113aaff783008204e1ff8cbf2fb8bb5f2af2204c).
These runs use one Aurora node, 12 MPI ranks and 12 distinct GPU tiles.
They use the original inline scri sampling, without the later cached-jet,
algebraic-projection or cross-face angular candidates.

| Grid | Accepted steps | Native status | Total seconds | RHS seconds | Exchange seconds | Closure seconds | Median seconds/step |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| n16 | 979 | failed | 338.410779 | 48.749153 | 30.028635 | 242.081176 | 0.339992 |
| n24 | 941 | failed | 759.633298 | 47.979275 | 433.585601 | 252.934530 | 0.794020 |
| n32 | 929 | checkpointed | 1403.264722 | 56.355926 | 1024.378333 | 283.650286 | 1.487436 |

Median step times are the median of adjacent history increments
`delta(wall_seconds)/delta(steps)`, discarding the first interval; there are
8, 11 and 9 retained intervals respectively. They include full-step and
diagnostic overhead. Diagnostic cadence differs: n16 records every 0.05
coordinate time, n24/n32 every 0.025. These are bounded, unstable controls,
not a throughput scaling study or a convergence result. In particular,
n32's native checkpoint status coexists with failed worker scientific
checks and does not admit a restart or stable evolution.

The frozen inputs to this audit are:

- n16 [report](reproduction/aurora/stability-v1/output/minkowski.report.json)
  and [history](reproduction/aurora/stability-v1/output/minkowski.history.csv),
  with report SHA-256
  `590230abbb4e597e6bba8355a42dcb318c76a7217a2b916b8f2edead75a7b090`.
- n24 [report](reproduction/aurora/resolution-control-v2/output/minkowski-n24.report.json)
  and [history](reproduction/aurora/resolution-control-v2/output/minkowski-n24.history.csv); report SHA-256
  `c1570c4fdca35e86158642fe361bb0383daec315c178db2d01599338125e0ebd`,
  history SHA-256
  `a567fd8d3f389afc4a414b738b53fdc22198eec665ff67435b406dfe9c025182`.
  This completed case was first read while the follow-up campaign remained
  active; its [terminal evidence package](reproduction/aurora/resolution-control-v2/README.md)
  is now retained and verified.
- n32 [report](reproduction/aurora/resolution-control-v1/output/minkowski-n32.report.json)
  and [history](reproduction/aurora/resolution-control-v1/output/minkowski-n32.history.csv),
  with report SHA-256
  `01d12b097ef8800cd4fc69cc9ccd330fcea35245da9d2cd74e5036a291059e6d`.

## What the timers measure

The frozen [Gather, RefreshBoundary, RHS and RK implementation](https://github.com/HengruiZhu99/athenak/blob/113aaff783008204e1ff8cbf2fb8bb5f2af2204c/tst/hyperbolic_shift_3d/coupled_mpi.cpp#L74-L138)
defines the following accounting:

- **Exchange:** blocking `MPI_Allgatherv` on persistent Kokkos device
  pointers. The preceding `Kokkos::fence()` is outside this timer. It
  includes MPI buffer handling, collective progress and waiting for other
  ranks; it cannot distinguish internal copies, transport or arrival skew.
- **Closure:** boundary projection/evaluation, their host summary
  reductions, error/statistic MPI reductions, and two surface gathers.
  Those gathers are also counted in exchange.
- **RHS:** bulk GPU evaluation and its local summary reduction. Boundary
  work, local padded-halo interpolation, subsequent error/maximum MPI
  reductions and RK copies are outside this timer. Despite its name,
  `HyperbolicCompositeExchange::Fill` is local GPU work, not an MPI call,
  and is not charged to exchange.
- **Diagnostics:** includes another boundary refresh and therefore
  overlaps closure and exchange. Reported cumulative timer values are
  independently `MPI_MAX`-reduced and may come from different ranks.
  Total wall time also includes setup, copies and checkpoint/output work.

Consequently these buckets **must not be added or subtracted as disjoint
wall-time components**. Dividing a cumulative bucket by accepted steps
also amortizes initialization, diagnostics, checkpoint gathers and any
failed-stage work.

A normal accepted RK4 step makes four volume and fourteen surface gathers:
four volume/surface RK materializations plus five boundary refreshes with
two surface gathers each. The application performs these synchronously;
the source does not establish what overlap MPI may perform internally.

Each rank receives all `31 * 19 * n^3` volume doubles and all
`31 * 6 * n^2` surface doubles. Ownership is contiguous, using integer
partitions of active cells/rows over 12 ranks. Exact buffer sizes are:

| Grid | Global volume bytes | Local volume bytes, min–max | Global surface bytes | Local surface bytes |
| --- | ---: | ---: | ---: | ---: |
| n16 | 19,300,352 | 1,608,280–1,608,528 | 380,928 | 31,744 |
| n24 | 65,138,688 | 5,428,224 | 857,088 | 71,424 |
| n32 | 154,402,816 | 12,866,736–12,866,984 | 1,523,712 | 126,976 |

These are allocation/receive sizes, **not measured wire traffic**. No
network or bus bandwidth should be inferred from them.

## Proposed bounded communication control

Use a standalone Kokkos/MPI program with the same compiler, MPI, backend,
environment and one-node/12-tile placement. Preserve the exact datatype,
separate send/receive buffers, counts and displacements above. Compare:

1. Device-buffer `MPI_Allgatherv`, matching the existing call.
2. Persistent host-buffer `MPI_Allgatherv`, with copies excluded.
3. Explicit device-to-host send copy, host collective and host-to-device
   receive copy; report each component and total elapsed time. Record the
   host allocation type rather than assuming it is pinned.

Fence producer work before timing, bracket explicit copies correctly,
warm up every route, and alternate measurement order. Validate complete
gathered data against deterministic rank/index values outside timing.
Retain per-rank samples and per-iteration maxima, with bounded repeats;
gather timing records after each batch. First synchronize ready ranks with
a barrier outside the timed call, then repeat without the extra barrier
to assess sensitivity to arrival skew. An additional bounded sequence can
reproduce four volume plus fourteen surface gathers per synthetic step.
This sequence measures communication, not coupled evolution throughput.

Record binary hash, MPI version, Kokkos backend, runtime environment,
buffer types, payloads and tile affinity. A device-only discontinuity
would motivate investigation of the GPU-buffer path or collective
selection; similar host behavior would broaden that investigation;
barrier sensitivity would motivate examining arrival imbalance. None of
these outcomes alone would identify a vendor bug. Production changes
should follow the measured control, not precede it.
