# Experimental native driver entry

Opt-in settings in the existing `<time>` block:

```
integrator = rk4_classical
subcycle_max_ratio = 2
subcycle_interval_cap = 0.002
subcycle_cycle_unit = synchronization
```

The ratio defaults to zero (disabled). Supported scope is single-rank vacuum VC
Cartoon with the qualified global telegraph gauge and zero shift/damping. The cap
is a maximum synchronization interval, further limited by all levels' spatial and
source bounds, scheduled physical output times and final time. Stability/corrector
failure can shorten the accepted interval.

`ncycle`, cycle-based outputs, and cycle-based regridding mean synchronization
cycles in this explicitly selected mode. `subcycling_intervals.csv` records
accepted leaf-block substeps separately. Existing campaign inputs must not be
changed to this mode until scientific/performance qualification is complete.

At each accepted interval the driver rebuilds native accepted boundaries and ADM,
runs final-stage diagnostics, and proceeds through existing stopping/AMR/output.
The hierarchy owner is discarded after topology changes and rebuilt from native
leaf fields on the next interval. This AMR connection is not yet tested against
actual refinement/coarsening events.

Local reproducible fixed-hierarchy test:

```
python scripts/subcycling/test_live_subcycling.py /path/to/athena /new/output/directory
```

Checks ratios1 and2, native versus frozen final fields and matched-endpoint
checkpoint/restart continuation, history time and accepted block-step counters.
It is not a Brill reproduction, dynamic-AMR qualification or a speedup benchmark.
