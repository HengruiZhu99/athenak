# GPU profiles preserve the tested trajectories

Aurora debug-scaling job **8988222** completed all six cases, scheduler F/exit0.
Source `c95023d2c8c52361a1450b04306a0227ba09d4ef` is separately frozen from
the unchanged c530 evolution controls. The launcher is the queue-isolated bundle
from `a551fc12c`. The allocation used one node and twelve distinct GPU tiles.

Independent collection verifies every worker artifact and bound input. Both
comparisons are bitwise equal over all 9,000,416 checkpoint payload doubles:
old full gather versus new disabled sparse, and new disabled versus enabled
sparse. Each oblique run uses dt=1/4096 for 20 steps. Enabled profiles have three
finite frames of 1,462 samples; the separate Minkowski run preserves sampled
values/derivatives at roundoff. This is a short observer-invariance result, not
new long-time or nonlinear stability evidence.

`offline-audit/results.json` independently reconstructs the final oblique and
Minkowski samples on Aurora from their completed GPU checkpoints. The standalone
reader executable is verified against the frozen build receipt. Both CSV outputs
are byte-identical to the online final frames; local collection rechecks the
checkpoint, online CSV and snapshot hashes and the exact bytes. This reader
never advances, floors or repairs an evolved state.

For the enabled oblique case, maximum reported setup was 0.01355s; collection
was 0.02639s and sampling/write 0.10819s over three frames. Application costs
are already included in diagnostic timing. End-to-end disabled/enabled times
were 17.0033/17.1554s in this single sequential pair. These are overhead
measurements, not a statistically qualified speedup or long-run I/O estimate.

The large checkpoints and CSVs are retained locally; verified Aurora cleanup
may remove their remote copies. `probes-independent-audit.json` records their
hashes. No horizon criteria, lapse floor, scri closure or equations changed.
