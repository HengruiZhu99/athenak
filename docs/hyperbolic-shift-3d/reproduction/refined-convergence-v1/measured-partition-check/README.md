# Static SMR ownership qualification

`../prepare_partition_probe.py --output FRESH_DIRECTORY` builds a separate
analysis-only driver against the unchanged native library, with the existing
`HYPERBOLIC_MEASURED_PARTITION` reader enabled. It changes only the constructor's
ownership-file argument and rejects nonstatic policies. It does not invoke the
uniform-grid `Repartition` implementation. Production files, binaries and the
active Aurora cases remain unchanged.

The preserved command records under `production/` and `unequal/` reproduce the
two-step, two-rank N12 experiment. For a fresh probe build, replace their output
prefixes and analysis binary path, and set `HYP3D_ANALYSIS_PARTITION` to that
build's `unequal.partition` for the unequal case. Preserve all other arguments.
The baseline uses the unchanged production binary. Never overwrite originals.

The complete373986735-byte checkpoints are byte-identical, including fields,
RK compensation, geometry, tracker, counters and checksum. The native RF2
finite/SPD/FNV verifier passed; the verification JSON binds the exact hash.
Reduction-only norm differences are below2.1e-16 absolute. These are local
SERIAL/MPI results, not GPU or long-time evidence. The deliberately uneven
one-third/two-thirds split tests ownership invariance, not performance.

`../propose_measured_partition.py` uses the two completed Aurora segments to
produce a candidate48-rank descriptor. It fits piecewise-constant local
RHS+halo cost in segment01 and checks the proxy on segment02. Observed maximum
local-cost/mean is1.261 and1.257; the proxy's new cuts predict1.00003 and1.00648.
This is not a speedup: communication, halo dependencies, occupancy and caches
change with the cuts, and closure/collectives are omitted. No Aurora allocation
has used this candidate. Adoption requires a separately bound paired GPU
runtime experiment with the same source/configuration/steps and checkpoint
parity, preserving the current physics controls.
