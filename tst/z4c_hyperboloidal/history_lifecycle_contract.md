# Native executable history lifecycle gates

This test uses the ordinary AthenaK executable and native input, with no custom
main, field replacement, admission override or alternate integrator. It evolves
the preserved N12 initial data for one classical RK4 cycle with the exact fixed
timestep. At each requested decomposition (default two and eight ranks), compare
history enabled with history disabled through the ordinary output cadence.

With `dcycle=1`, the original Driver writes history initially, after cycle one
and once more during finalization. Require the three coordinate-time/dt rows
`(0,dt)`, `(dt,dt)`, `(dt,dt)` and exact equality of the latter duplicate.
The cached FastFlow summary must contain just the two distinct accepted states.
Every reported integral must be finite and both volumes positive. Cross-rank
numeric output uses `1e-12 + 5e-10*abs(reference)` to allow changed reduction
order; record its maximum absolute difference.

Native shared restart files are written after diagnostics initially and at the
accepted state, including the ordinary final duplicate. Compare every stored
bulk/scri value and every ChiRecord member bitwise between history enabled and
disabled, and across decompositions. The comparisons include native topology,
atlas and numerical-policy identity, cycle/time/dt and complete fixed-stride
payloads. Parameters added by FastFlow and output counters may differ; they are
not evolved state. Preserve source, object and executable hashes and reject a
run if those inputs change while testing.

This checks scheduling, output caching and state preservation. Independent
constraint, quadrature and horizon oracles establish numerical correctness;
the full saved N12 history and GPU performance remain separate requirements.
