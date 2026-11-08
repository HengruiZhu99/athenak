# Bounded post-initialization capture

This addendum accompanies the two-node timing contract without changing the
already qualified timing/observer sources, their build proof, or its pinned
contract bytes.

Run a separate ordinary frozen `f3` process from the same cycle-2486 source:
`time/nlim=2486 time/tlim=2 time/ndiag=1000000000`. Use the same24-rank binding
and environment as the three continuation cases, in the `init-only` directory.
The original Driver performs initialization, zero RK cycles, and normal final
outputs. The resulting `native_n12_5M.00008.rst` captures actual owned state
after restart initialization. It is never the source of a timed continuation.

`Driver::Initialize` regenerates dependent values. `Diagnostics::Prepare` then
forms residuals, fills ghost/halo values and computes diagnostic normal
corrections; it does not project or mutate owned31-field/eight-record state.
`Driver::Finalize` writes history/horizon output before writing the restart.
The FastFlow seed and output counters therefore change independently of owned
state, and are not claimed unchanged.

The reader checks every padded31-field value against the source with the
existing short gates: absolute1e-12 plus5e-10 times reference magnitude, with
chi absolute1e-14. It reports bitwise equality and all differences separately.
All eight chi records must remain bitwise exact. No initialization tolerance is
selected from the result. This capture supplies actual native initial data for
the separate legacy post-load comparison; source payload equality alone does
not establish post-initialization equality across implementations.

There are still only three evolved native controls, each70 complete cycles.
This additional case has zero evolved cycles and is not a performance sample.
