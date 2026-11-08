# Coupled native AMR evolution gates

These gates precede the first run. `amr_evolution.cpp` starts from the original
reference problem generator without manufactured field or history replacement,
and calls the original `Driver::Execute` once for four complete RK4 cycles.
The input constructs a static-time adaptive module, then selects dynamic mode
only for the test Driver. This deliberately bypasses the constructor admission
gate; passing this harness alone does not enable ordinary adaptive inputs.
There are no timestep, RK-weight, state-update or stage-clock overrides.

At accepted cycle 1, a native user criterion refines the same three parents as
the transfer oracle: the Cartesian corner, first +X shell and outer +X shell.
At accepted cycle 3, it derefines only their descendants. Native closure,
equal-weight load balance, transport, plan rebuilding and timestep admission
remain authoritative. Cycle-start/accepted topology must be
106, 127, 127, 106, 106 leaves and 6, 9, 9, 6, 6 surface patches. Each actual
regrid increments the mesh sequence once; no other cycle increments it.

A real `hyperboloidal_chi` slope criterion runs before the user callback. The
callback sets the final flags, so the slope threshold does not determine this
test's mesh schedule. After every accepted step the observer poisons all bulk
ghosts. The callback checks every nominal-cell axial chi neighbor, taking the
outer radial endpoint from s0, after native accepted-halo refresh. This exercises
the derivative criterion, its origin, view rebinding and accepted-halo lifecycle.

Before every stage, poison RHS/residual/status workspaces and bulk ghosts.
After every stage, every evolved bulk point must have successful RHS status and
finite characteristic data, and every one of the 31 bulk/scri state and RHS
slots must be finite. Metric expansion, positive chi and physical lapse, and
the projected scri clock must hold. Surface compatibility must remain <=1e-7;
surface RHS clock error and characteristic CFL are recorded without a new CFL
rejection threshold. All 16 stage clocks must match native classical RK4.
At every cycle start, the four rebuilt RHS work lists must cover each local
evolved point exactly once, and the chi criterion must reference current u0.

The initial four cycles are expected to have zero chi-floor/nonpositive/nonfinite
events and zero corrections. All eight history members are included in complete
snapshots. Global record additive totals and extrema are captured before each
AMR call and must survive that accepted-step operation exactly. Nonzero seeded
record redistribution is covered separately by `amr_lifecycle.cpp`.

Complete snapshots at cycles 0 through 4 contain all 31 fields and eight history
values keyed by logical leaf identity and evolved local index. Per-leaf bounds
independently enumerate the required bulk points; every outer face must contain
all 12x12 scri points. The runner rejects missing/duplicate/foreign points and
requires bitwise identical canonical snapshots at two and eight local CPU MPI
ranks. Eight-rank snapshots must include ranks with no scri patches. This is a
bounded coupled AMR consistency test, not a convergence, long-time stability,
GPU performance, restart or diagnostic validation.
