# Next evolved attribution control

The radial18 initial-only result isolates a large spatial truncation error in
.90–.95. It does not establish that this accounts for the evolved exterior
constraint plateau. Continue the already submitted radial12 reference-v2 run
9038410 through its authorized disposition; do not alter its dt, source,
configuration, closure, restart policy or allocation.

After preserving its disposition, the useful bounded comparison is:

| Case | Grid | Requested dt | Target |
|---|---|---|---|
| preserved live reference | radial12 | .000125 | existing 10M run |
| time-refinement control | radial12 | .00003125 | fresh .6M |
| radial-refinement control | radial18 | .00003125 | fresh .6M |

The latter two are a proposed next experiment under the persistent goal, **not
submitted jobs**. Qualify the refined grid's characteristic limit first. Use
the same native scientific binary if its immutable runtime configuration
support suffices; freeze and hash-bind each distinct input and exact identity.
The physical frame, initial slice, inner SMR, angular grid and final scri shell
remain identical. Effective boundary penalty stays fixed; its numerical CFL
changes with dt and must be recorded. No restart from the initial probes or a
different formulation. The original reference-v2 data remain immutable.

Use fixed .1/.3/.6 cuts and the same native constraint operator, horizon flags
and masses, null and cumulative pole/clock histories, floor/sign counters,
radial field/constraint/gradient profiles, and native wall timings. Sampling
at .025 coordinate-time intervals would provide .075M rows through .6M,
including the old .463875M failure neighborhood. Compare the two equal-dt
cases for spatial effects; compare the two radial12 cases for temporal
sensitivity. Changing the grid also changes radial interface placement; do
not label this a proof about a single stencil in isolation.

Refining only this radial band raises checkpoint storage from858262703 to
951454990 bytes (~10.9%) while leaving232 Cartesian SMR leaves unchanged.
This is substantially cheaper in state size than uniform 3D refinement, but
the timestep penalty can dominate total cost. Current four-node radial12
runtime is ~.195s/step, and no refined-grid GPU timing exists yet. Do not turn
cell counts into a claimed speedup. Use the measured timing and spatial error
reduction to choose the later full10M/resolution family. No claim of fourth
order or a resolved boosted puncture follows from this initial control.
