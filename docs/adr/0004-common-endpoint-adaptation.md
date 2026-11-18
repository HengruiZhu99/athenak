# Regrid on the coarsest time grid

Evaluate spatial refinement criteria only after every block has accepted a complete base interval. The driver increments the coarse cycle, completes refinement and transfer, then writes synchronized outputs. Restart files include the accepted ownership and the refinement ages used for cooldown decisions.

Evaluate field criteria on current accepted values and collect flags by global block ID. The native refinement transfer uses a contiguous old layout. Stage that layout when flags can cause refinement or coarsening. Preserve accepted fields and persistent diagnostics during placement changes. After spatial transfer, rebuild the halo, agreement and scri plans, construct the new dependency graph, and select a placement subject to the configured block capacity.

New Cartesian leaves inherit their relative spatial levels, subject to the overlap promotion. Explicit spherical settings continue to apply to each physical radial region. Recompute automatic assignments from the new topology and accepted state. Validate all donor edges and the stability bounds against the original base interval before advancing.

A fixed-mesh temporal convergence measurement isolates integrator error. An adaptive comparison must align the physical regrid times and check the resulting topology before comparing fields.
