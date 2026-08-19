# U2-short v3 pre-evolution failure

Slurm job `57252715` was a separate one-GPU `shared_interactive` allocation.
It reused the authenticated v3 executable and attempted the exact cycle-4096
restart, but AthenaK rejected the supplemental input before restart loading
because it introduced a `<mesh_refinement>` block absent from the restart input
schema. No PDE step or O2 beta-advection evaluation ran.

The successor supplies the same exact replay-compatible source ID through a
replay-only environment input. Parameter and environment declarations must
match if both are present; default replay source binding remains strict.
