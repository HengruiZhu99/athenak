# U2-short v4 zero-work allocation failure

Slurm job `57253231` was a separate one-GPU `shared_interactive` allocation.
It stopped before configure because the inline fresh-build preparation sourced
the external Perlmutter profile before exporting its required `COLLAPSE_ROOT`.
No build, restart load, or PDE step occurred. The v5 successor exports the
exact staged campaign root before sourcing the unchanged profile.
