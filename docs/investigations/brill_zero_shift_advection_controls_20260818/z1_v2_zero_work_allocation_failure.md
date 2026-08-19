# Z1 v2 zero-work allocation failure

Slurm job `57251682` was a separate one-GPU `shared_interactive` allocation.
It was granted on `nid001017` and immediately relinquished before configure,
build, or AthenaK execution because the staged checkout was detached while the
run preflight required `HEAD` to have an upstream tracking branch. The v2 root
contains no `build/`, `run/`, or `evidence/z1/` directory.

The v3 successor keeps the source commit and all numerical/resource bytes
unchanged, but stages the already-pushed branch as an attached clean checkout.
