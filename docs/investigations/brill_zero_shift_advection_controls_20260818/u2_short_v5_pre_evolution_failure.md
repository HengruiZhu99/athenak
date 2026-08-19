# U2-short v5 pre-evolution failure

Slurm job `57253502` was a separate one-GPU `shared_interactive` allocation.
Fresh CUDA/MPI configure, build, and all five focused tests passed. AthenaK then
rejected `-r ... -i ...` before loading the restart because `CheckBlockNames()`
did not recognize the immutable internal `z4c_restart` and
`amr_history_restart` blocks already present in the restart parameter stream.
No U2 PDE step ran.

The v6 parser recognizes those two internal names syntactically. Existing later
checks still fail closed on fresh carrier injection or any post-capture carrier
mutation. A production-path restart-plus-input regression and the negative
carrier tests pass locally.
