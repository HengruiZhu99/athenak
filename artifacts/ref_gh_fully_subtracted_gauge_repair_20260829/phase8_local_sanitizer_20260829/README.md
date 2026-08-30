# Phase-8 local sanitizer discriminator

This compact bundle records a focused local discriminator following the Aurora
PVC write faults in jobs `8791456` and `8791465`.  It is not an Aurora result
and it is not positive-time trumpet-stability evidence.

The first Debug/Serial run at source commit `30ee309a` stopped before evolution
because UBSan diagnosed an invalid `bool` load while copying an
`OutputParameters` object.  The temporary object in `Outputs::Outputs` was not
value-initialized, so format-specific parsing left unrelated members
indeterminate before the implicit copy constructor copied the complete struct.
The preserved diagnostic is in `ubsan_initial_failure.txt`.

Commit `c19c6058` value-initializes that temporary object.  No Ref-GH equation,
numerical parameter, View extent, task dependency, or output definition was
changed.  With that correction, a fresh ASan+UBSan build with Kokkos bounds
checking completed the exact matched STANDARD/gamma0=1/gamma2=1/gauge-on
case for one full RK4 cycle on a `16^3` Serial grid.  The run reached
`t=0.01`, wrote its final histories, and exited zero.  `run_n16/run.log`
contains no sanitizer or Kokkos-bounds report.

The reduced grid is deliberately only a lifecycle and bounds discriminator.
It cannot qualify the 96^3, 12-PVC Phase-8 evolution or the GPU-aware MPI path.
A local eight-rank MPI sanitizer attempt was not usable because the local
OpenMPI launcher itself entered the Intel DRM `drm_sched_entity_flush` wait
before spawning ranks; it was terminated and is not treated as test evidence.

