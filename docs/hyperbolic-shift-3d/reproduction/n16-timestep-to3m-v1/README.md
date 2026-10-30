# n16 literal dt/dx experiment, stopping at 3M

User-requested direct experiment: run only n16 puncture initial data to coordinate
`t=1` (`3M`, mass `1/3`) with `dx=.18/16=.01125`, initially `dt=.0045`
(`dt/dx=.4`). On a recorded numerical failure, restart from initial data once at
`dt=.00225` (`dt/dx=.2`). Infrastructure interruptions are not numerical failures;
an admitted walltime checkpoint can resume the same attempt. Stop and pause after
completion or both numerical failures. No n24/n32 or 10M extension is authorized
by this bounded follow-up.

The target `hyperbolic_shift_3d_mpi_radial_batch_timestep_experiment` inherits the
fast field-batched radial, cached boundary, sparse-exchange implementation. It
bypasses only the conservative sum-of-directions bulk speed guard (still reports
that CFL); the `dt/.02 <= .25` relaxation guard remains. Classical RK4, log chi/log
lapse, and the physical lapse floor `1e-10` remain.

The numerical scri coefficients `4/h` and `8/h` receive one common scale
`min(1, .5*(1-1e-12)/(dt*12/h))`. Normal finite-difference weights are unchanged.
The dependent closure variable is calculated from the resulting geometric rates.
Angular dissipation receives a common coefficient cap keeping its estimated
negative-real-axis RK4 rate at most `2.2*(1-1e-12)` (reserving .5 of a 2.7 budget).
This is a bound on isolated numerical terms, not a proof that the complete
non-normal, coupled discretization is stable. Smaller steps can change these
numerical coefficients: this is not a clean temporal-convergence experiment.

`HYP3D_MPI_DT1` checkpoints cannot be read by V11 executables or vice versa.
Within this experiment the checkpoint also binds dt, mesh, angular coefficient,
normal operator, log-variable policy and floor. Source and binary hashes bind
each job. The controller's `timestep_experiment` phase records this explicit
exception to the normal preflight gate; it does not claim earlier GPU parity
qualifies the newly capped discretization. Existing production targets retain
the original timestep guards and uncapped coefficients.

The native penalty test evaluates a perturbed scri surface at an uncapped step
and both requested steps, checks finite rates, applied penalty bound, exact-null
projection and compatible dependent rates. This is an operator check, not an
additional evolution stability pilot. Results are in `penalty-unit-test.txt`.

Aurora resources: CompactBinaryMerger, one debug node at a time, 12 MPI ranks on
12 GPU tiles, SYCL/MPI Release build with `make -j64`. All campaign files remain
under the home root with the existing 5 GB storage guard. Large failed states and
checkpoints will be copied and hash-verified before remote cleanup.
