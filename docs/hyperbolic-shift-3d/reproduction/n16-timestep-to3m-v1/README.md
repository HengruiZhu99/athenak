# n16 literal dt/dx experiment, stopping at 3M

Initial user-requested direct experiment: run only n16 puncture initial data to coordinate
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

## Updated timestep sweep

After the first two attempts, the user requested lowering `dt/dx` down to the
original `0.0444` and explicitly checking reproduction with the optimized code.
The expanded controller policy admits `dt=.0045,.00225,.001125,.0005`; the previous
policy remains restricted to its original two steps. The next priority is a
`dt=.0005` reproduction split at coordinate `.15` and `1`, with horizon
`lmax=12, ntheta=24`, matching the earlier run. Both numerical caps are inactive
at this step. Compare state and RK compensation against the retained original
checkpoints; the DT1 identity itself differs intentionally from V11. Then test
`dt/dx=.1` to narrow the stable range. The target remains n16 through 3M.

The evolution executable and its original source manifest remain frozen. The
updated launcher is separately committed and hash-bound in each new job; there
is no rebuild or change to the evolution binary for this policy extension.

## Verified results so far

| dt/dx | dt | Last accepted t/M | Outcome | Job |
|---|---|---|---|---|
| 0.4 | .0045 | .108 | Invalid scri metric, stage after step 8 | 8988261 |
| 0.2 | .00225 | .14175 | Invalid scri metric, stage after step 21 | 8988303 |
| .0444444 | .0005 | 3 | Completed, original result exactly reproduced | 8988342 |
| 0.1 | .001125 | pending | Submitted after reproduction | 8988417 |

At both .4 and .2 all 1536 scri surface metric tensors have a negative eigenvalue
in the captured rejected stage. This is a numerical failure, not a horizon
quality flag. The applied penalty estimate stayed below .5. At .4 the angular
coefficient was 1.28166975375; at .2 it remained 2. Lapse-floor interventions were
zero. The .2 report's ordinary diagnostics belong to its last recorded time
`.10125M`, not its later last accepted time `.14175M`.

The original-step run exactly reproduces all saved scientific history columns
(excluding wall time), and all volume/surface state and RK compensation bytes
at both `.45M` and `3M`. Header configuration is identical except the intentional
DT1 magic. Both caps are inactive: penalty estimate .384, angular estimate
.381446848891, angular epsilon 2. Null surface maximum through the two segments
is 1.22129211454e-15; ghost maximum 1.55431212293e-15. No floor interventions.

The native runtime total is 446.628 seconds (7.44 minutes), versus the retained
1375.468 seconds (22.92 minutes), an observed 3.080x speedup across different
allocations. This includes diagnostics and both native startups but excludes
queue and launcher verification overhead; it is not a same-node controlled
benchmark. Inclusive throughput is 713254 actual evolved zone-cycles/s.

This establishes preservation of the original evolution by the performance
changes. It does not establish convergence or repair the original accuracy
limitations. At 3M the complete horizon candidate has radius .07519062316 and
expansion RMS .00139188621, exceeding the .001 acceptance threshold. Its radius
can be used provisionally; accepted mass and exterior constraints are absent
at that sample. The earlier 1D comparison and resolution mismatch remain.

The capped large-step variants change the boundary semidiscretization, so their
failures cannot be attributed solely to the larger dt, nor do isolated penalty
bounds prove stability of the coupled geometric/regularity system.

See `comparison.png` for the first failed attempts, `reproduction.png` for the
reproduced 3M trajectory against 1D, and the per-job verification files for hashes.
The 1D reference is the preserved N513 result; its radial spacing is 5.76 times
finer than the 3D core. Missing measurements are not interpolated or filled.
