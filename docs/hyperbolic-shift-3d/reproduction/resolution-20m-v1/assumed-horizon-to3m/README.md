# Exterior constraints with the candidate horizons assumed valid

At the user's request, use the recorded complete candidate radii even when the
FastFlow expansion tolerance is missed. The original histories did not retain
exterior norms at those times: the native diagnostic code computed full-domain
norms with cutoff zero, then discarded the exterior accumulator. Relabeling
those NaNs cannot recover a time series.

The saved N16 and N24 DT1 checkpoints permit a fresh diagnostic-only measurement
at3M. `hyperbolic_shift_3d_checkpoint_constraints` uses the native checkpoint
loader, boundary/halo refresh and device constraint kernel with an explicit
coordinate-sphere cutoff. It takes no evolution steps, does not search for a
horizon and writes no checkpoint. This separate local target does not modify
the frozen Aurora executables or running campaign. Physical source checkpoint
hashes are checked before and after measurement; all remain unchanged.

| Exterior normalized RMS at3M | N16 | N24 | N24 change |
|---|---:|---:|---:|
| Hamiltonian | .02292036827 | .02193294775 | -4.31% |
| Radial momentum | .04834331616 | .04815419029 | -0.39% |
| Conformal invariant momentum | .03548253861 | .04223195194 | +19.02% |
| Z4 scalar Theta | .0005795102002 | .0005713673634 | -1.41% |
| Radial Z4 vector | .006860852341 | .006094287035 | -11.17% |

Cutoffs are.07519062316027353 and.07502425967234379. These are each run's own
provisional mean coordinate radius, not identical integration domains. The
invariant momentum norm uses the conformal spatial metric; the radial momentum
is the coordinate covector component used in the 1D comparison. They need not
show the same resolution trend, especially without exact spherical symmetry.
No unique source of their differing trends is diagnosed by this measurement.
These results do not show uniform exterior-constraint convergence. The larger
full-domain reductions reported earlier therefore cannot be taken as equivalent
exterior improvements.

Validation: the same analyzer reproduces the saved accepted N16 exterior norms
at.45M and both runs' full-domain norms at3M to roundoff (absolute differences
below4e-15). `comparison.json` and `verification.json` retain the differences,
source hashes and checkpoint immutability result. `execution.json` records all
three exact commands, checkpoint/binary hashes and runtimes. Build with
`cmake --build build-hyperbolic-mpi --target hyperbolic_shift_3d_checkpoint_constraints -j2`.
The executable arguments are `n checkpoint dt cutoff fresh_output.json`.
`plot.py` verifies diagnostic parity before regenerating the figure.

The isolated3M values are recovered, not interpolated across the missing earlier
samples. Recovering those earlier missing values requires saved intermediate
states or a diagnostic replay; the three measured snapshots here do not supply
them. The N24/N32 long-evolution campaign remains unchanged.
