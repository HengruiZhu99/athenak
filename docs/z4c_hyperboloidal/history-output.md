# Native hyperboloidal history output

The standard history output dispatches to the hyperboloidal diagnostic
coordinator when that physics module is present. For example:

```text
<output1>
file_type = hst
dcycle = 10
data_format = %.17e
```

The original Driver controls initial, periodic and final output. The module
prepares accepted-state derivative data and the complete fixed-origin N12
FastFlow horizon at that cadence. Repeated consumers can reuse the same accepted
state's diagnostic workspace. `user_hist_only=true` suppresses this automatic
hyperboloidal history when user-defined history is selected.
Cache validity includes the owned-state version, mesh-update sequence, cycle
and coordinate time. If one rank invalidates its state, all ranks refresh
together because its values may supply another rank's derivative or horizon
stencil. A final history row at an already-output state reuses the cached sums
and does not duplicate the FastFlow summary/shape record.

The file is `<job/basename>.z4c_hyperboloidal.hst`. It contains native coordinate
time and dt followed by these ten values:

| Column | Meaning |
| --- | --- |
| Volume | Full compact-coordinate quadrature volume |
| H-rms | Full Hamiltonian constraint RMS |
| M-rms | Full momentum constraint RMS, using the physical metric contraction |
| Theta-rms | Full physical Theta RMS |
| Z-rms | Full Z RMS, using the physical metric contraction |
| eVolume | Quadrature volume outside the complete harmonic horizon |
| eH-rms | Horizon-exterior Hamiltonian constraint RMS |
| eM-rms | Horizon-exterior momentum constraint RMS |
| eTheta-rms | Horizon-exterior physical Theta RMS |
| eZ-rms | Horizon-exterior Z RMS |

For each region, the native history writer sums rank-local volumes and weighted
squared constraints, then computes `sqrt(global_sum/global_volume)`. A rank can
contribute zero volume. Both global volumes must be positive, and all global
integrals must be finite and nonnegative. The coordinator reports a failed or
incomplete horizon collectively before history output.

Weights use the map Jacobian, unique-coverage core/shell blend and the physical
inner-start half weight of the outer nodal layer. Extra evolved overlap cells
have zero quadrature weight while remaining subject to validity checks. The
full harmonic surface determines the exterior mask. See
[the diagnostic plan](native-diagnostics-plan.md) for the mathematical conventions
and independent comparison gates.

Use 17-digit output for numerical comparisons. Time remains in coordinate
units; the pinned N12 convention converts it to M units by `time_M=3*time`.
Horizon summary/shape records use the existing FastFlow output separately.

Local CPU MPI validation passed at two and eight ranks. Synthetic weighted sums
produce identical output across decompositions, including ranks with no local
exterior contribution. Invalid global integrals fail collectively. Actual
initial-data diagnostics preserve every owned field, all eight ChiRecord
members, RK/RHS buffers and stage observations. Tests also cover cache reuse,
one-rank mutation/restoration and a change of cycle/time alone.

The ordinary executable's one-cycle test writes the original initial,
periodic and final history rows. Turning history on leaves all 24,704,784
stored checkpoint scalars bitwise identical to history off, at both initial
and accepted states; the final payload is also identical between two and
eight ranks. The largest cross-rank history difference is
`2.6201263381153694e-13`, from volume summation order. Evidence and executable
hashes are recorded in
[native-history-validation.json](evidence/native-history-validation.json).
These checks establish output integration, scheduling and state preservation;
independent constraint/horizon comparisons and GPU performance are separate
requirements.

A same-time process restart from two to eight ranks also preserves the complete
checkpoint payload bitwise and reproduces all ten history columns within the
short arithmetic gate. The adopted mass/areal-radius checks pass. The additional
compact-coordinate mean/minimum radius checks introduced by the restart test
fail: a fresh search advances the saved monopole by one flow update, changing
the compact mean radius by `1.9020776903522219e-7` in M units. Restarting on the
same two ranks produces the same compact radii, while the areal-radius change
is only `5.448725914902752e-10` M. The original failure is retained separately
from the adopted areal-radius check; no tolerance is changed.
An independent replay of the available predecessor FastFlow source reproduces
both searches exactly: the prior-state monopole takes 11 iterations and the
saved current-state monopole takes two, with matching area, compact radii,
mass and expansion RMS between old and new solvers. This identifies the compact
radius shift as existing solver behavior under its unchanged stopping criteria.
It does not resolve the separate historical N12 binary provenance limitation.
The exact solver replay is documented in
[the warm-start review](fastflow-warmstart-review.md).

The same-object `restart::Unpack` cache checks pass at two and eight ranks:
cached constraints refresh exactly once and recover their original values and
complete state bitwise. See
[native-history-restart-validation.json](evidence/native-history-restart-validation.json)
for the separate passed and failed gates, expansion diagnostics and provenance.
