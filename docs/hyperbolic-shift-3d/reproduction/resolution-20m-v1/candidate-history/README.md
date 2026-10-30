# Measured exterior-constraint histories using candidate horizons

The user requested a time-series plot, accepting complete provisional horizons
as exterior cutoffs. Missing values were discarded in the original histories;
they cannot be recovered by plotting or interpolation. This bounded diagnostic
replay reconstructs the same N16/N24 DT1 evolution through3M, retaining exterior
norms whenever the finder returns a complete finite candidate with0<r<1.
The original horizon acceptance flags and expansion residuals remain unchanged.
There is no change to the evolution equations, timestep, floors or penalties.

N16 starts from the verified original.45M DT1 checkpoint and its original seed
sidecar, with dt=.0005. Its earlier0–.45M exterior history already exists and is
merged by the plotting script. N24 repeats the original initial data with
dt=.0003333333333333333. Both use diagnostics interval.05, horizon lmax12/ntheta24,
tolerance.001 and angular epsilon2. Each new checkpoint must be compared against
the original3M checkpoint before presenting the recovered series as a faithful
replay. Two short local timing attempts were intentionally stopped; their partial
files are not completed histories. N16's recovered.45M and.6M diagnostics agree
with the original records. Local runs were too slow; the full replay uses Aurora.

## Frozen remote replay

Remote root `/home/hzhu/hyperbolic-core-shell-20261028`.
Source `source-candidate-history-v1`, copied from the direct-chi/lapse scientific
snapshot with explicitly overlaid diagnostic target and controller changes.
Current manifest `source-candidate-history-v1-clean-manifest.json`, SHA256
`d0ec3281be30e03af4d7fadc12be7fcc1cb92027dc4ae534b7a1a5afd878bfe1`.
Use `source-candidate-history-v1/tst/hyperbolic_shift_3d/aurora/control.py`.
Current plan `campaigns/n16-n24-candidate-history-v1-attempt02/plan.json` executes N16 then N24
on one debug node using12 MPI ranks/GPU tiles, CompactBinaryMerger. The separate
target is `build-sycl-mpi-candidate-history-v1/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_mpi_candidate_diagnostics`.
`build.sh` uses the existing SYCL/MPI settings and `make -j64`.
The bound binary SHA256 and exact source/input bindings are recorded by bind.
Never submit until the binary is built, binding exists, no user debug allocation
is active/queued, and the terminal predecessor's matching root lock is released.

N24 long-evolution job8990986 failed at5.868999999999684M with invalid accepted
volume algebraic projection. Its checkpoint is unadmitted and must never resume.
Its evidence has been copied locally and verified. The next priority is this
requested diagnostic replay; after it is collected and plotted, proceed with
the prepared N32 campaign under the original resolution protocol.

## Collect and plot

Collect terminal scheduler/worker records. Copy the whole replay campaign to
`output/hyperbolic-core-shell/candidate-history-aurora-v1/`, verify every artifact
against worker hashes/sizes, then copy the two history/report files and compact
submission/binding/execution evidence into this document directory. Preserve
the original checkpoints in their local evidence directories.
Run `verify_checkpoints.py` and `plot.py`. The first requires exact DT1 headers
and compares all state/compensation components, recording whether bitwise equal;
it allows scaled differences below1e-8. The second requires both completed3M
histories, finite exterior values, and agreement with original full-domain norms,
minimum fields and candidate radii at shared times (rtol1e-7/atol1e-11). Inspect
and explain a failed comparison rather than relaxing its tolerance silently.
No missing-value interpolation is permitted. Display the resulting time-series
figure, commit/push the evidence, and mark the replay completed in the remote
campaign state. Then release the matching terminal lock and start N32 as already
authorized. Keep the hourly monitor until N32 is complete or numerically failed.

If the replay fails or stops early, preserve its output and diagnose the cause;
never present a partial trajectory as a completed3M series. Do not automatically
reuse a failed checkpoint. Keep the home footprint below5GB and verify local
copies before purging remote large artifacts. The original scientific binaries
and the prepared N32 inputs remain unchanged.

Replay job8991110 was submitted; binary SHA256 is
2888647aa38dfd7cb302a7f60832ea82bc22f78d1036e9595d4a1238d3fa3654.
Source/binary/input binding and storage checks passed (3.581GB at binding).
All54 existing controller tests passed; altered replay target, dt and angular
coefficient were separately rejected. The hourly monitor now prioritizes this
replay before the already authorized N32 run. No complete time series is claimed
until the replay and validation finish.

## Startup manifest repair

Job8991110 failed before starting any MPI evolution. The manifest accidentally
included three generated Python bytecode files inherited from the source-copy
operation. Importing the worker regenerated control.cpython-311.pyc, correctly
triggering the source-change guard. The first manifest and failure trace remain
preserved. A new manifest excludes only those three __pycache__/*.pyc files;
every other source hash and the evolution binary hash were explicitly verified
unchanged. A fresh attempt02 plan retains the same inputs and parameters.
This is an infrastructure retry, not a numerical-failure restart.

Replacement job8991129 was submitted using the fresh attempt02 plan. Worker
import followed by repeated full binding validation passed before submission.
Task storage at binding was3.497GB, with the requested output/atomic-write reserve.
