# SG1: restore the earlier gauge interval, retain the direct outer driver

The N16 accuracy investigation continues after LG1 failed at 2.9715M. The next
controlled test uses the IT1 interpolation interval 0.2--0.56 with the DO1 direct
outer driver and the DG1 restored IT1 constraint damping. Only the gauge weight
interval changes relative to DG1. The physical inner telegrapher is exact for
r<=0.2; the direct outer branch applies for r>=0.56. Its scri limit and original
restoring collar are unchanged. The initial lapse/shift rates match IT1 when
B=D=Lambda=0; B rates and subsequent evolution need not match.

This tests whether retaining the Cauchy inner gauge deep into the initial
hyperboloidal slicing transition generates the harmful pulse. It does not assume
the interval is optimal or exclude a discretization contribution. The shorter
interval is steeper than DO1, but it is the one previously evolved by IT1.

Fresh original N16 wormhole text, dt=0.0005, target coordinate time 10/3=10M.
Direct chi/lapse RK and KO; physical lapse floor 1e-10, positive chi floor 1e-8
inside r<0.03, finite nonpositive chi rescue everywhere. Preserve grid, FD4, KO,
angular epsilon=2, scri penalty scale=1, horizon lmax12/ntheta24/tolerance0.001,
and diagnostic interval0.05. Count pre-floor sign candidates, preserve horizon
quality flags and missing values. N24 is not authorized until reviewed N16
accuracy success; completion alone is insufficient.

Unique checkpoint HYP3D_MPI_SG1, header168, scheme3. No older checkpoint may be
loaded. The N16-only controller/helper reject wrong schemas, failed/unadmitted
states and changed parameters. Healthy walltime continuation requires full
artifact SHA256/length verification, admitted configuration and finite payload,
report/header matching, frozen bindings and forward progress. Preserve horizon
seed sidecars. Never resume a numerical failure or silently change science.

Use one CompactBinaryMerger/debug node, 12 MPI ranks/GPU tiles, worker3200s,
command3100s, exit reserve180s. Check all user debug jobs and saved submissions
before submitting; never duplicate queued/running/ambiguous jobs. Keep home task
footprint below5,000,000,000bytes including output and atomic checkpoint reserves.
Copy/hash-verify evidence locally before remote purge. Release only the matching
terminal root lock. Build SYCL/MPI/PVC with make -j64.

After verified completion compare full/candidate-exterior histories and matched
native endpoint constraints against IT1, DO1 and DG1, using one common cutoff
outside all candidate horizons. Locate radial error bands, assess floor/sign
counts and horizon acceptance. On failure collect and localize it without
restarting. Do not claim convergence, global hyperbolicity or accuracy from
survival. Frozen build and submission records will be appended once available.
