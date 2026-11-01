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

## Local verification and frozen source

Scientific source: `47ae0c35dcef877550ac4371e74d0948842b68b7`.
Nine gauge/device/characteristic checks passed, including the new prescribed
interval and unchanged damping oracle. Fresh two-step evolution to coordinate
0.001 and one-step restart to0.0015 both completed with SG1 schema; maximum
surface/ghost null residual stayed below1.2e-15. These are smoke checks only.
The first controller-suite invocation loaded a version missing SG1 in the
experimental executable allowlist. That was corrected before this source commit;
the focused campaign tests and the full rerun of all127 controller/campaign
tests passed.

`compare.py` requires verified healthy10M completion and matching native SG1
checkpoint diagnostics. It never accepts a failure checkpoint and never grants
N24 automatically. It records one common exterior cutoff, radial error bands,
segment-local sign/floor counts and horizon acceptance gaps. Syntax checking
is not endpoint validation; no SG1 long-time result is available yet.

## Submitted Aurora trial

Job **9018835** is submitted, plan
`campaigns/n16-short-gauge-fresh10m-v1-segment01/plan.json`, SHA256
`0b4eb0b2b886bf86c9cff3ea5f5a4b215f840158a82347d71907c410362d4f5f`.
Follow `short-gauge-10m-state.json` and saved submissions before any action.
Do not submit another copy. N24 remains gated on reviewed N16 accuracy success.

Frozen source manifest `source-short-gauge-v1-manifest.json`, SHA256
`a5fd4a1cf9924927dfa19ddf9d27e91c6d13a91a0d12b8e997e6868853801cff`.
Binary `build-sycl-mpi-short-gauge-v1/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_mpi_short_gauge`,
SHA256 `1b75a50ec11f8b3908de9172bd5baec57eb8bc17d382d1fba25173ab4352996f`.
The frozen launcher is `launcher-short-gauge-v1`; verify all files against its
`provenance.json`, including `short-gauge-science.json`, before using
`control.py` or `short_gauge_campaign.py`.

All546 archived source files and1486 Kokkos files were hash-verified. SYCL/PVC+MPI
build completed with oneapi/release/2025.3.1 and make -j64. Binding measured
3,684,233,216bytes and1,088,481,201bytes remaining after planned/atomic reserves.
There were no user debug jobs before submission; unrelated capacity jobs remain
held. The new20-minute monitor is `aurora-n16-original-interval-gauge-to-10m`.
It may continue healthy walltime checkpoints only; delete it after verified
trial disposition. The persistent investigation remains active.

## Frozen-state characteristic audit while SG1 evolves

`audit_characteristics.py` applies the SG1 gauge coefficients to the original
wormhole data and preserved IT1/DO1/DG1 10M states. It uses the audited native
geometric principal block, independently replaces the gauge block, and checks
the unchanged0.2--0.9 control against the native direct-outer matrix. Source-data,
coordinate and library hashes are recorded in `characteristic-audit.json`.

All active cells in0.1<r<0.95 retain positive normalized squared-speed gaps
(transverse-shift minus lapse, longitudinal-shift minus lapse, and lapse minus
physical) on these four sampled states. The smallest transverse/longitudinal
gaps across them are0.0487721/0.0967603, both on DG1. For each state,200 directional
26-variable symbols have the expected real spectrum and complete eigenspaces;
maximum spectral discrepancy is3.8e-15 and maximum balanced basis condition7.59.

These are SG1 coefficients on initial/older snapshots, not a trajectory of SG1.
They do not exclude future collisions, establish global strong hyperbolicity,
or prove stability or accuracy. The earlier LG1 audit was also benign on older
states before its actual trajectory developed reversed speed ordering and failed.
The running SG1 evolution remains necessary evidence.

## Provisional first evolution comparison

Scheduler job9018835 and its worker are running. The captured history reaches
0.30M; it is a live snapshot, not terminal evidence. At matched0.30M the exterior
Hamiltonian/momentum RMS values are SG1 0.094445/0.160757, DG1 0.097005/0.172294,
DO1 0.106743/0.188707 and IT1 0.059336/0.108752. All four accept a candidate horizon
at radius0.0647689519 (agreement to about1e-11). Thus the difference at this early
sample is not plausibly explained by materially different radial cutoffs.
The large full-domain Hamiltonian transient is shared: all four are approximately
17.9364. No new instability is inferred from that transient alone.

The shorter interval only modestly reduces the early exterior error relative to
DG1 and remains above IT1. That suggests transition extent alone does not remove
the initial discrepancy; it does not yet identify which remaining outer-driver
term matters or predict the later result. `progress/comparison-0p30M.json` records
exact values and source hashes. The fixed10M experiment continues unchanged;
N24 remains gated on terminal accuracy review.

The final comparison also records the accepted irreducible mass relative to the
known Schwarzschild mass M=1/3, including the final signed fractional error and
maximum absolute fractional error after4M. Failed horizon solves remain missing;
using their candidate surface as an exterior cutoff does not create a reliable
mass measurement. This supplements coordinate-volume constraint norms and avoids
using coordinate horizon radius alone to judge differing gauges. The native IT1
reference final fractional mass error is5.28210e-4; its DO1 counterpart is missing
because final horizon acceptance failed. The SG1 endpoint remains pending.

## Live measured comparison through3M

The SG1 history has passed3M, beyond the LG1 accepted failure time2.9715M.
This is not terminal success or N24 admission. `plot_progress.py` preserves the
captured history, hashes and measured comparison in `progress-to3m/`, without
interpolating missing mass data or extrapolating LG1 beyond2.85M.

At3M, SG1 candidate-exterior H/M/Theta RMS are0.018706/0.029535/0.00067265;
DG1 gives0.018066/0.054447/0.0023416 and IT1 gives0.012461/0.017604/0.00045626.
SG1's final candidate radius0.07948 is close to IT1's0.07982, but DG1's is0.05501.
These different cutoffs prevent treating the raw ratios as a same-domain
accuracy comparison. SG1's3M finder is unaccepted, so no mass value is supplied.
The final10M common-cutoff reconstruction and accepted mass review remain required.
