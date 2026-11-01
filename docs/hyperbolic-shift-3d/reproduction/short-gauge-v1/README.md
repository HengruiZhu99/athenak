# SG1: restore the earlier gauge interval, retain the direct outer driver

**Current:** Both fresh SG1 runs completed10M: N16 job9018835 and N24
job9019406. Both checkpoints and all worker artifacts passed remote/local
verification. See the final resolution comparison below. Earlier pending
sections are preparation chronology, not the current campaign state.

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

## Prepared N24 gate (not admission)

The local launcher code now supports N24 only with an explicit N16 accuracy review
supplied via `short_gauge_campaign.py --review PATH`. This does not modify the
running frozen N16 launcher or scientific source/binary. A new frozen launcher
would be created only after terminal N16 review; no N24 plan or admission exists
at preparation time.

The review must record verified N16 completion at10M, exact checkpoint/comparison
hashes, matching scientific source-manifest/binary hashes, a written rationale,
and explicit review of geometry, exterior constraints, accepted horizon mass and
floor/sign counts. It must set `n24_admitted:true` after actual evidence inspection.
The record is included among hash-bound N24 inputs. Structural checks alone are
not scientific proof and cannot create the review. The helper preserves fresh
original N24 data, dt=1/3000, SG1 schema and all other settings; healthy continuation
requires the same review. Missing/failed reviews, changed science, wrong timestep,
wrong schema and unadmitted checkpoints are rejected by focused tests.

Read-only preparation verified the original N24 text SHA256
`6c265e8dc309c70a2c7017a80eddd2d76f3a185b4f1d6108cc65b8bfc85be914`
(84,249,758bytes). A current footprint scan was3,766,644,736bytes; N24 planned
output plus atomic reserves are629,475,714bytes, leaving603,879,550bytes at that
instant. Capacity must be rechecked before actual binding/submission.

All129 controller/campaign tests passed after adding the N24 review gate. This
validates launcher behavior only; no N24 scientific admission has been issued.
The frozen running N16 source/binary/launcher remain unchanged.

## Verified N16 result and decision to run N24

Job9018835 completed10M in6667steps. Scheduler is terminal, worker passed, and
remote plus independent local artifact/header/configuration/finite-payload
verification passed. Admitted SG1 checkpoint SHA256
`d581c2860a1049c5a79568c90658b56463ebe8487229e06f7df3d34b27f80749`
has79,725,224bytes. All stored volume metrics are positive definite (minimum
metric eigenvalue0.8730); chi and lapse are positive. Small terminal records and
checkpoint geometry audit are preserved in `n16-completed/`. Large original
checkpoint remains local under output/hyperbolic-core-shell/short-gauge-v1/aurora.

`comparison/` contains verified native reconstruction on common cutoffr>0.1099,
outside every final candidate horizon. Final H/M-magnitude RMS:

| Gauge | Exterior H RMS | Exterior M RMS |
|---|---:|---:|
| Earlier IT1 |0.00808023|0.01240101|
| Wide DO1 |0.23278860|0.32155526|
| Wide DG1, restored damping |0.22462459|0.30648240|
| Original interval SG1 |0.01541497|0.02900764|

Relative to DG1, SG1 reduces H/M by14.57/10.57 times with only the gauge interval
changed. The former.16--.30 region contributes2.83% of full H-squared instead of
66.50%; this is a final-state localization, not proof of where errors originated.
The initial-source audit and the failure of moving the interval outward support
a strong gauge transient as the trigger. Narrowing the interval improves the
result despite a steeper weight, so insufficient resolution of the weight alone
is not an adequate explanation. The precise discretization/interface mechanism
has not been isolated.

SG1 accepts45/68horizon rows, including the final one; unaccepted rows remain
missing. Final fractional irreducible-mass error is+0.00341326 (0.3413%), worse
than IT1's0.0528%. Late exterior constraints are bounded but remain1.91/2.34times
IT1. Thus this is not a convergence or final-formulation accuracy claim.

There were56negative chi stage candidates and8negative final candidates, all at
the nearest8puncture points, r=0.00974279, beginning at coordinate0.10075. These
counts include rejected RK candidates. There were no lapse floor hits, surface
chi floor hits, or boundary-interpolation sign events. Scri residual maxima are
1.22e-15(surface) and1.55e-15(ghost enforcement). Extra inner chi interventions
relative to IT1/DG1 are recorded, not concealed.

After reviewing these results and the rendered figures, `n16-accuracy-review.json`
admits the user-requested fresh N24 test using identical science and dt/dx. The
reason is stable finite10M evolution, removal of the large overlap-region error
and bounded coarse-grid exterior errors with an accepted final horizon. N24 is
needed to determine whether the remaining error and mass drift decrease. The
review does not authorize changing equations, timestep ratio or floors, nor
restarting numerical failures. It does not claim global strong hyperbolicity or
convergence. The persistent goal remains active until the N24 investigation is
finished and assessed.

At the10M SG1 endpoint, r=0.775--0.9 contributes62.20% of the exterior H-squared
integral and75.02% of the exterior M-squared integral (common cutoff0.1099).
The.16--.30 region contributes23.44%/8.98% respectively. These are residual error
locations to compare at N24; they do not by themselves identify an outer-collar
or interface defect. The generated comparison keeps `n24_admitted:false` because
it cannot grant admission; the later explicit accuracy review is authoritative.


## N24 submitted after N16 review

Job **9019406** is already submitted. Current state is
`short-gauge-n24-10m-state.json`; plan
`campaigns/n24-short-gauge-fresh10m-v1-segment01/plan.json`, SHA256
`8290c012f0c3a5a982f5eb53a29bf6d85b6a17dd467d5a5be60ae63efaceccf6`.
Do not duplicate the saved submission. Fresh original N24 text starts at t=0,
dt=1/3000, target coordinate10/3. Scientific source, binary, gauge, floors and
diagnostics are identical to N16. There is no N24 result yet.

Use the new frozen `launcher-short-gauge-n24-v1`, from launcher commit
`ef70f5dc9809062ebb388f253beb58e8488ef804`; verify its provenance before use.
The older launcher is N16-only. Every N24 segment requires
`--review admissions/n16-short-gauge-10m-review-v1.json`, SHA256
`723e5cda25c9496b12b487bbe7f37c6578f95a2743feecb682c1fbb673ccc224`.
This review is bound as an input and must match the frozen scientific hashes.
Submission/binding/storage records and launcher provenance are in
`n24-submitted/`. Binding measured3,682,934,615bytes, with687,589,671bytes
headroom after planned output and atomic reserves. No user debug jobs were
present before submission. N16's terminal lock was released and its remote
checkpoint retired only after independent local SHA256/length verification.

The existing20-minute monitor now follows N24 and may continue healthy walltime
checkpoints only. Collect and independently verify artifacts and admitted SG1
configuration before continuation; N24 checkpoint length is234,413,736bytes.
Keep campaign-relative paths locally, preserve horizon-seed sidecars and the
bound review, check all user debug jobs, and release only the matching terminal
lock. Never resume numerical failures or change science automatically.

After N24 terminates, compare actual N16/N24 histories and any admitted endpoints
with matching SG1 diagnostics and one common exterior cutoff. Existing
`compare.py` compares N16 gauge variants; it is not a resolution-comparison
script. Retain horizon flags, missing data and separate floor/sign counters.
Assess the residual outer-shell and overlap-region errors and accepted horizon
mass errors without inferring an asymptotic order from two resolutions alone.


## Prepared N16/N24 comparison

`compare_resolutions.py` is the separate resolution comparison. It takes
`--root` pointing to the collected Aurora mirror and ordered `--n16-plans` and
`--n24-plans` lists. It requires terminal scheduler records and hash-verifies all
worker artifacts, frozen science, the N24 admission review and restart sequence.
It rejects resolution mixing, duplicate segments and live partial output.

Histories preserve each run's candidate cutoff and accepted horizon flags;
missing values are not interpolated. Segment sign/floor counts are summed, with
boundary interpolation events separate. Only two healthy, verified10M endpoints
trigger the native N-dependent SG1 reconstruction and common-cutoff comparison.
The endpoint diagnostic is checked against native full-domain norms. Failed
states are not reconstructed as accepted endpoints. Radial plots and fractions
for0.16--0.30 and0.775--0.9 locate the remaining error. Ratios are reported without
claiming an asymptotic convergence order from only two resolutions.

Preparation checks loaded the actual verified68-row N16 history and reproduced
its reviewed sign/floor aggregates. Wrong-resolution, duplicated-segment and
live-scheduler inputs were rejected. See
`checks/resolution-comparison-preparation.log`. Full N24 reconstruction and
rendered-figure review await collected N24 results; this preparation is not a
new scientific result.


Restart-history handling follows `coupled_mpi.cpp`'s `record()` semantics:
segment-local null/CFL maxima reset, and a newly initialized horizon finder may
report a different iteration count or candidate cutoff at the same physical
state. The merger requires repeated full-domain physical diagnostics to match,
retains the preceding segment's final observation, and records all differing
segment/finder columns explicitly. It does not reject a legitimate restart just
because a cumulative maximum reset, nor hide a changed full-domain constraint.
A synthetic merger check exercised both cases; synthetic rows were not saved as
scientific results.


## Characteristic samples on the completed SG1 state

The read-only `audit_characteristics.py --include-sg1-endpoint` adds the
hash-verified N16 SG110M checkpoint to the preserved initial/control audit.
Results are in `characteristic-audit-sg1-endpoint.json`; the earlier audit is
unchanged. All200 directional symbols sampled on SG1 had the expected real
spectrum and complete eigenspaces. Maximum spectrum discrepancy was3.78e-15;
maximum balanced eigenbasis condition was7.91. Over all active stored cells in
0.1<r<0.95, minimum transverse-shift-minus-lapse squared-speed coefficient gap
was0.11843, longitudinal-shift-minus-lapse0.23136, and lapse-minus-physical0.01367;
none was negative or zero. These are coefficient gaps before directional metric
factors, not coordinate speed differences.

This gives no evidence of a new sampled exterior characteristic collision in
the successful SG1 endpoint. It does not cover every direction, the time
trajectory, the puncture region r<=0.1 or scri, and is not a global hyperbolicity
proof. The matrix reconstruction remains checked against the native DO1 control
block before replacing the gauge coefficients. No evolution source changed.

## Provisional N24 progress through 3M

`progress-to3m-n24/` contains a hash-verified live history snapshot, measured
plots and provenance. N24 was running at capture and continues toward10M.
At3M its candidate-exterior H, momentum-magnitude and radial-momentum RMS are
0.6869,0.9636 and0.7589times N16 respectively. Each uses its own candidate
cutoff. Full-domain H is instead2.583times N16 at that instant and its history
remains irregular. Thus exterior errors improve to different degrees, but no
clean full-domain convergence or asymptotic order is established. Preserve the
unaccepted N16 horizon-mass gap at3M. Final common-cutoff and radial comparison
still requires collected, verified matched10M checkpoints.


## Final matched N16/N24 result at 10M

N24 job9019406 completed in one allocation (PBS exit0,worker passed), with10001
steps, dt=1/3000 and target coordinate10/3. Native execution took2178.74seconds
(36.31minutes); scheduler walltime was39m20s. It used12MPI ranks bound to12GPU
tiles. Source/binary/launcher/review provenance and all artifact hashes were
verified. The admitted SG1 checkpoint has234,413,736bytes, SHA256
`7082e7706c18c8e3e84f24c04f6c56492b45c372a8c1e3012e35a578aaf6a5e2`.
Independent local frozen-helper verification passed; every stored volume metric
is positive definite (minimum eigenvalue0.873066). Small records are in
`n24-completed/`; the full checkpoint is retained in the local output mirror.

`resolution-comparison/` contains measured histories and matching native
checkpoint reconstructions. Both native full-domain reconstructions agree with
the original reports. Use the common coordinate cutoff r>0.1099, outside both
final candidate horizons. These are normalized compactified-coordinate-volume
RMS norms, not proper-volume norms.

| Diagnostic at10M | N16 | N24 | N16/N24 |
|---|---:|---:|---:|
| Exterior Hamiltonian RMS |0.01541497|0.00820183|1.879|
| Exterior momentum magnitude RMS |0.02900764|0.02550556|1.137|
| Exterior radial momentum RMS |0.03081597|0.02610776|1.180|
| Exterior Theta RMS |0.00081293|0.00040453|2.010|
| Accepted fractional horizon mass error |+0.00341326|-0.00028682|—|

Both final horizons are accepted. N24 accepts58/68history rows versus45/68atN16;
all missing accepted mass values remain gaps. The N24 final mass error is
-0.02868%, versus+0.34133%atN16. This reduction at one time does not establish
an asymptotic convergence order; histories have gauge transients and mass
oscillations. Full-domain Hamiltonian transients also remain irregular, even
though final full-domain H/M improve. The final shorter-interval runs survive
10M at both resolutions with bounded late exterior errors.

### Remaining error location and floors

The0.16--0.30 band now contributes10.33% of N24 exterior H-squared and1.58% of
M-squared (N16:23.44%/8.98%). The0.775--0.9 outer-shell band contributes81.65%
and93.72% (N16:62.20%/75.02%). The momentum-magnitude peak in that radial band changes
little with refinement, limiting the overall momentum improvement to12%. This
localizes the remaining weak resolution trend; it does not yet isolate whether
its source is the gauge collar, shell discretization, or another truncation term.

N24 records968negative chi stage candidates and344negative final candidates.
Total chi-floor counts are1064stage and368final, including96/24positive
below-floor candidates respectively. Every intervention is at the nearest8
puncture cells, r=0.00649519; first negative candidate is at0.1545M. N16 had
56/8negative candidates. These segment-local counts include attempted stages
and are not directly an accuracy metric across different grids/timestep counts.
There were no lapse floor hits, surface chi floor hits, or boundary-interpolation
negative/zero events in either trial. The increase in puncture interventions and
irregular interior transient remain limitations; exterior improvement is not
explained by clipping exterior cells. `floor-localization.json` preserves the
counts/radii. Maximum N24 surface/ghost null residuals are1.55432e-15.

### What the controlled experiments establish

Moving the inner gauge farther outward was not the successful modification:
LG1's0.4--0.9 placement failed at2.9715M with an indefinite metric near r=.4228.
Restoring damping alone (DG1) had only a small effect. Restoring the earlier
0.2--0.56 gauge interval while retaining the direct outer driver and fixed
damping (SG1) reduced the N16 exterior H/M errors by14.57/10.57times and enabled
these matched fresh N16/N24 runs to10M. N24 was submitted only after explicit
N16 accuracy review; no failed state was resumed.

The leading explanation is the strong gauge transient generated when the
Cauchy inner gauge is extended into the initial hyperboloidal transition. The
exact initial forcing audit and controlled interval changes support it. The
shorter, steeper interval performs better, so an underresolved weight alone is
not an adequate explanation. Scri closure stayed at roundoff, and the sampled
SG1 exterior characteristic audit found no defect; neither observation excludes
other numerical mechanisms. The precise continuous-versus-discrete origin is
not proven, and the remaining outer-shell momentum error still needs separate
work. This completes the requested bounded gauge-placement investigation and
N16-then-N24 test, not a claim of global hyperbolicity, established high-order
convergence, or20M/100M stability.

### Collection and disposition

Both terminal allocation locks were released. The N24 remote checkpoint was
removed only after independent local SHA256/length/configuration/finite-payload
verification; the local original remains. The measured remote task footprint
after retirement was3,821,523,789bytes, below the5GB cap. Receipts are preserved
in `n24-completed/`. No continuation or replacement job is needed for this10M
trial. The monitor is retired after final comparison and evidence publication.

The native diagnostic deliberately refuses an existing output path. The first
verified rendering was preserved locally when regenerating the shorter figure
labels/consistent colors; the final rendering uses a fresh output directory.
No checkpoint or scientific executable was changed during figure preparation.
