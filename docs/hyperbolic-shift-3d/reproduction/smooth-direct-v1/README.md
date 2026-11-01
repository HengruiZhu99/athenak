# Fresh smooth-gauge direct-variable campaign

The user requested direct signed chi and alpha evolution with floors, from fresh
wormhole data at t=0: N16 toward20M first, then N24 toward20M. This supersedes the
GW1 log-variable continuation. N16 GW1 failed at9.1575M; N24 job9013517 was
cancelled by request, not classified as a numerical failure. Its terminal PBS
records and partial diagnostics are preserved. Neither3M nor failed checkpoints
are inputs to the first segments of this new campaign.

Target hyperbolic_shift_3d_mpi_smooth_direct combines the existing AC1 direct RK
and physical-variable KO treatment with smoothstep9 gauge blending on0.56–0.8
and complete-candidate exterior diagnostics. GD1 is a distinct168-byte checkpoint
schema with PositiveConfig.scheme3 and chi floor/radius1e-8/.03. No old target
changes its equations or checkpoint identity. Finite signed candidates are
counted before clipping; they do not terminate evolution. Physical lapse floor
is1e-10, positive chi floor1e-8 applies inside r<.03, and finite nonpositive chi
is rescued everywhere including boundary interpolants. Nonfinite arithmetic and
invalid metric geometry remain failures. Surface lapse remains fixed by scri
closure. No chi or lapse logarithmic RK/KO is used in GD1.

Both grids start from the original wormhole generator text, with SHA256:
- N16:35c1b65aeaaefac34552f9a63b279cabe839b09326adc9af1d2c767ecc29c8ca
- N24:6c265e8dc309c70a2c7017a80eddd2d76f3a185b4f1d6108cc65b8bfc85be914

Keep N16dt=.0005,N24dt=1/3000,dt/dx=.0444444,target coordinate20/3=20M,
angular epsilon2,scri scale1,original harmonic collar, horizon lmax12/ntheta24/
tolerance.001 and diagnostic interval.05. Preserve candidate-horizon quality
flags; reaching20M alone does not establish accuracy or convergence.

Use smooth_direct_campaign.py to create fresh segment01 from --initial, or
later sequential segments from --restart and independently verified --sha256.
It rejects GW1/AC1, unadmitted/configuration-mismatched/nonfinite checkpoints,
changed initial data and3M starts. Freeze smooth-direct-science.json beside the
launcher after the SYCL/MPI build and include it in launcher provenance.
For each terminal healthy walltime stop: collect the full campaign, copy and
hash/length verify all artifacts locally, run helper verify, release only its
matching terminal lock, then create/bind/submit the next segment once. Never
resume numerical failures; advance from N16 failure/completion to fresh N24.
On storage/integrity/scheduler failure, diagnose before retrying. Do not change
science, timestep or floors between segments. Sum segment-local candidate/floor
counters, which include rejected attempts; keep boundary interpolation counters
separate. After both terminate, compare histories, preserve gaps, commit/push
results and delete the hourly monitor.

Use CompactBinaryMerger/debug, one node/job at a time,12 MPI ranks/GPU tiles,
worker3200s,command3100s,native exit reserve180s. Check all user debug jobs before
submission. Build SYCL/MPI with make -j64 using the recent oneAPI configuration.
Keep task footprint below5,000,000,000 bytes including output/atomic reservations;
copy and hash-verify large evidence before purge. Stay quiet on unchanged monitor
states; notify progress, continuation, completion, failure or required action.

## Preparation checks

Scientific source commit cbad68bbda93ff08949945630425217a1e62ebf1. The controller
and campaign unit tests passed62 tests; native floor/sign accounting passed.
A fresh N16 two-step CPU run completed at coordinate.001, using GD1 and direct
chi/alpha RK/KO, with maximum null residual1.12e-15. Its GD1 restart completed
one further step at.0015, and the native executable rejected a GW1 checkpoint.
These are setup/restart checks, not long-time stability evidence.

The cancelled log N24 job9013517 last recorded coordinate1.45=4.35M; no admitted
terminal checkpoint was produced. PBS confirms cancellation(exit143), whereas
the interrupted worker file still says running. The latter is stale. Its last
partial diagnostics and terminal records are preserved in superseded-log-n24/.

Aurora configure initially failed because the partial source archive lacked
tst/elliptic_shift_1d/spherical_api.cpp. The missing tracked donor directory from
the same source commit was added before a successful configure; the failed log
is retained. No evolution used the incomplete source. The final source manifest
binds all copied source, including the unchanged Kokkos files.

## Frozen Aurora build and monitor protocol

Scientific source:cbad68bbda93ff08949945630425217a1e62ebf1. Manifest `source-smooth-direct-v1-manifest.json`
SHA256 `a7c265d827621328406992e1fb8f175d5a629eba6ddd618066e482c824d27b2c`.
Binary `build-sycl-mpi-smooth-direct-v1/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_mpi_smooth_direct`
SHA256 `1cb01219a803623d877ef9bac8a8fb3cc3619866da2ba776239ed825f7b9d942`.

Use `launcher-smooth-direct-v1/control.py` and `smooth_direct_campaign.py` after
checking every file against `provenance.json`, including the science identity
file. Follow `smooth-direct-fresh20m-state.json` and the current plan's saved
submission before any action. Never duplicate queued/running/ambiguous jobs.

Prepared first plans:
- `campaigns/n16-smooth-direct-fresh20m-v1-segment01/plan.json`
- `campaigns/n24-smooth-direct-fresh20m-v1-segment01/plan.json`

Both use `inputs/puncture-continuous-nN-smooth-v1.txt` at t=0. Later segments
use ONLY admitted GD1 checkpoints from the current resolution. There is no
GW1/AC1 conversion and no3M variable switch. Example first plan (already
prepared; do not duplicate):

    python3.11 launcher-smooth-direct-v1/smooth_direct_campaign.py plan --root "$PWD" --n 16 --segment 1 --initial inputs/puncture-continuous-n16-smooth-v1.txt

For healthy walltime results: `control.py collect`, copy full campaign to local
output retaining `campaigns/...` layout, verify artifact hashes/lengths and run
`smooth_direct_campaign.py verify --root ROOT --plan PLAN`. After verified
admission/configuration/forward progress and terminal scheduler, release matching
lock once using `control.py collect --release-terminal-lock`, then helper `plan
--n N --segment NEXT --restart VERIFIED_CHECKPOINT --sha256 VERIFIED_HASH`. Bind
with the frozen manifest, check ALL user debug jobs, submit once, update state.
The helper's plan generation itself does not authorize an unverified checkpoint.
Preserve horizon-seed sidecars and bound restart inputs until a successor has
verified output. N16 and N24 checkpoint lengths are79725224 and234413736 bytes.

For a numerical failure, collect/verify failure artifacts without treating the
failed checkpoint as admitted, mark that resolution failed and release only its
terminal lock. Start N24 from its own fresh generator data after N16 termination.
After both terminate, report and compare available full/candidate-exterior
histories with prior original/smooth results, distinguishing this direct-from-t0
protocol from earlier3M switches; commit/push small evidence and delete monitor.

## Initial submission

N16 job9013744 is submitted from fresh t=0 data toward20M. Plan SHA256
`b7cfd56141302eba8ac81748e215498ac0fa0ddd6d88e585d91bbec913945966`. N24's fresh segment01 is prepared and unsubmitted.
The all-user debug query was empty; binding measured3.005GB
and1.768GB headroom after N16 output/atomic
reservations. Hourly monitor `aurora-smooth-gauge-n16-then-n24-to-20m` is ACTIVE,
renamed Aurora fresh direct N16 then N24 to20M and updated to this new protocol.
It must not continue the superseded log-variable jobs.

## N16 verified completion

Job9013744 completed20M from fresh wormhole data in13334 RK steps. Native runtime
was2438.93s (40.65 minutes), with SYCL and12 MPI ranks. All configured worker
checks passed, and the admitted GD1 checkpoint was independently verified both
remotely and in the local copied campaign, including every recorded artifact's
SHA256/length, checkpoint configuration, finite payload and exact terminal time.
Checkpoint SHA256:6e36cb7b160be61223c7e2e5e1b6009c2bf41abefde27e6e30ea900da40d19f9.

There were48 negative chi stage candidates and16 negative final candidates, all
in the eight nearest puncture cells at r=.0097427858; all64 were floored. First
negative candidates occurred at0.3M and the minimum was-1.57669115e-6. There were
no zero candidates, lapse-floor interventions, surface-chi floor interventions
or negative scri-interpolated chi events. Counters count cell updates, not unique
timesteps. The configured floor remained active throughout the run.

Maximum null residuals:surface1.2213e-15,ghost1.5543e-15. At20M the finder accepted
the horizon(expansion RMS4.9086e-4); coordinate radius.1157804,areal radius.6668811,
irreducible mass.33344054. Final candidate-exterior H_L2=.02803899 and radial
Mr_L2=.006704283; full momentum-vector exterior M_L2=.01404840 is a different
diagnostic. Preserve the full history and finder flags in n16-completed/.
Completion does not demonstrate convergence; the matched N24 run is pending.
No large remote outputs were purged at this transition because sufficient
storage headroom remained. Full local campaign is preserved under
output/hyperbolic-core-shell/smooth-direct-v1/aurora/campaigns/.

## N24 submitted after N16 completion

N24 job9014872 is submitted from its own fresh t=0 initial data toward20M.
Plan SHA256:a7a19fb1ee54c3ffab5e24f5de207b48b4518d3079e35e3285df93f272860922.
The all-user debug query was empty. Binding measured3.144GB with1.227GB headroom
following N24 output/atomic reservations. The hourly monitor now follows N24;
N16 is terminal and must not be rerun. Frozen source/binary, floors and all
scientific settings remain unchanged. Small submission records are in
submission/n24/ and the current campaign state is preserved in submission/.

## Terminal N24 failure and comparison

N24 job9014872 failed numerically after the last accepted3.726M(step3726),
with `invalid accepted volume algebraic projection`. Its rejected final-step
candidate corresponds to3.727M. Last saved diagnostic time is3.6M; report norm
and horizon fields refer to that older diagnostic, not the failing candidate.
This was neither a walltime stop nor a storage failure. No failed restart is
admitted. All worker artifact hashes and lengths were independently verified
in the copied local campaign; see n24-failure/verification-and-location.json.

The saved rejected volume state is finite but has two indefinite conformal
metrics: cells69648 and70248, at(.00375,-.01125,.00375) and
(.00375,-.00375,.01125), both r=.01243734. Their minimum metric eigenvalues are
-.0124980 and-.0512716; determinants-.0651036 and-.2851343. Both chi and conformal
lapse are positive there. These are inner Cartesian points, well inside the
last candidate horizon radius.0825255. The determinant/trace algebraic repair
cannot admit these tensors. This identifies the immediate failure, not the
cause of their preceding growth. It is not a negative-chi/lapse stopping test
or a failure of null closure at scri(maximum residual1.55e-15).

N24 recorded1230 negative chi candidate updates(914 stage,316 final) in8 points,
with first occurrence.1545M and minimum-3.48358e-6. Its chi floor fired1261 times
(941 stage,320 final), also including positive below-floor values. It recorded
109396 negative lapse candidates(82040 stage,27356 final) across48 points, first
at1.744M; all were floored. No zero candidates or nonpositive scri-interpolated
chi were logged. Counters include the rejected attempt. These differ sharply
from N16's64 chi interventions and zero lapse interventions.

`constraints-final.png/pdf` compares every retained new N16/N24 exterior norm
with the preserved much finer N513 baseline; `protocol-comparison.png/pdf`
compares earlier original and smooth-log histories. Candidate exterior cutoffs
and original finder acceptance flags are retained. There is no interpolation
across missing values or extrapolation past the final diagnostics. Scripts
and source hashes accompany the figures. Native coordinate export uses the
same ContinuousCoreMesh and active-cell ordering as the evolution.

N24 exterior norms are smaller than N16 over much of their shared history,
but N24 loses metric positivity inside the horizon. N16's late Hamiltonian
exterior norm rises again toward20M. The combination does not establish a
stable convergent3D scheme. Direct-variable evolution with floors removes the
previous log-chi underflow failure path but is not sufficient for finer-grid
strong-field stability. The campaign terminates with N16 completed and N24
failed; no numerical retry or parameter change is made automatically.

Terminal cleanup: the failed N24 checkpoint and rejected-state capture were
copied locally and hash/length verified before remote removal; purge receipt
and matching terminal-lock release are retained in n24-failure/. Post-cleanup
remote task footprint was3.224GB. Both requested resolutions are terminal and
the hourly automation was deleted. No failed state was restarted.

To reproduce the coordinate export used by analyze_terminal.py:

    build-hyperbolic-mpi/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_continuous_positive_geometry 24 3 output/hyperbolic-core-shell/smooth-direct-v1/coordinates-n24.csv

Then run analyze_terminal.py and compare_terminal.py using the locally retained
raw campaign. The scripts produce evidence/plots only, never evolve or submit.
