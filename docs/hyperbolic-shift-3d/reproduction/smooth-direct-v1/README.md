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
