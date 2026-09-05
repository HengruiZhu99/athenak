# Direct lapse-gradient correction: frozen qualification plan

This plan is recorded before compilation tests or corrected evolution. The user
requested this gated correction only. The accepted criteria are recovered from
`docs/pc_gh_hybrid_projection.md` (Qualification sequence),
`qualification-runs-20260905/r16-smr128/REPORT.md`, and
`docs/pc_gh_qualification_log.md` (long-domain head-on comparison).

## Scope and provenance

Parent 80602ffc, branch codex/pc-gh-gamma2-20260904. Production src at HEAD is
identical to baseline source commit 1e6b0612. Preexisting dirty analysis/docs are
outside this patch. Change only the finite relaxation/projection L target to
2 D(rho*w), via one existing-FD accessor, and append diagnostics. Retain Ralpha,
p/Q/L/B curls, every other equation, rate, mask, projection switch and timing.
The continuum Ralpha projection scaling in the older plan is no longer an exact
discrete identity: Ralpha_new=(1-P) Ralpha_old + P delta, where
 delta=2 D(rho*w)-2(w D rho+rho D w). Direct RL scales by (1-P).
This patch makes no discrete AMR curl-commutation claim.

Remote work begins on della-vis1.princeton.edu. Established cuda_driver.py permits
that host and Slurm GPU allocations. The head A100 was idle at inspection.
Verified Slurm partition is `gputest` (not gpu-test), QoS `gpu-test` has MaxJobsPU=3;
partition MaxTime=15 days, QoS MaxWall unset. Use one job if needed, conservatively
2 hours with 15-minute clean checkpoint segments, explicit same-binary resumes,
and never resume a strict failure. Do not change unrelated jobs.

## Gate 1: discrete correctness

Test the actual shared C++ helper with Kokkos on 2D and 3D uniform periodic grids,
consistent scalar and target ghosts, FD2/FD4/FD6 (template radii 2/3/4), two grid
sizes and unequal directional spacings. Compare target to independently summed
coefficients. For pair ij, scale curl by
2 epsilon max|rho*w| ||D_i||_1 ||D_j||_1. Require max scaled curl <=64 and
scaled target discrepancy <=64. The factorized target is a negative control;
require at least one resolved nonzero curl per stencil order. Compile the full
CUDA production executable (which instantiates all three supported orders).
A test/build failure blocks stress until diagnosed and corrected within scope.

## Gate 2: matched fine-core stress

Use the exact saved `core-hybrids/R16/core256-R16/used_input.athinput` from remote
root `/scratch/gpfs/FPRETORI/hz0693/pcgh-hybrid-20260905-1317`.
Baseline executable SHA256 96bc34157896c164904a6ff426e84f4c97d95a7b8c8c1a43971940fc744e60b1;
input SHA256 ff69d9d88a75e4156735454b19b7cd91522d133ce4ff0f8fd9ced1e2d7105f80.
R16 lambda=1+15P, radii (.125,.5), FD6/RK3, CFL=.2, KO=.3,
dt ceiling .0125, both reduction and GH projection off, original eight-level
[-8,8]^3 hierarchy, finest M/256, target 6M, original output cadences.
Baseline strict metric failure is at 5.187818M.

Pass requires clean numerical completion at 6M, all strict checks and sampled
metric eigenvalues positive, finite full-domain and chi>=.0625 constraints,
and no unresolved runaway in any separate curl/reduction/constraint or localized
core/taper/interface history. Evaluate completed-step trends separately from RK
brackets with stale ghosts. Compare all common saved times and endpoint; preserve
locations, extrema and coordinate-volume norms. A strict failure is FAIL even
if delayed; survival with unexplained growing constraints/curls is INCONCLUSIVE.
For a reproducible runaway flag use consecutive half-M envelope doublings in
three bins, or a >10x late increase over the 2--3M envelope with a positive
log-linear trend. These are investigation triggers, not thresholds that grant
pass: unresolved growth below them also withholds promotion. Diagnosis must
report full-domain and excised behavior and minimum eigenvalues, not lifetime alone.

## Gate 3: original large-domain single-puncture ladder (conditional)

Only after Gate 2 passes, repeat preserved r16-smr128 inputs at M/8,M/10,M/12,
through 20M, boundaries +/-128M, original seven physical SMR regions and 400
blocks. Use native 3D coordinate-volume full/excised norms, separate curls,
and the existing common [-8,8]^2, 128x128 field sampler. Report unequal-spacing
orders and successive difference alignment, not only decreasing magnitudes.
Require clean endpoints/strict checks, positive pair orders in all five primary
constraint families at every half-M sample in 10--20M, declining field differences
in every resolved sector, and a common leading error profile. Operationally,
require cosine alignment >=.9 for each resolved field group and aggregate at
exact 20M and late samples whose interpolation sensitivity is <10% of each
difference. These conservative numerical definitions operationalize the original
'unresolved exterior convergence is not a pass' rule; they are not inherited
published thresholds. Unresolved interpolation sensitivity, nonaligned differences,
or unexplained refinement-growing core curls is INCONCLUSIVE and blocks binary.

## Gate 4: documented head-on (conditional)

Recovered geometry/input: `qualification-runs-20260904/regular-extension/inputs-v2/binary/headon-t100.athinput`,
generated from `inputs/z4c/twopuncture/bbh_headon_pcgh_cuda_r128_t100.athinput`
by make_inputs.py. Equal .5/.5 masses at x=+/-2.5, zero momenta/spins,
[-128,128]^3, adaptive chi refinement plus documented fixed regions,
finest M/16, FD6/RK3, CFL=.2, KO=.3, eta=2, kappa=1, regular advective system,
outflow/extrapolation=2, extraction 8/12/24/32/48/56M. Retain the qualified R16
candidate settings; no substitute geometry. Exact input must be frozen and
compared to the recovered source before any launch. The old projected method
failed at 73.79991M after sustained curl-Q growth (roughly 5.82M e-folding).

Acceptance requires evolution past that failure and clean to 100M, no unresolved
exponential constraint/curl/interface growth or metric deterioration, serialized
tracker continuity, symmetry checks and merger-window waveform comparison to
saved Z4c with explicit resolution uncertainty. Use u=t-r, junk u<20, merger
20<=u<=35; imaginary 22 is symmetry leakage. The older plans supply no numerical
waveform tolerance; do not invent one or call a single-resolution difference a
precision pass. If needed after earlier gates pass, recover further criteria or
ask the user about that unresolved specification. No binary is authorized by
survival alone.

On failure/inconclusive: preserve raw artifacts and hashes, diagnose, stop all
downstream runs, commit the focused implementation/analysis, and finish the report
with explicit decisions. A documented scientific failure completes this campaign.
