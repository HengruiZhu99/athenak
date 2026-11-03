# BC2 refined Cartesian core: completed 10M, accuracy remains qualified

Job9036411 completed13,334 steps to coordinate10/3 (10M), native return0,
workerpassed/PBS0. Native runtime2358.10s on two nodes/24 GPU tiles. Frozen
helper verified the admitted BC2 checkpoint, exact configuration, fixed scri
penalty, finite payload, forward progress, tracker/report agreement and all
source/binary/input/artifact bindings. All five worker artifacts were copied
and locally hash/length verified. The matching terminal allocation lock was
released. No continuation, other resolution, purge or AMR work was performed.

Checkpoint:259205840bytes, SHA256
`c59bdeeb6e1330954bd4fa0dba1a5f14d42a370ddc9e8122e9da1978293c1cbf`.
Raw data remain in output/boosted-refined-core/aurora/campaigns/
n16-boosted-refined-core-cfl-v2-c16-evolution-segment01, with remote originals
retained. The original BG1 failure and earlier launcher failures remain intact.

## What the comparison establishes

The combined core refinement and matched-CFL timestep reduction avoided the
coarse BG1 fatal metric failure at4.23M and reached10M. It does **not** establish
that spatial refinement alone fixed the problem: core spacing and timestep
both halved, MPI placement also changed, and the effective boundary penalty
was held fixed. There is no admitted BG1 endpoint at10M and no resolution order
can be inferred from these two runs.

BC2 improves the duration of accepted horizon tracking:25/68 recorded rows
accepted, last3.75M, versus BG1's4/29 rows and last1.5M. BC2's last accepted mass
error is+0.0011018% at3.75M; it is **not** a10M mass measurement. Every subsequent
BC2 horizon row is unaccepted. At10M expansion RMS.002427 exceeds tolerance.001.
Mass gaps are left blank through10M in trajectory-horizon.png.

Floors are a substantial qualification. BC2 chi-floor counts are22,188 stage /
7,484 final; pre-floor negative chi counts19,392 /7,000. Lapse-floor node
interventions total3,032, with negative-lapse stage/final counts2,272/760.
No surface or interpolated-scri sign interventions occurred. These are event
counts over different durations/step counts, not normalized accuracy measures.
BG1 had60/20 chi-floor counts and no lapse-floor events before failing.

The trajectory is strongly gauge dependent: BC2 recorded x from-.0751745 to
+.0495474, with final position+.0495474. Minimum sampled cube margin is.104826,
and conservative support margin.084826. The puncture did not reach the gauge
blend beginning at r=.2 in the recorded history. This is not proof that all
relevant stencils or propagated gauge effects are confined to the plateau.
The final volume metric is positive (minimum eigenvalue.816830) and finite.
This is an endpoint check, not a smoothness/convergence proof.

The29 common recorded times through4.2M are matched without interpolation.
Own-candidate exterior histories remain close before the coarse failure,
while BC2 full-domain Hamiltonian exhibits larger early spikes. At4.2M,
BG1/BC2 candidate H=.152595/.143458, M=.146533/.144947, but cuts differ
(.146991/.133107), so these are not same-domain error reductions. Fixed-cutoff
historical fields were not saved and are not reconstructed from these CSVs.

Scri null residuals remain<=1.56e-15. Pole/clock cumulative maxima6.03363/14.0130
are essentially unchanged from BG1's6.04077/14.01765. Algebraic null closure
therefore does not establish small regularity residuals or overall accuracy.

## Admitted endpoint and profiles

The matching native BC2 diagnostic reproduces full report H/M/Mr norms to
relative1e-9 or better. At fixed r>.3, H=.01448145953, M=.02303339626,
Mr=.01378313985. This cut exceeds all recorded candidate enclosing cutoffs
(maxBG1.220870, maxBC2.211392); no accepted final horizon exists. The diagnostic's
legacy `horizon_acceptance_assumed=true` metadata only reflects a supplied cut
and must not be interpreted as a successful horizon search.

Of the squared H/M integrals outside r=.3, region .3-.6 carries95.493%/91.761%,
.7-.775 carries3.892%/6.747%, and .775-.9 carries.123%/.457%. These are BC2
endpoint regional contributions, not matched-time reductions against BG1.
Raw radial integrals and bins are in endpoint-radial.json/radial-regions.json.

Read-only unsmoothed Cartesian and shell profiles are preserved. Their CPU
constraint formula reproduces native full-volume and r>.12 integrals with
maximum relative difference2.82e-16. The Cartesian line y=z=-.0028125 still
shows sharp metric/D/constraint structure near the tracked puncture, including
sampled H up to24.59 and Mnorm up to14.21. Radial projections reverse direction
near the coordinate origin; that projection feature must not be mistaken for
a physical discontinuity. Shell D gradients peak near r=.75, with directional
variation. These finite-grid profiles do not establish regularity or convergence.

## Metadata qualification and reproducibility

Frozen `cartesian_dt_over_dx` still uses legacy spacing .18/n and reports
.0222222. Actual refined spacing is .36/(4n)=.005625, hence dt/dx=.0444444,
matching coarse .0005/.01125. The measured characteristic CFLmax is.272775.
No frozen source or output was changed to correct this metadata; the independent
calculation is in endpoint-state-audit.json.

Reproduce preservation, identity checks, history figures and radial analysis:
`python3 docs/hyperbolic-shift-3d/reproduction/boosted-refined-core-v1/analyze_completed.py`.
Matching diagnostic command, flags and hashes are in endpoint-provenance.json.
Read-only profile source is ../endpoint_slices.cpp; exact compile/link/run
commands, working directories and hashes are in slices-build.json and
slices-run.json. They do not step or write checkpoints. Replot with
`python3 docs/hyperbolic-shift-3d/reproduction/boosted-refined-core-v1/plot_endpoint_profiles.py`.
All six figures were visually inspected. Original checkpoints were unchanged.

The authorized fixed-grid experiment and its comparison are complete. The
hourly monitor is retired after preservation/reporting. AMR remains a separate
thread task described by ../AMR-HANDOFF.md. No additional run is implied by this
result: improved survival is demonstrated, accurate boosted-hole evolution is
not yet established.
