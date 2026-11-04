# RF2 enlarged-center investigation (work in progress)

Branch `codex/refined-convergence`, based on `hyperboloidal_refined`21676d39e.
The persistent goal in GOAL.md is not complete. Stationary RF2 N16 SMR job9038055
reached10M, with worsening exterior errors. The original worker/PBS verification
failed on executable permissions; an independent identical-byte verifier passed
all scientific/artifact checks. All151 artifacts are preserved locally. No run is
currently queued or running from that historical campaign. The separate frozen BN1 campaign
and its final analysis are complete; its monitor was deleted. See state.json
before any action; never duplicate a run.

**Active experiment:** the coherent stationary outer gauge v2 is implemented,
locally qualified and frozen on Aurora. Fresh N16 job9038410 is running in
stationary-reference-v2, sourceb52f09723, four debug-scaling nodes48ranks,
dt=.000125, target10M. Build3464707 completed with-j64 in645.84s. All29 source,
build and launcher records are locally hash/length verified; actual binary
4220cf70111f829fc8aa3e244f0cd04ad57e2477ca74c294e08c0bb880c7104c.
Both user debug queues were empty before the single submission. Combined
footprint38.583GB leaves8.417GB after2GB output and1GB atomic reserves under50GB.
No purge was needed. Initial-only N12/N14 plus preserved N16 t0 diagnostics
show exterior H/M residual orders 3.89/3.66; no evolved convergence claim.
The SMR topology/boost-reference coverage audit also identifies the distinction
between puncture-only and horizon-neighborhood coverage. See
STATIONARY-REFERENCE-GAUGE.md and state.json for the
active plan and terminal verification/healthy continuation policy.
The v1 precursor build is preserved but must not launch: a local perturbation
check found a positive lapse response, repaired in v2 by consistently targeting
K-K_*. This is not a stability or convergence proof. Never repeat a successful
build/freeze or restart a failed state.

**Current update:** both enlarged stationary controls failed at the same
accepted step1237 (0.463875M): separated-driver9038253 and constant-driver9038285.
All151 artifacts from each run are locally hash/length verified and both terminal
locks are released. Neither may restart. The reference v2 job above is a fresh control.
The outer failed nominal fields (r>.875) differ by at most9.032e-4 scaled;
changing the inner driver did not prevent this early failure. See
`stationary20-constant-failure/` and `compare_driver_failures.py`.

`INITIAL-GAUGE-ATTRIBUTION.md` identifies an independent continuum initial-gauge
mismatch: the free shift removes the cancellation of the native slicing shear,
while the original restoration strongly drives a compatible shift away again.
This suggests a coherent reference-gauge control, not a shift-only replacement.
The full regularity and convergence goal remains unmet.

The failure message “nonpositive interpolated scri geometry” combines two
different checks: ordinary AMR halo geometry and surface interpolation. It
cannot alone locate the failure. The saved active capture is finite, has
positive chi everywhere and minimum metric eigenvalue0.232664. Read-only
reconstruction on the matching local topology found all3503736 non-boundary
halo rows valid. The13824 surface-dependent rows cannot be reconstructed
exactly: the old AMR driver omitted its already available failed surface.
Thus boundary-dependent interpolation is the remaining suspect in this
reconstruction, subject to CPU/SYCL arithmetic differences. No failed endpoint
is admitted and no slicing root cause is established.

At the last diagnostic0.45M, the horizon still passed (RMS3.18e-5), while fixed
r>.6 H/M were.06678/.62990 and cumulative scri pole/clock maxima161.81/53.04.
Null projection remained near roundoff. These maxima are not instantaneous
pole values, and roundoff nullness does not establish regularity. The history
plot preserves all four recorded times without inventing intermediate data.

The matched fresh constant-driver case changes only the inner shift policy,
retaining the large center, initial slice, three SMR levels, frame-aware halo
interpolation, twelve shells and dt.000125. It is not a continuation of the
failed state. A local diagnostic-only follow-up now separates halo and surface
error counts and saves failed-surface data separately; it has **not** changed
the frozen failed-run binary. Existing reduction results are reused, with no
new halo kernel or collective in healthy steps. No measured speedup is claimed.

## What is established

The original sampled boosted initial slice is spacelike and its spatial map
is nonsingular. The minimum normalized spacelike margin Q/Q0 is .9940224;
it is exactly1 in the suspect compact region .3–.4. This does not exclude
an evolved slicing pathology. Prior BN1 endpoint evidence instead identifies
large derivative disagreement between independently evolved copies near the
first shell interface .305. That is a numerical suspect, not causal proof.

`run_initial_audit.py` reproduces independent native initial-geometry checks
at968points per case for stationary and v=.1 configurations. The generalized
native initializer preserves the original4–12M fields exactly on these samples.
For the most enlarged20–40M slice the sampled minimum Q/Q0 is .9684319 and
boost-map detJ is .6943910; all sampled radial/angular map derivatives are
positive. Small absolute Q near scri is normalized by the unboosted Q0 rather
than interpreted as failure. The physical radial map is distinct from Omega;
changing the conformal frame does not replace that map with r/Omega.

The configuration/frame oracle calls the actual Cartesian Z4c RHS. The largest
sampled RHS difference is2.46844e-12; physical constraint frame-invariance and
map inversion checks pass. These are local consistency checks, not a coupled
stability or convergence proof. Production initial data and evolution use the
same configured Omega, native slicing interval, and map normalization k.

## Controlled candidates

| candidate | core half-width | shell overlap radius | flat Omega / end | gauge interval | native areal slice |
|---|---:|---:|---:|---:|---:|
| moderate | .24 | .21 | .24 / .30 | .36–.65 |6M–18M|
| large | .40 | .35 | .40 / .54 | .62–.78 |20M–40M|

Original plateau .5 controls are retained. Matched-plateau controls use
Omega_flat=(1-r_flat²)/2: .4712 and .42. They reduce peak sampled |Omega''|
from140.36 to40.01 (moderate), and79.73 to29.50 (large); the original RF1
narrow join gives171.25. `plot_initial_controls.py` produces the inspected
figure and hashes its raw inputs. The plot shows initial profiles only.

The larger configuration's inscribed Cartesian sphere is about11.86M in
isotropic radius, and gauge onset about18.4M isotropic /19.4M areal. The
20–40M label specifies the native areal slicing interval, not the Cartesian
box size. Its initial-map scale k=.1011686354 differs from original.3061422834.
At N16, unrefined physical Cartesian spacing is .3707M; three refinement levels
reduce it to .0463M. The moderate case with one level gives .0501M, comparable
to original BN1 .0551M. Enlarging the domain without accounting for this map
normalization would substantially underresolve the puncture.

Restoration coefficient4 and .6–.9 collar, KO8, original scri derivative,
fixed effective scri penalty, auxiliary driver and floors remain. Several
parameters change together in these exploratory configurations; isolate them
with controlled comparisons before attribution. The origin-centered .03
positive-chi floor patch has a changed physical size when k changes and needs
explicit audit. Gauge blending in the large case overlaps the restoring collar.

## Native SMR qualification

Reuse AthenaK Cartesian tree/transfer infrastructure with configurable core
bounds, overlap and initial refinement. RF2 identities bind all new parameters;
AMR identities additionally bind `dynamic|fixed|uniform|smr`. Policy changes
cannot silently restart a different formulation. SMR is constructed before
analytic fields are installed. Initial construction bypasses only the regrid
age interval; evolution retains its normal age rules.

The moderate matched-plateau stationary case at N12, dt=.00025, one level
produced120Cartesian leaves and passed two steps. Two MPI ranks through step1,
then one MPI rank through step2, reproduce all eight saved native state,
compensation and event arrays bit-for-bit versus uninterrupted MPI2.
`check_smr_restart.py` records hashes, configuration, topology and finite-array
checks. A changed policy is rejected. Horizon seed differs by2.76e-8 after
finder reinitialization; the checkpoint files therefore are not byte-identical,
even though evolved fields are. Final horizon mass is about.33333339545,
null residuals <=1.23e-15 and no floors occurred. Time is only .0015M.
This does not qualify long evolution, GPU behavior, moving refinement or
three-level SMR. The older `central6-smr-check` predates policy binding and is
retained as development evidence, not a restart input for the final binary.

The earlier uniform `central6-t0` development report incorrectly labeled its
slicing interval4–12M while identity/fields used6–18M. Report metadata is fixed
in source. Do not use that old report as a fully qualified production gate.

## Performance and remaining work

Existing optimized halo construction, sparse exchange, device evolution and
native AMR machinery are reused. Local MPI2 Serial measurements (not Aurora
scaling) show approximately10.3s per step; cumulative bulk halo time9.83s
versus RHS7.73s across the earlier two-step check. Setup is about20s and
performing full diagnostics/horizon finding every step adds seconds per step.
Timers overlap as stated in their native records; do not sum nested and
exclusive timers or extrapolate this CPU fixture to GPU costs.

The production extension below now supplies scheduled diagnostics, healthy
walltime checkpoints and complete candidate/fixed-cutoff histories. Its local
field equivalence and restart checks pass; the frozen SYCL build completed
with make-j64. The first GPU timing and stationary10M results are now measured below.
Next causal controls must test interface derivative/overlap behavior and
monitor evolved slice/metric regularity. Stationary qualification precedes
boosted runs, with resolved puncture spacing and controlled timestep, followed
by common-domain exterior and field-resolution analysis. Survival alone is
not convergence.

The analysis-only RF2 reader and its qualification are described in
INTERFACE-AUDIT.md. `build_endpoint_diagnostic.py` builds it locally without
rebuilding campaign binaries. For an independently admitted checkpoint use
`reader N checkpoint dt config smr fresh_prefix cutoff` under MPI, then retain
all rank CSVs and metadata. `check_endpoint_diagnostic.py` verifies the preserved
N12 fixture against its production history and confirms unchanged checkpoint
hash and rejection of a different configuration. Current qualified error is
2.10e-16 for native volume integrals and1.57e-16 for saved-history norms.
This is operator consistency, not an independent continuum-accuracy test.

SMR retains first-shell and outer-core independently evolved overlap cells.
Zero overlap in an individual rank's performance record does not mean that
the global formulation has removed them. The new reader preserves copy IDs
and positions rather than smoothing them into one curve.

## Production driver extension

Configured AMR accepts three optional trailing arguments:
`max_seconds diagnostic_interval target_time`. With these present, the target
is coordinate time (10M=10/3), the requested dt is held fixed after its
characteristic-bound check, and the final step is shortened to hit the target.
Diagnostics are scheduled on an absolute coordinate-time grid, with initial,
walltime and final rows preserved. A healthy walltime checkpoint returns3;
a numerical or I/O failure returns2 and cannot be used as a continuation.
The old per-step qualification invocation remains available. The synthetic
`dynamic` regrid fixture is rejected in production mode; production currently
uses fixed SMR, not a claimed moving adaptive criterion.

One native constraint pass accumulates candidate-enclosing and fixed cuts
r>.1,.3,.6, with H/M/Mr and shift/D derivative maxima. Full candidate acceptance,
expansion and mass gaps are recorded. Scri pole/clock values are cumulative
maxima. Counter snapshots are cumulative across restarts, with bulk, surface
and boundary events separate; segment differences must be taken rather than
summing cumulative reports. The native checkpoint retains detailed event data.
A restart topology reconstruction now explicitly permits initial multi-level
tree construction without the evolution regrid-age delay.

`check_production_controls.py` verifies identical native/compensation/event
arrays against the pre-extension two-step stationary test, both with reduced
diagnostic cadence and after a healthy walltime checkpoint (MPI2 to MPI1).
The horizon seed is not bit-identical because finder calls differ. This is
not a long-time or GPU qualification. Source/build provenance for the next
Aurora experiment is separate from all frozen RF1 and BN1 snapshots.


## Aurora production and deeper SMR control

Scientific commit b41c24e88790562d93a643be9c713631bb93d573 is frozen at
`/lus/flare/projects/CompactBinaryMerger/hzhu/refined-convergence-20261104-v1`.
The actual SYCL binary hash is
`1a694bca3e9bca020bee2a112911fde7ee28032c772972b5d2c05e67480efaca`.
All22 source/build/launcher records were locally hash/length verified.
Stationary6 N16 job9038055 uses four nodes48ranks, dt=.00025, target10M,
one SMR level, matched plateau.4712 and gauge interval.36–.65. The original
scri/restoration remains. Frozen BN1 N24 job9037997 has completed and its
follow-up is preserved. No cancellation or duplicate submission is authorized.
No GPU smoke gate.

`launcher/` binds source manifest, actual binary, configuration, launcher,
PBS script and restart (if present). The streaming verifier checks RF2 framing,
finite payload, positive chi/metric, exact array layout and native checksum;
matching native/worker/PBS disposition, identity, steps/time, tracker and
penalty are additionally required. A truncated checkpoint rejects. Failed
states are never restart inputs. Native return3 denotes a verified healthy
walltime checkpoint; continuation retains the same scientific identity.
Preserve every artifact locally with hash/length verification before any purge.

The three-level large-center layout with refinement radius.05 requests
8,8,64 blocks on successive levels; native balancing expands the third tree
past the test capacity (704blocks/rank versus264). This is not an evolved
slicing failure. A narrower .025 location criterion requests8,8,8, producing
120,176,232 Cartesian leaves; reconstruction of that native tree passes MPI2.
The narrowing is a separate configuration, not a silent change to .05.

The .025 variant also passes two local Serial MPI2 steps at N12, dt=.000125,
t=.00075M with no floors, finite/SPD/checksummed admitted RF2 state. Raw
history, report, checkpoint verification and performance are preserved here.
It is not a long-time or GPU test. The topology criterion intersects blocks,
so its input radius is not itself the exact refined coverage boundary; moving
puncture coverage still needs explicit verification. The large configuration
changes physical map scale and the physical extent of the fixed .03 chi patch.
Do not infer stability or improved accuracy from this short control.


### Startup disposition and measured launcher overhead

Job9038031 terminated before native evolution, worker/PBS2. Both requested MPI
rank environment variables were empty in Aurora's shell placement probe; the
masks and hosts were present, but positional parsing failed. No cancellation
was issued: the safety check detected that it was already terminal. All
recorded artifacts and scheduler/lock-release receipts are preserved under
startup-failure-v1/. It consumed296s in pre-native startup including a global
storage scan. This is infrastructure evidence, not a numerical failure.

Immutable launcher-v2 validates the observed four-host/12-distinct-mask layout;
a duplicate-mask regression rejects. The global storage scan runs in a
background guard, while the submission-time total-capacity check and a cheap
per-allocation output/atomic budget remain enforced. An old submission storage
snapshot (>300s) still triggers a fresh scan. Native walltime is bounded by
remaining worker time with180s exit reserve. The scientific binary, source,
configuration, operators and dt are unchanged.

The v2 submit preflight also discovered that Aurora's qstat -u ignored its JSON
format request. It failed before intent/qsub and created no job. The separate
hash-recorded submit_reviewed.py uses the observed text format, checks all user
jobs and both queues, and preserves the same one-shot intent/lock protocol.
Use it for future healthy continuations; frozen v2 control.py collect remains
the terminal verifier. No frozen launcher or science snapshot was edited.

Historical observation: job9038055 passed four-node placement and started native evolution,
reaching200steps/.15M at that observation. Its terminal disposition is now below.
Its plan hash is f1c5847369aceee53c3c16f4d045fcf619fc95aaa5841851156303b724df0f51.
Do not duplicate it. Follow state.json before any action.


## Stationary10M disposition and causal controls

Job9038055 completed13334 native steps in2414.76s (40.25min), dt=.00025.
The worker then could not execute launcher-v2/verify-checkpoint (mode0644),
so original worker terminal_failure/PBS2 remains immutable. Frozen collection
verified all151 artifact hashes and released only the matching allocation lock.
`review_terminal.py` executed an identical-byte copy from a separate review
path and passed every frozen controller scientific check. Its receipt qualifies
the endpoint for read-only analysis, not a restarted run or an accuracy claim.
Local preservation, original failure, independent review and figures are in
stationary6-terminal/. No frozen launcher/source/binary was edited.

The same-operator reader reproduces the full report to roundoff. At10M,
fixed r>.3 H/M/Mr =2.285843811/3.165588429/6.372486203; r>.6 H/M =
.0643872723/.1207173464. Of the squared error outside .3, .3–.6 contains
99.9362% H and99.8830% M. The peak is around .36–.40. Native-center profiles
show much larger angular error near shell face corners: sampled max |H|/M
in .3–.6 is118.68/139.84 there, versus .332/.461 near the face center.
These are different angular samples of a stationary spherical problem, not
an interpolated jump at identical coordinates. The metric/chi stay admissible,
but resolved accuracy and good gauge behavior have not been established.

Online horizons accept56/68 rows, last8.25075M; post expansion quality<=.01
passes58/68. Final RMS=.01580349 and final mass is missing. No lapse floor;
chi stage/final floors20944/7024. Tiny null residuals do not cure this failure.
Pole/clock2.72225/2.27961 are cumulative maxima, not current boundary values.
Full norms near the puncture are not used as an exterior accuracy criterion.

Exclusive per-rank median times: stepping2315.94s, diagnostics96.83s,
evolution setup35.98s, initialization7.30s, capture/write~1.88s. Nested RHS,
exchange, closure, halo and error synchronization timers overlap; they must
not be summed into a speedup estimate. Four-node data are not controlled strong
scaling. Measured step/closure/exchange costs should guide further performance
work; repeated launcher tests or output optimization are not priorities.

The enlarged20–40M N16 initial controls keep geometry, gauge, Cartesian SMR
and angular resolution fixed while varying radial shells. Core nominal fields
are checked bit-identical. Six shells give full H/M=.0572033/.2387504; eleven
shells refining .54–.875 barely change them. Nine shells targeting .875–1
reduce them to .00320683/.00143734. The compressed late initial-slicing change
lies around r=.90–.93, not the earlier gauge-stitching interval. Final shell
spacing and hence boundary stencil sampling/penalty CFL also change explicitly.
The nine-shell two-step Serial MPI2 check passes, CFL .188389 atdt=.000125;
it is not a long/GPU result. See initial-radial-controls/.

Further refining the Omega/interface region with twelve shells instead raises
H to .0421891, leaving M=.00143734. A read-only t0 control replacing **only
Cartesian-to-shell ghost values** in .33–.60 by exact values reduces H to
.00044628, with active fields and M unchanged. This isolates a numerical halo
interpolation contribution; it is not an evolution with analytic boundaries.
Frame-aware interpolation using only donor values and known Omega achieves
essentially the same reduction. Its bounded test, limitations and reproducer
are documented in INITIAL-INTERPOLATION.md. No production repair is frozen yet.

Separately, PRINCIPAL-AUDIT.md records radial frozen-symbol Jordan-block
evidence at lapse/shift speed coincidences in both initial configurations.
Pushing the gauge outward alone does not remove these defects. The inner
crossing also belongs to the corresponding spatial gauge, so it is not a
uniquely hyperboloidal effect. Neither audit alone attributes all late errors.
Next: validate a conformal-frame-aware interface repair and independently check
an explicitly well-separated gauge control before further stationary/boosted
production. Preserve stationary-first sequencing and the full convergence goal.


## Production frame and gauge controls

See [FRAME-AND-GAUGE-CONTROLS.md](FRAME-AND-GAUGE-CONTROLS.md) for the new
opt-in halo interpolation and separated inner-driver policies, derivation,
qualified tests and next paired stationary comparison. The N16 production
initial Hamiltonian norm reproduces the read-only interface improvement.
The two-step local N16 control is admitted with CFL .188389, fixed penalty
.6510416666660156 and penalty CFL .3125. Twelve radial shells, deeper SMR,
the larger center and dt .000125 differ from the completed stationary6 case;
that historical case is not a one-parameter gauge comparison. No new long
result or boosted convergence has been established.

The separate Aurora frame-separation-v1 build completed with source342715a90,
actual SYCL binary f548c09f458fad9f74508cb629ee0f49e4ec8808c824f211056fe5f186c2f892.
All30 frozen source/build/launcher records, including verifier executable mode,
are locally verified. Requested module oneapi2025.3.1 resolved compiler
IntelLLVM2025.3.2; actual compiler/MPI/module records are preserved, not assumed
identical to an earlier build. Build used-j64. Read state.json and saved
submission for the current stationary20 separated-control job.

The matching new endpoint reader reproduces the local two-step production
norms exactly and leaves its checkpoint hash unchanged. A separate read-only
frame-halo control on the older stationary6 10M state does not reduce its
large late exterior errors; see late-frame-control/ and the explicit support
qualification in FRAME-AND-GAUGE-CONTROLS.md.

Fresh stationary20 separated-control N16 job9038253 is submitted, plan
campaigns/stationary20-separated-n16-segment01/plan.json under nested remote
frame-separation-v1. Four debug-scaling nodes48ranks,dt=.000125,target10M,
three-level SMR and twelve radial shells. Submission checked all user jobs
and both debug queues; held capacity work was preserved. Storage footprint
36.046GB leaves10.954GB after2GB output and1GB atomic reserves under the50GB
combined cap. No purge was needed. Follow the saved job and immutable
launcher-v3 for terminal verification and healthy same-case continuations.
This historical submission was followed by the matched constant-driver job9038285;
both are now terminal failures as recorded at the top. No boosted run submitted.

### Initial native RHS and selective radial refinement (2026-11-04)

The initial geometry and its continuum equations are regular in the outer
transition, but the production discretization has substantial truncation error.
`analyze_initial_rhs.py` and `initial-native-rhs/` reconstruct the analytic t0
state on admitted N12/N14/N16 checkpoint topologies, evaluate the actual native
RHS, and reproduce the saved initial full/fixed-cut constraints. They take no RK
steps and leave checkpoint hashes unchanged. In .90<r<.95, the RMS rates of
chi, metric, A, K, Lambda, lapse and D fall with fitted residual orders 3.64–3.88.
Bulk KO removal in this analysis has exactly zero effect there: its original
taper is already zero. Face-center and corner profiles nearly coincide in the
peak band. These are initial semidiscrete residuals, not evolved convergence.

An N16 radial-only initial control reduces spacing fourfold in .90–.95 by
splitting two shells into eight; core, angular resolution, geometry, gauge,
SMR leaves and the last .95–1 scri shell are unchanged. The band RMS rates fall
by factors 206–246 (A:216.3, K:221.9, Lambda:206.0). Initial fixed r>.3 H and M
fall from .000264273845/.001457127855 to .000079709595/.000275602016,
factors 3.315/5.287. `compare_radial_initial_rhs.py` verifies exact reproduction
of the new initial constraints and unchanged checkpoint hash. The diagnostic
uses its admitted .00003125 probe timestep, versus .000125 in the baseline;
there is no time integration, and this is not an evolved matched-timestep test.
The smaller step would increase evolution cost, so the next evolved control
must distinguish radial from temporal changes. The original Aurora run remains
unchanged. This isolates a large radial discretization error, not the sole
cause of all late error or a proof of nonlinear regularity.

Reproduction: build the separate analysis reader with
`build_endpoint_diagnostic.py --body initial-native-rhs/body-v1.inc --output
output/refined-convergence/initial-native-rhs`, then `run_initial_native_rhs.py`.
The radial control reader instead uses `--body initial_native_rhs.inc --output
output/refined-convergence/initial-native-rhs-probe-dt`, then
`run_radial_initial_rhs.py`. Run the two analysis scripts above for plots.
Recorded compile/link commands, binaries/source hashes, commands and raw small
results are preserved; existing output directories/records are never overwritten
by evolution runners. Figures were visually inspected. These commands need the
preserved input checkpoints and the matching local production build.

`stationary-reference-v2-early/` is an immutable live snapshot through1.05M,
compared with the old .463875M failure without interpolation. The new coherent
gauge markedly reduces cumulative pole/clock maxima, while exterior constraint
errors still grow. It is not terminal admission. Later live observations belong
in state.json; do not reinterpret cumulative maxima as instantaneous values.

Performance: the local radial control required147.2s for zero-step setup,
checkpoint and initial diagnostics plus135.4s for the read-only RHS analysis.
These are Serial/MPI2 end-to-end measurements, not a GPU speedup comparison.
The running four-node original radial12 control takes about78s per400 steps,
roughly .195s/step, close to the old control. The reference-source cache avoids
per-RK quadrature, new kernels and new collectives; no speedup is claimed from
this structural improvement. Fresh storage admission consumed about378s of
startup and the worker correctly reduced its native time budget. No frozen
launcher or running scientific binary was modified for performance work.

The bounded evolved follow-up is specified in `RADIAL-CONTROL-PLAN.md` and
has not been submitted. It holds dt equal between radial12 and radial18 and
uses the existing radial12 evolution to assess temporal sensitivity separately.
This prevents crediting a spatial refinement for an improvement caused only by
its smaller step. The active four-node run continues unchanged.

The new radial18 grid's local two-step Serial/MPI2 CFL probe has now passed:
the unchanged native nonproduction timestep selector chooses
3.3745118776273217e-5 from a requested .000125 cap, with measured maximum CFL
.2, null/ghost residuals below1.12e-15, zero floor counts and an admitted
finite/SPD/FNV checkpoint. This confirms .00003125 as a suitable initial cap
for the proposed matched comparison; later CFL still requires monitoring.
Evidence/commands are in `reference-radial18-two-step/` and
`run_radial_cfl_probe.py`. This is neither a GPU check nor long evolution, and
its checkpoint will not be used as a fresh experiment's restart input. Runtime
235.14s is local end-to-end, not a GPU estimate or a scaling measurement.

### Bounded controls and storage traversal preparation

`launcher-v7/` and `prepare_radial_controls.py` prepare the equal-dt fresh
radial12/radial18 controls in `RADIAL-CONTROL-PLAN.md`. They reuse the exact
frozen b52 production binary and checkpoint verifier; only runtime grid input,
requested dt/target/sampling and launcher namespace are separately bound.
No rebuild, GPU smoke or change to launcher-v6 is required. Both controls
remain unsubmitted while the existing reference10M run is active.

The measured ~378s pre-native admission delay motivates a separate storage
traversal in v7. `storage_accounting.py` retains the50GB cap, all roots,
allocated-versus-logical maximum, inode deduplication and reservations. It
avoids traversing a directory twice and rejects unreadable directories rather
than silently missing their contents. `test_storage_accounting.py` checks
known inventories, nested/overlapping roots, hard links, root and child
symlinks, cycles, dangling links, missing roots and exact/over-cap behavior.
It matches the frozen scanner on all valid inventories. No running guard was
modified.

On an immutable Aurora source/build inventory with a nested duplicate root,
both implementations counted exactly655642624B/5610 unique inodes. Old/new
wall times were8.334/1.830s in that order and1.684/4.504s in reverse order.
These results are cache-sensitive and do **not** establish a reproducible
speedup or predict full admission time. The deterministic improvement is
removal of redundant walks; future allocations will record end-to-end startup
cost. All measurements, source hashes and integrity checks are preserved in
`storage-performance/`. No additional launcher tests or repeated physics
checks were performed.

The v7 bounded-control package is now installed alongside v6, not submitted.
All15 records, modes, reused verifier and unchanged scientific source/manifest/
binary hashes were checked. Archive SHA256
`e928da71a072799d74a069184b0fa53c0678b99c574d8335075d50f51f4a1b33`;
launcher manifest SHA256
`05f348183df6a55d46ed61d36d95bb7481134b4969793448ef93a5ff4a7e4acc`.
A complete new traversal counted38,583,739,636B over179187 unique inodes in
57.11s. This is an observed full scan, not a matched comparison with the earlier
startup delay. No plans/submissions were created and the active job remains
unchanged. See `radial-controls-preparation/`.
