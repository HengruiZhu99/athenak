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

The v7 bounded-control package was installed alongside v6 before submission;
its subsequent job dispositions are recorded below.
All15 records, modes, reused verifier and unchanged scientific source/manifest/
binary hashes were checked. Archive SHA256
`e928da71a072799d74a069184b0fa53c0678b99c574d8335075d50f51f4a1b33`;
launcher manifest SHA256
`05f348183df6a55d46ed61d36d95bb7481134b4969793448ef93a5ff4a7e4acc`.
A complete new traversal counted38,583,739,636B over179187 unique inodes in
57.11s. This is an observed full scan, not a matched comparison with the earlier
startup delay. No plans/submissions were created and the active job remains
unchanged. See `radial-controls-preparation/`.

### Admitted reference-v2 first segment and outer error localization

Job9038410 ended in a healthy walltime checkpoint at4.782375M/12753 steps.
Native/worker checkpointed and PBS3 agree; all152 worker artifacts are locally
hash/length verified and the matching lock is released. Job9038494 continues
the exact checkpoint on four nodes with unchanged science and dt; it is not
a new fresh run. Both input and full evidence remain preserved.

`stationary-reference-v2-segment01/` contains the verified records and
`analyze_reference_segment.py` / `analyze_reference_spatial.py` reproduce the
history, timing, radial-integral and unsmoothed profile plots. All33 recorded
horizons were accepted; maximum absolute mass error is0.00083656%. This
does not establish exterior accuracy: fixed r>.3 H/M at4.782375M are
.09861974/.67702628, with99.868%/99.843% of their squared integrals in .90–.95.
Both peak radial bins are .92–.93. The Omega and gauge joins contribute little
to this exterior integral at this time. Native profile rays show steep lapse
and shift gradients plus constraint oscillations in this same band.

The initial radial-only control already reduces this band's native RHS
residual206–246x. Its overlap with the evolved error strongly motivates
`RADIAL-CONTROL-PLAN.md`; it does not yet prove that radial refinement cures
late evolution, that the slicing remains uniformly regular, or that a boosted
puncture is resolved. No full-domain puncture spike is used as an exterior
accuracy measure. Scri null residual remains near roundoff; cumulative pole
and clock maxima are .00834451/.01375982, not instantaneous residuals.

`prepare_reference_endpoint_reader.py` creates a separate analysis-only
reader with an explicit exact backend identity calibration: native quadrature
yields local k=.10116863541976967 versus frozen Aurora .10116863541976966,
and the corresponding gamma differs in the final few bits. Only these known
values are calibrated; the full frozen identity is still required. Original
checkpoints, campaign binaries and production admission are never changed.
The wrong local identity is rejected. The read-only reconstruction reproduces
GPU full/fixed norms to5.8e-14 scaled and preserves the checkpoint hash;633
production source files match the frozen manifest. See
`reference-endpoint-reader/` and `reconstruct_reference_segment01.py`.

Measured four-node native command cost was2592.93s; rank-median stepping
2445.64s/12753=.19177s per step. Setup82.1s, initial state8.7s, diagnostics45.2s.
RHS/exchange/closure/halo/error-sync timing categories are nested and overlap;
they must not be added as an exclusive cost breakdown. No strong-scaling
speedup is established. RF2 floor/sign counters are cumulative across restart:
use final totals or segment differences, never sum cumulative reports.
First-segment chi floor totals are203472 stage/67832 final; lapse floors zero.

Current debug-scaling max_run is one job per user. Therefore the installed v7
matched-dt .6M controls follow the completed reference sequentially;
no new parallel launcher or duplicate submission is needed.

### Initial versus evolved slicing profiles

`reference_profile_oracle.cpp` evaluates the unchanged native initializer and
stationary gauge at exact evolved sample radii and a dense reference grid.
`run_reference_profile_oracle.py` builds this analysis-only program;
`compare_reference_profiles.py` reproduces the inspected figures and data in
`reference-profile-regularity/`. There is no evolution or checkpoint rewrite.
Independent1e-5/5e-6 derivative probes differ by at most1.70e-7 scaled.

The sharp lapse/shift transition is already in the analytic initial slice.
The initial maximum lapse gradient is29.48 and its sampled halfmaximum width
is .0213 in compact r: about13.63 original radial cell widths versus45.70 in
the proposed targeted grid (accounting for unchanged spacing below .9). At
exact evolved +x sample points, initial/evolved maximum lapse gradients are
29.395/29.273 and shift-deviation gradients30.259/30.092. Thus the steep
profile has not arisen from a new order-of-magnitude slicing collapse by
4.782375M. The metric and K acquire localized oscillatory deviations instead:
max |delta gtilde_rr|=.005490 and |delta K|=.05505 in the sampled .85–.96 band.
The evolved lapse differs by at most.003075 at these samples.

The native initial Schwarzschild-chart spacelike margin f/(f+R²S²) is
positive throughout the sampled .90–.95 band (.002537–.039112), and dr/dR is
positive (.002498–.018507). These are initial analytic checks, not a proof of
evolved coordinate regularity. Together with the initial RHS refinement
result and the evolved exterior localization they favor radial truncation
of a steep but initially regular slicing transition. The causal evolved
equal-dt controls remain necessary. No cell-count threshold alone proves
that a stencil is resolved, and puncture/boosted convergence remains open.

Rank-level cost evidence (`analyze_reference_rank_cost.py`,
`reference-rank-cost/`) shows28263–28264 active cells per rank but68096–249728
halo rows. Host RHS times range662.8–1025.3s and halo times49.4–380.6s.
Nominal sparse-send and receive payload sums agree exactly at1.399GB per
call over all48ranks. This is application accounting, not a network bandwidth
measurement; nested timers overlap and synchronization includes load waiting.
Existing measured-partition/repartition infrastructure is the appropriate
place for a later controlled load-balancing test, rather than assuming equal
cell counts mean balanced work. No running partition was changed. Figure
inspected, raw per-rank records and reproducer preserved.

A dedicated radial18 endpoint analysis reader is prepared in
`radial-endpoint-reader/`, using the same explicit frozen-backend k/gamma
calibration and strict full radial18 identity. The unchanged reference reader
body is bit-identical after generalizing the preparation CLI. The radial
reader rejects the local identity as expected. There is no admitted Aurora
radial18 endpoint yet: positive report/norm reproduction remains required
before any spatial result is interpreted. No production build changed.

### Boosted-follow-up coverage and topology cost

`probe_smr_topology.cpp` exercises the existing AthenaK octree/refinement,
balance, redistribution and registered native field-transfer lifecycle,
without evolving physical fields. `analyze_expanded_smr.py` verifies disjoint
leaf interiors, total core volume and rectangular finest-level unions.
Evidence and inspected coverage/cost plot: `expanded-smr-topology/`.

A wider all-direction location target produces1408 Cartesian leaves, with
a finest cube[-.1,.1]^3, versus232 leaves and[-.05,.05]^3 currently. The local
two-rank probe first exceeded its1040-total allocation capacity; this was a
topology-probe resource rejection, not numerical evolution failure. A separate
analysis allocation with2048/ranks+8 capacity completed. The production
four-node cap32blocks/rank would geometrically fit1408 leaves, but memory and
runtime have not been measured for this larger evolution. No guard changed.

Static rectangular target flags using the same Regrid API produce316 leaves
for an offset tube and400 for a symmetric tube. The latter's actual finest
union is[-.1,.1]×[-.05,.05]², with unchanged fine dx=.0015625. Together with
radial18 shells, its nominal cell count is1.494x the original narrow-SMR
radial18 configuration, versus4.459x for the wider cube. These are cell-count
ratios, not claimed speedups or exact checkpoint/memory estimates.

Under either sign of the reference k*v*t displacement at10M (magnitude.033723),
the symmetric tube accommodates an enclosing radius up to.04375 plus four
fine-cell widths; the original patch permits only.01003. The offset tube is
not robust to reversing the drift sign. These are geometric estimates: the
live gauge trajectory, evolving horizon and gradients must be checked, and
coverage is not resolution convergence. Rectangular target flags exist only
in this analysis tool, not yet as a production input policy. The frozen
stationary/radial controls are unchanged. Before a boosted run, qualify the
chosen static policy/initial fields and monitor actual fine-support margins;
use the existing AMR infrastructure rather than a separate mesh engine.

### Reference-v2 completed 10M; equal-dt controls in progress

Segment02 job9038494 completed10M/26667 total steps with native/worker/PBS
success and frozen terminal admission. All152 worker artifacts were locally
hash/length verified, the matching allocation lock released, and the unchanged
checkpoint reconstructed with the dedicated native analysis reader. Full and
fixed-domain report norms agree within5.0e-14 scaled. Evidence is in
`stationary-reference-v2-segment02/` and the joined histories in
`stationary-reference-v2-completed/`; figures were inspected.

All69 unique recorded horizon candidates were accepted online. Final mass
error is−.000344613%, maximum absolute recorded error.00105377%, and final
expansion RMS2.34891e-6. The native command sum is5405.71s (90.10minutes),
with median-rank stepping.191669s per step. Queue and inter-segment delays are
excluded; this is not a strong-scaling measurement.

The exterior error remains large: common r>.3 H/M=.1030394540/.7062455160,
with99.8754%/99.8558% of their squared integrals in .90–.95. The central
evolution and accurate horizon therefore do not establish global convergence.
RF2 cumulative floor/sign counters are unchanged after segment01; they must
not be summed across the restart. Read-only localization identifies all60864
stage and60864 final negative-chi candidates at the eight central cells,
r=.00135316, first encountered at.031875M. There are no lapse, surface or
boundary sign events. These are attempted pre-floor states, not negative
admitted fields. Their location is distinct from the dominant exterior error;
this alone does not exclude causal influence. See `reference-sign-localization/`.

Fresh time-control job9038572 used four nodes with the frozen v7 launcher,
unchanged radial12 grid and dt=.00003125, targeting.6M. It completed and was
preserved; paired radial18 job9038626 now uses the same dt/target. Neither
restarts from the completed10M state. No build, freeze,
smoke or long reference run is repeated.

Performance work is measurement-driven. The observed halo/work imbalance
motivates testing existing fixed ownership descriptors. The general
`Evolution::Repartition` entry point currently rebuilds `ContinuousCoreMesh`,
so it cannot be applied directly to this custom SMR descriptor. A separate
analysis-only static-SMR constructor probe reuses the existing strict measured
partition reader; it must pass MPI field/counter parity before a new production
policy is considered. Existing scientific snapshots remain unchanged.

Additional topology qualification: although the1408-leaf cube fits the stated
four-node block allocation geometrically, current RF2 restart/admission readers
cap leaf counts at1024. Thus that topology is not an admitted production
candidate without a separately versioned capacity change. The400-leaf
symmetric tube is below this cap. No capacity guard has been weakened.

The final10M profile comparison reuses the native analytic initial oracle at
the exact sampled radii (`reference-profile-regularity-segment02/`). Sampled
initial/final maximum lapse gradients are29.3953/29.2412 and shift-deviation
gradients30.2585/30.1101. Maximum sampled lapse change is.004659 and K change
.061804 in .85–.96. The steep profile remains close to its initial form, with
localized oscillatory deviations; there is no observed order-of-magnitude
collapse of the sampled outer lapse. This is not a uniform regularity proof.

Static-SMR constructor ownership qualification now passes both block-aligned
and partial-block cuts in two-rank SERIAL: the complete373986735-byte two-step
checkpoints are byte-identical to the preserved production baseline. See
`measured-partition-check/`. A48-rank candidate descriptor fitted to segment01
costs predicts lower local-work imbalance on segment02's independent timings,
but no new GPU partition has been run and no speedup is claimed. Explicitly
separate this performance preparation from the unchanged radial/time physics
controls.

The time-refinement control9038572 completed.6M/6400steps with native/worker/
PBS success and frozen admission. All152 artifacts are locally hash/length
verified; the unchanged checkpoint's native reconstruction reproduces full
and fixed report norms within1.02e-13 scaled. At the five common recorded
times0,.15,.30,.45,.60M, fourfold timestep reduction changes exterior H/M by
at most.0017933%/.0014662%. At.6M, fixed r>.3 H/M are.0821313833/.5660309414.
Thus the early exterior error is essentially insensitive to this timestep
refinement; this alone does not establish the spatial cause or time order.
See `time-control-comparison/`; its plots use only matched recorded times so
the different diagnostic cadences do not create a misleading visual gap.

Fresh paired radial18 job9038626 is now running on four nodes, with identical
dt=.00003125 and .6M target. Source, initial geometry/gauge, core/angles,
effective boundary penalty and final scri shell remain unchanged. Its bound
plan SHA256 is6aa4eee16a6fb122a561803934db61ce117166c140a6e65b9a39f14b88c68c8d.
No new performance partition is used in either physical control. Await full
terminal admission/preservation, then run `reconstruct_radial_control.py
--variant radial` and `compare_short_radial_controls.py`; never repeat the
completed time/reference evolutions or diagnostics.

### Bounded GPU ownership comparison prepared

The locally parity-qualified static-SMR ownership probe is now compiled for
Aurora in the separate `gpu-partition-package-v1` directory, against the
unchanged frozen libraries. Production source/object/library/binary hashes
were verified before and after; no existing campaign binary was rebuilt.
The probe binary SHA256 is
`c3a2970ddb1cd78908d233f54c45394375c67d1035f0c61653efdb80c0f2eca9`.
Build commands, source bindings and compiler output are in
`gpu-partition-build/`. This is not yet GPU parity or a measured speedup.

After the radial control completes and passes full admission, the proposed
bounded experiment uses its actual rank costs to choose contiguous ownership
cuts. It compares equal and weighted ownership from the same admitted restart,
for256 steps on the same four-node allocation, with identical solver, geometry,
mesh, timestep, effective boundary penalty and diagnostics. The existing
frozen RF2 verifier checks each result. Full checkpoint byte equality and
measured stepping time, distinct from startup/diagnostics/I/O, are required
before considering adoption. Unequal ownership can worsen communication;
the fitted cost proxy is not evidence of a speedup. No performance job is
submitted while radial9038626 is active, and no benchmark checkpoint is an
automatic physics continuation.

The comparison helpers `prepare_partition_pair.py`, `partition_pair_worker.py`
and `collect_partition_pair.py` preserve the shared allocation lock, explicit
bindings, queue review, storage reserves and terminal artifact verification.
The existing reviewed single-submit helper is reused. This adds no smoke gate
and does not repeat the completed physics controls.

### Evolved radial attribution established through 0.6M

Radial control9038626 completed0.6M/6400steps with native/worker/PBS success
and frozen admission. All152 artifacts are locally hash/length verified; the
951454990-byte checkpoint retains SHA256
`f1db846ab81fd3ad936fdeae4479f2e6b4b01525f204211aa85bae1e41713996`.
The dedicated radial18 native reader reproduces full/fixed report norms within
1.33e-13 scaled and leaves the checkpoint unchanged. Its positive Aurora
calibration is now verified, not merely the earlier rejection test.

At common fixed r>.3 and identical dt=.00003125, the radial12/radial18 endpoint
H norms are.0821313833/.00225673129 and M norms.5660309414/.00953642688:
reductions36.394x/59.355x. Squared integrals within .90–.95 fall by factors
5330/5567 approximately. This band holds24.82%/63.18% of the refined H²/M²,
instead of99.88%/99.85%. The remaining refined H² is distributed substantially
in .3–.6 (46.95%) and .85–.9 (27.26%); M² in .85–.9 accounts for30.39%.
The error reduction is not just a shift of the same large squared integral
into a different band. See `short-radial-controls-comparison/analysis.json`.

All9 horizon samples are accepted in each control, with identical final mass
.33333352438971814. Floor/sign totals are identical. Direct read-only comparison
of950272 nominal Cartesian cells gives maximum scaled primary-field change
4.24e-22 and compensation change1.22e-21; per-cell floor/sign arrays are
byte-identical. Appended evolved-overlap rows are explicitly excluded from
that nominal-core comparison. This is not a statement that all halo data or
future causal influence are identical.

The unsmoothed native profiles show the large outer metric and constraint
oscillations greatly reduced. Lapse/shift gradient peaks remain near30; the
underlying steep but smooth profile was not removed or reset. Together with
the negligible fourfold timestep effect, these controls demonstrate a major
early-time contribution from radial spatial discretization/interface placement
in the original transition band. They do not separate those two spatial
mechanisms completely, establish a long-time resolution order, or prove
nonlinear slicing regularity. The full10M stationary/boosted convergence goal
is still open. All three figures were visually inspected. Reproducers are
`compare_short_radial_controls.py` and `compare_radial_core.py`; native
reconstruction is already complete and must not be repeated.

Observed native-command times are1347.08s/1423.96s for time/radial controls,
including their setup and output; this is not strong scaling. Performance
job9038692 is now running, with plan SHA256
`8146783f417d65d447b1025306f634d7248f88017459490dfbdb1407e5c82be5`.
Both user debug queues were reviewed before the single submission; unrelated
held work is preserved. Its two256-step cases restart the admitted radial
endpoint, and use equal versus measured weighted ownership on the same four
nodes. Binding footprint42.377GB leaves3.623GB after2.8GB output and1.2GB atomic
reserves, within the50GB task cap. No purge or production adoption occurred.
The local watcher invokes the dedicated terminal collector once; saved
submission/state remain authoritative. Raw metadata is in
`gpu-partition-submission/`; no speedup is claimed until measurement/parity.

## Measured ownership result and next stationary family

Four-node GPU pair9038692 completed and all303 artifacts are locally hash/length verified. Both 256-step endpoints are byte-identical (SHA25663e64dd36d188be00f5d16b147ba02ef82aacbad3071337012af60a5b64d1cf8). Median stepping time was50.28009s with equal ownership and50.75285s with weighted ownership: the candidate was0.94% slower. Nominal sparse-exchange payload increased3.6%; RHS and closure time were essentially unchanged. The inspected zero-based plot and records are in `gpu-partition-comparison/`. This single ordered pair is not a strong-scaling study. **Do not adopt weighted ownership.** Frozen production remains unchanged.

The next stationary radial18 resolution family is N12/N14/N16 with dt=1/(2000N), four nodes/48ranks, to10M, using the existing frozen v7 worker. N12/N14 start fresh; N16 explicitly extends its admitted .6M radial control. No new geometry, gauge, floor, derivative or boundary equation is introduced. N12 two-step local MPI check passed requested dt1/24000 unchanged, CFL.185195, finite/admitted RF2 and matching scientific identity. Its record is `radial18-resolution-check/n12/`. The new plan writer binds each plan and uses the same terminal/continuation checks. Spatial/time resolution change together; survival is not convergence.

N12 stationary radial18 job **9038758** is submitted, plan SHA25679c7f0ffd7fcd0ef6161bcf2c448f53efbdc602efdc6793141fc34ebf1df8a83. Both user debug queues were empty; four nodes48ranks, dt1/24000, fresh native t0 to10M, storage44.282GB plus3GB output/atomic reserves under50GB. Current-task watcher collects once at terminal; no duplicate submission. N14 local two-step check also passed dt1/28000 unchanged, CFL.185205 and finite/admitted RF2; no N14 Aurora job yet.

## Static tube and boosted initial angular attribution

The optional origin-centered `smr_half_widths` input and live nominal-finest support diagnostics are implemented and locally qualified; see `SMR-TUBE.md`. The candidate uses400 leaves and expands coverage along both boost directions at unchanged finest spacing. MPI1/2 boosted two-step checkpoints are byte-identical, and all eight arrays in uninterrupted MPI2 versus MPI2-to-MPI1 four-step restart are identical. Old spherical-policy two-step checkpoints remain byte-identical; wrong restart identity and malformed inputs reject. The new candidate is not submitted and the active stationary binary is unchanged. Local wall-clock test elapsed times include host suspension and must not be used as scaling measurements.

The boosted native initial data have larger exterior errors than the stationary case. Independent Cartesian derivative refinement at66 exterior points gives H orders3.97–4.11 and physical-M orders3.95–4.00. Minimum sampled Q/Q0=.981123 and boost-map determinant=.736311 remain positive. This supports regular vacuum data at sampled points, not a uniform spacelikeness or evolved-convergence proof. Evidence/inspected figure: `boosted-initial-point-constraints/`.

A separate same-map shell-coordinate control isolates bulk truncation at48 fixed physical points using analytic stencil values. Eightfold radial-only refinement leaves max H/M approximately unchanged (ratios.998918/1.001385); angular-only refinement reduces them to.073415/.077833 of baseline before hitting radial error. Refining both directions gives H orders3.83/3.95/3.93 and M3.94/3.99/4.00. Thus puncture SMR and radial-only refinement cannot by themselves resolve the boosted outer angular structure. This control excludes interpolation, seams, scri closure, quadrature and evolution; do not claim it is a full-grid convergence result. Evidence/inspected figure: `boosted-shell-derivative-control/`.

N12 job9038758 was last observed running through.6M. The watcher then lost SSH; a bounded reconnect returned authentication denial. Renewal was requested once. Current terminal disposition is unknown, no collect or resubmission occurred, and the remote active lock must be reconciled before continuation. Local state records the outage; remote state is stale until access returns.
