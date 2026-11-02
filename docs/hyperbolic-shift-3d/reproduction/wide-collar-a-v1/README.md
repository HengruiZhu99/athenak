# A with a wider outer restoring collar

Authorized 2026-11-02: fresh N16 to10M on Aurora, compared with preserved A job9031760. The sole intended equation change is C(r)=S((r-.7)/.2) instead of S((r-.8)/.1), S(z)=z^8/(z^8+(1-z)^8), with exact plateaus outside [0,1]. The original C7 shape and restoring coefficient4 are retained, as is the outer plateau at r>=.9. The transition doubles in width and its maximum slope halves (80→40). Both support and center change; this is a width sensitivity test, not proof that a difference is caused solely by derivative size.

The inner gauge W remains .2–.56, original native initial slicing remains4M–8M with the C-infinity switch, KO8/originalscri, dt=.0005, direct chi/lapse, floors, damping, horizon diagnostics and12ranks are matched to A. The promising initial-slice polynomial is NOT used. Direct-outer mu is independent of C; adaptive J is disabled. Near-scri sources/closure remain those of A because C=1 there. A dedicated target/schema NW1 prevents mixing checkpoints.

The separate source/build/launcher are immutable after staging/freezing. Reuse audited native controller with one case and exact additional collar report checks, not a new scheduler. One local one-rank SERIAL two-step Minkowski/puncture reference and GPU parity precede the long run. Automatic healthy admitted continuation is authorized; no failed-state restart or parameter changes. Keep5GB cap and copy/hash before retirement. One node/debug-scaling,12MPI ranks,worker3200s/command3100s/native2920s.

Build budget: only two targets (one evolution, one diagnostic), versus ten targets in the previous builds, each measured about330MB on Aurora. Reserve400MB new build footprint plus100MB temporary headroom; the5GB cap and live capacity guard are unchanged. Evolution uses the existing output/atomic checkpoint reservations unchanged. Source packaging is included in measured footprint before build.

## Previous removal tests

There is no matched candidate-A puncture evolution with the restoring coefficient zero. B/E removed the extra C factor but retained restoration through W; they failed at9.3915M/9.3900M. Their failure is not evidence for a zero-restoration evolution. The shared-transition source unit test sets restoring=0 only to check which source terms change; it is not an evolution. An older different finite-angular pulse experiment set constraint and restoring damping to zero together and measured increased Z residual at t=.01; it does not isolate this term or establish puncture stability. See ../native-shared-transition-v1/STATUS.md and repository docs/hyperboloidal-layer-validation.md.

Comparison must retain accepted mass gaps, full/own-candidate constraints, scri null/pole/clock residuals, floor/sign counts and common-cutoff regional constraints, especially .775–.9. One N16 trial cannot demonstrate convergence. Record submitted job and state here before any further submission.

## Current handoff

Build completed and all frozen configuration/source/binary checks passed. Source6894a0e322146bf479e5aab5296fda0a0d012273, manifest SHA2562d2f514c89bc9819fe5b3f09d5f86fb9de50162e814c114bcd4e9f754bf6aa25, science SHA25636fd12f6606fb04ed6cfa8e760486daa6f483989c3b059418a1724c887ff08ec. Both actual binary hashes are in launcher/native-transition-science.json. Local one-rank SERIAL references passed both two-step cases and exact NW1 admission; generated launcher files are locally hash/length verified against final provenance.

GPU smoke9033185 completed successfully (PBS0, worker passed). Both NW1 two-step cases passed exact configuration, finite-payload and hash checks; all nine artifacts are locally hash/length verified. Physical SERIAL/GPU scaled differences are9.1874e-14(Minkowski)/1.3406e-13(puncture), compensation differences<4.45e-16 and null surface/ghost maxima<=1.111e-15. Evidence: smoke-verified/, smoke-local-verification.json and smoke-parity.json. This is short backend/partition agreement, not an evolution accuracy result. The matching terminal smoke lock was released.

Fresh A-width N16 job9033230 completed10M/6667steps with native return0, worker passed and PBS exit0. Plan campaigns/n16-wide-collar-a-v1-a-evolution-segment01/plan.json SHA2566e88807d96819f097847c9f7fbc3cb42ad3ab635433b64d40771ee987fc90ae0. The NW1 endpoint passed frozen schema/configuration/finite-payload/time/step/source/binary verification. All six worker artifacts, including checkpoint and horizon seed, are locally SHA256/length verified. Checkpoint SHA256d469401c8f6a679005b60e70d2da32805a988afe72a8c4a433f93b3759ede915,79725224bytes. The matching terminal lock was released. No continuation, replay or new evolution is required; no outputs were purged at disposition. The comparison is complete and the heartbeat is being retired.

N16 resolution qualification: the unchanged S8 profile rises from10% to90% across0.013647(original) or0.027294(wide) in compact radius. The outer nominal radial spacing is0.015625, giving0.873 or1.747cells across that rise. Neither width by itself guarantees an asymptotic regime; collar-profile.png marks actual outer cell centers.

Before GPU allocation, completed C/D endpoint copies were reverified locally and remotely, then only the two remote checkpoint copies were retired. Their original local files remain. Receipt retired-before-wide-collar.json records hashes, logical sizes and318930944allocated bytes freed.

## Completed comparison at 10M

The wider collar preserved stability over this N16 trial but did not give a uniform accuracy improvement. The outer constraint peak shifted inward as the restoring transition moved inward. The Hamiltonian constraint improved modestly over a fixed exterior region, while momentum became slightly worse. This supports sensitivity to the restoring profile; it does not establish loss of continuum regularity, underresolution as the sole cause, or an earlier asymptotic convergence regime.

The final accepted mean horizon radii are .1097382325 (A) and .1131595327 (A-wide). We therefore reconstructed both admitted endpoints with their matching native diagnostic binaries at the independent common coordinate cutoff r>.12, rather than comparing each run's own horizon cutoff or reusing .1099 as an exterior cutoff for A-wide. The identical coordinate-volume measure and original-scri diagnostic operators are used. Full-domain reconstruction agrees with each native report to roundoff; both checkpoint hashes were unchanged. Recorded cumulative cuts .1099/.12/.13 permit cutoff sensitivity checks; .1099 is below A-wide's final mean horizon radius and is only a fixed-radius sensitivity diagnostic. At .13 the qualitative conclusion remains the same.

| 10M measurement | Original A | A-wide | Wide versus A |
|---|---:|---:|---:|
| H RMS, r>.12 | .0153300963 | .0144442459 | -5.7785% |
| M RMS, r>.12 | .0289713080 | .0300451238 | +3.7065% |
| Radial M RMS, r>.12 | .0307122662 | .0319892899 | +4.1580% |
| Accepted horizon mass error | +.3413204% | +.3549687% | +.0136483 percentage points |
| Accepted horizon rows | 45/68 | 46/68 | Missing mass rows retained |

Regional numbers below are changes in the **absolute integrals of squared residuals**, not RMS changes or percentages of the domain. In .775–.9, H2/M2 fall82.5672%/44.7593%; in .9–1 they fall94.5182%/91.3384%. However, the newly affected .7–.775 band now holds42.7671%/45.3730% of the wide run's exterior H2/M2, versus1.1071%/1.9322% for A. Over the complete .7–.9 band, H2 decreases23.5627% but M2 increases17.0902%. In .16–.30, H2/M2 increase14.5824%/13.8895%. The improvement in the old outer band is therefore partly redistribution, not removal of the dominant error.

Both runs have56stage/8final negative-chi and floor events at the nearest eight puncture cells; there are no lapse, surface or boundary-interpolation sign interventions. Scri null surface/ghost maxima remain <=1.56e-15. Wide pole/clock maxima are .031458936/.023503144 versus .028167936/.022001227 for A. Algebraic null closure remains accurate but does not certify small constraint errors. The full and own-candidate exterior histories and accepted horizon gaps are preserved without interpolation.

Only one resolution was evolved, and collar support and center changed along with width. The unchanged C7 profile still spans only about1.75radial cells across its10–90% rise at N16. Thus this result neither proves high-order convergence nor rules out underresolution. A remains the baseline; A-wide is a useful sensitivity control rather than a clear replacement. The separate improved initial-slice study is not part of these evolutions.

### Evidence and reproduction

- `evolution-verified/`, `evolution-local-verification.json`: scheduler/worker/native disposition, exact admission, all six artifact hashes/lengths and terminal release.
- `comparison/comparison.json`, `histories.png`, `full-and-boundary.png`: separate A/A-wide identities, full and candidate-exterior histories, gaps, counter totals and boundary diagnostics.
- `comparison/{a,wide}-radial.json`, corresponding command/log records and `radial-comparison.json`: admitted endpoints, common-cutoff norms, regional integrals and cutoff controls.
- `comparison/local-diagnostic-source-verification.json`: local scientific source hashes checked against the frozen wide manifest; binary hashes and compile definitions are in command records. The original A binary hash matches the earlier preserved A diagnostic.
- `comparison/radial.png`: inspected radial error redistribution. All three comparison figures were visually inspected.

From the repository root, `python3 docs/hyperbolic-shift-3d/reproduction/wide-collar-a-v1/compare.py --plans campaigns/n16-wide-collar-a-v1-a-evolution-segment01/plan.json --output FRESH_DIRECTORY` reproduces histories. `python3 docs/hyperbolic-shift-3d/reproduction/wide-collar-a-v1/reconstruct_radial.py --output FRESH_DIRECTORY` records and runs matching local diagnostic commands; native output paths must be fresh (preserve existing comparison files before selecting a new output directory). `comparison/plot_radial.py` reproduces the radial summary and figure from saved native integrals. Raw full campaigns remain under output/wide-collar-a/aurora and output/native-shared-transition/aurora.
