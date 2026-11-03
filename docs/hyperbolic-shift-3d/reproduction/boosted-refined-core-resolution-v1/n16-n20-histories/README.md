# N16/N20 boosted BC2 history comparison

Reproduce with `python3 ../plot_n16_n20.py` from this directory, or invoke that
script using its repository-relative path. This only reads preserved histories,
reports and identity records; no evolution, checkpoint admission or new native
diagnostic execution. Inputs are hash/length checked against worker records,
with source/binary/plan/job identities retained in comparison.json.

`constraints.png` / PDF overlays logarithmic-y Hamiltonian, conformal momentum
norm and radial momentum component norm, both exterior and full-domain.
Radial momentum is the coordinate-radial contraction x^i M_i/r, not a separate
regularized norm. `diagnostics.png` / PDF includes each run's moving exterior
cutoff, puncture trajectory, expansion RMS, recorded mass error with missing
values left as gaps, scri null residuals and cumulative pole/clock maxima.
The vertical line marks N20's last accepted evolution time6.2082M; its last
recorded diagnostic is6.15M. N16 continues to10M. Curves connect saved samples
only, without resampling, smoothing or extrapolating N20 to the failure stage.

N20 generally has smaller exterior errors early, but improvements are not
uniform. At the last exactly shared sample6M, exterior H is5.681% lower and
conformal momentum6.667% higher than N16. Candidate cutoffs there are.1432801
and.1400439 respectively, so these ratios are descriptive and do not measure
fixed-domain convergence. Forty-one matching samples through6M were found,
with no interpolation; exact sample matches are listed in comparison.json.

Exterior norms decrease in the last saved N20 diagnostics even though the run
then fails at a localized near-puncture metric defect. Exterior averages can
therefore miss this local failure. Full-domain H and radial momentum show
large intermittent spikes in both resolutions. No monotonic stability cure,
isolated spatial-error attribution, or convergence order is established.

The updated post expansion tolerance1e-2 is shown with the original online
1e-3 threshold. N16 has65/68 and N20 has40/42 complete saved candidates passing
the new expansion criterion. Online masses are available only for25 and26
rows, both ending3.75M. Relabeling candidate expansion quality does not recover
unsaved masses or establish the native mass-stationarity criterion; gaps remain.
The original moving-domain constraint histories are unchanged.

Both PNG figures were visually inspected for readable axes/legends, log scales,
failure marker, correct curve endpoints and preserved mass gaps. Scientific
source/launcher/evolution parameters and active N24 job were not modified.
