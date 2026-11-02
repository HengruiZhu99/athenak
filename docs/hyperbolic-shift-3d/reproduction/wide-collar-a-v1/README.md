# A with a wider outer restoring collar

Authorized 2026-11-02: fresh N16 to10M on Aurora, compared with preserved A job9031760. The sole intended equation change is C(r)=S((r-.7)/.2) instead of S((r-.8)/.1), S(z)=z^8/(z^8+(1-z)^8), with exact plateaus outside [0,1]. The original C7 shape and restoring coefficient4 are retained, as is the outer plateau at r>=.9. The transition doubles in width and its maximum slope halves (80→40). Both support and center change; this is a width sensitivity test, not proof that a difference is caused solely by derivative size.

The inner gauge W remains .2–.56, original native initial slicing remains4M–8M with the C-infinity switch, KO8/originalscri, dt=.0005, direct chi/lapse, floors, damping, horizon diagnostics and12ranks are matched to A. The promising initial-slice polynomial is NOT used. Direct-outer mu is independent of C; adaptive J is disabled. Near-scri sources/closure remain those of A because C=1 there. A dedicated target/schema NW1 prevents mixing checkpoints.

The separate source/build/launcher are immutable after staging/freezing. Reuse audited native controller with one case and exact additional collar report checks, not a new scheduler. One local12-rank two-step Minkowski/puncture reference and GPU parity precede the long run. Automatic healthy admitted continuation is authorized; no failed-state restart or parameter changes. Keep5GB cap and copy/hash before retirement. One node/debug-scaling,12MPI ranks,worker3200s/command3100s/native2920s.

Build budget: only two targets (one evolution, one diagnostic), versus ten targets in the previous builds, each measured about330MB on Aurora. Reserve400MB new build footprint plus100MB temporary headroom; the5GB cap and live capacity guard are unchanged. Evolution uses the existing output/atomic checkpoint reservations unchanged. Source packaging is included in measured footprint before build.

## Previous removal tests

There is no matched candidate-A puncture evolution with the restoring coefficient zero. B/E removed the extra C factor but retained restoration through W; they failed at9.3915M/9.3900M. Their failure is not evidence for a zero-restoration evolution. The shared-transition source unit test sets restoring=0 only to check which source terms change; it is not an evolution. An older different finite-angular pulse experiment set constraint and restoring damping to zero together and measured increased Z residual at t=.01; it does not isolate this term or establish puncture stability. See ../native-shared-transition-v1/STATUS.md and ../../../hyperboloidal-layer-validation.md (repository docs/hyperboloidal-layer-validation.md).

Comparison must retain accepted mass gaps, full/own-candidate constraints, scri null/pole/clock residuals, floor/sign counts and common-cutoff regional constraints, especially .775–.9. One N16 trial cannot demonstrate convergence. Record submitted job and state here before any further submission.
