# Frozen principal-symbol audit: a separate gauge defect

This is a necessary-condition investigation, not a nonlinear well-posedness
proof or a sole-cause attribution of the evolved constraint growth. It changes
no campaign equations or binaries. The sampled initial geometry is regular and
spacelike; regular geometry alone does not guarantee a suitable gauge system.

`principal_probe.cpp` evaluates the actual ConformalRHS, HyperbolicDamping,
HyperbolicGauge and auxiliary-D RHS on the native stationary initial states.
The frozen radial pseudodifferential reduction treats chi, metric, lapse, B
and shift as coordinate variables, and A, K, Theta, Lambda and D as one-
derivative variables. It retains the corresponding zeroth/first/second-
derivative blocks. Lower-order damping/restoration, KO and discrete upwinding
are not principal terms. The two algebraic metric/A trace constraints are
projected out with an orthonormal null-space basis. Differential constraints
are not imposed; a free-evolution formulation must also control their modes.
The resulting principal matrix has dimension26. Trace-tangent leakage is
below2.3e-13 in the sweep; sampled eigenvalues are real to4.5e-14.

For inner weight s, radial inverse metric Q and configured inner Gamma g,

    c_shift^2 = (4/3) [(1-s) mu + s g] Q
    c_lapse^2 = {chi [(1-s) F + 2 s Omega alpha_bar]
                 + s^2 max(Omega^2 chi, chi_guard) kappa/tau} Q

for the retained direct-outer formulation. Coefficients are multiplied after
blending, as in the actual RHS. The characteristic advection is beta.n.
Equal speeds alone do not prove a defect: the eigenspace must be checked.

With g=k² the refined root tests find:

| initial configuration | inner crossing r | blending-region crossing r |
|---|---:|---:|
| moderate6–18M | .148445053843612 | .489225977767550 |
| enlarged20–40M | .066914250995437 | .696686469812034 |

At each crossing, for both lambda=beta.n±c_shift, T=P-lambda I has nullity1,
while T² has nullity2. Singular values of T separate the single roundoff-sized
value (~1e-18–1e-17) from the next one (1.37e-4 to6.41e-3). T² has two
roundoff-sized values, with its next value2.86e-5 to5.48e-4. The stated absolute
rank threshold is1e-10. This is numerical evidence of a size-two Jordan block,
not merely a poorly conditioned generic eigenvector calculation. Unit and
quarter-amplitude coefficient extractions agree to2.67e-15. Neighboring radii
split the two eigenvalues as expected. Full matrices and singular values are
preserved in principal-audit/.

The inner crossing occurs inside the exact flat-frame spatial-gauge plateau.
It therefore cannot uniquely be blamed on compactification or stitching;
the corresponding spatial telegrapher/Gamma system needs the same audit.
The second crossing is introduced as the inner gauge joins the outer one.
Neither is automatically the origin of the late observed face-corner errors.
The independent t0 ghost experiment identifies a separate numerical mechanism.

An analysis-only Gamma=1 sweep has positive longitudinal speed gap at all
sampled radii. Simply doubling k² clears this gap for the moderate initial
case but **not** for the enlarged one. The sampled minimum constant Gamma
needed for positive gap is about1.9236 k² (moderate) and2.6307 k² (enlarged).
These are initial radial speed bounds only: do not implement them as an
unqualified stability prescription or change a frozen run. A proposed gauge
repair must pass full symbol, evolving-state and scri-limit checks, with its
physical normalization and its difference from the spatial reference explicit.

Reproduction from a fresh output namespace:

    python3 build_principal_probe.py --mode sweep
    python3 build_principal_probe.py --mode crossings --amplitude 1
    python3 build_principal_probe.py --mode crossings --amplitude .25
    VECLIB_MAXIMUM_THREADS=1 python3 analyze_principal_probe.py

Scripts resolve the repository root from their own location. Existing probe
outputs reject overwrite. The sweep includes normalized and Gamma=1 controls;
its compressed full matrices, exact crossing matrices, compiler commands and
hashes are retained. Future work should test the corresponding spatial Z4c
principal subsystem independently and design a regular, explicitly bounded
gauge control before another long enlarged-center evolution.
