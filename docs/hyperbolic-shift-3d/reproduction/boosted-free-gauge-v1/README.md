# Boosted hole with freely specified initial hyperboloidal gauge

Branch `codex/boosted-free-initial-gauge`. This implements the user's correction:
use v=+0.1 in the usual asymptotically flat Cartesian boost chart, but choose the
initial lapse and shift freely, as in ordinary puncture evolutions. Preserve the
exact boosted geometry and latest six-shell evolution formulation. BX1's imposed
initial boost-drift shift is superseded, not silently relabeled as this experiment.
The unchanged exact embedding, induced metric/extrinsic curvature and regular
outer boost chart are derived in [the geometric construction](../boosted-puncture-v1/README.md).
The new target has `HYPERBOLIC_BOOST_FREE_INITIAL_GAUGE=1`, schema `HYP3D_MPI_BG1`,
and a separate immutable source/build/launcher. No GPU smoke gate.

## Check against the main-branch implementation

`src/pgen/tests/z4c_boosted_puncture.cpp` is byte-identical to origin/main at
c8d5c33e39089f005e416bf839652f6796fa6e82, SHA256
a990c524453970809abb5e2ebb2fa9b21ea4b479185d497eb326c3caf2466f5d.
It sets exact boosted Schwarzschild geometry (not Bowen-York data), and initially
sets the exact boosted-coordinate shift. It then calls `GaugePreCollapsedLapse`,
whose prescription is alpha=(psi4)^(-1/2)=det(gamma)^(-1/6). Zero initial shift
is an independent allowed gauge choice; it is not what that particular pgen sets.

For the inner lab Cartesian coordinates X, rest coordinates Y, boost Gamma and
rest isotropic radius rho:

    Gamma=1/sqrt(1-v²), Yx=Gamma Xx, Yy=Xy, Yz=Xz on T_lab=0,
    rho=|Y|, psi=1+M/(2rho), a=(1-M/(2rho))/psi,
    B0²=Gamma²(1-v² a²/psi⁴),
    gamma_AF=psi⁴ diag(B0²,1,1),
    alpha_pre=psi^(-2) B0^(-1/3).

The last factor must not be dropped for a boosted metric: the conformal factor
used by Z4c comes from the full determinant. The independent audit transcribes
the main-branch extrinsic curvature expressions and compares them with the exact
hyperboloidal embedding restricted to its inner Lorentz plateau. After undoing
the compact-coordinate Jacobian, metric and K agree to7.2e-14 and3.4e-16;
the new physical lapse agrees to2.8e-16. See af-gauge-verification.json.

## Free hyperboloidal lapse and shift

Let bar-alpha_0 and chi_0 be the native unboosted six-shell C6 reference data at
the same compact point; chi is the actual boosted conformal factor. Set

    bar-alpha = bar-alpha_0 sqrt(chi/chi_0),
    beta^i = -W(r) x^i,
    W(r)=I_z(5,5), z=(r-.2)/.36, with exact0/1 plateaus.

This W is the same smoothstep9 transition used for the retained inner/outer gauge
blend. Thus beta=0 for r<=.2 and beta=-x for r>=.56. No localized -v shift is
added. Auxiliary D and the other free auxiliary initial fields remain zero.
Metric, extrinsic curvature, Theta and geometric Lambda are unchanged bit-for-bit
from BX1 at the same points; only lapse and shift change.

Since x=kX inside, chi_0=k²/(Omega² psi_lab⁴), while
chi=k²/(Omega² psi⁴ B0^(2/3)). Therefore the lapse formula gives exactly
bar-alpha/Omega=psi^(-2) B0^(-1/3) in the inner boosted-isotropic plateau. It is
the main-branch pre-collapsed lapse there. At v=0 the lapse is precisely the
existing unboosted hyperboloidal pre-collapsed lapse. At scri both chi/chi_0 and
bar-alpha_0 tend to1; beta=-x and the existing null/dependent-D closure apply.
Multiplying a compact-coordinate determinant by Omega without the reference
normalization would not reproduce the Cartesian pre-collapsed prescription.
This is a free gauge prescription with the correct limits, not the exact
stationary lapse of the boosted four-coordinate transformation or a stability proof.

## Velocity conversion and target time

The geometric boost parameter v=.1 is defined in the usual lab Cartesian chart:
X_p=v T_lab. For x=f(X) and hyperboloidal time t=T_lab-h(X), the coordinate
conversion is

    dx^i/dt = (partial x^i/partial X^j V^j)/(1-partial_j h V^j).

On the inner plateau h=0 and x=kX, k=.306142283363079. Hence the reference-chart
velocity is kv=.0306142283363, not.1 and not Gamma*v. With M=1/3 this gives
cube-half-width.18 exit time17.638857M (conservative.16 support15.678984M).
The requested minimum with10M is10M, coordinate time10/3.

After resetting the evolution gauge, the tracker obeys dx_p/dt=-beta(x_p), so
its actual initial velocity is ZERO. The reference Lorentz-chart velocity is
not an imposed tracker velocity and does not predict the later live-gauge
trajectory. One cannot simultaneously prescribe zero shift at the puncture and
force that tracker to start with kv. The geometry still has the requested boost.
Retain the RK4 tracker, moving-center fast-flow finder and conservative core
support guard; record the measured trajectory and distinguish it from the
reference exit estimate. The invariant horizon mass reference is rest M, not Gamma M.

## Execution and preservation

N16 first to10M, dt.0005; fresh N20 dt.0004 then N24 dt1/3000 only after successful
N16. Preserve six shells, broad.6-.9 restoring collar coefficient4, C6 areal4M-12M
slice, KO8/originalscri, W.2-.56, damping, auxiliary-D equation, angular2/scri1,
matched clipped penalty scale.651041666666, floors and pre-floor accounting,
horizon12/24/tol.001,diag.05. The positive-chi floor patch stays origin-centered
as before; finite nonpositive rescue remains global. No parameter changes/replays
on numerical failure. Healthy exact same-formulation walltime checkpoints may
continue automatically after full identity/finitepayload/progress verification.
BG1's208-byte header includes(v,tracker x/y/z,support.16); never load BX1.

Use one Aurora debug-scaling node/12GPU tiles, CompactBinaryMerger,10GB cap
including atomic reserves, make-j64 build. Inspect saved state and ALL user jobs
before a single submission. Preserve source/build/launcher snapshots; local
hash/length preservation plus remote rehash is required before any output retirement.
Hourly quiet monitor must follow this revised gauge; do not advance BX1 higher
resolutions. Preserve all disposition records and previous unboosted baselines.

Current status: independent main-branch/gauge comparison and bounded N16 production
checks passed. Two SERIAL steps produced an admitted finite BG1 checkpoint; horizon
mass .333334653727, null residual <=1.12e-15, tracker displacement -8.334e-12 in x.
This is not a GPU or long stability result. BX1 job9034826 was cancelled while queued,
before any evolution; terminal scheduler verified and matching lock released.
Aurora build completed with make-j64 and frozen SYCL/MPI/PVC target checks.
All launcher/source-manifest files are locally hash/length verified in aurora-build/.
Fresh BG1 N16 job **9034868** is submitted to10M on one debug-scaling node/12ranks.
Plan SHA256 `072847e1eb5d7e1b190b51c78963c04c08781d8b5a0302e5a5cfb594c5a403ee`.
Scientific source `62cca6e4f1e7e453c0fc443efa8ec7d4e044085a`; science SHA256
`815010d107d95d860375f32c48deaeee4eff65fb339a4e7fe8f8db1451c7a5db`.
Hourly monitor follows this job; N20 then N24 remain gated on N16 success.
No long boosted result yet, no repeat of build/freeze/smoke.
