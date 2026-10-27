# Puncture position and chart transport

`PuncturePosition` owns only Cartesian position, coordinate velocity, time,
RK-stage identity and a configuration revision. It contains no regularity
exponent and supplies no parity, lapse/metric floor, excision radius or shift
boundary condition. The instantaneous elliptic operator remains an operator on
the physical Cartesian shift with its full evolved-metric coefficients.

The same position object can supply relative coordinates and radial factors to
coefficient/source evaluation, neighborhood diagnostics, AMR selection and
halo-plan construction. A neighborhood selection is metadata, not permission
to exclude its cells from residual or scientific acceptance norms.

## Tracker adaptation

The existing `CompactObjectTracker` exposes `GetPos(int)` and `SetPos(Real*)`.
Its velocity is private. Its black-hole `InterpolateVelocity` routine uses
`velocity=-beta`, but its position update is final-stage forward Euler. That
update must not be used as the motion of a stage-dependent coefficient chart.

`PuncturePosition::FromTracker` is a thin templated adapter to those actual
public position methods. It accepts the shift **already evaluated from the
converged composite stage solve**, and sets coordinate velocity to its negative.
It does not interpolate native Cauchy fields, run the old tracker integrator or
write to the tracker. `PublishAccepted` is a separate explicit action and
rejects intermediate RK stages. A future native owner can publish its accepted
stage-integrated position for existing tracking output and AMR consumers.

Changing position, velocity, time or stage through `SetKinematics` increments
the revision. `Capture`/`Matches` lets cached chart/halo/source data detect stale
kinematics. Because the public arrays also support simple manufactured data,
the stamp compares their values as well as the revision; a direct array edit
cannot silently retain a matching cache stamp. A source coefficient or chart
revision is separate from the geometry/solver hierarchy revision and should be
included in the owning stage's invalidation rules.

## Spatial and time derivatives

With `y=x-X(t)`, `rho²=y.y` and `v=Xdot`, at fixed Cartesian x,

    partial_i rho² = 2 y_i,     partial_ij rho² = 2 delta_ij,
    partial_t rho² = -2 y.v.

For a caller-selected fixed factor `f=rho^p`, at positive rho,

    partial_i f = p rho^(p-1) n_i,
    partial_ij f = p rho^(p-2) [delta_ij+(p-2)n_i n_j],
    partial_t f = -v^i partial_i f.

The implementation evaluates these with normalized directions to avoid
unnecessary intermediate inverse powers. It rejects nonfinite or
noninvertible numerical factors; it never inserts an epsilon radius.
`SquaredRadiusJet` is well-defined even at the coordinate endpoint, while the
general `RadialPowerJet` and division-based chart helper require explicit
endpoint data at rho=0. This distinction does not treat that point as a regular
physical center or select an endpoint asymptotic class.

For an evolved coefficient `F=f u`, the fixed-grid rate must satisfy

    u_t = [F_t - f_t u]/f.

`FixedChartRate` applies exactly that conversion away from the endpoint. It
does not choose p or infer that p is the solution's evolving exponent. If p
were changed in time, an additional `p_t log(rho)` term would be required; that
different adaptive chart is outside this fixed-factor helper.

The derivative at fixed moving coordinates is

    (partial_t F)_y = (partial_t F)_x + Xdot.grad(F).

Since the formulation transports fields with `+beta.grad`, `Xdot=-beta(X)`
cancels a constant translational shift locally. This uses a nonzero beta; it
does not pin beta at the puncture. These time terms belong to evolved
coordinate-dependent charts. An instantaneous spatial elliptic solve at a
fixed stage requires their spatial derivatives, not an additional time source.

## Scope and checks

`test_puncture_position.cpp` checks arbitrary translated coordinates, nonzero
constant shift transport, first/second spatial derivatives and moving factor
time derivatives for several fixed integer and noninteger powers. It also
checks revision invalidation, explicit endpoint rejection for divided charts,
nonfinite input rejection and accepted-versus-stage tracker publication. The
test has been prepared for the parent's bounded build; no result is claimed
before that execution.

This abstraction does not make the selected current gauge valid for a
translated wormhole. Its collapsed-inner lapse source uses global `|x|²` and
has the translated-end incompatibility derived in
`angular-af-endpoint-audit.md`. Numerical chart factors can use a moving center
only with their complete derivative/rate conversion; the continuum gauge must
not be silently translated or modified to obtain an apparent moving-puncture
success. Constant/linear and translated fractional elliptic manufactured tests
are compatibility checks, not moving-black-hole evolution evidence.

## Chart and neighborhood integration

`WormholeChart(x,puncture)` now delegates explicitly to the existing generic
position/velocity constructor after validating the kinematics. Its global
Omega, reference lapse and time/spatial derivative expressions are unchanged.
The centered `Isotropic`/`Asymptotic` factories and selected lapse gauge are
unchanged. The position test also checks this actual overload's three factor
time derivatives by finite differences of a translated center.

`PunctureNeighborhood` builds active-point and halo-target metadata from the
same position object. Position, velocity, time, stage or revision changes
invalidate it; mesh revision and active/halo counts are also checked. Queries
refresh before returning selections. Neither the active set nor the grid's
halo plans or field values are modified. A regrid that destroys the referenced
grid requires the owner to reconstruct the neighborhood object as well.

Its optional instantaneous algebraic scale is

    s(x)=1-(1/2)(1-rho²/R²)^4   for rho<R,
    s(x)=1                      otherwise.

It has explicit bounds `1/2<=s<=1`, inverse bound `1/s<=2`, and returns one
value for **every active cell**. It remains one at and near scri because the
requested support must lie strictly inside the outer sphere. A coincident
puncture/cell gets the finite value1/2 and is not discarded. This scale contains
no singular powers, empirical exponent or radius floor. The generator is a
component for bounded equilibration; it does not itself change the physical
elliptic problem or claim a conditioning improvement.

The physical Cartesian beta remains the correctness reference. Any use of
this scale must conjugate/reconstruct the discrete operator consistently and
check the original full-metric residual. An instantaneous algebraic scale has
no separate time source. An evolved coordinate-dependent chart still needs
the transport conversion above. Position affects the elliptic operator only
through its actual geometry/source inputs or a consistent algebraic scaling;
the full metric operator has no special puncture boundary datum or beta pin.
