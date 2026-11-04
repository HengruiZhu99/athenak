# Coherent stationary outer reference gauge (experimental)

This opt-in control follows INITIAL-GAUGE-ATTRIBUTION.md. Both original enlarged
stationary controls failed at.463875M despite changing the inner driver policy.
No stable or convergent evolution of this new option has yet been established.
The physical geometry, initial map and configured Omega are unchanged.

## Equations

Let S(R) be the existing C6 native slicing profile and

    a(R) = sqrt(1 - 2M/R + R² S²),
    alpha_* = Omega a, beta_*^i = -S x^i,
    K_* = -3S - R S_R, B_* = D_* = 0.

This is the stationary Schwarzschild Killing gauge only outside the central
region. The original precollapsed lapse and zero initial shift are retained
exactly for r<=Omega_flat_radius. Outside that radius, the lapse blends to
alpha_* with the existing C4 Smoothstep9, ending at gauge_begin. Initial shift
is -Sx there; S=0 in the affine inner-map region, so its switch preserves the
same zero shift through the central zone. The reference lapse is used only
outside the horizon and remains positive. This is not a sign-changing Killing
lapse imposed through the puncture.

Writing G_alpha and G_beta for the original *outer* gauge RHS, the new outer
RHS is

    G_beta_new(U) = G_beta(U) - G_beta(U_*),
    G_alpha_new(U) = G_alpha(U) - G_alpha(U_*)
                    + 2 xi (alpha-alpha_*) (alpha_*-A)/Omega,
    A = (1+r²)/2.

The last term turns the original lapse restoring factor into
-2 xi alpha(alpha-alpha_*)/Omega. At alpha=alpha_*>0, its derivative is negative,
-2 xi alpha_*/Omega. Simply subtracting the reference RHS without this term
would leave the possibly positive derivative -2xi(2alpha_*-A)/Omega.
This sign check is not a nonlinear stability or well-posedness proof.

The original complete-RHS weight multiplies the *entire* correction, so the
inner telegrapher equations and their principal symbol are unchanged. All new
sources are prescribed positional functions, affine in evolved lapse, and do
not change principal derivatives. The actual geometry, Lambda, auxiliary D,
constraint damping, lapse/chi floors, RK, transport and dissipation are evolved
normally. No field or constraint is reset to a reference solution. B_* and
D_* are zero; their existing equations vanish for the stationary geometric
reference outside the gauge blend, without a new auxiliary source.

## Scri and frame consistency

For S=1, beta_*=-x exactly, so the new shift source vanishes identically in an
open neighborhood of scri. Thus the old restoring-limit contribution to the
dependent D boundary condition is retained unchanged. The geometric sampled
normal derivatives and angular dissipation also remain unchanged.

There alpha_*-A=O(Omega³). The lapse-dependent correction is O(Omega²) for
bounded perturbations, while the constant reference subtraction vanishes at
least quadratically (the sampled stationary tail is fourth order until
roundoff). Both value and first normal limiting derivative are zero. The
existing unit lapse, null projection, dependent D and clock diagnostics at
scri therefore retain their limiting equations. The sources are evaluated as
zero at the exact boundary, not by dividing small numbers there. The measured
asymptotic control preserves its roundoff floor instead of declaring an order
from the smallest spacings.

Initial physical metric and extrinsic curvature are bit-identical to the
original configured data, both stationary and boosted, on the tested samples.
Only initial lapse/shift and outer gauge sources change. The same Omega is
used by initialization, evolution and source construction. The map r(R) is
still the native physical map, not r/Omega. The boosted geometry is not
replaced by a stationary reference; a future boosted test uses the same free
gauge construction on that exact boosted geometry.

## Implementation and performance scope

Set outer_gauge=stationary-reference in a separately bound configuration.
Its checkpoint identity adds outer_gauge=stationary-reference-v1 and the
initial-lapse blend policy; original identities are unchanged. No old
checkpoint can silently restart this formulation.

Native inversion/quadrature is evaluated once during setup, with existing
exact-radius caching. The device stores five prescribed scalars per owned
point (40 bytes/cell); no extra RK kernel, MPI call or quadrature is introduced.
The CPU reference table is created lazily only when the CPU RHS is requested,
avoiding a replicated global table during native distributed device evolution.
These are structural cost controls, not a measured speedup claim.

The point oracle compares source balance using independently differenced
reference lapse/shift, tests arbitrary nonzero gauge perturbations, exact
inner zero correction, unchanged stationary/boosted geometry, positive initial
lapse and the asymptotic source limits. The existing full CPU/device test
covers all fields and nontrivial derivatives, KO on/off, adaptive-source on/off
and partitioned/empty owned ranges. Local backend is Serial; this is not an
Aurora GPU parity assertion. No GPU smoke gate is being added.

Fresh stationary N16 is the next evolution control, on the matched12-shell,
three-level SMR grid, dt.000125, four nodes48ranks, target10M. There is no restart
from either failed control. Stationary regularity and resolution evidence must
precede a boosted stability/convergence claim.
