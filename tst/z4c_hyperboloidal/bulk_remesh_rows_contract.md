# Bulk remesh row accuracy gate

This geometry-only test constructs actual native N12/N24/N48 trees and donor
selectors, without calling `AddCoordinatesAndPhysics` or allocating evolution
fields. The old/new trees contain 106/127 leaves and differ by one corner core,
one first-shell, and one outer-layer parent. It probes their actual dyadic
parent/child maps in both directions. Nominal core points are excluded: their
ordinary native cell-centered transfer is covered by the lifecycle oracle.

Before running, the polynomial acceptance bound is fixed at
`4096*epsilon*max(1,sum(abs(weight*polynomial)),abs(expected))`. This conservative
roundoff allowance covers coordinate conversion, six-point coefficient
construction, and at most 216 tensor terms. Independent analytic chart
monomials include every one-axis degree through five, a mixed (1,2,3) term, and
the tensor (5,5,5) term. Constants are checked separately for every row.

An independent smooth `exp(2*u+3*v+4*w)` chart field must approach sixth-order
interpolation accuracy as N doubles. The predeclared finite-resolution lower
bound is observed order 4.5, with an exception only once relative error falls
below `4096*epsilon`. Maxima include endpoint extrapolation and one-sided rows,
so target locations approach chart boundaries as N increases.

Coarse overlap beyond the old fine family's evolved extent is explicitly
identified by `used_global`. Chart-polynomial gates do not apply there: each
chart carries independently evolved values. A chart-label field must exhibit
that distinction on at least one first-shell fallback point at each N.
Constants remain checked, and the error for the globally smooth physical
field `exp(40*x+30*y+20*z)` is recorded separately.

The initial archived run in `build-native/bulk-remesh-rows` recorded that
physical-fallback error without gating it. The subsequent test extension
explicitly applies the same general observed-order threshold 4.5 to both
doublings for core and first-shell fallback points. It also checks every such
row against an independent exact remainder identity for the globally smooth
physical field `F=10^6*(x^2+y^2+z^2)^3`. For one Cartesian tensor leaf,
`interpolated-exact = -10^6*sum_axis product(target_axis-six_axis_nodes)`;
all mixed terms have degree at most five in each axis and reproduce exactly.
For spherical donor rows, angular factors are constant and the identity is
`-10^6*product(target_radius-six_radial_nodes)`. An exactly sampled axis or
radius has zero remainder. The identity residual must obey the same fixed
4096-epsilon coefficient-norm bound. This additional mathematical prediction
validates finite-N behavior without estimating a tolerance from observed errors.
The original ungated report and its source are preserved separately.

Eight additional first-shell targets straddle all three internal refined-child
interfaces at quarter-cell offsets. Their radius is near 0.12 and their
directions are near the +X axis, placing them outside the Cartesian cube while
remaining inside the first shell. Every returned row must use one first-shell
leaf and require a one-sided stencil; the same polynomial and convergence
gates apply. This exercises the new fallback after the preserved centered
overlap selector fails to find a legal stencil.

Passing this test establishes interpolation behavior at these sampled points.
The full native lifecycle, poisoned storage, conservative histories, coupled
regrids, restart, and fixed N12 gates remain separate requirements.
