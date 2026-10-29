# Geometry-only CFL comparison

The standalone probe uses the native ShiftCoefficients and the same summed
characteristic-frequency formula as HyperbolicCompositeDevice, on exact
hyperboloidal Minkowski data. n16, core half-width .125, outer radii .45,.75,1.
Its V7/V9 controls reproduce the recorded native initial CFL values.

Kind0 is V7 without evolved overlap. Kind1 is current V9 (three extra evolved
points in every direction on core and transition blocks). Kinds2/3 replace
the transition map by a pure spherical annulus, retain three extra evolved
Cartesian cells, three angular overlap cells, and outward radial overlap, but
use geometric overlap with the core instead of inward radial extension.
Kind2 retains uniform face-plane ratios; kind3 uses uniform angles in
[-pi/4,pi/4] with ratios tan(angle). The inverse-Jacobian angular rows are
transformed by4/[pi*(1+ratio^2)].

At inner radius .125, kind2 reduces maximum CFL only from .271529 to .267864
at dt=.00025. Kind3 reduces it to .158173, giving a CFL=.4 timestep ceiling
of .00063222 versus .00036828 for V9. Moving the inner sphere to .10 or .075
with the original angular coordinates worsens CFL.

This probe evaluates geometry and initial characteristic speeds only. It does
not construct donor plans, establish centered interpolation coverage, define
unique volume quadrature, update scri angular operators, or evolve fields.
Equal node counts do not imply identical physical resolution between maps.
The modified descriptor is used only for Forward/Geometry, never ownership.
These results are not an admission of a spherical-shell evolution scheme.

Reference: Pollney et al., arXiv0910.3803, equations1–3 and sectionII: inflated-cube
angular coordinates are arctan ratios and the radial coordinate is a function
of r, with interpolation between overlapping Cartesian and exterior patches.
