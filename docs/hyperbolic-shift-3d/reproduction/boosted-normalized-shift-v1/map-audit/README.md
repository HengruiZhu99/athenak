# Spatial normalization versus nonlinear compactification

Reproduce from repository root with `c++ -O2 -std=c++17 -Isrc
 docs/hyperbolic-shift-3d/reproduction/boosted-normalized-shift-v1/map-audit/map.cpp
 -o /tmp/bn1-map-audit` then run that executable. This reads the native base map;
no evolution, campaign parameter, frozen source or checkpoint is modified.

The lab hybrid radius is sigma(rho)=rho+S(R)(R-rho), with
X=sigma(rho(r)) x/r. For J=dx/dX, tangential eigenvalue r/sigma and radial
eigenvalue1/(d sigma/dr). The map is exactly x=kX through r=.29738799806418;
the full inner driver has weight1 only throughr=.2, then blends to zero at.56.
The initial boost embedding is separate; this is the lab spatial map used by
that embedding, not the full nonlinear map from the rest-frame coordinates.

At r=.4 the radial scale differs from k by.246%, while inner weight=.3655.
At r=.5 the radial scale differs by5.89%, but inner weight=.00895. The tangential
scale is closer to k. Thus the driver fades while the map remains nearly affine,
but these numbers do not bound derivative-of-J or full gauge transformation errors.

Under constant x=kX and unchanged time, determinant-one conformal metric slots
are unchanged, Lambda_x=Lambda_X/k, beta_x=k beta_X, giving coefficient k².
For native chi, chi_native=k² chi_AF/Omega²: Omega cancels from gtilde, so no
additional Omega² factor belongs in this constant-map shift normalization.
This is exact for general fields in that chart, not just spherical initial data.

The nonlinear transition is not the exact pullback of the reference Gamma driver:
Jacobian derivatives and conformal connection transformation terms are absent
from the simple constant-coefficient blended prescription. It remains a chosen
coordinate gauge, not a claim of global AF gauge equivalence or well-posedness.
The existing outer gauge and scri closure are unchanged. Refinement alone would
not make two different continuum gauge prescriptions identical in the blend.
Other lapse/chi guards and finite-difference/KO differences remain independent.
