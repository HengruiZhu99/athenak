# Characteristic audit and a direct transition to the outer gauge

This audit does not change job9015938, its source snapshot, binary, timestep or
restart policy. Mode0 is the previous smooth donor; mode1 is IT1; mode2 is a
point-kernel proposal, not an evolution target. Its implementation lives only in
characteristics_api.cpp and the separate inner_characteristics test library.

## Symbol and verification

The26-variable comoving pseudodifferential reduction uses dchi_bar, five dg,
five A_bar, p/Omega, Theta/Omega, three Lambda, dalpha_bar, three dbeta, three dB,
and three D. Rescaling physical p and Theta by the frozen Omega makes the
geometric principal block equal to ConformalRHS with Omega=1. Derivatives of
Omega, forcing and damping are lower-order terms. Gauge entries are extracted
from the actual native IT1/donor point kernels; mode2 uses an explicit proposed
point kernel. Pole numerators are differenced before division to avoid apparent
rank loss from cancellation near scri. This is a limiting interior symbol, not
a complete boundary-system well-posedness proof or a uniform puncture estimate.

With Q=n_i g^{ij} n_j and E=n_i n_i, the speeds relative to beta are

    physical: +/- alpha_bar sqrt(chi_bar Q), multiplicity6 each
    lapse: +/- sqrt(chi_bar L Q + b(U E + V Q))
    longitudinal shift: +/- sqrt(4 mu_eff Q/3)
    transverse shift: +/- sqrt(mu_eff Q), multiplicity2 each
    zero: multiplicity6

For IT1, with inner weight s and donor w, eta, F, mu:

    L = (1-s)F + s 2 Omega alpha_bar
    U = (1-s) eta chi_bar w
    V = s Omega max(Omega^2 chi_bar,1e-5)
    b = (1-s)(.005/.02)w + s(2/Omega)
    mu_eff = (1-s)mu + s

The product of blended lapse/B coefficients creates cross terms. Blending
squared speeds directly would miss these terms and the IT1 speed bump.

Checks include the independently existing donor symbol, eigenvalues and SVD
nullspaces at repeated speeds, after power-of-two diagonal similarity balancing.
Basis condition numbers depend on this normalization and are not physical
stability constants. Generic scans are not proofs of uniform strong hyperbolicity.

## Direct-blend proposal

Use one C4 weight s=Smoothstep9((.9-r)/.7): physical Cauchy telegrapher gauge for
r<=.2, transitioning directly to the complete existing outer hyperboloidal gauge
by r=.9. Extend the OUTER shift choice mu_outer=1.5 chi_bar F inward, without the
mu=4 plateau. In that outer branch turn off the Euclidean lapse/B coupling and
B gradient forcing; B still advects and relaxes. Keep the complete donor outer
lapse/shift reference sources and restoring terms. In the symbol above this gives

    U = 0
    b = s(2/Omega)
    mu_eff = s + (1-s)1.5 chi_bar F

L and V retain their expressions. Thus all nonzero squared speeds carry the
same Q factor; their ordering no longer changes with propagation direction due
to a Euclidean-versus-physical-metric contraction. This removes the intermediate
old inner-driver regime. It does not discard the CMC sources needed by the outer
gauge. A future evolution implementation must also transition inner weak
constraint damping directly to outer damping; damping is absent from this
principal-symbol audit.

For r>=.9 the complete gauge point RHS agrees with the old outer gauge, including
at scri; the nonreference fixture's maximum difference is recorded in audit.json.
Consequently the asymptotic gauge is preserved, not merely its limiting speeds.
The limiting reference scri spectrum has the same3 incoming,16 outgoing and7
zero-coordinate-speed modes in all three cases. No claim that all modes are
outflow or that the boundary problem is proved well posed is made.

## Results

On201 reference radii,160 positive anisotropic random states/directions and528
radial/tangential/oblique directions from176 cells of the verified old N16 GD1
20M checkpoint, all889 cases per mode have real expected speeds and complete
bases. Maximum spectral error is below7e-11. An additional all-cell/all-direction
lapse-versus-longitudinal-shift separation bound checks all159200 cells of that
old snapshot: minimum (c_shift^2-c_lapse^2)/Q is.85187 for donor/IT1 and.62074 for
the candidate. This snapshot was evolved with GD1, not IT1 or the candidate.

The reference-state maximum lapse speed is1.39634 for IT1, versus1.00000 for the
direct proposal. Maximum longitudinal-shift speed is2.30940 versus1.41421.
These are frozen comoving speeds, not measured evolution throughput or a license
to increase the timestep without checking other limits.

The generic samples do NOT eliminate exact resonances. At r=.4, alpha_bar=.58,
chi_bar=1, metric diag(T^-2,T,T) with T=2.629651636224101 and tangential n=(0,1,0),
IT1 has a lapse/longitudinal-shift collision. Each colliding +/- eigenvalue has
algebraic multiplicity2 but geometric multiplicity1. Its radial squared-speed
margin remains20.70675. The old driver and direct proposal both have complete
bases on this same state; the candidate's tangential squared-speed margin is
.205624. Thus the IT1 handoff is not uniformly as robust as the old driver.

Both IT1 and the proposal retain a pure-inner resonance at r=.1, alpha_bar=.5,
chi_bar=1.353569192765124, metric=I. Lapse and longitudinal shift speeds coincide
at +/-sqrt(4/3), again with multiplicity2 but only one eigenvector. The direct
proposal removes the tested anisotropic handoff defect, not every possible bulk
defect. Negative separation by itself means reversed speed ordering, not complex
speeds or a defective eigenbasis; exact/near coincidences need the rank check.

See characteristics.png/pdf, audit.json, old-n16-all-direction-gap.json and saved
collision matrices. No candidate evolution has been run. These results favor a
matched direct-blend trial while preserving the current IT1 run as a comparison.

## Reproduction

Build target hyperbolic_shift_3d_inner_characteristics and run CTest
hyperbolic_shift_3d_inner_characteristics for the portable reference/random/
collision audit. For the additional snapshot audit, run

tst/hyperbolic_shift_3d/audit_inner_characteristics.py with --library LIBRARY,
--output DIRECTORY, --snapshot CHECKPOINT, --coordinates COORDINATES and
--snapshot-sha256 SHA256. The existing GD1 N16 checkpoint SHA256 is
6e36cb7b160be61223c7e2e5e1b6009c2bf41abefde27e6e30ea900da40d19f9.
Coordinates are regenerated by the continuous_positive_geometry target with
arguments16 3 OUTPUT. Their hash is recorded in audit.json. The standalone
scan_inner_characteristics.py accepts the same paths (--sha256 for the
checkpoint hash) and computes the all-cell/all-direction separation bound,
cross-checking the worst-direction result against the native matrix.

## Actual IT1 10M snapshot

While this audit ran, job9015938 completed10M. Its checkpoint and all evidence
were verified remotely and independently after local collection. IT1 checkpoint
SHA25637665a7936afc1e404c653a1ea8fc1051a437a31f3bc907dc5cadec76c483d9f.
The same528 sampled directions have real speeds and complete bases for all modes.
The all159200-cell worst-direction normalized gap is.7541603 for donor/IT1 and
.4395967 for the direct candidate, with no nonpositive margins. Native matrices
cross-check the analytic bound at its minimizing state/direction. These checks
are recorded in it1-10m/; they do not exclude a transient resonance at unsaved
intermediate times and do not constitute an evolution of the proposed gauge.
