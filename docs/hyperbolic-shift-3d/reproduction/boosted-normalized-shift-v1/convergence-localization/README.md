# BN1 N16/N20 convergence localization at 10M

These are read-only analyses of the admitted BN1 endpoints. No campaign source,
binary, evolution parameter, or checkpoint was changed. The regional analysis
reuses completed matching-operator radial diagnostics. The additional slice
reader is the preserved BC2 analysis-only `endpoint_slices.cpp`, compiled with
all BN1 diagnostic definitions (including normalized shift), not the BC2 binary.
Its CPU constraint integrals reproduce production native volume sums to zero
(N16) and 2.04e-16 relative (N20); both checkpoint hashes remain unchanged.
Commands, flags, source/binary/input/output hashes are in slice-provenance.json.

Reproduce with ../isolate_convergence.py, ../extract_localization.py, and
../plot_localization_slices.py. The extractor reuses existing successful output
and refuses identity inconsistencies; it performs no time evolution. PNG/PDF
figures were visually inspected. No new remote build or run was launched.

## What is isolated

At the common coordinate cutoff r>.3, outside all recorded complete candidate
horizon enclosures, N16→N20 Hamiltonian RMS increases 11.188% and momentum RMS
2.677%. The deterioration therefore is not solely a moving-cutoff artifact or
an integral over the puncture singularity. These coordinate domains are not
invariant physical domains, and no accepted final horizon/mass is inferred.

The region .3<r<.4 contains 95.740% of N20's exterior H-squared integral and
90.336% of its M-squared integral. Its integrals increase 36.215% and 37.933%.
In contrast, .6<r<.775 decreases 68.410% and 69.800%, and .775<r<.9 decreases
64.852% and 67.579%. For r>.4 the effective RMS residual orders are +2.050/+2.491;
for r>.6 they are +2.569/+2.670. These are two-point residual slopes with space
and time refined together, not field self-convergence or asymptotic order.
Changing the displayed cutoff is localization, not a cure for the poor inner
exterior behavior. The small contribution from r>.9 does not prove that scri
errors cannot propagate inward or that the separate pole compatibility issue
is harmless.

The increasing region intersects the first radial-shell interface r=.305,
inside the inner-gauge blend .2–.56. It is not the restoring collar .6–.9 or
the Cartesian/spherical coupling band .16–.18. The latter may still produce
outgoing disturbances; endpoint localization does not identify their origin.

## Native unsmoothed slices and overlap evidence

Plots use only nominal shell cells; evolved extra cells (zero diagnostic
quadrature weight) are retained separately for the overlap audit. The grid's
DiagnosticVolume in composite_difference.hpp excludes extra cells, so no
extra-cell double counting was introduced into regional norms. Rays sample
+x face center, +x near a corner, and -x face center; a Cartesian x-line is also
saved. Angular directions differ slightly across resolutions. No interpolation
or smoothing is applied and these are not field-difference convergence norms.

Chi, lapse, and radial shift-deviation derivative profiles nearly overlay;
constraint profiles contain oscillations over a few radial cells, particularly
near the first interface. The +x near-corner first nominal point outside r=.305
has H=-.59536 at N16 and -1.24325 at N20. Within EACH resolution, the independently
evolved layer-0 extra copy and layer-1 nominal copy occupy the same physical
coordinate (mismatch <=1e-12). At that point:

| N | r | absolute metric_rr copy difference | absolute H copy difference |
|---|---|---|---|
|16|.30953125|.000647942|.601349|
|20|.308625|.000567007|1.004765|

Small primary-field mismatch coupled to much larger derivative-based residual
mismatch makes the shell-interface/overlap derivative treatment a concrete
suspect. This is not proof of an interface bug: coordinates differ between
resolutions, nonlinear dynamics can feed the mismatch, and both copies use
finite stencils with different neighborhoods. A full truncation/RHS budget and
controlled intervention are still needed. It is too strong to conclude that a
smooth initial slice or simple broadening of the gauge transition fixes it.

The first-segment history already has larger N20 own-candidate H/M around
5.55–6M, before the N20 restart at9.3174M. Thus restart is not necessary for that
history deterioration. These historical cuts differ and no missing fixed-cutoff
history is fabricated. Restart/finder effects on the final state are not ruled out.

## Next discriminating controls

1. At the first shell interface, separate analytic/manufactured truncation error
   from evolved-copy mismatch using the actual non-spherical mapped geometry;
   inspect first/second derivatives and the geometric RHS on both copies.
2. On preserved endpoints, compare derivative evaluation with a common donor
   representation across this interface, without changing the fields, to test
   how much of the spike is diagnostic sensitivity. This is analysis-only and
   cannot establish evolution stability.
3. Only then select a single-change evolution control (interface coupling or
   radial redistribution) with gauge and time-step held fixed. Moving the gauge
   transition and changing the shell layout together would obscure attribution.

N24 job9037899 remains the independently running frozen BN1 case; this analysis
neither restarts nor changes it. Its endpoint will distinguish persistence of
this pattern from a two-grid fluctuation. Current evidence localizes the defect
but does not identify a proven root cause or establish convergence.

Latest live scheduler check during this audit: job9037899 running on four nodes;
last history3.60M/7200steps, PBS walltime30min08s, online horizon accepted with
expansionRMS1.052e-6. This is a live observation, not terminal verification.
