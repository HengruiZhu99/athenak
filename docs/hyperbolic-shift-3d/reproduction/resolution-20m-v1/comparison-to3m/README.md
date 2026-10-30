# N16 versus N24 through 3M

`plot.py` regenerates both PNG/PDF figures, the matched-time table and the
provenance/summary JSON. It verifies each source history against recorded worker
SHA256 and byte counts. N16 uses the optimized original-step reproduction from
job8988342 (two segments); N24 uses job8990772. Both evolve logarithmic chi and
lapse to3M, with the original penalties, angular epsilon2 and dt/dx=.0444444.
Core spacing is.01125 versus.0075; dt is.0005 versus.0003333333333333333.
Both use horizon lmax12/ntheta24 and expansion tolerance.001.

The N16 restart at.45M and endpoint-rounding duplicate records are merged by
time rounded to10decimal places, retaining the latest record. The denser N16
early sampling (.03M versus N24's.15M) is shown explicitly by the markers; early
peak amplitudes cannot be compared as equally sampled maxima. The matched-time
CSV uses coincident samples within1e-8M and performs no interpolation.

At3M, full-domain H_L2 decreases from.0518183 to.0346151 (33.20%), and the radial
momentum norm decreases from.0900247 to.0752266 (16.44%). These are volume norms
over the entire numerical domain, not exterior-only norms. Exterior norms use
each run's own accepted mean horizon radius as a coordinate-sphere cutoff;
they therefore do not integrate over exactly the same domain.

The provisional final coordinate radii are.07519062 and.07502426, differing by
0.2213%. Neither final candidate is accepted: expansion RMS is.00139189 and
.00145586, respectively, above.001. Open circles identify complete provisional
candidates. Accepted areal-radius error is abs(R_AH/(2M)-1), with M=1/3; it is
also the fractional irreducible-mass error. Missing accepted quantities and
exterior norms are not filled. Local finder acceptance alone does not prove
that a candidate is the outermost apparent horizon.

Both surface and ghost null residuals remain below1.56e-15. These saved columns
are running maxima within each segment; the N16 restart resets those maxima.
Their roundoff-scale differences are not a truncation-error convergence test.

The constraint and accepted areal-radius errors improve with refinement over
the displayed samples, while the horizon residual does not decrease uniformly.
Two grids and different horizon cutoffs are insufficient to establish a reliable
convergence order. N32 and longer histories remain necessary for that assessment.
