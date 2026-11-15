# Full-overlap N12/N16 comparison through 30 M0

Both original frozen science runs completed normally at the last full timestep
below 30 M0: N12 at 29.9987545132 (cycle 21301), N16 at 29.9994586762
(cycle 28402). Both use unmasked full-overlap agreement, kappa=1, taper width
0.015, KO=0.1 and the original resolution-dependent timestep. The optimized
executable was not substituted into either science campaign.

At the last identical recorded time for each resolution, the full/legacy
exterior RMS ratios are:

| Resolution | t/M0 | H | Momentum | Theta | Z |
|---|---:|---:|---:|---:|---:|
| N12 | 29.750889 | 0.8154 | 0.8010 | 0.9799 | 0.9845 |
| N16 | 29.750185 | 0.1320 | 0.1155 | 0.2656 | 0.3493 |

The N16 full-overlap constraints remain approximately level at late time,
while the archived legacy constraints rise. N12 shows a smaller benefit.
Whole-domain H/M ratios are 0.8211/0.8092 at N12 and 0.1381/0.1234 at N16.
The full/legacy whole-domain diagnostic volumes are identical at each
resolution; maximum relative exterior-volume differences are 8.25e-8 (N12)
and 2.17e-8 (N16). The observed reductions therefore are not explained by
changing the diagnostic horizon exclusion. All included histories are finite.

The previous masked N12 run failed at 18.75045 M0. The masked N16 run was
stopped at user request, with its last recorded history at 17.00026 M0;
its shorter curve is not evidence of a numerical failure. Masked curves have
a different activity geometry, so use them as context rather than a strict
same-volume comparison. Removing the mask and broadening/reweighting agreement
were combined changes; these curves do not separate their causal effects.

## Apparent convergence order

For a constraint norm C, assume C(h) is proportional to h^p and use
p = ln(C12/C16) / ln(16/12). N16 values are linearly interpolated to N12
recorded times inside their shared coverage; no extrapolation is used.
Near 30 M0 (29.9987545), the full-overlap estimates are:

| Domain | H | Momentum | Theta | Z |
|---|---:|---:|---:|---:|
| Exterior | 2.641 | 2.655 | 4.363 | 3.842 |
| Whole domain | 2.560 | 2.525 | 4.362 | 3.842 |

The latest exactly coincident sample at 29.0002513 M0 gives exterior orders
2.650, 2.670, 4.257 and 3.814 without temporal interpolation. H/M remain
near 2.6 at late time. The legacy H/M estimates turn negative as its N16
constraints grow above N12. Two resolutions cannot establish asymptotic
order, solution convergence, or the truncation order of the numerical method.
Constraint norms contain initial-data, boundary, interpolation and evolution
errors; a value above four for Theta is not proof of a higher-order method.
The exterior horizon selection and discrete quadrature differ with resolution;
whole-domain estimates provide a useful additional check.

## Reproduction and provenance

Run `python3 plot.py` with NumPy and Matplotlib. The script verifies all input
history hashes in `input-manifest.json`, checks finite values and unchanged
timesteps, combines restarted segments, and writes PNG/PDF figures, CSV ratios
and `assessment.json`. Paths in the manifest identify the source archives;
the included `histories/` files are sufficient to rerun the plots locally.
Duplicated legacy N16 rows differ by at most 3.40e-11 relative; the later
numbered segment wins after a 1e-9 relative consistency check. The discrepancy
is recorded, not claimed to be bitwise equality. Full/legacy comparisons use
identical recorded cycles and times, never interpolation. Only the cross-
resolution order estimate interpolates in time.

The archived legacy final checkpoints are one timestep later than each
full-overlap endpoint; final-checkpoint spatial comparisons must disclose
that offset. Scri panels show maxima over nodes and stages since the previous
dump, so restarted segments can change accumulation windows; no exact paired
ratio is claimed for those maxima.

`N12-exterior.png`, `N16-exterior.png`, `late-time-ratios.png` and
`apparent-convergence-order.png` are the principal figures. Whole-domain
and scri figures are included as complementary checks.

## Endpoint horizon diagnostics

The two horizon trackers agree on the final common-horizon mass within each
run. Tracker 0 reports mass/M0 = 0.96167190 (full N12), 0.96169294 (legacy
N12), 0.96096607 (full N16), and 0.96092400 (legacy N16). These are endpoint
comparisons with the one-step offset described above. The full/legacy mass
differences are approximately -2.19e-5 and +4.38e-5 relative, respectively.
All saved summary rows are finite. The native `hrms` column stores a squared
quantity: following the archived surface-analysis check, take its square root
to obtain expansion RMS. Full N12/N16 give 1.0541e-4 and 3.6872e-5 in native
units. Raw values and their square roots are both provided in
`endpoint-diagnostics/assessment.json`. Mass agreement is a diagnostic
consistency check, not a proof of fully converged apparent horizons.

## Final checkpoint checks

Read-only job 9091875 completed with exit zero. All four final checkpoints
passed finite full-field reductions and the independent Theta/volume check
against native diagnostics. The native observer verified zero evolution steps
and bitwise preservation of accepted fields and stored records; source
checkpoint hashes were unchanged. `checkpoints/` includes the receipts and
native band/full-overlap histories. Run `python3 plot_checkpoints.py` to
reproduce the spatial comparison.

At N16 the full/legacy Hamiltonian and momentum ratios in r=0.25–0.32 are
0.00916 and 0.00923; in r=0.32–0.40 they are 0.00990 and 0.00961. The
shell-corner Theta RMS in r=0.32–0.40 is 0.01014 of legacy. All six
full-overlap field-group mismatch ratios are 0.00593–0.02039 at N16, using
exactly the same diagnostic observer and sample volume within that resolution.
N12 mismatch ratios span 0.01812–0.30740. These endpoint comparisons retain
the one-timestep offset and should not be described as exact-time ratios.
Farther out (r>=0.8), fixed-band constraint ratios remain approximately one;
the interior agreement treatment has not removed the scri-region errors.

Historical chi-floor/nonpositive counters are identical between legacy and
full at each resolution, with no nonfinite records. Their maximum radii are
0.100699 (N12) and 0.101565 (N16), inside the central region. These cumulative
counts include the earlier evolution and are not a count of current bad cells.

`plot_two_resolutions.py` produces the two direct N12/N16 time-history panels:
`two-resolution-constraints` (exterior RMS) and `two-resolution-boundary`
(scri maxima). The RHS panel is an amplitude, not a zero-target error.
