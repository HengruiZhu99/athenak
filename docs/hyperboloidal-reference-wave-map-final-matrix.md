# Final native t2 matrix and independent flat-wave values

The original nine-process matrix is finished, with six failed processes and
three completed controls. The fixed wave-map acceptance comparisons cannot be
evaluated at t=2 because their required runs failed. No t6/t12 extension or
production adoption is admitted.

| Original case | Outcome | Reported abort time or completed target |
|---|---|---:|
| wave-map N16, large | failed | 1.3671453700579306 |
| wave-map N24, large | failed | 0.79277343749968521 |
| wave-map N32, large | failed | 1.5837303635638276 |
| C0 N16, large | failed | 1.9994906249988413 |
| C0 N24, large | completed | 2 |
| C0 N32, large | completed | 2 |
| wave-map N24, half timestep | failed | 0.79280598958281545 |
| wave-map N24, small | failed | 1.0180338541661471 |
| wave-map N24, stationary reference | completed | 2 |

Every completed control passed the unchanged analyzer on all 81 saved restart
states, including positivity/SPD, algebraic and native diagnostic gates. The
stationary reference's maximum saved field deviation is 1.1178731551542143e-13;
its maximum saved H/M/Z/Theta RMS is
7.015569536222238e-14 / 5.943916799832328e-14 /
1.415984624081873e-15 / 1.0728538605414713e-15.
The disturbed C0 controls have substantial growth:

| Control | Near-t1 sample | H/M/Z/Theta RMS at t=2 |
|---|---:|---|
| C0 N24 | 0.9999999999994967 | 2.7187330 / 2.8015061 / 0.65525886 / 0.060262799 |
| C0 N32 | 1.000021289062511 | 1.6758281 / 1.6469865 / 0.40504335 / 0.028154168 |

Completion is not stability acceptance. The bookkeeping retains the exact
near-t1 sample, distinguishes endpoint growth from saved-window peaks, and
does not interpolate or substitute partial histories for failed t2 states.
Its three required candidate comparisons are explicitly `not_evaluable`.

The separate N32 partial check covers 64 saved states ending at
t=1.5750509765619771. All saved fields passed the finite/positive/SPD/algebraic
gates; an independent parser matched all 1,600 field-extrema pairs exactly.
The last saved native H/M/Z/Theta RMS is
0.3591774657100985 / 2.4505096871160506 /
0.5273032780725571 / 0.03650777949200517.
The r>=.9 shell contains 0.9972562580057317 of summed squared native Z.
These are saved-state observations, not reconstruction of the unsaved abort
or evidence identifying its cause. The N32 process subsequently reports a
negative lapse at t=1.5837303635638276. Across N16/24/32, the closest active
Omega is nonmonotone, so the failure times do not define a convergence order.

## Independent exact-flat coordinate-wave control

A separate multiprecision Kirchhoff integral solves the physical Minkowski
coordinate-wave initial-value problem for the actual .2/.1, width .35 angular
pulse on the complete layered initial surface. It uses no native kernel,
finite angular-mode truncation or numerical PDE outer boundary. The fixed
one-shot scalar-values gate passed all 282 checks at 80/110 decimal digits:
96 coarea rows, 56 initial-data rows, 72 exact-control rows and 18 ray rows.
The full/axis refinement errors are below 6.06e-40 for the physical scalar u
and 3.61e-38 for phi=u/Omega. Independent saved-data readback also passed.

This result checks scalar integral values and initial data. It supplies no
analytic derivative, Jacobian, inverse-map, target-slice coverage, injectivity
or caustic verdict. A reference event at tau_reference is not the corresponding
native event until the full four-dimensional inverse map is solved. Neither
positive scalar values nor local principal hyperbolicity prove evolution
stability or the absence of coordinate breakdown.

## Reproducibility and later target

The completed controls and final bookkeeping are retained in the
[final native evidence](validation/hyperboloidal-reference-wave-map-native-t2-final-completed-controls-20261009/README.md).
The [N32 partial evidence](validation/hyperboloidal-reference-wave-map-native-t2-N32-partial-observations-20261009/README.md)
and [flat-IVP values](validation/hyperboloidal-reference-wave-map-flat-IVP-values-20261009/README.md)
have separate frozen archives. Exact logs and original failures are preserved;
arrays, executables, objects, NPY/NPZ/JSONL and files over 1 MiB are metadata-only.
The earlier [N24 controls](hyperboloidal-reference-wave-map-native-controls.md)
remain unchanged. No native evolution was repeated for this addendum.

The later black-hole target must survive the inner wormhole-to-trumpet
transition with the Minkowski hyperboloidal reference retained throughout.
The current global wave-map diagnostic has no selected moving-puncture inner
blend and has not established that target. No black-hole fixed-point RHS is
subtracted to manufacture stationarity.
