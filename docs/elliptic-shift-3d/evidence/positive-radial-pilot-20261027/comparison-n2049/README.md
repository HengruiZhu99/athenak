# Matched narrow-taper positive-storage comparison

This analysis compares the existing three N8, 13,312-cell, t=.0002 pilots
against the existing independent spherical N2049 reference at the same time.
**All three coupled pilots failed their recorded scientific gates.**

Both the pilots and this short reference use taper width **.08** (onset .72,
outer .8). The verified t10 family uses width **.24**. These are matched
narrow-taper numerical controls, not reproduction of that selected benchmark.
The raw pilot/reference files are unchanged. The comparison launches no
evolution and makes no spatial/time-convergence or collapse-safety claim.

The primary errors retain every active cell and its supplied CSV weight,
`map_volume*sqrt(det(g_tilde))/chi^(3/2)`. This is the Penrose spatial
proper-volume quadrature used by these pilots; an additional physical
`Omega^(-3)` volume factor is not introduced. Regional diagnostics overlap and
supplement the all-cell result. Positive alpha/chi relative errors use the
actual positive reference values, without denominator floors.

| Storage | Alpha RMS error | Chi RMS error | Radial shift RMS error | H RMS | M RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct positive |2.64820e-5|3.97325e-5|.0336908|3.35326|.138495|
| Log/reference |2.57526e-5|3.85969e-5|.0318223|3.63880|.140161|
| Square-root |2.73950e-5|4.10321e-5|.0357781|24.3480|.137755|

The r<.1 H RMS is8.41208/9.12830/61.0864 respectively. The square-root control
is worse in that neighborhood on this coarse wormhole grid; this does not
establish a universal ranking across changing puncture powers.

The original-vector solve residuals reported by the pilots have stage maximum
RMS5.59e-11–5.97e-11 and point maxima4.10e-10–5.30e-10. The comparison carries
those reports explicitly; derivative-incomplete CSV files do not permit an
independent recomputation of the full operator. The coupled tensor scri poles
are4.25604e-6/3.93162e-6/5.02810e-6, all above the unchanged1e-7 target.
Actual ghost clock/null/tangential residuals pass their recorded targets.

The N1025-to-N2049 reference differences, evaluated at every matched 3D cell
with mode1 weights, have alpha RMS2.53e-11, chi RMS1.14e-12, curvature RMS
5.58e-8 and radial-shift RMS4.04e-11. These are independent finite-grid
reference controls, not an exact continuum solution.

The sphere comparison retains all64 directions at each of19 radii. Raw
scatter includes mesh/transfer error and extraction interpolation error. For
example, maximum A_rr scatter is about.2014 for all modes, considerably larger
than the matched-cell A_rr RMS errors around5e-4 near this steep profile. No
extraction control or angular scatter is subtracted, and no radial averaging
enters an evolution.

`comparison.json` records all eight field errors, region splits, raw angular
means/scatter/extrema, unchanged boundary gates, reference controls, input
hashes and analysis provenance. Both PNG figures have been visually checked.
The earlier `analysis-initial.py` snapshot preserves the initial analysis
implementation; the current script and qualification are authoritative here.

Reproduce from the isolated repository root into an unused output directory:

```sh
OPENBLAS_NUM_THREADS=1 python3 tst/elliptic_shift_3d/compare_positive_radial.py \
  --pilot-root docs/elliptic-shift-3d/evidence/positive-radial-pilot-20261027 \
  --reference-root docs/elliptic-shift-3d/evidence/wormhole-radial-reference-20261027 \
  --output build-shift-3d/positive-radial-comparison
```

The script also accepts losslessly compressed `.csv.gz` inputs. It refuses to
overwrite existing comparison evidence without `--overwrite`, verifies time,
cell counts, angular counts and input stability, and checks that all-cell H/M
norms reproduce the supplied stdout values. No cells or directions are
silently removed when an input fails validation.
