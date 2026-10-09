# Wide flat-Penrose reference supplement: stopped before global evolution

The requested flat spatial Penrose geometry is the previously tested flat-height/FlatPowerReference exponent-one family, not a new candidate. The original broad(.2,.8) family and its rejected native tangent results remain unchanged. This fresh wide(.05,.95) supplement passes local consistency, endpoint and floating-tail checks, but worsens the actual initial discrete Hamiltonian source by2.23x atN24 and4.69x atN36. Per the parent instruction, no full global matrix, global propagation, native evolution, runtime option or production edit is made for this geometry.

## Reused formulas and numerical range

The unchanged compactification is `Omega=1-w+w*Omega_out`, with the production cutoff and `Omega_out=(S-r)(S+r)/(2aS)`. Set `d=-r Omega_prime`, `b=sqrt[d(2Omega+d)]`, `alpha=L=Omega-r Omega_prime`. The Penrose spatial metric is then exactlyI, with chi1, conformal metricI and Lambda0. The stationary extrinsic-curvature eigenvalues are `kr=-b_prime/L` and `kt=-b/(rL)`. The stored physical trace is

```
P=-(Omega*(b_prime+2*b/r)-3*b*Omega_prime)/L
```

and the tracefree radial anisotropy is `D=(-b_prime+b/r)/L`, giving `Atilde=D*(n*n-I/3)`. Its spatial derivative and the derivative of P are the existing general b-prime/b-second formulas; the original r*w/a trace shortcut is not used.

The private include overlay copies the byte-pinned existing `flat_power.hpp`, SHA7d3c22bb5e23627a5da83d542fd4100d109fad43612ee3f995c223bd50df1135. Only the base class name and constructor default exponent2->1 are changed for integration. All mathematical formulas and `ScaledPowerExp` are reused exactly. The original core and CMC/outer branches remain exact. This is a CPU research build, with no GPU portability claim. Apple arm64 long double has53 mantissa bits and no wider exponent range than double; it is not used as an extended-range workaround. The existing logarithmic sqrt(w) evaluation retains representable b-prime/b-second tails after w or b underflows, without floors or repairs.

The independent math review is separately frozen under `continuum/independent-flat-penrose-review/immutable-independent-flat-penrose-review-20261009/index.json`, SHA0925f36f411bf934342f24ce76876f9d95232de5309bb665590cb2705442bbf6. It confirms the metric/ADM/K/A/constraint identities and equivalence with the prior factorization; it does not admit a native/global test.

## Local actual-kernel and oracle gates

The fresh wide S1,a.5 supplement passes both Release and ASan/UBSan. Its214 axis/oblique points include exact core/outer branches, nextafter endpoints and small-Omega legacy behavior. It checks all nonconstant consumed reference fields and their first Cartesian derivatives; consumed second derivatives of lapse/shift and constant chi/metric/Lambda are also checked. The maximum H/M reference residuals are3.2641e-14/9.9590e-15, regular assembled actual ConformalRHS residual9.9476e-14, and assembled gauge residual6.4287e-15. First/second finite-difference jet errors are2.0100e-11/3.1079e-7 (normalized as in the reused local gate). The null identity residual is4.0202e-16. The small-Omega assembled raw RHS reaches1.3333 from legacy cancellation amplification and is byte-identical to the exact outer baseline; it is not included in the regular-region fixedpoint threshold or concealed as zero.

A fresh independent100-digit mpmath oracle checks1936 scalar values at121 transition radii, including g=-700..-1600 inner tails. Its Omega-prime formula is explicitly factored, so it does not lose an exponentially small derivative by differentiating an Omega value rounded to1. The maximum normalized error is1.2847e-14; maximum absolute error is1.1369e-13 in alpha_second. Resolved inner boost relative error is at most1.0617e-13. One tested point has b=0, b_prime=6.05423e-319 and b_second=7.54775e-313, confirming that derivatives surviving boost underflow are retained. Core/outer behavior is checked by actual C++ byte parity instead of approximate oracle equality.

The first oracle summary used an incorrect, unused denominator for its optional max_normalized_error summary field. That source/output/log is preserved under first-oracle-summary-normalization. The corrected oracle rerun uses each value's own `1+abs(exact)` denominator. All raw value comparisons and pass/fail tests were unchanged; no physical result depended on that summary field.

## Actual Cartesian initial discrete source

Both original and flat controls use the same copied, separately pinned source, actual CartesianPatch, strict-interior nonrecursive symmetric degree2 ray plan, fourth-order derivative/mixed/upwind stencils, native algebraic projector, actual constraint diagnostic and analytic reference reconstruction. Parameters are span2.1, S1,a.5, geometry(.05,.95), physical-P lapse, the original spatial-norm shift, kappa input10, kappa2=0 and KO=.1. The fixed pointwise pure-gauge seed is the original nonspherical lapse.1/shift.02/width.5 profile.

This is an actual initial projected discrete `C_h J_h` source, not a cached full matrix: RHS directional derivatives are centered about the analytic reference using gauge-seed multiplier1e-4; then the actual constraint derivative uses signed tangent intervals1e-6 and1e-7 with the native algebraic projection. The latter controls agree to about1e-7 relative or better in RMS norms. Both initial gauge states have only reference-roundoff constraints. This measures initial derivative leakage and does not assert subsequent instability from a nonzero source alone.

| N | Reference | Hdot RMS | Mdot RMS | Zdot RMS | Hdot peak | Mdot peak | Zdot peak |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 24 | Original | .47241362 | 1.73479804 | .04535568 | 5.59243664 | 8.67588167 | .53435205 |
| 24 | Flat p1 | 1.05489501 | 1.81683855 | .05363668 | 13.21549909 | 11.25706771 | .57784398 |
| 36 | Original | .08225203 | .61314557 | .00835275 | 1.12558540 | 5.28029059 | .15427989 |
| 36 | Flat p1 | .38541840 | .59451510 | .01395762 | 8.58621286 | 5.17042609 | .26487537 |

These displayed values average the positive/negative constraint tangent signs at interval1e-6. The complete signed1e-6/1e-7 values, source minima, maxima radii and radial squared-error budgets are retained in tangent-results.json and per-run JSONL. Flat/original RMS ratios are H2.2330/4.6858, M1.0473/.9696 and Z1.1826/1.6710 atN24/N36. H peak ratios are2.3631/7.6282. H is a scalar; M/Z use each reference's actual native Penrose inverse for contraction, and these two background metrics differ.

Flatness eliminates reference conformal-metric/connection distortion, but the larger lapse/height/curvature derivatives produce a worse native initial H source on both tested grids. This is consistent with the earlier broad-family rejection and does not justify another expensive global test. No convergence order, native pulse-stability acceptance, or continuum instability claim follows from two resolutions of this initial source.
