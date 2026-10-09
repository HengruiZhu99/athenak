# Independent earlier-support local attribution review

The reviewed scientific index is
`902d875e6da80bf3b10e7103aa59cf123c4d8a034862f0a5d9be6f9e220c56c0`,
with helper `562d01b4382c5f9c36afc83b29208f5db3d90ca37f12890c92c03b101eb36a69`.
The read-only review passes: 36 files, 376 unchanged inputs, eight successful
command records and three external executable identities are verified. The
separately pinned postprocessor is not silently included in the earlier
376-input snapshot. No tensor gate, compiler, propagation or native binary
is rerun by this review.

The API calls frozen `qnf::Gauge(...{.85,.95,0,true})`, then adds the weighted
null-residue beta pole. Mode0 uses the supplied gauge W, here (.45,.85);
mode1 uses fixed SmoothCutoff(.65,.85). Both use finite sigma5, input
physical_trace_lapse=false/preferred_source=true, inherited xi=1/a, and the
unchanged physical-P/Q alpha blend. Assembly must use qnf::Assemble to include
the beta pole once. These are explicit different continuations of the source
through 0<W<1; no frozen late-onset definition is modified.

For the target S1,a>=.5, geometric layer(.05,.95), and gauge(.45,.85),
`Omega'=w_geo'*(Omega_out-1)+w_geo*Omega_out'` is strictly negative outside
the geometric core: `Omega_out<=1`, `w_geo'>=0`, and
`w_geo*Omega_out'<0`. Both feedback supports lie there, so their Euclidean
gradient norm is nonzero. The helper rejects nonpositive norm instead of
flooring it. This is a fixed-target admission. Its base qnf guard still
requires supplied gauge_r1<=.85; that upper-bound guard alone would not admit
arbitrary gauge_r0/geometry choices or NaN/inf parameters. In particular,
support/reference projection must remain outside any exact geometric core.

Only an algebraic beta source changes. Direct four-dimensional connection
variation gives `deltaGamma4^0=0`,
`deltaGamma4^i=-deltaBetaDot^i/alpha^2` and hence
`deltaBoxOmega=V*sigma*deltaNraw/Omega`. This generic change identity applies
in the transition as well. The full preferred-source Box identity remains
W=1-only for the alpha blend; it cannot be extended to 0<W<1 by this result.
The factored weighted null numerator avoids avoidable live-lapse divisions.
Neither that factorization nor finite source tests establish a uniform
collapsed-lapse GH bound, positivity, hyperbolicity or smooth hierarchy.

The 192 recorded off-constraint source rows give generic deltaBox error
2.664535259100376e-15 and reference residual7.549466163688232e-15. The32
outer comparisons/160 leading gauge-pole columns are exactly unchanged.
All504 recorded complete principal cases pass. These preserve the previous
outer pole certificate and exact core behavior for the target parameters;
they do not prove nonlinear boundary closure.

Independent saved-matrix checking verifies every one of560 new matrices
against the frozen late form2 plus the analytically differentiated rank-one
weight change. Maximum error is7.105427357601002e-14. Only the real radial
beta row and alpha/chi/beta_radial/gtilde_radial value columns change; the
addition is k/direction-independent. Outer and W=0 baseline matrices are
exactly identical. Saved maximum roots recompute exactly. At threshold1e-8,
377/560 matrices retain positive primitive roots (1046 positive roots counted
over matrices, including repeated parameter samples).

For a=.5, sampled maximum Re(lambda) over radii/directions is:

| Coordinate k | Spatial-norm baseline | Late feedback | Early W | Early(.65,.85) |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1.518016 | 20.399344 | 1.518016 | 1.518016 |
| 4 | 1.531952 | 22.467616 | 3.343106 | 3.169341 |
| 16 | 1.269652 | 21.298014 | 8.745175 | 8.745175 |
| 64 | -.259155 | 20.168941 | 20.168941 | 20.168941 |
| 256 | -.253218 | 15.290290 | 15.137307 | 15.137307 |

Both early weights retain worst20.168940777918813 at r=.98,k64,radial.
The k16 positive value8.745174885766524 lies below coordinate Nyquist in
N16/span2.2 (22.84794657156213) and N24/span2.1(35.90391604102621).
Here 'resolved' means only below the sampling Nyquist: it does not certify
finite-difference accuracy, the actual discrete spectrum, KO effects or a
convergent mode. No direct Nyquist or exhaustive below-Nyquist sweep is inferred.

The earlier source substantially reduces the sampled onset-region growth,
but neither weight passes an all-local-negative screen. The parent-authorized
single mode0 global attribution control remains a separate next experiment,
with its own implementation/Jv/stage/short-canonical gates. This review
does not authorize native runs, t6, a stable formulation, physical subsidiary
classification, a primary-source energy transfer, or BH evolution.

An unmatched parenthesis in the draft independent checker was caught by a
syntax-only check before any scientific check; exact draft source/command/log
are retained under history. The final checker passes. All prior frozen gates
and production source bytes are unchanged.
