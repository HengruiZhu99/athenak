# Independent native-pulse flat-wave scalar values: PASS, limited scope

The fixed one-shot high-precision integral attempt passed all282 declared
checks. It consumed the actual .2/.1,width.35 angular lapse/shift pulse on the
complete layered Minkowski initial surface, without native source imports,
kernel calls, finite harmonic truncation, numerical outer boundary or PDE time
integration. This is not yet an inverse-map or coordinate-caustic result.

The accepted source is89c96ee729b8cc741757740534566c9557dd6fdce2e66213c2fe8ff1c7e8ffe7,
VALUES-PLAN6db2628b6d331a4ceaed226be29823550f2ce078becd2db55d77dca22e07c57f,
unchanged recipe664b5da4f11a1f2c483cd05ec3fb79eb863262f02d6be81c4352b53a22b0cb77.
Root reviewed the source/math/AST and97 dependency/runtime pins before the
single execution. The authorization is10e13705ce6123fce25d5181ddd185b1360cb3133395bdb121961435798907bf.

The run used80/110 decimal digits with the pinned Python3.9.6/mpmath1.3.0
runtime. It completed in557.0666s internally (557.2304s outer), exit0 with empty
stderr. All98 child and100 outer protected byte pins were unchanged. Exact
argv, environment, stdout/stderr, source-before copies and process receipt are
under values-invocation001. The complete scientific output is values-attempt001.

There are96 coarea rows,56 initial-data rows,72 exact-control rows and18 ray
rows. Coarea rules16/32/64/128, separate128x64/64x128 radial/azimuth rules, both
precision runs, exact zero controls, complete initial normal/determinant
identities, flat constant-velocity and pure-CMC l=0/1/2 scalar controls all
passed their original thresholds. Ray comparisons independently used center,
exact-core and a short exact-outer fixed-boost parameterization. They do not
validate a general layer-ray derivative oracle.

| Check | Maximum saved scaled error |
|---|---:|
| Coarea full/axis refinement, physical u | 6.05e-40 |
| Coarea full/axis refinement, conformal phi | 3.60e-38 |
| Identical-rule precision, u/phi | 2.07e-82 / 1.96e-82 |
| Initial normal data/determinant | 9.53e-79 |
| Independent exact scalar controls | 2.01e-80 |
| Height-defect full64/full128 | 8.86e-36 |
| Ray/coarea, u and phi | 5.71e-57 |

These are numerical convergence/readback errors, not outward interval bounds.
No threshold was changed after execution, and no retry was made. The original
unbounded-cache/u-only source a6d1938d and its plan/recipe/preparation remain
byte-exact in source-history/001-unbounded-cache-u-only. The reviewed revision
only bounded caches and added the same u/phi checks; its exact diff is retained.
An atomic source patch-context failure with no file mutation is recorded in
the source preparation history, distinct from a scientific failure.

The two stopped native failure coordinates/times were scalar *reference-event*
query labels. Without solving the full4D inverse Y(X)=Yhat(t_native,x_native),
they are not the corresponding native physical events. A scalar snapshot at
reference tau cannot be renamed native time. No Jacobian, target-time causal
margin, spacetime injectivity/coverage, reconstructed metric or global bound
was computed. Positive finite scalar values cannot establish the absence of
a coordinate caustic.

The next separate gate needs complete analytic first/second integral jets,
including graph/source derivatives, then the inverse-map and target-slice
coverage analysis. Fixed-Lorentz-ray differentiation can avoid moving coarea
endpoints but needs its own derivative convergence/oracle checks. Only after
those gates can this control assess whether the exact-flat RWM gauge itself
breaks before native Z4c. It does not prove lower-order/off-constraint Z4c
stability or supply a wormhole-to-trumpet black-hole gauge. The Minkowski
hyperboloidal reference remains retained throughout the later objective.

Exact scientific receipt:
b054132e54cbe1e63b217f2cb1d416ea87e955f5ef1dfa2c945936e127730493.
Exact outer receipt:
f6e5ee874b40543841c520f398a12c5e9f4609af29520848d72199c576e2f267.
checks.json:
230adc8bed460a364ddf8e41c8a34111fbb12a21829a0ae9c4c65d78aa521fae.
The stdlib-only saved-data verifier rehashes protected inputs/outputs and
rechecks saved tolerances without importing or rerunning the oracle.
