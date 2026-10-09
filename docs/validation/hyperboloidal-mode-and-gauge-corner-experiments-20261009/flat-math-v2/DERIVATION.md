# Derived flat-Penrose-spatial Minkowski layer: mathematical assessment

This is a new height choice, not the original b=rw/a family. The user authorized changes to both compactification and height. Keep the original Ω=1−w+wΩout, Ωout=(S²−r²)/(2aS), a≥S/2,0<r0<r1<S. For r≥0, Ωout≤1 and Ωout'≤0, hence Ω'=w'(Ωout−1)+wΩout'≤0. Set

```
d=−rΩ'≥0, L=Ω+d,
b=sqrt[d(2Ω+d)], A=L,
R=r/Ω, h_R=b/L.
```

Then A²=Ω²+b²=L² exactly, so the Minkowski height construction has Penrose spatial metric δij, χ=1, gtildeij=δij, Λi=0 throughout. This is a consistent derived geometry, not a deletion of the nonflat original reference's connection. It has h_R=0 in the exact Cauchy core and the standard outer b=r/a,A=L=(S²+r²)/(2aS). The physical spatial metric is Ω⁻²δij; it is not physically Euclidean. L>0 and 0≤h_R<1 for Ω>0. Reference outgoing/incoming speeds are L+b and −Ω²/(L+b), reaching2S/a and0 at scri.

For the logistic cutoff write g=−1/s+1/t and k=g'=(1/s²+1/t²)/(r1−r0). The useful positive factorization is

```
d=w F,
F=r[(1−w)k(1−Ωout)+r/(aS)]>0,
b=sqrt(w)*sqrt[F(2Ω+wF)].
```

Near r0, F grows at most s⁻² while sqrt(w)~exp(−1/(2s)+bounded). Thus b and every radial derivative vanish at the inner endpoint. At r1, b is already positive and its difference from r/a is smooth and flat. The derived reference is C∞. A floating implementation must preserve sqrt(w) before w underflows; sqrt of a rounded-zero w is insufficient. A private CPU long-double radial calculation or log-factor evaluation can be tested, but no public implementation is accepted here.

The extrinsic curvature must be recomputed from this b:

```
Kbar_R^R=−b'/L, Kbar_A^B=−b/(rL) delta_A^B,
Kbar=−(b'+2b/r)/L,
w_n=bΩ'/L,
Kphys=−[Ω(b'+2b/r)−3bΩ']/L.
```

Atilde=Kbarij−δij Kbar/3, with general radial derivatives retained. Beta=−b n and alpha=L; alpha'=−rΩ'',alpha''=−Ω''−rΩ'''. Do not use the original rw/a-specific physical trace shortcut or tangential-curvature derivative. The reference null and preferred-source identities retain Nhat=(Ω'/L)² and the original factored WhatOmega expression because they follow from the same Minkowski height/compactification construction.

The80-digit399-point comparison at S1,a.5,r0.05,r1.95 checks the algebraic identity, spatial-flat metric and interior causality. It does not evaluate the actual tensor kernel or justify evolution. The flat spatial metric comes with larger sampled curvature/lapse derivatives: |Kphys|max8.46787 versus6.09370, |Kphys'|max34.2236 versus16.7053, |alpha''|max68.573 versus13.276. Therefore no stability or accuracy improvement is assumed. A local actual-reference/derivative/endpoint/constraint/RHS gate must precede any global or native trial.

The first numerical assessment differentiated Ω near1 directly and lost the exponentially small derivative at80 digits. That strict positive-height check failed; exact source/log are preserved. The corrected assessment uses analytic factored Ω' and passes. This was a mathematical diagnostic failure; no native or production source was touched.
