# Independent finite-radius negative-screen review

The frozen input is `conformal-q-followup/immutable-Q-null-finite-frequency-negative-20261009/index.json`,
SHA `adaa2b1437054f4ba4cf6471eafb6f83af96a31e9ea376d723a2666abcf5e72a`.
The read-only review passes: all 19 frozen files, 371 source inputs, three
successful command records and the external executable identity are rechecked.
No tensor kernel, compiler, propagation or native binary is rerun.

The 1120 saved full20 matrices use forward column action `dq/dt=Lq`, with
`exp(+ik n.x)` perturbations. The cosine/value/second-derivative seed supplies
the real part and the positive first-derivative sine seed supplies the imaginary
part. Recomputed largest real parts and all recorded positive-root counts agree
exactly with the saved analysis. Positivity uses the recorded 1e-8 numerical
threshold. Every matrix is retained, so the complete spectra remain
reproducible; the compact root file stores the maximum root and positive count.

Independent direct differentiation of the algebraic source formulas reproduces
the alpha-only blend matrix difference to 7.105427357601002e-15 and the
sigma5-minus0 feedback difference to 2.273736754432321e-13. Both differences
are real and independent of wave number/direction. At the aligned reference
point, feedback changes only the radial-beta row and is rank one, coupling
the alpha, chi, beta_radial and gtilde_radial values. The alpha blend changes
only the alpha row; its P and live-derivative coefficients cancel. These
checks use independently evaluated scalar reference/weight quantities and
the saved matrices, without reevaluating the full tensor equations.

The spatial-norm baseline sign is consistent with outward Euclidean
`n=-dOmega/|dOmega|`: the source written with a negative Omega gradient equals
the archived baseline `-eta W[deltaBeta+C n deltaG/Ghat]`.
All forms inherit xi=1/a. The two Q variants coincide for r>=.85, the
physical-inner variant matches the baseline at W=0,r=.45, and sigma5 equals
sigma0 at its r=.85 onset. These matrix differences are exactly zero.

At a=.5 the blended Q candidate's largest sampled real part is
22.467615596732056 at r=.85,k4,radial, with imaginary part
-14.009667653364426. Its feedback weight is exactly zero at that radius,
and the matrix exactly equals original Q/sigma0. The baseline's maximum over
the same sampled point set is 1.531952335962095 at r=.45,k4,radial. The two
maxima occur at different radii; this is a worst-over-grid comparison. All
positive primitive roots are retained rather than classified as gauge or
physical subsidiary modes.

These are frozen reference-coefficient primitive generator roots in coordinate
time. They do not supply coefficient-gradient transport, physical constraint
subsidiary classification, global boundary behavior, eigenvalue convergence,
or an energy estimate. No primary-source theorem is invoked to transfer this
screen to continuum/global or native stability. Negative leading scri poles
cannot remove the reported finite-radius positive roots by inference.

Separately named earlier feedback weights can leave the outer pole and exact
Cauchy/gauge core unchanged for the target geometry. They are different
sources in 0<W<1 and cannot inherit the W=1 preferred-source/Box identity in
that transition. Their source/helper/frequency tests belong in a separate
gate. This review does not admit either earlier alternative.
