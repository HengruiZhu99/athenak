# Independent Einstein gauge/null invariance review

The negative sigma5 result is confirmed. The reviewed actual-kernel gate is
`4cfbc8ed743c46787f617c33fdbe9875ec092fd117e82f1daddf5e165f2ef29a`;
the independent root ADM/divergence-Box gate is
`78c6870117cf1fb74e1175cda6abc38b08b5ac9092f3fce7d904bf8dd958e585`.
Neither frozen tree is changed or rerun. The new review uses saved kernel rows
and a separate future-normal projection of Box, retaining the full varying
reference lapse. SymPy version, module hash, Python executable/version and
80-digit mpmath oracle are recorded in result.json.

The physical ADM metric equation gives, initially on exactly reference
geometric/P/Theta fields,

```
delta G_t = -2r^2 b'/a^2-2r^3 b/(a^3 Omega)-2r^2 f/(a^3 Omega),
delta wn = r b/(a h)+r^2 f/(a^2 h^2),
delta wn_t = b wn_ref'+beta_ref (delta wn)'
             -f beta_ref wn_ref'/h
             +h[-3wn_ref delta wn/Omega+Kbar_ref delta wn
                 +(f/h)'Omega'-sigma delta Nraw/Omega].
```

Here f=deltaalpha, b=deltabeta_rad, h=(1+r^2)/(2a),
wn_ref=-r^2/(a^2 h), Kbar_ref=-3/(a h). This follows from
`Box=Delta+Kbar*wn-n(wn)+acceleration.dOmega`, with future normal
`n=(partial_t-beta.grad)/alpha` and the preferred source
`delta Box=sigma delta Nraw/Omega`. It is an initial gauge-only identity, not a
closed equation for general perturbed spatial geometry.

This projection independently reproduces root's exact arbitrary-radial-F/B
transport and reaction identity, its sigma3 weighted reaction
`-(r^2+3)(3r^2-1)/(4ar^2)` and limit -2/a. A tangential initial shift cancels
analytically between the metric trace and spatial-divergence contributions.
Common alpha/beta rescaling also has exactly zero initial null rate. These
analytical cancellations do not replace an actual full-angular Taylor gate.

For b=Omega^m, f=0 the Q lapse, preferred Box and ADM metric equations
independently reconstruct every full20 first-RHS value. Z=0 preservation gives
Lambda_t=Delta b+(grad div b)/3; all angular derivatives of the radial vector
are retained. An80-digit evaluation matches the saved rows to4.39e-15;
Ndot/Qnumdot to8.29e-17. Exact limits for tested m2/3/4 are

```
Nraw_t/Omega^(m-1) -> 4(m+1-sigma)/a^3.
```

For m2,sigma5 the coefficient is -8/a^3, or -64 at a=.5. Direct next-N1 values
match within3.56e-15, while next R0 remains at roundoff. Initial physical ADM,
spatial Z and physicalTheta constraints are exactly zero in all480 saved rows.
The beta perturbation changes none of those initial data, and an outer-collar
cutoff can extend it smoothly into the core. Initial null/Q numerators are
quadratic, all first-jet conditions vanish, and the full4D source/Hessian
calculation uses correctly assembled metric derivatives. Its trace-curvature
value is inferred from the physical vacuum trace relation, as documented;
curvature time derivatives are not bounded by this check.

Thus the stated smooth-in-time first-jet ideal with Nraw=O(Omega^2) is not
invariant for sigma5 even on these Einstein initial data. This is not a proof
of finiteOmega amplitude blowup, lack of physical Einstein evolution, or a
global instability rate. It does not establish a fully higher-order compatible
spacetime ideal. Interchanging time and boundary limits may fail in a stiff
corner layer. Sigma3 cancels this one radial obstruction and is leading
Hurwitz only for K=kappa_input*a^2>3/2; no sigma3 candidate is admitted.

One precision clarification is external to the frozen reports: the saved
common-rescaling control has maximum actual |Ndot|=4.021028327534997e-9,
although the exact identity is zero. This is retained as a smallOmega numerical
cancellation limitation, separately checked at1e-8. Every other comparison in
this review is below2e-12. No uniform smallOmega precision bound is claimed.
The original report's unquantified word "roundoff" should not be interpreted as
a1e-14 bound for this particular control. The gate owner has been notified.

All26 actual-gate files, seven root-gate files,372 source inputs, five original
commands and two original executables are verified. Actual Release/ASan-UBSan
Debug JSON is byte-identical. The independent checker history retains symbolic
integer-power normalization and structural-limit assertion failures, followed
by the overly strict common-control threshold failure. Repairs affect only the
new checker; no original artifact or scientific source was changed. No kernel
compilation, propagation or native evolution was executed for this review.
