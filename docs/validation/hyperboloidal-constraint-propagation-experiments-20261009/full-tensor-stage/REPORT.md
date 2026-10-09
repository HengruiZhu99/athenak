# Native full tensor stage audit (completed interim evidence)

The actual final-only native SSPRK(3,3) tangent is **P_ref R3(dt J22)Lift**, whereas the projected continuous20 generator is **L20=P_ref J22Lift**. Both are constructed from the actual analytic LayerReference jets, strict spherical ghost plan, LoadMeshJet centered/mixed derivatives, Lx correction, and KO mask. This audit verifies their relationship; it makes no pulse stability, global spectral, boundary closure, or SBP claim. No production files changed. All provisional long t2/Krylov/Ritz results are excluded from this catalog.

Parameters: N16, span2.2, h=.1375,1640 strict ball nodes,32800 free20/36080 raw22 unknowns, S1,a.5,wide(.05,.95),kappa10,nativeKO.1,symmetric quadratic ghosts. The actual physical lapse branch is used; the private norm control has xi2,eta6,C2/3 and the frozen native wrapper. Three dormant B fields are excluded from both physical-gauge tangent systems.

Production headers and native lifecycle sources match27c19d20696ea6dd4704032c51dfd026218f64f2 byte-for-byte. Launch HEAD b37a20f2d7a8ccc42148f1a814a80b2957e17b53. The copied full22 build receipt records1153/1155 compiler dependencies, exact AppleClang flags/static Kokkos archives, executable hashes, and frozen wrapper hashes. Executables, CSR matrices, geometry/vector arrays remain at their immutable scratch paths; only metadata/hashes are archived here.

Six retrospective **implementation audit** thresholds and all observed values are in six-audit-gates.json. They are numerical consistency checks, not pre-registered physical acceptance criteria. Projected-v1 receipts retain the full six-amplitude Jv sweep. Full22 receipts retain all raw-amplitude and native one-step sweeps, including failures to be asymptotic at larger amplitudes.

Reference Prepare change is exactly0. Native projected reference RHS maximum6.0836e-12; H/M/Z=3.2395e-14/1.3306e-15/6.6431e-16. All63576 donor references are strictly active and nonrecursive; constant-weight sum error1.3323e-15. Native Lift/Restrict and Prepare derivatives converge to the analytic tangent/ghost fill; at epsilon1e-6 the maxima are below1e-8.

Full22 CSR matvec agrees with actual cached stencil application at approximately1e-15 relative. Raw native centered RHS derivatives agree at approximately1e-10 to1e-9 when amplitude reaches the asymptotic range. The nonlinear native one-step derivative agrees with P_ref R3(dtJ22)Lift within4e-10 relative to input at epsilon1e-5 for tested dt/2,dt,2dt. Nominal pole dt=.03 minOmega=8.0859375e-5.

For smooth gauge/geometry seeds, J22Lift algebraic-normal fractions are roundoff (~2.4e-12/~5.5e-13), and one-step native22 versus every-RHS projected20 differences scale as dt^3. Controlled shell/white random seeds have normal fractions~1.96e-4/~2.73e-4 and asymptotic dt^2 one-step differences. Thus normal-stage coupling is real, but cannot explain the instantaneous smooth-gauge constraint tangent defect.

At t=.01,124 native final-only steps and248 half steps are compared with124 every-RHS projected steps. Relative to the initial Euclidean free20 vector, native versus projected differences are~2.28e-6 for the gauge pulse and~9.8e-5 for shell data; halving dt approximately halves accumulated differences. These are short-time consistency observations, not long-time bounds or invariant energy statements.

Exact block similarity multiplies {chi,g,alpha,beta} by1/h and leaves {P,A,Lambda,Theta} unchanged; outputs are unscaled. It preserves the spectrum and t=.01 states to2e-15 relative, but reduces basic norm1 without improving measured Taylor pilot cost (~16.3–16.6s). A redundant production balanced long run was cancelled after preserving its completed pilot; no long result is claimed.

Sources and exact original commands are under sources/. Original receipt paths/hashes, full scalar observations, short run timings and projection lifecycle are retained. The manifest covers every archived byte except itself, including this report.
