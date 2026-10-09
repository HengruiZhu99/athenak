# Frozen tensor C1 identity gate; no native acceptance

Launch b37a20f2d7a8ccc42148f1a814a80b2957e17b53, unchanged runtime27c19d20. All9 compile/run/check commands passed, including Release and ASan/UBSan. Recorded374inputs unchanged.

-384 nonlinear actual fulltensor rows cover a=.5/.75/1/2, wide layer, eight radii, unrestricted finite physicalTheta and spatialZ perturbations, kappa_input5/10 andkappa2=0/.3.
-Mechanical Appendix-B C1 physicalKij error1.294544e-14, physicalTheta4.298339e-12. Actual metric/Gamma time differentiation verifies the surviving extra spatialZ shift term and separately derived connection repair within8.052864e-13.
-C0 switch and Einstein-sector additions vanish exactly in every row. No lapse/shift RHS or kappa_input/alpha convention changed.
-360 actual full20 principal cases retain complete basis; kernel error3.56e-15, left-eigenfield error8.89e-16.
-AtOmega=2.4999375e-05, physical tensor terms diverge. Absolute metric/ADM cancellation residuals reach{'metric_absolute_residual': 0.01953125, 'C0_ADM_absolute_residual': 0.0234375, 'C1_ADM_absolute_residual': 0.0234375}; both raw and termwise normalized residuals are recorded.
-Flat affine independent actual counterexample: mechanical C1 addition0 but spatialZ covector residual1.0000000000000004, exact extra1; separate Lambda repair error1.11e-16. The initial strict binary64 equality assertion failed and is fully preserved; accepted exact symbolic identities plus1e-14 floating tolerance pass.

The generic scratch header has explicit regular/simple/double-pole parts and rejectsOmega<=0. The printed Appendix-B C1 and separately derived covariant connection completion are distinguished. This is a finiteOmega nonlinear tensor identity/principal gate only. SmallOmega full20 spectra/semigroups/RK stiffness, nonlinear scri closure, boundary/global stability and native pulse acceptance remain unresolved. Do not implement a native C1 option from this gate alone.
