# Independent boundary review of audit-draft.md

Read-only review against the frozen C0 mode diagnostic396899199b3e94afdf28133c3db094690c18c094aa86e38ad42edfc8829c1d69, finite-step action3e212ec5e437c0190f687fcd3b95d52671fe814b36c9e62469924a4a01e241b0, and constraint-Taylor derivation/check sources5f013948f01b54416065ce2bd1cf182e1b61c8e8d50a84afb1264ea128925217. No scientific rerun or frozen-source edit was made.

One numerical/units correction is needed in the finite-step paragraph. The sentence saying state residuals3.57e-6/9.66e-7/2.51e-7 are “much larger than the generator residual” compares different units, and its quarter-step number is actually smaller than the numerical generator residual5.89e-7. Replace it with:

> State residuals are3.57e-6/9.66e-7/2.51e-7 per unit input. Dividing by dt gives .04419/.02389/.01241 per unit time, far above the projected20 RK3 residual per unit time of about5.887e-7; intermediate algebraic-normal feedback dominates this directional discrepancy.

The rest of the finite-step values and approximate-mode limitations are correct. In particular, “exact cached” refers to the polynomial action of the original finite-difference Jacobian cache, and the Rayleigh growth values are not certified finite-step eigenvalues. The reported1.80e-11 native one-step agreement is in Euclidean state L2 per unit input at perturbation size1e-4.

Two small scope clarifications are recommended:

1. Open the Taylor section by identifying it as a **linear actual-kernel audit** using physical-P lapse and spatial-norm shift (xi=1/a,rho=1.5), S=1, kappa input10, with the two stated kappa2 reference values. Before the displayed gauge-witness RHS rates, say “the linearized actual RHS has”. The final paragraph already says “initial linear”, but putting this next to the formulas avoids reading those rates as a finite-amplitude nonlinear statement or as applying to every gauge choice. The unchanged spatial/extrinsic-curvature initial data do indeed have exactly vanishing initial Einstein constraints; that fact does not by itself extend the linear gauge/null time rates nonlinearly.

2. The .9999999987 overlap is the unweighted **free20 Euclidean squared projection onto span{Re(v),Im(v)}**, from classify_candidate.py. The preceding61.6%Lambda/etc contents instead use lifted raw22 components with reference-volume weights. The existing generic “raw-component overlap” is not wrong, but naming free20 makes those two contractions unambiguous. Neither is an energy fraction or a biorthogonal dynamical modal amplitude.

The constraint-map ranks/counts, M0/Z0/Theta1 and Theta_t0 identities, the Omega-squared-P witness, initial gauge-witness Nraw/Q behavior and listed scalar corner rates agree with the frozen actual/symbolic evidence. The smooth-corner versus finite-Q blowup/current-pulse qualifications are appropriate. No other correction found.
