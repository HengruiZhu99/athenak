# C0 prescribed damping profile: finite-Omega local gate

All 12 commands passed with empty stderr. The 378 unique recorded inputs (365 tracked src/CMake files plus 13 audit/helper/test files) were unchanged before and after. Production implementation remains 27c19d20696ea6dd4704032c51dfd026218f64f2. No tracked source, runtime option or native evolution was changed by this gate.

- Release and Debug ASan/UBSan actual tensor checks agree on 4004 reference/perturbed points at a=.5,.75,1,2, including Omega=0 parts. The only nonlinear part changes are the two Theta damping entries; normalized identity error 1.41328e-16. Einstein-sector additions are exactly zero, the profile is exactly zero at core Omega=1, and no double-pole part is introduced. Reference assembled RHS maximum is 1.04273e-12 over positive-Omega points.
- The actual 360-case principal extraction remains complete: max symbol error 3.55272e-15, left eigenfield error 8.88179e-16, maximum normalized basis condition 11.5292. This includes oblique and SPD transformed cases, both lapse flags and the existing transition parameters.
- Eight exact actual20 zero-Omega pole matrices (base/profile at four radii a) are rationally reconstructed with error below 1e-11. Every nonzero characteristic factor has strictly positive exact Hurwitz minors. All have nullity(P)=nullity(P^2)=5: five semisimple zero modes, with all other pole roots in the left half-plane. The k10,a.5 profile polynomial is

```
lambda^5 (lambda+6)^3 (lambda+8)^2
 (lambda^2+10lambda+80)^2 (3lambda^2+30lambda+304)
 (lambda^4+42lambda^3+584lambda^2+3200lambda+7680)/3.
```

- 760 actual full20 finite-Omega matrices include base/profile, reference and finite .01 perturbations, a=.5,.75,1,2, and k0 at all radii; a=.5 also includes k32/64/128/256, the three actual |k|=pi*N/2.2 Nyquist frequencies N24/36/48, and radial/oblique directions. All actual pointwise profile differences equal the two predicted Theta-column entries. Positive *finite* primitive roots remain (e.g. the reference profile worst sampled is Re lambda=3.14600 at r=.3,k0, nearly unchanged from base3.14512); these are recorded without a physical subsidiary interpretation. This is not an all-negative primitive spectrum claim.
- 96 scalar RK3 checks at the actual N24/36/48 Omega minima and pole.03 timesteps include the above diagnostic and Nyquist frequencies. Every sampled nonpositive-Re primitive root lies inside the scalar RK3 stability region, max excess0. Positive finite primitive roots are retained and excluded from that damping test. At reference k0, raw RK3 norm2 for the profile is 1.54205/1.54784/1.54816 at N24/36/48 and tends toward1.54997 at Omega1e-5, vs base1.53439. No contractivity claim follows from eigenvalue admission.
- 1000 coefficient-aware actual dual20 -> eight-constraint rows at kappa_input10,a.5 include radial/oblique k0..256 and five coefficient-derivative refinement levels. Finest max scaled closure error is 7.45302e-9. Initial gauge constraint columns are exactly zero; differentiated gauge RHS cancels to the expected coefficient-FD/roundoff level (finest absolute maximum4.35027e-4 in the large outer matrix; scaled7.15635e-9). Omitting only d kappa2 yields scaled error.00800463. All 200 sampled local eight-constraint generators have negative roots; worst Re lambda=-1.13138363 at r=.3,k2,oblique. These coefficient-aware local operators do not settle a global boundary problem or energy estimate.

The first successful pass, before adding native Nyquist frequencies, is retained separately in `first-pass-before-Nyquist`. There was no failed scientific/compile run in this profile gate. The older unrelated indicial exploration remains outside this gate and supplies no admission condition.

This evidence permits a separately validated **finite-Omega exploratory** native/global test. It establishes no finite-pulse acceptance, nonlinear constraint-manifold preservation, scri regularity, black-hole compatibility or continuum energy theorem. The helper has no floors, stronger Theta assumptions or physical-boundary falloff forcing. See DERIVATION.md for exact formulas and the restricted primary flat damping comparison.
