# Actual configuration derivative and reference/frame control

The private continuum bridge now supplies the radial derivative of the actual linearized configuration source about the stationary Minkowski layer reference. Release and Debug ASan/UBSan gates pass with byte-identical numerical output. This is a source/reduction checkpoint; no radial Z4c matrix, propagation, nonlinear pulse or black-hole result is accepted here. Production remains implementation27c19d20696ea6dd4704032c51dfd026218f64f2.

The unchanged baseline uses physical P storage/evolution, physical-trace lapse, preferred source off, spatial-norm shift feedback, a=.5,S=1, geometry.05–.95, gauge.45–.85, xi=2,rho=1.5 and kappa1=10/alpha,kappa2=0. No Q/null feedback or C1 extension is included.

## Complete configuration reduction

A first Cartesian spatial jet over the perturbation dual differentiates only chi, the six symmetric metric entries, lapse and shift. These equations require at most second configuration jets and first A jets. No momentum evolution equation is differentiated. The templated gauge retains the unchanged physical-P/spatial-norm formula and adds the beta pole once.

The geometric configuration equations are

```
chi_t = beta.dchi - 2 chi div(beta)/3
        + 2 alpha chi (P+2 Theta_physical-3 omega_n)/(3 Omega),
gtilde_t,ij = -2 alpha Atilde_ij + Lie_beta(gtilde)_ij
             - 2 gtilde_ij div(beta)/3.
```

All reference, cutoff, lapse, shift, Omega and metric coefficients retain their consumed spatial derivatives. The shared Geometry helper also evaluates curvature for its finite-validity check; finite placeholders for unused higher jets are varied in both background and perturbation-dual components. They change neither the selected configuration values nor their mixed spatial/perturbation derivatives: measured difference exactly0. This tests equation-output independence without claiming that higher analytic reference jets have been supplied.

Let e_a^i and e^a_i be the reference Penrose orthonormal frame/coframe. The radial frame is e_n^i=c n^i, c=alpha_hat/L. Define

```
h_ab = e_a^i e_b^j delta gtilde_ij/chi_hat,
S_ab = e_a^i e_b^j [delta Atilde_ij
       - gtilde_ref,ij (Aref^kl delta gtilde_kl)/3]/chi_hat,
chart(T) = (T_nn,T_nT,T_nU,(T_TT-T_UU)/2,T_TU).
U = (delta alpha/alpha_hat,delta chi/chi_hat,chart(h),frame(delta beta)/alpha_hat),
V = (delta P/Omega,delta Theta_physical/Omega,chart(S),chi_hat frame(delta Lambda)).
```

The chart is not a Frobenius-orthonormal STF basis. Here vector frame components use the coframe. The reduction is q=D_s U with D_s=c partial_r, including all derivatives of the complete reference-normalized field. It consumes U,U',U'' and V,V' only. In particular, V_t retains the metric-RHS contribution in the A trace subtraction. Unused second A-reference derivatives do not enter this reduction or its source. The fixed reference maps are also applied to the raw source to obtain U_t,D_s U_t,V_t.

## Measured scope and results

The new derivative check covers m=0, all8/16/20 channels in J=0/1/2, five radii .025,.30,.60,.85,.98 and the single oblique ray(.36,-.48,.8):220 cases. Independent fourth-order radial differences use h=.001,.0005,.00025,.000125. It retains220 configuration and220 complete-map sequences.

| Check | Recorded error |
|---|---:|
|Configuration values versus unchanged actual full22 action, scaled|3.9876504894905472e−16|
|Unused mixed higher-jet extensions|0|
|Input/output algebraic normals, scaled|1.3335285594754532e−16 /5.549497778660224e−16|
|Input/output algebraic normals, absolute|1.3335285594754532e−16 /5.684341886080802e−14|
|Configuration derivative final h, scaled|6.15406963974717e−9|
|Complete U/V map derivative final h, scaled|1.8247330498736127e−8|

Both final-h errors pass the predeclared2e−7 gate. There is measured fourth-order evidence in67 configuration sequences and71 map sequences;153 and149 are within tolerance with order unclassified. Those cases are not automatically called roundoff-limited. This checkpoint is not an all-m/all-angle derivative certificate. At r=.98 the continuum source-binding stencil can extend past the separate finite boundary rb=.98 while staying inside scri; it is not the strict-inside-rb physical constraint-rate gate.

Release compilation took1.875716709s and the gate.581850916s; Debug ASan/UBSan compilation took1.038829916s and the gate13.346314125s. The builds record1060 and1062 compiler dependencies respectively, plus four link archives each; the union contains1066 distinct dependency/library inputs. Compile and accepted run stderr are empty. Release flags are C++17,-O3,-DNDEBUG; Debug uses-O0,-g,-fsanitize=address,undefined,-fno-omit-frame-pointer. Complete commands, compiler version, dependency hashes and production readback are retained.

The accepted bridge source SHA256 is65cc3df6ff7655a28beb61aab445055f7d84123c07101e6e4e5cfd6ab4251438. Recorded executable SHA256 values are eaf81162dabb6cefea329020cfdd47a04ce0a734155a70410d0c38c38ccb7f58 and eaee555e728f85b44cb051e5c6fa0469b7078d580423d860da29348877a2b7ba. Numerical stdout is byte-identical, SHA256 da02927e9d7784a3bfd9abdc7ee6788d4ee454d0cb26793b16e4a813b8c4f69a. Build/run HEAD was2e0aa3b0d5ade2fef4d86807a4569fef61f6b202; this is distinct from compiled production27c19d2.

The reused builder overwrote both accepted executable paths in later API
builds and had not saved those binaries in the attempt folders. This is a
preservation error: the recorded executable hashes are historical receipt
values, not fresh readbacks of retained binaries. Exact source/build/dependency
records and scientific outputs remain; neither accepted gate is rewritten or
rerun to conceal the loss. Future builds retain unique per-attempt executable
paths. Superseded build-latest pointer JSONs are reconstructed additively from
the retained attempt/receipt/executable fields and must match their original
gate-recorded SHA256 before collection; current pointers remain untouched.

## Preserved failures and remaining work

The first compiler attempt failed on a name collision with a frozen helper and a mixed-size auto declaration. No numerical run followed that attempt. A subsequent derivative readback exposed a real adapter error: floating normalization crossed a screen-axis selector at n_z=.8 and generated artificial frame jumps. The failed source/output/sequences are retained. The corrected check locks the center normal and screen along every radial FD ray; the equations and tolerance are unchanged. Screen O(2) covariance remains a separate required matrix/trace gate. An independent review also preserved a source-pin race that aborted before issuing a PASS; stable chunk002 was then reviewed from exact copies.

The point-energy source implements the declared weak/strong densities and independent remainder/Gamma volume formula, but this checkpoint does not numerically admit the energy assembly. Coupled mass conditioning, symmetrizer gradients, independently assembled bulk identity, boundary forcing, incoming sector ranks, exact core and transition/collar physical subsidiary rates remain separate gates. Eigenvalues and propagation are not included. The eventual substantial angular Minkowski pulse and the later inner wormhole-to-trumpet single BH must still be validated with the Minkowski hyperboloidal reference throughout.
