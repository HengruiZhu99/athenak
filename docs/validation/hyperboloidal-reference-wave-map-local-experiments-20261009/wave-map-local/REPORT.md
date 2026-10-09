# Local physical reference wave-map prototype: PASS, no evolution admission

The private templated helper `reference_wave_map.hpp` implements the reviewed physical reference wave-map gauge using actual `LayerPoint<T>`/`Z4cJet<T>` conventions. Stored/evolved P=Kphysical-2Theta physical and the full Lambda/Z coupling remain intact. Global harmonic slicing/integrated shift are a diagnostic choice. C0 production stays byte-preserved at implementation27c19d20696ea6dd4704032c51dfd026218f64f2; there is no BH reference, numerical gauge-RHS subtraction, Q full subtraction, floor, moving-puncture blend or exact-scri claim.

Final accepted attempt: `attempts/1791563579962029000/receipt.json`, SHA25645f789e122a78a3c3b651dcc1317d83d7f09cb4931032ca841952ece881ed8d2. Fixed recipe0274853113737d3cdbb13d54d96b5d61bed0dd8ba8c0feedcc13cbe512f1dc68 preceded compilation. Launch HEAD35d348712d11ae4db9ce8568476b27b0f64e113a is separately recorded. The helper SHA25656d61c56bf37bf33591bec74c029c686c7d39f7abbe62dd090becda8e7176e28 was unchanged across all attempts. Release executable78349cd4bc458e35bdc52a0ba7a6578eaac8dc39f6d235f22597d3ec0b457416 and Debug ASan/UBSan77f4995d9319d7e2a7162718774330593c47797d19957827838fd3fd75455dec produce identical numeric JSON. All five recorded final commands exit0 with empty stderr.374 source inputs comprise365 production files,8 local source/plan files and the reviewed derivation; compiler-emitted dependency counts are1053 Release/1055 Debug. Exact flags/commands/compiler/source/dependency/executable hashes and timings are retained.

The fixed S1 grid covers a=.5,.75,1,2, wide(.05,.95)/narrow(.2,.8) geometric layers and pure-CMC controls,21 radial points throughr=.99999 and3 orientations. There are756 reference and756 off-constraint comparisons,132 exact-core rows,480 tiny-positive-lapse assembled-gauge controls, and1440 det/trace-compatible dual directional controls. The independent connection construction evaluates the full embedding Jacobian/Hessians with direct height derivatives. A separate stationary ADM four-metric supplies an independent physical/conformal connection. Full spacetime inverse contractions, including time/mixed entries and s=bar(g)^bc bar(ghat)_bc, are used in the source comparison. The actual C0 tensor RHS supplies metric time derivatives in the full-Z source check; this does not assert a covariant C0 subsidiary system.

Scaled errors use |x-y|/max(1,|x|,|y|); absolute maxima are labeled below.

| Check | Result |
|---|---:|
| embedding scaled connection |4.1388700468951706e-13|
| independent ADM scaled connection |6.994405055138486e-15|
| full conformal connection transformation |5.6343818499726694e-15|
| physical/conformal Fbar forms |7.65940045763247e-15|
| actual C0 full-Z conformal source |7.519809427924962e-15|
| actual C0 full-Z physical source |3.219646771412954e-15|
| factored reference gauge absolute max |0 exactly|
| direct raw reference gauge absolute max |9.256188232607907e-11|
| direct raw reference C0 geometry absolute max |1.3322676295501878e-10|
| factored/unfactored gauge |1.0369965715386872e-13|
| stationary B identity |4.107825191113079e-15|
| exact harmonic core |3.469446951953614e-18|

The gauge directional FD maxima for eps1e-4,5e-5,2.5e-5 are1.6987370550123912e-6,4.246035406698775e-7,1.0661608689126492e-7. The corresponding Omega*Fbar maxima are1.293765356268538e-7,3.234564702969592e-8,8.0910225458091e-9. The final levels pass the fixed2e-7 thresholds with the expected approximately fourth error reduction when eps halves. This is a local source/jet derivative test, not a principal symbol or differential operator. Both returned poles are assembled exactly once. The helper factors exact stationary Minkowski identities and inverse-metric deviations; no live-lapse division occurs in assembled gauge rows. Tiny positive lapse checks show finite values (maximum401139.53008238465) and do not establish bounded source functions, positivity preservation, or puncture hyperbolicity.

All first failures remain retained. Attempt1791563251078338000 failed the direct binary64 embedding cancellation check at2.850070010831953e-6 against2e-7; all other rows passed. Attempt1791563327489692000 failed compilation mechanically when a long-double radial argument conflicted with double cutoff parameters. Attempt1791563350407249000 still failed embedding at3.8150884679912554e-6: this arm64 compiler's long double has no additional significand precision. No tolerance/grid/helper equation changed. Independent embedding arithmetic was then changed to explicit FMA double-double (approx106 bits, not rigorous intervals); subtraction-cancellation/division/sqrt sanity residual is2.0637160873731504e-33. Attempt1791563435278429000 passed. A final additive harness refinement rejects nonfinite comparison operands explicitly; the accepted final run has identical numeric output to that prior pass. The helper stayed byte-identical throughout.

For a finite-amplitude inverse coordinate map Y=X+epsilon a phi, the inverse at fixed Y gives delta g=-Lie_(a phi) eta. The active linear-coordinate convention therefore uses xi=-a phi. The nonlinear oracle and complete pure-coordinate lift are separate gates. No spectra, operators, propagation, native evolution, gauge adoption or BH acceptance were run or inferred here. The later goal remains a substantial finite angular pulse and single-BH wormhole-to-trumpet transition with the Minkowski hyperboloidal reference retained.
