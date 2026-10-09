# Live C0 damping coefficient: finite-Ω experiment

The live coefficient gives an exact algebraic cancellation in the C0 subsidiary system, but does not suppress the later growth of the tested Cartesian discrete operator. It remains private research code. Production implementation is `27c19d20696ea6dd4704032c51dfd026218f64f2`; no live-coefficient option is enabled publicly.

Use physical Θ, P=Kphys−2Θphys, κinput=αbar κ1=10, and V=SmoothCutoff(r;.15,.3). The candidate is

```
κ2 = V [Ω−1+2 β^j Ω_j/κinput],
κeff = κinput(1+κ2),
σ = [2 β·dΩ−κeff]/Ω
  = (1−V)[2 β·dΩ−κinput]/Ω−V κinput.
```

Thus σ=−κinput wherever V=1 for arbitrary live states. Its full gradient retains V', live shift derivatives and the Ω Hessian. The coefficient-aware subsidiary corrections include −4K κinput κ2 Θ/Ω and 2∂i(κinput κ2 Θ/Ω). Omitting the coefficient gradient fails the actual chain check by .360234. The earlier prescribed profile agrees on the a=S/2 outer reference, but does not share this general live identity.

The uncut coefficient violates κeff≤10 by a reproducible 4.2005e−9 for the initial angular pulse near r=.1058. A 50-digit interval calculation proves the stated initial pulse interval for the chosen turn-on; it does not prove preservation by evolution. No clamp or floor is used. The flat constant-coefficient damping result's admissible interval is −1<κ2≤0 under the separately checked quartic convention; it does not transfer an energy estimate to this variable-coefficient C0 problem.

The actual local gate has 17 successful commands, 384 unchanged inputs, 4004 nonlinear Release/ASan-UBSan states, 360 complete principal-symbol cases, 1064 finite-Ω full20 matrices, 16 leading pole matrices, 4660 coefficient-aware chain rows and 192 actual-span frequency/RK samples. Rationally reconstructed analytic reference pole matrices have five semisimple zero modes and Hurwitz nonzero factors; finite-Θ samples are checked numerically. Positive finite-Ω primitive roots and bounded inner subsidiary roots are retained. Scalar RK checks exclude positive-real roots; they do not prove a nonnormal propagator bound. The independent algebra/matrix/grid review passes.

The private AthenaK build changes only the helper include and four live/reference evolution/pole-diagnostic κ2 arguments. Removing them restores the entire original Cartesian header exactly. Six translation units are rebuilt; 182 original objects and four libraries remain unchanged. The base catalog has 365 src/root-CMake inputs plus four auxiliary input/analysis files, and four separate spatial-norm overlays.

| Native preflight | Result |
| --- | --- |
| Reference to t=.05 | Exit0, 16.4408s, maximum binary64 drift1.32352e−14 |
| Angular pulse to t=.02 | Exit0, 6.84234s; H/M/Z=.00307913226/.00489515082/.00119479669 |
| Pulse versus original spatial-norm | H/M/Z ratios .989874/1.000225/.997324 |
| All six saved states | All25 binary64 fields, binary32 casts, initial equality, determinant/trace and positive spatial metric checks pass |
| Sampled live interval | κ2∈[−.1971507732,0], κeff∈[8.028492268,10] |

A separate utility verifies the manual/helper coefficient identity to2.22e−16 and σ=−10 in the V=1 cells to2.274e−13. These six snapshots do not establish future coefficient bounds. Historical `physical_metric_eigen_*` keys measure the Penrose spatial metric gtilde/χ; positivity is equivalent to physical SPD for Ω>0, but the magnitudes differ by Ω⁻².

Executable SHA256 is `580b6043f906ea37ae4870db2cd63d68f344c634c28319d88dafe53308d56005`; build receipt is `5ade5d3c80b68913301957a7093929a6793af5da8a9fc19c4215f140d7e49a72`. Native compact index is `a0db07fe7aad1a52afc9a609d1a0d5666f5e9e358eb51ba9ce96508cc20b252c`. The live global oracle confirms only the predicted P/Θ←Θ matrix changes at the reference. Short independent canonical checks pass. The [longer screen](hyperboloidal-long-window-and-resolution-audit.md) leaves gauge component amplification essentially unchanged at t6, so no long native pulse is admitted by this candidate.

Reproducible helpers, interval assessment, independent reviews, build/run commands, logs, binary64 audits and global histories are in the [compact archive](validation/hyperboloidal-live-damping-and-scri-hierarchy-experiments-20261009/README.md). Its 433 cataloged blobs total15,004,327 bytes; catalog SHA256 is `5ffe46220daccb65edf0f49ce9aa8e9b43f24f449dd821a33635644f7d50b724`. Large binaries, objects and state/matrix arrays are metadata-only. No stable pulse, closed exact-scri formulation, puncture evolution or black-hole acceptance follows.
