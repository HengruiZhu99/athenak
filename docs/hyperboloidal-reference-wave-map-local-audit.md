# Physical reference wave-map gauge: local implementation checkpoint

This checkpoint implements a private alternative gauge and complete higher reference jets. The gauge passes local nonlinear algebra and dual derivative checks. Both original broad-coordinate Einstein-tangent tests retain their failed outer geometry gates. There is no gauge adoption, evolution, full characteristic acceptance, or exact-scri closure in this checkpoint.

The implementation branch is `z4c_hyperboloidal_layer`. Production `src/` and root CMake remain byte-identical to implementation `27c19d20696ea6dd4704032c51dfd026218f64f2`. The later acceptance target includes a substantial angular Minkowski gauge disturbance followed by a single hole surviving the inner wormhole-to-trumpet transition, with the Minkowski hyperboloidal reference retained throughout.

## Gauge and storage conventions

Keep physical `P=Kphysical-2ThetaPhysical`, fixed time-independent Omega, and `Lambda=GammaTilde+2*gtildeInv*Zcov`. Let `g=Omega^-2*bar(g)` and let `ghat` be the physical Minkowski reference from the same layer height and radial compactification. The proposed physical reference wave-map equation is

```
g^bc (Gamma[g]^a_bc-Gamma[ghat]^a_bc)+2 Zphysical^a = 0.
```

Define `Hhat^a=bar(g)^bc Gamma[ghat]^a_bc`, using the live inverse four-metric. The equivalent conformal source is

```
Fbar^a = Hhat^a - 2 bar(g)^ai Omega_i/Omega,
GammaBar^a + 2 Zbar^a = Fbar^a.
```

Its equivalent form using the conformal reference connection contains the full four-dimensional trace `s=bar(g)^bc bar(ghat)_bc`, not a spatial trace. The physical-P lapse and shift rows are

```
D0 alpha = -(alpha^2 P + alpha beta^i Omega_i)/Omega
           - alpha^3 Hhat^0,
D0 beta^i = alpha^2 [chi Lambda^i
             + gtildeInv^ij(chi_j/2-chi alpha_j/alpha)
             + 2 barGammaInv^ij Omega_j/Omega
             - Hhat^i-beta^i Hhat^0].
```

Here `alpha` is the stored conformal lapse and `D0=partial_t-beta^i partial_i`. The private templated helper factors deviations and exact stationary reference identities before assembling both lapse and shift pole terms once. It does not subtract a numerical reference RHS, evaluate Q by a full cancellation, add an Omega floor, or divide assembled gauge rows by the live lapse.

For the stationary embedding `Yhat^0=t+h(r)`, `Yhat^I=x^I/Omega`, all temporal reference connection slots vanish. With `G=Omega Gamma[ghat]`, the implemented finite-Omega identities are

```
G^k_ij = -delta^k_i Omega_j-delta^k_j Omega_i
         -x^k Omega Omega_ij/L,
G^0_ij = L(Omega b'-b Omega') n_i n_j/alphaHat^3
         + b(delta_ij-n_i n_j)/(alphaHat*r).
```

The origin uses its isotropic core limit. Independent comparators construct the full embedding Jacobian/Hessian and, separately, the stationary ADM four-metric connection. The exact core is globally harmonic for this diagnostic candidate. A moving-puncture inner blend has not been selected; local tiny-positive-lapse arithmetic checks do not establish puncture hyperbolicity or the requested black-hole transition.

The wave-map construction is our candidate for the actual C0/physical-P kernel. The reference connection approach is established in the [wave-gauge formulation](https://arxiv.org/pdf/1711.00195); preferred conformal gauge and hyperboloidal gauge sources are discussed by [Zenginoglu](https://arxiv.org/pdf/0808.0810). Their results do not prove stability or regular closure for this implementation.

## Complete higher reference derivatives

The private backend supplies Omega through fourth radial order and the configuration/reference quantities through the third orders required by the Einstein coordinate lift. It computes tiny cutoff complements and derivative tails separately. No absent fourth derivative of L is consumed. The origin remains exactly Cauchy and avoids radial division.

Release and ASan/UBSan Debug reference outputs agree byte-for-byte over 38 radii, including endpoints and extreme cutoff tails. Agreement with previously consumed reference jets is at most `3.09e-14` scaled. Independent 100/130-digit radial checks pass all 2,052 comparisons; the worst native scaled error is `4.21e-12`. The original failed tail-classifier attempt remains archived. Its correction changes the classifier, preserves source outputs and thresholds, and retains the failure.

An additive root readback compares the saved owner C++ Cartesian exports at eight off-axis points to independent radial formulas and closed Cartesian chain rules. All 53 fields, including physical lapse/metric/K and stored fields, are checked through all 20 ordinary Cartesian multiindices of total degree at most three: 8,480 comparisons per precision. Native scaled error is `3.5684e-13` against `2e-10`; 100/130-digit agreement is `6.2028e-98` against `1e-80`. This readback performs no source/API calls and does not override a coordinate-lift failure.

## Local gauge results

The fixed grid covers four height parameters, two layer widths and a pure-CMC control, 21 radii through `.99999`, and three orientations. There are 756 reference and 756 finite off-constraint comparisons, 132 core controls, 480 tiny-positive-lapse gauge-only controls, and 1,440 determinant/trace-compatible dual directions.

| Local check | Maximum |
|---|---:|
| embedding scaled connection | `4.1389e-13` |
| independent ADM scaled connection | `6.9944e-15` |
| full conformal connection transformation | `5.6344e-15` |
| two physical/conformal source forms | `7.6594e-15` |
| actual C0 full-Z conformal source identity | `7.5198e-15` |
| factored reference gauge | exactly zero |
| raw reference gauge, absolute | `9.2562e-11` |
| raw C0 reference geometry, absolute | `1.3323e-10` |
| final gauge directional FD error | `1.0662e-7` |
| final scaled-source directional FD error | `8.0910e-9` |

The final FD levels pass the predeclared `2e-7` gate. Release and Debug ASan/UBSan outputs match exactly. The accepted receipt is `45f789e122a78a3c3b651dcc1317d83d7f09cb4931032ca841952ece881ed8d2`; Release executable SHA256 is `78349cd4bc458e35bdc52a0ba7a6578eaac8dc39f6d235f22597d3ec0b457416`, Debug is `77f4995d9319d7e2a7162718774330593c47797d19957827838fd3fd75455dec`. Exact source/flags/commands, 374 input pins and 1,053/1,055 compiler dependency hashes are saved. Root independently rehashed these inputs and dependencies and checked all saved thresholds.

The first direct embedding comparator failed through cancellation in binary64. A long-double template compile failure and a subsequent long-double numerical failure are retained; Apple arm64 long double has binary64 precision. Only independent comparator arithmetic changed to FMA double-double, with sanity residual `2.06e-33`; helper/grid/tolerances stayed unchanged. This arithmetic is not rigorous interval arithmetic. The final harness also explicitly rejects nonfinite comparison operands and reproduces the prior passing numeric output.

## Retained Einstein-coordinate failures

The independent active-Lie lift uses four freely prescribed radial coordinate displacement/velocity envelopes and the complete fixed-Omega reference. Algebraic normals, physical constraints, core rates, actual gauge attribution and raw22/free20 binding pass. Nevertheless the original full-radius geometric point-action gate fails: the entrywise threshold is `5e-10`, while the first maximum is `2.747e-7` at `.995`.

An independently reviewed conformal factoring of the same lift lowers that maximum to `2.9802e-8`, still a failure. Its fixed five-level FD sequences all pass `2e-7`; Release and ASan/UBSan outputs match. No `.98` subset is retroactively accepted. These two failed controls and their as-built executables remain frozen. A new bounded physical-inertial witness family is a separate experiment outside this checkpoint.

## Scri and later black-hole scope

The additive radial identity for this source is

```
boxBar Omega = rho4[Omega''-Omega' L'/L+4 Omega'^2/Omega]
               + T4 Omega Omega'/(rL) + 2 Omega_i Zbar^i,
rho4=n_i n_j bar(g)^ij,
T4=(delta_ij-n_i n_j)bar(g)^ij.
```

For Einstein solutions, bounded T4 and `rho4=O(Omega^2)` imply `boxBar Omega=O(Omega)`. The reference recovers its exact `Omega*What` identity. General live fields need not satisfy `boxBar Omega=Omega*What` at finite Omega. No preservation or constraint/shear/Theta closure has been proved. The off-constraint Z term remains explicit.

The later black-hole physical initial foliation must also be derived consistently. Applying the Minkowski height slope directly to Schwarzschild time leaves a `4M/R` physical radial metric term and produces a compactified radial metric growing as `1/Omega`. The physical initial height therefore needs appropriate mass-dependent outgoing asymptotics, while the requested reference stays Minkowski. This observation does not select wormhole initial data, an inner gauge blend, or a trumpet fixed point.

## Reproduction and remaining gates

The [compact archive](validation/hyperboloidal-reference-wave-map-local-experiments-20261009/README.md) contains 414 byte-preserved small files, source and all failed attempts, with large arrays/comparisons/binaries represented by hashes and local origins. Catalog SHA256 is `e92403c8c05f57c1f1228e686e372838d6b61c1c5d730deba495e2d4cf5850fa`. Run its `verify_archive.py` with `python3 -B` for saved-artifact checks. Compact verification does not replay omitted comparison arithmetic or executables. Original frozen capsules remain unchanged.

Next gates are actual constrained20 principal completeness, independent coordinate-wave accelerations, finite-amplitude exact-flat RHS consistency, lower-order/constraint growth, and actual native angular evolution. Exact-scri regular closure and boundary stability remain unresolved. Production code has not changed, so previously passing production regressions were not rerun for this private checkpoint.
