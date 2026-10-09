# Physical reference wave-map gauge: separate finite-Omega candidate

This is a new research candidate, not a modification or acceptance of C0. It uses the same Minkowski hyperboloidal reference and external time-independent Omega. No black-hole reference subtraction is permitted. The initial question is whether an analytically controlled Einstein-sector gauge can guide the lower-order source choice. Boundary closure, Z4c subsidiary behavior, inner Bona-Masso blending and exact-scri regularity remain separate requirements.

Let g=Omega^-2 barg and ghat=Omega^-2 barghat denote physical four-metrics, signature -+++. Impose the physical wave-map gauge

```
g^{bc} (Gamma[g]^a_bc-Gamma[ghat]^a_bc) + 2 Zphysical^a = 0.
```

The wave-map connection difference is a tensor. Its established geometric definition is used here; the conversion and proposed tests below are our derivations. Hintz and Vasy, section 3 equation 3.1, give the background-connection gauge [primary paper](https://arxiv.org/pdf/1711.00195). General metric-dependent conformal wave sources and the distinct preferred-conformal-gauge condition are discussed by Zenginoglu, sections 2–3 [primary paper](https://arxiv.org/pdf/0808.0810). Neither paper proves stability of this AthenaK Z4c discretization or the proposed interior blend.

At fixed Omega, with s=barg^{bc} barghat_bc, define

```
Fbar^a = barg^{bc} Gamma[barghat]^a_bc
         - (4 barg^{ai}-s barghat^{ai}) Omega_i/Omega.
```

The gauge is equivalently barg^{bc} Gamma[barg]^a_bc+2 Zbar^a=Fbar^a. This follows by contracting the complete conformal connection transformation; the four in the numerator is the spacetime dimension. At the reference s=4 and the additional numerator vanishes. Fbar is algebraic in the live four-metric and first reference derivatives. It can contain off-reference 1/Omega terms. These are derived terms, not a claim of a regular exact-scri extension. Finite-Omega source tests precede any asymptotic or nonlinear adoption.

An alternative implementation uses the physical reference connection directly:

```
Hhat^a = barg^{bc} Gamma[ghat]^a_bc,
Fbar^a = Hhat^a - 2 barg^{ai} Omega_i/Omega.
```

Since the Minkowski reference is built by Y^0=t+h(R), Y^I=x^I/Omega, its physical connection satisfies Gamma[ghat]^a_0b=0 and Gamma[ghat]^a_ij=(dY)^-1{}^a_A partial_i partial_j Y^A. In particular, no height integral is needed to compute the connection: h_r=bL/(alpha_hat Omega²) and its derivative suffice. A point implementation must verify this embedding formula against the ADM-derived four-connection.

The stored variables have P=Kphysical-2Thetaphysical, Q=(P-3 omega_n)/Omega, omega_n=-beta^i Omega_i/alpha. Write gammaBarInv^{ij}=chi gtildeInv^{ij}. The exact gauge rows are

```
D0 alpha = -alpha² Q-alpha³ Fbar^0,
D0 beta^i = alpha² [chi Lambda^i
             + gtildeInv^{ij}(partial_j chi/2-chi partial_j alpha/alpha)
             - Fbar^i-beta^i Fbar^0].
```

Equivalently, retaining physical P storage,

```
D0 alpha = -(alpha² P+alpha beta^i Omega_i)/Omega-alpha³ Hhat^0,
D0 beta^i = alpha² [chi Lambda^i
             + gtildeInv^{ij}(partial_j chi/2-chi partial_j alpha/alpha)
             + 2 gammaBarInv^{ij} Omega_j/Omega
             - Hhat^i-beta^i Hhat^0].
```

The lapse/shift identities follow from GammaBar^0+2Zbar^0=-Q/alpha-D0alpha/alpha³ and GammaBar^i+beta^i GammaBar^0=GammaBarSpatial^i-gammaBarInv^{ij}partial_j logalpha-D0beta^i/alpha². The stored Lambda convention is GammaTilde^i+2 gtildeInv^{ij}Z_j. Thus no constraint term may be silently omitted when checking the two implementations. Do not evaluate Q by subtracting full background quantities in a production implementation. A future finite-Omega prototype must factor reference deviations in each pole and measure cancellation separately.

At the exact Cauchy core, ghat is inertial Minkowski. This candidate has harmonic slicing and harmonic integrated shift (f=q=mu=1, epsilon_alpha=1, epsilon_chi=1/2), not the current inner f=3, mu=3/8 gauge. Its principal source is algebraic in the live metric, so the harmonic principal block is unchanged. A moving-puncture inner blend is not selected by this preparation and must undergo the actual-symbol and black-hole-transition checks later.

For a pure-coordinate perturbation delta g=Lie_xi ghat, linearizing the connection-difference gauge on the Einstein sector gives nablaHat^b nablaHat_b xi^a+RicHat^a_b xi^b=0. The physical reference is flat. In reference inertial components Xi^A=(partial_a Y^A)xi^a, this is four scalar Minkowski wave equations. For compact finite-energy data with a suitable causal boundary treatment, their physical Killing energy does not exhibit exponential growth. This statement concerns the analytic Einstein-sector gauge, and supplies no conclusion about off-constraint Z4c modes, a projected coordinate discretization or the existing spherical mask.

A useful finite-amplitude manufactured oracle follows from the same geometry. Let X^A be physical inertial coordinates and choose four functions Y^A(X)=X^A+epsilon a^A phi(X) with box_eta phi=0. Where this map is invertible, set Y^A=Yhat^A(x), invert for X(x), and pull back eta. Then the metric is exactly flat and satisfies the physical reference wave-map gauge: the reference inertial coordinates Yhat^A(x) are harmonic scalars in the physical metric. For phi=(F(T-R)-F(T+R))/R, the regular center limit is -2F'(T); spatial a^A gives angular gauge content. This is a route to a nonlinear gauge oracle, not an evolution result. Invertibility, complete ADM/storage jets and source residuals require explicit checks. A plane-wave oracle can validate local formulas but is not acceptable evidence for a localized smooth hyperboloidal pulse at scri.

Before any actual source query: independently verify both conformal contractions and ADM gauge identities on nontrivial positive metrics, compare embedding/ADM reference connections, compare pure-coordinate accelerations with the four inertial wave equations, and derive the exact-core oracle. No finite matrix, boundary condition, propagator or native equation adoption is authorized by this source document alone.
