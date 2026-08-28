# Published sphGR reference gauge

## Authority chain

The Figure-3 comparison in [arXiv:2607.10843v1](https://arxiv.org/abs/2607.10843)
states that sphGR evolves BSSN with a shock-avoiding Bona-Masso slicing and
vanishing shift.  The paper does not print the gauge equation or parameter in
its Figure-3 section, so the mapping is completed from the cited sphGR gauge
literature:

- [Baumgarte, Gundlach, and Hilditch (2023), arXiv:2303.05530](https://arxiv.org/abs/2303.05530)
  gives the sphGR Bona-Masso equation and identifies the shock-avoiding family.
- [Baumgarte and Hilditch (2022), arXiv:2207.06376](https://arxiv.org/abs/2207.06376)
  defines the family, discusses `kappa=1`, and explicitly permits negative lapse.
- [Baumgarte, Gundlach, and Hilditch (2026), arXiv:2606.27431](https://arxiv.org/abs/2606.27431)
  documents the current sphGR vacuum-collapse choice `f(alpha)=1+1/alpha^2`,
  initial lapse `alpha=1`, and vanishing shift.

The locally frozen Figure-3 source is
`/home/hzhu/Desktop/research/gr/collapse/artifacts/axisymmetric_cartoon_z4c_2026-08-10/continuation/literature/arxiv2607_10843v1_freeze_2026-08-11/original/2607.10843v1.tar`.

## Continuum system used for this campaign

With the standard numerical-relativity sign convention

```text
K_ij = -(1/2) L_n gamma_ij,
K = gamma^ij K_ij,
```

the Bona-Masso equation is

```text
(partial_t - beta^i partial_i) alpha = -alpha^2 f(alpha) K,
f(alpha) = 1 + kappa/alpha^2,
kappa = 1.
```

Therefore the evolved equation is

```text
partial_t alpha = beta^i partial_i alpha - (alpha^2 + 1) K,
beta^i = 0.
```

The Figure-3 run starts with

```text
alpha(t=0) = 1,
beta^i(t=0) = 0.
```

There is no lapse floor, clipping, absolute value, or positive regularization
in the lapse RHS.  The lapse is allowed to cross zero and become negative.

### Sign-source qualification

The 2023 sphGR paper prints the standard minus-sign equation above.  The 2026
Nakamura paper prints a plus sign while not defining a different extrinsic-
curvature convention in that section.  Its stated `f(alpha)`, `kappa=1`, unit
initial lapse, zero shift, and negative-lapse behavior agree with the earlier
sphGR description.  This campaign follows the explicit 2023 sphGR equation
and AthenaK's standard `K_ij=-(1/2)L_n gamma_ij` convention.  The isolated plus
sign in the 2026 source is recorded as a literature ambiguity, not silently
used to reverse AthenaK's curvature convention.

## Gauge characteristic speed

In a spatial direction with physical inverse metric component
`gamma^{nn}`, the Bona-Masso characteristics relative to the shift have speed

```text
v_gauge = |alpha| sqrt(f(alpha) gamma^{nn})
        = sqrt((alpha^2 + 1) gamma^{nn}).
```

The physical light-cone speed magnitude is

```text
v_light = |alpha| sqrt(gamma^{nn}).
```

The absolute value is a CFL magnitude only.  It is not inserted into the
lapse evolution equation.

## Frozen Figure-3 physical data

```text
A = -0.047
rho0 = 5
z0 = 0
sigma_rho = sigma_z = 1
```

No time, amplitude, or curvature rescaling is permitted in the comparison.
