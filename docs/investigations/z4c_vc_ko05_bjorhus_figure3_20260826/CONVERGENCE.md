# Three-resolution convergence assessment

## Verdict

KO=0.5 does not establish a convergent collapse evolution. The matched-tree
campaign is close to fourth order at early times, but the order degrades first
outside `r<=4M` around coordinate time 8M and becomes nonconvergent around the
rho approximately 5M mode. Terminal amplitudes are strongly nonmonotone: N256
is catastrophically worse than both N128 and N512, while N512 is still badly
unstable.

The history quantities are squared inventories. Every reported amplitude order
uses

```text
q = 0.5 * log2(I_h / I_h/2).
```

## Representative proper-volume regional orders

The offline common-lattice diagnostic uses
`2*pi*rho*sqrt(gamma)*drho*dz` at matched `h=0.25`. It is useful for spatial
localization but filters scales below `h=0.25` and is not the production leaf
quadrature.

Each entry is `q128_256 / q256_512` for C, H, M, and Z.

| t | region | C | H | M | Z |
|---:|---|---:|---:|---:|---:|
| 2 | r<=4 | 4.07/3.97 | 3.87/3.75 | 4.19/3.98 | 4.09/4.01 |
| 4 | r<=4 | 3.77/3.96 | 3.74/3.98 | 3.76/3.97 | 3.84/3.93 |
| 6 | r<=4 | 3.53/3.90 | 3.47/3.90 | 3.56/3.90 | 3.74/3.89 |
| 8 | r<=4 | 3.58/3.88 | 3.50/3.87 | 3.61/3.87 | 3.93/3.95 |
| 8 | r<=8 | 1.36/-0.20 | 2.21/2.02 | 0.63/-0.64 | 2.12/3.23 |
| 10 | r<=8 | -0.57/-0.20 | -0.85/1.53 | -1.19/-0.63 | 1.59/1.49 |
| 12 | r<=4 | 3.39/0.74 | 3.14/0.62 | 3.55/0.90 | 3.88/1.13 |
| 14 | r<=8 | -1.32/-1.50 | -2.15/-0.46 | -0.89/-1.97 | -1.29/-1.05 |

The retained fourth-order behavior inside `r<=4M` while `r<=8M` and
`r<=12M` degrade is direct evidence that the first loss of convergence is
localized outside the central region. It is consistent with, but does not by
itself prove, an AMR/interface or bulk high-frequency mode around rho=5M.

## Terminal state

| quantity | N128 | N256 | N512 |
|---|---:|---:|---:|
| C squared inventory | 2.344e1 | 2.677e18 | 1.774e5 |
| max abs Kretschmann | 9.441e1 | 8.434e30 | 4.244e9 |
| max abs K | 1.339e1 | 8.382e7 | 3.572e2 |
| dt | 2.888e-8 | 5.615e-9 | 3.081e-9 |
| C maximum rho | 4.849 | 4.880 | 4.906 |

This nonmonotone ordering rules out a convergence claim. Higher resolution did
not simply cure the mode, although N512 was less catastrophic than N256 at the
forced common terminal time.

## Diagnostic measure

The Cartoon history reduction already uses the physical axisymmetric ring
measure. The jumps and late growth are not a fictitious collapsed-y
normalization artifact.

Figures and compact tables are under `analysis/history/` and
`analysis/regional_constraints/`.
