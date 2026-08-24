# Three-resolution convergence analysis

## Method

N128/N256/N512 share the same physical MeshBlock tree. N256 records every
accepted physical-time leaf hierarchy; N128 and N512 replay it using 16, 32,
and 64 cells per physical MeshBlock respectively.

History norms are compared at common central proper time. Fields are
interpolated to common coordinate times and sampled on the nested N128 vertex
lattice with spacing 0.25. The finest owning leaf supplies a shared vertex.
Errors are ring-weighted over the requested region:

```text
E128_256 = ||u128-u256||
E256_512 = ||u256-u512||
Q = E128_256/E256_512
p = log2(Q)
```

The offline evolved-minus-metric Gamma diagnostic uses O4 derivatives on that
common lattice. It is a comparison diagnostic, not the production RHS value.

Regions are trusted core, axis, block interiors, same-level seams,
coarse/fine neighborhoods, full domain, and the outer two common-grid layers.
The trusted core shrinks using the integrated maximum coordinate speed.

## Exact hierarchy control

- N128: 114/114 replay events exact through authority event 114, final tree
  checksum `163bcb25912a8344`.
- N512: 24/24 replay events exact through the event reached before its state
  failure, checksum `89f3599d5f78b40e`.

Native shadow diagnostics differ strongly with resolution, as expected for a
forced common tree; replay exactness does not imply that the tree is preferred
by each resolution.

## Initial and early behavior

At `tau_c≈0.1268`, before the catastrophic transaction, global history orders
are:

| C | H | M | Z |
|---:|---:|---:|---:|
| 5.84 | 3.58 | 9.44 | 7.64 |

Initial common-lattice orders include C `7.59`, H `3.78`, Z `7.59`,
Kretschmann `3.79`, and Gamma incompatibility `3.81`. Exactly zero symmetry
components have undefined order and are retained as NaN rather than assigned a
fictitious rate.

## Loss of convergence

Authority event 3 at `t=0.2979919497` removes six leaves and drops the maximum
logical level from 5 to 4. The global norms jump by increasingly large factors
with resolution. By `tau_c≈0.2497`, their effective orders are:

| C | H | M | Z |
|---:|---:|---:|---:|
| -3.26 | -3.57 | -2.62 | -0.54 |

Selected trusted-core field orders are:

| coordinate time | chi | K | Gamma incompat. x | C | H | M | Z | Kretschmann |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 1.05 | 0.47 | 1.08 | 0.63 | -0.64 | 0.33 | -0.07 | -0.04 |
| 0.50 | -0.45 | -0.08 | -0.65 | 0.31 | -1.17 | 0.75 | 1.33 | -0.19 |
| 1.00 | -1.06 | -1.02 | -0.62 | -3.14 | -1.36 | -3.33 | -0.64 | -2.79 |
| 1.50 | -1.63 | -1.08 | -1.84 | -0.44 | -0.72 | -0.74 | -1.07 | -0.04 |
| 2.00 | -1.44 | -1.43 | -1.69 | -2.98 | -1.29 | -3.01 | -2.29 | -3.05 |

At `t=1`, axis orders for C/H/M/Z and Kretschmann are
`5.41/3.75/7.84/7.72/5.44`, while coarse/fine-neighborhood orders are
`-3.18/-1.36/-3.64/-0.58/-1.84`. The outer-layer counterparts are mostly
positive. This excludes both the axis and outer boundary as the first bad
region and supports an AMR-interface disposition.

## Admissibility and synchronization

At common times through `t=2`, chi and all conformal-metric Sylvester pivots
remain positive. Sampled same-level and coarse/fine shared-node spreads are
zero. N512 subsequently develops a negative determinant/SPD pivot at
`t=2.4953913`, one fine spacing from a MeshBlock edge.

Therefore the data do not establish convergence. They establish early
resolution-worsening after AMR event 3.

