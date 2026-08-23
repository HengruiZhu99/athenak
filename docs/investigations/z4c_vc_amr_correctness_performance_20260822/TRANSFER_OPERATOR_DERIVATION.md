# Native vertex-centered AMR transfer

## Geometry and restriction

A dyadically nested coarse vertex is a fine vertex.  With active starts
`i_s, I_s`, the coincident map is

```text
i = i_s + 2 (I - I_s)
```

and likewise in each active transverse direction.  Restriction is therefore
point injection,

```text
U(I,J,K) = u(i_s+2(I-I_s), j_s+2(J-J_s), k_s+2(K-K_s)).
```

Collapsed dimensions use their unique stored index.  There is no averaging,
coordinate search, or interpolation at a coincident vertex.  Consequently a
coarse value prolongated to a coincident child and restricted again obeys
`R P = I` exactly (bitwise, absent an intervening evolution operation).

## Midpoint prolongation

For fine offset `n = i-i_s`, even `n` is injected.  For odd `n`, the containing
coarse interval is selected with mathematical floor division:

```text
I_left = I_s + floor(n/2).
```

This distinction matters for negative odd ghosts: `floor(-1/2)=-1`, whereas
C++ integer division would produce zero.

The centered one-dimensional midpoint weights are:

| q | stencil weights, left to right |
|---:|---|
| 4 | `[-1, 9, 9, -1] / 16` |
| 6 | `[3, -25, 150, 150, -25, 3] / 256` |
| 8 | `[-5, 49, -245, 1225, 1225, -245, 49, -5] / 2048` |

They are symmetric, sum to one, and reproduce polynomials through degree
`q-1`.  Multidimensional prolongation is the tensor product of only the
noncoincident active directions.  Coincident and collapsed directions use the
identity.

## Why transfer order exceeds bulk order

Let the point interpolation error be `e = O(h^q)`.  An interface-consumed
first derivative sees `D e = O(h^(q-1))`; a second derivative sees
`D2 e = O(h^(q-2))`.  Because vacuum Z4c contains second spatial derivatives,
maintaining bulk truncation order `p` requires

```text
q - 2 >= p,
```

hence:

| bulk Z4c order p | midpoint transfer order q |
|---:|---:|
| 2 | 4 |
| 4 | 6 |
| 6 | 8 |

This mapping is implemented by `TransferOrderForSpatialOrder` in
`src/mesh/vertex_amr.hpp`.

## Coarse halo requirement

For fine ghost width `g` and midpoint order `q`, the required coarse-cache
ghost width is

```text
g_c = max(g, floor((g-1)/2) + q/2).
```

With the standard `g=4`, q8 needs `g_c=5`.  Centered interpolation is allowed
only when each active coarse MeshBlock direction can supply that halo in one
hop.  Otherwise construction fails closed and asks for a larger MeshBlock.
This avoids silent one-sided closure and stale multi-hop cache values.

The wider `g_c` is a persistent coarse-cache communication requirement.  It is
not the halo copied from an old fine MeshBlock when constructing a newly
refined child.  That migration uses only

```text
r = q/2 - 1,
```

so an N16/q8 child sends its eight-interval half plus three vertices on each
side.  Source and target cardinalities are both 15 in each active direction.
Using `g_c=5` here would request an upper source endpoint one past the allocated
fine array; this distributed-only defect was fixed in `480de5f7`.

## Shared-vertex synchronization

Every duplicate storage location is grouped by an integer canonical physical
vertex key.  Contributors are deterministically ordered.

- If all contributors are at the same level, their arithmetic mean is written
  back symmetrically to every copy.
- If levels differ, only contributors at the finest level enter the mean.  The
  result is written to every coincident copy, so the fine solution is
  authoritative and the coarse value is its exact injection.
- Hanging fine vertices are reconstructed and never treated as independent
  contributors.
- Physical-boundary and axis vertices follow explicit boundary/parity rules.

This is deterministic with respect to MeshBlock enumeration.  The present MPI
implementation uses a global gather; correctness is qualified before any
performance replacement is considered.

## Verification

Durable unit tests cover weights, moments, symmetry, tensor products,
collapsed dimensions, face/edge/corner bounds, exact injection, `R P = I`,
positive finite `chi`, parity, and deterministic topology grouping.  Evolution
evidence is summarized in `CORRECTNESS_MATRIX.md`.
