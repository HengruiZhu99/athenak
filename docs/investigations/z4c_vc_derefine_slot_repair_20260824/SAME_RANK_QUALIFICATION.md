# Same-rank native-VC derefinement qualification

## Expanded-test red finding

After the 2D A5/A6 staging repair passed, the expanded one-rank matrix found a
separate 3D family-selection defect. The 2D nonconstant O4/q6 and O6/q8 cases,
the original O2/q4 authority-map case, and the three-family constant case all
passed. All three 3D nonconstant order cases failed at the existing
coincident-sibling consistency gate.

The constant 3D diagnostic completed the transaction but reported three
families for a hierarchy containing exactly two octets. Its audit identified
old GIDs 0, 2, and 22 as family starts. The refined hierarchy shows that old
GID 2 is child `(lx1,lx2,lx3)=(0,1,0)` of the first octet, not its lower
child. In 3D, `Mesh::two_d` is false, so the family-start predicate checked
the `lx1` and `lx3` parity bits but omitted `lx2`. It therefore processed the
same octet again from a sliding, invalid source window beginning at old GID 2.

This is distinct from the A5/A6 destination-slot corruption. The exact repair
is to treat x2 as active in both 2D and 3D when selecting the all-lower child:

```text
(two_d || three_d) && (lower_child.lx2 & 1)
```

No tolerance or transfer rule was changed. The x2 active-dimension predicate
was repaired in both the production family selector and its independent audit.

## Green matrix

Command:

```text
ctest --test-dir build/vc-derefine-release-openmp \
  -R '^athena\.z4c_vc_multi_family_derefine' \
  --output-on-failure --timeout 180
```

All seven focused one-rank tests pass:

| dimensions/state | bulk/transfer | families | old/new leaves | slot shifts |
|---|---:|---:|---:|---:|
| 2D nonconstant | O2/q4 | 2 | 38/32 | 0, -3 |
| 2D nonconstant | O4/q6 | 3 | 41/32 | 0, -3, -6 |
| 2D nonconstant | O6/q8 | 3 | 41/32 | 0, -3, -6 |
| 2D constant | O2/q4 | 3 | 41/32 | 0, -3, -6 |
| 3D nonconstant | O2/q4 | 2 | 30/16 | 0, -7 |
| 3D nonconstant | O4/q6 | 2 | 30/16 | 0, -7 |
| 3D nonconstant | O6/q8 | 2 | 30/16 | 0, -7 |

Every reconstructed parent matches the independent coincident-node oracle at
A5 and A6 for all 25 variables. Every `a5_modified_live_old_gids` and
`a6_bad_unaffected_old_gids` list is empty. The executable SHA-256 for this
matrix is
`d55b477cdf63467749278194e768717f56b65ee5ef811d8a99cd877b47465987`.

This matrix covers stationary and left-moving parent slots, separated
families, and adjacent refined-family source runs. A same-rank move-right case
requires a load-balance partition change and is covered with the MPI ownership
matrix rather than synthesized in one-rank storage. A mixed refine/derefine
transaction remains to be added.
