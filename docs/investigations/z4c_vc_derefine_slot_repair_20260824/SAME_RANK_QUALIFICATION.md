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

No tolerance or transfer rule should be changed. Green results and the rest
of the matrix are pending that predicate repair.
