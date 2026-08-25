# Rout=128 SMR layout proof

Both domains use the same 4 by 8 root MeshBlock topology and the same
MeshBlock physical extents at corresponding logical locations.  Enlarging each
active coordinate extent by eight makes the new root spacing eight times
coarser.  Three dyadic minimum levels restore the old inner spacing exactly.

| family | old root dx | new root dx | static depth | new inner dx | old physical ceiling | new physical ceiling |
|---|---:|---:|---:|---:|---:|---:|
| N128 | 0.25 | 2.0 | 3 | 0.25 | 20 | 23 |
| N256 | 0.125 | 1.0 | 3 | 0.125 | 20 | 23 |
| N512 | 0.0625 | 0.5 | 3 | 0.0625 | 20 | 23 |

For every family,

```text
new_root_dx / 2^23 = old_root_dx / 2^20.
```

`mesh_refinement/num_levels` includes physical root level 0, so the old value
21 permits physical levels 0 through 20.  The new value 24 permits 0 through
23.  `z4c_amr/max_ref_lev` is therefore moved from 20 to 23.  The root logical
level is 3 for a 4 by 8 root topology, so the corresponding maximum logical
levels are 23 and 26.

The local N256 zero-step preflight independently reports 104 initial leaves,
physical levels 0 through 3 for the seeded tree, physical ceiling 23, and
logical ceiling 26.  The exact inner-state comparison establishes identical
vertex positions and initialization to deterministic precision.

The persistent radius floors conservatively enlarge the initially seeded box
footprint after AMR begins.  They do not alter any inner spacing, absolute
finest spacing, dchi threshold, or physical boundary condition.
