# Revised 20M puncture campaign mesh preflight

Date: 2026-08-17

Both revised controlling inputs were evaluated with AthenaK mesh-only mode.
They independently produced the same fixed-refinement tree:

```text
Root grid = 4 x 4 x 4 MeshBlocks
Total number of MeshBlocks = 232
Number of physical levels of refinement = 3 (4 levels total)
Physical level 0: 56 MeshBlocks
Physical level 1: 56 MeshBlocks
Physical level 2: 56 MeshBlocks
Physical level 3: 64 MeshBlocks
```

The leaf-block coordinate envelopes at physical levels 0--3 are respectively
`[-32,32]^3`, `[-16,16]^3`, `[-8,8]^3`, and `[-4,4]^3`.  The deepest level
therefore covers the required `[-2,2]^3` cube, with block-aligned over-coverage
to `[-4,4]^3`.  For 32, 48, and 64 active cells per MeshBlock, the finest
cell widths are exactly `1/16`, `1/24`, and `1/32 M`.

Commands (with the matching FO-GH and Z4c executables) were:

```text
athena -m -i inputs/fo_gh/fo_gh_puncture_compare_smr.athinput
athena -m -i inputs/z4c/onepuncture/z4c_onepuncture_compare_smr.athinput
```

The FO-GH MPI-enabled local executable printed the complete correct summary
and wrote all 232 blocks, then segfaulted during mesh-only shutdown.  The Z4c
executable completed normally.  This shutdown defect is retained as a preflight
failure to diagnose; it does not change the independently matching tree data.

The previous `[-128,128]^3`, 3,200-block campaign was superseded before its
coarse FO-GH run reached `t=2M`; allocation 57195786 was cancelled.  None of
that partial run is evidence for the revised campaign.
