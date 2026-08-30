# Source-scoped per-kernel split: negative gate

Aurora job `8791895` ran on `x4514c1s6b0n0` from exact commit
`580298eeaf6405e467485e7b629f270b8ae79779`, with Kokkos
`6739bc623081648af9e752b616d9671527922cbf` and the frozen input SHA-256
`6d483ded11b70d640f4a166fd21757f956802622d4e3994ceda71ae8649235eb`.

The compile database showed one and only one flagged source:

```text
src/ref_gh/ref_gh_calcrhs.cpp
```

The executable SHA-256 was
`c93fef980f015c08f9c1cebc43d18da028a69d4e094eef571cc48f0c944669e5`.
The 12-rank gate classified `FAIL_LEVEL_ZERO`, with four `NotPresent` writes,
status 143, and no positive-time history. All twelve ranks had reported the
final diagnostic history fence before the asynchronously surfaced fault.

Conclusion: splitting only the RHS source object is insufficient. This does
not invalidate the earlier global per-kernel pass; it narrows the required
scope to the final device-link image. No scientific stability claim follows.
