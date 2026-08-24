# Native-VC multi-family derefinement red/green record

## Base red result

The production-path regression was run against the unmodified same-rank
`DerefineVCSameRank` staging behavior. The executable was a Release OpenMP
build with MPI disabled, so the test exercised exactly one rank without the
local OpenMPI singleton launcher.

Command:

```text
ctest --test-dir build/vc-derefine-release-openmp \
  -R '^athena.z4c_vc_multi_family_derefine$' \
  --output-on-failure --timeout 90
```

Result: return code 8; one focused test failed in 0.11 seconds as expected.

The independently generated slot audit reproduced the authority event map:

- old children 16:19 produced parent 16 in slot 16 and remained correct;
- old children 29:32 produced parent 26 in destination slot 26 during A5;
- A5 therefore changed still-live old GID 26 before its A6 copy to new GID 23;
- A6 copied the clobbered slot 26 into new slot 23;
- A6 then copied old lower-child slot 29 over the correct parent in new slot 26.

The first A5 live-source mismatch was variable 0 at `(k,j,i)=(0,2,2)`,
with absolute difference `1.7100119219515264e-06` and ULP distance
`13839494550`. The first A6 parent mismatch was variable 0 at
`(0,2,3)`, with absolute difference `7.7654515262537416e-08` and ULP
distance `349724846`.

Evidence:

- `red_base_slot_audit.json`
- audit SHA-256: `a112680f30119b671228a5e26751ab8d5e3542a05712a67da09b1f34ebfbf6ef`
- executable SHA-256: `033c627e3c93ad30dd987477dc0b1462cd593f97683e5580d80d7789c1b2a507`

This is direct evidence of the predicted A5/A6 slot-lifecycle corruption,
not a norm-only or constant-state discriminator.

## Green result

Pending the minimal staging repair.
