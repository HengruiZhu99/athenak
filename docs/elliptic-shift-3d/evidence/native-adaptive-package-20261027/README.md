# Regular native adaptive AMR evidence capsule

This capsule covers 22 native adaptive phase directories, including failed
attempts. It retains the actual Serial Driver controls, independent exact-RK
replays, restart and ADM checks, refined/coarsened region diagnostics, source
archives, compiler/link receipts, and default-path regressions. The implemented
option is experimental and restricted to regular Minkowski/angular lapse gauge
pulses. Wormhole adaptation and unsupported native configurations remain guarded.

`manifest.json` lists the per-directory Git-facing evidence. `chunk-index.json`
records every scientific file, its size, SHA256 and ordered chunks. The eight
`data-chunks.tar.gz.partNNN` files concatenate to the exact lossless gzip capsule;
each is at most 32 MiB. The unsplit capsule, individual gzip copies, executables,
object files and expanded private source trees remain local and ignored.

Verify all shipped data and archived source files, using only the split archive:

```sh
python3 tst/elliptic_shift_3d/rehydrate_native_adaptive_evidence.py \
  docs/elliptic-shift-3d/evidence/native-adaptive-package-20261027 \
  --sources --check-only --force-parts
```

Omit `--check-only` to restore exact originals into their recorded repository
paths, or supply `--destination /absolute/new/tree`. Existing differing files are
rejected. `rehydration-check.json` records verification of 564 scientific files
and 2,999 source members. Raw data are retained for every cell and region.

Rebuild using the frozen source/configuration and recorded compiler/link commands.
Numerical controls use the frozen preflight binary identified in their manifests.
The later current shared binary reproduces its small adaptive snapshots and all
seven accepted sidecars byte for byte. Native inputs do not expose the parent's
optional affine beta operator; these tests use its default OFF path. Independent
reference builds also use the already committed native static source/vendor
capsule identified in the qualification receipts. No shipped executable is needed
to audit the source, data, or independent reproduction commands.

The three lost qualification attempts are preserved explicitly: pre-regrid versus
post-regrid snapshot labeling, cumulative-time subtraction versus actual stored
RK timestep, and a compile error in the early cyclic-zoom guard. Their corrected
controls retain the original strict acceptance criteria.

See `../../native-adaptive-regular-integration.md` for implementation details,
exact test cases, costs, numerical limits and unsupported configurations.
`ready-for-commit.json` separately inventories the complete owned source and
Git-facing evidence; its own hash is reported to the parent.
