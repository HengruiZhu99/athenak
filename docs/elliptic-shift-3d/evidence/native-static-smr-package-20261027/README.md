# Native static-SMR evidence capsule

The manifest enumerates12 evidence directories, including failed receipt and
post-run assertion attempts. Source/header trees are lossless deterministic
archives. The content-addressed data capsule reconstructs175 scientific output
files byte-for-byte; no floating-point encoding or field interpretation changes.
All raw originals and per-file gzip copies remain local. Git excludes those raw
copies, executable/object files and private source/build trees.

Verify without writes:

```
python3 tst/elliptic_shift_3d/rehydrate_native_static_evidence.py docs/elliptic-shift-3d/evidence/native-static-smr-package-20261027 --sources --check-only
```

Remove `--check-only` to restore raw scientific files and archived source trees.
`--destination /tmp/native-evidence-copy` restores a separate copy instead. The
`local-gzip-files.json` receipts describe local-only gzip alternatives; their
bytes are represented in the capsule and are not shipped separately. Use this
native capsule tool, not the generic per-file gzip restorer.

MachO executables and objects are intentionally omitted. Rebuild AthenaK from the
recorded source/configuration before replaying executable commands. The final
shared build/control manifests record the current validated binary hash; earlier
frozen-header relink results are separate and retain their own provenance.

Read `docs/elliptic-shift-3d/native-static-smr-integration.md` for validation scope.
Static native state/RK/restart/ADM mechanisms pass. Wormhole scientific gates
still fail; dynamic native AMR, MPI/GPU and native shell ownership remain guarded.
