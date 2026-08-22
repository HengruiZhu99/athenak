# Cell-centered compatibility

## Authorities

- historical CC authority: `6daa774d7451dbc5f7cac640c6e32a6fd11de7f9`
  (tree `cbb702f4da954cf630da261790d5c21ef3142235`)
- candidate production source: `99a4eb5ba7713f7de73239cf75a27c1fb9ac6cbb`
- comparable configuration: GNU 13.3, Release, OpenMP+Serial, MPI off,
  double precision, built-in pgens, IrisK importer off, unit tests on

## Exact selector gate

`athena.z4c_cc_selector_equivalence` runs the same one-cycle Kerr half-plane
case with the centering selector omitted and with `grid_centering=cell`.
It requires exact history and timestep bytes, exact binary payload arrays, and
exact restart payload bytes after the parameter header. It passes.

Frozen hashes include:

- history: `4896c333ceda81d99cf1e4c15a28996d73c999c6222d4b83e770c9f4f4d0f598`
- timestep contract: `dad954f5938eea76aca74493ec5bd1ac8c66cdc67ac7ad24225988c19e5e3037`
- evolved final binary payload: `cba796c4feb0af6d89f15b0278ae2cf00f882f1347953079884f19519b5aeced`
- constraint final binary payload: `dd0e1b8dbf64d2014a5d49f26b7b3f188145c6c66571ccbb44e456a9dc5b6591`

## Historical authority gate

`z4c_cc_historical_equivalence_test.py` runs the same input with separately
built candidate and historical executables and compares the same payloads.
The result is `pass`; the retained JSON is
`evidence/local/cc-historical-equivalence.json`.

## Shared-source audit

The branch touches shared driver, mesh-refinement, pgen, output, and build
registration files. VC-native storage, transfer, topology, history, restart,
and output paths are selected before their device loops. Generic CC active,
stored, and buffer counts remain derived from `RegionIndcs`; native-VC counts
remain in the separate `Z4cGridLayout`/VC boundary classes. The deterministic
test-pgen schedule accepts CC explicitly, and three CC refine/derefine
companions pass in 2D Cartesian, 2D Cartoon, and 3D Cartesian.

The output test records the only intentional whole-file metadata delta: the
shared schedule no longer materializes an unused default parameter. The raw
post-parameter-header CC payload remains exact, with SHA-256
`b04b85bbb0b6f4227a1795ba507d7ddc8e19159c080bc4f5e6c4800ab7dc2618`.
No CC numerical payload changed.
