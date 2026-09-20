# Static-SMR local three-cycle preflight

The separately authorized preflight passed on eight local CPU MPI ranks using immutable binary SHA256 `67ff395e3c7af43425ef13fe4f0ebecef00256db9450f6ac912f5ab39620f1f8`. Both cases use the preferred232-block16³ static mesh,29 blocks/rank, levels0–3, centered M=R0=1 trumpet and radial outer layer. **No longer evolution or Aurora job was started.**

| Case | Ending | Actual dt | Total elapsed | Evolution seconds/cycle |
|---|---|---:|---:|---:|
| Zero residual | cycle3, t=.075M | .025M | 29.46s | 9.030s |
| Compact lapse pulse | cycle3, t=.075M | .025M | 28.67s | 8.921s |

Total elapsed includes setup and per-rank outputs; seconds/cycle is measured from the cycle0–3 progress records. These are CPU measurements, not a new GPU throughput calibration. Both applications exited0 on the requested cycle limit, with finite histories and no primitive-recovery or invalid-state messages.

The zero checkpoint has **exactly zero residual entries over all25 fields, all232 blocks and all active/ghost cells**. The pulse's initial checkpoint has only lapse residual nonzero, with peak9.78948e−9; metric/extrinsic-curvature/Theta residuals are initially exactly zero. At the final pulse checkpoint, active maxima are:

| Residual | Maximum absolute value |
|---|---:|
| lapse | 9.58446e−9 |
| Khat | 7.36675e−9 |
| chi | 8.43918e−11 |
| Theta | 1.70602e−10 |
| shift component | 7.44114e−11 |

This demonstrates a nonzero geometric/gauge response rather than a reset/frozen evolution. It **does not demonstrate perturbation stability** over.075M. The raw finite-difference background Hamiltonian diagnostic is already about1.32972e−4 at initialization near r=1.066M; exact residual preservation must not be confused with exact discrete ADM Hamiltonian cancellation.

## Separate level-aware validator

`../check_smr_trumpet_checkpoint.py` is independent of the existing uniform-only checker and does not broaden it. It validates the little-endian double MHD+residual-Z4c layout, common cohort headers/time/cycle, exact payload sizes, the fixed equal-cost contiguous gid partition, finite entire payloads, and raw full alpha/chi/conformal-metric Sylvester minors including every ghost cell. It uses no floors, clipping or repairs.

For each stored leaf at logical location`(lx,ly,lz,level)`, relative level`ell=level-root_level` gives

```
dx_leaf = (domain_max-domain_min) / root_nx / 2**ell
leaf_min = domain_min + logical_location * block_nx * dx_leaf
x(i) = leaf_min + (i-nghost+.5) * dx_leaf
```

All232 leaf bounds and spacings match the independently generated AthenaK mesh-only output exactly, including every refinement level. Domain coverage and no overlapping leaf interiors are also checked. Raw minimum alpha=.0976791 and chi=.00954121 occur near the puncture; the pulse's minimum conformal determinant is.9999999999999997. Every saved metric passes, including refinement-interface and physical ghosts.

Checkpoint files do not embed a per-block gid or rank signature. The validator verifies the rank-file names, common full logical-location table, deterministic equal-cost partition and each file's expected block count; it cannot authenticate semantically swapped same-sized payloads without an external provenance hash. Exact source/output hashes are therefore retained in the validation records.

Scope guards reject AMR, an unsupported background, wrong mass and wrong chi exponent. Other restricted assumptions are documented in the validator and checked explicitly, including centered geometry, no inner excision/projector, the residual lapse interpretation and SMR coarse indices.

## Deliberate rejection test

A private copy of the zero checkpoint was modified at rank0/gid21, relative level3, first inner-x ghost at **(−4.0625,−2.9375,−2.9375)M**. Setting only residual gxx to−2 makes raw full gxx, its second leading minor and determinant all−1. The validator rejects exactly that one fine-level interface ghost and reports the correct rank, gid, level, array index and coordinates. All original cohort hashes remained unchanged. The malformed file is never passed to AthenaK.

The remaining rank files in that private test are hard links to the originals and are read-only test inputs. `test_validator.py` creates a new mutation directory and refuses to overwrite an existing one. Raw restart files remain external research artifacts; the compact manifest records their hashes rather than packaging them.

## Reproduce checks without evolution (requires external raw checkpoints)

The compact package contains reports only. Restore the original raw `zero/rst`
and `pulse/rst` trees in a separate working directory, then run the packaged
validator with those paths. The following commands assume that layout, with
the validator and mesh-audit paths adjusted to this checkout.

```
python3 ../check_smr_trumpet_checkpoint.py zero --ranks 8 --cycle 3 --exact-zero --mesh-audit ../mesh-audit.json
python3 ../check_smr_trumpet_checkpoint.py pulse --ranks 8 --cycle 3 --mesh-audit ../mesh-audit.json
python3 summarize_preflight.py
```

`run_three_cycles.py` is the explicit local launch record and should not be rerun as part of ordinary validation. `results.json`, individual checkpoint validation JSONs, `validator-regression.json`, input hashes and application logs contain the evidence.
