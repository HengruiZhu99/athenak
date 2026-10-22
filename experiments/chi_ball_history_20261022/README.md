# Chi-masked 3D history diagnostic

The Z4c interior histories now measure cell centers satisfying exactly
`sqrt(x*x+y*y+z*z) < 16` and `chi >= 0.0625`, centered on the coordinate
origin (coordinates in M for the supplied unit-mass decks). There is no
inner-radius cut: cells inside 2M contribute whenever they pass the chi mask.
These are 3D cell-volume integrals, not norms reconstructed from line slices.

Defaults are `history_interior_radius=16`, `excise_chi=0.0625`, and
`history_inner_radius=0`. The outer radius and chi threshold remain configurable;
outer radius zero disables the extra histories. A nonzero legacy inner radius
now raises an error rather than silently applying a different mask.

The original nine full-domain integral columns remain unchanged. The next nine
columns retain their positions, with one explicit header change:

| File column (one-based, including time and dt) | Meaning |
| --- | --- |
| 12–14 | Integral of H squared, M squared, C squared over the chi-masked ball |
| 15 | Physical volume of the retained cells |
| 16 | `boundary-exposed-volume`: retained physical volume failing the boundary-distance screen |
| 17 | Full-grid sampled speed-bound violation volume, including puncture cells |
| 18 | Coordinate volume of retained cells |
| 19 | Physical volume inside the ball excluded by chi |
| 20 | Full-grid speed-bound violation volume restricted to chi-retained cells |

Physical weights are `dx*dy*dz*sqrt(abs(det(g_ij)))`. For a normalized L2 norm,
use `sqrt(integral / retained_physical_volume)`; a zero retained volume makes
the normalized norm undefined. M2 and C2 are already squared constraint fields.
The boundary screen still uses distance to the nearest domain face compared
with `history_boundary_buffer + history_boundary_speed*time` (defaults 4 and 8).
It now reports exposure **without removing cells from the norm**. Nonzero
exposure or speed flags preclude claiming boundary safety from this screen.
The speed estimates are sampled diagnostics, not a causality proof.

The chi mask evolves and differs across resolutions, so retained volume may
change. Nonzero chi-excluded volume is expected, not a failure. The historical
fixed-shell audit's constant-volume, zero-chi-exclusion and horizon-inside-2M
requirements do not apply. This diagnostic alone does not establish convergence
or boundary-safe 20M evolution on a static boosted-hole mesh.

`inputs/` contains revised telegrapher N32/N48/N64 4M boosted-hole decks. Their
physics and mesh match the October 1 decks; only the job basename and diagnostic
mask parameters differ. Historical decks, manifests, histories, sharing archives
and the pinned Aurora executable are preserved. No new runs are submitted by
this change. Existing history data cannot be converted to this new 3D mask
without the full volume fields; fresh evolution with the new binary is needed.
Do not use the old campaign controller/audit to submit or validate these decks.

Local validation (Python needs numpy and h5py for the existing binary reader):

```sh
cmake --build build/local-serial -j 8
build/analysis-env/bin/python experiments/chi_ball_history_20261022/check_history.py
python3 experiments/single_hole_gauges_20261001/check_local.py
```

The first test independently sums full 3D ADM, chi and constraint dumps at
initialization and after one evolution step. It checks retained cells below 2M,
chi exclusion, the outer sphere, boundary reporting without norm changes,
disabled diagnostics and rejection of legacy inner cuts. Binary fields are
stored as float32, so comparison to the double-precision history uses a matching
roundoff tolerance.
