# Angular KO floor in the outer shells

## Parameters

All in `<z4c_hyperboloidal>`.

| Parameter | Default | Meaning |
|---|---|---|
| `ko_angular_floor` | 0 | Lower bound on the radial taper of angular KO lines in the spherical shells, in units of `ko`. Zero reproduces the earlier operator and checkpoint identity exactly. |
| `ko_angular_begin`, `ko_angular_full` | 0.55, 0.65 | The floor rises smoothly from zero to its full value between these compactified radii. |
| `ko_angular_fade`, `ko_angular_end` | 0.91, 0.96 | The floor falls smoothly back to zero between these radii. |
| `restart_with_new_dissipation` | false | Accept a checkpoint whose identity differs only in `ko` or the angular floor. |

The taper of an angular line is `max([max(0, 1 - r^2/0.64)]^6, floor(r))`. Radial lines and the
Cartesian blocks keep the bulk taper alone.

A restart can only add parameters through an input file: `-r checkpoint -i override.athinput`
with

```
<z4c_hyperboloidal>
ko_angular_floor = 0.65
restart_with_new_dissipation = true
```

The command line modifies existing parameters only.

## Why

N24 head-on and spinning-plunge evolutions grow a checkerboard layer along the angular patch
edges of the shells between r = 0.6 and scri, at about 0.2 per M0, until the monitored
characteristic CFL bound stops the run near 112 M0. The bulk taper is 3.9e-7 at r = 0.765 and zero
from r = 0.8, so nothing damps it.

Paired restarts from the N24 head-on checkpoint at 95 M0 showed that the layer

- sits in the cells beside patch edges: damping lines that touch four edge cells has the full
  effect, two cells 80%, one cell 45%, the interior alone 1%;
- is carried by Theta, P, chi and the lapse; damping the conformal metric, A, Lambda, the shift,
  B or D does nothing;
- does not depend on the seam interpolation order (6, 4 or 2 points) or on the temporal level.

The effect of the floor on the growth rate is linear in its value: 0.2 changes the exterior
Hamiltonian growth rate by -0.17 per M0 and 0.65 by -0.54, against a production rate of +0.20.

## Checks made on Aurora (N24, production checkpoints)

- Zero floor: checkpoints identical bit for bit to the production runs, head-on (static
  refinement) and spinning (Cartesian AMR).
- Floor 0.65: a run stopped after 111 cycles and continued from its own checkpoint is identical
  bit for bit to the uninterrupted run.
- Spinning AMR case across a regrid: the regrid happens at the production cycle with and without
  the floor.

Stability to 120 M0 and the effect on the Bondi news had not been measured when this note was
written; see the result archive `results/n24-seam-instability-20261123/diagnosis.md`.
