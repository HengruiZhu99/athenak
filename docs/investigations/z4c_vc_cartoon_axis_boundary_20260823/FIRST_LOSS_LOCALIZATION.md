# First fixed-grid Brill loss of convergence

Date: 2026-08-23

## Configuration

All decisive runs are fixed-grid native-VC/O4 Brill evolutions with no AMR:

```text
A=-0.047
rho=[0,16], z=[-16,16]
N128/N256/N512 on the same physical 4x8 MeshBlock lattice
RK4, CFL=0.15, KO diss=0.02
max-|K|-scaled telegraph lapse, tau=kappa=1
Gamma-driver shift, eta=2
kappa1=kappa2=0, no chi floor
initial alpha=psi^-2
```

The comparison uses the 128x32 global IrisK coefficient control. Its SHA-256
is `1b5f0efc3f080215ed7d7994194ba63ea123415bfd8e74c54ca1fd72680aea10`
and its ADM mass is `2.6606354586228815`. This file was generated with the
existing local IrisK exporter from a dirty/uncommitted IrisK worktree; it is a
diagnostic control, not a newly qualified upstream initial-data release.

## Initial-data discriminator

The original 48x32 coefficient file was spectrally under-resolved at the
origin. Angular `l>0` modes plateaued at `1e-8--1e-7`, so a nominally smooth
origin value left a defect amplified by a finite-difference second derivative
as `h^-2`. A 96x32 control reduced the plateau to roughly `1e-12` but still
showed negative pointwise origin orders. The 128x32 control reduced it to
roundoff and changed the origin result decisively:

| Initial point/region | observed N128/N256/N512 order |
|---|---:|
| axis-center Theta RHS | 2.721 |
| axis-center H | 3.893 |
| axis-core constraints, worst significant | 3.824 |
| core MeshBlock interior RHS, worst significant | 3.925 |

Thus the former immediate axis-center divergence was an upstream coefficient
resolution defect, not evidence against the repaired Cartoon axis formulas.

## Earliest evolved loss

With quadratic physical-ghost extrapolation (`extrap_order=3`), the earliest
meaningful negative three-resolution order at central proper time
`tau_c=0.125 M` is on a physical outer face:

| region | family/component | order | `|N256-N512|` RMS | worst common point |
|---|---|---:|---:|---|
| outer layer 0 | Theta RHS | -3.262 | 7.565e-7 | `(rho,z)=(16,0)` |
| outer layer 1 | C constraint | -1.309 | 5.711e-10 | `(15.875,0)` |
| core `r<=8`, block interior | worst RHS (`Azz`) | +3.578 | 8.832e-6 | aggregate |
| axis core `|z|<=8` | worst RHS (`Theta`) | +1.620 | 1.912e-8 | aggregate |
| full domain | worst significant | +3.593 | 4.579e-7 | aggregate |

The whole-axis aggregate also has a small negative Theta order at this time,
but its worst points include the outer axis/z corners. It is therefore a
boundary observation, not an independent central-axis result.

The face-local loss persists and moves among the Sommerfeld and adjacent
bulk fields:

| `tau_c/M` | outer layer | worst field | order | worst point |
|---:|---:|---|---:|---|
| 0.50 | 0 | Theta RHS | -0.234 | `(16,0)` |
| 0.50 | 1 | Ayy RHS | -2.042 | `(15.875,0)` |
| 0.75 | 1 | Axx RHS | -3.846 | `(15.875,0)` |
| 1.00 | 0 | Ayy RHS | -2.023 | `(16,0)` |
| 1.25 | 0 | Axx RHS | -3.499 | `(16,0)` |
| 1.25 | 1 | Axx state | -3.557 | `(15.875,0)` |

At `tau_c=0.75`, the worst axis-core row is the active `x2` gauge-driver
component reported as `By`, with order `-1.765` at `(rho,z)=(0,4)`.
That point is also a same-level MeshBlock seam. It is a later seam-axis
observation and has not been isolated to parity filling, same-level halo
completion, or the gauge RHS. It does not precede the outer-face loss.

## Global and core behavior

All N128/N256/N512 quadratic-ghost runs reached `t=5 M` and
`tau_c≈3.07987 M`. At common proper times through `tau_c=3 M`, all four
axisymmetric history norms (`C,H,M,Z`) decrease monotonically with resolution;
their three-resolution self-difference orders are positive. The minimum over
the sampled history table is `2.630`.

At exact dense samples through `tau_c=1.25 M`, the core MeshBlock-interior
state/RHS/constraint orders remain approximately `3.4--3.9`, except for a
lower-order gauge transient around `tau_c=0.75`. Shared native vertices are
bitwise identical. The physical face nevertheless fails the strict local
gate at every evolved sample.

The history diagnostic is already normalized with the proper axisymmetric
ring measure. In native VC Cartoon mode it uses

```text
2*pi*rho*dx1*dx2*w_rho*w_z*sqrt(det(gamma))
```

with nodal trapezoid weights and no fictitious collapsed-`y` spacing. The
earlier constraint jump is therefore not a `dx3` normalization artifact.

## Observation, inference, hypothesis

Established observations:

- the 48x32 initial-data coefficients are under-resolved at the origin;
- the 128x32 coefficient control restores positive axis-center and core
  initial convergence;
- the earliest evolved negative order is on the outer face at `(16,0)`;
- the central core remains substantially better in the same time window;
- no AMR operation exists in these runs.

Supported inference:

- the first fixed-grid physical limiter is the outer boundary closure, not
  the Cartoon axis or AMR transfer;
- the later seam-axis `By` result merits a separate bounded diagnostic but is
  not the source of the first global loss.

Open hypotheses:

- low-order physical ghost completion contaminates centered second
  derivatives in the non-Sommerfeld variables and adjacent layers;
- the partial Sommerfeld overwrite and bulk ghost closure form an
  inconsistent coupled boundary operator;
- the later seam-axis gauge result may be a same-level derivative/halo effect
  or simply a non-asymptotic near-zero gauge component.

No unique boundary term-family source is established because the dense
diagnostic records the total post-boundary RHS, while named RHS-term telemetry
was not spatially resolved over the physical face.
