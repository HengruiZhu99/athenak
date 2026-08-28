# Reference shock-avoiding Figure-3 campaign handoff

## Verdict

`AURORA_PVC_QUALIFIED; N256_REACHED_TLIM; PARTIAL_FIGURE3_AGREEMENT; CONSTRAINT_INVALID_AT_FIRST_PEAK; CONVERGENCE_NOT_RUN`

The corrected shock-avoiding/effectively-zero-shift implementation passed a
fresh Aurora PVC qualification. The N256 reference run then reached coordinate
time `30 M`, but it did not qualify as a Figure-3 reproduction: its apparent
first curvature peak occurs at almost the published proper time, while the
constraint integrals are already undergoing catastrophic growth. The run ends
at central proper time `11.2863 M`, before the published deep-minimum and
rebound region near `12.6--13.2 M`. The conditional N128/N512 replay campaign
was therefore not run and no convergence claim is supported.

## Narrow source correction and PVC qualification

The literature-aligned lapse/shift choice is

```text
partial_t alpha = beta^i partial_i alpha
                - (alpha^2 + 1) (Khat + 2 Theta),
alpha(t=0) = 1, beta^i = 0.
```

The lapse RHS was already correct. The original admissibility implementation,
however, accessed host-owned `Z4c::opt` from a device lambda. Host and
bounds-checked CPU tests passed, but the Aurora PVC kernel faulted. Job
`8789460` bisected the first bad source commit to
`151bf20d13c4838793f1c26e0b6b6e669cb7b765`. The repair copies the required
boolean policy to a scalar before entering the Kokkos lambda. It does not change
the gauge equation, AMR operators, damping, KO dissipation, or boundary
conditions.

Source repair commit:

```text
commit f8303c6be7eb214fa1e91b646123ee0d434b3698
tree   7a585ca487b12351b084eb425bb812775849b001
```

Aurora qualification job `8789659` used one PVC charged to
`CompactBinaryMerger`, exited zero, and wrote
`AURORA_PVC_QUALIFICATION_PASS`. Eleven focused tests passed, including native
vertex-centered Cartoon production kernels, multilevel O4 Cartoon paths, and a
two-cycle shock-avoiding/prescribed-zero-shift smoke. Static multilevel Cartoon
constraint RMS values were approximately `1.43e-13--2.58e-13`.

Exact build authority:

```text
AthenaK commit        f8303c6be7eb214fa1e91b646123ee0d434b3698
AthenaK executable    aae7ccb8739fb4951221ad7be69ea0e220548b52d402086f57d7857fa2c97a13
CMake cache           8da40bcb47564d9184119ca207f9847a33a3d1b5bd2930627d705cda8fb36386
Kokkos commit         6739bc623081648af9e752b616d9671527922cbf
IrisK source          620acca67c2736d9add98ecae3ec76f0f2800b29
IrisK library         380a90d5b1d9762fe7f9076edcb27fb4a209f4cd8c070da376c36284a438c7a1
Brill coefficients    1b5f0efc3f080215ed7d7994194ba63ea123415bfd8e74c54ca1fd72680aea10
N256 input            6c694cf871a3d694d745f0fb58b279b6cd07516463ac8ad54f1c91d2689c90ba
```

Qualification evidence manifest SHA-256:
`8acb38caa52c15236c3537e9d074bf76e9429409dc5120d66c7d75ce18e4ba27`.

## N256 execution

Science job `8789703` used one Aurora PVC in the debug queue under
`CompactBinaryMerger`, exited zero, and recorded `REACHED_TLIM`.

The run used native vertex-centered axisymmetric Cartoon SO(2), an N256 root
grid (`128 x 256` active vertices represented by `32 x 32` MeshBlocks), outer
radius `128 M`, initial refinement radii `64/32/16 M`, O4 finite differences,
q6 prolongation, RK4 at CFL `0.15`, KO `0.5`, `dchi_max=0.02`, derefinement at
`0.25 dchi_max`, no Z4c constraint damping, shock-avoiding lapse with
`kappa=1`, unit initial lapse, prescribed zero shift, and telegraph gauge off.

```text
Imported ADM mass                       2.6606354586228815
Initial proper-box C_rms                2.384524686e-05
Initial proper-box H_rms                2.378410596e-05
Final coordinate time                   30.0 M
Cycles                                  5791
Final central proper time               11.2863067801 M
Minimum sampled axis lapse              0.1434111145
Axis lapse zero crossings               none
Maximum refinement level                5
Maximum MeshBlocks                      212
```

The hierarchy changed in the first two cycles, from 104 blocks reaching
physical level 3 to 212 blocks reaching physical level 5. The first retained
history row with the final topology is at `t=0.0249029532 M`. The hierarchy then
remained fixed through `t=30 M`; later logs report no refinement requests and
rejected derefinement requests. Thus the late runaway is not synchronized with
repeated topology changes in this run.

## Figure-3 comparison and constraint qualification

No time, amplitude, or vertical offset was fitted. The direct apparent
first-peak comparison is:

| Curve | central proper time | `log10(abs(Kretschmann))` |
|---|---:|---:|
| AthenaK N256 | 10.30333 | 5.01349 |
| bamps | 10.30683 | 5.47778 |
| Prague | 10.31154 | 5.48688 |
| sphGR | 10.31384 | 5.47833 |

The timing is close, and the AthenaK curve follows the published curves well
at early proper time, but its peak is about `0.47 dex` low. More importantly,
the constraint squared-integrals grow rapidly before and through this point:

| Quantity | Initial | Maximum | Time of maximum | Final |
|---|---:|---:|---:|---:|
| C | 6.8505e-4 | 48.2330 | 25.3091 | 3.1583 |
| H | 6.8152e-4 | 41.1314 | 25.3091 | 2.3658 |
| M | 0 | 9.53227 | 25.4556 | 0.64059 |
| Z | 8.8166e-7 | 0.0550105 | 26.3318 | 0.0366208 |

The C integral first crosses `0.01`, `0.1`, `1`, and `10` at coordinate times
`20.12825`, `21.99447`, `23.37376`, and `24.40678 M`, respectively. The global
and sampled-axis Kretschmann maximum is `1.0315458e5` at coordinate time
`25.37186 M`, central proper time `10.30333 M`, inside this constraint
catastrophe. Consequently the apparent peak-time agreement is descriptive,
not scientific reproduction evidence.

The run did not reach the published deep minimum or rebound. No horizon finder
was enabled (`num_horizons=0`), so the artifacts support no horizon statement.

## Diagnostic measure and AMR interpretation

The Cartoon history measure is already the physical axisymmetric ring measure,

```text
2 pi rho dx1 dx2 sqrt(abs(det gamma)),
```

with vertex trapezoid weights and canonical ownership of shared nodes. There is
no fictitious collapsed-y factor in these constraint histories.

Current evidence shows a late instability on a fixed hierarchy after an early
single refinement event. It does not isolate whether the source is bulk
evolution, a persistent AMR-interface mode, a missed short scale in a non-chi
field, or their combination. The observation that the chi sensor requests no
additional refinement while curvature and constraints explode makes a bounded
all-field resolution/spectral audit a natural next discriminator, but it is
not itself proof that chi is the wrong refinement variable.

## Setup-only attempts

Job `8789668` terminated during a one-second preflight because Aurora does not
export `PBS_NNODES`; no science ran. Job `8789692` initialized IrisK but stopped
before evolution because the AMR-history authority parent directory was
missing. These are setup failures, not science evidence.

## Claim boundary

Supported:

- the narrow device-capture correction is host and Aurora-PVC qualified;
- the exact N256 run reached its coordinate-time limit with finite history
  rows;
- its early central-curvature curve and apparent peak time partially agree
  with the published Figure-3 curves;
- its first apparent peak is constraint invalid;
- the constraint normalization is already the correct axisymmetric measure.

Not supported:

- a full Figure-3 reproduction;
- stable self-similar collapse through the deep minimum and rebound;
- N128/N256/N512 convergence;
- a horizon conclusion;
- identification of a unique source bug or continuum formulation failure.

The smallest useful next step is a bounded, location-resolved diagnostic on the
existing pre-runaway state: compare high-frequency content and cheap
O6-versus-O4/undivided-difference indicators across all evolved Z4c variables,
and separate bulk active-state growth from coarse-fine/interface ghosts before
changing production numerics.
