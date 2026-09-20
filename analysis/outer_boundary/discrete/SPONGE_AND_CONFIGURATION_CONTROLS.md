# Absorbing-layer and configuration-boundary controls

These are bounded frozen-coefficient discrete tests. They keep production G=1,
kappa1=.1, kappa2=0, eta=2 and lapse residual damping=.1. They do not reset or
clip evolved fields. No source or campaign changes were made for these tests.

## Exact existing outer-sponge semantics

The production problem generator already implements a residual source
`rhs -= sigma(x)*u_residual` on all 25 evolved fields. The physical 20-field
linear subsystem modeled here excludes the inactive auxiliary B fields.

For `outer_sponge_geometry=face`, the code takes the minimum distance to any
Cartesian physical face, then

```
s = clamp(1 - distance/width, 0, 1)
sigma = rate * s^3 * (10 - 15s + 6s^2).
```

The radial alternative uses the same smootherstep from a start radius to a
ramp-end radius, then holds the maximum rate to the boundary. The explicit
source timestep cap is `1/max_rate`, which is safely above dt=.0375M for the
.02--.1/M rates tested. The source acts on residuals, hence preserves zero
exactly without a threshold or hard reset.

**Ordering matters:** production applies UserRHS before zero_rate. Configuration
RHS damping is therefore differentiated by the boundary map; momentum RHS
damping in incoming combinations can be overwritten. The primary results below
implement that exact ordering. A post-boundary diagonal source was tested only
as a distinct diagnostic.

## Two-dimensional spectra

All tests use dx=32M, linear ghosts and production D6/KO8. Rates are in 1/M.

| Grid | Width/cells | Rate | Largest real eigenvalue |
|---|---:|---:|---:|
| 12x12, no sponge | — | 0 | .008968 |
| 12x12 | 4 | .02 | .001385 |
| 12x12 | 4 | .05 | .0002317 |
| 12x12 | 4 | .1 | .00005283 |
| 12x12 | 6 | .02 | <2e-15 |
| 12x12 | 6 | .1 | <2e-15 |
| 16x16 | 4 | .02 | .001419 |
| 16x16 | 6 | .02 | .001366 |
| 16x16 | 6 | .1 | .0002833 |
| 16x16 | 8 | .02 | <2e-14 |

The nominal passes damp nearly the entire small test domain. The wider-domain
checks below show why they cannot be called a robust stability result.

At 12x12/width4, applying only a diagonal momentum-field sponge of rate .1
leaves gamma=.001127, worse than damping all fields. This is a momentum-only
comparison, not a pure incoming-characteristic projector. Applying all-field
damping after zero_rate at rate .02 gives gamma=.001274; changing ordering alone
does not cure the mode.

A separate source discriminator smoothly tapered kappa1 from .1 in the interior
to exactly zero at the outermost active cells over four cells. It still grows at
.007534/M. This taper is not recommended as a fix.

## Long-strip check and mode location

`strip_model.py` uses 64 normal points (2048M) and a Fourier tangential direction.
Only the outer eight cells (256M) at each end receive the sponge, leaving 48
undamped interior cells. It retains the original zero_rate boundary and exact
pre-boundary source ordering. The source characteristic reference is identical
to production; the strip is flat and omits background gradients/AMR/corners.

| Tangential kh | Rate .02: max real | Rate .1: max real |
|---|---:|---:|
| pi/8 | .001585 | .001245 |
| pi/4 | .00001286 | .0003167 |
| pi/2 | <4e-16 | <4e-16 |

Increasing the sponge rate is not monotonically better. At kh=pi/8 the original
mode has gamma=.005635 and peaks 48M from a physical face. With rates .02/.1 its
full-state peak moves to 272/304M from the face, just inside the sponge onset.
The fraction of weighted full-state eigenvector norm in the outer 256M changes
from 98.3% to 40.9%/15.9%. The normalized eigenpair residuals are about 1e-15.

See `sponge-surviving-modes.png` and its JSON. The layer suppresses the original
near-face mode but leaves a growing mode near its transition. This is mitigation,
not elimination, and not proof that all growth has the same continuum origin.

## All-configuration boundary discriminators

Use the unchanged volume relation `q_t=C p+B q`, where C is an invertible local
10x10 matrix, q=(chi,h_STF[5],alpha,beta[3]) and
p=(Khat,Theta,A_STF[5],Gamma[3]). The reflecting differential control imposes
`q_tt=0`; the common outgoing control imposes `q_tt+Dn q_t=0` in the flat model.
The compatible momentum update is

```
p_t = -C^{-1} [B(q_t) + v Dn(q_t)] , v=0 or 1.
```

No configuration RHS or state is overwritten. These are practical reflecting/
radiative controls, not claimed physical-constraint-preserving boundaries.

For 12x12/dx32, both remain weakly unstable under the production ghost scheme:

| Control | Linear ghosts | Quadratic ghosts |
|---|---:|---:|
| Reflecting q_tt=0 | .0003471 + .006026i | .0001749 + .005721i |
| Common outgoing | .0004353 + .003159i | .0005299 + .003443i |

A compact initial lapse pulse with exactly compatible zero boundary q and q_t
was evolved by the matrix exponential through 5000M. Both controls remain
finite, but weighted full-state norms increase while Theta stays small. The
final Theta maxima are 4.30e-13 and 6.66e-13; final full-state weighted norms are
8.59e-9 and 3.27e-8. This does not override the positive eigenvalues.

The reflecting control preserves boundary q to about 9e-23 for that compatible
pulse. It would permit linear boundary drift if the initial q_t were nonzero:
q_tt=0 alone does not choose the integration constants. The common outgoing
control similarly preserves its initial boundary wave residual, not an
arbitrary subsequently imposed homogeneous value. These neutral-data issues
must be addressed explicitly in any implementation.

## Interpretation

Neither a smooth sponge, localized kappa taper, nor these all-q controls pass
the bounded robustness checks. An outer sponge may extend the usable time of a
specific run, but would alter the residual spacetime in its layer and must not
be advertised as a consistent boundary cure or promoted from one coarse-box
pass. Full-domain matter/AMR/MPI validation remains necessary.

## Source-order recheck and GPU diagnostic comparison

The requested post-CPBC source-order discriminator had already been measured;
no additional scan was run. For 12x12 cells, dx=32M, width=128M and rate=.02/M,
adding `-sigma*u_residual` to **all q and p fields after** the unchanged
zero_rate closure still gives gamma=.001274113662/M, with four positive
eigenvalues. RK3 at dt=.0375M gives the same growth rate. The matched original
pre-CPBC order gives .001385008843/M. Both maps preserve exact zero. Thus the
closure overwriting momentum damping is not sufficient to explain or remove
this growing mode. Evidence:
[`results/sponge-after-n12-r02-w4.json`](results/sponge-after-n12-r02-w4.json)
and [`results/sponge-before-n12-r02-w4.json`](results/sponge-before-n12-r02-w4.json).
This is a negative corner-model result, not a new long-strip or resolution
validation of the post-CPBC order.

The GPU8841948 comparison requires matching diagnostics. In the saved
rate=.02/M, width=256M strip eigenvector, the **weighted full-state norm** peaks
272M from the face and has 40.94% inside the layer, whereas **Theta** peaks
144M from the face and 99.1231% of Theta squared lies inside the layer. The latter
agrees closely with the reported GPU active Theta peak at 144M and 99.1344% of
Theta squared inside the sponge. The strip uses flat coordinate-volume weights;
the GPU diagnostic uses proper-volume weights on the evolving geometry. The
earlier onset-location claim applies only
to the strip's weighted full state; the GPU Theta diagnostic does not establish
that full-state location. This agreement also does not by itself identify the
same eigenmode in the evolving three-dimensional background. Both strip
diagnostics follow from the existing arrays in
[`results/sponge-surviving-modes.json`](results/sponge-surviving-modes.json):
use the `rate=.02` entry, cell-center distance to the nearest face, and square
`theta_normalized` before summing over distances below 256M.
