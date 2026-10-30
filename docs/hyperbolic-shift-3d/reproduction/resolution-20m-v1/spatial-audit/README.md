# Spatial localization of the exterior constraint plateau at 3M

The native diagnostic-only checkpoint reader now optionally accepts a comma-
separated list of increasing radial cutoffs. It evaluates the unchanged native
constraint kernel on each saved state, using the same refreshed halos and scri
state. No evolution steps are taken and the frozen Aurora executables are not
modified. Each cumulative entry contains unnormalized squared-constraint
integrals and compactified-coordinate volume outside its cutoff. Differences
between adjacent cumulative entries give disjoint radial bands. Band RMS is
sqrt(difference of integrals / difference of volumes).

The matched N16/N24/N32 DT1 checkpoints are all exactly at3M, before switching
to direct chi/lapse evolution. The common exterior cutoff is r=0.0753, outside
all three candidate mean radii. This is intentionally distinct from the
own-horizon cutoffs in the time-history plot. Discrete inclusion of a cell layer
can change an exterior norm even for a small change of cutoff.

`execution.json` records commands, elapsed times, executable and checkpoint
SHA256s and before/after checkpoint immutability checks. `plot.py` verifies that
band sums reconstruct full and exterior norms, and independently compares the
full-domain norms to the original histories (absolute differences below1e-11).
`outer-fine/plot.py` checks repeated measurements at every common cutoff.
No empty radial bin is interpolated or counted as zero error.

## Findings

At3M, the r=0.6–0.8 band contributes the following fractions of the exterior
squared-constraint integrals:

| Grid | Hamiltonian | Radial momentum | Invariant momentum |
|---|---:|---:|---:|
| N16 | 54.1% | 55.0% | 81.8% |
| N24 | 88.2% | 88.3% | 93.1% |
| N32 | 95.3% | 94.0% | 94.4% |

Within this band, N24/N16 band-RMS ratios are1.246 for H and1.253 for Mr.
N32/N24 ratios are1.0002 and0.9945: the error is effectively flat under the
second refinement. By contrast, both refinements substantially reduce errors
near the horizon and throughout most of r=0.0753–0.4, including the core/shell
blend r=0.16–0.18. The extreme outer rows do not dominate these integrals.

Finer bands put84.7% of N32 exterior H squared integral and82.9% of Mr in
r=0.675–0.75 alone. N32's largest measured band RMS is at0.675–0.7; N24's is
at0.7–0.725. Thus the error is not exclusively a spike at the r=0.75 interface.
These are spatial measurements at3M; they do not determine where the error was
first generated or its propagation direction.

## Concrete suspect and discriminating checks

`ShiftCoefficients` in src/z4c/hyperboloidal/hyperbolic_shift.hpp defines
w(r)=SmoothCollar((0.8-r)/0.24), where SmoothCollar uses eighth powers.
Although its support transition spans0.56–0.8, its central10–90% change has width
only0.032752737, centered at0.68. In the0.45–0.75 shell, dr=0.3/N, so that
width spans1.75,2.62,3.49 radial cells at N16,N24,N32. The gauge transition is
poorly resolved at all three grids and overlaps the error concentration. The
nearby shell interface at0.75 remains another candidate. Correlation is not
proof: a source-term inconsistency or angular/interpatch error could also yield
this behavior. Radial averages alone cannot distinguish angular face seams.

The first useful controlled comparison is finer radial sampling of0.56–0.8 at
fixed core resolution and unchanged gauge coefficients. Separately moving the
0.75 shell interface while retaining the same gauge would distinguish an
interface-tied defect from a gauge-transition-tied one. Broadening the gauge
transition is a separate equation change and should not be mixed into the
current frozen convergence campaign.

The exterior plateau and N24's later indefinite-metric failure at r=0.01244
are distinct observations; the latter does not explain the3M exterior plateau.
No formal convergence order or unique causal diagnosis is claimed.

Regenerate plots with `python3 plot.py` and `python3 outer-fine/plot.py` from
this directory. Build the reader with the existing
hyperbolic_shift_3d_checkpoint_constraints CMake target. Exact measurement
commands are preserved in the two execution.json files; outputs must be fresh.
