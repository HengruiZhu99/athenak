# Long-time residual Z4c boundary investigation

**A wider sponge alone does not cure the instability.** Reduced damping gives
much better vacuum behavior, and a coupled exterior-boundary reference removes
the known growing linear mode without changing the original damping sources.
Neither result is clearance for nonlinear black-hole, matter or AMR evolution.
The production campaign, executable, checkpoints and monitor remain unchanged.

## Completed evolution tests

All use the G=1 background-adapted gauge, sixth-order volume differences and
RK3. The weak-curvature patch has no horizon or refinement. The large cube
has an evolved residual Minkowski background, no forced metric override and
no matter feedback. Here M denotes the reference production mass unit.

| Test | kappa1 / shift eta / lapse damping | Actual outcome |
|---|---|---|
| Weak-curvature patch, no sponge | .01 / 2 / .1 | Invalid-state failure at 5540.4 M |
| Same patch | .01 / .2 / .01 | Failure at 8290.2 M |
| Same patch | .1 / .2 / .01 | Failure at 3990 M |
| Same patch | 0 / 0 / 0 | Clean walltime stop at 12212.4 M, below target |
| Same patch | 0 / .02 / .01 | **Reached 50000 M**; final Theta RMS 7.56e-17, maximum 4.71e-16 |
| Large cube, no sponge | .1 / 2 / .1 | Failure at 5650.2 M |
| Large cube, broad radial sponge | .1 / 2 / .1 | Failure at 6420 M |
| Same cube and sponge | 0 / 2 / .1 | **Reached 20000 M**; final Theta RMS 5.84e-16, maximum 2.13e-14 |

All final rank files in the two target-completed controls passed full-payload
finite checks and positive-definite full/ghost metric checks. Their seeds were
compact lapse perturbations, not direct constraint pulses. Small Theta alone
would be insufficient: metric and gauge residuals in the GPU 20000 M control
also decreased over the late checkpoints from 5000.4 to 20000 M.

The cube is [-2048,2048]^3, dx=64 M, with eight 32^3 blocks/eight GPU MPI ranks.
The weak patch uses eight 8^3 blocks/four CPU MPI ranks and dx=32 M. Do not
interpret the different geometries or initial perturbations as a convergence
sequence. [Local results](local-analysis/README.md) and
[GPU results](gpu/results/README.md) contain plots, inputs and validation.

## Where the growing mode appears

In the large-cube controls with kappa1=.1, Theta RMS grows exponentially over
4000–5000 M: gamma=0.006704/M without the layer and 0.005747/M with it. More
than 99.68% of the proper-volume Theta-squared integral lies within 256 M of
physical faces at 5000.4 M. These are late amplification profiles, not a
measurement of first injection. The first printed invalid states are fourth
physical ghost cells; the later active-cell abort is a separate diagnostic.

The original boundary has positive eigenvalues in the complete coupled
source-plus-boundary model. Removing one physical-constraint branch or
increasing sponge damping is insufficient. The previously repaired ghost-read
defect is real, but is independent of this remaining mode.

## Broader, smoother sponge

The radial rate is zero for r<=512 M. With s=clamp((r-512)/1280,0,1),

```
sigma(r) = (0.001/M) * s^3 * (10 - 15*s + 6*s^2).
```

The C2 ramp spans 20 coarse cells and reaches its constant rate at 1792 M,
before the nearest boundary at 2048 M. It continuously relaxes residuals;
it never clips metrics or resets small residuals. The source timestep cap is
retained at 1000 M, but does not automatically bound explicit shift damping.
The eta=2 controls use dt=.6 M; eta=.02 uses dt=3.2 M after short timestep checks.

The [discrete screen](discrete/README.md) finds that pushing the boundary out
alone is not monotonic. At fixed dx=64 M and a 512 M layer, doubling a strip's
length from 4096 to 8192 M increases one mild rate from 1.71e-6 to 5.20e-6/M.
A 2048 M layer in the larger strip reduces it to 2.98e-7/M. These frozen strip
results do not include radial geometry, black-hole gradients or AMR.

At kappa1=0, the matched wide-layer analogue with eta=2 retains a mild k=0
rate of 1.213e-6/M. Lowering lapse damping alone leaves it unchanged; lowering
eta to .02 removes a resolved positive rate at the four sampled tangential
wavenumbers. This is the reason for the leading reduced-source candidate,
not a claim that every mode is stable.

## Direct constraint pulses and retained incoming data

GPU job 8842248 uses Theta=A exp[-r^2/(2*384^2)] with A=1e-6, kappa1=0,
eta=.02, and lapse damping .01 or .1 in two independent fresh starts.
Job 8842283 repeats the primary with A=1e-7. The primary's cached histories
reach 30800 M; a later status-only observation reached 45571.2 M. **Final
stopping reasons and final checkpoint validity remain unverified** because
Aurora authentication expired. Elapsed calendar time is not completion.

The primary's late exterior RMS is about 1.26e-13. This cannot automatically
be called roundoff: the Gaussian already has nonzero incoming boundary data.
At (2016,32,32) M its initial incoming scalar C1 is 1.027692619e-12. Across the
first three RK3 cycles (9.6 M), Theta at that point changes by 7.46%, but C1
stays within 1.3 ppm of its initial value. Other incoming scalar traces behave
similarly. The boundary's `zero_rate` condition freezes an incoming rate,
rather than setting the incoming state to zero. Because the sponge precedes
CPBC, the boundary correction can cancel its incoming-rate damping.

This is evidence for retained initial data, distinct from the positive
boundary mode. Replacing dC/dt=0 by dC/dt=-nu*C merely changes a modal row's
factor from lambda to lambda+nu and leaves its old positive roots unchanged.
[Trace audit and spatial profiles](theta-propagation/README.md) preserve all
terms and distinguish short local checkpoints from incomplete GPU samples.

The new optional `outer_sponge_test_theta_pulse_profile = compact` gives an
exactly supported bump, exp[1-1/(1-q^2)] for |q|<1 and zero otherwise. For the
centered monopole fixture it is smooth and exactly zero near every physical
boundary. Gaussian remains the default. The compiled regression verifies
support, zero preservation, nonzero response, MPI parity and default Gaussian
compatibility with the previous executable. This changes diagnostic initial
data only; it does not alter the evolution or boundary equations. The
[longer local compact controls](compact-theta/README.md) show that a small
incoming trace is also acquired from initially zero boundary support. Its
approximately quadratic amplitude scaling distinguishes it from the original
linear growing mode; it must not automatically be treated as a numerical
defect or removed by resetting fields.

## A coupled boundary reference

The [full-source analysis](theory/README.md) retains the constraint, gauge and
radiation fields together. Its exact exterior Dirichlet-to-Neumann map
excludes previously demonstrated positive boundary roots. A simpler fitted
map and a truncated exterior closure were rejected when new positive roots
appeared. Stable auxiliary poles alone do not establish a stable boundary.

The [coupled exterior prototype](robust-boundary/README.md) tests a compatible
discrete reference, including original RK3 and bulk sources. It is a linear
planar pilot with finite-exterior memory, not an installed AthenaK CPBC.
Large transient amplification, sampled tangential frequencies, exterior size,
convolution cost, corners and refinement coupling remain explicit limits.

## Remaining gates and reproduction

A 232-block static-SMR centered trumpet fixture passes an eight-rank,
three-cycle zero/pulse preflight. All residual and ghost entries stay exactly
zero in its zero case; the pulse has finite physical response. Its endpoint
is only .075 M. [Strong-field plan](strongfield-plan/README.md) records the
mesh, full ghost validation and deliberate invalid-interface rejection test.
Long strong-field stability remains untested with this candidate. A
[guarded two-node Aurora draft](strongfield-plan/aurora-next/README.md) is
prepared but not submitted; renewed SSH access and environment checks are
required before its zero gate and long pulse can run.

Removing kappa damping does not remove the physical constraint equations,
but it also removes their bulk friction. For the frozen damped wave,
lambda=-sigma +/- sqrt(sigma^2-c^2*k^2); at small k the slow branch decays as
-c^2*k^2/(2*sigma). Stronger damping can slow long-wave relaxation. This bulk
relation is distinct from the positive boundary eigenvalues.

Scripts, exact inputs, lossless histories and hashes are included. Raw rank
checkpoints remain in the external local/Aurora study. Validators accept
`ATHENA_REGRESSION_PATH=/path/to/athenak/tst/regression` or discover it from
this checkout, and reject unsupported geometry. No failed run is restarted.
Nonlinear boundary validation, long strong-field/SMR evolution, then matter
and spinning-background tests remain required before production resumption.
