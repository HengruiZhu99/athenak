# Wider and gentler residual-sponge screen

The most promising **screened** combination is uniform `kappa1=0`,
`eta=0.02`, lapse-residual damping `0.01`, and a residual sponge rising smoothly
from radius512M to1792M with maximum rate0.001/M, inside a boundary at2048M.
The volume gauge remains G=1. In the flat face-normal analogue at dx64M,
the largest real eigenvalue is below3.1e-15/M at each of four sampled tangential
wavenumbers. This is a candidate for nonlinear validation, not a stability proof.

With the original eta2/lapse0.1 and uniform kappa0, the same profile retains
a mild k=0 mode, gamma=1.21324e-6/M: approximately6.25% exponential amplification
over50000M. It meets the requested doubling budget
`ln(2)/50000 = 1.38629436e-5/M`. No nonzero-kappa or wide-taper case tested here
met that budget across its sampled modes.

## Model and scope

This is analysis only. No AthenaK source, campaign records, executable, or
scheduler job was changed. The copied operator is the original production
`zero_rate` closure, with G1, sixth-order volume derivatives, KO8 coefficient0.5,
linear ghost extrapolation and the active D2 boundary derivative. All20 active
linearized fields are retained. Inactive auxiliary B fields are omitted.
The reference is frozen flat alpha=chi=1, beta=0; it has no background gradients,
matter, black-hole interior, radial curvature, AMR or MPI interfaces.

The sponge is applied to all residual fields before CPBC, matching production.
For distance d to the nearest normal boundary and layer onset/end distances
d_start>d_end, its coefficient is

```
s = clamp((d_start-d)/(d_start-d_end),0,1)
sigma = rate * s^3 * (10-15s+6s^2)
```

The matched radial normal profile has d_start=1536M and d_end=256M:
on the radial normal through the origin this is start512/end1792 at R2048.
The interior1024M interval has no sponge. This matches the one-dimensional
profile only; the two transverse directions remain Fourier modes, not a sphere.
Uniform kappa0 also removes damping from the core. It must not be confused with
the separately tested taper that preserves kappa0.1 in the core.

Dense eigenproblems are split by exact z-reflection parity into14- and6-field
sectors. Numeric characteristic eigenvectors leave cross-sector entries below
1.1e-18; a full20-field reference spectrum reproduces the leading value within
1.3e-15. Default matrices match the archived reference bitwise. Source-affinity,
variable-kappa local terms and the added eta parameter are checked in
`study_regression.py`; results are in `default-regression.json`.

## Fixed-domain width/rate screen

At fixed L8192M, dx128M and kh=pi/32, with original kappa0.1/eta2/lapse0.1:

| Width [M] | rate.001/M | rate.005/M | rate.02/M |
|---:|---:|---:|---:|
|512|.00147617|.00140073|.00155462|
|1024|.00142167|.00110875|.00123950|
|2048|.00129681|.00055830|.00065551|

These are growth rates in1/M. A broader layer helps, but simply lowering the
rate is not monotonically better and none of these rates is near the budget.
At width2048/rate.005, reducing kappa0.1 to0.03 or0.01 gives gamma=.00022538
or.00016750 at the same k. Nonzero damping can substantially delay the failure
without making growth mild enough for50000M. This screen is adaptive; the
width256 cases were tested at dx32 with kappa0, rather than treating a two-cell
layer at dx128 as a resolved width comparison.

## Uniform kappa0: width, resolution and domain checks

With eta2/lapse0.1, L4096M and dx32M, kh=pi/64:

| Width [M] | rate.001/M | rate.005/M |
|---:|---:|---:|
|256|1.7283e-6|2.3355e-6|
|512|no resolved positive|1.0155e-6|
|1024|no resolved positive|no resolved positive|

Longer wavelengths expose a small surviving scalar mode for width512/rate.001:
the maximum sampled growth is at k=0, gamma=1.699018e-6/M, decreasing to
1.66474e-6,1.56155e-6 and1.14808e-6 at kh=pi/512,pi/256 andpi/128. The sampled
pi/64,pi/32,pi/16,pi/8,pi/4,pi/2 andpi modes have no resolved positive value.
Its Theta peak lies48M from the face, with99.9978% of Theta squared inside the
layer. Its weighted full-state peak is144M from the face, with84.66% inside.

The k=0 rate is1.71218e-6 at dx64 and1.69902e-6 at dx32, with all physical
parameters and L4096 held fixed. This is not disappearing rapidly with
resolution. Increasing the domain alone is also not monotonic: at dx64,
L4096 to8192 raises this mode from1.71218e-6 to5.20105e-6, still below budget.

For a genuinely wider layer2048M, rate.001, L8192, dx64 and uniform kappa0,
the largest sampled rate is2.98221e-7 at k=0, only1.50% amplification over50000M.
At fixed physical k=pi/8192, dx128 to64 changes gamma from1.57468e-7 to1.52388e-7.
The undamped interval is4096M; this result does not rely on damping the whole
domain. The sampled kh=pi/64 andpi/8 at dx64 have no resolved positive value.

Applying the width512/rate.001 source after CPBC does not remove its surviving
mode: k=0 gamma becomes1.78077e-6, compared with1.69902e-6 before CPBC. Both
are exactly zero-preserving. No further source-order scan was justified.

Two-dimensional combined-corner controls with kappa0, width512/rate.001 and
L2048 have no resolved positive eigenvalue at dx256 or128. These are coarse
two-/four-cell-layer checks, not production-resolution corner validation.
A matched coarse kappa0.01 control remains strongly unstable, gamma=.00086052.

## Wide kappa taper is still unstable

In the R2048/start512/end1792 normal profile, tapering kappa smoothly from0.1
in the core to0 by1792M, alongside a rate.005 sponge, yields gamma=.00025834,
.00047619 and.00079592 at kh=pi/64,pi/32 andpi/8. The surviving mode is scalar;
Theta peaks at |x|992M and99.616% of Theta squared lies in the transition/layer.
Weighted full-state amplitude peaks at |x|736M. The outer zero-kappa plateau
therefore leaves a growing mode within the transition rather than establishing
a stable core.
Mode localization is consistent with transition sensitivity; it does not
identify a particular variable-coefficient term as the sole cause.

## Final profile requested for the GPU comparison

All geometry/profile parameters below match the start512/end1792/rate.001
normal analogue at R2048 and dx64. Uniform kappa0; G1 and original zero_rate.

| kh/pi | eta2, lapse0.1 | eta0.02, lapse0.01 |
|---:|---:|---:|
|0|1.213243e-6|<3.1e-15|
|1/64|1.583912e-7|<8.7e-16|
|1/32|<3.0e-15|<8.0e-16|
|1/8|<1.2e-15|<1.4e-16|

A follow-up one-factor comparison (`gauge_factorial.py`, eight additional
spectra in `gauge-factorial/`) separates the two gauge coefficients. At k=0,
eta2/lapse0.01 retains gamma=1.213243e-6, while eta0.02/lapse0.1 has no resolved
positive value (largest real part1.913e-15). At kh=pi/64 their rates are
1.583915e-7 and1.274e-15, respectively; both mixed tuples have no resolved
positive mode at pi/32 and pi/8. Thus reducing **shift eta**, rather than lapse
damping, removes this mild mode at the four sampled wavenumbers. This attribution
is limited to the frozen flat normal-profile operator; it does not establish
nonlinear stability, spherical geometry, AMR, finite normal shift, or all
unsampled tangential wavenumbers. The directly evolved weaker-gauge candidate
still uses eta0.02/lapse0.01; a separate lapse0.1 ablation is being tested.

## Independent50000M evolution and nonnormality

`transient.py` propagates two directions with a sparse matrix exponential,
independent of the eigenvalue calculation. It uses the **width512/rate.001,
kappa0/eta2/lapse0.1,L4096,dx32,k=0** case, not the final weaker-gauge candidate.
The measured eigenmode grows1.088663615-fold through50000M, matching
exp(gamma*t) to2.2e-9 relative error. Its eigenpair residual is3.6e-16.
A compact pure-lapse initial direction instead reaches weighted full-state
norm0.27226 of its initial value at50000M. This direction was sampled every
10000M; these samples do not exclude earlier transient peaks. Amplitudes are
unit-normalized for this linear test and scale with the imposed perturbation.

The requested doubling budget bounds exponential mode growth, not transient
growth of a nonnormal operator. Two initial directions cannot bound all
perturbations, and neutral modes may still allow secular amplification.
No result here establishes a full nonlinear stability theorem.

## Files and reproduction

- `summary.csv`: all70 recorded spectra, parameters, growth and50000M factors;
  includes reference/check cases, not70 distinct nonlinear experiments.
- `results/`: original small JSON records; no matrices or full grids.
- `screen-summary.png`: visually checked white-background comparison.
- `manifest.json` / `reference-provenance.json`: model scope and source hashes.
- `strip_reference.py`: preserved unparameterized production-reference model.
- `strip_model.py`: analysis-only kappa, eta, source order and smooth-profile
  parameterization; no AthenaK source edits.

Run from this directory with NumPy, SciPy and Matplotlib installed:

```
OPENBLAS_NUM_THREADS=1 python3 study_regression.py
OPENBLAS_NUM_THREADS=1 python3 reproduce_shortlist.py
OPENBLAS_NUM_THREADS=1 python3 transient.py
OPENBLAS_NUM_THREADS=1 python3 summarize.py
```

The shortlist reproducer writes a separate folder and preserves existing files.

The eight gauge-factorial spectra supplement the original70-case manifest;
`gauge-factorial-provenance.json` records their inputs/source hashes. Raw
`rk3_growth` fields in the strip screens use fixed dt0.0375M and are **not**
checks of the actual GPUdt3.2M. Conclusions about modes here use the
semidiscrete eigenvalue field `real`; nonlinear timesteps are tested separately.
