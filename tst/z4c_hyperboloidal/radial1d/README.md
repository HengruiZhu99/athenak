# Surgical accuracy/cost experiments, 2026-11-16

**Subsequent interpretation:** the [bulk–boundary analysis](../bulk_boundary/README.md)
identifies a constraint-compatible formal noninteger branch for increased lapse
damping at fixed shift restoration. Its predicted normal-derivative error
orders match the stronger-damping experiments. The coarse-grid gains below
remain observations, but the earlier recommendation to explore 0.375–0.5
does not imply improved asymptotic smoothness. Retain the original production
pair pending joint regularity and stability qualification.

The radial tests support cheaper **scri residual accuracy at coarse resolution**
from modest lapse-damping changes. They do **not** establish earlier entry into
a uniformly fourth-order regime. No production defaults or bulk operators have
been changed. The nine-point endpoint and compact damping-boost experiments
are not recommended for adoption.

## Provenance and scope

Production kernels: simulation commit `f90ab38472405907f21347afc19a7c9e90a1d665`.
The driver was adapted from Claude's untracked radial assessment in
`/Users/hz0693/research/hyperboloidal/.claude/worktrees/nonradiative-assessment`.
It uses the production point kernels, algebraic chart, endpoint compatibility
solve, penalties and normal lift, on a single +x radial line. The modifications
add an optional 7/9-point endpoint comparison and reproducibility metadata.
Default evolution and all saved profiles are byte-identical to the original
driver built against the same production headers in the N12 control test.

This is a reduced model: angular derivatives are reconstructed from radial
functions; there are no angular finite differences, seams, overlaps, physical
news calibration, or binary evolution. The inner five points are frozen and
the RHS is smoothly tapered near the inner boundary, r=0.5. Constraint results
below concern only 0.7 <= r <= 0.95, away from this boundary and the normal lift.
The profile writer's constraint columns use unlifted jets and must not be
interpreted as native constraints in the last two bulk layers.

Raw runs, original/modified source snapshots, executable hashes, commands,
wall times, the rejected compact-boost patch, and plots are preserved at:

`/Users/hz0693/research/results/nonradiative-surgical-20261116`

All runs were local. No cluster allocation was used. No paper claim has been
updated from these reduced-model experiments.

## Results

Here N denotes the equivalent outer-shell resolution, h=0.06/N, not the total
number of points in the radial domain. Physical time t/M=3t. Peaks use the
common native-time interval [0,0.8], excluding final-step overshoot. The
diagnostic is max over time and tensor components of |T_ab| at scri. Its target
is zero for a regular nonradiative solution; it is not a full-state error norm.

| N | Baseline damping 0.25 | Damping 0.375 | Damping 0.5 |
|---:|---:|---:|---:|
| 12 | 3.317e-2 | 1.728e-2 | 8.660e-3 |
| 16 | 2.181e-2 | 5.518e-3 | 2.804e-3 |
| 20 | 1.180e-2 | 2.875e-3 | 7.553e-4 |
| 32 | 3.159e-3 | 6.737e-4 | 2.144e-4 |
| 64 | 3.513e-4 | 6.461e-5 | 6.045e-5 |
| 128 | not rerun | not run | 1.974e-5 |

At N12/16/20, damping 0.5 reduces the control residual by 3.83/7.78/15.6.
However, its successive high-resolution orders are approximately 1.83
(32 to 64) and 1.61 (64 to 128). The impressive coarse-grid reductions are
not evidence for fourth-order asymptotics. At 32 to 64 the baseline order is
3.17 and the 0.375 order is 3.38. The 0.375 choice is a useful compromise to
test, not an established optimum; it has only received the short control sweep.

Wide-pulse data use amplitude 0.001, center 0.86, half-width 0.08, and
exp(-6z^2/(1-z^2)) for |z|<1. They include both the pulse and background
relaxation, without subtraction of a control.

| N | Wide pulse, damping 0.25 | Wide pulse, damping 0.5 |
|---:|---:|---:|
| 16 | 2.050e-2 | 2.818e-3 |
| 32 | 4.780e-3 | 8.227e-4 |
| 64 | 1.720e-3 | 2.261e-4 |

At N16 the 0.5 pulse run has a smaller residual than the baseline N32 run,
and takes 4.41 versus 17.25 seconds: a measured 3.91-fold reduction in radial
run time for this residual threshold. These are workstation timings with two
independent single-thread runs allowed concurrently, not a controlled 3D
performance benchmark. There is no extra stencil or stage in the constant
damping change. No 3D speedup is claimed.

The control sweeps at damping 0.25 and 0.5 also completed through 4.8 M at
N12/20/32/64. For 0.5 the full-window peaks are 9.565e-3, 7.553e-4, 2.144e-4,
and 6.045e-5. These are bounded-duration stability checks, not long-time
stability proofs. Halving dt changes the signed tensor history by at most
2.66e-10 for the N20 control and 4.13e-9 for the N16 wide pulse; clock differences
are at most 4.62e-10 and 8.46e-9. Histories are compared at common times.

Interior constraints improve much less. At the saved time nearest 2.4 M,
N20 max |H| changes from 6.868e-3 to 4.191e-3 and max |M_r| from 6.773e-2
to 4.094e-2. At N64 the respective pairs are 1.092e-4/6.700e-5 and
1.057e-3/6.288e-4. These are maxima on the fixed interior annulus, not
whole-domain or scri constraint norms. Smaller scri residuals therefore do
not justify reducing the grid by the same factor for every observable.

## Rejected alterations

1. Increasing the endpoint interpolant from seven to nine donors while
   retaining the same last-three-bulk-node closure footprint, normal lift and
   penalties raises the N12/N16 control peaks to 6.142e-2/4.164e-2. Increasing
   interpolation order alone does not improve this coarse-resolution problem.
2. A compact boost of 0.25 to the lapse damping, multiplied by
   S9((r-0.70)/0.10) S9((0.97-r)/0.07), leaves the asymptotic damping at 0.25.
   S9 is the existing clamped C4 smoothstep. Both the evolved gauge and
   prescribed stationary source used this profile. It gives little benefit:
   N12/16/20/32 control peaks are 3.429e-2/2.047e-2/1.166e-2/3.050e-3.
   The production header was restored after the experiment; the patch and
   exact test source remain in the raw-results directory. With zero boost,
   all saved N12 output is byte-identical to baseline.

## Follow-up: larger constant damping

Fifteen additional runs test 0.75 and 1.0 with the same executable, grid,
initial data, time window and wider pulse. The peak scri tensor residuals
over 0–2.4 M are:

| Damping | Control N16 | Control N32 | Control N64 |
|---:|---:|---:|---:|
| 0.5 | 2.804e-3 | 2.144e-4 | 6.045e-5 |
| 0.75 | 1.131e-3 | 3.703e-4 | 1.756e-4 |
| 1.0 | 3.167e-3 | 2.029e-3 | 1.191e-3 |

| Damping | Wide N16 | Wide N32 | Wide N64 |
|---:|---:|---:|---:|
| 0.5 | 2.818e-3 | 8.227e-4 | 2.261e-4 |
| 0.75 | 1.138e-3 | 3.624e-4 | 1.719e-4 |
| 1.0 | 3.160e-3 | 2.025e-3 | 1.189e-3 |

The 0.75 choice improves the coarse N16 data and this wide-pulse window,
but its control residual is 2.90 times larger than 0.5 at N64. Increasing
to 1.0 is worse for both families at N64: factors 19.7 and 5.26 relative
to 0.5. The last control doubling gives orders about 1.08 and 0.77 for
0.75 and 1.0, respectively. Neither an error floor nor eventual asymptotic
order is established by these limited resolution sequences.

All fifteen runs completed with finite scri histories. N32 controls were
also extended to 4.8 M; full-window peaks are 3.703e-4 for 0.75 and
2.029e-3 for 1.0. Halving dt for the N16 wide pulse at damping 1.0 changes
the tensor history by at most 2.09e-9 and the clock by 6.13e-9. Thus that
case's degraded accuracy is not repaired by the tested timestep reduction.
These are local radial results, not native 3D validation. A global increase
to 1.0 is not recommended on this evidence. Keep 0.375–0.5 as the primary
candidates; 0.75 is an optional comparator, not a uniform improvement.

## Interpretation and next native test

The existing input parameter `outer_lapse_damping` changes the constant part
of xi in `hyperbolic_shift.hpp`. The lapse pole contains
`-2*xi*alpha*(alpha-reference_alpha)`. The production stationary source and
scri limit already use this parameter consistently. Changing it modifies the
continuum gauge and boundary coefficients; it leaves the interior principal
derivative coefficients, finite differences, independent variables, normal
lift, penalties, RK stages and KO operators unchanged. Initial radial field
values were identical in the baseline and 0.5 N20 runs.

More precisely, combining the bare lapse pole with the prescribed stationary
source gives the following change at fixed fields when kappa_alpha is changed:

    Delta(dt alpha) = -2 W Delta(kappa_alpha) alpha (alpha-alpha_ref) / Omega.

Here W is the existing outer-gauge blend weight and alpha_ref is the prescribed
stationary conformal lapse. This follows by combining the xi-dependent pieces
of `DonorHyperbolicGauge` with `lapse_offset` and `lapse_linear` in
`StationaryGaugeReference::At`. The unadjusted donor uses (1+r^2)/2 as its
reference; the source correction is essential to the full interpretation.
At scri alpha_ref=1 and its radial derivative is 1. Alpha=1 is already
enforced algebraically at the boundary. The finite contribution to the free
lapse clock is +2 kappa_alpha (alpha_n-1), as recorded explicitly in
`scri_rate_diagnostic.hpp`. Increasing the coefficient therefore changes the
nearby relaxation and sensitivity to normal derivatives, even though the
boundary lapse value itself remains exactly 1.

Use 0.375 and 0.5 as opt-in candidates, keeping 0.25 as the control. Do not
change a shared default or resume an old checkpoint under a changed gauge
identity. A native comparison should regenerate initial checkpoints with the
candidate input and preserve executable/input provenance.

The next decisive experiment is a two-node debug-queue N16/N24/N32 control
and wide-pulse comparison, with a fixed physical time window, native Hamiltonian
and momentum constraints, full scri tensor, lapse clock, false-news modes and
measured time-to-tolerance. Follow a successful nonradiative comparison with a
resolved nonspherical perturbation and a nonzero-news calibration. The radial
surrogate cannot establish preservation of a physical waveform.

If high-order convergence, rather than a coarse error threshold, remains the
limiting requirement, first investigate the endpoint lapse/trace boundary
layer and the chart-dependent angular error. A consistent chart-permutation
test is a useful discriminator before altering the chart. These experiments
give no reason to replace the bulk scheme or widen the endpoint stencil.

## Reproduction

From a checkout of the stated production commit containing this directory:

```sh
sh tst/z4c_hyperboloidal/radial1d/build.sh EXISTING_REPO_WITH_build-binary OUTPUT_BINARY
python3 tst/z4c_hyperboloidal/radial1d/surgical.py OUTPUT_BINARY NEW_RESULTS --phase screen
python3 tst/z4c_hyperboloidal/radial1d/surgical.py OUTPUT_BINARY NEW_RESULTS --phase validate
python3 tst/z4c_hyperboloidal/radial1d/surgical.py OUTPUT_BINARY NEW_RESULTS --phase refine
python3 tst/z4c_hyperboloidal/radial1d/surgical.py OUTPUT_BINARY NEW_RESULTS --phase stronger
python3 tst/z4c_hyperboloidal/radial1d/summarize_surgical.py NEW_RESULTS
```

The build script reads the existing Kokkos build but compiles against this
checkout's production headers. The launcher refuses to overwrite an existing
case. NumPy and Matplotlib are required for summarization. The optional third
build argument selects a source snapshot for parity tests. Compact-boost
reproduction additionally requires applying the archived patch in an isolated
checkout and using the archived `run_boost.py`; it is excluded from the
recommended default source.
