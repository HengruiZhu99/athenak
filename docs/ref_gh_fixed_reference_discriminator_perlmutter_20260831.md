# Ref-GH fixed-reference causal discriminator

## Scope and claim boundary

This report records the bounded follow-up to the moving-reference versus
hard-freeze experiment.  It asks whether the reduction/curl growth is caused
by a fixed intermediate blended reference, a mode seeded during motion, or the
reference stopping transient.  It does not tune parameters or change the
equations.

Both new cases retain STANDARD Phi ordering, the relative-damped gauge, FD4,
RK4, CFL 0.05, KO 0.02, gamma0=gamma2=1, and the existing fixed-core blend and
window.  This is one-resolution causal evidence, not a convergence campaign,
formal stability analysis, or stable/convergent trumpet evidence.

## Source and validation

The work started from pushed commit
`fb6da483428eecc38774aa26085efb8c2ddc53fe`.  Three source commits implement
the discriminator:

- `551806232976f84021a36d1382654baf623d10f2`: construct the fresh physical
  Schwarzschild wormhole and project its metric and first derivatives into a
  static xi=0.25 reference;
- `360b424f48e8256c3031785047ec25977dc1a47c`: add the analytic smooth-stop
  profile and its source-unit oracle; and
- `47e57894df8547318132b3a78ad3271c118c2f4a`: make nested Perlmutter launch
  steps safe while retaining the same run sequence.

The direct-fixed initialization uses `ProjectPhysicalMetricToReference`; it is
not an input-only reference change.  A deterministic 64-sample, off-axis
source-unit oracle verifies:

| check | A100 Linf result |
|---|---:|
| coordinate physical metric | 1.89399e-16 |
| coordinate first derivatives | 3.96853e-16 |
| lapse/shift/spatial metric | 2.22045e-16 |
| reference temporal jet | exactly 0 |
| nontrivial first-order-state scale | 2.07388e-1 |

The runtime initialization independently reports physical state error zero,
coordinate metric error 7.77e-16, lapse error 5.55e-16, shift error 7.01e-18,
first-derivative error 8.55e-16, and reference temporal jet exactly zero.  The
smooth-stop oracle checks the analytic integrated quintic profile, including
its mixed derivatives, with derivative error 4.55956e-10; its final reference
jet is exactly zero and a moving sample has nonzero jet 2.89527e-6.  A fresh
post-run Release/Serial rebuild and the complete source-unit suite also pass
without tolerance changes.

The numerical executable used source commit `47e57894...`.  A later
bookkeeping-only cleanup removes misleading pre-task `initial_GH_L2`,
`initial_reduction_L2`, and `initial_curl_L2` fields: the attempted calculation
occurred before the output task made those values authoritative.  The actual
t=0 history values below, not the zeros printed in the historical GPU run log,
are the constraint evidence.

## Perlmutter provenance

- Allocation: `57786944`, account `m3328_g`, QOS `gpu_interactive`
- Nodes: `nid001089`, `nid001152`, `nid001153`, `nid001172`
- Devices: 16 distinct NVIDIA A100-SXM4-40GB UUIDs, four per node
- Runtime: CUDA 13.2, Kokkos 4.7.2 CUDA+Serial/AMPERE80, Cray MPICH 9.1.0
  with `libmpi_gtl_cuda.so.0`
- Executable SHA-256:
  `80481d854bfe8932ba5dd2b66b89d3e1bccd3a95159a5ed4607833cde1eed4d9`
- Domain/tree: `[-24M,24M]^3`, 328 MeshBlocks distributed 20 or 21 per rank,
  physical-level counts 208/56/64, finest h=1/16M
- Direct fixed: fresh t=0 to 4.2M, 1017 cycles, exit 0, bad-state count 0
- Smooth stop: common clean t=2M checkpoint to 5.2M, 1260 total cycles,
  exit 0, bad-state count 0
- Scheduler: allocation completed with exit `0:0` in 00:51:14 and was released

All 16 rank/device mappings, build configuration, exact commands, inputs, and
compact output hashes are in the artifact bundle.  The smooth branch retained
the seed basename for its power history; therefore its raw `run_status.txt`
incorrectly says `latest_power_history_time=0`.  The retained file
`refgh_reference_motion_seed.ref_gh_power.hst` demonstrably reaches 5.2M and
was consumed by the analysis.  The harness now discovers the history by
suffix rather than assuming the case basename.

## Direct fixed xi=0.25 result

The continuum projection is roundoff accurate, but the discretely sampled
fresh state starts with a large constraint impulse:

| t=0 whole-domain RMS | direct fixed | clean moving state at t=2M | ratio |
|---|---:|---:|---:|
| GH | 8.04854e-3 | 1.05902e-4 | 76.0 |
| reduction | 2.61305e-4 | 1.74349e-5 | 15.0 |
| curl | 7.14034e-4 | 3.09458e-4 | 2.31 |

This violates the clean-comparison premise in the controlling decision tree.
The impulse initially decays: log-linear slopes over 0.15--1.0M are
-0.5023/M, -0.4006/M, and -0.2328/M for GH, reduction, and curl.  A slower
growth sector subsequently appears.  Fits over 2.8--4.2M give:

| quantity | slope /M | R2 |
|---|---:|---:|
| GH RMS | +0.18451 | 0.9990 |
| reduction RMS | +0.28461 | 0.9476 |
| curl RMS | +0.32220 | 0.9890 |
| Pi RHS Linf | +0.50270 | 0.9448 |
| Phi RHS Linf | +0.28707 | 0.9913 |
| frame-correction source | +0.61384 | 0.9999 |

At 4.2M, GH/reduction/curl RMS are 6.37444e-3, 1.52194e-4, and
1.13438e-3.  Their localized maxima occur at r=0.736M, 0.409M, and 0.379M;
the Pi RHS, Phi RHS, and frame-correction maxima occur at r=0.533M, 0.409M,
and 0.409M.  The fixed blend region is therefore suspect, but the contaminated
initial state prevents classifying its operator as independently unstable.

## Smooth-stop result

The secondary case forks the clean moving trajectory at 2M with xi=0.25 and
xi_dot=0.125/M.  It smoothly brings xi_dot to zero over 2--3M using an
integrated quintic profile, reaches xi=0.3125, and then holds xi, xi_dot, and
xi_ddot exactly fixed.  No state reprojection or Phi-ordering switch occurs.
Across all 222 retained post-stop max-location records, both reference frame
and reference connection time-derivative sectors are exactly zero.

The fit excludes the stopping interval and begins at 3.8M, 0.8M after the
analytic stop.  Over 3.8--5.2M:

| quantity | slope /M | R2 |
|---|---:|---:|
| GH RMS | +0.45463 | 0.9997 |
| reduction RMS | +0.45602 | 0.9851 |
| curl RMS | +0.42111 | 0.9882 |
| Pi RHS Linf | +0.57336 | 0.9591 |
| Phi RHS Linf | +0.59401 | 0.9850 |
| frame-correction source | +0.62087 | 0.9987 |

At 5.2M, GH/reduction/curl RMS are 1.05669e-3, 4.98175e-4, and
3.97149e-3.  Their maxima localize at r=0.409M, 0.311M, and 0.540M;
Pi RHS, Phi RHS, and frame-correction maxima localize at r=0.471M, 0.540M,
and 0.409M.  Thus reduction/curl growth continues after all prescribed
reference temporal sectors have become exactly zero, in and around the same
fixed-core/blend region.

For context, endpoint values at the common comparison time 4.2M are:

| case | GH RMS | reduction RMS | curl RMS |
|---|---:|---:|---:|
| continued motion | 4.13903e-3 | 9.80800e-4 | 1.27852e-2 |
| abrupt hard freeze | 1.26001e-2 | 2.99474e-4 | 2.78592e-3 |
| smooth stop | 6.66224e-4 | 3.33558e-4 | 2.47747e-3 |

Smooth stopping avoids the abrupt hard-freeze GH impulse and greatly reduces
all three norms relative to continued motion.  It does not eliminate the
post-stop reduction/curl mode.

## Conclusion and smallest next step

The discriminator is **ambiguous** under the required decision tree.
Ongoing reference motion is a strong amplifier, and an abrupt stop injects a
separate GH/gauge transient, but ongoing motion is not the sole cause: the
clean smooth-stop branch continues to grow after every reference temporal
sector is exactly zero.  These data cannot distinguish:

1. a slower instability of the fixed intermediate blended operator; from
2. a persistent reduction/curl mode seeded before or during the moving
   transition.

The contaminated direct-fixed run supplies suggestive localization, not the
missing causal separation.  Correlation and co-localization are not proof of
formulation instability.

The smallest next step is a remote, equation-level/discrete-operator audit of
the fixed blended region around r=0.3--0.6M, together with an
initialization-only gauge- and reduction-compatible projection oracle that
preserves the same physical metric and first derivatives without the observed
discrete t=0 impulse.  No additional evolution, tuning, or resolution ladder
should precede that audit.

No stable or convergent trumpet evolution is claimed.

## Evidence locations

Compact evidence is committed under
`artifacts/ref_gh_fixed_reference_discriminator_perlmutter_20260831/`.
Large checkpoints remain on Perlmutter under:

`/pscratch/sd/h/hzhu/refgh-fixed-reference-20260831.uk93aa/fixed_reference_discriminator_57786944`

The reused clean t=2M checkpoint is:

`/pscratch/sd/h/hzhu/refgh-reference-motion-freeze-20260831.7Y5n8O/reference_motion_freeze_57779143/seed_to_t2/rst/refgh_reference_motion_seed.00001.rst`

Its SHA-256 is
`68ab3a2dcfddebb79065e923b330faaa993c71fa023930a63f4180894dcbb279`.
The per-case `restart_sizes.tsv` files record all uncommitted checkpoint paths
and sizes.  The approximately 10.24 GB restart files are not committed.
