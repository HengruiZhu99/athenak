# Matched head-on BBH comparison

This directory contains the post-processing script for the deliberately
low-resolution Z4c/PC-GH comparison in
`inputs/z4c/twopuncture/bbh_headon_{z4c,pcgh}.athinput`.

Both inputs use the same TwoPunctures data (two nonspinning, zero-momentum
punctures of bare mass 0.5 at x = +/-2.5), the domain `[-16,16]^3`, RK3 with
CFL 0.2, sixth-order finite differences with four ghost cells, chi-based AMR,
and initial level-4 refined boxes around both punctures.  The finest spacing is
M/16 and the waveform is extracted at coordinate radius 8M.  All evolution
constraint-damping coefficients are zero.  PC-GH additionally projects its
continuum-zero GH gauge constraints after each RK stage and its defining
first-derivative constraints after each complete step; these are discrete
constraint projections, not damping terms.
The PC-GH comparison uses `z4c_mp_hyperbolic`: the direct `z4c_mp` realization
has a defective characteristic shell at `alpha*chi=2/3` (initially near
`r=7.15M`), unacceptably close to the `r=8M` extraction sphere. The switched
shift retains the moving-puncture limit while completing the characteristic
basis throughout the `0 < alpha <= 1` puncture domain.
Both formulations use the same shift-integrated coordinate puncture tracker,
with positions written every completed RK3 step.

After building AthenaK with `PROBLEM=z4c_two_puncture`, run each input from a
separate directory so their `waveforms/` output directories do not collide.
For example:

```sh
mkdir -p runs/z4c runs/pcgh
(cd runs/z4c && ../../build/src/athena \
  -i ../../inputs/z4c/twopuncture/bbh_headon_z4c.athinput)
(cd runs/pcgh && ../../build/src/athena \
  -i ../../inputs/z4c/twopuncture/bbh_headon_pcgh.athinput)
```

Generate the comparison artifacts with:

```sh
python3 analysis/pc_gh_bbh/plot_comparison.py \
  --z4c-history runs/z4c/bbh_z4c.z4c.user.hst \
  --pcgh-history runs/pcgh/bbh_pcgh.pcgh.hst \
  --z4c-wave-dir runs/z4c/waveforms \
  --pcgh-wave-dir runs/pcgh/waveforms \
  --z4c-trackers runs/z4c/bbh_z4c.co_0.txt runs/z4c/bbh_z4c.co_1.txt \
  --pcgh-trackers runs/pcgh/bbh_pcgh.co_0.txt runs/pcgh/bbh_pcgh.co_1.txt \
  --pcgh-boundedness runs/pcgh/bbh_pcgh.pcgh-boundedness.dat \
  --output-dir runs/comparison
```

The domain is intentionally too small for precision wave physics: a radial
null signal can travel from the outer boundary to the extraction sphere in
about 8M.  Treat the plots as a workflow and boundedness qualification, not a
converged waveform result.
