# Paused 20M FO-GH/Z4c campaign: compact evidence

The campaign was paused after all three FO-GH resolutions developed a
resolution-worsening timestep collapse. No Z4c production case ran. The
Perlmutter allocation was released; large restart and field outputs are omitted.

| case | dx_min | valid 2M restart | timestep collapse |
|---|---:|---|---:|
| coarse | 1/16 | yes | 3.431611M (`dt=0`) |
| medium | 1/24 | yes | 3.024995M (`dt=0`) |
| fine | 1/32 | yes | 2.658676M (`dt=3.66e-212`) |

The earlier failure at higher resolution is a failed stability/convergence
gate. It is consistent with a formulation or high-frequency semidiscrete
instability, but does not prove the cause. The faces are at 32M and the
pre-collapse maximum characteristic speed is about 0.964, so a simple
center-to-face estimate is about 33.2M; boundary arrival cannot explain failure
before 3.5M.

## Provenance and passed launch gates

- scratch: `/pscratch/sd/h/hzhu/fo-gh-20m-20260817.57196231`
- allocation `57196231`; nodes `nid001133,nid001136`; eight A100 40GB GPUs
- production source `c7f3950e`; Kokkos gitlink
  `6739bc623081648af9e752b616d9671527922cbf`
- Cray CUDA-aware MPI device-buffer ring passed across both nodes with eight
  distinct GPU UUIDs and `MPICH_GPU_SUPPORT_ENABLED=1`
- exact eight-rank Minkowski was zero; the 232-block tree assigned 29 blocks/rank
- direct/restarted startup checkpoints were bit-for-bit identical
- one/eight-rank comparison passed; worst relative difference `1.014e-14`
- peak memory was about 9.0, 20.5, and 39.6 GiB/GPU

`z4c_start_gpu_dmon.txt` and `z4c_start.log` record a scheduler wait only. The
Z4c executable never launched; they are not Z4c runtime evidence.

Reproduce the compact CSVs with:

```sh
python3 tst/test_suite/fo_gh/analyze_fogh_instability.py \
  docs/fo_gh_artifacts/perlmutter_20m_20260817_partial \
  --output-dir docs/fo_gh_artifacts/perlmutter_20m_20260817_partial/analysis
```

The common unmasked ADM momentum columns are identically zero despite nonzero
native FO-GH momentum histories. A post-campaign audit confirmed uninitialized
tensor reads in the common ADM operator; both its H and M columns in this bundle
are therefore invalid and retained only as failure evidence. The repaired
operator has ordering-sensitive non-diagonal and curvilinear flat-space
regressions, but these histories have not been regenerated. The diagnostic did
not feed evolution and cannot explain the FO-GH collapse. Do not resume the
campaign before the remaining formulation review in
`FORMULATION_CODE_REVIEW.md` is resolved.
