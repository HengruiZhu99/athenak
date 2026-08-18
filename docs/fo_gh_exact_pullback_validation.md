# Exact-pullback FO-GH validation record

## Controlling outcome

**FORMULATION NOT ESTABLISHED.**  The exact transformed improved-driver
regularity audit fails before implementation of a new Einstein evolution
system.  Consequently the hyperbolicity, frozen-spectrum, G0--G4, new-vs-old,
and AMR promotion gates were intentionally not run.  This document separates
the corrected-source control evidence from the analytic formulation stop.

## Corrected-source control provenance

- branch parent: `3a9ba3bb5997e3d3071fed875b2fb0a1672303a8`;
- control source: a tracked archive of that exact commit;
- Kokkos gitlink: `6739bc623081648af9e752b616d9671527922cbf`;
- scratch: `/pscratch/sd/h/hzhu/fo-gh-exact-pullback-20260818.20260818T063217.173370`;
- build allocation: `57208476`, nodes `nid001145,nid001148`;
- control allocation: `57208600`, nodes `nid001145,nid001148`;
- QOS: `gpu_interactive` routed through the live `interactive` request;
- compiler: Cray `cc`/`CC` host wrappers with Kokkos `nvcc_wrapper`;
- CUDA/Kokkos: CUDA 12.9, Kokkos 4.7.2, `Cuda` + `Serial`, `AMPERE80`;
- MPI: Cray MPICH with `Athena_ENABLE_MPI=ON` and
  `MPICH_GPU_SUPPORT_ENABLED=1`;
- executable SHA-256:
  `8f6be381da9f553022dc70db1426049b5fac422c9c211355c1cca5c12d4e041b`.

All compilation occurred inside an allocated compute-node `srun`; the recorded
build host was `nid001145`.  The eight control ranks were mapped explicitly by
`SLURM_LOCALID` to eight distinct A100-SXM4-40GB UUIDs, four on each node.

## Grid evidence

For 32, 48, and 64 active cells per MeshBlock, the committed input reproduced
the same tree:

| physical level | MeshBlocks |
|---:|---:|
| 0 | 56 |
| 1 | 56 |
| 2 | 56 |
| 3 | 64 |
| total | 232 |

The domain is \([-32M,32M]^3\), with finest coverage throughout
\([-2M,2M]^3\), `nghost=4`, and finest spacings \(1/16\), \(1/24\), and
\(1/32M\).  AthenaK printed the correct tree and wrote `mesh_structure.dat`,
then its one-rank CUDA mesh-only teardown segfaulted for each resolution.  This
is recorded as a mesh-only teardown defect, not an evolution result.

## Old-formulation control

The control uses RK4, fourth-order finite differences, CFL 0.025,
\(\epsilon_{KO}=0.02\), and unchanged \(\kappa=\mu_H=\eta_H=1\),
\(\eta_\beta=2\).  Outer faces at 32M exclude boundary arrival before 4M for
the observed maximum characteristic speed near 0.964.

| resolution | \(\Delta x_{min}\) | last regular step | first collapsed step | outcome |
|---|---:|---|---|---|
| coarse | 1/16 | cycle 1650, 3.413010M, `dt=2.066873e-3` | cycle 1660, 3.431611M, `dt=0` | failed before 4M |
| medium | 1/24 | cycle 1530, 2.108827M, `dt=1.377991e-3` | not reached | stopped by request while finite |
| fine | 1/32 | not run | not run | campaign stopped before launch |

The coarse result is bit-for-time identical to the archived pre-diagnostic-fix
failure.  This is expected because commit `3a9ba3bb` repairs only common ADM
diagnostics and does not feed the evolution RHS.  The regenerated common ADM
histories use fixed coordinate regions and the repaired non-diagonal operator;
the old archived common H/M histories remain invalid.

The unmodified control does not emit a full first-bad-cell state/RHS record.
A diagnostic-only `fo_gh/fail_closed_dt` option was added and verified in a
local Serial build: its default-zero path completed a 97-cycle smoke test, and
an intentionally high threshold produced a nonzero fail-closed exit with a
`FO_GH_FIRST_BAD_STATE` record.  It was not rebuilt or replayed on Perlmutter
after the formulation stop, so no GPU first-bad-state claim is made.

The campaign was stopped at the user's request. Allocation `57208600` and its
medium-resolution step were cancelled; no production or stability claim is
made for medium, fine, or the requested 20M interval.

## Exact-driver audit

Command:

```sh
PYTHONPATH=tst/test_suite python3 \
  tst/test_suite/fo_gh/exact_driver_pullback_audit.py \
  --samples 256 --maximum-n 64
```

Results:

```text
weighted_driver_dense_oracle=PASS
maximum_relative_mismatch=1.20738018701417634e-15
regular_gauge_target_oracle=PASS
gauge_target_maximum_relative_mismatch=4.20925626520668695e-16
normal_h_projection_term_power=-2.182000
required_z_perp_power=-2.182000
FORMULATION NOT ESTABLISHED
```

The expected process exit status is 2.  Direct invocation of all three test
functions in `test_exact_driver_pullback_audit.py` passed.  The local system
Python lacks an importable `pytest`, so no pytest-runner success is claimed.

## Gate ledger

| gate | evidence | status |
|---|---|---|
| exact regular gauge-target identities | 256 random states | pass |
| exact standard-to-weighted driver RHS | dense 4x4 matrix oracle, 256 states | pass |
| every driver intermediate finite at trumpet | power audit gives \(-2p\) | **fail** |
| regular \(z_\perp\) evolved field | exact stationary balance requires \(-2p\) | **fail** |
| parent-to-regular Einstein RHS | stopped before implementation | not run |
| regular principal symbol/symmetrizer | singular map blocks gate | not run |
| frozen Fourier spectrum | earlier gate failed | not run |
| smooth-equivalence suite | earlier gate failed | not run |
| G0--G4 and new puncture runs | prohibited after stop | not run |

## Exact control build and launch commands

```sh
export NVCC_WRAPPER_DEFAULT_COMPILER=CC
cmake -S . -B build_control -DCMAKE_BUILD_TYPE=Release \
  -DPROBLEM=built_in_pgens -DAthena_ENABLE_MPI=ON \
  -DCMAKE_C_COMPILER=cc \
  -DCMAKE_CXX_COMPILER="$PWD/kokkos/bin/nvcc_wrapper" \
  -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON
cmake --build build_control -j 32

export MPICH_GPU_SUPPORT_ENABLED=1
srun -N 2 -n 8 -c 32 --gpus-per-task=1 --gpu-bind=none \
  build_control/src/athena --kokkos-map-device-id-by=mpi_rank \
  -i inputs/fo_gh/fo_gh_puncture_compare_smr.athinput \
  time/tlim=4.0 \
  mesh/nx1=128 mesh/nx2=128 mesh/nx3=128 \
  meshblock/nx1=32 meshblock/nx2=32 meshblock/nx3=32
```

Replace both triples by `192/48` and `256/64` for medium and fine.
