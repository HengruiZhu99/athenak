# Photon-ring parameter searches

This directory runs a sequence of AthenaK evolutions, classifies collapse or dispersion, and follows the inner and outer marginal surfaces back toward formation. One Slurm dispatcher drives either a width scan or an initial ADM spin scan. Each survey has its own directory on a filesystem shared by the login and compute nodes. The solver, MPI launcher, GPU visibility, queue resources and environment modules belong to the site profile.

## Build and initialize

Use a double-precision MPI build of this AthenaK branch and its Kokkos submodule. The following commands run from the repository root. They show a CUDA build and a spin scan. Choose the architecture for your GPU and load the site's compatible MPI toolchain first. For a host build, omit the CUDA and architecture flags. Use the existing CMake backend options for other accelerators. The sparse vertex exchange defaults to host staging. Set `-DAthena_VC_DEVICE_MPI=ON` when the MPI implementation supports device pointers. Other AthenaK communication paths may also require GPU-aware MPI.

```bash
git submodule update --init --recursive
scripts/photon_ring/build.sh build-ring \
  -DCMAKE_CXX_COMPILER="$PWD/kokkos/bin/nvcc_wrapper" \
  -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON
code="$PWD/scripts/photon_ring"
python3 "$code/init.py" "$SCRATCH/ring-spin" --name ring-spin --axis spin \
  --athena "$PWD/build-ring/src/athena" \
  --calibration "$IRISK/scripts/photon_torus" \
  --operators "$IRISK/build/src/iris_lobatto_operators"
```

Use Python 3.11 or later. Compute stages need NumPy and SciPy. The dispatcher uses the standard library. The IrisK directory supplies `solve.py`, `measure.py` and `spectral.py`. For the original wide-ring section use the production IrisK solver with the local lower bound on the circumferential radius over the launch support. An existing campaign's `calibration` directory and its `read.sh` reader can be used as the two initialization arguments; copy the associated `48.dat` alongside that reader. Keep the operator and the AthenaK binary accessible to compute nodes. Initialization pins the executable by SHA-256.

Choose `--axis width` to vary the coordinate minor radius at fixed initial ADM mass and spin. Spin scans keep the coordinate width, Eulerian photon energy, energy-weighted coordinate centroid and angular spreads fixed while calibrating the measured initial ADM spin. Both axes use the supplied reflected 288-square SMR deck, 36-square blocks, 24 by 48 angular grid, and 0.0625 spacing on the initial support. The checkpoint and mesh checks are specific to this family.

## Configure and monitor

1. Edit `cluster.json` in the campaign directory. Set the Slurm flags under `common`, `evolve`, `fit` and `calibrate`. Set `launcher` to an argument array preceding the executable, for example `["srun","--ntasks=4","--gpus-per-task=1","--gpu-bind=single:1"]` for a site with one visible GPU per rank, or `["mpiexec","-n","4"]` for a CPU build. Set the allocation to match. Inspect the site-specific MPI binding before a production GPU run. For StellarAI pass `--profile "$code/profiles/stellarai.json" --site "$code/profiles/stellarai.sh"` at initialization. Other sites can supply their own `site.sh` to load modules, select a Python environment via `PHOTON_PYTHON`, or set MPI variables.
2. Edit `scan.json` before dispatch. Choose distinct decimal strings for the coarse samples, a positive parameter tolerance, and either or both objectives, `threshold` and `onset spin`. Give concurrently active campaigns distinct job prefixes. Optional `archived` entries map a parameter string to an existing read-only case directory with its validated final checkpoint and geometry. Fresh campaigns evolve all coarse points.
3. Run the dry check and inspect the proposed stages. Then invoke the same monitor without the flag from the campaign directory. It locks per campaign, checks the queue and job journal, validates published checkpoints, and retries recoverable Slurm interruptions. A failed science stage retains its scratch directory for diagnosis.

```bash
cd "$SCRATCH/ring-spin"
python3 "$code/advance.py" --dry-run
python3 "$code/advance.py"
```

For hourly dispatch, use the site's permitted scheduler on a login node where `squeue`, `sacct` and `sbatch` work. For example, the following crontab entry uses absolute paths and a login shell. Replace the paths with those for this campaign. A site can also run `advance.py` from a systemd user timer. Slurm is the implemented queue backend; the physics stages can be run under another scheduler through `stage.sh`.

```cron
0 * * * * /bin/bash -lc 'cd /shared/ring-spin && python3 /shared/athenak/scripts/photon_ring/advance.py' >> /shared/ring-spin/monitor.log 2>&1
```

To run a single already-prepared stage inside an allocation, use `"$code/stage.sh" "$PWD" "$code/worker.py" 0.9508`. The replay stages take `replay.py PARAMETER plan|evolve|reduce`. Use one worker per case. The monitor writes `targets.json` and `survey-state.json`. Each case retains its initial-data metadata, segment histories, checkpoints and replay result.

## Interpretation and verification

Bisection uses adjacent, deterministically classified outcomes and decimal midpoints. Ambiguous samples interrupt a bracket. The maximum-spin search places a golden-section probe adjacent to competitive measured peaks. It ranks the enclosing horizon on the first verified strictly trapped slice and uses fit-order and temporal differences as error estimates. Several local maxima or a spatial-resolution error can limit that inference. A fold reconstructed between samples is recorded separately in `replay/turning-point.json`.

The worker keeps one-mass segments and the original two-mass post-event tail. Replay uses the latest retained checkpoint preceding the birth window, or regenerates the missing prefix from the original handoff. During the dense window it checks finder and ADM output every five cycles. The science diagnostics, strict-surface checks and paired order-40/48 fits are common to all cluster profiles.

Run `python3 "$code/test_campaign.py"` to exercise both planners, dispatch recovery, shell argument passing and the two launcher profiles without a scheduler. The repository's `boltzmann_equatorial_amr` CTest exercises native parity, AMR, MPI and restart. The historical files under `tst/unit/boltzmann/equator/smr` document the earlier fixed-bracket campaign.

### Horizon formation diagnostics

On the eight-GPU reference run at initial ADM spin 0.9672875, the centered finder took 175.6 seconds across 1,065 calls over 53,255 cycles. Evolution wall time with finder calls subtracted averaged 0.143 seconds per cycle, including output. At the measured call cost, using the finder every five cycles throughout that trajectory would add about 23% to this baseline before additional geometry output or offline fits. Production uses a 50-cycle survey and retains **every** one-mass segment restart. Near a classified collapse the separate replay uses five-cycle finder calls and matching native geometry.

The reduction runs through the completed two-mass production tail. In `replay/result.json`, `formation_history` records the reconstructed fold, the first resolved marginal pair and the first slice with a verified strictly trapped witness between the two surfaces. `replay/horizon-history.csv` contains one row per branch per sampled epoch. It records signed `zeta = 8πJ/A`, area, signed horizon angular momentum, Christodoulou spin, and the principal stability eigenvalue both in code units and multiplied by the square of the initial ADM mass. The stability operator uses outward unit-normal displacements in the spatial slice and the outgoing null normal n+s.

Each row identifies the replay or production geometry. The outer fit can continue independently after the inner branch stops resolving. This continuation requires a positive principal eigenvalue, consistent order-40 and order-48 shapes, at least 20 cells across the minimum coordinate radius at order 48, and whole-meridian expansion checks. Unresolved rows have empty measurement fields. An empty strict-witness field means the witness was not tested on that epoch. Reconstructed values between evolved slices carry their own entry and fit variants. Run `python3 "$code/test_horizon_history.py"` with NumPy and SciPy to check the branch-loss and replay-to-production handoff.
