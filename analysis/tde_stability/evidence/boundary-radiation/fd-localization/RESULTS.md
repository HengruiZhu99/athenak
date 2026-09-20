# First boundary injection and finite-difference compatibility audit

Read-only runtime replay and standalone operators; no AthenaK source or Aurora job was changed here. The saved v1 helper is deliberately used in the compiled first-stage replay so that it matches the actual v1 data. The later D4 inner-derivative accuracy repair is separately identified.

## First recorded amplification in the stage data

`snapshot_probe.cpp` replays the actual helper with the saved background, immutable volume RHS and exact initial lapse pulse from `../stages-v1/pulse_r1_t1`. The active domain is [-2,2]^3, dx=0.25, eight 8^3 blocks, four ghosts; the pulse amplitude is 1e-8, centered at (0.75,0,0) with compact radius 0.5.

At cycle0/stage1, the largest pre-boundary Theta RHS is 3.573465265e-11 at (0.625,0.125,-0.125), r=0.649519. After the boundary update it is 3.451395051e-10 at (1.875,-0.125,-0.125), r=1.883315, gid1, logical level1/relative level0. The latter is a physical exterior face, about 9.66 times the earlier volume maximum.

At that exterior cell the helper returns FTheta=0 and FQ=(-4.676248971e-10,4.697125111e-11,4.697125111e-11). The boundary adds Theta RHS 3.451395051e-10 and Khat RHS -1.178146234e-8. The scalar characteristic correction identities reproduce the actual saved pre/post RHS to 4.10e-27 for C1 and 5.38e-24 for C2. Thus the early amplification is not a scalar sign/index transcription error. The coupled characteristic solve can introduce Theta through its Gamma/gauge correction even when FTheta itself vanishes.

By cycle2/stage2 (cycle-start time0.075M), the largest post-boundary Theta RHS is 7.781071065e-10 at (1.875,0.375,-0.125), still on the outer face. These observations identify an early numerical injection/amplification step. They do not establish that it is the only eigenmode source or the physical origin of every previous long-run failure.

## Ghost failures occur later and are a different diagnostic

For v1 cubic ghosts, the first C2P invalid metric was logged at cycle-start9.7124999999999524M, cycle259, stage1, rank0/gid1, relative level0, cell(15,0,0), coordinate(2.875,-2.875,-2.875), ghost depths(4,4,4). Its second leading principal minor was negative despite a positive determinant. Active invalid metrics appear at14.025M.

For v1 linear ghosts, the first C2P invalid metric was at cycle-start22.350000000000133M, cycle596, stage1, rank0/gid1, relative level0, cell(15,11,9), coordinate(2.875,-0.125,-0.625), ghost depths(4,0,0). Its first minor and determinant were negative. Active invalid metrics appear at24M.

These are first recorded events per rank, not an inferred exact physical onset. A maximum in an active-cell history cannot detect an earlier ghost-only loss of positive definiteness. No valid evolution claim is based on positive determinant alone.

## What the separate radial finite-difference model shows

`radial_fd.py` discretizes the independently verified variable-coefficient spherical continuum volume system using D6 centered derivatives, the actual sixth-order shift upwind derivative, KO8 coefficient -epsilon/256 (epsilon=0.5), and linear/cubic four-ghost polynomial extension. It compares inner D2/D4 metric-defined Q, outer D2 transport, and the original p-only boundary correction. Analytic radial angular terms and an artificial inner gauge condition remain; it is not a Cartesian discretization of the background.

At outer radius4, cubic tau1 spectra are nearly neutral. Linear ghosts retain smaller positive branches, which decrease with the D4 inner derivative. At outer radius1.875, the largest reported linear branches are approximately0.095/M (D2 inner) and0.064/M (D4 inner), while cubic branches are about0.007–0.010/M. These scans do not reproduce the fast 3D cubic failure and cannot certify it stable.

The subsequent full Cartesian face/Fourier analysis in `../fd-symbol` does reproduce a fast oblique mode with nonzero normal shift. That is a stronger discriminator than further radial sweeps. The v2 one-block runtime failure also shows that internal block-edge stencil switching is not a sufficient explanation. See `../PUBLISHED_BOUNDARY_PLAN.md` for the matched published-boundary comparison and its negative direct-allocation result.

## Files

- `first-stage-helper.json`, `first-stage-maxima.json`: actual compiled-helper replay and scalar identity check.
- `early-stage-localization.json`: coordinates and pre/post maxima from saved RK data.
- `analyze_snapshots.py`, `snapshot_probe.cpp`: reproducible analysis and helper probe; `snapshot-build.json` records the build command.
- `radial_fd.py`, `radial-fd-scan.json`, `radial-fd-near-boundary-scan.json`: bounded negative compatibility models.

No binaries or large field snapshots are required in a compact evidence package.
