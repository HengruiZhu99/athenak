# Della R=128M head-on BBH comparison

## Outcome

- Z4c completed the requested evolution to `t=100M` (`8000` cycles).
- PC-GH stopped at `t=73.79991M` when the strict diagnostic found a non-positive
  conformal metric.  It was not restarted or tuned after this failure, as requested.
- The physical merger waveform is present for both formulations at extraction radii
  `8M`, `12M`, `24M`, and `32M`.  PC-GH data later than retarded time `u=t-r=35M`
  are increasingly contaminated by the numerical instability and must not be
  interpreted as a physical ringdown.
- The long binary test therefore reinforces the existing classification
  `PARTIAL IMPROVEMENT`; it is not a production qualification of PC-GH.

## Configuration and provenance

Both runs used the same committed executable, source commit
`b81b44d658f3b81584e94ce79b92656c112ff908`, a Cartesian domain
`[-128M,128M]^3`, finest spacing `dx=M/16`, sixth-order spatial derivatives,
RK3, `CFL=0.2`, and extraction radii `8,12,24,32,48,56M`.

Every face is `outflow`, not periodic.  In this code path, `outflow` applies the
formulation's Sommerfeld RHS at the physical boundary and fills ghost zones with
the selected polynomial extrapolation.  Both inputs set `extrap_order=2`.
The exact used inputs, executable/input hashes, source commit, and source status
are retained under `raw/*/used_input.athinput`, `raw/*/provenance.sha256`,
`raw/*/git-commit.txt`, and `raw/*/git-status.txt`.

The original remote run directories are:

- `/scratch/gpfs/FPRETORI/hz0693/pcgh-z4c-gpu-r128/z4c-r128-t100`
- `/scratch/gpfs/FPRETORI/hz0693/pcgh-z4c-gpu-r128/pcgh-r128-t100`

The portable local copy intentionally excludes restart files and the CUDA core
dump.  It contains 42 Z4c slice files (21 state plus 21 constraint slices at
`t=0,5,...,100M`), 30 PC-GH slice files (15 plus 15 at `t=0,5,...,70M`), complex
waveforms at all six extraction radii for each formulation, histories, boundedness data,
inputs, logs, and provenance.

## Evolution and constraint behavior

Z4c remained stable.  At `t=100M`, its volume RMS Hamiltonian and momentum
constraints were `1.70697e-5` and `1.22632e-5`.  Their run-wide maxima were
`3.54809e-4` at `t=4.375M` and `1.40358e-4` at `t=6.5M`.

PC-GH initially tracks Z4c well, but its constraints turn upward after about
`40M`:

| time | Hamiltonian RMS | momentum RMS |
|---:|---:|---:|
| 30M | 1.05846e-5 | 7.42850e-6 |
| 40M | 1.56256e-5 | 1.95140e-5 |
| 50M | 5.63424e-5 | 1.04681e-4 |
| 60M | 2.85797e-4 | 6.21830e-4 |
| 70M | 2.25634e-3 | 4.42374e-3 |
| 73.7603M | 5.77223e-2 | 5.57140e-2 |

The strict diagnostic then failed at
`(x,y,z)=(-0.34375,-0.96875,-0.03125)M` with
`det(gTilde)=-20.19875` during the post-RK update at `t=73.79991M`.

The concrete precursor is growth of the first-order `Q` reduction/curl sector on
fixed refinement interfaces.  From `t=50M` to `70M`, the maximum curl-Q constraint
has an exponential e-folding time of about `5.82M` (doubling time `4.03M`).
It grows from `0.1425` at `50M` to `0.6632` at `60M`, `6.0715` at
`69.754M`, and `18.8909` at `73.2009M`; the minimum conformal-metric eigenvalue
falls to `0.3652` and the maximum gradient RHS reaches `42.36` by the latter time.
At `69.754M`, `dRQ_prolong_max=2.1761e-2` and
`dRQ_reduction_project_max=1.0043e-1`; at `73.2009M` they are `1.3456e-1`
and `1.9963`, respectively.  The slice maps show square, grid-aligned rings on
the inner fixed-refinement interfaces.  There is no simultaneous dynamic-AMR
event.  This identifies an SMR prolongation/reduction-projection feedback loop,
not the outer boundary, as the late failure mechanism.

There is no honest local one-line repair for this instability: it calls for
constraint-preserving prolongation or a redesign of the first-order reduction
exchange.  No damping, floor, KO, or resolution tuning was added to conceal it.

## Z4c constraint spikes

The Z4c Hamiltonian spikes are puncture/grid-phase sampling artifacts.  The
history reduction includes a cell only when `chi >= 0.0625`; as a puncture moves
across cell centers, the sharp excision-mask boundary changes the high-curvature
cells entering the norm.  For the 16 successive peaks while
`0.75M <= |x_p| <= 1.75M`, the median puncture displacement between peaks is
`1.0695` finest cells and the mean is `0.9963` cells.  None of the largest
adjacent jumps brackets a dynamic-AMR change.  The spikes therefore do not mark
a continuum instability or regrid event.

## Puncture trajectories

PC-GH tracker positions were not serialized in the restart format used by this
run, so its ODE track jumps back toward the initial position after a restart.
The plotted PC-GH track is instead reconstructed from the two lapse minima in
each saved xy slice, exactly as requested.  The Z4c slice-minimum trajectory is
overlaid with its continuous ODE track.

At every common saved time through `70M`, the PC-GH and Z4c slice-minimum tracks
agree to at most `0.001225M`, far below the finest-cell width `0.0625M`.
The Z4c slice position differs from its continuous ODE position by at most
`0.0361M`, consistent with cell-center quantization.

The restart defect is fixed for future runs by commit `5733d131`: PC-GH tracker
positions are now written and read only when the new restart header flag is
present, preserving compatibility with older restart files.  Serial and two-rank
MPI restart regressions agree bit-for-bit.  The final CUDA tree was rebuilt
successfully with `make -j32`; this production run was not repeated.

## Waveform interpretation and symmetry

The early pulse (`u<20M`) is initial-data/gauge junk.  The coherent pulse in
`20M <= u <= 35M` is the physical merger signal.  In that window, the PC-GH
real-mode peak differs from Z4c by `-2.46%`, `-2.56%`, `-12.86%`, and `-15.96%`
at `r=8,12,24,32M`; peak-time offsets are at most `0.50M`.  At `r=48M` the
physical window is incomplete, and at `r=56M` it has not arrived before the
PC-GH stop.

For an exactly reflection-symmetric head-on collision, the imaginary `(2,2)`
mode should vanish.  The observed nonzero values are numerical symmetry leakage,
not a physical polarization.  In the merger window, `||Im||_2/||Re||_2` for
Z4c is `1.20e-4`, `3.54e-5`, `5.73e-5`, and `1.17e-4` at the four inner radii;
for PC-GH it is `1.41e-4`, `1.01e-3`, `3.33e-2`, and `1.95e-3`.

Two concrete symmetry defects were found in the code audit.  The direct
`z4c_mp` PC-GH gauge has a defective characteristic surface at
`alpha*chi=2/3`; the production input therefore uses the complete
`z4c_mp_hyperbolic` switch.  PC-GH reflection parity had also been treated as
scalar for all components; commit `25bed824` supplies correct scalar, vector,
tensor, `Q_kij`, and `B_i^j` parities.  The latter bug does not affect these
full-domain outflow runs, but it is fixed for symmetry-reduced runs.  The later
PC-GH imaginary-mode growth is spatially and temporally consistent with the same
SMR reduction instability described above.

## Products

- `plots/puncture_trajectory_overlay.png`: lapse-minimum trajectories, with Z4c
  ODE track on top
- `plots/puncture_trajectory_from_slice_minima.csv`: reconstructed positions
- `plots/waveform_22_merger_window_overlay.png`: uncontaminated physical window
- `plots/waveform_22_radii_overlay.png`: complete available histories, including
  visibly contaminated late PC-GH data
- `plots/waveform_22_radius_consistency.png`: retarded-time/radius comparison and
  imaginary-mode leakage
- `plots/constraint_overlay_with_amr.png`: constraints and all AMR changes
- `plots/pcgh_instability_slices_t50_t70.png`: Hamiltonian, curl-Q, and minimum
  metric eigenvalue on the xy slices
- `plots/long_comparison_summary.json`: machine-readable numerical summary
