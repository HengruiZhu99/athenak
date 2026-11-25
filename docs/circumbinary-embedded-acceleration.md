# Accelerating disk relaxation at 100:1

These sweeps use the same embedded secondary, circular 25M orbit, vertical-field torus, physical interior excision and two B200s as the [initial comparison](circumbinary-embedded-secondary.md). The solver accepts separate parameters for the radius where it holds gas primitives fixed and the distance over which it searches for mesh blocks to coarsen. Scaling the first radius down while scaling the search factor up keeps the same refinement footprint and allows more of the accretion flow to evolve. The constrained-transport field remains active.

## Locating the timestep limit

In two 20-step probes the radii, dwell prescription and mesh were identical. The second probe added the primary as a zoom center. Both used one accepted step per intermediate zone. The table gives the actual timestep and the spacing of the midplane leaf containing each hole, verified from slices. Rates during transitions are unsuitable for estimating a complete relaxation run.

| Coarsenings from finest | Secondary spacing (M) | Primary spacing, secondary-only / both (M) | Timestep, secondary-only / both (M) |
| ---: | ---: | ---: | ---: |
| 0 | 1/1024 | 1/8 / 1/8 | 0.000337 / 0.000337 |
| 6 | 1/16 | 1/8 / 1/8 | 0.02096 / 0.02096 |
| 7 | 1/8 | 1/8 / 1/8 | 0.04060 / 0.04060 |
| 8 | 1/4 | 1/8 / 1/4 | 0.04947 / 0.07674 |
| 9 | 1/2 | 1/8 / 1/2 | 0.04947 / 0.14485 |

At the outermost level the spacing at the pressure maximum on the positive x axis remains M/2. Deeper sweeps also coarsen neighboring blocks around the embedded secondary. The sampled disk radius alone cannot establish that all of the disk is unaffected.

## Frequent returns with a smaller frozen region

The short runs start on the original 2,872-block finest mesh. The base frozen radius is 0.3375 secondary masses. The three- and four-level variants have maximum frozen radii of 1.35 and 2.7 secondary masses. Both dwell 0.014M on the finest mesh and 0.084M on the coarsest mesh, passing through each intermediate zone in one step. The matched fixed control ran for the same 0.94M with the same output cadence. Each visit uses trapezoidal time averages over adjacent qualified samples. The last four-level return has one sample and supplies no time average.

| Sweep | Observed returns | Speedup | Secondary horizon unsigned-flux error | Secondary horizon net mass-rate error |
| --- | ---: | ---: | ---: | ---: |
| Three levels | 9 | 2.60× | within 0.38% | within 3.6% |
| Four levels | 9 | 3.67× | within 9.4% on 8 measured visits | up to 57.2% on first visit |

The original finest primary grid fails the horizon stencil qualification. Its 5-primary-mass transfer sphere supplies a separate observable.

## Deep relaxation and controls

For the primary horizon comparison, an additional refinement level gives M/16 on finest visits. The secondary retains M/1024. The initial mesh has 4,160 blocks. A short ten-level return is matched against continuous finest evolution through 0.94M on this mesh. The control and both-hole run share the physical excision settings and the history and slice cadence.

Longer runs evolve for 52M. They spend 6M at the outer level and 0.060M at the finest level on each sweep. Intermediate levels take one accepted step in each direction. The ten-level sweep freezes a sphere of radius 2.281M around each active center at the outer dwell. This encloses the primary horizon when both centers participate. A seven-level control holds the primary resolved and limits the secondary frozen radius to 0.285M. The common disk and output cadence allow comparisons of elapsed time and the evolved flow. The secondary transfer sphere at 20 secondary masses lies inside all three outer masks. Horizon and transfer samples must pass native coverage checks individually.

At the first short ten-level return both modeled horizons passed the stencil test. Relative to the continuous finest control, the primary horizon net mass rate was 1.17% higher. The primary flux was negligible in both runs. The secondary horizon flux was 90.33% lower and its mass rate was 77.65% lower. The 20-secondary-mass transfer sphere also showed a 19.0% flux deficit and 69.8% mass-rate deficit. A resolved stencil is necessary to measure the returned state, but the state also needs time to recover physically.

In the matched 52M ten-level runs, coarsening the primary reduced loop time from 1,163.14 to 716.59 seconds. Each completed eight returns on the same two B200s. Holding the more finely resolved primary fixed capped the timestep at 0.02474M. Coarsening it lifted the cap to 0.14485M. The common qualified interval on the primary's 5-mass transfer sphere totaled 46.45M. Over that interval, the both-hole run gave a 3.65% higher net mass inflow and 5.68% lower unsigned flux.

At 52M, the area-weighted relative L1 differences between those two midplanes were 0.62% in density and 1.62% in vertical field at barycentric radii 12–24M. At 24–40M the differences were 0.017% and 0.057%. Within 3–6M of the primary, the corresponding differences were 45% and 32%. These are spatially resolved differences on a common raster. An integral over azimuth can be much less sensitive to local errors.

The seven-level control reached the same time in 1,328.01 seconds. Compared against that smaller-mask control, the ten-level, two-hole run at 52M differs by 1.48% in density and 2.31% in the stored vertical field at 12–24M, by 0.85% and 1.11% at 24–40M, and by 3.92% and 4.37% within 2.5–5M of the secondary. The two masks produce distinct local accumulations. Midplane peak density reached 90.5 in the seven-level control and 2.49 in the two-hole run, with each peak lying near its own secondary mask edge. Peak absolute face fields over the runs were 56.7 and 1.80, respectively. Neither run recorded a nonfinite face. These differences preclude interpreting the masked interior as a resolved accretion solution.

The midplane comparisons sample a common Cartesian raster with 0.125M pixels. Each native leaf contributes its cell values from the requested equatorial slice. The sampled cell-center height and resolution therefore vary with the leaf hierarchy. Differences include both dynamical and finite-resolution effects.

## Wall time and longer outer residence

| Configuration | Time evolved (M) | Finest returns | Steps | Measured seconds | Projected B200 GPU-hours per orbit |
| --- | ---: | ---: | ---: | ---: | ---: |
| Always finest | 0.94 | 0 | 2,788 | 919.8 | 426.98 |
| Seven levels, secondary | 52 | 8 | 4,126 | 1328.0 | 11.14 |
| Ten levels, secondary | 52 | 8 | 3,756 | 1163.1 | 9.76 |
| Ten levels, both, 6M outer dwell | 52 | 8 | 2,075 | 716.6 | 6.01 |
| Ten levels, both, 24M outer dwell | 198 | 8 | 3,083 | 934.7 | 2.06 |

Lengthening the outer dwell to 24M reached 198M (0.252 binary orbits), with the same 0.060M finest dwell. At least 97.7% of simulated time was covered by consecutive history samples in the same outer visit. Both horizons qualified on every finest sample and were masked at the outer level. Over the full run, the largest relative CT divergence was 1.60e-14, the peak absolute face field was 6.06, and the maximum fractional change in grid rest mass was 1.37%. Open boundaries and physical sinks contribute to this grid mass change.

At the common 52M comparison, the longer dwell changes the density and vertical field relative to the 6M dwell by 0.13% and 0.42% at 12–24M, and by 0.034% and 0.13% at 24–40M. Against the seven-level control, the differences are 1.51% and 2.42% at 12–24M, 0.87% and 1.12% at 24–40M, and 4.43% and 4.88% at 2.5–5M from the secondary. The 24M-dwell maps are interpolated between the 51.095M and 52.543M slices during a single outer visit. The seven-level control also uses zoom. Continuous finest evolution supplies an independent short comparison through 0.94M. The 198M run covers disk startup. Long-term inflow equilibrium and magnetic saturation require longer runs.

The projected cost multiplies solver loop seconds by two GPUs and divides by the measured number of binary orbits and 3,600. It includes actual mesh changes, history diagnostics and midplane writes. The long dwell advances simulated time 207.3× faster than the 0.94M finest timing would project, and 2.92× faster than the 6M two-hole dwell. The always-finest estimate extrapolates a separate short run with denser physical output. Each configuration was timed once. StellarAI jobs 69672, 69673, 69680, 69681, 69682 and 69703 ran sequentially on node sai-n01g7, each using two B200s and two MPI ranks. [Numerical results](circumbinary-embedded-acceleration.json)

To reproduce the decks and analyze the native output, place logs, histories, and slices in one subdirectory per generated case. Supply the original fixed pilot output as the second analysis argument.

```sh
python3 scripts/circumbinary_mad/embedded_sweeps.py decks
python3 scripts/circumbinary_mad/embedded_sweep_analysis.py runs original/fixed > measurements.json
```
