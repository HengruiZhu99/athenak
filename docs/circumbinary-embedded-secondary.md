# Cyclic zoom around a 100:1 embedded secondary

The pilot uses one torus centered on the binary barycenter, with inner edge at 12M, pressure maximum at 28M and outer edge at 63.49M. At binary separation 25M, the secondary orbits at barycentric radius 24.7525M inside the dense disk. AthenaK takes `q=m2/m1=0.01`, with total mass one and both aligned spins +0.7. The initial vertical-field vector potential uses radial power 2, falloff 1500M and a minimum target plasma beta of 100. The run uses a circular corotating frame and six diode boundaries. The initial disk is smooth and the secondary immediately perturbs it. This gives a startup test in accreting gas.

## Mesh, dwell times and controls

The domain extends to ±128M in each dimension. The primary retains spacing M/8, the secondary reaches M/1024 and the disk has a separate midplane refinement hierarchy. Mesh blocks have 16 × 16 × 8 cells. At the pressure maximum the initial mass-weighted median vertical MRI wavelength spans about 4.4 cells in the midplane. Production turbulence and flux saturation require a finer disk and longer evolution.

Only the secondary cycles. The four-zone sweep visits 0, 1, 2, 3, 2, 1 and returns to 0, with nominal dwells of 0.024, 0.012, 0.018 and 0.192M by zone. The two-zone alternative uses 0.080M on the finest mesh and 0.160M one level coarser. Both follow the same 0.94M interval, about 95 secondary masses or 0.00120 binary orbits. The short finest dwells deliberately expose transition errors. At the nominal outer mask radii of 10.8 and 2.7 secondary masses, the local time estimates from $t_{\rm dyn}=\sqrt{r^3/m_2}$ are 35.5 and 4.4 secondary masses. The two outer dwells span about 0.55 and 3.6 of these times. The two-zone run and the four-zone run share the same static, always-finest control.

The 1:1 timing control keeps the torus radii, binary separation, primary refinement throughout, and secondary-only sweep. The finest spacing is M/16 around both holes. The secondary is at radius 12.5M, close to the torus edge. The four-zone dwells, output intervals and total duration are scaled by secondary mass, giving 47.47M. The initial 1:1 secondary horizon interpolation stencil intersects the excision region at some angles and fails surface qualification. Only timing is taken from that pair. Each speedup compares cyclic and static runs at the *same* mass ratio and physical duration.

## Puncture stability

The first 100:1 pair used the equal-mass campaign's smooth puncture settings. Both meshes developed runaway gas density inside the secondary horizon. The cyclic run ended with 6.54 × 10²¹ times its initial grid rest mass, while the static run reached 7.87 × 10⁴⁹ times its initial mass by 0.162M. The constrained-transport divergence monitor alone missed the failure.

The pilot decks now impose the full hydrodynamic target inside boosted Kerr–Schild radius 1.0125 hole masses, with a smooth taper to zero weight at 1.35. The puncture flux mask extends to 1.5525, inside the modeled horizon at 1.7141. Magnetic damping is disabled. Both members of each timing pair use this prescription. On the first four-zone return, at $t\simeq0.282M$, the midplane density and densitized vertical field differ from the static control by 14.5% and 16.0% in area-weighted relative L1 norm over the annulus from 3 to 15 secondary masses. A ring is visible near the maximum zoom-mask radius.

## Measured cost and flux differences

All timing runs were sequential on StellarAI node sai-n01g7 with two B200s. The 100:1 static and four-zone pair was job 69577, the equal-mass pair was 69573 and the 100:1 two-zone test was 69582. The binary orbit period is 785.398M. The projected cost is twice the solver loop wall time in hours, divided by the number of binary orbits evolved. It includes the pilot output cadence and mesh transitions. Queueing, setup, production checkpoint traffic and long-term changes in the accretion flow lie outside this estimate. Each configuration has one timing measurement.

| Configuration | Steps | Loop seconds | Projected B200 GPU-hours per binary orbit | Speedup over static |
| --- | ---: | ---: | ---: | ---: |
| 100:1, fixed | 2,788 | 687.2 | 319.00 | 1.00× |
| 100:1, four zones | 745 | 195.8 | 90.87 | 3.51× |
| 100:1, two zones | 1,871 | 471.9 | 219.06 | 1.46× |
| 1:1, fixed | 2,067 | 407.3 | 3.74 | 1.00× |
| 1:1, four zones | 2,067 | 406.9 | 3.74 | 1.00× |

At 100:1 the four-zone median timestep increases from 0.00033720M to 0.00269383M. At 1:1 both runs take 2,067 steps at 0.02297047M, apart from final time clipping. The fixed fine mesh at the primary sets the equal-mass timestep. These comparisons use secondary-only zoom. The equal-mass campaign that cycles **both** holes uses a different mesh and schedule.

| 100:1 visit | Window in M | Horizon flux, cyclic | Horizon flux, static | Horizon difference | 20m₂ flux difference | 20m₂ net mass-rate difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Four zones, return 1 | 0.282–0.300 | 3.2685e-05 | 5.1133e-05 | -36.1% | -0.00153% | -0.00077% |
| Four zones, return 2 | 0.564–0.582 | 7.8963e-05 | 0.00012245 | -35.5% | -0.00018% | -0.00213% |
| Four zones, return 3 | 0.846–0.864 | 0.0001866 | 0.00020801 | -10.3% | -0.00082% | -0.00137% |
| Two zones, return 1 | 0.246–0.318 | 4.9125e-05 | 4.9211e-05 | -0.174% | -0.00003% | +0.00005% |
| Two zones, return 2 | 0.486–0.558 | 0.00010839 | 0.00010842 | -0.025% | -0.00003% | +0.00003% |
| Two zones, return 3 | 0.726–0.798 | 0.00017847 | 0.0001785 | -0.017% | -0.00002% | +0.00002% |

Across the corrected 100:1 histories, every monitored magnetic face was finite. The largest relative divergence was $4.63\times10^{-15}$ or smaller, and the grid rest mass stayed within 1.58 parts per million of its initial value. The two-zone schedule saved 31% while matching the mean horizon flux within 0.18% on each of the three returns. The four-zone schedule saved 72%, with much larger errors at the horizon. The transfer flux at 20 secondary masses agreed within 0.002% on all six finest visits. These are measurements against the matched static startup on this grid. Resolution convergence, late-time disk turbulence and horizon-flux saturation remain untested for this torus.

## Running the comparison

Generate each pair from the archived aligned-run parameters. The static and cyclic decks of a pair share the initial mesh, gas, field, sink, and diagnostics.

```sh
python3 scripts/circumbinary_mad/embedded_pilot.py decks/deep
python3 scripts/circumbinary_mad/embedded_pilot.py decks/shallow --shallow
python3 scripts/circumbinary_mad/embedded_pilot.py decks/equal --mass-ratio 1
```

Run the dynbbh executable on two B200s with one MPI rank per GPU. Within each results directory, put the solver standard output in `cyclic/log` or `fixed/log` alongside the histories and `bin/` slices. The shallow analysis can reference the deep pair's fixed directory. The comparison script checks completion, the initial mesh and fluxes, common endpoints and qualified flux samples. It interpolates the fixed series to the cyclic history times and uses trapezoidal averages over each finest visit. The plotted flux is half the unsigned magnetic flux in raw code units. H2 is a coordinate transfer sphere of radius 20 secondary masses. BH2_Kerr is the boosted component Kerr horizon.

```sh
for pair in deep shallow; do
    python3 scripts/circumbinary_mad/embedded_comparison.py "$pair" > "$pair/report.json"
done
python3 scripts/circumbinary_mad/embedded_comparison.py equal --timing-only > equal/report.json
python3 scripts/circumbinary_mad/embedded_figure.py deep figures --movie --summary equal shallow
python3 scripts/circumbinary_mad/embedded_figure.py shallow figures/two-zone --movie
```

The plot requires NumPy and Matplotlib. Movie output also requires imageio-ffmpeg.
