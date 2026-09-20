# Early strong-field SMR response: job 8843091

This is a frozen collection made at **2026-09-20 21:31:21 UTC**, while the pulse was still running in Aurora's `debug` queue. Histories extend to **61M**; validated active-cell Theta snapshots extend to **50M**. This is not a final job result or long-term stability clearance. No job configuration, executable, submission, restart, or scheduler state was changed by this analysis.

The exact-zero control passed its submitted all-24-rank checkpoint validator at **0.075M / 3 cycles**: 232 blocks, all payloads finite, zero invalid metric cells including all four ghost layers, and every residual component exactly zero. The authoritative completed-gate record is [zero/run-validation.json](zero/run-validation.json). A second independent zero read exceeded the tool's output wait, and its output was not recovered; we make no additional validation claim from that attempt. That reader had finished before the completed pulse checkpoint was inspected.

The completed **50M / cycle 2000 pulse checkpoint independently passed** the level-aware all-24-rank validator: matching headers, all 232 blocks and payloads finite, no invalid full metric cells in active or ghost zones, and nonzero residual evolution. Raw full metric minima were lapse 0.0976791, chi 0.00954121, first conformal metric minor 0.999999999838, second minor 0.999999999919, and determinant 0.9999999999999997. See [pulse-checkpoint-50M.json](pulse-checkpoint-50M.json). This checks an already completed checkpoint while the application continues; it is not a terminal application check.

The run uses the Schwarzschild trumpet, background-adapted gauge, sixth-order volume derivatives, static refinement with dx = 1, 0.5, 0.25, and 0.125M, and outer `zero_rate` characteristic treatment with linear ghosts. Kappa1 = 0, shift eta = 0.02, lapse-residual damping = 0.01. The outer sponge starts at 8M, ramps smoothly over 20M, and reaches rate 0.05/M. No residual-field zeroing or damping is applied inside the horizon. The fluid atmosphere is present but its stress-energy feedback is disabled. The pulse perturbs the lapse with amplitude 1e-8 around x = 2.5M and width 0.75M.

The early pulse response rises and then falls. Maximum absolute Theta reached **1.42646e-9 at 15M**, and was **2.05816e-10 at 61M**. The lapse residual fell from **9.78948e-9** to **8.11617e-12**. At 61M, exterior proper-volume Theta RMS was **3.40330e-12**, and the metric-invalid history counter remained zero. This short transient does not show a sustained growing envelope; it cannot exclude a later unstable mode.

The raw Hamiltonian maximum is already approximately **1.32972e-4 at r = 1.06617M at t = 0**, and is nearly unchanged at 61M. It is dominated by the discretization error of the analytic background. The exterior integral of H squared changed by at most 22.9 parts per million over this interval and by −0.348 parts per million at the last row. This comparison of scalar norms is **not** a calculation of the spatial field H(t)−H(0), so its near-horizon maximum must not be labeled a newly localized instability.

The profiler validated all **24 rank files in each of 11 complete Theta cohorts**, at 0, 5, ..., 50M: identical time/cycle/input headers, finite complete payloads, matching physical and logical geometry, expected rank/gid ownership, and all 232 unique blocks covering the domain exactly. No snapshots were excluded. These float32 dumps contain active cells, not ghosts; their regional RMS values use coordinate volume and must not be equated to proper-volume history RMS.

| Time | Maximum absolute Theta | Peak (x, y, z)/M | r/M | Rank / gid / relative level |
|---:|---:|---|---:|---|
| 5M | 2.27643e-10 | (3.5625, −0.1875, −3.6875) | 5.13071 | 5 / 50 / 3 |
| 10M | 3.18569e-10 | (7.625, −7.875, −2.625) | 11.2715 | 5 / 46 / 2 |
| 15M | 1.42646e-9 | (−7.125, −0.125, −0.125) | 7.12719 | 2 / 20 / 2 |
| 25M | 4.20831e-10 | (−0.6875, −0.0625, −0.1875) | 0.715345 | 3 / 28 / 3 |
| 50M | 6.56714e-10 | (3.1875, −0.0625, −0.0625) | 3.18873 | 6 / 54 / 3 |

The first nonzero saved frame at 5M has its peak 0.3125M (2.5 fine cells) from the nested ±4M refinement cube. At 10M the peak is 0.125M from the ±8M cube. Later peaks move into the central region and sometimes inside the horizon. These positions warrant matched interface controls if growth appears, but they do **not** identify the first injection or prove that refinement creates an unstable mode. At 5M the outer-face band maximum is only 3.78e-20; by 50M it is 1.58e-12.

The latest sampled throughput window (47.5–60M) is **0.242 seconds/cycle**, or **9.69 seconds/M** at dt = 0.025M. At that rate another 940M would take approximately **2.53 hours**, an extrapolation exceeding this job's remaining one-hour allocation. A clean walltime stop would not constitute reaching 1000M.

- [summary.json](summary.json): frozen histories, throughput, raw-background comparisons and limitations.
- [theta-profiles.json](theta-profiles.json): all validated snapshots, maxima with coordinates/ownership/level, radial bins, horizon/sponge/outer-face and interface bands.
- [early-response.png](early-response.png): early gauge and constraint response.
- [profile_theta_smr.py](profile_theta_smr.py): read-only 24-rank binary profiler; `analyze_snapshot.py` reproduces the local summary and figure.

The nested cubic interface bands have half-width two local cell widths. They overlap radial bands. No stage-level injection audit, full spatial Hamiltonian subtraction, or final pulse completion check has been performed in this packet.
