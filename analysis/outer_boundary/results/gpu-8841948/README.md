# GPU face-sponge control: job8841948

The resolved face sponge completes5000M with finite fields and valid full ghost-inclusive checkpoints, but retains a slow exponential residual mode. This is a mitigation in a displaced weak-field box, not a validated production cure.

The exact original production GPU binary (SHA256 `fb74375038ffdd5f132e5305b9299c9672803b37e7eaedaff8ebb19bb7ad143c`) runs four32³ blocks/four MPI ranks on one Aurora node. The box is x,y∈[0,2048],z∈[−1536,−512]M, dx32M. It contains no BH interior, AMR or physical star. The G1 adapted gauge, lapse damping0.1, κ1=0.1/κ2=0, η2, linear ghost extrapolation and original zero_rate boundary are retained. dt0.6M differs from production0.0375M. The pulse is unchanged between controls. This job does not test the subsequently fixed tangent RHS stencil.

The zero control remained exactly zero through3RK cycles/1.8M, with all4 matching per-rank checkpoint headers, finite payloads and SPD metrics including ghosts. The baseline then aborted at2375.4M/cycle3959 at active(2000,2000,−1040), rank3/gid3. Its first printed primitive-recovery failure is in the fourth +x ghost at(2160,2000,−1040), determinant−0.0275428, between neighboring history log times2361 and2362.2M. Buffering and missing per-error timestamps prevent assigning a sharper injection time. Its only checkpoint is initialt0; the checkpoint check therefore does not validate its late failed state.

The independent fresh sponge adds the existing smooth residual damping layer with width256M and rate0.02/M. It reaches exactly5000M/cycle8334, application and PBS exit0, in497.24 application-wall seconds. It did not stop at the24-minute cap. All4 final checkpoints match and remain finite/SPD including ghosts; all sampled bad-metric counts are zero. Final max|Theta| is4.80174e−12, ThetaRMS1.43944e−13, max|δα|4.26077e−10.

| Window (M) | Theta maximum γ/M⁻¹ | Theta RMS γ/M⁻¹ | lapse residual γ/M⁻¹ |
|---|---:|---:|---:|
|2000-3000|0.00124357|0.00127547|0.00126315|
|3000-4000|0.00131882|0.00135534|0.001336|
|4000-5000|0.00135959|0.00138587|0.00136412|

RMS is sqrt(the stored proper-volume squared-field integral divided by stored proper volume). The label `Theta-norm` is a truncated `Theta-norm2`; it has not already been square-rooted. Over4000–5000M, Theta maximum and lapse-residual fits have R²>0.999998. Theta maximum grows3.893× and RMS3.997×, giving e-fold times736M and722M. Finite target completion does not establish saturation.

The first-ghost observation, eventual active failure, and growing constraint norms diagnose different events. No saved spatial stage profiles are available in this old-binary GPU job to determine which operator first seeded its mode. The separate local forensic replay provides that information only for its smaller one-block proxy.

## Evidence

- `results.json`: full window fits, endpoints, checkpoint records and first-error localization.
- `histories.npz`: complete numeric user/Z4c/MHD history tables with column labels. Raw logs remain at the path in `provenance.json`; checkpoint checks are packaged separately.
- `gpu-sponge-comparison.png`: constraint and lapse amplitudes; the dotted line marks the baseline’s first recorded ghost-error bracket, not a measured exact onset.
- `plot.py`: portable NumPy/Matplotlib reproduction and validation of the stored late growth fits.

No production campaign, job or monitor was changed. Separate fixed-stencil GPU verification job8841975 in debug-scaling rejected a zero-test input before evolution. It provides no GPU verification result; corrected-input replacement is tracked in the main investigation report.

## Final spatial localization

Decoded checkpoint logical locations give the active Theta and lapse maxima at(1616,1904,−1008)M, rank3/gid3, absolutelevel1/relative0. They lie144M from the+y boundary, inside the256M sponge where sigma=0.00768055/M. They do not peak beyond its onset. The physical-ghost Theta maximum is2.07766e−12 at(−112,−112,−1648), a triple fourth-ghost corner; physical-ghost lapse maximum1.34153e−10 is at(1616,2160,−1008). Internal ghosts are reported separately and omitted from the active profile to avoid double counting.

The profile groups all active cells by nearest physical-face distance; it does not establish the mode’s initial origin. 99.134% of the final proper-volume Theta² integral is inside the layer. `spatial-profile.json` retains decoded geometry and signed values; the original checkpoint decoder and its hash are recorded in `provenance.json`. The active Theta² integral matches the final global history, and proper volumes agree to2.22e-16 relative.

The startup resolution diagnostic prints `dx_current=0 diameter_cells=inf` before mesh construction. This is not an evolved-field failure; array finiteness and explicit invalid-state markers were checked separately.
