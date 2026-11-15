# Full-overlap N12 rerun with scri recording

This directory records the completed fresh rerun of the latest validated
full-overlap binary through **59.9989173525 M0** (cycle 42603). `campaign.json`
contains its target cycle, step, source revision and executable hash.
The initial data, overlap strength/taper, dissipation, time step and optimized
GPU arithmetic are unchanged. `run.py` extends the copied input's time/cycle
limits and adds native scri output every `.01` code units (`.03 M0`).

## Completed evolution and radiation diagnostic

Seven sequential allocations completed the evolution in **5.99264 hours** of
summed segment execution time, excluding build, admission, queue and analysis
time. All segments exited successfully, their histories contained finite
numbers, and their checkpoint hashes and restart chain were verified. The
recorder supplied 2001 accepted-state records. The final checkpoint and
extraction provenance are recorded in [`final/run-result.json`](final/run-result.json).

The extracted curve is a **formal news diagnostic, not a validated physical
waveform**. See the [plot](final/news_diagnostic.png), [PDF](final/news_diagnostic.pdf)
and [quadrupole CSV](final/news20.csv). The first and last three samples are
excluded from the full-cadence extraction. Its transported time coordinate is
distinct from the native evolution time and is itself subject to the frame
regularity limitation below.

The peak formal `|N20|` is `0.0543002`. On common transported-time intervals,
changing the angular fit from L=8 to L=6 changes this mode by at most
`2.329e-6` (0.00429% of its peak); halving the recording sample rate changes it
by at most `1.101e-4` (0.203%). These are **postprocessing sensitivity checks**,
not evolution-resolution convergence or physical waveform error estimates.
The full conformal-Hessian trace-free defect reaches `0.0413262`, and the
transported round-cut RMS defect reaches `1.31866`. Extending the run has not
removed these regularity and frame limitations. We have not inferred a
physical strain, radiated energy or ringdown fit from this curve.

The final Weyl numerator probe samples through `59.7905 M0` and has maximum
pointwise magnitude `5.843e-4`; see
[`final/weyl-numerator-60M.json`](final/weyl-numerator-60M.json). This remains a
zero-target regularity diagnostic, **not** an extracted `psi4^0` waveform.

The final directory contains compact receipts, scri histories, extraction
summaries and plots. Complete mode archives and per-record extraction
diagnostics remain under the remote analysis directories named in the run
receipt; `download-manifest.json` records hashes of the downloaded artifacts.
All six extraction source files and both Weyl-probe scripts were checked
against the committed source. The evolution and analysis controllers have
finished; no continuation beyond the requested target was submitted.

## GPU admission

Job `9092194` compared the selected prior GPU executable against the cleaned
executable with recording enabled, for eight steps from both fresh initial
data and the original `26 M0` checkpoint. Both complete accepted restart
payloads were bitwise identical: **1,361,093,760 bytes per comparison**.
See `gpu-admission.json`. The isolated build reused only objects whose source
dependency hashes were unchanged; `build-receipt.json` records the
dependency audit, compile/link commands and executable digest.

`fresh-scri-admission.json` and `late-scri-admission.json` additionally compare
every recorded bulk, edge and endpoint value with its same-cycle checkpoint:
**423,026 scalar values per case**, all bitwise identical. Independent native
row and normal-lift reconstruction gave maximum errors of zero and
`1.55e-12`, respectively, below the `2e-11` admission tolerance. These checks
include all 24 MPI ranks.

The first science launch failed before evolution with a scheduler/process
launcher error, `No reply from shepherd after 108s` (exit 127). Job `9092227`
reused the successful admission and began the fresh evolution. Sequential
continuations use verified final checkpoints, with one science job active at a
time. The final segment and radiation-diagnostic receipts are in `final/`.

## Offline extraction

The reusable tools and conventions are documented in
[`scri-news.md`](../../scri-news.md). The analytic logs here were produced with
NumPy 2.1.3 and Python 3.11 on Aurora. The full pipeline recovered a known
quadrupolar news signal from angle-dependent native-time cuts with maximum
absolute error `1.7e-12`. This validates the manufactured extractor cases;
it does not establish convergence or vacuum regularity of the N12 evolution.

Bulk checkpoint and recorder archives remain at the remote root recorded in
`campaign.json`; they are not committed to Git. Scripts here are campaign
receipts with fixed Aurora paths, not general cluster submission commands.
The copied base input retains its historical limits; `run.py` supplies the
actual target and output block. Its inherited `configuration.json` is used
only to verify the initial-data manifest; `campaign.json` is the authoritative
description of this rerun.

## What the previous N12/N16 runs establish

`compare_previous_scri.py` verifies the archived input hashes and recomputes
maxima over the common interval `[27.5003839904, 29.0002512997] M0`, combining
all restart subwindows without interpolating extrema. The two resolutions'
window endpoints differ by less than `1e-11 M0`. The resulting N16/N12 ratios
are 0.196 for `chiQ-1`, 0.673 for connection regularity, 0.702 for shear
regularity, and 0.853 for the unsolved clock residual. The solved compatibility
residual is already at roundoff. See `previous-N12-N16-scri.json`.

These are encouraging decreases, but two grids do not establish decay to zero
or asymptotic order. In particular, the clock residual improves only about 15%
in this late window. These native regularity diagnostics are not waveform
error bars, nor are they the new four-metric/round-cut diagnostics: those were
not recorded in the previous N16 run. The present N12 extraction therefore
does not establish a nonconvergent continuum obstruction, and the earlier
N12/N16 constraint-norm improvements do not establish waveform convergence.

Direct extraction at scri is possible for the finite radiative coefficient
`psi4^0 = lim R Psi4`, using the regularized conformal Weyl tensor and a specified
inertial frame. Raw physical `Psi4` tends to zero. Such an independent extraction
would be a useful check against the time derivative of news, but it still
requires regularity cancellations and can require higher derivatives; it is
not implemented by merely dividing endpoint curvature by zero. See
[Babiuc et al.](https://arxiv.org/html/0808.0861v2#S3.SS2).

## Long-trajectory comparison at 26 M0

The fresh rerun and the original frozen science run have identical checkpoint
time, cycle, mesh/atlas, formulation identity and step at cycle 18462
(`26.0005166810 M0`). Their complete payloads are **not bitwise identical**.
The maximum absolute field difference is `3.92013e-7` in `Lambda_y`; maximum
metric-component differences are below `1.11e-8`, and lapse/shift differences
are below `2.75e-10`. Across all 31 stored endpoint fields, the largest absolute
scri difference is `7.43e-12`. The full receipt includes per-field maxima and
unweighted relative L2 differences over the stored payload.

The runs use different restart schedules and the original science run used the
pre-optimization executable. This comparison does not isolate the cause of the
small differences. The eight-step fresh/late admission comparisons above were
bitwise exact against the selected optimized executable; do not extend that
claim to this entire fresh trajectory. These tiny endpoint-state differences
do not explain the much larger measured scri regularity defects.

## Direct-scri Weyl feasibility check

`check_scri_weyl.py` independently constructs the conformal four-dimensional
Weyl tensor from metric jets. Its Ricci contraction matches the existing
geometry routine and its traces vanish. On the manufactured asymptotic wave,
the boundary radiation numerator vanishes and its one-sided native radial
derivative recovers the known `psi4^0` to `3.9e-13`; the zero-wave case is also
checked. This is an analytic feasibility test, not a completed N12 Psi4
extractor.

`check_n12_weyl_numerator.py` checks the zero-target numerator
`Cbar_abcd n^a mbar^b n^c mbar^d` directly at the recorded endpoint, with
`n=grad(Omega)` and a native orthonormal angular dyad. It does not divide by
Omega or identify the numerator as the waveform. Sampling every tenth usable
record through `26.4906 M0` gives a largest pointwise magnitude `5.56e-4` and
largest angular RMS `1.41e-4`; the receipt is `weyl-numerator-27M.json`.
These nonzero discrete residuals require a regularized limit and resolution
checks for a physical `psi4^0` interpretation. They are not fractional waveform
errors and, with only this N12 dataset, do not establish nonconvergence.
