# Direct-Theta radial propagation diagnostics

This is an independent view of the full active-cell `z4c_Theta` binary outputs
from Aurora diagnostic job **8842248**. It changes neither the simulation nor
the scheduler. The primary case uses `shift_eta=0.02`, lapse-residual damping
`0.01`; the independent ablation uses lapse damping `0.1`. Both use `kappa1=0`,
G=1, the wide radial sponge (512–1792M, maximum rate 0.001/M), and the same
1e-6 initial centered Gaussian Theta perturbation of width 384M.

## Archived validated sample, not final job status

At this collection, primary has **373 complete snapshots through 23808M**;
no ablation snapshots had been collected at that time. Final C/D collection is blocked by unavailable Aurora authentication; no completion is inferred. Every included snapshot passes all
eight-rank matching headers/time/cycle, complete payload, finite Theta, unique
logical blocks, matching physical bounds, and exact 64³ active-cell coverage.
There were no exclusions. This is not a replacement for the independent raw
checkpoint, ghost-cell, SPD, and full-field validation.

`theta-radial-spacetime.png` shows signed radial means and RMS. The alternating
bands propagate outward, while the core has repeated sign-changing oscillations
whose envelope decays. The core RMS falls from 6.18e-7 initially to 3.01e-14 at
23808M. Ramp and plateau RMS are 1.25e-13 and 1.34e-13; after roughly 20000M they
approach a low-amplitude plateau. This interval does not establish eventual
saturation or a late growing mode. The final point is above an oscillatory
minimum, which is not by itself exponential growth or a returning pulse.

The latest active maximum is 4.64e-13 at (-2016,-32,-32)M. A tiny late maximum
near a face is not evidence for the origin of a resolved unstable mode.
Initial Gaussian tails already reach the outer domain, and the pulse is a
direct constraint violation with coupled dynamics, so these profiles do not
measure a reflection coefficient. No inward/outward characteristic projection of these binary dumps
has been performed; see the separate fixed-face checkpoint audit below. Decaying core oscillations may include propagation,
coupling, and boundary return; the present plot does not separate them.

## Definitions and limits

* Binary Theta is **float32**, promoted to float64 for accumulation. This
  preserves small absolute values, but not double-precision relative accuracy.
* Means, RMS, and squared integrals use **coordinate volume dx³**, not the
  physical metric volume used by checkpoint/history diagnostics. The binary
  dumps contain no metric from which to construct proper volume.
* Radial bins are 64M wide; r>2048M includes only the portion of each shell
  inside the cube, not a complete sphere. Dashed lines mark 512/1792M; the
  dotted 2048M line marks the inscribed sphere.
* Core r≤512M, ramp 512<r<1792M, and plateau r≥1792M partition the active cells.
  The separately stored distance-to-face≤256M band overlaps these regions.
* `propagation-summary.json` gives descriptive log-linear fits and explicitly
  records each actual fitted interval. These are not eigenvalues or a stability
  theorem. Do not infer a reflection coefficient from the core norm alone.

`aggregate.py` parses the format implemented by `src/outputs/binary.cpp`,
rejects incomplete/inconsistent/nonfinite cohorts, and caches unchanged valid
cohorts using file size and nanosecond modification time. It records SHA256
for all included rank files, avoiding transfer of large raw dumps.
`test_reader.py` independently checks constant-field coverage and rejection of
truncation, time/cycle mismatch, duplicate coverage, and nonfinite payloads.
`plot.py` consumes only compact NPZ profiles. Scripts require Python3, NumPy,
and Matplotlib locally; remote aggregation requires only NumPy.

## Refresh both cases

The command below reads simulation outputs and writes only the separate remote
analysis directory. It copies compact profiles/manifests and regenerates plots.
It does not submit or modify jobs. Run it using a Python with Matplotlib.

```sh
python3 refresh.py \
  --socket /Users/hz0693/.ssh/codex-control/9a355d7bb9bdc04368c5de4b6d3474a2c1512802 \
  --run /lus/flare/projects/MHDTidal/hzhu/tde_1e4_solar_review/runs/long_theta_loweta_8842248 \
  --remote-analysis /lus/flare/projects/MHDTidal/hzhu/tde_1e4_solar_review/long-sponge-study/theta-propagation-8842248
```

`status.json` and each case's `manifest.json` are authoritative for subsequent
refreshes; the numeric narrative above describes only the stated collection.
Incomplete latest dumps remain excluded with a reason and may be accepted on
a later refresh after all ranks finish writing.

## Incoming-trace audit and portable checks

`INCOMING_TRACE.md` audits the actual `zero_rate` boundary rows and the sponge/CPBC
ordering. At the fixed active face cell `(2016,32,32)M`, the Gaussian has a
nonzero incoming characteristic trace before evolution. The two JSON trace
records show all terms in short local MPI8 gates at 0 and 9.6M. They are not late
GPU checkpoint measurements. The audit explains why changing `d_t C=0` to
`d_t C=-nu C` can remove a retained initial trace but cannot alone remove an
existing positive root of the same homogeneous boundary operator.

Run the portable reader/analytic checks from this directory, using Python3
with NumPy (Matplotlib is additionally needed for figures):

```sh
python3 test_reader.py
python3 test_trace_initial.py
python3 plot.py .
```

The trace test creates temporary synthetic checkpoints for both supported resolutions
and compares against the independently specified face value
`Theta0=1.027692618952470e-12` for amplitude 1e-6 and width 384M. It also checks
the 0.1-amplitude fixture, scalar incoming coefficients, archived initial
values and rejection of an unsupported curved background. No raw simulation
checkpoint is included in this package.

`source-provenance.json` records archived source-file hashes and immutable
CPU/GPU executable hashes; `provenance/` contains input decks and existing
all-rank gate checks. The GPU binary manifest contains hashes for all eight
rank files for every accepted snapshot. Absolute paths in these records are
historical provenance; scripts accept new run/output paths and do not require
those local historical directories. `checkpoint_reader.py` is a standalone
read-only parser, with unrelated run-launch helpers removed.

After restoring a verified Aurora connection, refresh C using the command
above and the current authorized socket. D has run subdirectory
`theta_amplitude`, output prefix `theta_loweta_amp1e7_coremask`, and parent run
`/lus/flare/projects/MHDTidal/hzhu/tde_1e4_solar_review/runs/long_theta_amp_8842283`.
Use a separate analysis directory and
`--case theta_amplitude=theta_loweta_amp1e7_coremask` for that run. Refresh
verifies the ALCF hostname and existing Flare project directory before writes;
its only writes are to the selected analysis directory and local output.
The current archived plots must not be relabeled as final results.

## Compact-support initial data

At the audited source state, the direct-Theta seed supports Gaussian radius,
width, amplitude and optional dipole only. Every existing characteristic family
also uses a Gaussian, optionally with a Gaussian transverse envelope. There is
no input-only compact-support Theta profile. `constraint_scalar_theta` is a
coupled outgoing characteristic pulse, not a replacement for an otherwise
matched pure-Theta perturbation; it also retains Gaussian tails and may present
incoming components at other cube faces.

A narrower Gaussian can make boundary data negligible but is not compact and
changes the wavelength spectrum. The recommended exact compatibility test is an
opt-in smooth compact initial Theta profile, with support ending well inside all
boundary stencils and the Gaussian default preserved. Keep it as a distinct
initial-data diagnostic, not an evolved-state cleanup. The opt-in compact profile is now implemented separately; the archived GPU C/D runs used the original Gaussian.

The fixed-face checker now also supports the uniform 32³ global mesh with
16³ blocks, deriving dimensions, spacing and point coordinates from the header.
It reads pulse amplitude/profile/width/radius from the input metadata. A zero
initial trace is reported with a null ratio and its absolute change, avoiding
misleading division by zero. The default CLI selection is initial and latest
available checkpoints; `--indices` requests specific file sequence numbers.
Existing archived 64³ GPU and CPU-gate measurements remain unchanged.

The subsequent four local compact/Gaussian controls, amplitude scaling and
timestep comparison are archived separately in
`../compact-theta/trace-comparison.md`. They show an acquired quadratic trace
even with exactly zero incoming initial data. Earlier initial-tail evidence
therefore must not be treated as the complete explanation for late plateaus.
The pre-existing gate JSONs retain the hash of the checker version used when
created; current generalized-checker hashes are in the package manifest and
new compact-control trace JSONs. Their initial analytic values are rechecked
by the portable regression.
