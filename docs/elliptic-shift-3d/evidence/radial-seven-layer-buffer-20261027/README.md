# Exact beta halo and radial mesh controls

These static controls retain the actual sampled M=1/3 isotropic wormhole,
Minkowski reference, independent8193-node shift integral, physical
`C_ij=-2 alpha A_ij` source and direct future-scri vector data. There is no
solve, evolution, source balancing, tolerance change or production admission.
The same full source/operator diagnostics include every active cell, physical
and conformal volume RMS, physical L2, maxima and six fixed physical-radius
regions. The JSON/CSV retains all interfaces and scri cells.

## Exact internal beta halos do not remove the dominant error

On the original focused N8 mesh `.62,.70,1`, replacing only the internal beta
halo values (active beta, source and outer DV unchanged) changes compact
exact-shift consistency RMS2.3444553 to2.2702145. The centered first-pass
metric-rate Frobenius defect changes.1189801 to.1176136, only about1%.
The corresponding composed centered residual changes3.0071849 to2.9832048.
Thus interpolation error alone is not the dominant defect. Evidence is in
`../radial-beta-halo-control-20261027`.

## Six layers expose a coarse-stencil/front overlap

The requested physical mesh `.45,.55,.60,.65,.70,1` keeps the Cartesian halfwidth
.25, eight Cartesian blocks, N angular cells per face axis and global physical
radial interface interpolation. The related resolutions are:

|N|Active cells|Compact exact-shift defect RMS|Metric-rate defect RMS|Metric-rate maximum|
|---:|---:|---:|---:|---:|
|6|9,504|2.1516647|.1241899|.5708432|
|8|22,528|1.5349589|.0758132|.4170101|
|10|44,000|.9816787|.0387938|.2461664|

Observed adjacent orders are1.174/2.003 for the compact consistency defect and
1.716/3.003 for the metric-rate defect. This is not a verified fourth-order
regime. The maximum metric-rate error stays at the first cell of the coarse
`.70->1` shell: radii.725/.71875/.715. At N8 its centered radial stencil samples
`.64375,.68125,.75625,.79375`, reaching backward across the steep physical
front even when those halo samples are exact. Exact beta halos improve the N8
six-layer compact RMS only from1.5349589 to1.5244868.

The N6/N10 centered controls use two ghost rings to bound storage; centered
composition exchanges its intermediate tensor rate and never needs the extra
rings. The N8 storage-contract check explicitly repeats the prior four-ring
case with two rings: all common reported norms are bitwise equal (difference0).
This is a storage control, not another physical resolution or parameter choice.

## Buffering the front improves the mechanism control

Adding a `.70->.75` buffer gives seven layers
`.45,.55,.60,.65,.70,.75,1`, 25,600 active cells at N8. The first coarse outer
cell is.765625; its farthest centered radial sample is.703125, beyond the
physical front endpoint.6919875. Therefore the coarse stencil no longer
straddles that front. Keeping all other inputs fixed gives:

|Quantity|Six-layer N8|Buffered seven-layer N8|
|---|---:|---:|
|Compact exact-shift consistency RMS|1.5349589|.4152547|
|Compact consistency maximum|8.1545693|4.3139047|
|Metric-rate defect RMS|.0758132|.00600543|
|Metric-rate maximum|.4170101|.0453157|

The new maximum is at r=.678125 inside the front, rather than at the first
coarse outer cell. Exact internal beta halos change buffered RMS only to
.4135319 and the metric-rate RMS to.00598337. This identifies underresolved
target-block derivative stencils, rather than transferred halo values, as the
dominant mechanism removed by the buffer. The remaining front error is material;
this mesh is not scientifically accepted. The centered composed operator also
retains its known checkerboard defect and is not adopted.

The independent exact-beta-first-jet oracle, with actual curvature unchanged,
continues to give residual RMS around3.6e-14 and metric-rate defect4.5e-16.
It is a diagnostic oracle, not an unknown-shift implementation. No new lapse,
puncture parity, curvature constraint or conformal-flatness condition was imposed.

## Reproduction and provenance

Each directory contains the compiler command, complete stdout/stderr, exact
source/header snapshots taken **before** compilation, library and input hashes,
all-cell CSV, and exact-data generation receipt. Source, binary, libraries and
generated inputs stayed unchanged during every run. Each process is limited to
180 seconds; actual diagnostic runtimes are about1–3 seconds. The sampled shift
integral is independently checked at4097/8193 nodes by the existing generator.
Original research jobs were not touched.

```sh
python3 tst/elliptic_shift_3d/run_radial_metric_divergence.py \
  --build build-shift-3d --n 8 --radial-flags 1 --centered-only \
  --radii .45,.55,.60,.65,.70,.75,1 \
  --output build-shift-3d/reproduced-buffer-control
```

Omitting `--data` regenerates coordinates and exact initial data using the
existing `wormhole_exact_jets.py`. The runner refuses to overwrite a manifest.
Six-layer N6/N10 receipts are in `../radial-six-layer-n6-20261027` and
`../radial-six-layer-n10-20261027`; the N8 ghost-storage equivalence receipt is
in `../radial-six-layer-n8-ghost-contract-20261027`.

The next coupled pilot can test this buffered mesh as an explicitly provisional
control. These static improvements do not replace the unchanged coupled
constraint, scri, elliptic-residual or radial-reproduction acceptance targets.
