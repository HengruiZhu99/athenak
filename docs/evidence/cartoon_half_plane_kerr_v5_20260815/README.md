# Half-plane Kerr v5 evidence handoff

This directory is a compact, reviewable archive of the existing AthenaK
half-plane SO(2) Kerr convergence attempt. No simulation was launched while
assembling it. The raw Perlmutter root remains at
`/pscratch/sd/h/hzhu/axisymmetric-cartoon-half-plane-kerr-f95a0580-v5-20260813`.

## Provenance

- AthenaK commit: `f95a05802621b76cff2894d562c38df4b0d09661`
- source tree: `4a03dc2ec051463372d812cfd053ca4db3d0698a`
- CUDA executable SHA-256:
  `c14f6ac88144ad27b097e6f95249167a37d32cc50de2923894e66f10de2adeba`
- Slurm job: `56882670` (`cartoon-hp-kerr-f95a0580-v5`)
- allocation: one A100 node, four MPI ranks, four A100 GPUs
- terminal accounting: parent `FAILED 1:0`; step `.0` completed and step `.1`
  failed; `mode=failure numbered_steps=2`
- final source evidence records the expected and observed commit/tree as equal
  and the checkout as clean.

The case is a single Kerr black hole with `M=1` and dimensionless spin `0.5`.
It uses O6 spatial differencing, RK4, `dchi_max=0.02`, no chi floor, and a
target time of `5M`. The initial lapse is pre-collapsed (`alpha=psi^-2`). The
evolution gauge is AthenaK's default moving-puncture choice: advective 1+log
lapse and advective Gamma-driver shift with `eta=2`. Telegraph lapse,
slow-start lapse, scale-selective damping, and alternative paper-matched gauge
options are disabled.

## Result and limitation

- `h32` (`dx_min=M/32`) completed to `5M`. Its last accepted origin horizon
  was at `t=2.834375M`, with `M_AH=1.0022786994483526` and
  `chi_AH=0.5090688179661913`; late constraint values became large.
- `h48` (`dx_min=M/48`) initially produced the more accurate horizon:
  `M_AH=1.0004919689381744`, `chi_AH=0.5010703108945663`. It failed at cycle
  2178, about `4.560417M`, because the axis-central diagnostic support became
  nonfinite. Its last accepted origin horizon was at `t=1.95625M`.
- `h64` was not run because the prospective convergence gate had already
  failed. There is no convergence or horizon qualification claim.

The bounded CPU/MPI4 replay reproduced the h48 failure. Native AthenaK
cycle-resolved data localize the growing mode to `rho=2.5h=0.0520833M`, near
`z=+/-0.28M`, inside the apparent horizon. Equatorial parity stayed near
`1e-6` until the terminal growth, and the outer boundary was causally
irrelevant. The trace-Ricci contribution to conformal A and shift-derivative
terms in Gamma dominate; the largest explicit Gamma damping term is about
`106`, versus about `1.40e6` for the Gamma contraction. The evidence therefore
does not identify explicit Z4c damping as the immediate stiff trigger.

The unresolved diagnostic limitation is narrower: the existing evidence does
not distinguish a continuum puncture-interior moving-puncture mode from a
nonlinear SO(2) derivative-stability defect. No threshold, floor, clipping,
dissipation, damping coefficient, gauge coefficient, or numerical order was
changed to obtain this archive.

## Directory map

- `evidence_index.json`: compact machine-readable disposition and hashes.
- `remote_terminal/`: live-fetched root/detached manifests, settled Slurm
  accounting, clean-source record, configure/build/CTest logs, inputs,
  commands, and terminal statuses.
- `run_evidence/`: h32/h48 stdout/stderr, history files, horizon tables, and
  campaign state used by the offline analysis.
- `analysis/`: strict diagnostic JSON, CSV tables, and five plot families in
  PDF and PNG form (constraints, horizons, near-axis growth/slices, RHS terms).
- `failure_region/`: four compact native near-axis snapshots bracketing the
  first diagnosed growth region.
- `campaign_bundle/`: the frozen campaign contract and launch/analyzer
  harness, included for provenance only.
- `analyze_default_gauge.py`: deterministic offline analysis script; running
  it is optional and is not a numerical evolution.

Large binary dumps and restart files are deliberately omitted from Git. The
copied remote `SHA256SUMS` authenticates their identities in the immutable
Perlmutter root, while `SHA256SUMS.sha256` authenticates that manifest. The
top-level package checksum files authenticate every file actually committed
here.

Verify the compact archive with:

```bash
sha256sum -c SHA256SUMS
sha256sum -c SHA256SUMS.sha256
```
