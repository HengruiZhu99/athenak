# Coupled RK3 memory compression: bounded external study

No AthenaK source, running job, or production input was changed. This is a
frozen, linear, planar matrix study. It does **not** supply a production boundary
condition. The new strong-field GPU diagnostic is owned by the coordinating
agent and is independent of this study.

## Decision

There is a realizable, causal, full-source memory closure, and a constructive
way to preserve closed-system stability during reduction. Present compression
is insufficient to justify a C++ implementation: accurate tested reductions
retain 60–70% of a small exterior, have dense updates, and require a nonlocal,
ill-conditioned coupled metric. Aggressive exterior-only truncation restores
exponential growth despite every retained exterior pole lying inside the unit
circle. Stability of the auxiliary memory alone is not the necessary test.

## Common model and comparison

The same 20-field trace-free Z4c flat model, original G1 gauge, kappa=.1,
eta=2, lapse damping=.1, sixth-order differences, KO dissipation and explicit
RK3 dt=.6M are retained. All input/output source coupling is included.
Spacing h=32M; fields Khat/Theta/A/Gamma are multiplied by h in the comparison
norm. The principal reported fixture has 64 periodic normal cells, 32 retained
cells and 640 exterior scalar states, with a single complex Fourier tangential
angle pi/8 or pi/32. The initial state is the leading eigenvector of the old
32-cell zero_rate strip, normalized to norm one and extended by zero.

The pi/8 old eigenvalue is +.00563488434476/M. No result here establishes a
strong-field, nonlinear, AMR, all-wavevector, or corner stability bound.

These short periodic fixtures intentionally test **compression against the
identical finite reference**. At length2048M the continuum crossing scale is
only about1448M at speed sqrt(2); repeated wrap is present at50000M. The long
times below measure the accuracy/stability of the finite realization, not
an open exterior. This is separate from the earlier8192-cell, domain-doubled
uncompressed experiment in ../robust-boundary.

## Exterior-only balanced truncation

Partition the exact complete RK3 step as R=[[A,C],[B,E]]. For zero initial
exterior, memory kernels are C E^j B. Discrete reachability and observability
Gramians solve P=E P E*+BB* and Q=E*Q E+C*C. Square-root balanced truncation
reduces E,B,C, preserving the unmodified A block. All roots of both E_r and
the **reconnected** [[A,C_r],[B_r,E_r]] are computed independently. Zero stays
exactly zero, without a threshold or reset.

For pi/8,640 exterior states:

| Reduced states | Closed growth /M | Interior relative error at50000M |
|---|---:|---:|
|128|+.00320164|not evolved|
|256|+9.3088e-5|1.00966e4|
|384|-3.29578e-8|8.42382e-5|
|448|-3.29914e-8|2.81841e-8|

All these exterior-only reduced E_r matrices have spectral radius below one.
At384 states, the largest sampled intermediate interior error is.1389% at
10000M, and full selected-state transient growth to norm21.73 remains.
Gramian relative residuals are2.52e-11 and1.52e-11; tiny negative Gramian
eigenvalues around1e-14 are reported and discarded only in their numerical
square roots, not in the evolved physical fields. At448 states the balancing
biorthogonality error is4.15e-7, so formal error bounds must not be treated as
rigorous floating-point certificates.

The longer tangential wavelength is harder. At pi/32,256 states gives
gamma=+.000755345/M;320 gives+8.35916e-5/M. At384 the closed gamma is
+3.15e-10/M and final interior error12.6%; at448 it is numerically neutral
(about-1.22e-12/M) with final error7.96e-5. The exact finite reference itself
has norm183.7 at50000M for this seed despite no resolved positive spectrum;
long transient amplification cannot be hidden by reporting only eigenvalues.

Balanced truncation and its exact-arithmetic transfer-error estimate follow
the discrete Stein-equation formulation in
[Chahlaoui, A posteriori error bounds for discrete balanced truncation](https://eprints.maths.manchester.ac.uk/1464/1/LAA_Chahlaoui.pdf).
Our closed-feedback eigenvalue tests are additional and indispensable.

## A reduction that preserves coupled stability in exact arithmetic

`protected_interior.py` constructs the complete Lyapunov metric H satisfying
R* H R-H=-I. Since the finite complete R is block circulant, H is computed
from20x20 Fourier systems. Write H in I/E blocks and set

```
T = -H_EE^{-1} H_EI
Phi = [[I, 0], [T, V]],      V* H_EE V = I.
```

Then Phi*H Phi is block diagonal, and the first metric-Galerkin test row is
exactly [I,0]. Therefore the retained physical interior update is the original
top row evaluated on the lifted exterior, uE=T uI+V z:

```
A_t = A+C T
B_t = B+E T-T A_t
E_t = E-T C
uI' = A_t uI+C V z
z'  = V* H_EE [B_t uI+E_t V z].
```

Metric-orthogonal projection cannot increase the complete H norm. Thus the
reduced closed step is contractive in exact arithmetic for an SPD H and
stable complete R, independent of which full-rank V is selected. The original
bulk stencil is not edited: the added closure acts through the existing
interior/exterior coupling C. A_t is not asserted equal to A; the difference
is an instantaneous part of the approximated exterior response.

V is built from balanced directions of the shifted exterior realization, then
orthogonalized in H_EE. One extra protected direction exactly represents
-T uI0 for this selected seed, so the physical initial exterior is zero to
about2e-12 relative. This protects one initial direction, **not arbitrary
future initial data**. General initial states need additional initialization
coverage and a quantified projection error; zero initial state remains exact.

In the64/32,pi/8 fixture, the complete H has min eigenvalue1.0547 and max
2.52825e10 (condition number about2.40e10). The normalized Lyapunov residual
is6.51e-7. All tested reduced orders16,64,128,256,320,384,448 have closed
spectral radius below one. Low orders are inaccurate and therefore rejected.
At448 states the final interior error is.598%, with.0881% at10000M; full
state error is2.61% at50000M. At384 final interior error is9.85%. Stability
alone does not make these accurate boundary approximations. This metric
prototype has not been tested at pi/32, larger exterior, or other parameters.

## Cost and next concrete check

Direct memory summation costs O(number_of_steps squared) without convolution
acceleration. The recurrence realization uses fixed memory and O(r^2+r p)
work per step per tangential mode, where p is the number of coupled boundary
ports. Forr448, a dense auxiliary update alone has200704 complex entries;
the640-by448 input/output maps add573440 entries. This can cost more than
keeping32 sparse exterior cells. Tangential FFTs, six faces, corners, AMR,
variable background and timestep changes still require separate designs.

The protected-coordinate lifting additionally stores a dense exterior/interior
map and instantaneous boundary correction. Fourier construction avoids a
large full Lyapunov solve, but extracting/factoring H_EE and storing T remain
large dense operations. Extrapolating the tiny fixture's cost to8192 cells is
not acceptable. No efficient implementation is claimed.

The most useful next bounded check is **accuracy-directed reduction in the
complete Lyapunov metric**, with protected initial-data families and a genuine
large-exterior reference: use snapshot/rational-Krylov directions for the
shifted E_t realization over0–50000M, increase order until outgoing-pulse and
constraint responses converge, and repeat at pi/32 and a second exterior
size. Before any solver implementation require <1% signed interior error
throughout the window, no positive closed modes beyond eigenvalue residuals,
preserved zero, and a lower measured cost than the sparse exterior. If the
nonlocal metric cannot be compressed while meeting those tests, retain the
exact elimination only as a reference and improve a local source-compatible
boundary against that reference.

## Reproduction and artifacts

The scripts use NumPy/SciPy and import the archived discrete model from the
portable sibling path `../robust-boundary/discrete`. An optional
`ATHENAK_BOUNDARY_DISCRETE_MODEL` environment variable can select a different
model directory, whose source hashes must then be revalidated. The included
manifest hashes all packaged files and the exact imported model source; no
job is launched.

```
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 balanced_memory.py --total 64 --inside 32 --orders 128 256 384 448 512
python3 balanced_memory.py --total 64 --inside 32 --tangent-divisor 32 --orders 256 320 384 448 512
python3 balanced_memory.py --total 32 --inside 8 --orders 176 192 208 224 256 272 288 304 320
python3 protected_interior.py --total 64 --inside 32 --orders 16 64 128 256 320 384 448
python3 plot_results.py
```

Set the thread environment above for all commands. Orders excluded for
numerically insufficient Gramian rank are explicitly recorded. JSON files
contain full spectra summaries, selected-state samples, numerical residuals,
and exact-zero checks; only those finite fixtures were tested. No reset,
clipping, bulk-gauge change, or claimed production fix is involved.

An independent read-only review by the incoming-trace subagent checked the
Fourier/component ordering, weighted symbol, lifting algebra, metric
orthogonality, selected-seed initialization, and exact-arithmetic contraction
argument. No indexing defect was found. The review explicitly distinguishes
the exterior transfer bound from closed-system error, and contraction from
Euclidean monotonicity or accurate physical response. No independent rerun
was performed. The top-row diagnostic compares the constructed update with
a separate multiplication of the original full operator by the lifted basis.

## Repository packaging

This package preserves the external numerical JSON results and rendered figures
byte for byte. Only the model import path in `balanced_memory.py`, the sibling
reference above, and these packaging notes were adjusted. The numerical
algorithms are unchanged. `external-provenance.json` retains the original
external script/model hashes. `portable-smoke.json` records the single-order
portable-path rerun and comparison; it is not a repeated full parameter sweep.
