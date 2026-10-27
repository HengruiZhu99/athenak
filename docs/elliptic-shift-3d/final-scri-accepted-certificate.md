# Exact final accepted scri certificate, 2026-10-27

The reported N10 pole `1.3540399227646582e-7` is the surface maximum at the
**final accepted t=.0002 state**, not a maximum over RK history.
`CompositeTensorEvolution::BoundaryRates` resets `boundary_pole` on every RHS,
and `coupled.cpp` calls `RHS(pilot.state,dt)` after the restart check and before
printing. Earlier reports saying “stage-history maximum” were incorrect; the
related-family and midpoint reports now contain this correction. Numerical
results and scientific acceptance targets have not changed.

This phase captures the exact failed accepted state. One private I/O-only patch
inserts a v16 snapshot write immediately after that existing final RHS. It copies
all volume fields, all 22 surface fields, the freshly solved global shift,
geometry, telegraph state and all current parameters. There are no changes to
PDE headers, sampler, source, solver, tolerance, loop/RK arithmetic, or boundary
conditions. The snapshot writer is copied from the driver's existing midpoint
writer, with `step=steps` and the suffix `.final`.

## Physics-preserving capture controls

Before the only nonlinear rerun, the original frozen binary and I/O-patched
binary each ran the same small two-step Minkowski control. Their volume and
64-direction sphere CSV files are byte-identical, and an independently evaluated
original-vector/full-tensor-pole certificate passes.

Exactly one N10 case then ran with the original 50,000-cell mesh, initializer and
all numerical controls. It finished in 102.80 seconds under its 180-second owned
process cap. Its complete final volume CSV and sphere CSV are **byte-identical**
to the related-family baseline. The entire midpoint checkpoint payload—volume,
surface, shift, geometry and parameters—is also byte-identical except for the
64-byte source tag and its dependent checksum. Thus the captured final surface
comes from the same numerical evolution rather than a changed discretization.
The original scientific failure remains exit 1.

The I/O-only patch, compiler commands, frozen sources/Kokkos dependencies,
lossless executable, baseline/current hashes, stdout/stderr and snapshot are
under `evidence/final-scri-io-capture-20261027/`. The source archive SHA256 is
`49f115ce4a6d798923b3924e7136ca97ca282a99224a192ad33e2deff2bc76a2`.
Both compiler dependency proofs contain zero live project paths. External SDK
hashes remain unchanged.

## Fresh accepted-state original vector equation

Unlike the earlier midpoint warm-start diagnostic, the final snapshot is written
**after** a fresh accepted-state solve. Independent recomputation of the original
vector source and full operator, including the inhomogeneous physical shift
boundary lift, gives:

| Residual | Value | Unchanged target |
|---|---:|---:|
| Penrose-volume physical RMS | 2.94226362662e-11 | 1e-10 |
| Conformal-volume RMS | 1.99278487504e-11 | 1e-10 |
| Point maximum | 2.07336769975e-9 | 1e-7 |

All pass. No potential or preconditioner residual is substituted. No solve or
time advance occurs in the certificate executable. It reconstructs the same
stored fields, physical boundary datum and derivatives from the frozen state,
and applies the original vector equation again. All input fields, shift and
surface arrays remain exactly unchanged.

## Exact failed pole and source balance

The independently reconstructed pole is `1.3540399227646587e-7`, agreeing with
the driver's reported final value to roundoff. It is LambdaZ, signed negative,
at surface row 533, face 5, logical j3,k3:

```
x = (-0.276172385369497, -0.276172385369497, -0.920574617898323).
```

All 600 boundary points and all 13 scalar/vector/tensor numerator components are
retained. The source decomposition at that maximum is

| Contribution | Pole numerator |
|---|---:|
| Metric normal first jets | +1.13873446247e-6 |
| Metric angular first jets | -1.14935023524e-6 |
| Independent evolved Lambda | +2.72781178919e-9 |
| Khat normal derivative | -6.72602296416e-7 |
| Theta normal derivative | +5.57147428888e-9 |
| Khat/Theta angular derivatives | -2.90171976449e-9 |
| Curvature contraction with grad Omega | +5.42416510595e-7 |
| Sum | -1.35403992276e-7 |

The Khat/Theta normal contributions sum to `-6.67030822127e-7`. Raw coordinate
radial data at this point are

```
Khat_r = -5.47974928040e-7
Theta_r = +9.07825690990e-9
Khat = -3.0000000064434391
Theta = -3.12276766142e-11.
```

Normal denotes the coordinate-radial derivative of global Cartesian components;
the complete physical kernels still perform all metric contractions. The two
metric derivative terms largely cancel. The largest uncancelled balance is
Khat normal versus curvature, rather than independent Lambda. This identifies
where the actual failed final numerator arises; it does not establish whether
the remaining discrepancy comes from normal differentiation, accumulated source
truncation, or the coupled boundary update. No field, normal coefficient, Z4
connection or constraint has been projected or reset to force that balance.

The full ConformalRHS tensor pole and independent first-jet pole helper agree
within 1.34e-15. The exact additive decomposition error is below 3.56e-15 over all
points, with the LambdaZ sum error `5.3e-23` at the maximum. A second read-only
certificate adds the Khat/Theta split; **every pre-existing certificate column
remains bitwise identical**. Its build/read-only N10 costs were 4.47/5.98 seconds;
no second nonlinear case was run. Actual final clock/null/tangent residuals are
0,4.44e-16,4.13e-16.

The unchanged pole target `1e-7` still fails, and the byte-identical family's
Hamiltonian RMS `0.0614299` still fails its `1e-5` reference target. Passing the
original shift solve and algebraic scri data does not override either failure.
The earlier midpoint's smaller pole and the rejected eight-node constructor do
not change this final accepted-state result.

## Reproduction and artifacts

```
PYTHONDONTWRITEBYTECODE=1 python3 tst/elliptic_shift_3d/capture_final_scri_state.py \
  --family docs/elliptic-shift-3d/evidence/related-scri-family-20261027 \
  --output docs/elliptic-shift-3d/evidence/NEW-final-capture
PYTHONDONTWRITEBYTECODE=1 python3 tst/elliptic_shift_3d/refine_final_scri_certificate.py \
  --capture docs/elliptic-shift-3d/evidence/final-scri-io-capture-20261027 \
  --output docs/elliptic-shift-3d/evidence/NEW-final-certificate
PYTHONDONTWRITEBYTECODE=1 python3 tst/elliptic_shift_3d/analyze_final_scri_state.py \
  --capture docs/elliptic-shift-3d/evidence/final-scri-io-capture-20261027 \
  --certificate docs/elliptic-shift-3d/evidence/final-scri-certificate-20261027 \
  --output docs/elliptic-shift-3d/evidence/NEW-final-analysis
```

The first command explicitly reruns the tiny Minkowski control and one bounded
nonlinear case; the second and third are read-only frozen-state diagnostics.
Raw complete snapshots and lossless all-row CSV.gz, input/output hashes, field
and component maxima, original residuals and plots are in the corresponding
`final-scri-io-capture`, `final-scri-certificate`, and `final-scri-analysis`
evidence directories. No additional coupled case or tolerance change is claimed.
