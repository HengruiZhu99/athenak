# Finite-sphere mass extraction on the accepted N10 state

The finite-sphere Hawking mass was evaluated from the actual full 3D final
`t=.0002` N10 snapshot, using metric, chi, curvature, Khat and independent Theta
jets in global Cartesian components. No radial average enters the expansion
kernel. All eight requested radii and three angular quadratures are retained.
**This extraction is not qualified:** a large static interpolation error and a
nonconvergent angular control occur at the exact shell interface r=.7. Mass
agreement elsewhere cannot override that failure or the failed evolution
constraint/scri gates.

No evolution, shift solve, source, gauge, field, mesh, PDE header, sampler, or
shared CMake/binary was changed. This is a read-only diagnostic of the frozen
accepted state. It is a finite-surface Hawking functional, not a Bondi extraction
or a generic 3D Misner–Sharp mass.

## Geometry and normalization

The kernel is `surface_mass.hpp`, committed in
`8004c20118ac561715c613fad670307b5d0c52ed`; see
[geometric-mass-diagnostic.md](geometric-mass-diagnostic.md) for its independent
physical ADM derivation/tests. For each angular point, the diagnostic computes
full first/second Cartesian jets of 22 stored components with
`CompositeDifference::SampleJets`, then decodes the actual positive-storage2
chart with `UnpackComposite`. The input field buffer is component-major padded
storage containing the actual active values. `SampleJets` accesses only active
cells in a six-point tensor-product owner-block interpolant; no ghost values or
radial averaging are substituted. It uses the map's full J/H chain rule.

The physical metric is `g_phys=g_tilde/(chi*Omega^2)` and
`Omega=(1-r^2)/2>0` on every requested surface. Actual null expansions and physical
area are integrated:

```
A = sum(weight_solid_angle * dA/dsolid_angle)
I = sum(weight_solid_angle * theta_plus * theta_minus * dA/dsolid_angle)
R_A = sqrt(A/(4*pi))
M_H = R_A/2 * (1 + I/(16*pi)).
```

This surface area includes the physical Omega factor; it is distinct from the
Penrose-volume weights used for evolution constraint RMS. No floor, exclusion,
clipping, endpoint limit, chosen seam owner, or radius offset is used.
Independent Lambda remains untouched; it is not needed by this geometric mass
kernel. The trace of A and independent Theta are retained.

The spherical normalization was checked independently from the selected stored
N2049/N1025 PunctureChart states. With `K=Khat+2Theta`,

```
R = r/(Omega*sqrt(chi))
normalR = R*(K/3 - Omega*A_rr/2) = -R*p/2
spatial = (1+r^2)/(2*Omega) - r*chi_r/(2*chi) = R*k/2
M_MS = R/2*(1 + normalR^2 - spatial^2).
```

Thus the kernel's p enters with the required factor 1/2 before squaring. The
`PunctureChart.parts(state)` diagnostic column 10 and this independent formula
agree at the requested radii within 8.30e-13. The p/2 identity agrees exactly;
the k/2 identity agrees within 1.14e-13 over all finite radial grid points.
The physical chi, lapse, Khat and A_rr reconstructed from the stored reference
states match the archived profiles bitwise. No 1D evolution is rerun.

## Controls and data provenance

The exact final v16 snapshot is the I/O-only accepted-state capture certified
against the original related-family volume/sphere CSVs. Its source tag and input
hash are recorded in the manifest. The original M=1/3 isotropic hyperboloidal
wormhole initializer is the same 50,000-cell initial dataset; it is reconstructed
in the exact 3D storage convention, then sampled with the **same** interpolant
as the final state. It is not replaced with time-symmetric Schwarzschild data.

Controls:

- Sampled hyperboloidal Minkowski mass: max absolute value 4.44e-13. Negative
  roundoff values are retained.
- The original analytic hyperboloidal wormhole evaluated directly with its
  independent initial radial jets: max error from 1/3 is 6.04e-13. This isolates
  the physical expansion/curvature normalization from 3D interpolation.
- N1025 versus N2049 selected radial Misner–Sharp masses differ by at most
  2.16e-13 at the eight finite radii.
- Independent reintegration of every raw area/expansion CSV agrees with the C++
  integrated mass within 7.47e-14.

Angular quadratures are Gauss–Legendre in mu and uniform midpoint phi:
8×16,16×32,32×64 =128/512/2048 points per surface. Every raw direction, solid-angle
weight, owner block/face/layer, validity flag, area density, mean curvature,
curvature trace and both null expansions is preserved. There are 65,536 raw
samples across all modes/controls, **zero failed samples**, and no discarded
requested radius. Integrals use extended-precision accumulation; double versus
extended final evaluation is recorded separately. No convergence claim is made
from three angular quadratures alone.

## Measured masses and unresolved extraction error

| Coordinate r | Initial, same3D sampling (2048) | Final3D (2048) | Final−initial |
|---|---:|---:|---:|
| .05 | .332675270833 | .332675270264 | -5.69e-10 |
| .1 | .333003060779 | .333003053174 | -7.61e-9 |
| .2 | .333329720948 | .333329719831 | -1.12e-9 |
| .3 | .333313727976 | .333313813662 | +8.57e-8 |
| .5 | .333341039569 | .333341072999 | +3.34e-8 |
| .7 | .355303541921 | .354953131776 | -3.50e-4 |
| .85 | .333333448400 | .333334252156 | +8.04e-7 |
| .95 | .333334012323 | .333336371350 | +2.36e-6 |

The corresponding selected radial reference is approximately 1/3 at every radius.
The last column includes physical/numerical evolution **and changed extraction
error**. It is not a pure physical mass-drift estimate, and the initial error
cannot simply be subtracted as if it were unchanged in the evolved fields.
At r=.95 the 512→2048 final change is 1.36e-6, comparable to the final−initial
change; even that small apparent drift is not accurately resolved by this control.
At r=.05 the same-grid initial bias is−6.58e-4, much larger than the evolved
change. The core is underresolved for this owner-polynomial log-field extraction.

At the exact shell boundary r=.7 the initial masses are
`.3554248290/.3523013483/.3553035419` for 128/512/2048 points. The nonmonotonic
changes are not smooth quadrature convergence. Raw owner data show that floating
point evaluation of the same exact sphere selects either side of the radial
interface. Inner layer 4 receives solid-angle fractions
`.957726/.822460/.952580`; the remaining directions use outer layer 5.
The one-sided interpolation limits are different even for exact initial data:

| Initial r=.7 owner side | Physical mean curvature k | Curvature trace p |
|---|---:|---:|
| Layer 4, inner side | 2.09573161 | approximately 2.0003869 |
| Layer 5, outer side | 2.09746496 | 2.0000000 |

The initial conformal metric is flat, so its mean-curvature jump is from the
sampled chi value/first derivative, not an evolved metric anisotropy. Curvature
sampling also contributes. Increasing angular quadrature changes the floating
owner-side mixture instead of resolving a unique C1 interpolant. All directions
and both sides remain in the reported integral; this failure is not hidden by
moving the radius or selecting an owner.

The diagnostic's six-point local-owner interpolation differs from the evolution's
fourth-order/support-enriched derivative machinery. The large initial mass error
therefore cannot be assigned directly to physical mass loss or to evolution
constraint error. A consistent cross-layer jet reconstruction needs its own
qualification. Face seams/corners and core extraction require separate controls;
a radial-interface fix alone would not establish the complete diagnostic.

## Reproducibility and scope

The standalone code was compiled from compiler-reported frozen project
headers/Kokkos dependencies, with zero live project paths. Sources, SDK hashes,
reference inputs and the copied 1D diagnostic library remain unchanged. The copied
library matches the selected reference SHA256
`1c6d85d5cbcdaf4bf6fef0b4bee536cbc4e3ec31af47f22ba7cae8f95983de5b`.
The complete source capsule is
`evidence/coupled-surface-mass-20261027/compiled-source.tar.gz`, SHA256
`4415573621bc9370ae9c838e96fe48eca401d5f60adb02c1d392ecfd4a3f12e0`.
Each extraction process took at most 2.07 seconds; all had an owned 180-second cap
and at most 4096 query points. Only local Serial CPU was used.

```
PYTHONDONTWRITEBYTECODE=1 python3 tst/elliptic_shift_3d/run_coupled_surface_mass.py \
  --capture docs/elliptic-shift-3d/evidence/final-scri-io-capture-20261027 \
  --output docs/elliptic-shift-3d/evidence/NEW-coupled-mass
PYTHONDONTWRITEBYTECODE=1 python3 tst/elliptic_shift_3d/analyze_coupled_surface_mass.py \
  --input docs/elliptic-shift-3d/evidence/coupled-surface-mass-20261027 \
  --output docs/elliptic-shift-3d/evidence/NEW-coupled-mass-analysis
```

Raw lossless CSV.gz, source/command/input hashes, scalar-reference mass arrays,
complete error-budget JSON, owner-side counts, mass arrays and plots are in the
`coupled-surface-mass` and `coupled-surface-mass-analysis` evidence directories.
The earlier 128-point pilot is preserved separately. This phase provides an
explicit failed extraction control and finite-surface measurements; it does not
qualify the full scientific evolution milestone, mass conservation, a Bondi mass,
or a generic 3D Misner–Sharp diagnostic.
