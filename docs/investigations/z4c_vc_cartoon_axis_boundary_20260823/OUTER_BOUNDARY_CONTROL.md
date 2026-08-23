# Fixed-grid outer-boundary control

Date: 2026-08-23

## Source audit

The state ghost fill in `src/bvals/physics/z4c_bcs.cpp` extrapolates each
physical ghost independently from interior active values. For a centered O4
bulk operator:

| input `extrap_order` | polynomial degree | ghost error | centered D1 boundary error | centered D2 boundary error |
|---:|---:|---:|---:|---:|
| 2 | 1 | `O(h^2)` | `O(h)` | `O(1)` |
| 3 | 2 | `O(h^3)` | `O(h^2)` | `O(h)` |
| 4 | 3 | `O(h^4)` | `O(h^3)` | `O(h^2)` |

Therefore none of the existing ghost options supplies an O4-consistent
centered second derivative at the active boundary or all adjacent stencil
layers. An O4 D2 value obtained through an extrapolated ghost would require at
least `O(h^6)` ghost accuracy (a degree-five polynomial), whose large
one-sided coefficients would need an independent stability analysis.

`src/z4c/z4c_Sbc.cpp` overwrites only `Theta`, `Khat`, three `Gamma`
components, and six `Atilde` components at the active physical face. The
remaining evolved variables retain the bulk RHS evaluated through physical
ghosts. Adjacent active layers also retain bulk centered derivatives. Hence a
formally high-order first derivative in the Sommerfeld overwrite alone cannot
make the coupled boundary closure O4.

The source now uses the configured O2/O4/O6 stencil and a matching one-sided
physical-normal derivative only for native-VC Cartoon. Legacy CC and Cartesian
closures remain on their original O2 path and pass their exact fingerprints.
The one-sided formula is polynomially verified, but it is only a partial
repair because the bulk ghost-dependent fields remain.

## Controlled outcomes

All controls used the corrected 128x32 IrisK coefficients.

### Linear ghosts (`extrap_order=2`)

All resolutions reached `tau_c>3 M`, but exact localization became globally
nonconvergent by `tau_c=0.5--0.75`. At `tau_c=1.25`, the full C constraint had
order `-3.895`, and outer-layer C had order `-3.940`.

### Quadratic ghosts (`extrap_order=3`)

All resolutions reached `tau_c>3 M`. Global history C/H/M/Z norms decrease
monotonically with resolution at every sampled proper time through `3 M`, and
the exact full-domain worst significant orders remain positive through
`tau_c=1.25 M`. The strict face-local gate still fails: Theta/A RHS and A state
rows have negative order at amplitudes `1e-7--1e-6`.

This is a strong diagnostic improvement, not a production qualification.
`extrap_order=3` remains an explicit input choice; the default is unchanged.

### Cubic ghosts (`extrap_order=4`)

Perlmutter job `57484730` used one 80-GB GPU. N128 and N256 reached `t=5 M`.
N512 ran away and failed at

```text
time=4.3165046742246815
cycle=1437
checkpoint=POST_RK_UPDATE
rho=15.8125, z=16
reason=nonpositive_metric_pivot_1
det(gtilde)=-39.0093793564.
```

The failure is at the outer corner and is preceded by timestep collapse.
Thus simply increasing extrapolation degree is not a stable repair.

## Moved-boundary and CC control

The final bounded control moves the VC boundary to
`rho=[0,32], z=[-32,32]` at unchanged inner spacing and MeshBlock dimensions,
and runs a matched fixed-grid CC/O4 Cartoon case on the base domain. All six
N128/N256/N512 cases reached the exact target coordinate time
`2.0295751268186133` (`tau_c≈1.25 M`). Detailed numerical reductions are in
`evidence/domain-cc-controls-nr128nt32-extrap3-analysis/` and are summarized in
the main report. The moved-boundary VC sequence has minimum significant
orders `3.587` for core state and `3.712` for core constraints, while its
outermost state/constraint subsets still reach `-0.110` and `-0.333`.
The base-domain/moved-domain inner `r<=8` maximum pointwise difference is
`5.96034e-8` over all saved state and constraint components.

The matched CC control shows the same qualitative separation: core
constraints have minimum order `3.721`, while the outermost constraint subset
has order `-0.666`. Its state sequence is only about `1.25` order in the core,
so it is not a positive O4 authority; it nevertheless confirms that the
outer closure is not a uniquely native-VC axis effect.

The CC result is an auxiliary comparator only; its half-cell radial geometry
is not a continuum authority for the evolved VC axis.

## Disposition and smallest next repair

Disposition: `NONCONVERGENT` under the strict local boundary gate.

The next source correction should not be a higher-degree ghost extrapolation.
The smallest decisive design is a manufactured semidiscrete boundary problem
that includes the actual coupled Z4c characteristic/Sommerfeld overwrite,
bulk first/second/mixed derivatives, KO, corners, and the first four active
layers. Use it to derive either:

1. one-sided/biased O4 first, second, and mixed derivative closures for every
   field whose bulk RHS reaches the boundary stencil; or
2. a characteristic/SAT boundary operator with an energy estimate and a
   compatible corner closure.

Promotion requires a stable manufactured spectrum and no negative local
N128/N256/N512 order before rerunning the physical gate. Floors, extra KO,
constraint damping, and relaxed admissibility checks are not acceptable
substitutes.
