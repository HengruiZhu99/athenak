# Flat composed-operator energy audit

**The unadopted composed candidate fails the flat zero-Dirichlet energy identity.**
The saved physical negative-Rayleigh mode has negative discrete energy while its
computed strain energy is positive. This is a discrete coercivity counterexample,
not evidence that the continuum flat shift equation is nonelliptic, and not a
proof that the discrete matrix is singular.

No Arnoldi or evolution was rerun. This read-only diagnostic uses
`composed-arnoldi-20261027/composed-negative.witness.csv.gz`, whose `Pv0..Pv2`
columns are the actual physical beta mode z, rather than its Krylov input v.
`APv0..APv2` are A z. The new independent action reproduces every saved A z
component **bitwise**. All 10,800 cells and all 32,400 vector components remain.

The exact configuration is flat g=I, chi=alpha=1,
`CompositeMesh(.25,{.45,.55,.60,.65,.70,.75,1},1,6,2,2)`, fourth-order volume
stencils, radial-interface interpolation, seven-node Hermite scri sampling,
affine OFF, and zero beta on the actual unit-sphere outer boundary. There is no
core AMR or coarse/fine interface in this witness. The candidate's `Rates(z)`
constructs all six global Cartesian strain components and their actual surface
traces, then `Apply` differentiates those rates with its original global halo
contract. No rate, field, boundary or operator was corrected.

## Identity and measured failure

For a smooth vector field with beta=0 at the unit sphere,

```
H_ij = beta_i,j + beta_j,i - (2/3) delta_ij div(beta)
A beta = -div(H)
E = integral beta dot A beta
  = integral (partial_j beta_i) H_ij
  = (1/2) integral H_ij H_ij = Q >= 0.
```

The outer integration-by-parts term is zero because beta is zero, not because
H is zero. In Q, each off-diagonal symmetric component occurs **twice**. The
maximum nodal trace(H) residual is1.666e-16. The volume weight is the positive map
cell volume: g=I and chi=1. There is no Omega factor, floor or region exclusion.
The table gives **unnormalized** volume integrals; the Stage physical dot is the
corresponding integral divided by total numerical volume 4.216699622675386.

| Quantity | Negative physical witness | Smooth zero-boundary control |
|---|---:|---:|
| Integral beta² | 4.22191351113e-6 | 0.533572554721 |
| Discrete E | -0.00361952404542 | 6.57158666763 |
| Positive Q | 0.00157118665627 | 6.58313335062 |
| E-Q | -0.00519071070169 | -0.0115466829925 |
| E / integral beta² | -857.318378473 | 12.3162006919 |
| Q / integral beta² | 372.150365499 | 12.3378410159 |
| (E-Q)/Q | -3.30368812704 | -0.00175397981137 |

The negative Rayleigh quotient is independently recovered using Python scalar
`math.fsum` over the saved fields/volumes, not a projected Hessenberg quotient.
The fresh C++ integrals agree with that independent reduction. This establishes
that the actual divergence/strain pair is not an energy-adjoint pair in the
chosen volume inner product. It does not by itself isolate one interpolation
row, establish the full spectrum, or imply that changing the preconditioner
would repair the identity.

## Localization, with physical regional-flux caveat

The regional E-Q entries below are **contributions to the global defect**. A
subregion has a nonzero interface flux in the continuum, so its own E-Q need
not vanish. These fractions are localization evidence, not separate regional
integration-by-parts tests. Overlapping bands are explicitly labelled.

| Region | E-Q contribution | Fraction of whole negative defect |
|---|---:|---:|
| Cartesian cells | -0.00438763850929 | 84.5287% |
| First cube-to-sphere shell layer | see manifest | 15.1772% |
| Cartesian plus first shell layer (disjoint) | see manifest | 99.7058% |
| Any two-cell patch stencil band (overlapping previous regions) | -0.00518714562312 | 99.9313% |
| Patch interior outside those bands | -3.56507856742e-6 | 0.0687% |
| Outer scri stencil band (overlapping shell regions) | -2.37612763116e-5 | 0.4578% |

The most negative weighted contribution occurs at point 1673, Cartesian block 7,
logical `(5,2,4)`, x=(0.229166666667,0.104166666667,0.1875), immediately near the
cube boundary. Its energy density is -0.5052090240, positive strain density
0.01998558236 and weighted defect-3.799150799e-5. The full top 20 list and every
point are retained. The first layer and Cartesian concentration identifies the
cube/transition interface as the first target for derivative/transfer adjointness
investigation, rather than the scri boundary. It is not a claim that all global
error can be attributed uniquely to the interface interpolation, since volume
quadrature, nodal differentiation and rate boundary lift also enter the pair.

## Independent smooth control and quadrature separation

The independent control is

```
beta = (1-r²)^2 * (1 + x/5 - 3y/10,
                  3/10 + 2yz/5,
                 -1/5 + xz/2).
```

It has exactly zero beta and first derivatives at the sphere. Its analytic
Cartesian jets are generated independently by automatic differentiation. The
Python analysis separately constructs rational multivariate polynomials,
differentiates their exact coefficients and integrates unit-ball monomials.
This proves the continuum control identity **as an exact rational coefficient**:

```
E_exact/pi = Q_exact/pi = 3014144/1447875
E_exact = Q_exact = 6.540076075118193.
```

Using the analytic jets but the same numerical cell volumes gives E=6.58078321918
and Q=6.58323796557, so quadrature alone contributes-0.00245474638918 to E-Q.
The extra nodal differentiation/transfer contribution is-0.00909193660337.
The small positive smooth-mode Rayleigh quotient and 0.1754% relative energy
defect do not qualify high-frequency coercivity; the saved negative mode is an
explicit counterexample. Conversely, the negative witness does not establish
that every smooth resolved mode is unstable.

A replacement must address a compatible gradient/divergence and interface flux
or adjoint construction, together with actual boundary lift, continuum accuracy
and variable-metric terms. Simply composing centered first derivatives or
altering the Krylov tolerance does not supply that proof. No such replacement
is adopted by this audit.

## Reproduction and provenance

New diagnostic files are:

- `tst/elliptic_shift_3d/audit_composed_energy.cpp`;
- `tst/elliptic_shift_3d/run_composed_energy_audit.py`;
- `tst/elliptic_shift_3d/analyze_composed_energy.py`.

Evidence is `evidence/composed-energy-20261027` and
`evidence/composed-energy-analysis-exact-20261027`. The initial floating-gamma
analysis remains frozen in `composed-energy-analysis-20261027`; the exact-rational
version changes only independent analytic moment evaluation. The new compiled
source archive SHA256 is
`62d6d3bc025afcb612df52fde7f75badbcc83dc2c0af1c5777acb0967251ac86`.
It privately restores the Arnoldi source capsule and copies the hashed original
runtime libraries. Compiler dependency inspection finds **zero live project
header dependencies**. Library hashes remain unchanged. The build took 2.99 s and
the actual two-case action/strain audit 2.20 s, with one worker and a 60 s cap per
process. Full per-point beta, A beta, all six H components, weights and energy
contributions are retained in lossless CSV.gz files.

```
python3 tst/elliptic_shift_3d/run_composed_energy_audit.py --output NEW_DIRECTORY
python3 tst/elliptic_shift_3d/analyze_composed_energy.py --input ENERGY_DIRECTORY --output NEW_DIRECTORY
```

The analysis includes `global-energy-versus-strain.png`, exact control integrals,
independent reductions, region fractions and top 20 negative contributions.
This is a standalone host diagnostic. It makes no production AthenaK/MPI/GPU,
scientific-evolution success or full spectral-gap claim.
