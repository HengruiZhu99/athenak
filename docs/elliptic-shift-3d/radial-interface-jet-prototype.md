# AD jets of the global radial interpolant: prototype qualification

`radial_interface_jet_probe.hpp` is a **new host diagnostic**, not a replacement
for production `CompositeDifference::SampleJets`, the PDE operator, or the
qualified Hermite boundary sampler. It removes the artificial dependence of a
shell query's interpolation jets on which adjacent radial shell block owns that
same point. The initial-slice mass control confirms the targeted r=.7 seam
improvement. **Global angular accuracy and complete mass extraction remain
unqualified; the evolved final state has not been sampled with this helper.**

## Same value interpolant, differentiated consistently

The prototype differentiates the existing `Grid.InterpolateShellRadius` value
algorithm, with the same six angular nodes in each direction and the same six
nearest physical-radius nodes along each angular source line. Its geometry-only
cache follows every aligned shell layer and face subdivision. For the initial
cube-to-sphere layer, the source-line node radii include that line's actual
angle-dependent lower cube radius. Every donor is an active cell.

The query variables are global Cartesian X,Y,Z. The code evaluates complete
jets of `r=|x|` and face-angle quotients, then differentiates the complete product
of radial and angular Lagrange weights. **Each angular source line's node radii
are constants**, not functions of the query angle. This retains normal/angular
mixed derivatives without inventing derivatives of moving donor locations.
There are no component-frame transformations. Global Cartesian field component
labels stay unchanged.

Terms with zero value weight but nonzero derivative weight are retained. The
existing value routine's `abs(weight)>1e-18` value-only pruning cannot be copied
into a jet calculation. Anchored field differences give exactly zero derivative
jets for constant input. A local scalar quotient derivative and the analytic
radius Hessian supply the full Cartesian chain rule directly.

At the same physical point, forcing either adjacent radial block with the same
face/angular bounds selects identical donor IDs and bitwise-identical complete
weight jets. No radius offset, arbitrary owner tie-break, averaging of fields,
profile subtraction, PDE source subtraction, or postprocessing projection is
used. Cartesian-core sampling deliberately retains the old local sampler.
Angular face/subpatch boundaries still inherit the existing value interpolant's
separate one-sided angular polynomials.

The prototype guards are declared before tests: each radial factor condition
`<1e8`, value L1`<=8192`, scaled gradient L1`<=1e5`, scaled Hessian L1`<=1e6`, and
partition reproduction`<1e-10`. The value bound reflects that six-point one-sided
half-cell tensor extrapolation can already have L1 approximately`18.25^3`.
These are new value-interpolant coefficient guards; they do **not** relax or
replace the existing Hermite sampler's combined-factor`1e8` guard. The latter
continues to reject the previously tested eight-node configuration.

The host prototype requires geometry revision/rebuild discipline. Shape and
stale-revision controls reject. It is not a production lifetime/foreign-plan,
MPI, GPU, shell-AMR, global angular communication, or Cartesian/shell interface
implementation.

## Independent mathematical controls

Ten configurations cover 1/2/7 shells at three affordable related N values, plus
face subdivision 2. Every queried radial interface, exact angular face edge and
corner, and both admissible angular owners are retained. Tests use globally
smooth Cartesian manufactured component fields and exact radial powers, plus a
local complete polynomial in physical radius and face angles. The latter is
differentiated independently in global Cartesian coordinates, including all
mixed terms; it tests the full chain independently of the black-hole profile.

Results from the final frozen matrix:

- 13,230 adjacent-radial-owner comparisons: donor IDs and every value/first/
  second derivative weight are **bitwise identical**.
- Constant error exactly 0.
- Radial polynomial jet error at the finite query surfaces at most 5.16e-10.
- Complete mixed radial/angular polynomial Cartesian jet error at most 7.79e-10.
- Existing global value interpolation agrees within 1.17e-13.
- Twenty negative shape/stale-geometry controls reject as intended.
- Maximum measured value L1`1031.21`, radial condition`44321.54`, partition
  error`7.44e-13`; all declared geometry guards pass.

Those mathematical consistency controls do not certify the coarse angular
approximation. For seven shells, full Cartesian Hessian max errors for the
smooth global manufactured fields are 31.78/8.02/3.93 at N6/8/10. The last observed
rate is 3.20, below the intended fourth order. Angular-owner Hessian jumps are
37.43/9.87/4.21. The one-shell N6/12/18 and two-shell N6/10/14 controls reach finer
angular scales and reduce much farther, but do not establish that the affordable
seven-shell family is asymptotic. No global fourth-order or exact angular C1
contract is claimed. The production sampler is left unchanged.

## Original initial-slice mass control only

A private extraction executable switches only the diagnostic sampling call to
the new jets. It uses the exact same 50,000-cell original hyperboloidal isotropic
M=1/3 initializer, positive storage, physical expansion kernel, eight radii,
and 128/512/2048 Gauss–Legendre×phi angular quadratures as the failed baseline.
It also repeats sampled Minkowski and direct original-wormhole controls.

The r=.7 results are

| Angular points | Old owner-only jets | New global radial-chain jets |
|---|---:|---:|
| 128 | .355424829037 | .333333333344850 |
| 512 | .352301348316 | .333333333344834 |
| 2048 | .355303541921 | .333333333344853 |

The new error from 1/3 is about 1.15e-11 and the quadrature spread is about 2e-14.
The same exact sphere can retain different *geometric primary owners*, but its
radial-chain value/jet no longer uses different inner/outer one-sided fits.
This removes the identified radial seam artifact without choosing an owner or
changing any input field.

Every other radius remains in the control. At 2048 angular points the r=.05
initial mass is still`.332675270833`; the Cartesian-core sampler is unchanged
and underresolved. At r=.3 the mass improves from`.333313727976` to
`.333323758568`, but a significant error and nonmonotonic angular variation
remain. Other angular/interpolation errors also remain. The targeted seam repair
therefore does not qualify the full mass extraction or physical evolution.

The private executable **explicitly rejects `mode=final`** before field sampling;
its negative control records that rejection. No evolved final-state mass was
computed with the unqualified helper, and no coupled run, elliptic solve, or PDE
change occurred. The rejection is a scientific qualification guard, not a request
for additional user permission or a mathematical impossibility.

## Frozen evidence and reproduction

- `evidence/radial-interface-jet-mms-20261027/`: exact commands, all-case JSON,
  source and dependency proof. Source archive SHA256
  `da3ca4f1ea2c8f559b187f17f797c7ae4616c37062949a4134f750874daa8c9a`.
- `evidence/radial-interface-initial-mass-20261027/`: initial/Mink raw complete
  directions, physical areas and expansions, failure/rejection evidence and
  sampling-only patch. Source archive SHA256
  `ab08e8dcbc26f20d4a9e21549e12acf36bd60d32ab338aa2f02b29394e638518`.
- `evidence/radial-interface-jet-analysis-20261027/`: complete controls, observed
  rates, owner-side mass data and plots. Earlier pilot matrices are preserved
  separately. An intermediate matrix evaluated the mixed/negative controls but
  omitted their values from JSON; the final source adds those output fields and
  reruns the matrix, retaining the earlier output.

All builds use private source/Kokkos capsules, with zero live project dependency
paths. Every process is bounded by 180 seconds and uses one sequential CPU worker.
No shared CMake, production binary, existing job or physical kernel is edited.

```
PYTHONDONTWRITEBYTECODE=1 python3 tst/elliptic_shift_3d/run_radial_interface_jet_probe.py \
  --output docs/elliptic-shift-3d/evidence/NEW-radial-jet-mms --matrix all
PYTHONDONTWRITEBYTECODE=1 python3 tst/elliptic_shift_3d/run_radial_interface_mass_control.py \
  --output docs/elliptic-shift-3d/evidence/NEW-initial-mass-control
PYTHONDONTWRITEBYTECODE=1 python3 tst/elliptic_shift_3d/analyze_radial_interface_jet_probe.py \
  --mms docs/elliptic-shift-3d/evidence/radial-interface-jet-mms-20261027 \
  --mass docs/elliptic-shift-3d/evidence/radial-interface-initial-mass-20261027 \
  --output docs/elliptic-shift-3d/evidence/NEW-radial-jet-analysis
```

The remaining work is a separately qualified global angular/Cartesian-interface
jet contract and an extraction control for the core. The present result is a
reproducible radial-interface correction with explicit remaining failures, not
production integration or completion of the scientific milestone.
