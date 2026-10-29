# Optional seven-node radial closure: CPU control, not stability admission

This control replaces the last **three radial cell rows** of the outer shell with direct one-sided derivatives using six active centers and the evolved scri endpoint. It also uses seven radial nodes for surface jets and sampled complete pole-numerator differentiation. Penalty coefficients remain **4 N/h − 8 C/h**. Angular differences remain centered five-point with cross-face Cartesian-component interpolation. All first, second and mixed Cartesian chain-rule terms are retained; radial transport uses the same one-sided derivative in this region, while existing upwinding remains elsewhere.

The purpose is to test the ghost-free radial coupling used by the successful 1D approach. It is **not an exact copy of the N513 donor operator**: the 3D half-cell layout, mapped angular derivatives, bulk fourth-order discretization and coordinate geometry remain different. This is not an SBP operator or a discrete-energy stability proof. No regularity field is reset, no spherical averaging is introduced, and the independent Lambda/Theta and final dependent-D ordering are preserved.

`HyperbolicComposite::RadialClosure::donor_seven` is an explicit CPU-only option, requiring sampled-normal mode and cross-face surface jets. Existing callers retain their six-node V5 or analytic-jet behavior. Device scri/sampled-normal constructors reject seven-node inputs; no MPI target, checkpoint schema or Aurora admission has been added for this control. The bounded CPU duration harness supplies compensated RK4 and final algebraic projection separately; the legacy `Model::Step` is not used for this candidate.

## Consistency and unchanged-path checks

The scalar derivative component test passes:

| Check | Result |
|---|---:|
| Constant first/second derivative | exactly zero |
| Degree-six radial polynomial, mixed derivatives and full mapped J/H | 1.11e-14 maximum scaled error |
| Smooth Cartesian exponential, first-derivative RMS n8 / n16 | 7.65634e-4 / 5.41257e-5 |
| Smooth Cartesian exponential, second-derivative RMS n8 / n16 | 2.96978e-3 / 1.08302e-4 |
| Measured orders over these two resolutions | 3.8223 / 4.7772 |
| Exterior radial ghosts poisoned with NaN | all selected derivatives remain finite |

The exponential fixture supplies analytic padded volume values and exercises interpolated surface angular donors. It tests the derivative plan and chart conversion, not the accuracy of a complete numerical halo exchange or the full PDE boundary rate. The polynomial check uses rows whose angular stencils stay inside a face. Constants and the exponential cover all six faces and seams. These are component results, not evolution convergence.

The existing sampled-normal device component passes (Serial backend). The CPU sampled-normal regression initially rejected an overly broad guard permitting seven nodes in legacy six-node mode. The guard was corrected to require the node count selected by the immutable closure option; the same regression then passes in 11.04 seconds. Both the initial failure and corrected result are retained. The initial component-build attempt required CMake regeneration after adding the target; it was a missing-target setup error, not a compilation or numerical failure.

## Complete coupled action

A fresh candidate library evaluates both saved growing directions at amplitudes 1e-4, 1e-5 and 1e-6. At all **twelve signed inputs**, the rebuilt default V5 full RHS is bitwise identical to the preserved pre-change V5 library. Candidate actions vary by at most 4.01e-7 relative across these amplitudes. Surface and ghost null residuals remain below 1.23e-15, and dependent-D compatibility below 9.42e-22.

At amplitude 1e-5:

| Observed input direction | New whole-state projection | New volume projection | New surface projection | Whole-state shape residual |
|---|---:|---:|---:|---:|
| V4 at t=.30 | +100.9265 | −114.1006 | +111.4509 | .5066 |
| V5 at t=.60 | +158.6059 | −64.3369 | +184.3487 | .4680 |

Thus the control changes both sides of the coupling, opposing the old volume direction while increasing the old surface direction. It is **not a demonstrated improvement**. The large shape residuals mean these projections cannot substitute for a coupled trajectory or eigenvalue calculation. A bounded fresh n6 Minkowski comparison was launched to observe that trajectory; no result from it is claimed in this package. It uses unchanged state/constraint/null limits and records any wall-bound outcome separately. Its local output directory is `output/hyperbolic-core-shell/scri-donor-seven-duration-v1`.

The action completed in 61.27 seconds with one process, under a 350-second internal and 385-second external bound. [action.json](action.json) contains all norms, amplitudes and source identities. The build manifest binds the precise pre-guard-repair action snapshot preserved in `action-source/`; the subsequent guard repair changes rejection of inconsistent configurations, not the valid seven-node arithmetic exercised here. [additional-source.json](additional-source.json) binds the test/control/duration drivers. Full state arrays and shared libraries remain in local output, not Git. No Aurora jobs were submitted for this control.

Reproduce the action with a new build directory:

```sh
python3 tst/hyperbolic_shift_3d/scri_candidate_replay.py build --output output/donor-seven-build-new
python3 tst/hyperbolic_shift_3d/scri_donor_seven_control.py \
  --library output/donor-seven-build-new/candidate.dylib \
  --reference output/hyperbolic-core-shell/scri-v5-direction-build-v1/candidate.dylib \
  --v4-checkpoint output/hyperbolic-core-shell/aurora/linear-window-dissipative-v2/output/snapshot-030.checkpoint \
  --v5-checkpoint output/hyperbolic-core-shell/aurora/stability-sampled-normal-minkowski-v1/output/minkowski.checkpoint \
  --output output/donor-seven-action-new
```

The component target is `hyperbolic_shift_3d_scri_radial_difference`; the explicit duration target is `hyperbolic_shift_3d_scri_radial_duration`. The latter accepts `variant n target output-prefix`, with variants 1 (V5) / 2 (seven-node control), n6 only, target at most 2, dt=.008/n and a 325-second native bound. It stops on the first recorded science failure and never issues checkpoint admission. Use an external 350-second timeout as in the captured local driver. Broader stability, GPU parity, puncture evolution and convergence remain unverified.
