# Full-coupled sampled-normal directional action

The optional CPU sampled-numerator closure changes the projection of the measured `.30` V4 direction from **+72.9635 to −22.3588**. The complete candidate action includes final dependent-D reconstruction and refreshed ghosts. Its relative shape residual is **0.6888**, however, so the negative projection does not establish a decaying eigenmode or stable evolution. No fields were evolved. Independently measured manufactured surface-rate errors are roughly third order; this candidate does not preserve the assumed fourth-order accuracy of the original discretization.

The input is the source- and worker-bound n16 epsilon2 checkpoint from [the early-window acquisition](../aurora/linear-window-dissipative-v2/acquisition/README.md). It is scientifically unadmitted despite its native capture-completed flag. The existing [directional replay](../scri-directional-replay-v1/README.md) defines the tangent coordinates and identified the normal term; the [manufactured accuracy study](../scri-sampled-accuracy-v1/README.md) assesses truncation separately.

At maximum coordinate amplitude `1e-5`, using unweighted point-coordinate/Frobenius norms:

| Region/fields | Default AD projection | Sampled-normal projection | Candidate shape residual |
|---|---:|---:|---:|
| Whole direction | +72.96352870 | −22.35878134 | .688793 |
| Surface | +72.96472155 | −27.02306463 | .0978650 |
| Volume | +72.93915717 | +72.93916912 | .0147464 |
| Volume Lambda | +72.42216249 | +72.42216249 | .0227110 |
| Volume D | +72.39675966 | +72.39762972 | .0230981 |

The candidate-minus-default action norm is `97.6593` globally, dominated by the surface. Its volume norm is `.000407051`, entirely in D for this direction. The existing volume component therefore continues to point approximately along the original growing direction at this instant; the modified surface response will change its later forcing. This single action cannot predict that subsequent coupled trajectory.

The adapter evaluates `T0 RHS(P(U0+E z))` at Minkowski. It uses the same native 26 volume/19 surface coordinates, algebraic retraction, null reconstruction and complete `EvaluateRHS` for both immutable normal selectors. The sampled selector first supplies provisional geometric ghosts, computes the complete numerator samples on the same five half-cell donors plus endpoint, finalizes all boundary rates including angular Q, reconstructs dependent D, and refreshes ghosts before bulk evaluation. This is a tangent generator, not the accepted RK map or a Jacobian evaluated at the nonlinear checkpoint. No term formula is implemented in the adapter.

The four amplitudes `1e-4,3e-5,1e-5,3e-6` give candidate-action changes of at most `5.20e-7` relative to the `1e-5` result. Default AD is **bitwise identical**, including signed zero, to the frozen prior v4 oracle for the background and all eight signed perturbations. Independent repeated/interleaved candidate calls are also bitwise identical. These checks compare complete arrays, not just norms.

Across all calls, actual surface and ghost null residuals are at most `1.11e-15`; dependent-D compatibility is at most `8.35e-22`. The bulk speed-gap lower bound is `.833165`, maximum bulk CFL `.323920`, and isolated angular-filter CFL `.402273`. None is a full-system stability certificate. Background RHS maxima are `7.71e-12` for default AD and `9.87e-12` for the sampled candidate at n16. The process made 9 default, 12 candidate and 9 frozen-reference RHS evaluations, completing in 44.33 seconds. The external timeout was 385 seconds and internal budget 350 seconds; one local computational slot was used.

The run used separate new `scri_candidate_probe.cpp` and `scri_candidate_replay.py` files; old oracle sources/libraries were unchanged. `source.json` records every captured native-header hash and the resulting library identity. The exact modified CPU header, adapter, driver and shared checkpoint decoder are retained under `source-tree/`; the other unchanged native headers are identified by hash. The prototype header was uncommitted when built, so the recorded Git HEAD alone is **not** its source identity. `build.json`, immutable source snapshot and binary hash bind the actual code. The library itself and full checkpoint are omitted from this portable package.

Reproduction requires the captured header/adapter source and the unchanged native dependencies identified in `source.json`, plus the original raw checkpoint and frozen v4 library. Build into a fresh directory, then run:

```sh
python3 tst/hyperbolic_shift_3d/scri_candidate_replay.py build --output output/candidate-repeat
python3 tst/hyperbolic_shift_3d/scri_candidate_replay.py run \
  --library output/candidate-repeat/candidate.dylib \
  --frozen-library output/hyperbolic-core-shell/scri-directional-replay-v4/probe.dylib \
  --frozen-manifest docs/hyperbolic-shift-3d/reproduction/scri-directional-replay-v1/v4/source.json \
  --checkpoint output/hyperbolic-core-shell/aurora/linear-window-dissipative-v2/output/snapshot-030.checkpoint \
  --output output/candidate-action-repeat
```

Use `.so` on Linux. The build refuses source changes during compilation. The run verifies both libraries against their manifests, binds the checkpoint to the original worker artifact, checks its exact n16/epsilon2/dt/.30 configuration, and requires full default-array frozen parity before reporting the candidate action. Rebuilding against a later working tree creates a new source binding. A stable Minkowski duration, puncture evolution, and convergence comparison remain separate work.
