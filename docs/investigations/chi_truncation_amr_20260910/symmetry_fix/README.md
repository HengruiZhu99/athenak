# VC Cartoon AMR reflection and derefinement repair

The N512 error-based mesh first loses z-reflection symmetry at t=5.6536244035322305, cycle4134, event12. At t25 there are50unmatched leaves of356. Native binary logical locations and coordinate bounds agree exactly with the independently recorded AMR tree; this is not a rendering error. The original dchi tree remains symmetric through25.

Per-block dchi and error-sensor maxima previously omitted shared vertices assigned to another block's canonical diagnostic ownership. That ownership prevents double counting in global sums, but makes per-block support depend on ordering and can remove non-reflected faces from mirrored blocks. The repair includes all active vertices in each block's maximum and matching dchi argmax. Global integral ownership and evolution shared-node synchronization are unchanged.

The D5/D6 estimator now groups reflected sample pairs before summing. Reversing the seven-point stencil reverses the odd derivative and preserves the even derivative with the same arithmetic grouping, including shifted block-edge stencils.

The former child derefinement factor0.25 is insufficient for a smooth fourth-order error: refinement divides the estimate by16, causing immediate eligibility for derefinement. The new chi_error_parent_derefine_factor (default0.25, strictly between0and1) caps the effective child factor at parent_factor/16. The existing chi_error_derefine_factor remains supported as an additional stricter cap. Default effective child factor is1/64. Refinement thresholds, resolution scaling, radius floors and maximum levels remain unchanged. This is a predicted smooth-parent error bound, not a full coarse evolution estimate; it does not guarantee absence of cycling on nonsmooth data.

CARTOON_AMR_TRANSACTION formerly printed literal mirror_mismatches=0. It now prints unchecked. This field is explicitly not a measured symmetry diagnostic; audit saved AMR trees for actual reflection mismatches. Cartoon SO(2) alone does not require equatorial reflection, so the repair does not forcibly mirror fields or meshes for arbitrary Cartoon data.

Validation: CUDA build, sensor polynomial/scale/Nyquist/shifted-stencil/reflection/hysteresis tests and radius AABB regression passed. Chi AMR refresh static test passed. Short fresh-data GPU evolution tests are recorded separately when complete; no long production campaign resumed. A remaining asymmetric evolved mesh is not evidence of physical symmetry breaking and requires follow-up on mirrored evolved fields and AMR decisions.

## Completed GPU validation

Shared-interactive job58158999 completed0:0 in8m38s. Immutable test binary is under /pscratch/sd/h/hzhu/chi-truncation-amr-20260910/validation_symmetry/athena, built from implementation commit c802bcaf77f143cf3e83ce267cc93d5f59f2daa7; executable.sha256 records its hash. All three modified production source files were independently hash-matched between local and Perlmutter source. Both runs began from the same original spectral initial data, not from asymmetric evolved checkpoints.

| Test | Original AMR events | Fixed events | Original deletions | Fixed deletions | Fixed unmatched leaves |
|---|---:|---:|---:|---:|---:|
| N128 to nominal t8 |351|11|1329|12|0 throughout|
| N512 to nominal t6 |30|6|111|12|0 throughout|

Original history comparison samples are at8.001912 and6.007515; fixed runs end exactly8and6. Events include the initial mesh record. Native field files are float32: chi and lapse are reflection-identical at saved precision in both final snapshots. Largest Khat reflection differences are7.28e-12and9.31e-10, respectively; Theta differences1.16e-10and1.82e-12. See field-reflection.json for scales and provenance. These checks do not assert exact double-precision evolution symmetry or long-time convergence.

At final N128 C-norm2 is0.07677 versus0.07475 originally; N5120.00008529 versus0.00008558. Changing the AMR support and coarsening rule changes the mesh and trajectory. Reduced churn and restored short-time mesh symmetry do not by themselves prove lower evolution error. No long campaign has been resumed. Re-run fresh longer resolution comparisons before interpreting late-time classification.
