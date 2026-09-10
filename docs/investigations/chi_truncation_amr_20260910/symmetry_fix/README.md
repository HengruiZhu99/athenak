# VC Cartoon AMR reflection and derefinement repair

The N512 error-based mesh first loses z-reflection symmetry at t=5.6536244035322305, cycle4134, event12. At t25 there are50unmatched leaves of356. Native binary logical locations and coordinate bounds agree exactly with the independently recorded AMR tree; this is not a rendering error. The original dchi tree remains symmetric through25.

Per-block dchi and error-sensor maxima previously omitted shared vertices assigned to another block's canonical diagnostic ownership. That ownership prevents double counting in global sums, but makes per-block support depend on ordering and can remove non-reflected faces from mirrored blocks. The repair includes all active vertices in each block's maximum and matching dchi argmax. Global integral ownership and evolution shared-node synchronization are unchanged.

The D5/D6 estimator now groups reflected sample pairs before summing. Reversing the seven-point stencil reverses the odd derivative and preserves the even derivative with the same arithmetic grouping, including shifted block-edge stencils.

The former child derefinement factor0.25 is insufficient for a smooth fourth-order error: refinement divides the estimate by16, causing immediate eligibility for derefinement. The new chi_error_parent_derefine_factor (default0.25, strictly between0and1) caps the effective child factor at parent_factor/16. The existing chi_error_derefine_factor remains supported as an additional stricter cap. Default effective child factor is1/64. Refinement thresholds, resolution scaling, radius floors and maximum levels remain unchanged. This is a predicted smooth-parent error bound, not a full coarse evolution estimate; it does not guarantee absence of cycling on nonsmooth data.

CARTOON_AMR_TRANSACTION formerly printed literal mirror_mismatches=0. It now prints unchecked. This field is explicitly not a measured symmetry diagnostic; audit saved AMR trees for actual reflection mismatches. Cartoon SO(2) alone does not require equatorial reflection, so the repair does not forcibly mirror fields or meshes for arbitrary Cartoon data.

Validation: CUDA build, sensor polynomial/scale/Nyquist/shifted-stencil/reflection/hysteresis tests and radius AABB regression passed. Chi AMR refresh static test passed. Short fresh-data GPU evolution tests are recorded separately when complete; no long production campaign resumed. A remaining asymmetric evolved mesh is not evidence of physical symmetry breaking and requires follow-up on mirrored evolved fields and AMR decisions.
