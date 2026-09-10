# χ derivative truncation-error AMR

Scope: implement and qualify the higher-derivative approach requested from Rashti et al., arXiv:2312.05438v2 §3.2, then run A=-0.04896875 at N128/N256/N512 against the preserved dchi series. This replaces the earlier proposed full-evolution Richardson shadow approach.

For the existing fourth-order centered spatial operator, estimate leading errors E1=h^4 |D5 chi|/30 and E2=h^4 |D6 chi|/90 using seven-point second-order D5/D6 stencils. Take max(length*E1,length^2*E2) over active directions and canonical diagnostic vertices per block. The fixed normalization length defaults to1 in existing units. E2 detects Nyquist modes that an odd derivative cannot see. This is a derivative-error proxy; it excludes gauge coupling, temporal integration error, boundary truncation order and accumulated global error. It is not a certified evolution-error bound.

Input method=chi_truncation, required chi_error_max, chi_error_reference_nx1, chi_error_length, chi_error_derefine_factor. Fourth-order scheme only; require at least3 ghosts. Scale tolerance by(reference_nx1/current_nx1)^4 with unchanged physical domain and fixed normalization length. Existing minimum physical refinement regions and caps remain. All other physics, initial coefficients, lapse stopping and device synchronization remain unchanged. Do not resume the failed bisection.

Qualification gates: analytic stencil consistency and h4 scaling; Nyquist response; finite-value/input guards; same-level and AMR-interface stencil behavior; axis and outer-boundary response; CUDA build and short GPU runs; dchi trajectory parity with new executable; establish and record one reference threshold before series launch. No threshold retuning per resolution or after looking at classification.

Compare full histories (C2 and component norms, global lapse, Kretschmann), runtime and AMR counts/levels, common-time values, and failure locations. Three resolutions do not guarantee convergence, especially with different live AMR trees or near-threshold classification changes. Preserve all failures; do not tune numerical guards to make runs pass.

## Implementation details and caveats

The seven-point sensor reads ±3 native neighbors after the ordinary evolution ghost synchronization, uses canonical VC diagnostic owners, and skips suppressed directions. It retains existing radius floors and maximum-level enforcement. Both refine and derefine limits scale with resolution; hysteresis defaults to0.25. A conservative floating-point cancellation floor, 256*epsilon*max|chi|, is subtracted from each unscaled derivative numerator to prevent roundoff amplified by h^-1/h^-2 from forcing refinement. Nonfinite samples or estimates are fatal.

Unlike the paper's sixth-order first-derivative proxy and h^4 local error band, this fourth-order adaptation uses a fixed reference tolerance scaled only between root resolutions. Scaling a fourth-order sensor by the *local* h^4 at every AMR level would largely cancel its dependence on level in smooth data and would not provide the intended error-control mechanism. The normalization length is fixed in physical code units; it must be rescaled if the physical units/domain change.

The estimator sees the existing boundary and inter-level ghost data, so lack of smoothness at such interfaces can trigger refinement. The leading smooth-data Taylor estimate is not a proof of accuracy there. Do not claim CC/full3D qualification based on the VC Cartoon tests. The first production comparison remains VC Cartoon only.
