# Separate Padé13 exponential replay

The preceding limited J0/N8 run remains failed at t=.25 because SciPy expm
raises a floating-point divide-by-zero error in its matrix-power selection.
Fresh isolation reproduces the error even for identity and zero products with
correct finite output arrays. This identifies no C-level cause and changes no
failed classification. Warnings remain errors.

This fresh copy replaces only the exponential implementation with the separately
reviewed real binary64 fixed degree-13 Padé scaling-and-squaring helper. Literal
unoptimized einsum performs every matrix product; the rational system uses a
linear solve. Its formula and norm-one threshold follow Al-Mohy and Higham
(2009), equations 3.5–3.6, Table 3.1 / Algorithm 3.1. It is not Algorithm 5.1.
Primary source: https://eprints.maths.manchester.ac.uk/1217/1/paper9.pdf

The helper has separate exact coefficient/Taylor, synthetic and independent
100/130-digit mpmath checks. It supplies no arbitrary-nonnormal forward-error
certificate. Overscaling and squaring roundoff remain possible. Keep the
original half-time product consistency gate at2e-7 and short RK3 gate unchanged.
Record every exponential's scaling and rational solve residual; use the existing
2e-9 algebra tolerance on that solve residual. This residual is not a matrix
exponential forward-error bound. Underflow, overflow and nonfinite arithmetic
remain errors in the helper. No fallback, warning filter or clipping is added.

All generator, E transform, spectrum, seed amplitudes/envelopes, SAT, propagation
times, guards, RK3 and saved physical-readback payload are unchanged. Original
PLAN/addendum/scope are retained verbatim as history; this additional file and
helper are mandatory reviewed admission pins. The full nongauge continuum
comparator stays unresolved. This remains a limited finite-matrix diagnostic,
with no PDE, native Cartesian, nonlinear, scri or black-hole stability admission.
Scientific replay requires root/independent source review and exact new
admission. Never reuse the failed output path. Start with J0/N8/rb=.98 only.
