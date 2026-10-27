Matched diagnostic histories, frozen Gamma-driver coefficient-four candidate.

Three-grid-t9: 394 exact common samples through 27M. H and M exterior RMS decrease at every sample. Theta and Z tie at their exactly zero initial values, then decrease. Peak and time-RMS AH mass errors decrease, but pointwise medium/fine ordering fails at 18.375M and 22.95M, with errors of order 1e-9 and 1e-10, respectively. These exceptions are retained; they are not called roundoff.

Two-grid-t23: 954 common samples through 69M. H, M, and absolute relative AH mass error decrease at every sample. This is two-grid duration evidence, not a three-grid field convergence order.

The norms exclude nodes inside the outermost apparent horizon and retain the run diagnostic exterior normalization. Time RMS uses trapezoidal weights on actual diagnostic times. No interpolation, smoothing, or removal of mass-error crossings is used. Initial zero values are omitted only where a logarithmic plot cannot display them; all arrays retain them.

Source/checkpoint/library/ancestry verification preceded comparison. The np.trapezoid implementation reproduces the original three-grid assessment exactly; an explicit trapezoidal weighted sum independently reproduces all time-RMS values. Raw input hashes and all matched arrays are archived. The three-grid trajectory assessment remains a separate field-based test; no uniform puncture-regularity or 100M convergence claim is made.
