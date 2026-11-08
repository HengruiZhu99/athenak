# Saved-state CPU/Aurora attribution

The native Aurora saved-step3551 replay passes the unchanged historical GPU
reference gate. The local Apple arm64 replay still fails that gate. This
comparison preserves both outcomes; it does not relax the gate or establish
trajectory equivalence. In particular, the separate eight-step GPU-versus-CPU
field comparison remains failed pending the unchanged GPU evolution control.

Both dumps contain all 229,392 evolved points, with identical canonical leaf
and cell identities. Within each platform, native and independent legacy
coordinates matched bitwise. Across platforms, 72,030 coordinate components at
54,072 points differ, by at most 3 ULP or 2.22e-16. The runtime constructs these
coordinates on the host before uploading them to the device.

The strict `sqrt(x*x+y*y+z*z) > .94` mask differs at 118 points, all with positive
quadrature weights and all with changed coordinate bits. There are 64 points
selected only on the local CPU and 54 only on Aurora, giving 9,874 versus 9,864
selected points. Every flip lies in the diagnostic rounding band around .94.
The host and device radius bytes and masks agree at **every point within each
platform**. Thus these records do not support a device-versus-host square-root
mask discrepancy on Aurora: the cross-platform coordinate inputs differ.
The complete harmonic horizon exterior masks agree across platforms.

For attribution, the stored pointwise constraints, weights and masks were
independently recombined in canonical order with `math.fsum`. These are
counterfactual controls, not replacement device/MPI observables. Their original
CPU/GPU combinations reproduce the recorded outer RMS values within 8.7e-18.
Starting from CPU values, the following changes isolate the dominant effect:

| Control change | H RMS change | M RMS change | Theta RMS change | Z RMS change |
|---|---:|---:|---:|---:|
| Substitute Aurora mask only | +1.51058925e-6 | +7.15957892e-7 | +2.75870863e-8 | -1.71518681e-7 |
| Then substitute Aurora weights | 0 | 0 | 0 | 0 |
| Then substitute Aurora constraints | -1.43735726e-10 | -6.66897093e-14 | 0 | -2.98155597e-18 |

All eight field/weight/mask combinations and both orientations' weighted
contributions are retained in the full analysis. Weights differ at 17,990
points, with maximum absolute difference 6.78e-20; their substitution makes no
change to these rounded RMS controls. The full-domain H difference, which
cannot involve a cutoff mask, remains 5.81003e-11. This analysis does not assign
that smaller difference to one particular operation.

A separate source review identified a concrete portability issue in derived
parameter resolution. `NativeWormhole::Gaussian` multiplies and accumulates in
`long double`; the integral feeds `exp`, compact radius and isotropic scale.
`CentralZ4cConfiguration::Resolve` derives all nine differing identity doubles
from those two quantities. Compiler predefined macros confirm a 53-bit
mantissa with eight-byte storage on Apple arm64 and a 64-bit mantissa with
sixteen-byte storage on Aurora. The differing quadrature precision is real;
its contribution has not been isolated from other compiler/libm differences.
The strict numerical-identity mismatch remains recorded.

Reproducible analysis and exact hexadecimal coordinates for all 118 mask flips
are under `build-native/saved-state-cross-architecture-attribution/`. Compact
results and artifact hashes are in
[evidence/native-saved-state-cross-architecture-attribution.json](evidence/native-saved-state-cross-architecture-attribution.json).
The original CPU and Aurora audit files were not modified, and no runtime
source or tolerance changed.
