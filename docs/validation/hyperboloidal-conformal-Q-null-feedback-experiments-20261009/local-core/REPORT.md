# Local conformal-Q/null-feedback gate — exploratory PASS

The recommended shared helper is qnf::Gauge(p,u,g,{.85,.95,5,true}) followed
by qnf::Assemble. The input must explicitly have physical_trace_lapse=false,
preferred_source=true and, for the proposed a=.5 comparison,
scri_lapse_damping=2. The helper does not override the input. The true parameter
preserves the physical-P lapse inside W=0 and blends to original harmonic Q
at W=1. Physical P storage and the complete C0 tensor RHS remain unchanged.

The exact original Q/preferred control was already rejected. The new part is
its null-residue beta pole, together with the explicitly named alpha-only inner
blend. DERIVATION.md gives the code equations and comparisons with the prior
physical-P/projected/null-feedback native failure.

Final receipt: 382 unchanged input paths = 365 tracked src/root-CMake paths
plus 17 current/history scratch sources. All 14 commands returned zero with
empty stderr. Launch HEAD is ad68e9701060cec99651368e36377f482653bc32;
compiled production implementation remains 27c19d20696ea6dd4704032c51dfd026218f64f2.
The compiler flags, binaries, source-before/after hashes and failed exploratory
receipts are retained. Release and Debug ASan/UBSan reproduce identical
full20 and nonlinear JSON data.

Completed evidence:

- Sixteen actual dual full20 reference pole matrices, a=.5,.75,1,2,
  kappa_input=5/10, sigma=0/5. All eight sigma5 matrices have eleven negative
  nonzero roots and nine semisimple zeros. Exact reconstructed rank M=rank M²=11.
  Maximum rational reconstruction discrepancy 3.553e-15. Sigma0 reproduces the
  known positive normal cubic, including +2.57170948731154 at a1/k5.
- Exact normal-factor identity and Routh threshold
  K>3/4, sigma>6K/(4K-3), K=kappa_input*a², S=1. Sigma5 requires K>15/14.
  Exact assertions apply to the rationally reconstructed analytic reference
  pole matrix, not a general nonlinear Omega0 state or uniform PDE proof.
- 960 directional finite-difference comparisons against the actual dual pole
  Jacobian. Max errors at eps1e-3,1e-4,1e-5 are 4.000e-5,4.000e-7,4.037e-9.
- 504 actual principal matrices, both alpha-blend variants, alpha=.2/1/3,
  chi=.4/1/2, aligned and oblique SPD determinant-one frames, fourteen radii
  through the gauge/source collars. Kernel error 3.553e-15; eigenfield error
  8.882e-16; normalized basis condition <=11.530. Complete harmonic endpoint.
- 640 nonlinear reference/blend samples for xi1.5 and xi=1/a. Reference gauge
  fixed-point residual <=7.550e-15; factored/blended production-equation
  discrepancy <=8.882e-16. This covers the native target xi2 transition.
- 160 independent four-dimensional metric/Christoffel source tests on
  off-constraint SPD states, including nonflat r=.85–.95. Gamma4+2Z4 identity
  error <=7.471e-12; Box extension error <=8.413e-12. Exact source/Box claims
  are restricted to W=1; the inner blend uses a chosen algebraic extension.
- Weighted null numerator versus alpha²deltaN discrepancy <=1.601e-15.
  Forty-eight tiny-positive-alpha exact-core cases and 320 noncore cases
  (r=.5/.9, zero/generic beta, both variants, alpha down to1e-320) have finite
  actual gauge sources. No uniform GH-source/lapse-hyperbolicity assertion.
  Two-sided cutoff source jump at h1e-8 is <=2.491e-8.
- The Einstein gauge witness deltaAlpha=Omega,deltaBeta=-Omega n has exactly
  zero initial H/M/Z/Theta and R0 in the actual dual kernel. Fourth-order
  corner limits satisfy alphaDot0=1/a²,betaDot0=0,PDot0=3/a² and
  NrawDot0=QnumeratorDot0=0, error <=1.889e-13.
- The archived physical-P/projected/sigma helper also cancels that initial
  null/Q corner. A separate cheap actual gauge finite difference reproduces
  alphaDot0=-(1+2xi*a)/a²,betaDot0=(2+2xi*a)/a², with corner error8.117e-9.
  The present test therefore does not erase the earlier negative native result.
- The finite-Q counterexample remains: P-Pref=.01Omega yields
  Omega QDot->.02/a²,ThetaDot->-.02/a², and a nonzero Lambda leading pole.
  It lacks full first-jet/Einstein compatibility. No finite-Q-amplitude blowup
  claim follows from its nonzero initial corner derivative.

Failed exploratory stages are preserved under history: a nonlinear probe
compile failed on a local identifier and missing Omega convenience function;
a JSON reader failed on hand-written `.001` metadata; and the first SymPy
structural assertion compared generators with different assumptions. Their
exact source/output/receipt context is retained. The fixes do not modify the
final mathematical helper. The earlier prior-corner eps1e-5 probe is retained
alongside the final more accurate eps1e-3 comparison.

Not completed or accepted: finite-Omega lower-order/frozen-frequency stability,
full20 first-jet R0 closure, coefficient-aware/global boundary stability,
nonlinear Einstein/null persistence, an energy estimate, native pulse
improvement, or BH/trumpet compatibility. Pole and principal checks do not
imply any of these. Positive finite-Omega roots must be retained in follow-up.
No native compilation/evolution or tracked production edits occurred here.

Reproduce from repository root with the existing numpy/sympy Python:

    python build-layer-research/continuum/conformal-q-null-feedback/run_audit.py

The frozen index is the authoritative byte identity. Do not rerun collectors
against the immutable snapshot or mutate prior controls.
