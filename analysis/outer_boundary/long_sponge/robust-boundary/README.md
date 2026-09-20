# Coupled exterior boundary prototype

A causal **full-state exterior-memory boundary** removes the demonstrated exponential boundary eigenmode in a frozen, planar, sixth-order residual-Z4c test, while retaining the original G1 gauge and all production damping. A full-PDE RK3 pilot reaches 50000.4M. This is a numerical reference for developing a boundary condition, not an implemented or validated AthenaK black-hole boundary.

The distinction matters: the coupled pilot still has large transient amplification, and neither a simple rational fit nor an approximate normal exterior buffer was uniformly satisfactory. A small incoming-trace relaxation has a separate, narrower purpose.

## Full-PDE comparison

The retained interval has 32 cells at dx32M, tangential Fourier angle pi/8 (wavelength512M), flat coefficients alpha=chi=1 and beta=0. All20 trace-free evolved fields, original G1, kappa=.1, eta=2, lapse damping=.1, sixth-order derivatives and production KO dissipation are retained. No sponge is used. The reference exterior extends the identical discrete volume operator to8192 periodic normal cells, length262144M.

The initial direction is the actual leading eigenvector of the original finite-strip `zero_rate` operator. Its eigenvalue is +.00563488434476/M, and its weighted state norm is normalized to1. The weight multiplies the ten momentum-like fields by dx; it is a comparison norm, **not a proven physical energy**. Amplitudes below are linear normalized amplitudes, not a nonlinear metric evolution; the entire direction can be scaled arbitrarily.

| t/M | Original zero_rate norm | Coupled exterior full norm | Coupled retained-interior norm | Coupled max Theta |
|---:|---:|---:|---:|---:|
| 0 | 1 | 1 | 1 | 1.06296e-4 |
| 4999.8 | 1.71991e12 | 21.7304 | 20.1138 | 6.07188e-6 |
| 10000.2 | 2.96812e24 | 3.52297 | 3.23023 | 7.24019e-6 |
| 19999.8 | 8.77999e48 | .369778 | .335039 | 6.33441e-7 |
| 30000 | 2.60601e73 | .126653 | .0685314 | 4.73965e-8 |
| 50000.4 | 2.29582e122 | .0962348 | .00999365 | 3.74113e-10 |

These values use **the original explicit RK3 polynomial**, dt.3M. The old-boundary result is the exact scalar RK3 propagation of its verified eigenvector, not a nonlinear simulation that was allowed to overflow. The coupled result propagates all normal Fourier modes with matrix powers, which is algebraically the same discrete linear recurrence. It does not diagonalize ill-conditioned eigenvectors. Repeating dt.6M at the identical output times gives final signed-state differences of8.26e-5 relative weighted L2 globally,4.69e-6 in the retained interval, and5.07e-10 for the full Theta L2. The decay is therefore not an artifact of replacing explicit RK3 by an implicit integrator.

All8192 normal Fourier symbols at this **one tangential frequency** have Re(lambda)<0; the largest is about-3.27873e-8/M. RK3 amplification factors also lie below1. This does not establish stability at all tangential frequencies, variable coefficients, corners, a puncture, or AMR interfaces.

The full operator is substantially nonnormal in the chosen norm. Among257 normal Fourier samples and the listed times, the induced norm reaches about1761 at4999.8M. The selected unstable-boundary eigenvector has a much smaller but still substantial transient of21.73. These are sampled values, not global bounds, and no small-amplitude nonlinear stability theorem follows. The later Theta bump is retained in the plot rather than hidden by a fitted monotone envelope.

The final signed arrays also provide linear physical-constraint diagnostics with the same sixth-order derivatives. Full unweighted discrete complex-amplitude l2 final/initial ratios at50000.4M are Theta9.05e-6, Hamiltonian H1.95e-5, momentum M1.19e-5, and connection Q (or Z=Q/2)6.49e-7. Here H=partial_i partial_j h_ij+2 Laplacian chi, M_i=partial_j A_ij−(2/3)partial_i(Khat+2Theta), and Q_i=Gamma_i−partial_j h_ij. Diagonal Hessians use D2 and mixed derivatives use D1_iD1_j. Because D2 differs from D1 squared, H+div Q does not exactly reproduce the discrete Theta RHS identity. These l2 norms have no sqrt(dx) or proper-volume factor; ratios compare the same extended periodic initial/final state, including derivative terms from the initial zero-extension. These are flat linear physical-constraint diagnostics, not nonlinear H/M norms or a proof of discrete constraint preservation. `linear-constraints.json` records initial/final norms and normal peak locations; no unmeasured intermediate H/M evolution is inferred.

## What boundary was actually tested?

Let R=I+dt L+(dt L)^2/2+(dt L)^3/6 be the **complete** full-source discrete RK3 step. Partition its state into retained interior I and exterior E. Eliminating E exactly gives

```
uI[n+1] = RII uI[n] + RIE REE^n uE[0]
           + sum(j=0..n-1) RIE REE^(n-1-j) REI uI[j].
```

This is a causal boundary-memory operator for all fields jointly. It retains the same gauge, constraint, physical-radiation and source couplings as the exterior volume evolution. It does not set physical fields to zero or freeze their evolution. With zero initial data and zero exterior memory its floating-point response is exactly zero; a nonzero state gives a nonzero response.

`rk3_memory_check.py` explicitly constructs these matrix convolution coefficients in a small full-PDE example and compares their application to direct interior+exterior evolution:16steps agree to5.22e-16 relative, and literal zero remains zero. This is more than a boundary-forcing impulse test. The longer pilot uses direct equivalent exterior evolution to avoid expensive uncompressed history convolution. Thus direct history evaluation is tested for16steps, while50000.4M refers to equivalent full-exterior PDE propagation, not a50000M history-summation run. The internal exterior propagator is checked separately: in the actual32-cell/8-cell-interior fixture, rho(R_EE)=.99994258372 at dt.6M (implied growth-9.57e-5/M), with no eigenvalue above1+1e-10. Sampled kernel Frobenius norms decrease from4.245e-4 to1.987e-7 by~50000M. This excludes an internally growing kernel for this fixture only; it is not a proof for every larger exterior or tangential mode. It does not claim that the memory is already installed at an AthenaK mesh face.

The full periodic exterior is finite. The8192-cell domain's continuum-speed wrap time is roughly262144/sqrt(2)=185364M, well beyond50000M. This is only a travel-time estimate, not a strict finite-difference domain-of-dependence argument. An RK3 exterior-domain doubling at50000.4M changes retained signed states by4.71e-12 relative weighted L2, max absolute8.95e-15. An independent BDF2 full-PDE check at50000M doubles the exterior from4096 to8192cells; retained signed states differ by3.62e-11 relative weighted L2, with max absolute difference7.06e-14. The earlier2048-cell pilot is used only through20000M; its physical wrap time is about46341M, so it must not be presented as a clean50000M exterior test.

The BDF2 Schur-complement version is also independently checked:16steps of full and eliminated PDE solves agree to3.23e-15, with exact zero response. It is a second discretization crosscheck, not the basis for claiming compatibility with existing RK3.

## Implementation cost and the next falsifiable step

The reference shows that the measured exponential mode can be removed by changing the boundary while leaving the G1 bulk and production damping unchanged. It supplies a target that a cheaper approximation must match. It is not an inexpensive local replacement for the existing ten momentum RHS rows:

- A complete RK3 step enlarges the stencil reach. With radius4KO terms, its polynomial can reach12normal cells; a compatible memory operator can act on that interface band, not just the outermost cell.
- The exact memory contains all coupled fields and tangential dependence. Separate local face formulas require treatment of tangential Fourier coupling and face intersections. Independent copies on six faces are not justified at edges/corners.
- Retaining the entire exterior state is expensive. Direct convolution also has quadratic history cost without compression. Kernel compression must be checked as a **closed feedback system**, not only by a pointwise fit or stable auxiliary poles.
- The reference coefficients are frozen and flat. Curved/variable-background compatibility and discrete physical constraint preservation are additional requirements; this is not a proof that the present nonlinear boundary is constraint preserving.

The next bounded implementation candidate is a planar periodic-tangent **discrete transparent boundary** using this full RK3 memory, first without compression, then with validated structured compression. It should reproduce the signed-mode and zero/nonzero-response tests here before being generalized. A full black-hole C++ implementation is not supported by these tests alone.

## Negative controls: why simpler replacements were not promoted

Incoming-state relaxation alone, `dC_in/dt=-nu C_in`, changes the Laplace boundary row from lambda C_in=0 to(lambda+nu)C_in=0. Every pre-existing root with Re(lambda)>0 therefore survives. Direct matrix checks at three original roots and nu=0,2e-5,.001,.1 leave boundary singular values around1e-14. Such relaxation can remove retained neutral incoming initial data, but cannot be advertised as curing the original positive roots.

A184-pole rational fit to the exact coupled continuum DtN map used only strictly left-half-plane poles. Its training error was~.2%, and it excluded the old roots, yet it created new verified roots, including lambda=.000250530+i*.000754886/M. The full PDE and boundary defects at that root are2.42e-13 and1.10e-12. A low-frequency held-out relative fit error becomes enormous. Stable memory poles and a good training error do not establish stable feedback; this candidate is rejected.

A finite stretched normal exterior buffer with a consistent half-cell flux also remains unvalidated. A coarse96-node case has a verified root with growth1.00727e-4/M. A refined256-node case has growth6.67693e-7/M, within a practical50k growth budget but still positive; the sampled normal and tangential closure approximations are not a proven transparent boundary. These tests coupled an approximate continuum-normal map to continuum interior traces and must not be confused with the exact identical-stencil RK3 elimination above.

## Continuum CQ operator test, kept separate

`cq_reference.py` makes a time-domain BDF2 convolution quadrature from the exact frozen full-source exterior DtN map N:

```
sum W[j] zeta^j = N((3/2-2 zeta+zeta^2/2)/dt, k).
```

For a smooth causal chi/lapse trace, its finite nonzero normal response converges under dt16/8/4M. Differences relative to dt4 through16000M have ratio4.999, the expected second-order ratio5. Independent generating-function checks agree within1.12e-11; zero response is exactly zero. This **operator-response test alone** does not establish closed-domain stability; the separate full-PDE tests above address that distinction. Matching the interior and boundary discretization matters. Convolution-based nonreflecting kernels have primary precedents in [Lubich and Schaedle](https://doi.org/10.1137/S1064827501388741); preservation of convolution coercivity requires additional hypotheses and compatible time discretizations, as discussed by [Banjai and Lubich](https://arxiv.org/abs/1702.08385). No scalar-wave theorem is transferred to this Z4c operator.

## Reproduce

Python with NumPy/SciPy/Matplotlib. Vendored model hashes are recorded. No external source checkout is required; binary final arrays are generated rather than committed.

```
OPENBLAS_NUM_THREADS=1 python3 rk3_memory_check.py
OPENBLAS_NUM_THREADS=1 python3 exact_elimination_check.py
OPENBLAS_NUM_THREADS=1 python3 coupled_rk3_pilot.py
OPENBLAS_NUM_THREADS=1 python3 coupled_rk3_pilot.py 4096
OPENBLAS_NUM_THREADS=1 python3 exterior_memory_internal.py
OPENBLAS_NUM_THREADS=1 python3 coupled_pde_long.py 4096
OPENBLAS_NUM_THREADS=1 python3 coupled_pde_long.py 8192
OPENBLAS_NUM_THREADS=1 python3 compare_long.py
OPENBLAS_NUM_THREADS=1 python3 constraint_diagnostics.py
OPENBLAS_NUM_THREADS=1 python3 cq_reference.py --output cq-reference.json
OPENBLAS_NUM_THREADS=1 python3 plot_results.py
```

The stored results and plot report a bounded frozen linear investigation. No Aurora jobs, production inputs, C++ equations, or production restart state were changed by this package.

Independent reviews by the incoming-trace and GPU-results agents checked Fourier signs, component ordering, norm weights, RK3 polynomial/powers, memory indices and scope. No indexing defect was found; their requested qualifications about periodic extent, stage-composed halo, transient growth and one tangential frequency are incorporated. The GPU-results reviewer additionally reproduced a complex random-state interior RHS match1.86e-15, unit initial weighted norm, eigenmode defect1.49e-13, and exactly zero initial exterior. Numerical crosschecks remain the reproducible scripts/results above.
