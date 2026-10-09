# Full-C1 saved-state approximate mode comparison

The same cheap reduced-state diagnostic finds a reproducible full-C1
projected-continuous N16 pair near `1.915659 +/- 6.776526 i`. The four
noise-screened exports have actual generator residuals `3.69e-7`--`5.78e-7`
in generator units (`5.23e-8`--`8.20e-8` relative to max(1,|lambda|)).
The original C0 pair is near `1.916339 +/- 6.790995 i`. The approximate
growth change is only -.000680 (about -.0355%), and frequency change is
-.01447 (about -.213%). This provides no evidence that the covariant repair
removes the observed discrete growing direction. Small residuals do not give
a certified eigenvalue error bound for these non-normal operators, so these
numbers remain approximate/pseudospectral evidence, not a spectral theorem.

No new propagation, native evolution, sparse LU, production edit, or frozen
input mutation was performed. Only the existing C1/C0 matrices and histories
were read, followed by instantaneous native centered-RHS/constraint probes.

## Inputs and like-for-like method

The original C1 global manifest is `5a442325...dee4b`; its 89 files are
reverified. The new C1 t6 index is `836b8f11...2335c`, with 23 files,
2,485,735 bytes. The original C0 mode index `39689919...1d69` is also
reverified. Exact full hashes and artifact sizes are in `input-pins.json`.
J20 is forward column action, free20 point-major in active-cell k/j/i order,
shape32800 by32800, N16/span2.2. Saved arrays have 241 times and two separate
unit seeds, gauge_pulse and shell_random. C0/C1 initial arrays, saved times,
and reference coordinate/weight/inverse-metric metadata are bitwise equal.

The C1 t6 history is exploratory Krylov, with the independently checked old
t0--2 prefix agreeing within3.878e-14, local pair error<=9.835e-11, and
actual curve/state defect<=2.825e-10 per unit time. It is not an independent
long canonical evolution. The original history generation's invalid-matmul
warning remains in its own immutable source/log bundle with no assigned
cause; finite states and scalars were checked there.

The nine window/seed/stride/raw-column choices, SVD rank ladder, residual
definitions and 1e-10 heuristic singular-fraction screen are the same as the
C0 diagnostic. The exact source changes are recorded in `preparation.json`.
Dense projections use explicit finite-checked contractions from the start.
All nine searches completed in47.393219417s, exit0, empty stderr. The screen
is a heuristic, not a rigorous history-error bound. All reduced roots and
residuals are retained. Residuals are never divided by ||J||. DMD uses the
principal complex logarithm, so temporal aliases are not resolved.

## Window/rank evidence and failures

Combined2--6 Ritz residuals at ranks16/32/48/64/80/96 are
.00102/8.48e-5/6.99e-6/1.11e-6/9.93e-7/1.84e-7. The last rank has a
singular fraction1.15e-11, below the heuristic screen. The four exports use
exactly the same choices as C0:

| Export | Window/stride | Method/rank | Residual | Singular fraction |
| --- | --- | --- | ---: | ---: |
| candidate0 | 4--6 /1 | Ritz /32 | 5.77604e-7 | 3.05138e-9 |
| candidate1 | 4--6 /1 | DMD /32 | 5.75484e-7 | 3.05138e-9 |
| candidate2 | 2--6 /2 | DMD /80 | 3.68636e-7 | 6.01042e-10 |
| candidate3 | 4--6 /2, raw | Ritz /24 | 4.64232e-7 | 2.22691e-9 |

The four independently rebuilt vectors agree within1.50e-7 after phase
alignment. Separate4--6 stride-two/four Ritz32 residuals are
5.98e-7/6.04e-7. The C1 vector's phase-aligned component distance from C0
candidate0 is.16053, so similar rates do not mean an identical vector.

The full0--6 rank96 Ritz/DMD rightmost residuals remain5.00e-5/4.62e-5.
Shell-only rank24 residuals are1.43e-4/9.55e-5 at singular fraction1.87e-10,
and high-rank reductions below the screen do not resolve them. Gauge-only
rank16 reaches4.55e-7 but is close to the screen(1.26e-10). Its rank24 Ritz
rightmost real root2.9357 has residual7.84 and singular fraction1.48e-13:
this is an unaccepted spurious reduced root. The late5--6 rank24 residual
remains about2e-6 at singular fraction1.51e-10; its higher-rank reductions
are noise limited. All of these controls remain in the archive.

## Instantaneous actual RHS and constraint content

The pinned original C1 **free20** server is used with the same single-double
protocol as C0: f gives actual projected centered RHS, d gives the seven
actual native constraints per point. Real/imaginary parts are probed
separately: ten RHS probes over five amplitudes, six constraint probes over
three amplitudes. No timestep is taken. At amplitude1e-4, actual RHS residual
is5.98816e-7 and cached/native Jv difference1.71947e-7. At smaller amplitudes,
roundoff eventually dominates. Constraint amplitude controls agree within
9.57e-10 relative. The diagnostic takes2.501712791s after server startup;
that duration excludes server initialization/cache construction.

For a unit complex free20 vector:

| Quantity | C0 | C1 | C1/C0 |
| --- | ---: | ---: | ---: |
| H RMS | .0190958 | .0204058 | 1.06860 |
| M RMS | .0217401 | .0262319 | 1.20662 |
| Z RMS | .00811944 | .00842522 | 1.03766 |
| Theta RMS | .000510414 | .000625189 | 1.22487 |

Thus both approximate vectors carry constraints. For C1, r>.95 contains
44.79% of raw22 component squared norm,72.86% of M squared norm,78.47% of Z
squared norm, and.4767% of H squared norm. H peaks at r=.83355; M,Z and
component amplitude peak at r=.99865. The C0 H peak was r=.22802. These
pointwise amplitude patterns and norm ratios are descriptive component-space
facts, not an identification of a continuum characteristic branch.

C1 raw22 squared fractions are approximately Lambda58.98%, P21.02%,
A14.34%, lapse3.689%, metric1.101%, chi.739%, Theta.0574%, shift.0803%.
The active-neighbor difference/h roughness is5.57847 per axis versus C0
5.64020. This is not a Fourier wavenumber; no low-frequency/stencil label
follows. Orthogonal real/imaginary-pair overlap at t6 is.9999999986 for
gauge and.999458724 for shell. These are raw component overlaps, not
biorthogonal mode amplitudes or invariant energies.

This comparison concerns the projected continuous J20 only. The already
frozen C0 final-only native RK map test shows that intermediate normal-stage
feedback changes the finite-step map. No C1 finite-step eigenmode or actual
native long-time claim is made here. No continuum eigenvalue, globally
rightmost spectrum, perturbation bound, or nonlinear stability is certified.

The compact freeze contains sources, exact command records, finite JSON,
logs and original catalog copies. Matrices, saved histories, lift, exported
complex vectors and original native binary are external artifacts by hash.
