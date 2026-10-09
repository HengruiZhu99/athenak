# Saved-state discrete mode diagnostic

This read-only diagnostic identifies a reproducible approximate pair of the
original C0 spatial-norm **projected continuous** N16 generator,
`lambda ~= 1.916339 +/- 6.790995 i`. The exported vectors have actual cached
generator residuals `3.77e-7` to `5.89e-7` in generator units and `5.34e-8` to
`8.34e-8` relative to `max(1,|lambda|)`. These are finite-grid approximate or
pseudospectral vectors. No eigenvalue perturbation bound, globally rightmost
spectrum, continuum eigenvalue, or stability conclusion is established.

The calculation uses only existing matrices and saved histories. It does not
perform sparse LU, new propagation, native evolution, or source modification.
The later actual nonlinear RHS/constraint oracle makes centered finite
differences of this vector at one instant; it also does not evolve it.

## Inputs and method

The original forward operator is `dq/dt=J@q`, with shape 32800 by 32800,
15,109,351 nonzeros, and 1640 active points. Coordinates are free20 point-major
in original active-cell k/j/i order. The original global grid has N16 and span
2.2; this is distinct from the span-2.1 native evolution grids. The unit seed
histories contain 241 times from 0 through 6 at spacing .025 and two seeds,
`gauge_pulse` and `shell_random`. Original index and large artifact hashes are
pinned in the final verification receipt.

Nine searches retain all reduced roots and residuals. Their windows are
0--6, 2--6, 4--6, and 5--6, with separate gauge, shell, combined, raw-column,
and stride-one/two/four controls. The SVD rank ladder is 4, 8, 12, 16, 24, 32,
48, 64, 80, 96, capped by the available rank and singular-value fraction
`>1e-14`. Usually X columns are scaled to unit Euclidean norm and the paired
Y columns receive the same scale; one raw-column control is retained.

Rayleigh--Ritz projects the actual J onto the SVD subspace. Projected DMD uses
paired states and the principal complex logarithm; temporal aliases are not
resolved. Every reported residual uses the actual J action, with selected
vectors independently checked by a fresh sparse matrix-vector product. It is
never divided by the very large norm of J. The `1e-10` singular-value fraction
screen is a conservative heuristic, not a rigorous bound on history error.
Raw component units are not a physical or proved energy norm.

## Convergence and retained negative controls

For the combined 2--6 window, Ritz residual falls from .00254 at rank 12 to
8.41e-5 at rank 32, 5.11e-6 at 48, 1.10e-6 at 64, and 8.34e-7 at 80.
The rank-96 residual is 3.11e-7 but its singular fraction is 1.28e-11, below
the heuristic screen. Four separately reconstructed exports pass the screen:

| Export | Window/stride | Method/rank | Residual | Singular fraction |
| --- | --- | --- | ---: | ---: |
| candidate0 | 4--6 / 1 | Ritz / 32 | 5.88677e-7 | 3.21853e-9 |
| candidate1 | 4--6 / 1 | DMD / 32 | 5.86733e-7 | 3.21853e-9 |
| candidate2 | 2--6 / 2 | DMD / 80 | 3.76907e-7 | 5.91452e-10 |
| candidate3 | 4--6 / 2, raw | Ritz / 24 | 5.04370e-7 | 2.23296e-9 |

Their phase-aligned vector distances are at most 1.67e-7. Other 4--6
stride-two/four Ritz rank-32 residuals are 6.03e-7 and 6.06e-7.

The complete 0--6 subspace still has rank-96 rightmost residual about 7.34e-5
(DMD 5.70e-5). Gauge-only high-rank reduction creates an unaccepted real root
near 3.0177 with residual 13.37. The shell-only search has residual about
2.1e-4 at a noise-screened rank 24; its best high-rank residual remains about
8.3e-5 below the screen. The 5--6 window gets sub-1e-6 residual only below
the screen. These are retained failures of reliable mode identification,
not additional accepted modes.

NumPy emitted divide/overflow/invalid floating-status warnings at dense
matmul calls during the original nine searches and first export. Their
finite JSON results, logs, source and warning text remain archived. The final
export does not suppress warnings: explicit `einsum(optimize=False)`
contractions, finite assertions, and agreement against the original reports
independently reproduce the four exports with a warning-free log. The host
warning mechanism is not established by this diagnostic. N20 original
search stdout logs do not capture a full stderr stream; the final verifier
independently rebuilds the selected N20 subspaces with explicit contractions.

## Actual RHS and constraint content

The original free20 native server tests real and imaginary parts separately.
Five centered-RHS amplitudes give ten probes; three constraint amplitudes
give six probes. At amplitude 1e-4, the actual nonlinear centered RHS has
generator residual 5.92383e-7 and differs from cached Jv by 6.74736e-8.
Smaller differences eventually become roundoff limited. Constraint amplitude
controls agree to about 1.12e-9 relative.

For the unit complex free20 candidate the native H/M/Z RMS amplitudes are
.0190958/.0217401/.00811944 and Theta RMS is .000510414. It is therefore
constraint carrying rather than a discrete constraint-free gauge direction.
Raw22 component squared fractions are approximately Lambda 61.6%, P 19.7%,
A 13.5%, lapse 3.44%, metric 1.03%, chi .674%, shift .0874%, Theta .0388%.
At r>.95 lie 46.7% of component squared norm, 66.7% of momentum squared
norm, and 85.6% of Z squared norm, but only .640% of H squared norm. H peaks
at r=.2280, M at .9600, and Z/component amplitude at .99865.

An active-neighbor difference/h norm is about 5.6402 per coordinate axis;
pi/h is 22.8479. This roughness diagnostic is not a Fourier wavenumber and
cannot classify the candidate as a low-frequency or stencil branch. The
data do not distinguish a continuum subsidiary mode, gauge-originated
constraint leakage, or a discrete stencil contribution.

Orthogonal projection onto the real/imaginary pair accounts for
.9999999987 of the gauge seed squared component norm at t6 and .9990436341
of the shell seed. These are component-space overlaps, not biorthogonal
mode amplitudes or invariant energy fractions.

## N20 control and finite-step distinction

The separately pinned N20/span2.2 gauge-only saved history gives a nearby
pair `1.9182803 +/- 6.8055088 i`. The 2--6 rank-32 Ritz residual is
3.82388e-7 at singular fraction 1.41009e-9. The 4--6 stride-one/two rank-16
residuals are 3.69718e-7/4.14048e-7, with singular fractions
1.42453e-10/1.23835e-10. Higher-rank results below the screen are retained.
The final verifier independently reconstructs these three selected vectors.
Two grids with different layer sampling do not establish spatial order or
a continuum eigenvalue.

Boundary's **separate** frozen action test is
`boundary/full-tensor-mode-finite-step-20261009/immutable-mode-final-step-20261009`,
index `3e212ec5e437c0190f687fcd3b95d52671fe814b36c9e62469924a4a01e241b0`.
The exact native final-only map is `P_ref R3(dt J22) Lift`, not RK3 of J20.
For candidate0 at nominal dt=8.0859375e-5 and half/quarter steps, its
directional Rayleigh growth is 1.92909/1.92314/1.91985, while state residual
per unit time is .04419/.02389/.01241. The projected20 RK3 residual per unit
time remains about 5.887e-7. Thus the exported continuous-generator vector
is not an eigenvector of the finite-step native map; intermediate normal
feedback matters. All four vectors agree, and an actual nonlinear one-step
finite-difference oracle confirms the cached map action within 1.80e-11.
This is an action test, not a certified finite-step eigenmode or stability
result. The separate bundle preserves its own sources and receipts.

All inputs and exports are hash pinned. Binary arrays, matrices, lift and
native executable are metadata-only external artifacts in this compact
freeze. No source or frozen input was modified.
