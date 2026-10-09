# Independent flat-height identity and endpoint review

The reviewed v2 index is
`cc8a2983de6a2e91f0ff55f4bad6928c0524fe3109e88060be4959407bf48f5e`.
No mathematical correction is needed. Its explicit source/platform correction
correctly identifies the proposal as the existing `FlatPowerReference` p=1
family. The original statement of novelty remains historical in v1; only the
new wide-layer comparison was not included in those earlier negative controls.

For the existing parameter domain `a>=S/2`, `0<r0<r1<S`,
`Omega_out<=1`, `Omega_out'<=0`, so the blended Omega is monotone and positive
inside scri. Setting `d=-r Omega'`, `L=Omega+d`,
`b=sqrt[d(2Omega+d)]` gives `L^2=Omega^2+b^2`. Pulling back physical Minkowski
with `R=r/Omega`, `R'=L/Omega^2`, and `h_R=b/L` gives the conformal metric
`-Omega^2 dt^2-2b dt dr+dr^2+r^2 dOmega_sphere^2`. Thus its Penrose spatial
metric is flat, alpha=L, beta=-b n, chi=1, gtilde=I and Lambda=0. The physical
spatial metric remains `Omega^-2 I`. The scalar/constraint proof assumes the
PDE domain Omega>0; it does not use a continuation outside scri.

The sign convention in the current kernel is
`Kbar_ij=(Lie_beta gamma_ij-d_t gamma_ij)/(2alpha)`. Consequently
`Kbar_rad=-b'/L`, `Kbar_tan=-b/(rL)`, and
`Atilde_ij=D(n_i n_j-delta_ij/3)`, with `D=(-b'+b/r)/L`.
The physical eigenvalues are
`Krad=(-Omega b'+b Omega')/L` and, particularly simply, `Ktan=-b/r`.
The physical trace is `P=Krad-2b/r`. This is identical to the consumed
general-b formula in the prior header, rather than the original rw/a shortcut.
The independent symbolic script verifies the physical Hamiltonian and radial
momentum constraints exactly using `b b'=L L'-Omega Omega'`. The flat metric
and stationary ADM identities explain the exact geometric fixed point, but
this short review is not an independent tensor-kernel gate.

Root's factor `d=w*r*((1-w)*g'*(1-Omega_out)+r/(aS))` exactly equals the
existing p=1 inner log-boost factor with `w=e/(1+e)`, `e=exp(g)`.
At r0, `b` is proportional to `exp(-1/(2s))/s` times a smooth positive
factor. Every derivative is flat there even though the polynomial factors
diverge. At r1, b approaches the positive value r1/a, so a smooth square
root of its positive argument supplies C-infinity CMC matching. Exact
core and outer branches avoid radial divisions at the origin and preserve
the old outer formula, jets, speeds, and local certificates.

Floating-point evaluation must retain representable b derivatives after w
or b itself underflows. The prior header's separate evaluations
`exp(logb)`, `sign(c1)*exp(logb+log(abs(c1)))`, and
`sign(c2)*exp(logb+log(abs(c2)))` supply b,b',b'' without sqrt of a
rounded-zero w. Apple arm64's recorded long-double exponent/mantissa macros
give no extra range over double, so the v2 correction is appropriate.
This is not a uniform numerical-range guarantee for arbitrarily extreme
parameter scales. The stored 80-digit sample maxima are mathematical profile
measurements, not a native or stability result.

No existing source, frozen archive, or evolution was modified. The already
adverse broad/gentle p1 and cutoff-power controls remain valid prior evidence;
the new wide-layer instantaneous comparison is a separate limited test.
