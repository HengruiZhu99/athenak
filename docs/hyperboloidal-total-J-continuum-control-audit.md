# Actual continuum total-J tensor control

The actual linearized C0 physical-P/spatial-norm continuum kernel closes
within the tested J=0,1,2 angular sectors. Independent angles, m components
and Cartesian rotations agree to 2.15e-13 after a well-conditioned fit.
This supplies a full-tensor continuum control for radial work; it establishes
no radial solver, boundary stability or finite-pulse result.

The baseline is S=1,a=.5, geometry(.05,.95), gauge(.45,.85), physical-trace
lapse enabled, preferred source disabled, xi=2,eta=6,C=(1-1/1.5)/.5,
kappa_input10 and kappa2=0. No Q/null feedback or C1 change is included.
Production remains `27c19d20696ea6dd4704032c51dfd026218f64f2`.

## Regular Cartesian representation

Normalized solid spherical harmonics couple to constant Cartesian spin0,
spin1 and STF spin2 tensors through Clebsch-Gordan coefficients. The phase
i^(L+s-J) makes m0 real, with exact negative-m conjugacy. Each component is
a regular Cartesian polynomial r^L Y_Lm times W_L(rho),rho=r^2. Value,
gradient and Hessian evaluation never divides by r.

Scalar sectors are alpha,physical P,physical Theta and the trace of the
Penrose spatial metric perturbation. Beta/Lambda supply vector sectors;
the remaining metric and independent A data supply STF sectors. J0/J1/J2
have8/16/20 radial amplitudes. Physical metric h=delta bar-gamma converts as

```
delta chi=-(chi_ref/3)*(bar-gamma_ref^-1:h),
delta gtilde=chi_ref*h+bar-gamma_ref*delta chi,
delta A=TF_gref(S)+(gref/3)*(Aref^ij*delta gtilde_ij).
```

All consumed spatial reference-coefficient derivatives are retained. The
last A term is required for nonzero background A. Lambda stays independent;
there is no extra chi prescribed alongside all six metric components.

The independently reviewed math-only gate checks69 all-m records with exact
rotation,norm,parity,conjugacy and homogeneity identities. Exact stacked
ranks give8/16/20. Compiled polynomial/origin and nonconstant anisotropic
metric/A conversion errors are9.50e-16/9.30e-16 scaled. Its Release and
ASan-UBSan outputs are byte identical. These precede the actual PDE gate.

## Actual binding and independent local checks

The private native spatial-norm wrapper dispatches that gauge only for
double; dual scalars silently take its production-gauge fallback. The new
bridge uses the frozen generic-dual C0/spatial-norm formula and checks it
against the actual double wrapper. The omitted-feedback negative control
differs by31.57894. The beta pole is added exactly once, and fixed constant10
analytic reference subtraction matches the native Cartesian patch.

| Actual local check | Observed maximum |
| --- | ---: |
| Reference source-subtracted RHS / physical constraints | 7.11e-15 / 3.08e-14 |
| Double/dual response, best amplitude / all amplitudes | 1.20e-10 / 3.87e-8 scaled |
| Raw22 input / output algebraic normals | 9.76e-17 / 1.56e-15 scaled |
| Metric first/second and A first tangent identities | 1.90e-15 scaled |
| Native point-projector derivative / coordinate-zz chart | 1.94e-11 / 1.12e-16 scaled |
| Reference coefficient-map jet FD, finest step | 7.14e-9 scaled |
| Independent solid-harmonic Laplacian | 2.25e-15 scaled |
| Core TT hxy=z²,Axy=z RHS/constraint oracle | exactly zero error |
| Pure-gauge initial constraints | exactly zero |
| Coefficient-aware pure-gauge Cdot, finest step | 2.15e-7 scaled |

The double/dual check covers200 full22 responses at five amplitudes,20 free
columns,reference/finite constrained SPD states and five radii. The Cdot
check differentiates the complete actual RHS before applying physical
H/M/Z/Theta. At r=.98 its largest lapse/shift H residuals reach
3.34e-5/3.96e-5 absolute after approximately fourth-order decrease over
.001/.0005/.00025. Other components reach cancellation floors. This is not
a frozen-symbol comparison or a uniform order claim past those floors.

A separate independent Cartesian flat-core formula checks all22 RHS rows
and eight physical diagnostics in1404 cases per build,including the origin,
all channels,nonnegative m/real-imaginary parts and three radial jets.
Errors are1.67e-16/1.29e-16 scaled. Negative m follows exact conjugacy rather
than separate kernel queries. The TT oracle gives hdot_xy=-2z,Adot_xy=-1.

Actual bridge Release/ASan-UBSan both pass declared gates,in .49s/6.71s.
Their full JSON differs in48 FD cancellation entries,at most4.32e-10; this
is distinct from basis-gate byte equality. An initial bookkeeping assertion
requiring equality and its prior report are preserved. No science was rerun
to correct that statement. Executable hashes are
`8e9ae32418150c4350cde7d8beca78f014aea74e1c83e046427feef157371562`
and `9925dff649ba5795b1d0b087734215dc1e78d1d617ded537b591ac27f3f61bbe`.
Build receipts capture1055/1057 dependencies plus four link archives. Root
readback verifies1062 unique build/link paths across the four accepted
bridge and independent-oracle builds. Point-projector checks invoke no
mesh ghosts or finite-RK step; output normals apply to this stationary
source-subtracted reference.

## Angular coefficient action

At13 positive radii from .025 to sqrt(.9973046875),with minimum
Omega=.0026953125,each channel receives independent envelope jets
(W,W_rho,W_rhorho)=(1,0,0),(0,1,0),(0,0,1). The matrices describe

```
Wdot=B0(r)*W+B1(r)*W_rho+B2(r)*W_rhorho.
```

J2 has60 input actions and three20x20 matrices. These are rho derivatives
of W; derivatives of the solid-harmonic r^L factor are already included.

Twelve oblique fit directions are stacked in raw22 Cartesian space and
solved with column-scaled SVD. Eight unused directions,independent m1/m2
and paired rotations validate the same matrices. All39 fits have full rank.
Maximum scaled condition is3.43164; raw condition3.16742e6 reflects explicit
r^L column magnitudes. No singular values are dropped. Worst scaled errors
are1.41e-13 fit,2.03e-13 unused angles,1.87e-13 m and2.15e-13 rotations.
Raw input/output normals stay below2.52e-16/1.03e-13. The96720 actual
evaluations take2.697s.

The first NumPy matmul analysis emitted floating-status warnings despite
finite output. Its exact source/report/coefficients remain saved. Accepted
einsum reanalysis promotes RuntimeWarnings to errors,has empty stderr and
returns identical matrices from the same pinned kernel batch. Tolerances
were unchanged and no PDE evaluation was rerun for that reanalysis.

## Limits

At r=0 the value fit loses rank. Origin regularity is checked through
polynomial/full-kernel oracles; positive-radius fits supply no origin
evolution equation. Nonlinear products can mix J sectors,so this is a
linearized angular closure result. Cartesian finite-h Dxx,DxDx,Lx and KO
anisotropy is absent. A later radial discretization changes the bulk operator
too and cannot uniquely attribute prior growth to primitive ghosts.

No radial operator,eigenvalue,propagation,exact-scri closure or stable finite
pulse is accepted. The later black-hole target remains the inner
wormhole-to-trumpet transition with a Minkowski hyperboloidal reference.

The [frozen evidence archive](validation/hyperboloidal-total-J-continuum-control-experiments-20261009/README.md)
contains160 cataloged blobs,4,678,897 bytes and59 finite JSON files. Its catalog
SHA256 is `1d20e5ab3590d3b34097f190c30bc397ae63365deeb398dca6d3350d808dff01`.
Original basis/actual-angular index SHA256 values are
`414f241e986e0c46b7d94820fb060b704166d973284854c53d2e8c64ee6c489e` and
`b4131aa02f093b7744d3de1b600e829b7213a59779672978bd84c71f6912c513`.
Read-only archive verification passes without scratch dependencies or
scientific reruns. Root and independent reviews are retained. All NumPy binary
arrays, executables and payloads larger than1MiB remain metadata only.
