# Finite-Omega C1 actual-kernel stiffness gate (scratch only)

This gate permits a bounded, strict-interior exploratory native preflight. It
does not accept C1 as a stabilization, a regular scri system, a global evolution
estimate, or a nonlinear constraint-manifold closure. No tracked runtime file,
live gauge, boundary plan, derivative stencil, projection, damping normalization,
Omega floor or physical-Theta falloff is changed.

The compiled production identity is implementation `27c19d20`; this fresh run's
launch HEAD is `aef47b0ab6484546887fbeef72eaa7449fff064d`. Its four compile/run/check
commands pass and all 370 recorded source inputs are unchanged (365 tracked
src/CMake files and five audit/helper inputs). The C1 math
header is the previously frozen `908d655ad8a43261d8e0cd66b3c0da485fc131b57a5203019aea01acc21771b7`.
The nine-command nonlinear tensor/principal gate remains separate and frozen.

## Actual full20 generators

`full20.cpp` uses exact dual-number differentiation of ConformalRHS plus actual
physical-P InteriorLayerGauge. The independent20 variables are alpha, chi,
physical P, physical Theta, beta(3), determinant-one metric(5), trace-free A(5),
and Lambda(3). Algebraic constraints and their spatial derivative jets are
reconstructed. Cosine/sine perturbation jets give the complex continuum Fourier
matrix without a finite perturbation step. Runtime kappa1=kappa_input/alpha is
varied with the live lapse; kappa2=0. The rho1.5 norm gauge, where selected, is
identical to the frozen failed-pulse gauge; its shift pole is explicitly
assembled. Preferred-source projection remains off.

Form0 is unchanged C0; form1 adds the mechanical Appendix-B C1 terms; form2 also
adds the independently derived covector connection repair
`Delta Lambda^i=-2 Ztilde^j d_j beta^i`. The C1 helper assembles separate regular,
simple-pole and double-pole arrays. It adds its live value without reference-RHS
subtraction. The exact Einstein sector makes the addition zero.

There are 1920 Fourier matrices: a=.5, kappa5/10, both gauges, all three forms,
reference and an unrestricted .01 deterministic full-jet perturbation, eight
radii .75 through the N48 outermost radius, k=0/32/64/128/256, radial/oblique.
The off-constraint perturbation has finite physical Theta, spatial Z and spatial
jets, with no asymptotic weights. Another 384 zero-frequency matrices use
a=.5/.75/1/2 and Omega=1e-2,1e-3,1e-4,1e-5. These are local continuum generators,
not the native stencil/ghost operator or a coefficient-aware subsidiary system.
The highest continuum frequencies exceed finite-grid Nyquist and are retained
as diagnostic samples, not resolved native modes.

Raw analytic-reference full RHS cancellation is at most 2.32e-13 in the Fourier
set and 1.09e-10 in the smaller-Omega set. Native C0 already subtracts its own
analytic reference roundoff; any private C1 integration must add the C1 delta
after that existing subtraction, without subtracting a C1 reference residual.

## A double-pole entry is not an Omega-squared eigenvalue timestep

At reference scri on the positive x axis, S=1,
`Omega^2 d(Lambda_x,t)/dTheta -> -2/a^2`; the measured coefficient is
-7.9999599999 for a=.5,Omega=1e-5. This is a genuine double pole for arbitrary
finite physical Theta. It must not be hidden by dividing a simple-pole numerator.

Define, for analysis only,

```
T = diag(1 on alpha/chi/P/Theta/beta/gtilde, Omega on Atilde/Lambda),
B = Omega T L T^-1.
```

Both momentum groups matter: scaling Lambda alone leaves the simple-pole
Atilde<-Lambda block unbalanced. With the stated T, all sampled zero-frequency
B matrices remain bounded: maximum 2-norm54.37280039, maximum spectral radius
23.00039859, over all radii, kappas, forms, gauges and reference/off-constraint
jets. In the repaired a.5,kappa10,norm-gauge reference sequence,
Omega max|lambda| tends20.3161, while ||B|| tends54.3728. The corresponding
unrestricted perturbed sequence also stays bounded. This is sampled
O(1/Omega) eigenvalue stiffness, rather than an inferred O(1/Omega^2) eigenvalue
stiffness. T is not imposed on runtime data and is not a uniformly equivalent
unweighted norm as Omega tends zero.

Small positive primitive frozen roots are retained. For a.5,kappa10 at k0 the
mechanical/repaired reference roots tend approximately .63549/.63550 in the
production gauge, or1.22440/1.22441 in the norm gauge. The repaired norm-gauge
Fourier sample has max Re(lambda)=2.80932473 at the N48 outer radius,k128,
oblique. The earlier C0 norm-gauge transition k0 root also remains positive.
No all-roots-negative claim is made, and these primitive roots are not identified
as physical subsidiary eigenmodes by their frozen constraint residues.

Kappa5 must be distinguished: at a=.5 the repaired norm-gauge k0 sequence has
`Omega Re(lambda_max) -> .22045277` on the reference and .10493749 on the .01
off-constraint background. This is a positive leading frozen pole, not a
bounded slow root. Consequently this report does not support a kappa5 native
candidate. The authorized limited preflight retains kappa10 throughout.

## Exact propagators and actual timestep checks

The actual native pole.03 steps are .000427734375 (N24),
.000115104166666667 (N36), and .00009755859375 (N48), using their actual strict-
interior Omega minima .0142578125/.00383680555555556/.003251953125. At each step,
only sampled points within that grid's outer radius are included. Across 5040
local matrix/step cases, every sampled Re(lambda)<=0 eigenvalue satisfies
`|1+dt lambda+(dt lambda)^2/2+(dt lambda)^3/6|<=1`; maximum excess is zero.
Positive roots have the corresponding small exact growth and are not discarded.
This is a local scalar RK stability check, not proof that the native full22
final-projection RK step or the boundary generator is stable.

Exact matrix exponentials are computed independently of eigendecomposition by
32-term Taylor scaling/squaring, using unoptimized einsum; the scaled infinity
norm is <=.5. Its truncation is below1e-44 before floating roundoff. Step-halving
composition is checked on all 810 reported propagators. Both physical and
analytical weighted component 2-norms are retained; neither is claimed to be a
symmetrizer or geometric energy.

At k0, reference a.5,kappa10,norm gauge:

| Grid | C0 raw RK norm | repaired C1 raw RK norm | repaired C1 exact norm | repaired C1 weighted RK norm |
| --- | ---: | ---: | ---: | ---: |
| N24 | 1.52697 | 12.95109 | 12.99668 | 1.51830 |
| N36 | 1.53239 | 48.30424 | 48.48761 | 1.52352 |
| N48 | 1.53270 | 57.01233 | 57.22983 | 1.52382 |

At Omega1e-5, the repaired k0 raw RK norm is18583.22699 and raw exact norm has
the same inverse-Omega growth. Omega times the raw RK norm tends .18583227;
the weighted RK/exact norms are1.52545139/1.51758652. Thus the large amplification
is present in the exact local semigroup, not an RK eigenvalue instability. A
bounded physical-Theta error can immediately produce a large stored Lambda
error; reducing dt does not remove that exact continuum transient over a fixed
time. There is no uniform raw-norm stability claim.

Weighted one-step RK versus exact matrix relative error at native k0 is about
1.52--1.54%, and reaches2.08% in diagnostic k256 samples. The same C0 values are
about1.49--1.50% and2.04%. This fast-mode temporal error is retained explicitly;
pole.03 is not an accuracy guarantee. The finite-Omega preflight is justified
only as an exploratory comparison at a fixed mesh and step. A successful long
pulse would still require actual timestep and resolution controls.

## Next finite-Omega experiment and limits

The separate native/global builders may now test the unchanged ng3 norm-gauge
baseline plus explicit repaired C1 delta: reference fixed point, actual RHS/Jv,
actual final-only RK step and short finite pulse. Reference subtraction must
retain the semantics stated above. No long pulse is authorized by this report
alone; the parent reviews those actual gates. Neither arbitrary Theta falloff
nor a special undamped vector-wave polarization is justified for this damped
constraint system. Finite-Q and nonlinear regularity counterexamples remain.

`receipt.json` records exact commands/compiler/source hashes, `check-report.json`
contains all numerical rows, and the immutable index hashes the large matrices
and binary without committing them. The native/source and covariance/principal
identity gates remain separate from any future propagation receipt.
