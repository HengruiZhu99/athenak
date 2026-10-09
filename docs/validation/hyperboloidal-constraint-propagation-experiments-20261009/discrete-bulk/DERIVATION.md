# Exploratory bulk composed-derivative gate

This is an ignored, read-only audit of the actual `ConformalRHS`,
`InteriorLayerGauge`, `EvolvedConstraints`, and native finite-difference/KO
functions. It changes no production or private native evolution source.
The constant background has positive alpha/chi, det-one SPD conformal metric,
Omega=1 with zero derivatives, zero curvature and connection, and constant
shift. Gauge restoring rates and physical-P pole damping are zero in this
principal-only test; physical-P slicing and the actual coupled shift remain.

## Exact stencil and flat discrete constraint defect

The actual fourth-order centered first derivative is

    D = [1,-8,0,8,-1]/(12h), offsets -2..2.

Its composition is

    D² = [1,-16,64,16,-130,16,64,-16,1]/(144h²), offsets -4..4.

It is a fourth-order approximation with leading error `-h^4 f^(6)/15`.
It requires radius-four coverage; increasing the halo must not select the
native sixth-order `Dx<4>` or `Lx<4>`. Fourth-order `Dx<3>`, radius-three
upwind/KO and mixed `Dx_i Dx_j` stay as audited here.

Write `S_ii=Dxx4_i`, `S_ij=D_i D_j` for i!=j and
`delta_i=S_ii-D_i²`. For a constant flat Cauchy background alpha=chi=Omega=1,
the actual kernel and constraint map give the following exact initial gauge
columns of `C_h L_h` (all omitted constraint components vanish):

    lapse a: M_i,t = D_i sum_(j!=i) delta_j a.
    shift b: H_t = -2 sum_i D_i sum_(j!=i) delta_j b_i.
             Z_i,t = [sum_j delta_j/2 + delta_i/6] b_i.

The last `delta_i/6` is required: the actual Lambda equation uses the diagonal
second derivative in its one-third gradient-divergence term too. The first
exploratory check omitted this term, failed, and is preserved separately.
Replacing every diagonal Hessian by D_i², in both RHS and the diagnostic
constraint map, makes all these gauge columns vanish to roundoff. A RHS-only
change with the original diagnostic is a distinct attribution/acceptance test
and must not be described as exact discrete Bianchi closure.

For theta=k h and t=sin²(theta/2),

    D(theta) = i sin(theta)(4-cos(theta))/(3h) = i p,
    delta(theta) = -16 t³(2+t)/(9h²) = -k^6 h^4/18 + O(h^6).

Thus the existing inconsistency is fourth-order for resolved waves, can be
large near the grid scale, and is hidden by some one-dimensional lapse/H
tests. KO/upwind do not generate these initial gauge columns: the static
constraint tangent contains no lapse or shift. Common scalar operators also
commute with the full constant-coefficient constraint map.

## Complete actual-kernel principal basis for p != 0

The C++ probe directly calls `Dx<3>`, `Dxx<3>`, `Dxy<3>`, a nested `Dx<3>`,
`Lx<3>`, and `InteriorKOSixth` on real sine/cosine fields. The resulting jets
seed the full actual 20-field kernel and constraint map through exact dual
derivatives. Algebraic det/trace conditions are imposed before extraction.
There is no finite perturbation epsilon.

For the composed scheme all Hessian symbols are `-p_i p_j`. Let
`s²=chi gtilde^ij p_i p_j` and orient an orthonormal coframe along p. In the
standard derivative-based 20-field reduction the extracted generator is

    L_red = i alpha s M(W,alpha) + i beta.p I.

The pre-existing exact complete characteristic basis of M is checked against
this independent full-kernel extraction, including radial/oblique directions,
det-one non-diagonal SPD metrics, alpha=.2/1/3, chi=.4/1/2 and W=0/transition/1.
Its normalized condition is finite through W=1. This establishes the bulk
principal relation for every nonzero modified covector, not a uniform basis
in the primitive variables at p=0.

Interior uniform KO has symbol `q=-sum_i sin^6(theta_i/2)/h_i`. The actual
upwind advection has, in each direction,

    Re Lx_i = -8 |beta_i| sin^6(theta_i/2)/(3h_i),
    Im Lx_i = beta_i sin(theta_i)
              [cos²(theta_i)-3cos(theta_i)+5]/(3h_i).

Replacing centered advection and adding common epsilon*KO therefore gives
`lambda_j=i alpha s v_j + sum_i Lx_i + epsilon q`; the same complete basis
works. This covers the constant-coefficient bulk common operator only.
The native active-line `-D3^T D3` KO and ghost closure near the boundary are
not common Fourier scalars and require the separate global audit.

## Exact Nyquist Jordan block and a uniform-h bound

If every theta_i is 0 or pi, p=0 and composed Hessians vanish. In primitive
variables the remaining principal links P->alpha/chi, Theta->chi,
A->metric, and Lambda->beta form a nilpotent matrix N with N²=0. The true
zero spatial mode has no KO damping and can have ordinary polynomial
zero-frequency gauge/curvature evolution. For a nonzero Nyquist mode, with
m Nyquist axes on an isotropic grid, gamma>=epsilon*m/h and

    exp[t(N-gamma I)] = exp(-gamma t)(I+t N).

Use a D+ norm weighting the position variables alpha,chi,metric,beta by
`zeta=sqrt(1+4m/h²)` and leaving P,Theta,A,Lambda unweighted. N maps only
momenta into positions, so `||N||_Dplus=zeta ||N||`. Therefore, for h<=hmax,

    ||exp[t(N-gamma I)]||_Dplus
      <= 1 + ||N|| sqrt(hmax²+4m)/(e epsilon m).

This is a uniform-h finite propagator bound for the exact nonzero Nyquist
Jordan blocks. It can have substantial transient amplification; it is not
an energy contraction or a claim of a uniform Nyquist diagonalizer. The
script also evaluates exact propagators at four h values and nine scaled
times. Near-Nyquist, variable-coefficient and full boundary estimates are
not established by this exact-Nyquist bound.

## Limits and literature

Analytic reference gradients and nonlinear/discrete product rules are absent
from this flat frozen calculation. Nonflat coefficient-aware and global
native gauges may still produce constraint error. No nonlinear Bianchi,
scri regularity, stability, or black-hole evolution claim follows.

The prescription in [arXiv:1111.2177](https://arxiv.org/pdf/1111.2177) is
different: it uses scalar Hessians D1_i D1_j plus an isotropic
`(Delta2-sum D1²)/3` correction and repeated D1 for gradient-divergence.
The discussion following Eq.46 explicitly notes that its semidiscrete
constraint subsystem does not close; a D0²-type closure has too weak a norm
for that paper's estimates. Its stability result cannot be invoked for this
blanket composed-Hessian experiment. The direct gate here supplies only the
stated bulk results, alongside a separate KO/Jordan bound.
