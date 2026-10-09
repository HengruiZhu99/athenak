# Negative tensor-discretization transfer audit

The hypothetical arXiv:1111.2177 discretization does not retain a complete
principal basis with the current physical-P coupled transition gauge. This is
a full20 actual-kernel matched counterexample, not a native evolution result.
No production or private native source was changed.

The paper retains the raw Laplacian and substitutes repeated first derivatives
for scalar tracefree Hessians and vector gradient-divergence (Eqs27–29). Its
constraint subsystem is explicitly not closed after Eq46. For the actual
constrained flat conformal variables this means only:

    delta A_ij = -chi [(D_iD_j-S_ij) delta alpha]^TF
                  + alpha/2 [(D_iD_j-S_ij) delta chi]^TF,
    delta Lambda^i = gtilde^ij (D_jD_k-S_jk) delta beta^k/3.

Every metric, trace and Theta Laplacian is retained. These are linear,
constant-background principal corrections, not a proposed nonlinear extension.
The probe applies them inside an exact dual call to the actual geometric/gauge
kernel, then transforms the full20 output to the physical orthonormal reduction.

Put nu²=p²/ell², where i p is the centered first-derivative symbol and -ell²
is the standard Laplacian symbol. The scalar characteristic polynomial is

    (lambda²-f)(lambda²-1)/3 *
    [3lambda^4 - lambda²(mu nu²+3mu-nu² W-nu²+4)
      +4mu-nu² W].

For W=9/10, alpha=1, f=6/5, mu=15/16 and nu²=7/17 it becomes

    (lambda²-1)(5lambda²-6)²(408lambda²-383)/10200.

Exact SymPy rank checks give nullity2 for M²-(6/5)I and nullity4 for its square.
Both repeated roots +/-sqrt(6/5) are defective. The actual fourth-order native
symbols attain the ratio at theta=2.15246627117 on a radial grid mode and the
actual smooth gauge cutoff attains W=.9. The check receipt gives authoritative
coordinates and full20 matrix errors. A common KO scalar shifts the defective
block but does not create a complete basis. A separate dissipative energy
argument could in principle bound transient growth; this audit supplies none.

The existing static geometric constraint map also has a nonzero lapse source:

    M_i,t = (2/3) D_i sum_j(S_jj-D_j²) delta alpha,
    Z_i,t = sum_j(S_jj-D_j²) delta beta_i/2.

In particular a one-dimensional lapse perturbation now sources M, unlike the
corresponding native standard-Hessian flat column. The full20 probe verifies
these identities as well. This transfer is rejected as a basis/closure solution;
there is no authorization or mathematical reason to run it natively.

An exact fourth-order repeated Dx4 has outer coefficient1/144 at offset4.
It cannot be implemented with only radius3 coverage while keeping the actual
centered fourth-order first derivative. A different compact or staggered
factorization would change the derivative/variable representation and needs
its own full tensor energy analysis.

Primary source: https://arxiv.org/pdf/1111.2177, Eqs27–29 and discussion after46.
The paper's constant-coefficient proof is parameter dependent and does not
cover the current coupled transition gauge, nonflat hyperboloidal background,
or native boundary/active-line KO operators.
