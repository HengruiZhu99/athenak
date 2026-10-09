# Held targeted coupled-inner principal gate

SOURCE ONLY. No compilation, import, exact-algebra execution, kernel query,
eigensolve or evolution is admitted. Root must review the exact sources first.
The authoritative pencil note is ASSESSMENT-v2.md1b3f607a...; the original
ce8aad transcription error and one-line correction remain preserved.

This proposal has two deliberately separate routes:

1. A general-coefficient gauge principal law with f>0,mu>0,ea1 and
   ec=2mu²/(1+mu)², inserted into the unchanged actual ConformalRHS20
   extraction. It checks the matrix and exact scalar decoupling, including
   q1,mu1,f1 and q=f collisions. It does not validate nonlinear lower-order
   reference-wave-map sources.
2. The full frozen physical-reference wave-map helper plus the displayed
   reference-deviation correction, at constant frozen reference/positiveOmega,
   with f and mu derived directly from alpha,chi,W,G0. This checks that the
   actual added source has the claimed derivative-order coefficients. Its
   constant reference connection is zero and valid by construction. It is
   not an audit of a nonflat layer reference or asymptotic source hierarchy.

probe.cpp is copied from the actual production kernel_symbol.cpp extraction;
all geometry field lifts and tensor/connection derivative-order splits remain
unchanged. Gauge calls alone are replaced, and the complete alpha+beta pole
assembler is used once. The scalar state order remains
(ell,cchi,h,pi,vartheta,A,Lambda,beta), with A_t containing +2Lambda/3.
The old wrong displayed beta term is not propagated into this source.

The fixed future probe grid is216 ordinary candidate cases:
alpha{.05,1,3},chi{.1,1},W{0,.5,1},G0{.375,.75},two orthonormal/oblique-SPD
frames. Eight additional nonharmonic mu1 candidate cases use alpha1,chiG0,
W{0,.5},G0{.375,.75},two frames. Two q=f candidate cases use mu2,W0,
alpha54/29,chi=G0/(2alpha²),G0.375,two frames.36 general-coefficient cases
use mu{1/8,3/8,3/4,1,2,8},f{1,3,q},two frames at alpha=chi1.
Total262 actual20 matrices. No singular puncture endpoint is sampled.

The exact Fraction source uses a separate literal scalar8 matrix, explicit
T/inverse and four-wave target. It will check exact T M=N T and inverse
identities for the same rational general coefficients, plus the q-1 and
positivity polynomial coefficient identities. These fixed exact tests support
the pencil general proof; they are not a numerical all-parameter proof.

If root releases later, require each actual matrix to match its independent
literal20 coefficient matrix at maxabs2e-12, the explicit scalar transformed
wave identity at maxabs1e-10 and relative1e-11, and the complete canceled
left basis rank20 with scaled eigenfield residual1e-10. Repeated-root
nullities are checked against the exact direct-sum multiplicities; numerical
nullity thresholds must be declared before execution. Run Release and ASan/UB
Debug with exact command/source/dependency pins and byte-equal matrix outputs.
All failures remain preserved. No eigenvalue sign, pole, nonlinear closure,
uniform puncture hyperbolicity or evolution acceptance follows from this gate.

A later nonlinear helper gate would separately need actual nonflat reference
identity, collapsed/high-contrast lapse arithmetic, variable-coefficient
linearization, reference connection/4D source convention and all-a outer/core
identity controls. It cannot be inferred from the present constant-reference
principal extraction. No native/BH adoption is admitted.
