Two positive-real-part eigenvalues are certified for each declared rounded
J0 finite matrix at N8,N12,N16. This is an exact finite-matrix statement;
it does not establish continuum eigenvalues or a physical subsidiary instability.
No eigensolve, operator assembly, propagation or native evolution was called.

| N | dimension | outward upper nu | outward upper epsilon | certified positive | certified negative | outward lower Re of each positive root |
|---:|---:|---:|---:|---:|---:|---:|
|8|64|9.328536670097686e-13|1.3967996651666166e-10|2|62|2.0493330858359444|
|12|96|4.634094818433476e-11|1.817947875447094e-9|2|94|1.8456513955764864|
|16|128|1.183056277494174e-10|7.032380143598307e-10|2|126|1.9128322211424449|

The displayed endpoints enclose exact rational bounds stored in certificate.json;
the sign and cluster tests use those exact rationals, not the displayed decimals.
Each positive root is counted in an isolated singleton disc. All other discs
are strictly in the left half-plane. The exact target is the elementwise
round-to-nearest/ties-even binary64 sum of the saved Jbulk and Jsat entries,
subsequently treated as an exact matrix. Direct binary64 sum binding was also
checked. No error claim is made for a different unrounded quadrature operator.

The full saved energy eigenvectors only propose V=L^-T Q; an ordinary complex
Gauss-Jordan calculation proposes W. These floating proposals are not trusted
as inverse/eigenvector identities. Exact integer dyadic products form N=WV and
R=WJV-ND. The exact bounds nu>=||I-N||inf<1 and
epsilon=||R||inf/(1-nu) prove the similarity enclosure. The conservative
complex modulus bound is |Re z|+|Im z|. Closed touching discs are merged, with
strict separation from outside discs; the D+s N^-1 R homotopy proves the
cluster counts with algebraic multiplicity. No diagonalizability or forward
error bound for individual computed eigenpairs is assumed.

The released synthetic suite passed13 tests in.005s, including independent
Fraction checks of integer complex products, outward normal/subnormal rounding
and256 fixed-seed rational interval checks; positive/negative, repeated,
overlapping, touching, wrong-center, defective-Jordan, nu=1 and extreme-scale
cases; and a dense non-unitary complex exact similarity with zero residual.
Actual runs took.303,.889,1.962s. Every run has empty stderr, unchanged sources
and an exact admission. Typed N12/N16 hashes matched the original authoritative
receipts before computation. There were no failed or tuned attempts in this
certificate stage.

Exact J,V,W,lambda hex-pair inputs are retained as large_payload. Sources,
PLAN, synthetic release/receipt/log, per-N admissions, rational certificates,
runtime logs and input context are frozen. External NPZ files are recorded by
path/hash/size and are not duplicated into this capsule. The full verifier can
replay the exact encoded certificate without NumPy; metadata-only verification
explicitly skips proof replay if the large hex input is absent.

The original failed SciPy exponential, both failed ordinary-FD projection gates
and unresolved generic nongauge continuum comparator remain unchanged and
linked. These certificates do not identify why the finite discretization grows,
prove a constraint propagation identity, or imply nonlinear/exact-scri/CPBC/native
instability. Finite-pulse stability and later single-BH wormhole-to-trumpet
formation with the Minkowski hyperboloidal reference remain unresolved.
