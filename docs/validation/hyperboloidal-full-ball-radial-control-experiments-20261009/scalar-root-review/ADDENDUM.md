# Additive root review of the scalar dense-mass model

The root independently read the accepted source and checked the half-integral weights, modal normalization/derivatives, full physical stiffness, integration-by-parts flux, scalar boundary cancellation, congruence and L=0 static constant. No correction was requested and no numerical rerun was performed.

For L=0 the implemented stiffness rule has the integrable weight rho^(-1/2)/2 and Q=2rho W_rho. Thus its derivative integrand equals 2rho^(3/2) W_rho V_rho algebraically. The source does not use a separate rho^(3/2) quadrature or evaluate a singular endpoint. The base report already records the general rho^(L-1/2) quadrature; this note makes the L=0 factorization explicit.

The base immutable index remains byte-identical at SHA256 2a61499d58ceae69d0259208b790e2688156e9a6e0496f44dcd52bf27665197e. This additive review admits no actual Z4c radial operator, boundary map or evolution.
