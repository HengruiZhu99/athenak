# Complete ghost consumers in the necessary TT model

The earlier `check_short_halo_closure.py` included ghost reconstruction in Dq but
added a global KO matrix to all three variables. It did not include reconstruction
at KO's q-ghost consumers. Its no-KO exact result remains valid; its KO treatment
was incomplete and is not a certificate for the implemented transfer.

Let D and Q denote the global centered derivative and KO matrix, respectively.
At a q ghost, reconstruction changes the value by (D_shift-D_owner)h. Accumulating
this change at all derivative consumers gives K_D h, and at all KO consumers gives
K_Q h. The corrected same-level periodic model is

    h_t = v + Q h
    v_t = D q + K_D h + Q v
    q_t = D v - (q-Dh) + Q q + K_Q h

Here lambda=1 and wave speed=1; this is only the TT subsystem. Consequently
E=q-Dh satisfies E_t=-E+Q E+K_Q h. That last term is a discrete tangency defect,
not the continuum candidate's removed source mixing. Its norm grows like h^-2
on unrestricted grid data, so norm growth alone is not a smooth-error growth
rate. Applied to sin(2*pi*x), FD6's measured defect approaches fifth order;
FD4 approaches third order, and FD2 needs no stencil shift at its KO consumers.

`check_halo_ko_coupling.py` constructs the actual shifted ghost correction for
both consumers, FD2/4/6, blocks 8/16/32/64, KO 0/0.3, and SSPRK3 dt=0.2h. Raw
floating spectra retain 31 threshold flags near neutral Jordan modes. Constants
have D1=Q1=K_D1=K_Q1=0 exactly by stencil moments. The constant h/v span is
therefore invariant with a zero-frequency Jordan block. When Q=K_D=0, the
centered derivative's checkerboard h/v span is also invariant. The test verifies
these identities and computes the quotient spectrum without altering the matrix
or the physical production evolution. The quotient does not flag a growing mode
or RK spectral-radius violation in these cases. Thresholds are unchanged.

The first quotient calculation emitted NumPy matmul divide/overflow/invalid
warnings even though all matrices/results were finite and bounded (entries are
far below floating overflow). The final calculation uses explicit non-BLAS
`einsum` contractions and checks finite matrices and neutral invariance, without
suppressing warnings. Both earlier JSON outputs and the final output are retained.
This is a diagnostic implementation correction, not evidence of a physical mode.

The quotient spectrum plus the exact neutral block is still not a uniform energy
estimate, a bound on nonnormal transient amplification, an AMR interface result,
or a full Einstein RK/KO stability proof. No promotion gate is passed by this
necessary screen alone. A projection/reset policy has not been included in this
model. The frozen dt is an investigated value, not a newly qualified CFL bound.
