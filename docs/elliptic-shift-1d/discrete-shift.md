# Direct discrete integration of the elliptic shift

The continuum metric-derived g=1 condition is unchanged. Let
W=(beta+r)/r. Bounded regular-center metric-connection freezing excludes the
r^-3 homogeneous metric-rate mode, leaving

    W_r=3 alpha Ar/(2r),
    W(S)=-[c+(2u+u^2)chi]/[1+alpha sqrt(chi)] at S=1.

Lambda is absent from this condition. The outer datum is the stable deviation
form of beta(S)=-alpha(S)sqrt(chi(S)). With the lapse clock, alpha(S)=1.
The independent curvature/Khat/Theta/Lambda fields are never projected.
Analytic beta derivatives are beta_r=-1+W+3alpha Ar/2 and
beta_rr=3(alpha_r Ar+alpha Ar_r+alpha Ar/r)/2.

## Discrete operator and arithmetic control

`shift_ode.py` constructs the banded collocation equation h D_r W=h source,
with the last row replaced by the stated outer Dirichlet datum. Even reflection
supplies regular-center parity. The centered nodal-origin derivative row is
identically zero and is omitted. A biased reflected origin row is not zero;
it must remain in the system. Omitting it, as in the first nodal biased probe,
selects a parasitic static mode. The corrected closure passes independent dense
and analytic-function checks on both origin layouts and finite cutoffs.

A no-pivot banded LU factorization costs O(N w^2) once; each solve costs O(N w).
Small/nonfinite pivots halt setup rather than regularizing the matrix. Selected
half-cell grids through N2048 have minimum scaled pivots above.3, with factor
growth below1. Dense solves independently verify small cases. Polynomial and
smooth manufactured solutions establish consistency. No universal matrix
conditioning theorem is inferred from those numerical checks.

Optional iterative refinement computes the residual in the platform long-double type
and solves a correction with the same factors. On this arm64 host long double has53 significand bits, the same as binary64;
these passes are working-precision residual corrections, not an extended-precision
claim. The 0/2/4 refinement controls
change arithmetic accuracy while preserving this discrete operator. They differ
from changing quadrature degree, which changes the sampled-source integration
discretization. Both types of controls and their failures remain recorded.
Every compiled stage records the maximum physical ODE residual. The independent
expanded second-order residual, computed from sampled beta derivatives, remains
in the reports; it includes spatial product-rule/differentiation error and is
not replaced by the small LU residual.

The Python and C++ integrators use the same supplied band factors, and their
triangular solves, residual arithmetic and RK updates are independently
implemented. Native tensor rates, independent free constraints, scalar sources,
all-stage monitors and short bitwise restarts are checked. ABI23 adds the
operator and stage residual. It does not alter existing frozen binaries.

## Centered failure and biased candidate

The centered ODE matrix solves accurately but its *coupled evolution* is
unstable. Principal pulse cases fail near.7073/.2970/.1055 at N512/1024/2048.
Halving dt does not repair the N1024 failure. The retained finite-grid Jacobian
finds an outer-localized mode, dominated by Ar and Lambda, with real growth
rates .5343/2.9573/11.9043/44.6570 at N64/128/256/512. Two finite-difference
amplitudes give matching rates. Unit or manufactured success alone was therefore
insufficient; this centered variant remains failed.

Replacing only the shift derivative by a one-point right-biased derivative
removes that growing mode in the tested N32--512 spectra. Dominant real parts
remain about-.7055. These are finite-dimensional diagnostics on the full nodal
phase, not a continuum energy estimate or a nonlinear stability theorem. The
unchanged physical PDE and native rate audits motivate the new nonlinear study.

The biased candidate has completed the exact t1 controls at N512/1024/2048.
Bulk/endpoint field orders are5.036/4.821. Accepted bulk H/M peak orders exceed
5.15; RK4 is resolved with CFL.1/.05/.025. Finest 0/4-refinement field changes
are below.2% of the Richardson spatial error in bulk and.007% near scri.
All predeclared t1 convergence/arithmetic gates pass. KO.25/.5/1 controls all
converge: bulk orders5.103/5.036/5.006 and endpoint4.494/4.821/4.933.
The three standard and six KO continuations reach t6, with finest stage and
Bondi gates passing. However a subsequent full common-time field comparison
finds transient endpoint order1.28 at t=.5, despite the final-t1 order4.82.
The sampled peak order is1.86, so the full principal result is **not admitted**.
Fixed-grid timestep error at that frame is5.76e-14, far smaller than spatial
error; the next diagnosis is spatial refinement, not a new time integrator.

With an integer-step clock, N1024/2048/4096 reduces the sampled endpoint
peak difference from6.08e-8 to2.21e-10. The resulting unusually high ratio
is not evidence of eighth-order accuracy. An N8192 check is running to test
its asymptotic interpretation; exact-clock t6 continuations are also running.
No final principal admission or full-goal completion is implied.
