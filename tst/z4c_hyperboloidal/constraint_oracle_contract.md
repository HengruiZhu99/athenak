# Native accepted-state constraint oracle

Predeclared gate, before first execution. Uses all 229392 N12 independently
evolved bulk points, including zero-weight overlap and endpoint reconstruction,
on initial data, a nonspherical perturbation, and pinned saved step3551 data.
The independent predecessor SphericalTransfer3D restores its own jets and normal
correction; it does not call the new native diagnostic implementation.

Compare chi, lapse, Theta, H, all three momentum and Z covectors, their conformal
and physical norms, physical trace, null residual, determinant/tracefree residuals,
and physical Theta (19 components). Gate: abs(delta)<=1e-12+5e-10*abs(reference),
with chi absolute term1e-14. Report component maxima, weighted errors and worst
locations. Every diagnostic weight must equal the legacy weight exactly. Global
full/fixed-r>.94 RMS obey the saved replay gate5e-11*(1+abs(reference)).

Native full-harmonic FastFlow runs on initial and saved3551 states. Saved full
and horizon-exterior RMS use the existing history gate5e-10+1e-6*abs(reference);
areal radius and irreducible mass in M use1e-10+1e-7*abs(reference). Record complete
surface coefficients and mean radius/expansion/iteration count. The legacy
snapshot/binary provenance limitations in N12-reference.md still apply.

Unused storage and residual workspace begin poisoned. A repeated Refresh at each
state must not re-evaluate constraints. Complete mutation and lifecycle checks
are separate history-output tests. This fixture includes N12 central SMR; it is
not a claim of diagnostics validity after AMR. Run2/8 with the identical frozen
executable, source/dependency/object/checkpoint hashes, and unchanged sources.
