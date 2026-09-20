# Linear spherical full-state volume operator

The eight-field volume operator passes an exact constraint-propagation check: applying the physical linear constraint map to its RHS gives the independently derived four-field Theta/Q/H/M system, including every trumpet coefficient and its radial derivative. This completes the **volume consistency** gate for the proposed annulus boundary experiment. It is not a boundary, spectrum, or stability result. No AthenaK source, production job or Aurora queue was changed.

The background is the analytic vacuum R0=M=1 trumpet, with `alpha=r/(r+1)`, `chi=alpha²`, radial shift `b=r/(r+1)²`, and `K=1/(r+1)²`. This operator uses the current Z4c geometric equations and original damping product `sigma=kappa*alpha`, with Gamma damping `-2sigma Q`. It does **not** include the experimental coupled CCZ4 sources, modified Gamma damping factor, analytic-jet truncation experiment, KO dissipation, or a finite-difference stencil.

The fixed-background adapted G2 gauge is included exactly:

```
ell_t   = b ell' - 2 alpha k
shift_t = b shift' + 2 Gamma - 2 shift.
```

In particular, adapted lapse advection does not include `delta beta*alpha_bg'`, and the lapse product does not include the standard full-minus-background `delta alpha*K_bg` term. Those would define another gauge.

## State and API

The state is `U=(u,h,k,Theta,A,Gamma,ell,shift)`. Here `u=delta chi`, `h=delta gtilde_nn` with tangential perturbation `-h/2`, `k=delta Khat`, `A` is the trace-free radial projection of `delta Atilde`, and `Gamma` is the radial evolved conformal connection perturbation. Consequently `delta Atilde_rr=A-2K*h/3`, not A alone. The background Cartesian conformal A has radial eigenvalue `-4K/3` and tangential eigenvalues `2K/3`.

`radial_operator.coefficients(rad,rate=.1)` returns `(A2,A1,A0)` such that

```
U_t = A2(r) U'' + A1(r) U' + A0(r) U.
```

Each array has shape `rad.shape+(8,8)`; a scalar radius returns three8×8 arrays. The rational symbolic entries are in `coefficients.json`. `symbolic_coefficients(force=True)` regenerates them with exact first-order dual arithmetic, without differencing nearby finite-amplitude states. Tensor derivatives include the spherical changes of the Cartesian radial basis. The equations are evaluated on the positive x axis; rotational symmetry determines other rays.

## Independent checks

`verify_operator.py` performs two different checks:

- Exact symbolic intertwining `C L = L_constraint C` for Theta, Q, H and the covariant radial momentum M. The constraint map is the independently derived physical map in [NEXT_BOUNDARY.md](../NEXT_BOUNDARY.md), with `Q=Gamma-h'-3h/r`. All four differences simplify identically to zero for arbitrary eight-field functions and arbitrary constant kappa. Coefficient derivatives are retained; this is not a frozen-coefficient test.
- Thirty-five arbitrary radial field/first-derivative/second-derivative vectors across radii .2,.37,.75,1,2,4,8M are reconstructed as full Cartesian tensor jets and passed through the independent NumPy `discrete-symbol/point_operator.py` transcription using complex-step differentiation. Maximum absolute RHS discrepancy is5.69e-14 and maximum relative discrepancy7.67e-16. This path is independently implemented arithmetic, not a fresh execution of the C++ kernel.

The exact background geometric RHS vanishes. The induced A evolution preserves the linear algebraic trace constraint; extracting A by trace-free projection equals `delta Atilde_rr_t+2K*h_t/3`. All results are in `validation.json`.

`check_physical_directions.py` additionally verifies exact zero H/M/Q/Theta for a Schwarzschild mass variation along the R0=M family and for an arbitrary radial spatial-coordinate variation. The current code's C1/C2 incoming amplitudes are generally nonzero for both. They must not be treated as the physical constraints themselves. Volume constraint preservation follows from the intertwining identity; these directions are not asserted to satisfy separately prescribed gauge boundary data or to remain stationary in adapted gauge. See `physical-directions.json`.

The independently owned [boundary work](../radial-full-boundary/) checked all eight actual CPBC gauge/light principal left eigenrows against these volume coefficients at r=.2,.5,1,2,4M, with maximum relative error1.40e-16. The frozen principal construction uses momenta `(k,Theta,A,Gamma)` and derivative fields `(u',h',ell',shift')`. This validates the variable mapping and inner gauge characteristic count, not a lower-order boundary closure.

At r=.2M both physical light directions and both lapse directions leave the annulus toward smaller r. G2 retains one incoming longitudinal shift-gauge direction there, which needs its own data. Omitting that condition because physical constraints are outflow would leave the full problem incomplete. Boundary implementation, finite-grid spectra, off-grid residual checks, and transient behavior remain owned by the separate boundary experiment. The omitted puncture interval r<.2M and nonspherical modes are not represented.

## Reproduction

Use NumPy and the existing local SymPy package, with one BLAS thread:

```
python radial_operator.py --force
python verify_operator.py
python check_physical_directions.py
```

The scripts read the existing review-local SymPy path and do not install dependencies. `manifest.json` records source/evidence hashes. No evolved state, checkpoint or executable is produced by this work.
