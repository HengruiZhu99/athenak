# Full-field Z4c Bjorhus compatibility boundary RHS

## Scope and qualification boundary

This branch adds the default-off runtime selection

```text
<z4c>/boundary_rhs = full_constraint_bjorhus
```

to the full evolved Z4c state. It does not introduce an analytic background,
residual variables, residual gauge evolution, SAT penalties, floors, or a new
boundary transfer path. The established `sommerfeld` path remains the default.

The implemented treatment is a method-of-lines (MOL) compatibility projection:
after the complete volume RHS is available, it modifies only `Theta` and
`Gamma^i` RHS entries so that four incoming principal constraint-rate rows
vanish. This is mathematically non-overdetermining, but it does **not** preserve
all paired outgoing characteristic rates for generic incoming data. The latter
limitation is derived and tested explicitly below.

This implementation is a bounded numerical feature, not a qualified production
outer boundary condition and not evidence of Brill-collapse convergence.

## Physical frame and constraint variables

Let `C = chi > 0`, and let `gtilde_ij` be the positive-definite conformal metric.
At a physical boundary, the code constructs an outward conformal-unit covector
`s_i` and vector `s^i`:

```math
gtilde^{ij}s_i s_j = 1.
```

At a face intersection, the unnormalized covector is the deterministic sum of
the incident coordinate-face covectors and is normalized with `gtilde^{ij}`.
One kernel owns each point in the precedence order x1, x2, x3; no atomic owner
selection is used.

The Z4 variable represented by the connection sector is

```math
Delta Gamma^i = Gamma^i_evolved - Gamma^i_metric,
qquad
Z_i = 1/2 gtilde_ij Delta Gamma^j.
```

The metric-derivative terms below are the principal contribution of
`Gamma^i_metric`; they are essential to targeting `Z_i`, rather than merely
damping the evolved `Gamma^i` array.

## Incoming principal rows

Freeze the coefficients and use an orthonormal boundary frame with normal `s`
and tangential indices `A=1,2`. Define the trace-free projections

```math
A_ss = s^i s^j Atilde_ij,
qquad
A_sA = s^i Atilde_iA,
```

and denote the corresponding normal derivatives of `chi` and
`gtilde_ij` by `d_chi`, `d_ss`, and `d_sA`. The four incoming light-cone rows
used by the implementation are

```math
w_1^+ = sqrt(C) Theta + C/2 Gamma_s + d_chi,
```

```math
w_2^+ =
  1/sqrt(C) (4/3 Khat + 2/3 Theta - 2 A_ss)
  - Gamma_s + d_ss,
```

```math
w_A^+ = -2/sqrt(C) A_sA - Gamma_A + d_sA.
```

The boundary task evaluates their **rates** from the already-complete volume
RHS, including normal derivatives of the configuration RHS. It requires exactly
one positive and one negative physical-speed member,

```math
lambda_+ = beta_s + alpha sqrt(C) > 0,
qquad
lambda_- = beta_s - alpha sqrt(C) < 0.
```

Invalid metric, lapse, conformal factor, characteristic speed, or local solve
causes a fail-closed abort.

## Sparse Theta/Gamma compatibility solve

Write the four incoming volume-RHS projections as `p_1`, `p_2`, and the
tangential covector `p_A`. Corrections only to `dot Theta` and `dot Gamma^i`
change the incoming rows by

```math
delta dot(w_1^+) = sqrt(C) delta dot(Theta)
                  + C/2 delta dot(Gamma_s),
```

```math
delta dot(w_2^+) = 2/(3 sqrt(C)) delta dot(Theta)
                  - delta dot(Gamma_s),
```

```math
delta dot(w_A^+) = -delta dot(Gamma_A).
```

The scalar determinant is `-4 sqrt(C)/3`, so the map is nonsingular for
`C > 0`. Setting the corrected incoming rates to zero gives

```math
delta dot(Theta) =
  -3/(4 sqrt(C)) (p_1 + C p_2/2),
```

```math
delta dot(Gamma_s) = 3 p_2/4 - p_1/(2C),
qquad
delta dot(Gamma_A) = p_A.
```

In coordinate components, `p_i` is projected into the tangent plane, raised
with `gtilde^{ij}`, and combined with the independently solved normal component.
The production kernel reprojects the corrected rates and aborts if their
residual exceeds a roundoff-scaled tolerance.

This is a zero-incoming-**rate** condition. It preserves homogeneous incoming
data if initialized consistently; it does not instantaneously erase a
pre-existing nonzero incoming characteristic amplitude.

## Why exact outgoing-rate preservation is not claimed

The same sparse correction changes the paired outgoing rows by

```math
delta dot(w_1^-) = -sqrt(C) delta dot(Theta)
                   + C/2 delta dot(Gamma_s),
```

```math
delta dot(w_2^-) = -2/(3 sqrt(C)) delta dot(Theta)
                   - delta dot(Gamma_s),
```

```math
delta dot(w_A^-) = -delta dot(Gamma_A).
```

The outgoing map is itself nonsingular on these four correction components.
Therefore requiring all outgoing changes to vanish forces the sparse correction
to be zero. Generic nonzero incoming rates cannot simultaneously be cancelled.
The requested eight conditions (four incoming targets plus four outgoing-rate
preservation conditions) are overdetermined for four correction degrees of
freedom.

The manufactured incoming-pulse test records this rather than hiding it. For
its fixed data, all incoming residuals vanish to roundoff while the maximum
induced outgoing-rate projection is `0.686111111111...`.

The smallest algebraic full-field extension needs four additional independent
directions:

1. one scalar `Khat/A_ss` principal combination;
2. one independent scalar configuration-derivative-rate direction, such as a
   normal `dot(chi)` reduction variable or an equivalent coupled boundary
   stencil correction;
3. the two tangential `A_sA` rates.

Pointwise `Khat` and `A_ss` corrections supply only one principal scalar
combination, so adding those two array entries alone does not close the rank
deficiency. A clean exact projector therefore requires either a first-order
reduction with independently evolved normal-derivative variables or a coupled
full-field boundary-stencil solve. Neither is introduced here.

## Discrete boundary treatment

- `rho=0` in Cartoon SO(2) is always owned by the axis parity/regularity path,
  including its intersection with a physical z boundary.
- CPBC is applied at physical `rho=rho_max`, `z=zmin`, and `z=zmax` points in
  the native-VC Cartoon path. The collapsed x3 direction is never a Cartoon
  physical face.
- Incident and oblique coordinate contributions use fixed O2 derivatives.
  At a physical face, the stencil points inward. At a local tangential
  MeshBlock edge, it also points into stage-current active data because
  `CalcRHS` does not refresh RHS ghosts before this task. Interior points use a
  centered O2 derivative. The suppressed Cartoon derivative remains the
  analytic tensor-aware symmetry derivative.
- The same implementation is instantiated for cell-centered Z4c and Cartesian
  3D where the existing layouts provide a clean mapping.
- Existing shared-vertex synchronization after the RK update remains the
  authority for duplicate native-VC state values. This boundary kernel uses no
  atomics and does not add a separate shared-RHS synchronization transaction.
