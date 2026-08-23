# O4 VC transfer selection

## Decision

```text
selected_o4_transfer = NOT_ESTABLISHED
```

The matched q4 midpoint prolongator is rejected as a production-qualified O4 operator because it creates an approximately second-order semidiscrete RHS defect and only 2.16–2.73 order dynamic-AMR convergence.

The elevated q6 prolongator is the sole surviving candidate: it gives approximately fourth-order interface RHS and 4.12–4.56 order dynamic-AMR convergence. It also reduces smooth-mode image content, although its coefficient L1 norm and localized-profile overshoot are slightly larger than q4.

q6 is **not selected for production** because the governing physical prerequisite failed. The fixed-grid Brill evolution is nonconvergent at the axis and outer boundary, so the planned common-tree nonlinear q4/q6 comparison cannot currently distinguish transfer behavior from the bulk/axis/boundary defect.

The default therefore remains unchanged. The explicit q4/q6 selector is retained as a diagnostic and future qualification mechanism.

## Evidence summary

| Criterion | q4 | q6 |
|---|---|---|
| Polynomial reproduction | cubic | quintic |
| State interface order | approximately 4 | approximately 6 |
| RHS interface order | approximately 2 | approximately 4 |
| Dynamic 2D order | 2.16–2.73 | 4.12–4.56 |
| Dynamic 3D order | 2.20–2.71 | 4.17–4.56 |
| Exact coincident restriction | yes | yes |
| CUDA/MPI/restart | passed | passed |
| Nonlinear common-tree Brill | not run | not run |
| Production disposition | `SUBORDER` | `NOT_ESTABLISHED` |

## Re-entry gate

After the fixed-grid Cartoon axis/boundary convergence defect is localized and corrected, repeat Phase 6 unchanged. Only an O4-compatible fixed-grid result authorizes the common-tree q4/q6 comparison and final production selection.
