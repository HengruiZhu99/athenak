# Figure-3 level-20 zero-damping KO/shift controls

This is a prospective three-case continuation of authenticated Perlmutter job
`56979632`.  Every case uses the exact same source, executable, IrisK data,
N128/O6/RK4 grid, pre-collapsed telegrapher lapse with
`(tau,kappa)=(1,1)`, `dchi_max=0.02`, physical levels 0--20, target `t=20`,
four-rank/four-A100 layout, and strict chi gates.

All three disable Z4c constraint damping explicitly:

```text
z4c/damp_kappa1=0
z4c/damp_kappa2=0
z4c/target_kappa1=0
z4c/damp_kappa1_max_K=false
z4c/roll_kappa=false
```

The cases run sequentially in one allocation:

1. fixed Gamma-driver `eta=2`, KO `diss=0.02`;
2. fixed Gamma-driver `eta=2`, KO `diss=0.5`;
3. zero shift, KO `diss=0.5`.

Case 2 differs from case 1 only by the 25-fold KO increase.  Case 3 differs
from case 2 only by setting `shift_Gamma`, `shift_alpha2Gamma`, `shift_H`,
`shift_advect`, and `shift_eta` to zero.  KO `0.5` is intentionally aggressive
and is treated as a diagnostic rather than an assumed production setting.

The frozen arXiv:2607.10843v1 source states that Prague and sphGR use BSSN,
so neither has AthenaK's Z4c kappa damping.  BAMPS uses generalized harmonic
evolution, and the paper does not establish that all of its constraint-damping
terms are zero.  These are therefore zero-Z4c-damping controls, not a claim
that all paper codes use one undamped formulation.

No build or source edit is performed.  Any numerical failure is retained as
the outcome; no floor, threshold relaxation, or adaptive retry is authorized.
