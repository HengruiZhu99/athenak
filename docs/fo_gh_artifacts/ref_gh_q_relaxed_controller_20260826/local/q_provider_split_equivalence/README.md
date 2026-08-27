# Local q-controlled provider split evidence

Commit `5ac381803ed1ac04da24ff626e4cda1ac57564f8` specializes the staged
q-controlled provider launch to evaluate and store one profile jet per work
item. The profile interpolation, automatic-differentiation algebra, q
trajectory, and cache layout are unchanged.

The retained OpenMP source-unit/cache gate and full-output RK4 cycle pass. The
cycle used the same domain, grid, pulse, and output settings as the pre-split
comparison. The physical trumpet summary and four empty outer-region histories
are bitwise identical. Across nonzero histories, the maximum absolute
difference is `1.14e-12`; the principal Ref-GH history differs by at most
`7.11e-14` absolute and `1.45e-15` relative. These are roundoff-level changes,
not a bitwise identity claim.

Restart files were intentionally omitted.
