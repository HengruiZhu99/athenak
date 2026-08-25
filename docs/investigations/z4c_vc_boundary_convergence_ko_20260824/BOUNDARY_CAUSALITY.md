# Conservative boundary causality audit

The calculation does not assume unit characteristic speed.  For each
authenticated Rout=16 run, the recorded maximum coordinate characteristic
speed is integrated with the trapezoidal rule:

```text
D(t) = integral_0^t vmax_coordinate(t') dt'.
```

A region `r<=R` is conservatively protected from the nearest outer face while
`D(t) < 16-R`.

## Results through t=6.5 M

| resolution | max recorded speed | D(6.5) | protected radius at 6.5 | first possible reach r<=12 | first possible reach r<=8 | r<=4 reached? |
|---|---:|---:|---:|---:|---:|---|
| N128 | 1.84836 | 9.33699 | 6.66301 | 2.66637 | 5.55891 | no |
| N256 | 1.85077 | 9.34580 | 6.65420 | 2.66390 | 5.55362 | no |
| N512 | 1.85111 | 9.34706 | 6.65294 | 2.66354 | 5.55288 | no |

The `r<=4` diagnostic remains conservatively causally protected for the full
tau~4 interval.  `r<=8` loses that protection only near coordinate time 5.55,
while `r<=12` can be reached from the old boundary after about 2.664.

These are conservative reach bounds, not proof that a boundary error is
actually launched or dominates a constraint family.  They make `r<=4` the
cleanest existing discriminator and show why Z in larger radii requires the
Rout=128 control.

## Rout=128 protection check

The same integration on the new common-tree runs gives:

| resolution | max recorded speed | D(6.5) | innermost possible boundary reach |
|---|---:|---:|---:|
| N128 | 1.03101 | 3.39587 | 124.604M |
| N256 | 1.23648 | 3.59801 | 124.402M |
| N512 | 1.34122 | 3.66046 | 124.340M |

Thus the Rout=128 physical boundary is conservatively unable to reach even
the outer diagnostic shell near 120M, much less `r<=12M`, by `t=6.5M`.

Machine-readable traces are
`analysis/small_domain_causality_n{128,256,512}.csv`.
