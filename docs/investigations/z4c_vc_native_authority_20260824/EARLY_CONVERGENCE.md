# Early common-tree convergence

N128, N256, and N512 all reached `t=2.5 M` using the same accepted four-state
logical hierarchy at identical physical event times. N256 record and replay
have byte-identical authority, history, constraint summary, timestep contract,
and axis diagnostic files. All 33 binary and six restart numerical payloads
are bitwise identical after the parameter header.

The complete binary/restart containers are not byte-identical because their
authenticated parameter headers intentionally record `record` versus `replay`
mode and different basenames. The repository integration-test contract defines
numerical payload identity as bytes after `<par_end>`. See
`analysis/record_replay_verification.json`.

At `t=2.5 M`, every global constraint decreases monotonically:

| Resolution | C | H | M | Z |
|---|---:|---:|---:|---:|
| N128 | 2.5117e-2 | 7.3200e-3 | 4.2247e-3 | 3.0319e-3 |
| N256 | 2.8460e-4 | 6.7890e-5 | 5.2302e-5 | 3.8756e-5 |
| N512 | 9.3225e-5 | 1.8973e-5 | 7.0151e-6 | 1.6736e-5 |

Trusted-core field analysis through `t=2 M` finds positive effective order for
every nonzero evolved component. Minimum chi is 0.3211, the minimum conformal
metric SPD pivot is 0.1974, and same-level and coarse-fine shared-node spreads
are exactly zero on the sampled common lattice. Event 3 reduces, rather than
spikes, C/H/M/Z at every resolution.

This establishes the repaired early gate only. It is not a full collapse or
Figure-3 qualification.
