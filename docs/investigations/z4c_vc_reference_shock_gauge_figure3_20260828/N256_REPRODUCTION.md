# N256 reproduction status

## Disposition

`REACHED_TLIM; PARTIAL_CURVE_AGREEMENT; SCIENTIFIC_GATE_FAILED`

Aurora job `8789703` reached coordinate time `30 M` in 5791 cycles. It used one
PVC charged to `CompactBinaryMerger` and exited normally. All 1449 retained
history rows are finite, the minimum sampled axis lapse is `0.1434111145`, and
the final central proper time is `11.2863067801 M`.

The direct, unshifted AthenaK curvature curve follows the published Figure-3
curves at early proper time. Its apparent first peak occurs at proper time
`10.30333 M`, close to the published `10.30683--10.31384 M` range, but is about
`0.47 dex` lower in amplitude. The Hamiltonian and combined constraint
integrals are already increasing catastrophically at that peak. The run also
ends before the published deep minimum and rebound near proper time
`12.6--13.2 M`.

This is therefore partial descriptive agreement, not a qualified Figure-3
reproduction. No horizon finder was enabled, so no horizon conclusion is
available.

Authoritative numerical details and artifact hashes are in `REPORT.md` and
`EVIDENCE_MANIFEST.json`. The direct overlay with a constraint-validity panel is
`analysis/aurora_n256/figures/figure3_with_constraint_validity.png`.
