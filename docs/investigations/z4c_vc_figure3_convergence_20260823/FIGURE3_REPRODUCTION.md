# Figure-3 reproduction disposition

## Authority

The binding is Figure 3 of `arXiv:2607.10843v1`: origin
`log10(abs(R_abcd R^abcd))` versus accumulated central proper time for the
`A=-0.047`, off-center Brill family. Width/code units are plotted directly;
there is no ADM-mass rescaling. Exact authority paths and hashes are in
`AUTHORITY.md`.

The rendered PDF reference begins at
`I(0)=5.6539688247323253e-4`. AthenaK gives:

| Resolution | Initial I(0) | Relative difference from rendered reference |
|---|---:|---:|
| N128 | 5.653739335385570e-4 | -4.0589e-5 |
| N256 | 5.653950047976614e-4 | -3.3210e-6 |
| N512 | 5.653963300593342e-4 | -9.7704e-7 |

This validates the initial quantity, unit convention, and corrected IrisK
coefficient sampling. It does not validate the evolution.

## Outcome

The required long endpoint was not attempted after the early convergence gate
failed. N512 terminates at central proper time `1.53686`; N256 is already in a
protected-interior runaway by `tau_c≈2.383`. The publication curve extends far
beyond this interval into the collapse feature.

The four requested plots are preserved as bounded partial overlays:

- `figures/figure3_vc_N128.pdf/png`
- `figures/figure3_vc_N256.pdf/png`
- `figures/figure3_vc_N512.pdf/png`
- `figures/figure3_vc_overlay.pdf/png`

They use the authenticated curve extraction and unmodified axes/definitions.
They are explicitly not a Figure-3 reproduction claim.

Disposition: `NOT_REPRODUCED_EARLY_NUMERICAL_GATE_FAILED`.

