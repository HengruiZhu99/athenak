# Finite-ball projection constraint diagnostic

The analytic projected-constraint point check passes for J0, N8, rb=.98 and the same polynomial interpolants used in the failed FD attempts. It measures substantial constraint production by the radial projection on gauge-only seeds. The complete nongauge continuum comparator remains unresolved; both FD attempts remain failed. This stage supplies no stability acceptance.

At each of 21 locations, the existing actual-source API evaluates every channel with envelopes 1, rho, rho². Linearity in W, W_rho, W_rhorho recovers the constraint map through a triangular solve. Four independent held envelopes test that map, with a separate constraint8 scale; its RHS22 portion also matches the prior source-batch action. Complete reference/angular coefficient jets and the physical lift are retained. No finite difference of a frozen coefficient or value-only restriction is substituted.

The 1,176 API rows and 294 analytic point records pass. Maximum held constraint-row scaled error is 1.419e−14; independent RHS binding is 6.390e−14. Root reconstructs all rows with separate scalar loops and compensated summation, agreeing within 1.301e−14. Independent source/map review finds no correction. All 84 gauge initial constraint vectors are exactly zero in these outputs.

| Gauge seed | Maximum sampled bulk constraint-rate norm | Maximum sampled SAT constraint-rate norm |
|---|---:|---:|
| Constant lapse | 986.113 | .011204 |
| Shift, W=rho³ | 2651.108 | .214873 |
| Lapse, W=exp(−8rho) | 14.3114 | .000616 |
| Shift, W=exp(−8rho) | 5.89942 | .002164 |

These are Euclidean norms of the physical Cartesian components (H, Mx, My, Mz, Zx, Zy, Zz, Theta), at fixed witness amplitudes. They are sample measurements, not integrated energies or normalized growth rates. For these gauge seeds, the continuum constraint-rate zero is a separately derived positive-Omega Einstein-sector identity: constraints are independent of lapse/shift, the ADM gauge variation is tangent to H=M=0 on the reference, and C0 additions vanish at Theta=Z=0. The earlier actual-source/ADM and continuum-rate controls support this derivation. It does not turn either failed FD sequence into a pass. For general nongauge seeds, C_ref[L_actual Phi(X)] is still missing here.

The original five-level FD run stops at constant lapse, r=.96: its continuum last increment 1.841e−6 exceeds 2e−7 despite resolved fourth-order behavior and a small extrapolated zero residual. The separately authorized half-step run stops at r=.30 with cancellation amplification and an extrapolated norm 2.090e−7. Their complete levels, partial measurements and failure receipts remain preserved. Available projected FD values agree with the analytic map to about 3e−8 scaled. No threshold was relaxed.

The [byte-preserved archive](validation/hyperboloidal-projection-constraint-defect-experiments-20261009/README.md) has 141 cataloged blobs, 2,675,294 bytes and 42 finite JSON files; catalog SHA256 is `9096e73edd1a2eb9c205bc4841fad6e9e32fd5a5072c6fb359fcc4e9aa3cd07f`. The original 224-file frozen index is `970a117d2790f668fc3e086de7477c4ea94b683a8772c27a751f2625a623ef28`. All local frozen bytes were freshly rehashed. Raw calls, maps, operators and executable bytes remain metadata only in Git.

Catalog verification passes. The original frozen compact verifier fails because its index labels one small NPZ as a source file, while the Git policy omits every NPZ. The original verifier/index and this failure remain unchanged. The [additive catalog-aware verifier](validation/verify_hyperboloidal_projection_compact_cases.py) admits only explicitly cataloged omissions with matching index hashes/sizes and verifies the same saved-case arithmetic. It passes without scientific queries. Exact failure and review logs are retained in the [separate compact review](validation/hyperboloidal-projection-constraint-defect-compact-review-20261009/catalog.json). The archive's original compact command is historical and has this stated limitation.

No production equation changes occur. Degree refinement and finite-matrix growth are separate controls. Exact scri, a substantial nonlinear angular Minkowski pulse, and a single black hole surviving its inner wormhole-to-trumpet transition with the Minkowski hyperboloidal reference throughout remain unvalidated.

The later [limited finite-matrix growth control](hyperboloidal-limited-finite-growth-audit.md) records N8/N12/N16 gauge growth and saved physical-constraint readbacks while keeping the original full comparator unresolved.
