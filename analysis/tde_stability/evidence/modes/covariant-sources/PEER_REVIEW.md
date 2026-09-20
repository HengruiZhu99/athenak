# Independent review of coupled covariant constraint sources

**Verdict: no blocking implementation defect found for the bounded, isolated vacuum pilots.** The coupled source formulas match the earlier continuum derivation, the full/background paths apply the same operator, and the saved point/stencil regressions check a genuine nonzero response. This is not a stability result or approval to change production. Review was read-only with respect to AthenaK source; the reviewer added only this report and `peer_check.py`/`peer-check-results.json`.

Reviewed source: `athenak-covariant-sources/src/z4c/{z4c.cpp,z4c.hpp,z4c_calcrhs.cpp,covariant_sources.hpp}`. The immutable regression/pilot executable has SHA256 `7456bbb2cbcef90cdfddbe6d1515220f5e4e02981b799c71e40adb81bf52cd8f`. The later, separate `main.cpp` mode-analysis hook is outside this review.

## Formula and integration audit

- `Q=Gamma_evolved−Gamma_metric` uses the contracted conformal Christoffels calculated from the same full/background metric and its existing centered stencil. The helper uses physical `Z^i=chi Q^i/2`, rather than accidentally raising Z with the conformal metric.
- The E tensor implements `D_i Z_j+D_j Z_i−gtilde_k(i partial_j)Q^k`, including derivatives of the conformal metric and chi. Its trace is contracted with the conformal inverse, then multiplied by chi where the physical trace is required. The A source subtracts its conformal trace and includes `−2alpha Theta A_ij`.
- Khat receives `chi Q·grad(alpha)=2Z·grad(alpha)`. The `−2alpha K Theta` term in full K cancels when converting `Khat=K−2Theta`; the patch correctly avoids adding it again. Theta receives the coupled E, `−alpha K Theta`, and lapse-gradient terms.
- Gamma includes `−2 gtilde^ij Theta partial_j(alpha)`, `−(2/3)alpha K Q^i`, and the Q-dependent shift terms. Both `dg[c][a][b]` and `dbeta[c][a]` use the derivative index first. The actual extraction and `Q[b]*dbeta[b][a]` contraction are consistent; no index transpose bug was found.
- The option also changes Gamma damping from `−2sigma Q` to the quoted CCZ4 convention `−sigma Q`, in the standard RHS, direct residual RHS, and forensic terms. Khat/Theta damping remain consistent. This switch therefore tests the coupled covariant sources **and** the published Gamma damping factor together; it does not isolate either change alone.
- Full/background source evaluations use identical helper code and immutable input views; their difference is added to the existing direct residual RHS. No background Q is assumed zero. The standard full RHS also includes the correction. Default-off geometry contributions are initialized to zero, with no mutations of input views, stencil reads, cache ownership, task dependencies, projection, matter recovery, or the boundary operator.
- Guards require `chi_psi_power=-4` and `use_z4c=true`. Chi divisions use the existing guarded chi; continuum formula equivalence assumes the guard is inactive. No new clipping, resetting, lapse division, or freeze region is introduced.
- Forensic geometry storage correctly grows from 126 to 137 entries per full/background state: Khat+Theta+3Gamma+6A add 11 entries. Background offset 137 and allocation 274 match. Four new named RHS-term groups expose the correction.

The local helper is constraint-surface preserving: `Theta=Q=0` produces exactly zero correction. The finite residual guarantee is narrower and precise: identical full/background state and stencil evaluations yield identical finite source values, whose subtraction is exactly zero. It does not make an arbitrary nonzero perturbation stable.

## Validation evidence reviewed

`point_source_test.py` calls the actual C++ helper at 64 arbitrary nonconstant determinant-one metric/derivative states plus 8 constraint-surface states. Its independent reference constructs physical Christoffels and `D_i Z_j` directly, retaining arbitrary Q derivatives before cancelling the absorbed Ricci term. Maximum absolute source error is `1.73e−18`; trace-free A residual is `9.22e−19`; constraint-surface correction is exactly zero.

`check_validation.py` independently reconstructs the existing sixth-order centered stencil over all 4096 active cells of the saved late state at cycle 6667, `t=500.025M`. It uses physical Christoffels to reconstruct E, rather than calling the helper or copying its expanded E formula. With bitwise-identical on/off input arrays, source differences agree with the actual volume RHS differences to maximum absolute error `3.76e−16` (relative L2 `3.86e−12`; largest expected source `1.16e−4`). Unchanged fields have exactly zero RHS difference. This checks actual tensor indexing and stencil wiring, not only an isolated formula.

The saved regressions also demonstrate:

- 18 zero residual RHS/state arrays remain exactly zero and finite across 3 complete RK3 steps.
- 54 serialized default-off arrays are bitwise identical to the previous lapse-damping executable; this specific comparison has `damp_lapse_scaled=true`, `kappa1=.3`, and `ccz4_covariant_sources=false`.
- Initial low-atmosphere full input and RHS arrays are bitwise unchanged on/off, while the matter response is nonzero: maximum Theta RHS `1.92734546e−13` for density `1e−14`.

The independent reviewer additionally checked all 9 zero-control full/background input pairs byte-for-byte, including ghost cells, and repeated the 54-array default-off and atmosphere checks. Signed scalar covariant increments in the late forensic logs agree with matched on/off RHS increments to their printed precision. Local diagnostic term sums omit the separate KO contribution while printed RHS includes it, so their absolute late sums are not expected to equal the final RHS; the matched increment is the appropriate check. Tiny nonzero recomputed algebraic diagnostics at an exactly zero RHS are pre-existing and do not come from these added source terms.

## Limits and nonblocking observations

- The runtime validation exercises the direct residual branch on one CPU/OpenMP block. The standard full RHS path was source-audited, but not independently runtime-tested here. No MPI/GPU, AMR, long-term, nonlinear, or full-star stability result is implied.
- The higher-density atmosphere attempt (`rho=1e−9`) failed the existing CPBC matter-energy ceiling with the option **off**, before the matched on case. Its failure is retained in the records. The successful `rho=1e−14` test does not validate matter evolution above that boundary ceiling; the ceiling was not relaxed.
- Both new options default false. The source adds first-gradient/nonlinear coefficient terms that are lower order in the linearized constraint sector; this is a geometric formulation change, not an additional independent damping knob.
- The gauge and boundary choices remain those of the existing residual implementation. Calling the added terms “covariant CCZ4 sources” is accurate; calling the entire resulting gauge/boundary/residual solver an independently validated CCZ4 implementation would overstate the evidence.
- A comment in `BuildStandardPointwiseRHS` says “Keep the original Gamma damping convention”; it is accurate for lapse scaling alone, but the covariant option deliberately changes that factor. This is a documentation nit, not a physics/code blocker.
