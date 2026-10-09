# Additive correction to the original σ3 Fourier freeze

The original46-file index97eca136515520bb7a551b32013ce4faa7bafde0d3a079cee2620f8367e95aa3 and every indexed file remain byte-identical. Its four C++ Release/Debug compilation/execution commands have empty stderr, and their actual raw22/intrinsic20 matrix payloads and metadata are byte-identical across build modes. Its fifth command emitted NumPy RuntimeWarnings during complex-by-real matrix products despite producing finite roots and a passing numerical summary. Thus the original REPORT.md sentence “All stderr files are empty” is incorrect; check.stderr is2433bytes. It is preserved together with the original source, receipt, numerical results, and prose.

The exact warning command was:

```
/Users/hz0693/Documents/Codex/2026-10-06/referenced-chatgpt-conversation-this-is-an/work/venv/bin/python /Users/hz0693/research/hyperboloidal/build-layer-research/continuum/q-sigma3-frozen-fourier/check_fourier22.py
```

Warnings were reported at original checker lines12,45,46, for matrix products constructing values-only R J22 B and its analytic rank-one comparison. The fresh additive checker uses unoptimized np.einsum for every such product, np.seterr(all='raise'), and warnings.simplefilter('error'). It reads the pinned original C++ matrix payloads without recompiling or rerunning any frozen source. No acceptance tolerance changes.

The accepted additive command is the same Python interpreter with the fresh q-sigma3-frozen-fourier-deterministic/check_fourier22.py path. It completes with exit0 and empty stderr. All original381 source/dependency inputs and all46 original frozen files verify unchanged; the fresh receipt has385 inputs, comprising those381 plus the old index and three additive scripts. Its one recorded checker command took0.9631315seconds.

The raw22 and intrinsic20 eigenvalue arrays are identical. Values-only RJB20 root ordering changes with arithmetic reduction order, so array equality is not claimed: the maximum bidirectional nearest-root distance is7.162510439644634e−12, and the maximum change in a row's max Re λ is1.5918377727075494e−12. Every parameter row has exactly the same positive-root count. All old matrix/control comparisons and independent rank-one bounds are unchanged. A nearest-root-set distance is the stated numerical comparison, not an exact multiplicity/conditioning theorem.

The accepted conclusion remains the negative local screen: σ3 late retains the σ-independent onset-gap maximum, and σ3 early retains positive primitive roots. No new C++/gauge formula, principal change, native/global run, subsidiary classification, nonlinear hierarchy proof, or stability admission follows. Use the new warning-free check-report.json and roots.npz for reanalysis, while retaining the original provenance/error history.
