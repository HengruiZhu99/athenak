# Elliptic-shift hyperboloidal evolution: PRD-style working manuscript

Evidence snapshot: 26 October 2026, updated at the requested finest-grid stopping point. The original experiment and its failures are retained. A new section reports the separately authorized spherical conformal telegrapher follow-up. Completion of its fixed-parameter spatial study does not complete the unrun robustness controls or the broader research program.

## Read and build

- `main.pdf`: compiled 23-page manuscript with thirteen vector figures and four tables, including five new figures for the telegrapher follow-up.
- `main.tex`, `sections/`, `references.tex`: editable LaTeX source.
- Requires REVTeX 4.2, AMS packages, graphicx, booktabs, hyperref, and standard T1 fonts. Tested with TeX Live 2026.

From this directory:

```sh
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The bibliography is embedded in `references.tex`; no BibTeX step is needed. REVTeX may emit a harmless message about the absent auto-note `.bbl`. The supplied manuscript has no unresolved citations/references or overfull boxes. The expanded two-column draft emits REVTeX deferred-float warnings; floatfix places all floats, whose presence and layout were checked in the rendered PDF. Author names and affiliations have intentionally not been invented.

## Figures and supporting evidence

```sh
python3 make_figures.py
python3 make_telegraph_figures.py
```

Requires NumPy and Matplotlib; plotting uses only the included `data/` files. Together these regenerate thirteen PDF and PNG figures without the simulation repository or binaries. Existing figures can be used directly without Python. The input data are deterministic saved numerical results, not fitted or synthetic substitutes. Guide slopes are labeled in the figures.

`data/evidence.json` preserves the source review records and selected history columns. `data/manifests/` retains complete inputs and source/executable bindings for plotted histories. Compressed `.npz` files contain selected exact-time inner profiles. `data/support/` includes the saved implementation test log, formulation notes, and acceptance protocols. The progress log is chronological historical evidence, not a current acceptance certificate.

`source-manifest.json` identifies the 50 original source files by repository-relative path and SHA-256 hash. `data/failure_inventory.csv` lists 85 explicitly failed/rejected manifests, including historical prototypes and intentional rejection fixtures. Completed-but-scientifically-unsuccessful runs are also discussed in the paper. The count is not a count of independent physical failures.

`collect_evidence.py` documents the read-only export from the original repository layout. It is not needed to compile the paper or redraw its figures and cannot reconstruct the full simulation archive from this bundle. The full source trees, executables, and checkpoint archives remain in the numerical workspace; hashes identify their bindings.

## Interpretation

The historical positive results are finite-pulse field convergence, a separate already-trumpet control, and a coupled dispersal mass–flux check. The follow-up adds completed N=1025/2049/4097 telegrapher trajectories through t=10, with full-domain constraints, lapse profiles and derivatives, and stage-characteristic refinement. Its measured spatial verdict is preserved in `data/telegraph/acceptance.json`. Wormhole-to-trumpet evolution, coupled collapse, ringing, tails, and comprehensive long-duration stability remain unfinished. Several runs reached their requested time while failing scientific acceptance. The paper preserves those distinctions, formulation changes, and unsuccessful repairs.

See `integrity-audit.md` for the document checks and evidence mapping.

## Telegrapher update and stopping scope

`sections/telegraph.tex` gives the added lapse/flux equations, the AF endpoint derivation, the frozen characteristic speeds and damping calculation, the fixed mapped grid, and the numerical/diagnostic repairs. `sections/telegraph-results.tex` describes the evidence. Its numerical macros and table are generated from the assessed records, not manually copied from plots.

`data/telegraph/` contains completed spatial manifests, profile extracts, constraint and horizon histories, characteristic evidence, the acceptance report, the available independent timestep evidence, implementation test receipts, and the predeclared protocols. Its own `source-manifest.json` records the original data and review hashes. `collect_telegraph.py` documents the read-only export and requires completed spatial trajectories; it is not required for a standalone build. `make_telegraph_figures.py` uses only the bundled export.

No inner excision or constraint masking is used in the new full-domain runs. The radial RMS downweights small radii, so unweighted Hamiltonian maxima are retained. A nonmonotone late lapse bump and strong characteristic compression remain visible. Ordered finite-grid rays are not proof of continuum shock avoidance. Independent quadrature, KO-amplitude and adjacent relaxation-parameter studies were not run at the requested stopping point. No robust-stability or completed wormhole-to-trumpet claim is made.

Final status: the fixed candidate passes the declared spatial checks through t=10. The quarter-step control is paused at t=8.35, so full-duration temporal order remains unverified. Further experiments are paused at the user-requested stopping point.
