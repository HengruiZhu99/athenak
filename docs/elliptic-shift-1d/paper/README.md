# Elliptic-shift hyperboloidal evolution: PRD-style working manuscript

Evidence snapshot: 27 October 2026, with the completed spatial study, terminated unfloored and floored duration trials, and the separate attenuation candidate validated through t90. The original experiment and its failures are retained. A new section reports the separately authorized spherical conformal telegrapher follow-up. Completion of its fixed-parameter spatial study does not complete the unrun robustness controls or the broader research program.

## Read and build

- `main.pdf`: compiled 29-page manuscript with seventeen vector figures and five tables, including the telegrapher follow-up, terminal exterior diagnostics, and localized lapse-underflow diagnosis.
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
python3 make_long_exterior_status.py
python3 make_lapse_stop_figure.py
python3 make_floored_long_figure.py
```

Requires NumPy and Matplotlib; plotting uses only the included `data/` files. Together with the attenuation command below, these regenerate seventeen PDF and PNG figures without the simulation repository or binaries. Existing figures can be used directly without Python. The input data are deterministic saved numerical results, not fitted or synthetic substitutes. Guide slopes are labeled in the figures.

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

Current status: the fixed candidate passes the declared spatial checks through t=10. The quarter-step control remains paused at t=8.35, so full-duration temporal order remains unverified. The separate unfloored N=1025 continuation failed at t=43.836474609375; saved exterior constraints remain nearly steady through t=43.8 and the final relative apparent-horizon mass error is 1.18e-7. Offline replay identifies fourth-stage lapse underflow at the second positive-radius node. The new configurable physical-lapse floor defaults to 1e-10; zero disables it. Seven targeted test groups pass, and a separate native checkpoint control advances one additional time unit. The fresh floored wormhole trial also failed, at t=84.798291015625 on the coupled-speed timestep guard. Saved exterior constraints remain nearly steady through t=84.75, with relative horizon-mass error 2.10e-7. Offline replay localizes the guard to shift-dominated transport at node3, r=5.53e-9; the lapse remains at its floor. The target500 was not reached.
## Appendix baseline and long-duration snapshot

The original Figure 6 is now Figure 17 in Appendix A, explicitly labeled as the pre-telegrapher baseline. Its data, failure interpretation, mathematical content, and citations are retained. The introduction and main results now emphasize the completed three-grid telegrapher study.

Figure 13 reports the terminal single-grid duration pilot, not an extension of the three-grid convergence claim. `data/long-puncture/` contains 877 distinct validated saved profiles through t=43.8, the failed manifest, verified parent checkpoint hash, launch command, and predeclared t=500 target. `collect_long_puncture.py` documents the workspace export; `--case`, `--protocol`, `--through`, and `--output` select a separate trial or interim snapshot. Its default case remains the historical unfloored trial. The output directory must not already exist. It checks frozen bindings, exact restart overlap where applicable, profile finiteness, reconstructed full-domain norms, and independently recomputed apparent horizons. `make_long_exterior_status.py` redraws Figure 13 from bundled data.

Figure 14 uses `data/lapse-floor/` and `make_lapse_stop_figure.py` to show the localized underflow. The bundle includes stage replay, isolated floor probes, native regression and targeted implementation-test receipts, the new long-run protocol, and full input. The explicit JSON input is `common.lapse_floor`; the driver also accepts `--lapse-floor`. The floor caps physical alpha/Omega at ordinary positive-radius nodes. The AF coefficient endpoint and scri clock remain unchanged.

The original failed run remains in `output/elliptic-shift-1d/telegraph-long-puncture-500-v1`; the separate fresh floored trial is `output/elliptic-shift-1d/telegraph-lapse-floor-long-v1`. The bundle is a fixed evidence snapshot and does not track subsequent progress automatically. The revised long-run budget excludes the apparent-horizon interior, normalizes RMS norms over the exterior, and treats apparent-horizon mass conservation as the primary invariant. Puncture mass estimates are archived diagnostics, not acceptance quantities. The original full-domain three-grid assessment remains unchanged.

The user-authorized budget amendment is preserved in `data/long-puncture/exterior-budget.json`. The optional `python3 make_long_puncture_figure.py --full-domain-control` produces a separately labeled full-domain diagnostic plot; it is not the primary long-run budget or the plot used in the paper.

## Terminal floored duration trial

Figure15 uses `data/floored-long-puncture/` to report the separate fresh floored run: 1696 saved states through t84.75, native failure at84.798291015625, selected raw lapse profiles, floor-correction histories, and exact frozen-kernel replay. `make_floored_long_figure.py` regenerates the four-panel figure. `make_long_exterior_status.py --data data/floored-long-puncture --output figures/telegraph_floored_exterior_status` produces the supplemental two-panel exterior plot. The original unfloored export remains unchanged. Both trials failed to reach500; neither supplies long-time stability or exterior convergence evidence.

The floor remains confined to r<=6.29e-7, well inside r_AH~0.10765. Terminal mapped Courant number9.94e11 exceeds0.4, while dt/tau=0.0024414 remains within its bound. The derived shift at the offending node is3.93e7; a lower lapse bound does not constrain that shift. No post-failure parameter change or restart was made. The original floor-only monitoring assessment is complete. The monitor now follows the separately authorized attenuation trials; it does not restart either failed trajectory.


## Attenuated duration candidate: interim, not terminal

`sections/telegraph-repair.tex` describes total-lapse RK stages, positive-radius inner stencils, and the explicit gauge multiplier eta=(1+epsilon)*alpha/(alpha+epsilon*A). The existing chi weighting remains. Epsilon1e-4 is used in fresh N513/N1025 trials targeting coordinate t500 in compactification units. The N1025 predeclared exterior budget passes through the fixed t90 export; the duration verdict remains pending. The earlier spatial convergence result concerns a different, unattenuated candidate and is not silently transferred to this modification.

Figure16 is regenerated with:

```sh
python3 make_repair_duration_figure.py --data data/attenuated-duration --output figures/telegraph_repair_duration
```

`data/attenuated-duration/` includes all1801 validated saved-state diagnostic rows and profile receipts through90, selected raw lapse profiles, captured stage extrema through92.65, the protocol, and the pending assessment. The source arrays retain five negative neighboring lapse differences near the puncture. The maximum inner fractional profile change from25 to90 is3.05e-4; this is finite-grid settling evidence, not elimination of oscillations. No radial smoothing is applied to the plot. The failed controls and their ~7.2% coarse-grid AH errors at31.25 are preserved in `data/regularity-repair-controls/`; the nine targeted numerical test-group receipts are in `data/regularity-repair-tests/`.

The source collector hashes and copies the same captured manifest bytes, avoiding a live-read receipt race. The portable assessment helper can check the supplied export without accessing running jobs:

```sh
python3 data/attenuated-duration/assess_telegraph_duration.py --export data/attenuated-duration --output /tmp/attenuated-duration-assessment.json
```

Use a new output filename. A healthy interim export always remains pending; only reaching the declared target with the full accuracy budget can pass the duration gate. The gate has seven separate tests, including negative controls for missing histories, missing horizons, nonfinite constraints, excessive mass/boundary error, and terminal failures with otherwise healthy exterior diagnostics.

`python3 package_source.py` refreshes the sibling portable source ZIP and verifies its CRC and every listed SHA-256 digest. Generated TeX logs, auxiliary files, and QA images are excluded.
