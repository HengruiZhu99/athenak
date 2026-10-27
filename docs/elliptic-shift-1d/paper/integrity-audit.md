# Manuscript integrity and artifact audit

## Original manuscript audit (retained historical record)

Snapshot: 2026-10-26. New manuscript assembled from the saved experiment, not a revision with an original prose baseline. No numerical simulation, build of the numerical solver, or CTest suite was rerun for this writing task. The paused numerical goal remains incomplete.

## Scientific scope and source checks

- The nonlinear formulation, lapse variants, scalar-Theta difference, boundary closure, puncture charts, and scalar sources were compared with the current repository formulation documentation and implementation during drafting. Appendix A explicitly restricts the displayed canceled system to metric freezing (`g=1`); the general metric-derived shift equation is given in the main text.
- Free evolution, algebraic determinant/trace reduction, an elliptic metric connection condition, and independently evolved Z4 fields are distinguished. No Hamiltonian or momentum projection is implied.
- Symmetric restoring, the modified physical-log source, rational alternatives, and reference-lapse damping are disclosed as distinct choices. CMC-source results are labeled separate controls.
- Self-convergence numbers, mass/flux discrepancies, accepted-state maxima, and reported-stage histories retain their original scope. Sampled histories are not presented as all-state maxima. Finite precision and transient failures are explicit.
- The existing implementation verification is 50/50 tests, 53.38 seconds, in `data/support/ctest-v41.log`. This is implementation evidence, not a proof of nonlinear stability or a complete physical validation.
- The fixed-background scalar oracle is separate from coupled dispersal. Failed collapse trajectories do not establish reliable horizon formation. No valid quasinormal-mode or late-tail extraction is claimed.
- Original fixed-amplitude noise remains failed; derivative-balanced noise is a separate experiment with different resolution scaling.
- Bibliographic metadata and primary-source links were checked during drafting. Literature context is not described as completed reproduction. The paper explicitly notes the incomplete reproduction program.

## Figure and claim mapping

| Figure / topic | Portable evidence |
| --- | --- |
| 1, algorithm | Formulation and numerical implementation; schematic only |
| 2, pulse convergence | `gauge` review in `data/evidence.json` |
| 3, arithmetic floor | `gauge`, `subtracted`, and `arithmetic` reviews |
| 4, shift spectrum and transient order | `modes` and `gauge_old` reviews |
| 5, CMC control | `cmc` review and `cmc256/512/1024` histories |
| 6, steep wormhole front | `front513/1025/2049.npz` and `front` review |
| 7, retained domain | `domain1009/2017/4033` histories and manifests; `compact1009.npz`, `global1009.npz` |
| 8, scalar tests | `fixed_scalar`, `scalar`, `dispersal512/1024/2048`, `collapse512/1024/2048` |
| Noise control | `balanced_noise` review; saved noise protocol and chronological progress log |
| Rational-source failures | `rational` review |
| Failure inventory | 85 original manifest paths and digests in `data/failure_inventory.csv` |

## Typesetting and delivery checks

- REVTeX 4.2 PRD two-column format; 16 pages, eight vector figures, three tables, and nine bibliographic entries.
- Two-pass local PDF compilation completed successfully. No undefined references/citations and no overfull boxes in the final build log.
- All rendered pages inspected in a contact sheet, with detailed inspection of the figures, failure table, and wide equation appendix. Float order was adjusted to keep result figures and the failure table before the appendices. No clipping or overlapping elements observed.
- Sources and plotting inputs are included; full original simulation outputs are not duplicated in this paper bundle.
- This audit records document verification, not a new mathematical proof or independent rerun of the experiments.


## Telegrapher revision: scope and preservation

The user authorized updating the existing manuscript and stopping numerical work after the finest grid finishes. Exact pre-edit manuscript files, including the old PDF and uncommitted sources, were saved in the task scratch directory `tmp/paper-telegraph-before-v1`. The old manuscript was compiled with the same local TeX configuration before editing. Its existing `nameref` and underfull-box messages are distinguished from new build issues.

The revision preserves the original Figure 6 failure and all historical results, and adds a separate spherical conformal telegrapher section. Existing equation bodies, citation keys, labels, and numerical tables were retained. Historical-status wording in the abstract, numerical verification section, results introduction, conclusions, and reproducibility appendix was updated so it does not misdescribe the later experiment. The new section and numerical additions are authorized scientific updates, not prose-only edits.

The new lapse/flux equations, AF coefficient limit, coupled characteristic speeds, frozen dispersion relation, fixed grid map, dissipation operator, and passive-ray handling were checked against `telegraph.py`, `telegraph_integrator.cpp`, `radial_grid.py`, and the recorded formulation/test evidence. In particular, the rational C7 coupling ramp is distinguished from the polynomial C7 grid-map smoothstep, and the flat-reference spherical contraction is not described as a covariant physical-lapse transformation. The frozen-subsystem damping result is not promoted to nonlinear stability or shock avoidance.

The 55-group solver test receipt is historical evidence for the implemented kernel. The later two analysis CTest groups are a separate targeted check, not a claim that all 56 groups were rerun. No additional numerical suite or robustness matrix was launched for this writing revision.

The export script rejects incomplete spatial trajectories, checks frozen source/executable bindings and profile finiteness, recomputes the plotted radial RMS values and compares them with saved history, and independently checks the apparent-horizon locations. Table maxima come from the all-accepted-state/stage records, not sparse snapshots. The raw profile and review hashes are retained. All new figure scripts operate on the portable data export and perform no numerical evolution.

The original full-domain acceptance criteria are unchanged. Pointwise inner Hamiltonian errors remain counted despite small radial RMS or accurate exterior mass. Nonmonotone lapse structure and characteristic compression are displayed; ordered finite-grid characteristics do not prove continuum noncrossing. Unrun quadrature, KO-amplitude, and adjacent fixed-parameter controls remain incomplete. The quarter-step status is captured at the requested stopping point and cannot be passed off as a complete time-refinement study.


## Final telegrapher evidence and delivery checks

- All three spatial trajectories (1025/2049/4097 points) completed t=10. The independent half-step trajectory completed t=10. At the user's requested stopping point, the quarter-step worker and dispatcher were suspended with preserved process/checkpoint state at saved time8.35. No additional robustness matrix was dispatched; historical suspended workers were untouched.
- The unchanged spatial assessment passes all listed gates. Bulk/endpoint field orders are5.8881/5.6296. Finest-pair H/M/Theta/Z RMS orders are6.5231/6.4834/6.6171/6.4905. Finest accepted H RMS and interior pointwise maxima are1.52811e-8 and2.28987e-5; maximum matched gradient discrepancy is0.103294%. Scri pole maximum is4.21225e-9; clock is zero and nullness is at roundoff. This does not complete the independent time/robustness protocol.
- Characteristic position and log-J peak-difference orders are4.71856 and3.58878. All81 inward rays remain active at t10 and all81 outward rays are terminal. The finest minimum active inward log-J is-17.55444612. The paper does not infer continuum crossing/noncrossing from finite-grid positivity.
- The new source-only diff was manually reviewed. The protected-span comparison has no removed original mathematical expressions, citations, or labels; additions are the authorized follow-up equations, references, numerical statements and figure labels. This lexical check is limited and is not a scientific proof. Bibliographic entries and new citation associations were also inspected.
- The updated PDF compiles successfully and has23 pages,13 figure captions and4 table captions. All pages were rendered and visually inspected, with detailed checks of the new plots, numerical table and provenance appendix. The original Figure6 remains unchanged. The first lapse-difference figure had overlapping panel letters and scientific-notation offsets; these were corrected before final inspection.
- The final build has no undefined citations/references, no overfull boxes and no pending label-rerun warning. The pre-existing nameref/underfull messages remain. REVTeX also reports deferred-float placement warnings for this expanded two-column draft; floatfix places every float, and all13 figures and4 tables were verified in the rendered PDF. The warnings are disclosed rather than represented as a warning-free build.
- Figure-page placement options were adjusted without altering scientific content. The new ray figure explicitly zooms the outward panel to the exit interval. All numerical figures use raw saved or independently reconstructed data; no smoothing, invented samples, or statistical uncertainty bars were added.
- This is a verified manuscript update at the requested stopping point, not completion of the original robust-convergence or broader physical-reproduction goal.

## Appendix relocation and exterior long-run budget (26 October 2026)

The exact pre-edit 23-page sources/PDF were preserved under `tmp/paper-appendix-before-v1`. The user authorized moving the original Figure 6 to an appendix, emphasizing the telegrapher results, recompiling, and trying a low-resolution puncture continuation to t=500. The old failure paragraph, original plot data, equations, citations, label, and caveats are preserved in Appendix A; it is now Figure 14. The introduction, main-results signposting, abstract, conclusion, and reproducibility map were updated.

A separate N=1025 continuation starts at the verified t=10 checkpoint with identical frozen source, libraries, scheme, and parameters. The target is500. It remains running at this snapshot; no completed long-time-stability claim is made. The portable export is explicitly limited to t<=20 (401 distinct saved times including the parent). Numerical bindings and exact restart overlap were checked; saved fields were checked for finiteness, full-domain RMS values were independently reconstructed and matched to histories, and apparent-horizon positions and masses were independently recomputed from the saved expansion and areal radius.

At the user's subsequent direction, the primary long-run budget excludes nodes inside the apparent horizon and normalizes RMS values using only exterior radial weights. Apparent-horizon mass is the primary invariant. Puncture mass estimates are archived diagnostics and do not enter acceptance. This authorized interpretation change is recorded in `data/long-puncture/exterior-budget.json`; it does not alter the evolution or retroactively rewrite the historical full-domain three-grid assessment. The main new Figure 13 uses exterior norms, compares interior-inclusive and exterior Hamiltonian maxima for localization, and plots relative horizon-mass error. An optional separately named full-domain diagnostic figure is retained outside the manuscript.

At t20, exterior H/M/Theta/Z RMS values are1.79432e-7/2.43681e-6/4.99927e-8/5.21150e-7, exterior H maximum5.87584e-6, and relative horizon-mass error6.42771e-8. The initial transient remains included: peak sampled relative horizon-mass error is2.63630e-6. Large interior errors are not used to reject the revised exterior budget.

The source diff was reviewed. The protected-span comparison for all included section sources has no removed original math, citation, or label spans; additions are authorized budget definitions, numerical evidence, labels and references. No bibliography entries were changed. The lexical check does not establish scientific validity. The numerical kernel was not modified, so no new solver-test-suite claim is made. Both plotting modes were executed and the complete export was rerun after adding the independent horizon check; all401 numerical records remained identical.

The final PDF compiles successfully:24 pages,14 figures,4 tables, all verified by text extraction and rendered-page inspection. The new figure was inspected at full size; panel-label/legend overlap was corrected. There are no undefined references/citations, overfull boxes, or label-rerun warnings. Existing nameref and REVTeX deferred-float warnings remain; every float is present and visually intact. The source-only diff and limited protected-span report accompany the portable export.

The quarter-step control and historical suspended jobs remain paused. Only the separately authorized long continuation is running. The PDF and bundle are fixed evidence snapshots, not a live view of its later output.


## Terminal lapse diagnosis and configurable floor (26 October 2026, local time)

The original unfloored duration trial has terminated at t=43.836474609375. Its 877 distinct saved profiles through t=43.8 replace the prior interim t=20 export. The collector verifies source/library bindings, exact restart overlap, finite fields, full-domain norms, and independently recomputed horizons. Exterior weighting excludes r<r_AH and renormalizes over the retained nodes. Final relative apparent-horizon mass error is 1.1800542e-7; final exterior H RMS is 1.7943228e-7. The whole initial transient remains included.

Offline frozen-kernel replay localizes the failure to node two at r=3.628948e-9 during RK stage four, where log(alpha)=-1109.306 underflows on reconstruction. The source terms and stages are preserved in data/lapse-floor. The isolated native zero-floor control reproduces the failure; default physical floor1e-10 advances 20480 steps to t=44.836474609375, with final relative AH error1.2034525e-7 and exterior H RMS1.7943227e-7. This is a short control, not long-time stability. Seven targeted telegrapher test groups pass; the entire historical suite was not rerun. A fresh floored N1025 wormhole trial targets t500 and remains in progress. The fixed three-grid evidence remains unfloored.

The exact pre-edit source/PDF is preserved in tmp/paper-before-terminal-v1. The source diff and protected-span comparison are bundled in data/lapse-floor. Every removed lexical span was inspected: these are superseded t20 bounds, numerical values, and interior diagnostic examples replaced by terminal diagnostics. No citation or label was removed. The original scheme equations and historical failure appendix remain. The added max-floor equation is an explicitly authorized method change, disclosed as finite-grid regularization with unchanged AF coefficient and scri clock. New numerical claims were checked against replay, native regression, and test receipts; claims of long-duration survival are withheld.

Two LaTeX passes succeed. The final PDF has25 pages,15 figure captions, and4 table captions. All pages were rendered and inspected in a contact sheet, with the new Figure13 exterior histories and Figure14 lapse diagnosis checked at page scale. Original Figure6 remains in AppendixA as Figure15. No undefined citations/references, pending label-rerun warnings, or overfull boxes occur; the pre-existing nameref message and underfull typesetting diagnostics remain. This audit checks this revision and its evidence associations, not a proof of the broader physical formulation.


## Terminal floored trial assessment (26 October 2026, local time)

The fresh N1025 floor1e-10 trial failed at t84.798291015625 after1736669 steps. All1696 saved profiles through84.75 were exported separately to data/floored-long-puncture; the original unfloored data are unchanged. Source/library hashes, finite fields, reconstructed full-domain norms, and independently recomputed horizons pass the collector checks. Final exterior H/M/Theta/Z RMS are1.79432247e-7/2.43681080e-6/4.99926502e-8/5.21149969e-7. Final relative AH mass error2.10449126e-7 remains below the initial-transient maximum2.63629888e-6.

An isolated frozen-kernel checkpoint replay reproduces the speed guard at stage1, node3, r5.53026834e-9. Shift3.93017191e7 dominates both characteristic speeds, mapped Courant number9.93706483e11 exceeds0.4, and dt/tau0.00244140625 does not exceed its bound. The lapse remains at1e-10. Floor activity stays inside6.29212027e-7, but terminal maximum log correction11950.571 is substantial clipping. No successful t500, convergence, or continuum-shock claim is made. No production state or parameters were changed during this assessment. The subsequent user request authorizes investigating and testing a repair separately.

Exact pre-edit sources/PDF are preserved in tmp/paper-before-floor-terminal-v1. The source-only diff and protected-span report are bundled with the new evidence: no protected spans were removed, and additions are the authorized terminal numerical findings, new figure/section labels, and references. Abstract, main results, conclusion, and reproducibility appendix agree on failure versus saved exterior accuracy. Two LaTeX passes succeed with26 pages,16 figures,4 tables, no undefined references/citations, no overfull boxes, and no pending label-rerun warning. The existing nameref and underfull diagnostics remain. Every page was rendered and checked in a contact sheet; the new Figure15 was inspected at page scale. Original Figure6 remains in AppendixA as Figure16. New plots use raw saved data; terminal diagnostics are marked separately from sampled histories.
