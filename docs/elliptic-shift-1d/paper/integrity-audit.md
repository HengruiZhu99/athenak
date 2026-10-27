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
