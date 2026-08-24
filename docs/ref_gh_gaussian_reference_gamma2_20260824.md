# Ref-GH Gaussian-reference / gamma2 report, 2026-08-24

This branch contains a figure-backed LaTeX report that packages the completed
T4/T5 fixed-stitch evidence and derives the controlled Gaussian-reference and
full-standard-gamma2 investigation requested next.

- LaTeX source: `ref_gh_gaussian_reference_gamma2_20260824.tex`
- rendered PDF: `ref_gh_gaussian_reference_gamma2_20260824.pdf`
- reproducible plotting script:
  `../scripts/ref_gh/plot_ref_gh_gaussian_gamma2_report.py`
- generated figures and manifest:
  `fo_gh_artifacts/ref_gh_gaussian_reference_gamma2_20260824/`

The report distinguishes completed fixed-stitch evidence from unexecuted
Gaussian/gamma2 work.  No source implementation or new simulation was made for
this reporting task, and no trumpet stability or convergence claim is made.

The parent branch was fetched before branch creation.  Local HEAD, its
upstream, and the remote parent all matched exactly at
`e5579269f2e979d246ab162a288b140e7076d666`.  The report branch is
`codex/ref-gh-gaussian-reference-gamma2-20260824`.

Regenerate from the repository root:

```bash
python3 scripts/ref_gh/plot_ref_gh_gaussian_gamma2_report.py
cd docs
pdflatex -interaction=nonstopmode -halt-on-error \
  ref_gh_gaussian_reference_gamma2_20260824.tex
pdflatex -interaction=nonstopmode -halt-on-error \
  ref_gh_gaussian_reference_gamma2_20260824.tex
```
