# PC-GH scheme review — investigation paused

[Read the comprehensive PDF](../../output/pdf/pcgh-scheme-review-20260907.pdf) · [Editable LaTeX source](scheme-review.tex) · [Saved evidence and stop record](../../qualification-runs-20260907/pcgh-clean-reduction/review-pause/README.md)

The report contains the full intrinsic 50-field equations, the actual b81b44d6 collision equations and projection schedule, the intermediate direct-lapse attempt, discrete implementation limitations, nine existing production figures, failure comparisons, and a proposed way to combine intrinsic geometry with an explicitly defined gradient closure. Measured findings are separated from conjectures. No solver tests or evolutions were run to prepare it.

The old binary failed at total coordinate time 73.79991M with a non-positive metric. The new coarse uniform puncture failed near 11.03M after large physical Hamiltonian growth and an alpha=2 domain crossing. These are not matched problems or resolutions. New SMR/finer runs were stopped on request, and no intrinsic binary evolution was run.

All three active Slurm jobs and the direct Della run were terminated; the Della queue was verified empty. No investigation job or build remains running. The unfinished health-performance edit was preserved externally and the production source restored to 9763080.

To format the report from this directory (document formatting only):

```sh
mkdir -p ../../output/pdf
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../../output/pdf scheme-review.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../../output/pdf scheme-review.tex
```

Those commands produce `scheme-review.pdf`; the delivered reading copy is named `pcgh-scheme-review-20260907.pdf`. Existing figure pixels are copied unchanged; hashes and original evidence paths are retained. The repository’s older geometry-only TeX record and historical “ongoing” notes are superseded by this pause-state review.
