# Reconciliation with comparison-to3m/constraints.png

The earlier figure's top panels use full-domain norms. Its lower panels use
exterior norms only when the horizon finder meets its acceptance tolerance.
The updated candidate-history figure uses exterior norms in every panel and
includes complete provisional horizon candidates as requested. Its invariant
momentum norm is also a different quantity from the radial momentum norm.

The replay changed only diagnostic cutoff selection. Final checkpoints are
byte-for-byte identical to the originals. All shared finite exterior H, Mr and
invariant-M measurements are unchanged. Previously missing measurements were
recovered; they were not interpolated. See checkpoint-comparison.json and
history-verification.json for the original verification.

At 3M:

| Norm | N16 | N24 |
|---|---:|---:|
| Full-domain H | 0.0518183125 | 0.0346150721 |
| Exterior H | 0.0229203683 | 0.0219329477 |
| Full-domain radial Mr | 0.0900247325 | 0.0752266330 |
| Exterior radial Mr | 0.0483433162 | 0.0481541903 |
| Exterior invariant momentum | 0.0354825386 | 0.0422319519 |

Thus the roughly 33% Hamiltonian improvement in the full-domain norm becomes
only 4.3% outside each candidate horizon; radial momentum improves by 16.4%
in the full domain but only 0.39% outside. The invariant exterior momentum
norm is 19.0% larger at N24. These are distinct diagnostics, not contradictory
values of one diagnostic. The cutoffs differ slightly between resolutions;
these comparisons alone do not establish convergence.
