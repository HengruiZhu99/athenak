# Compiled intrinsic symbol checkpoint

The full 50-field CPU and CUDA point kernels pass 52 sampled symbol cases
(2652 point calls each). This includes all three admitted speed coincidences
at alpha=0.8 and 1.4 with seven offsets each, six separated inner/transition
states, and four states approaching/reaching the excluded alpha=2,w=0.5 point.
These are characteristic checks, not evolution or a puncture-uniform theorem.

At every state the checker builds each column from an oblique unit derivative
jet minus the zero-jet source. After removing beta.n advection, it checks the
full image against the 20-column normal embedding and its action against the
pinned analytical scalar/vector/tensor blocks. Full left/right SVD nullspaces
construct spectral projectors; multiplicities, eigenrelations, completeness,
idempotence and pairwise annihilation use the frozen PLAN.md thresholds.
This does not infer diagonalizability from numerically real eigenvalues.

CPU/CUDA normalized matrix disagreement is 4.864e-16. Maximum analytical block
error is 3.871e-15. All admitted ladders have largest projector norm no greater
than their sampled endpoint maximum (ratio 1.0). At w=0.5, the largest norm grows
from about 182 to 2005 to 20242 as alpha increases from 1.8 to 1.98 to 1.998.
At alpha=2 the two nonzero eigenspaces each have dimension 9 instead of the
algebraic multiplicity 10. PASS for this negative control means the defect was
detected; that state is excluded, not qualified.

The analysis initially emitted NumPy pseudoinverse matmul warnings while
returning finite values. A warnings-as-errors replay isolated the warning to
that product. The final checker evaluates the mathematically identical image
projector U*U^T from the thin SVD of the full-rank normal embedding, using
explicit einsum products. Both saved batches pass with warnings treated as
errors; raw compiled matrices are byte-identical to the first analyses. The
warning control and first results remain in the external manifest. No platform
library defect is claimed proved and no tolerance was loosened.

## Evidence and reproduction

Compact results, all sampled states, per-case projector measurements, a plot,
backend comparison and hashed raw inventories are in
qualification-runs-20260907/pcgh-clean-reduction/intrinsic-symbol-001/.
The manifest identifies the exact d8f3110 kernel source, pinned mathematical
reference, existing CPU/CUDA executable hashes and prior build provenance.
CUDA ran directly on authorized della-vis1 with 36969 MiB free; the 2652-point
batch took 0.58 seconds wall time. No evolution or Slurm job was launched.

From this worktree, with numpy/sympy installed:

```sh
python -W error analysis/pc_gh_clean_reduction/check_intrinsic_symbol.py \
  --binary /absolute/path/to/intrinsic_rhs \
  --reference-dir /absolute/path/to/candidate_20260906/analysis \
  --output /absolute/path/to/new-output-directory
```

For CUDA, copy generated input.txt and states.json into a unique remote run,
run `intrinsic_rhs input.txt output.txt kokkos.txt`, save its log as run.log and
its executable SHA256 as binary-sha256.txt, then copy the batch back. Analyze
with the same command plus `--replay /absolute/path/to/copied-batch`. Replay
requires byte-identical regenerated input and equal states; it never executes
the supplied local binary and reports the saved remote executable hash.
The output directory must not exist.

The proof scope remains compact positive-field subsets of the stated domain;
conditioning near excluded boundaries and w approaching zero is not uniformly
controlled by these finite tests. Full subsidiary, Fourier/transient and
discrete RK/KO checks remain, followed by intrinsic mesh/transfer/restart
integration and ordered physical qualification.
