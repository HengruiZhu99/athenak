# Prompt for a read-only external reviewer

Please review the AthenaK source and evidence at:

- repository: https://github.com/HengruiZhu99/athenak
- branch: `codex/cartoon-2d-high-order-restriction-20260815`
- source repair commit: `345dd31d59cebd9c0c7231be43dcc6a72524bcc7`
- report: https://github.com/HengruiZhu99/athenak/blob/codex/cartoon-2d-high-order-restriction-20260815/docs/investigations/brill_r16_2d_restriction_fix_20260815/README.md

You have read-only access and cannot run the code.  Please audit the
mathematics and source semantics of the collapsed-x3 2D O2/O4/O6
cell-centered restriction repair, especially whether its tensor weights,
edge handling, indexing, and use in both AMR regridding and same-level coarse
boundary refresh are consistent with the existing high-order prolongation.
Confirm that generic non-Z4c physics and the 3D path are unchanged.

Then assess the paired N128/N256 evidence.  Typical constraint jumps at AMR
topology changes shrink after the repair, but N128 fails at nearly the same
time and repaired N256 fails much earlier at the strict-positive chi
parent-stencil gate.  No Figure 3 or convergence claim is made.  Please reason
about the most likely remaining mechanism and propose the smallest bounded
diagnostic that can distinguish (a) invalid chi produced by the RK evolution,
(b) positivity loss/overshoot during high-order restriction, and (c) another
AMR/gauge/formulation coupling.  Do not recommend floors, threshold
relaxation, or a new large parameter sweep unless you can first justify why a
narrow location-resolved diagnostic is insufficient.
