# Rejected reference-workspace symmetry compression

This directory preserves the two checked Aurora PVC failures that rejected an
attempt to compress analytically symmetric reference metric jets.  No
performance benchmark or scientific evolution was accepted from either job.

- Job `8774093`, commit `056aafcd21532b2c9d19572add32f2fa3ecb51d3`,
  compressed the metric two-jets and inverse-metric value/first-derivative jets.
  The checked cache oracle failed in spin derivatives with conditioned scaled
  Linf `5.94778e-13`, above the unchanged `5.68434e-14` tolerance.
- Job `8774143`, commit `dfdf9feefbdcdd517f97fc811c2b22759d4b494f`,
  restored the orientation-specific metric two-jets but retained compressed
  inverse-metric jets.  The same oracle category failed at `9.96623e-13`.

The analytic tensors are symmetric, but the separately evaluated orientations
do not share an identical floating-point accumulation path near the trumpet
puncture.  Sharing one stored orientation therefore violated this campaign's
strict equivalence gate.  Commit `47d0ad99402ec47c99988a1bf1e7843762706589`
reverted the experiment completely; its Ref-GH source is identical to the
previously passing `3d9da36fd11eedb5f38862f3cc054744b483c649` state.

`SHA256SUMS` covers every compact file here except itself.  Full failed build
evidence remains under the corresponding `symmetric_workspace_8774093.*` and
`inverse_workspace_8774143.*` directories in the Aurora campaign root.
