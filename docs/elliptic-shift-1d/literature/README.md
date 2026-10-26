# Reproduction inventory

`reproduction-matrix.json` contains 68 explicitly accounted test/control rows
from the six requested primary sources and the mandatory additional experiments.
Each row includes source URL, section/figure, equations, initial data, reference
and gauge, damping, grid, dissipation, resolutions, CFL/dt, duration, measured
quantities, expected result, status and missing information. **No row is marked
reproduced.** This is an initial inventory; complete follow-up full-text auditing
and figure-axis extraction remain unfinished.

The physical experiments can sometimes share trajectories, while a different
formulation, reference metric, time coordinate or gauge must remain labeled.
For example, the 2023 BH paper evolves GBSSN and its gauge-adapted stationary
data differ from the CMC trumpet data. The 2024 reference-metric study has no
scalar evolution; adding one would be an extension, not reproduction of that
paper. The 2023 conformal-diagram paper shares trumpet relaxation physics with
the earlier BH paper and adds eikonal extraction and scalar-collapse examples.
The initial-data constructions and stationary diagrams are identified separately.

The inventory also lists five directly relevant follow-ups/comments, including
the 2024 spherical generalized-harmonic study and the 2025 Einstein-Maxwell-scalar
studies. Their neutral vacuum/massless-scalar tests require detailed extraction.
Charged-scalar, Maxwell and hydrodynamic neutron-star cases require additional
physics and are not silently treated as mandatory neutral-scalar evolutions.

Primary texts were retrieved from arXiv HTML; `retrieval.json` records URLs and
content hashes. Full retrieved text is cached locally and excluded from Git.
The matrix is generated reproducibly by `tst/elliptic_shift_1d/literature_matrix.py`.
Numbers absent from the inspected source text remain null rather than invented.
