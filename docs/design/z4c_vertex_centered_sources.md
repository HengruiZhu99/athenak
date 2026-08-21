# Sources for the vertex-centered Z4c design

This file records provenance for mathematical and implementation ideas. It is
not permission to copy source mechanically. AthenaK remains a MeshBlockPack,
Kokkos, device-buffer implementation; adapted code must carry local
BSD-compatible attribution naming the source file and pinned commit.

## Pinned code reference

GR-Athena++ repository: <https://github.com/computationalrelativity/gr-athena>

Pinned commit: `8583eb9b13639ef7cef65b93f12e9858884527a9`

Reviewed files and the concepts taken from them:

- [`src/mesh/meshblock.cpp`](https://github.com/computationalrelativity/gr-athena/blob/8583eb9b13639ef7cef65b93f12e9858884527a9/src/mesh/meshblock.cpp): separate cell and vertex active/stored index geometry; VC has one additional endpoint in every active direction.
- [`src/mesh/mesh_refinement_vc.cpp`](https://github.com/computationalrelativity/gr-athena/blob/8583eb9b13639ef7cef65b93f12e9858884527a9/src/mesh/mesh_refinement_vc.cpp): coincident-node injection and symmetric Lagrange prolongation semantics.
- [`src/mesh/mesh_refinement.hpp`](https://github.com/computationalrelativity/gr-athena/blob/8583eb9b13639ef7cef65b93f12e9858884527a9/src/mesh/mesh_refinement.hpp): distinct VC refinement entry points and coarse-index contracts.
- [`src/comm/amr_registry.cpp`](https://github.com/computationalrelativity/gr-athena/blob/8583eb9b13639ef7cef65b93f12e9858884527a9/src/comm/amr_registry.cpp): sampling-specific AMR payload registration and dispatch.
- [`src/comm/refinement_ops.cpp`](https://github.com/computationalrelativity/gr-athena/blob/8583eb9b13639ef7cef65b93f12e9858884527a9/src/comm/refinement_ops.cpp): dispatch from VC sampling to injection restriction and Lagrange prolongation.
- [`src/comm/node_multiplicity.cpp`](https://github.com/computationalrelativity/gr-athena/blob/8583eb9b13639ef7cef65b93f12e9858884527a9/src/comm/node_multiplicity.cpp): topology-derived multiplicity for shared vertices. AthenaK adopts the semantic need for multiplicity, but replaces accumulation with an explicit deterministic gather plan.
- [`src/comm/comm_channel.cpp`](https://github.com/computationalrelativity/gr-athena/blob/8583eb9b13639ef7cef65b93f12e9858884527a9/src/comm/comm_channel.cpp), [`comm_enums.hpp`](https://github.com/computationalrelativity/gr-athena/blob/8583eb9b13639ef7cef65b93f12e9858884527a9/src/comm/comm_enums.hpp): separate VC channel ranges and sampling/operator categories.
- [`src/z4c/z4c.cpp`](https://github.com/computationalrelativity/gr-athena/blob/8583eb9b13639ef7cef65b93f12e9858884527a9/src/z4c/z4c.cpp), [`z4c_macro.hpp`](https://github.com/computationalrelativity/gr-athena/blob/8583eb9b13639ef7cef65b93f12e9858884527a9/src/z4c/z4c_macro.hpp): Z4c arrays, coordinates, coarse caches, communication and AMR operators selected consistently by sampling.
- [`src/defs.hpp.in`](https://github.com/computationalrelativity/gr-athena/blob/8583eb9b13639ef7cef65b93f12e9858884527a9/src/defs.hpp.in): distinct VC/CC ghost and interpolation configuration concepts.

AthenaK-specific decisions are: one executable with run-time host selection;
packed Kokkos views rather than per-MeshBlock `AthenaArray`; canonical dyadic
node keys; sorted one-work-item gathers rather than atomics; a separate
`MeshBoundaryValuesVC`; explicit `nvc_tosend`; immutable restart schema; and a
vacuum-only qualification boundary.

## Papers

### Pretorius, 2005

F. Pretorius, *Numerical Relativity Using a Generalized Harmonic
Decomposition*, Class. Quantum Grav. 22 (2005) 425,
[arXiv:gr-qc/0407110](https://arxiv.org/abs/gr-qc/0407110).

Used for the modified-Cartoon strategy: evolve the reduced Cartesian
hyperplane, derive suppressed-direction derivatives from Killing symmetry,
avoid off-plane interpolation, and retain AMR in the reduced domain. This is a
mathematical strategy, not a source-code dependency.

### Cook et al., 2016

W. G. Cook et al., *Dimensional reduction in numerical relativity: Modified
cartoon formalism and regularization*,
[arXiv:1603.00362](https://arxiv.org/abs/1603.00362).

Section 4 and Appendix C supply the SO(2) Cartesian component identities.
Appendix B/C supplies analytic limits at the quasi-radial origin. The paper
explicitly treats a vertex-centered origin and motivates retaining all tensor
components in SO(2), computing only the suppressed derivatives analytically.
AthenaK maps the paper's active radial coordinate to `x1=rho`, axial coordinate
to `x2=z`, and suppressed Cartesian coordinate to `x3=y`.

### Daszuta et al., 2021

B. Daszuta et al., *GRAthena++: puncture evolutions on vertex-centered
oct-tree AMR*, [arXiv:2101.08289](https://arxiv.org/abs/2101.08289).

Used for nodal AMR geometry, coincident-node injection, symmetric
prolongation, complementary coarse representation, and the need to account for
vertex sharing. The source at the pinned commit, rather than prose alone, is
the authority for exact data-flow details.

### Daszuta et al., 2024

B. Daszuta et al., *Numerical relativity simulations of compact binaries:
comparison of cell- and vertex-centered adaptive meshes*,
[arXiv:2406.09139](https://arxiv.org/abs/2406.09139).

Used for coexistence of CC and VC spacetime sampling, sampling-specific
communication/AMR operators, coarse/fine staggering, and the performance and
coupling tradeoffs of keeping matter cell centered.

## Local source authority

The cell-centered regression base is AthenaK commit
`6daa774d7451dbc5f7cac640c6e32a6fd11de7f9`, tree
`cbb702f4da954cf630da261790d5c21ef3142235`, with Kokkos
`6739bc623081648af9e752b616d9671527922cbf`.

The principal local files to refactor are `src/z4c/z4c.{hpp,cpp}`,
`cartoon_derivatives.hpp`, `cartoon_axis_{parity,boundary}.hpp`,
`z4c_{calcrhs,adm,tasks,update,newdt,amr,restart}.cpp`,
`src/bvals/*`, `src/mesh/{mesh_refinement,load_balance}.cpp`, the ADM and
output/restart implementations, and every Z4c consumer enumerated in the
architecture document. Any additional file becomes part of the implementation
inventory and evidence manifest.

