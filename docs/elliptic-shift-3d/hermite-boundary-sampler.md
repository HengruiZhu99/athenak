# Reusable host Hermite boundary sampler

`src/mesh/cubed_sphere/hermite_boundary_sampler.hpp` turns the qualified boundary-jet prototype into a stage-safe host component. `surface_polynomial_weights.hpp` contains its field-independent angular weights. Neither header is wired into the shared evolution engine or checkpoint format in this change.

## Public interface

```cpp
cubed_sphere::HermiteBoundarySampler sampler(*scri);  // stable lvalue objects
const std::uint64_t epoch = ++boundary_epoch;
auto metric = sampler.Prepare(padded_state, scri->values, 22, epoch);
auto shift  = sampler.Prepare(padded_beta,  scri->beta,    3, epoch);

for (std::size_t p = 0; p < scri->rows.size(); ++p) {
  auto u = sampler.Sample(p, metric);  // ScriSurface::Jets, 22 components
  auto b = sampler.Sample(p, shift);   // same geometry and epoch, 3 components
}
```

Padded data are component-major with stride `grid.count`; surface data are point-major. Components1..64 are accepted. `Prepare` validates shapes and every input entry, computes current coordinate-radial normal data, and returns an immutable value object. It owns F,G,H and the raw logical q0/q00 arrays. No input-vector references or current normal fields remain in the sampler. The `Prepared` also pins a shared immutable geometry frame containing native query geometry and angular plans.

Read-only getters are `components()`, `epoch()`, `trace()`, `radial_first()`, `radial_second()`, `raw_q0()` and `raw_q00()`. Prepared snapshots can be copied or moved. A moved-from object is rejected. The sampler itself is neither copyable nor movable; direct rvalue Surface construction is rejected at compile time. The referenced Grid and Surface must have stable addresses and ordinary C++ lifetimes that outlive the sampler. Moving their containing engine after construction is outside this contract.

A successful greater epoch invalidates sampling of all older snapshots. Several packs, including22 and3 components with different values, may coexist at the same epoch. Every preparation recomputes normals, even at the same epoch. Failed shape, nonfinite, overflow or backwards-epoch preparations do not advance the epoch or invalidate existing valid data. Foreign and expired-owner snapshots are rejected. Their owned data remain readable for diagnostics after the original sampler is destroyed, but they cannot be sampled through another sampler.

`SampleAt(block,q,prepared)` is an additional exact-boundary query interface used by seam/corner tests. Native points route through the exact native contract. It accepts q0=1 and angular coordinates in[0,1]; internal cross-face derivative stencils may extend their angular queries beyond a face through global physical ownership.

## Normal data and map derivatives

Here G=partial_r(component), with r=|x| the compactified Cartesian coordinate radius. This is the Euclidean reference-radial direction, not an uncompactified areal radius or metric-unit Penrose normal. Global Cartesian components remain unchanged; only derivative directions transform. The caller applies evolved-metric contractions and physical source/boundary equations afterward.

The native radial fit and its6/7-node data are unchanged. `Prepare` stores its raw F_q0 and F_q0q0 directly. Given coordinate-map derivatives,

    G = F_q0 / r_s,
    H = (F_q0q0 − r_ss G) / r_s².

The sampler derives r_s, r_ss and r_sA from the inverse map J/H. It inverts J, recovers the forward second derivative

    x^i_AB = −(J_inverse)^i_C H^C_jk x^j_A x^k_B,

and differentiates r=|x|. It includes r_ss for a nonlinear radial map. Independent analytic-differentiation tests use a nonlinear forward map with nonzero r_ss and angular r_sA. Singular, nonfinite, reversed, non-radial and second-jet-bending maps are rejected. Current CompositeMesh maps satisfy the required radial-ray boundary contract.

Only f=F|Sigma, G and H are interpolated across faces. Thus a prescribed constant trace has zero angular derivatives independently of interior values. Mixed derivatives obey

    F_q0A = r_s D_A G + r_sA G,

with centered fourth-order global angular differences. All Cartesian first/second derivatives use the full J/H chain rule. At native points `Sample` uses the stored raw q0/q00 directly, avoiding a divide/multiply round trip. This is numerical differentiation of arbitrary3D fields; it does not average angular data, project differential constraints, replace evolved Lambda, impose conformal flatness, or freeze normal data across stages.

The component-agnostic cache contains only geometry, angular/radial weights, diagnostics and epoch bookkeeping. Its21-term angular QR checks rank, triangular-factor infinity condition, polynomial reproduction and value-weight L1<=8. The reported factor-condition indicator multiplies that QR-factor condition by the normal Vandermonde condition; it is not a measured condition number of a complete jet matrix.

## Geometry integrity and cost

Construction captures n/ng/order/padded/count, mesh descriptor/revision, active IDs, all active position/J/H data and surface row metadata. It validates consistency with the actual map. `Prepare` and explicit `ValidateGeometry()` perform the full O(grid) integrity audit.

`Sample` performs cheap owner/epoch, storage/shape/revision and queried-row/block checks. It deliberately does not scan all47,088 active points per boundary query. An illegal in-place change to an unrelated derived coordinate/ID without changing the mesh revision is caught by the next preparation or explicit full audit, not necessarily by a sample at another row. Legitimate regridding must rebuild the sampler before preparing new data. No promise of automatic detection of every unversioned mutation inside each sample is made.

This implementation uses host storage. Calls that prepare/mutate epoch state and geometry are serialized by the caller; it is not a distributed or device sampler. It consumes the current ScriSurface outer-boundary row contract. Future shell-AMR boundary selection and communication need their own integration and validation.

On the actual frozen47,088-cell mesh with216 boundary points, three repetitions measured approximately:

| Operation | Time |
|---|---:|
| Construct sampler and geometry plans |0.016s |
| Prepare22 |0.0022s |
| Prepare3 |0.0007s |
| Sample all216 points,22 components |0.0036s |
| Sample all216 points,3 components |0.0006s |

The combined stage preparation/sampling cost was about7ms. Beta boundary values were synthetic zero for this timing-only control; the mesh and volume warm start came from the real frozen checkpoint. There was no evolution or shift solve. This is one local small-mesh cost measurement, not a scalability claim.

## Validation and source qualification

Final evidence is `evidence/hermite-sampler-final-20261027`:

- 33 rejection controls cover shape, nonfinite and derived-overflow data, backwards/stale/foreign/moved/expired snapshots, transactional failures, different22/3 packs at one epoch, geometry changes, and invalid maps. Nonlinear map conversion error is8.89e-16.
- All20 qualified MMS configurations pass:1/2/7 shell layers,6/7 radial nodes, three related resolutions and face subdivision. All boundary trace values are bitwise equal to the qualified prototype. Complete first/second jets differ by at most5.51e-14, reflecting the general inverse-map evaluation and exact raw normal storage. No claim of bitwise equality of every derivative is made.
- Zero-trace, intrinsic surface Hessian, quadratic-normal and mixed controls remain at floating-point precision; whole-field constants are exact. Coarse angular errors remain underresolved where previously identified.
- The complete Minkowski tensor-limit source check is unchanged: pole6.66e-16, tensor limit4.44e-15 and clock rate9.88e-16.
- The frozen wormhole pole remains9.14445e-7, above the unchanged1e-7 reference gate. This component alone does not establish coupled evolution success.

The runner copies actual current geometry/interface headers and the new sampler into a private capsule, removes the live project-source include fallback, and records source/runtime hashes. Dependency probes verify all four test binaries use zero live project headers. All204 runtime header/library hashes remained unchanged. Sources, commands, stdout/stderr, result manifests and lossless compressed tables are retained; intermediate attempts are separate.

```sh
python3 tst/elliptic_shift_3d/run_hermite_sampler_validation.py --output NEW_DIRECTORY
```

Every process has a180-second limit and uses one worker. No shared CMake, engine, checkpoint, native owner or original1D checkout was changed. The next integration step is an optional engine sampling scheme with a checkpointed scheme identifier, fresh preparation at every stage, restart/regrid reconstruction, and bounded coupled validation with all original scientific gates retained.
