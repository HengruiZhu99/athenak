# Comparing the 3D runner with the preserved gamma4 control

The preserved 1D control uses M=1/3, bulk mu=4, J disabled, and reached coordinate t=33.3333984375, or 100.0001953125 M. Its scientific source is `ef3e9ba0d345854861beaa9675e66de6b8b62db5`, archived in donor `c0ae7472a7aecc92cc72312c33d4cdc417a26291`. The portable reference is `reproduction/original-1d-reference/history.csv` with its adjacent `provenance.json`. Completion does not establish late gauge settling or uniform puncture convergence.

The fidelity audit rechecked the original source-manifest hashes against current `puncture.py`, `gamma_driver.py`, `constraint_damping.py`, `spherical_api.cpp`, `conformal_rhs.hpp`, `conformal_constraints.hpp` and `spherical_tensor.hpp`; all seven match exactly. Existing native evidence covers 105 radial gauge point cases with maximum difference 3.410605131648481e−13, radial symbol reduction below 5e−16, and the spherical reduction of the full coordinate-sphere expansion. No new donor evolutions or tests were run for this audit. See `evidence/native/audit.json`, `tst/hyperbolic_shift_3d/audit.py:47`, and `tst/hyperbolic_shift_3d/test_tensor_scri.cpp:67`.

## Quantities to compare

Use normalized compactified **coordinate volume**, not physical proper volume:

\[
\|f\|_{\rm ext}=\left(\frac{\int_{r>r_H} f^2\,d^3x}{\int_{r>r_H}d^3x}\right)^{1/2}.
\]

For spherical data the 4π factor cancels and this becomes the 1D r²dr norm. The 1D code uses r² times the grid-map Jacobian with half-weight endpoints; 3D uses mapped Cartesian cell-center volume weights. These are different quadratures for the same continuum measure. The 1D scri endpoint contributes its quadrature weight, whereas the 3D volume sum excludes the separate surface atlas. Sources: `tst/hyperbolic_shift_1d/puncture.py:174`, `src/mesh/cubed_sphere/composite_difference.hpp:39`, and `src/z4c/hyperboloidal/hyperbolic_composite_device.hpp:332`.

| Preserved 1D field | Matching 3D quantity | Required interpretation |
|---|---|---|
| `H_exterior_rms` | `H_exterior_L2` | Same native physical ADM Hamiltonian scalar. |
| `M_exterior_rms` | `Mr_exterior_L2` | RMS of coordinate radial covector component nᶦMᵢ, n=x/r. `M_exterior_L2` instead contracts with the inverse Penrose metric. |
| `Theta_exterior_rms` | `Theta_exterior_L2` | Exterior RMS of physical Theta; `Theta_L2` is full-domain. |
| `Z_exterior_rms` | `Zr_exterior_L2` | Exterior RMS of nᶦZᵢ; `Z_L2` is a full-domain Penrose invariant norm. |
| `max_relative_null` | `null_surface` / report `max_null_surface` | Same normalized null identity at alpha=1, comparing running maxima. `null_ghost` is an additional 3D off-node closure check. |
| `apparent_horizon_areal_radius` | `horizon_areal_radius` | Physical R_A=sqrt(area/4π). |
| `apparent_horizon_mass` | `horizon_irreducible_mass`, or `horizon_areal_radius/2` | `horizon_mass` remains Christodoulou mass and must not be silently relabeled. |
| `apparent_horizon_r` | `horizon_radius` | Compact coordinate radius; the 3D value is the mean radius of the found surface. |

The donor adapter writes H, M_r, Theta, Z_r explicitly in `tst/elliptic_shift_1d/spherical_api.cpp:42`. The common constraint definitions are in `conformal_rhs.hpp:EvolvedConstraints` (line 74) and `conformal_constraints.hpp:Z4Constraints` (line 136). In spherical symmetry, the invariant momentum and Z norms include a factor sqrt(chi/g_rr), and their physical invariant norms include an additional Omega. There is no such factor in the donor's radial covector RMS. The 3D accumulator stores both radial and invariant sums at `hyperbolic_composite_device.hpp:360`; the frozen source-v10 MPI output omits exterior Theta and radial Z. A subsequent local diagnostic-only change appends `Theta_exterior_L2`, `Zr_L2`, `Zr_exterior_L2` and `horizon_irreducible_mass` to histories and reports, using the existing reduced sums and FastFlow result. Old raw histories remain unchanged. Missing horizons retain NaN CSV/null JSON exterior and horizon values. The new columns require a later bound executable; snapshot reanalysis remains available for older data. Local one/two-rank nonzero-field and target-zero checkpoint-equivalence evidence is in `reproduction/comparison-diagnostics-local-v1/`.

The native MPI exterior cutoff is its contemporaneous FastFlow mean radius. The old matched-time `compare_coupled.py:40` uses the contemporaneous **1D** horizon cutoff for both datasets. For precise discretization comparisons, use a common specified cutoff; otherwise disclose that different measured horizon locations also change the norm's integration region. Missing horizons give unavailable exterior norms, not zero. A mean-radius sphere is valid for this spherical comparison; it is not the exterior of a distorted 3D surface.

## Horizon geometry and root choice

The physical tensors used by FastFlow are

\[
\gamma_{ij}=\widetilde\gamma_{ij}/(\chi\Omega^2),\qquad
K_{ij}=\widetilde A_{ij}/(\chi\Omega)+\gamma_{ij}(P+2\Theta)/3.
\]

Metric derivatives include both chi and Omega gradients (`src/z4c/hyperboloidal/horizon_geometry.hpp:14`). FastFlow integrates the physical area and uses the outward expansion Dᵢsᶦ+Kᵢⱼsᶦsʲ−K; its irreducible mass is R_A/2, while `result.mass` includes the measured spin (`src/z4c/fastflow.cpp:749`, `:923`, `:1385`). The MPI runner currently assigns `result.mass` to `horizon_mass` (`tst/hyperbolic_shift_3d/coupled_mpi.cpp:351`). The spherical 1D area formula is R_A=r/[Omega sqrt(chi sqrt(g_rr))], consistent with det(gtilde)=1 and the same physical reconstruction. Existing analytic CMC/displaced Schwarzschild, wormhole, Omega=1 reduction and sampled initial-data evidence are documented in `horizon.md`.

There is a concrete late-time root-selection issue: the preserved 100 M reference records **five outward crossings** (nine total marginal crossings), and `tst/elliptic_shift_1d/horizon.py:12` selects the outermost outward crossing. FastFlow is a local continuation search. Before claiming late-time horizon-radius agreement, verify that it finds the same outermost horizon. Equal small mass errors alone do not establish that root identity. This audit does not infer that the extra reference crossings are physically resolved. A read-only diagnostic replay of the retained final N513 profile with its exact original library locates the outward roots at r=0.0039101585, 0.0112332663, 0.0190954350, 0.0266262710 and 0.1705596331. Thus the four inner crossings are clearly separated from the selected outer horizon. Their area masses are approximately 0.24218, 0.24322, 0.25470, 0.25819 and 0.33333333234. The last radius matches the preserved record exactly. `reproduction/original-1d-reference/crossings-audit.json` records brackets and input hashes; the single diagnostic RHS call took no evolution step, activated no floor, and left the profile unchanged.

For the black-hole comparison use t/M=3t, R_A/M=3R_A if dimensionless areal radius is desired, and mass error |3M_irr−1|. Keep compact coordinate r unscaled unless the axis explicitly says r/M. The initial coordinate horizon is approximately 0.06292449, not M/2 in this compact isotropic map. Minkowski/pulse use coordinate time, without an invented black-hole mass. The audit found that `tst/hyperbolic_shift_3d/baseline.py` wrote `t_over_M=3*t` for all kinds, although `compare_coupled.py` correctly chose coordinate time for non-puncture plots. The reporting wrapper now writes JSON null for non-puncture `t_over_M`; existing evidence files retain their original bytes and must be interpreted using `kind` and `t`. No donor evolution source or saved trajectory was changed.

## Initial data, gauge and numerical differences

The initial data use the same `IsotropicWormhole(1/3,96)` embedded Schwarzschild slice and isotropic map, total chi, conformal lapse Omega times physical lapse, and integrated initial shift. The Cartesian lift has gtilde=I, A_ij=(3/2)A_rr(n_i n_j−delta_ij/3), physical P+3, Theta, and beta+x; B, D and J initially vanish. Sources: `tst/hyperbolic_shift_3d/initial_data.py:15` and `tst/hyperbolic_shift_1d/puncture.py:40`. The shift uses 4097-node donor quadrature followed by beta/r interpolation; the retained n16 input records a 2049→4097 difference of 3.22493e−6. This is a known initial-data interpolation error, separate from evolution error. Its library binary hash differs from the preserved duration executable, while the audited donor equation sources match; do not claim identical binaries.

Gauge coefficients match: mu_bulk=4, ratio=2, D damping=1, restoring=4, telegrapher kappa=.005/tau=.02/epsilon=1e−4, inner Z4=8, kappa1=4, kappa2=−.99, telegrapher window .56–.8 and shift collar .8–.9. The separated radial-metric coefficient term is annihilated by w*C=0, as used in the 3D coefficient. The appendix-C scalar source adjustment and inner damping correction also match (`hyperbolic_shift.hpp:13`, `:37`, `:137`; `puncture.py:87`). J must stay disabled for this original-driver comparison.

The main remaining fidelity differences are disclosed discretization choices, not a discovered coefficient bug: the donor uses a logarithmic/rescaled puncture chart, seven-point radial derivatives, an asymptotic r=0 coefficient slot and odd regular-shift treatment; 3D evolves direct fields on an even cell-centered Cartesian core, fourth-order mapped derivatives, seam interpolation and block-local KO8. The 1D floor was inactive in the preserved 100 M control; 3D has no lapse floor. The donor parametrization enforces unit determinant/traceless curvature at all stages, whereas the V2/V3 candidate projects only accepted final stages. Spatial resolution and finite-time stability must therefore be established for the 3D discretization itself; pointwise equation parity is not a transferred 100 M stability result.

Before a long-time comparison: retain matched physical times and source/input identities; select the explicit exterior/radial diagnostics (or reanalyze old snapshots); compare area-derived mass and the same horizon root; state the exterior mask convention; preserve failed-horizon and stale-diagnostic nulls; and keep angular variation alongside radial-profile errors. These are concrete outstanding comparison requirements, not changes to the frozen source-v10 preflight.

## Current V11 comparison through 3M

The discretization discussion above describes the earlier V2/V3 control. V11
now has logarithmic chi/lapse RK and reference KO, continuous Cartesian seams,
and a counted physical lapse floor. Its near-horizon spacing is 5.76 times
coarser and timestep 10.24 times larger than the preserved N513 1D trajectory.
[The current comparison and plots](reproduction/continuous-positive-vs-1d/README.md)
use radial momentum and irreducible mass consistently, preserve failed horizon
samples, and distinguish behavioral agreement from equal-resolution accuracy.
