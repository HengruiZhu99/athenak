//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE for details
//========================================================================================
//! \file kerr_puncture.cpp
//! \brief Reusable input-selected single-Kerr puncture initializer.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "pgen/z4c/kerr_puncture.hpp"
#include "z4c/cartoon_axis_boundary.hpp"
#include "z4c/stored_domain_bounds.hpp"
#include "z4c/z4c.hpp"
#include "z4c/z4c_amr.hpp"
#include "z4c/z4c_symmetry.hpp"

namespace {

[[noreturn]] void Fail(const std::string &message) {
  std::cout << "### FATAL ERROR in " << __FILE__ << std::endl
            << message << std::endl;
  std::exit(EXIT_FAILURE);
}

bool BlockContainsCellCenter(const Real target, const int start,
                             const int end, const int active_start,
                             const int active_count, const Real minimum,
                             const Real maximum) {
  const Real scale = std::max({Real{1}, std::abs(target), std::abs(minimum),
                               std::abs(maximum)});
  const Real tolerance =
      Real{32} * std::numeric_limits<Real>::epsilon() * scale;
  for (int index = start; index <= end; ++index) {
    const Real coordinate = CellCenterX(index - active_start, active_count,
                                        minimum, maximum);
    if (std::abs(coordinate - target) <= tolerance) return true;
  }
  return false;
}

bool MeshSamplesPuncture(Mesh *mesh,
                         const kerr_puncture::CoordinateMap map,
                         const Real axial_center) {
  MeshBlockPack *pack = mesh->pmb_pack;
  const auto &indices = mesh->mb_indcs;
  const auto bounds = z4c::MakeStoredDomainBounds(indices);
  pack->pmb->mb_size.sync_host();
  const auto size = pack->pmb->mb_size.h_view;
  for (int block = 0; block < pack->nmb_thispack; ++block) {
    const Real target1 = 0.0;
    const Real target2 =
        map == kerr_puncture::CoordinateMap::cartesian_xyz ? Real{0}
                                                           : axial_center;
    const Real target3 =
        map == kerr_puncture::CoordinateMap::cartesian_xyz ? axial_center
                                                           : Real{0};
    if (BlockContainsCellCenter(target1, bounds.is, bounds.ie, indices.is,
                                indices.nx1, size(block).x1min,
                                size(block).x1max) &&
        BlockContainsCellCenter(target2, bounds.js, bounds.je, indices.js,
                                indices.nx2, size(block).x2min,
                                size(block).x2max) &&
        BlockContainsCellCenter(target3, bounds.ks, bounds.ke, indices.ks,
                                indices.nx3, size(block).x3min,
                                size(block).x3max)) {
      return true;
    }
  }
  return false;
}

template <kerr_puncture::CoordinateMap Map,
          kerr_puncture::GaugeChoice Gauge>
void FillPhysicalAdm(MeshBlockPack *pack,
                     const kerr_puncture::Parameters<Real> parameters) {
  auto &indices = pack->pmesh->mb_indcs;
  auto &size = pack->pmb->mb_size;
  const auto bounds = z4c::MakeStoredDomainBounds(indices);
  auto &adm = pack->padm->adm;
  auto &mb_bcs = pack->pmb->mb_bcs;
  const int nmb = pack->nmb_thispack;
  const int is = indices.is;
  const int js = indices.js;
  const int ks = indices.ks;
  Kokkos::deep_copy(pack->padm->u_adm, 0.0);
  Kokkos::deep_copy(pack->pz4c->u0, 0.0);
  par_for(
      "initialize Kerr puncture ADM fields", DevExeSpace(), 0, nmb - 1,
      bounds.ks, bounds.ke, bounds.js, bounds.je, bounds.is, bounds.ie,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        if constexpr (Map ==
                      kerr_puncture::CoordinateMap::half_rho_z_suppressed_y_v2) {
          if (mb_bcs.d_view(m, BoundaryFace::inner_x1) == BoundaryFlag::axis &&
              i < is) {
            return;
          }
        }
        const Real x1 = CellCenterX(i - is, indices.nx1,
                                    size.d_view(m).x1min,
                                    size.d_view(m).x1max);
        const Real x2 = CellCenterX(j - js, indices.nx2,
                                    size.d_view(m).x2min,
                                    size.d_view(m).x2max);
        const Real x3 = CellCenterX(k - ks, indices.nx3,
                                    size.d_view(m).x3min,
                                    size.d_view(m).x3max);
        const auto data =
            kerr_puncture::Evaluate<Map, Gauge>(x1, x2, x3, parameters);
        // MeshSamplesPuncture rejects this case on the host.  Keep this branch
        // finite as a defensive fallback so a failed precondition never emits
        // NaN/Infinity from a device kernel.
        if (!data.physical_adm_available) {
          adm.psi4(m, k, j, i) = 1.0;
          adm.alpha(m, k, j, i) = data.lapse;
          for (int a = 0; a < 3; ++a) {
            adm.beta_u(m, a, k, j, i) = data.shift[a];
            for (int b = a; b < 3; ++b) {
              adm.g_dd(m, a, b, k, j, i) = a == b ? Real{1} : Real{0};
              adm.vK_dd(m, a, b, k, j, i) = 0.0;
            }
          }
          return;
        }
        adm.psi4(m, k, j, i) = data.psi4;
        adm.alpha(m, k, j, i) = data.lapse;
        for (int a = 0; a < 3; ++a) {
          adm.beta_u(m, a, k, j, i) = data.shift[a];
          for (int b = a; b < 3; ++b) {
            adm.g_dd(m, a, b, k, j, i) = data.spatial_metric(a, b);
            adm.vK_dd(m, a, b, k, j, i) =
                data.extrinsic_curvature(a, b);
          }
        }
      });
  Kokkos::fence();

  if constexpr (Map ==
                kerr_puncture::CoordinateMap::half_rho_z_suppressed_y_v2) {
    auto &adm_state = pack->padm->u_adm;
    auto &z4c_state = pack->pz4c->u0;
    const int ng = indices.ng;
    par_for(
        "derive Kerr puncture ADM axis ghosts", DevExeSpace(), 0, nmb - 1,
        0, adm::ADM::I_ADM_PSI4, bounds.ks, bounds.ke, bounds.js, bounds.je,
        KOKKOS_LAMBDA(const int m, const int n, const int k, const int j) {
          if (mb_bcs.d_view(m, BoundaryFace::inner_x1) == BoundaryFlag::axis &&
              !z4c::FillAdmAxisGhostLine(adm_state, m, n, k, j, is, ng)) {
            Kokkos::abort("invalid ADM component in Kerr axis parity fill");
          }
        });
    par_for(
        "derive Kerr puncture gauge axis ghosts", DevExeSpace(), 0, nmb - 1,
        z4c::Z4c::I_Z4C_ALPHA, z4c::Z4c::I_Z4C_BETAZ,
        bounds.ks, bounds.ke, bounds.js, bounds.je,
        KOKKOS_LAMBDA(const int m, const int n, const int k, const int j) {
          if (mb_bcs.d_view(m, BoundaryFace::inner_x1) == BoundaryFlag::axis &&
              !z4c::FillZ4cAxisGhostLine(z4c_state, m, n, k, j, is, ng)) {
            Kokkos::abort("invalid gauge component in Kerr axis parity fill");
          }
        });
    Kokkos::fence();
  }
}

void ConvertAdmAndComputeConstraints(MeshBlockPack *pack,
                                     ParameterInput *pin) {
  // AMR requires an even allocated ghost width, so an O4 evolution legitimately
  // uses four stored ghosts with the three-point-half-width Z4c stencil.  Dispatch
  // mathematical operators from the configured stencil, never from allocation size.
  switch (pack->pz4c->opt.fd_stencil) {
    case 2:
      pack->pz4c->ADMToZ4c<2>(pack, pin);
      break;
    case 3:
      pack->pz4c->ADMToZ4c<3>(pack, pin);
      break;
    case 4:
      pack->pz4c->ADMToZ4c<4>(pack, pin);
      break;
    default:
      Fail("kerr_puncture supports Z4c stencil widths 2, 3, or 4");
  }
  pack->pz4c->ReconstructAxisParityGhosts();
  pack->pz4c->Z4cToADM(pack);
  switch (pack->pz4c->opt.fd_stencil) {
    case 2:
      pack->pz4c->ADMConstraints<2>(pack);
      break;
    case 3:
      pack->pz4c->ADMConstraints<3>(pack);
      break;
    case 4:
      pack->pz4c->ADMConstraints<4>(pack);
      break;
  }
}

}  // namespace

void KerrPunctureRefinementCondition(MeshBlockPack *pack) {
  pack->pz4c->pamr->Refine(pack);
}

void ConfigureKerrPuncture(ProblemGenerator *generator, Mesh *mesh,
                           ParameterInput *pin, const bool restart) {
  // This must precede the restart-safe initializer: refinement callbacks are
  // runtime state and are not restored from the checkpoint payload.
  generator->user_ref_func = KerrPunctureRefinementCondition;
  InitializeKerrPuncture(mesh, pin, restart);
}

void InitializeKerrPuncture(Mesh *mesh, ParameterInput *pin,
                            const bool restart) {
  if (restart) return;
  MeshBlockPack *pack = mesh->pmb_pack;
  if (pack->pz4c == nullptr || pack->padm == nullptr) {
    Fail("kerr_puncture requires vacuum Z4c with allocated ADM scratch fields");
  }

  const Real mass = pin->GetOrAddReal("problem", "M", 1.0);
  const Real chi = pin->GetOrAddReal("problem", "chi", 0.0);
  const Real axial_center = pin->GetOrAddReal("problem", "z_h", 0.0);
  const std::string gauge_name =
      pin->GetOrAddString("problem", "initial_gauge", "precollapsed");
  if (!std::isfinite(mass) || mass <= 0.0) {
    Fail("<problem>/M must be finite and positive");
  }
  if (!std::isfinite(chi) || !(std::abs(chi) < 1.0)) {
    Fail("<problem>/chi must be finite and satisfy |chi| < 1");
  }
  if (!std::isfinite(axial_center)) {
    Fail("<problem>/z_h must be finite");
  }
  kerr_puncture::GaugeChoice gauge;
  if (gauge_name == "precollapsed") {
    gauge = kerr_puncture::GaugeChoice::pre_collapsed;
  } else if (gauge_name == "stationary") {
    gauge = kerr_puncture::GaugeChoice::stationary;
  } else {
    Fail("<problem>/initial_gauge must be precollapsed or stationary, not '" +
         gauge_name + "'");
  }
  const auto map =
      pack->z4c_symmetry.mode == z4c::Z4cSymmetryMode::cartoon_so2
          ? kerr_puncture::CoordinateMap::half_rho_z_suppressed_y_v2
          : kerr_puncture::CoordinateMap::cartesian_xyz;
  if (MeshSamplesPuncture(mesh, map, axial_center)) {
    Fail("kerr_puncture requires a cell-centered topology that does not sample "
         "r=0: the physical ADM carrier diverges at the second asymptotic end; "
         "use an even transverse grid with the axis between cell centers. The "
         "analytic conformal puncture limit is retained by the point evaluator; "
         "no epsilon clipping is permitted");
  }

  const kerr_puncture::Parameters<Real> parameters{mass, chi, axial_center};
  if (map == kerr_puncture::CoordinateMap::cartesian_xyz) {
    if (gauge == kerr_puncture::GaugeChoice::pre_collapsed) {
      FillPhysicalAdm<kerr_puncture::CoordinateMap::cartesian_xyz,
                      kerr_puncture::GaugeChoice::pre_collapsed>(pack,
                                                                 parameters);
    } else {
      FillPhysicalAdm<kerr_puncture::CoordinateMap::cartesian_xyz,
                      kerr_puncture::GaugeChoice::stationary>(pack,
                                                              parameters);
    }
  } else if (gauge == kerr_puncture::GaugeChoice::pre_collapsed) {
    FillPhysicalAdm<
        kerr_puncture::CoordinateMap::half_rho_z_suppressed_y_v2,
        kerr_puncture::GaugeChoice::pre_collapsed>(pack, parameters);
  } else {
    FillPhysicalAdm<
        kerr_puncture::CoordinateMap::half_rho_z_suppressed_y_v2,
        kerr_puncture::GaugeChoice::stationary>(pack, parameters);
  }
  ConvertAdmAndComputeConstraints(pack, pin);

  if (global_variable::my_rank == 0) {
    const Real spin = chi * mass;
    const Real r_plus = mass + std::sqrt(mass * mass - spin * spin);
    std::cout << "Initialized arXiv:1001.4077 Kerr puncture: M=" << mass
              << " chi=" << chi << " z_h=" << axial_center
              << " horizon_radius=" << r_plus / 4.0
              << " initial_gauge=" << gauge_name << std::endl;
  }
}
