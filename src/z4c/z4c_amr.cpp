//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================

#include <algorithm>
#include <array>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "z4c/z4c_amr.hpp"
#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/nghbr_index.hpp"
#include "parameter_input.hpp"
#include "z4c/compact_object_tracker.hpp"
#include "z4c/amr_shadow_sensor.hpp"
#include "z4c/z4c.hpp"

#define SQ(X) ((X)*(X))

namespace z4c {

// set some parameters
Z4c_AMR::Z4c_AMR(ParameterInput *pin) {
  std::string ref_method = pin->GetOrAddString("z4c_amr", "method", "trivial");
  if (ref_method == "trivial") {
    method = Trivial;
  } else if (ref_method == "tracker") {
    method = Tracker;
    if (!pin->DoesParameterExist("z4c", "co_0_type")) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line "
                << __LINE__ << std::endl
                << "<z4c_amr>/method=tracker requires at least one "
                   "<z4c>/co_0_type compact-object tracker"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  } else if (ref_method == "chi") {
    method = Chi;
    chi_thresh = pin->GetOrAddReal("z4c_amr", "chi_min", 0.2);
  } else if (ref_method == "dchi" || ref_method == "dchi_max") {
    method = dChi;
    dchi_thresh = pin->GetOrAddReal("z4c_amr", "dchi_max", 0.01);
    dchi_derefine_factor =
        pin->GetOrAddReal("z4c_amr", "dchi_derefine_factor", 0.25);
    if (!(dchi_derefine_factor > 0.0 && dchi_derefine_factor < 1.0)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line "
                << __LINE__ << std::endl
                << "<z4c_amr>/dchi_derefine_factor must be strictly between 0 and 1, but is "
                << dchi_derefine_factor << std::endl;
      std::exit(EXIT_FAILURE);
    }
    dchi_shadow_nyquist =
        pin->GetOrAddBoolean("z4c_amr", "dchi_shadow_nyquist", false);
    capture_replay_dchi =
        pin->DoesParameterExist("mesh_refinement", "amr_history_mode") &&
        pin->GetString("mesh_refinement", "amr_history_mode") == "replay";
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line "
              << __LINE__ << std::endl;
    std::cout << "Unknown refinement strategy: " << ref_method << std::endl;
    std::exit(EXIT_FAILURE);
  }

  for (int nr = 0; nr < 16; ++nr) {
    std::string name = "radius_" + std::to_string(nr) + "_rad";
    if (pin->DoesParameterExist("z4c_amr", name)) {
      radius.push_back(pin->GetReal("z4c_amr", name));
      reflevel.push_back(pin->GetOrAddInteger(
          "z4c_amr", "radius_" + std::to_string(nr) + "_reflevel", -1));
    } else {
      break;
    }
  }
  // num_levels includes root level 0, so the largest physical refinement level is
  // num_levels - 1.  AthenaK's mesh tree enforces the same hard upper bound.
  int num_levels = pin->GetOrAddInteger("mesh_refinement", "num_levels", 1);
  max_ref_lev =
      pin->GetOrAddInteger("z4c_amr", "max_ref_lev", num_levels - 1);
  if ((method == Chi || method == dChi) &&
      (max_ref_lev < 0 || max_ref_lev >= num_levels)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line "
              << __LINE__ << std::endl
              << "<z4c_amr>/max_ref_lev must be between 0 and "
              << num_levels - 1 << " for <mesh_refinement>/num_levels="
              << num_levels << ", but is " << max_ref_lev << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

// 1: refines, -1: de-refines, 0: does nothing
void Z4c_AMR::Refine(MeshBlockPack *pmy_pack) {
  if (method == Tracker) {
    RefineTracker(pmy_pack);
  } else if (method == Chi) {
    RefineChiMin(pmy_pack);
  } else if (method == dChi) {
    RefineDchiMax(pmy_pack);
  }
  RefineRadii(pmy_pack);
}

// refine region within a certain distance from each compact object
// using exact minimum distance via AABB clamping, which correctly handles
// all cases: tracker nearest to a face, edge, or corner of the block.
void Z4c_AMR::RefineTracker(MeshBlockPack *pmbp) {
  Mesh *pmesh       = pmbp->pmesh;
  auto &refine_flag = pmesh->pmr->refine_flag;
  auto &size        = pmbp->pmb->mb_size;
  int nmb           = pmbp->nmb_thispack;
  int mbs           = pmesh->gids_eachrank[global_variable::my_rank];

  std::vector<int> flag;
  flag.reserve(pmbp->pz4c->ptracker.size());

  for (int m = 0; m < nmb; ++m) {
    // current refinement level
    int level = pmesh->lloc_eachmb[m + mbs].level - pmesh->root_level;

    // extract MeshBlock bounds
    Real &x1min = size.h_view(m).x1min;
    Real &x1max = size.h_view(m).x1max;
    Real &x2min = size.h_view(m).x2min;
    Real &x2max = size.h_view(m).x2max;
    Real &x3min = size.h_view(m).x3min;
    Real &x3max = size.h_view(m).x3max;

    flag.clear();
    for (auto &pt : pmbp->pz4c->ptracker) {
      // clamp tracker position to box bounds: closest point on the box
      Real cx = fmax(x1min, fmin(pt->GetPos(0), x1max));
      Real cy = fmax(x2min, fmin(pt->GetPos(1), x2max));
      Real cz = fmax(x3min, fmin(pt->GetPos(2), x3max));

      Real dmin2 = SQ(pt->GetPos(0) - cx) \
                   + SQ(pt->GetPos(1) - cy) \
                   + SQ(pt->GetPos(2) - cz);

      // safety net for radius = 0: dmin2 = 0 inside the block but 0 < SQ(0) is false
      bool iscontained =
        (pt->GetPos(0) >= x1min && pt->GetPos(0) <= x1max) &&
        (pt->GetPos(1) >= x2min && pt->GetPos(1) <= x2max) &&
        (pt->GetPos(2) >= x3min && pt->GetPos(2) <= x3max);

      if (dmin2 < SQ(pt->GetRadius()) || iscontained) {
        if (pt->GetReflevel() < 0 || level < pt->GetReflevel()) {
          flag.push_back(1);
        } else if (level == pt->GetReflevel()) {
          flag.push_back(0);
        } else {
          flag.push_back(-1);
        }
      } else {
        flag.push_back(-1);
      }
    }
    refine_flag.h_view(m + mbs) = *std::max_element(flag.begin(), flag.end());
  }

  // sync host and device
  refine_flag.template modify<HostMemSpace>();
  refine_flag.template sync<DevExeSpace>();
}

// refine based on min{chi}
void Z4c_AMR::RefineChiMin(MeshBlockPack *pmbp) {
  if (pmbp->pz4c->layout.centering == Z4cGridCentering::vertex) {
    RefineChiMinImpl<VertexCenteredZ4c>(pmbp);
  } else {
    RefineChiMinImpl<CellCenteredZ4c>(pmbp);
  }
}

template <typename Centering>
void Z4c_AMR::RefineChiMinImpl(MeshBlockPack *pmbp) {
  Mesh *pmesh       = pmbp->pmesh;
  int nmb           = pmbp->nmb_thispack;
  int mbs           = pmesh->gids_eachrank[global_variable::my_rank];
  auto &refine_flag = pmesh->pmr->refine_flag;
  const auto bounds = pmbp->pz4c->layout;
  const int is = bounds.is, nx1 = bounds.ie - bounds.is + 1;
  const int js = bounds.js, nx2 = bounds.je - bounds.js + 1;
  const int ks = bounds.ks, nx3 = bounds.ke - bounds.ks + 1;
  const int nkji = nx3 * nx2 * nx1;
  const int nji  = nx2 * nx1;
  auto &u0       = pmbp->pz4c->u0;
  int I_Z4C_CHI  = pmbp->pz4c->I_Z4C_CHI;
  // note: we need this to prevent capture by this in the lambda expr.
  auto chi_thresh = this->chi_thresh;
  auto root_lev = pmesh->root_level;
  auto max_ref_lev = this->max_ref_lev;

  par_for_outer(
    "Z4c_AMR::ChiMin", DevExeSpace(), 0, 0, 0, (nmb - 1),
    KOKKOS_LAMBDA(TeamMember_t tmember, const int m) {
      Real team_dmin;
      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(tmember, nkji),
        [=](const int idx, Real &dmin) {
          int k = (idx) / nji;
          int j = (idx - k * nji) / nx1;
          int i = (idx - k * nji - j * nx1) + is;
          j += js;
          k += ks;
          dmin = fmin(u0(m, I_Z4C_CHI, k, j, i), dmin);
        },
        Kokkos::Min<Real>(team_dmin));

      if (team_dmin < chi_thresh) {
        refine_flag.d_view(m + mbs) = 1;
      }
      if (team_dmin > 1.25 * chi_thresh) {
        refine_flag.d_view(m + mbs) = -1;
      }
    });

  // sync host and device
  refine_flag.template modify<DevExeSpace>();
  refine_flag.template sync<HostMemSpace>();

  // enforce maximum refinement level
  for (int m = 0; m < nmb; ++m) {
    // current refinement level
    int level = pmesh->lloc_eachmb[m + mbs].level - pmesh->root_level;

    if (level > max_ref_lev) {
      // derefine mbs above the maximum set refinement level
      refine_flag.h_view(m + mbs) = -1;
    } else if (level == max_ref_lev && refine_flag.h_view(m + mbs) == 1) {
      // avoid over refining
      refine_flag.h_view(m + mbs) = 0;
    }
  }

  // sync host and device
  refine_flag.template modify<HostMemSpace>();
  refine_flag.template sync<DevExeSpace>();
}

// refine based on max{dchi}
void Z4c_AMR::RefineDchiMax(MeshBlockPack *pmbp) {
  if (pmbp->pz4c->layout.centering == Z4cGridCentering::vertex) {
    RefineDchiMaxImpl<VertexCenteredZ4c>(pmbp);
  } else {
    RefineDchiMaxImpl<CellCenteredZ4c>(pmbp);
  }
}

template <typename Centering>
void Z4c_AMR::RefineDchiMaxImpl(MeshBlockPack *pmbp) {
  Mesh *pmesh       = pmbp->pmesh;
  int nmb           = pmbp->nmb_thispack;
  int mbs           = pmesh->gids_eachrank[global_variable::my_rank];
  auto &refine_flag = pmesh->pmr->refine_flag;
  const auto bounds = pmbp->pz4c->layout;
  const int is = bounds.is, nx1 = bounds.ie - bounds.is + 1;
  const int js = bounds.js, nx2 = bounds.je - bounds.js + 1;
  const int ks = bounds.ks, nx3 = bounds.ke - bounds.ks + 1;
  const int nkji = nx3 * nx2 * nx1;
  const int nji  = nx2 * nx1;
  auto &u0       = pmbp->pz4c->u0;
  int I_Z4C_CHI  = pmbp->pz4c->I_Z4C_CHI;
  // note: we need this to prevent capture by this in the lambda expr.
  auto dchi_thresh = this->dchi_thresh;
  auto dchi_derefine_factor = this->dchi_derefine_factor;
  auto root_lev = pmesh->root_level;
  auto max_ref_lev = this->max_ref_lev;
  DvceArray1D<Real> block_dchi;
  DvceArray1D<int> block_dchi_ordinal;
  if (capture_replay_dchi) {
    Kokkos::realloc(block_dchi, nmb);
    Kokkos::realloc(block_dchi_ordinal, nmb);
    Kokkos::deep_copy(block_dchi, 0.0);
    Kokkos::deep_copy(block_dchi_ordinal, std::numeric_limits<int>::max());
  }
  const auto capture_dchi = capture_replay_dchi;

  if (dchi_shadow_nyquist) WriteDchiShadow(pmbp);

  par_for_outer(
    "Z4c_AMR::DchiMax", DevExeSpace(), 0, 0, 0, (nmb - 1),
    KOKKOS_LAMBDA(TeamMember_t tmember, const int m) {
      Real team_dmax;
      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(tmember, nkji),
        [=](const int idx, Real &dmax) {
          int k = (idx) / nji;
          int j = (idx - k * nji) / nx1;
          int i = (idx - k * nji - j * nx1) + is;
          j += js;
          k += ks;
          // This is 2*dx*|grad(chi)| on an isotropic mesh, not |grad(chi)|.
          // Since chi is dimensionless, the indicator is dimensionless and follows a
          // self-similar feature without introducing a preferred physical length.
          Real d2 = SQR(u0(m,I_Z4C_CHI,k,j,i+1) - u0(m,I_Z4C_CHI,k,j,i-1));
          d2 += SQR(u0(m,I_Z4C_CHI,k,j+1,i) - u0(m,I_Z4C_CHI,k,j-1,i));
          if (nx3 > 1) {
            d2 += SQR(u0(m,I_Z4C_CHI,k+1,j,i) - u0(m,I_Z4C_CHI,k-1,j,i));
          }
          dmax = fmax((sqrt(d2)), dmax);
        },
        Kokkos::Max<Real>(team_dmax));

      if (team_dmax > dchi_thresh) {
        refine_flag.d_view(m + mbs) = 1;
      }
      if (team_dmax < dchi_derefine_factor * dchi_thresh) {
        refine_flag.d_view(m + mbs) = -1;
      }
      if (capture_dchi) block_dchi(m) = team_dmax;
    });

  if (capture_replay_dchi) {
    par_for("Z4c_AMR::DchiArgmax", DevExeSpace(), 0, nmb - 1, ks, ks + nx3 - 1,
            js, js + nx2 - 1, is, is + nx1 - 1,
        KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
          Real d2 = SQR(u0(m, I_Z4C_CHI, k, j, i + 1) -
                        u0(m, I_Z4C_CHI, k, j, i - 1));
          d2 += SQR(u0(m, I_Z4C_CHI, k, j + 1, i) -
                    u0(m, I_Z4C_CHI, k, j - 1, i));
          if (nx3 > 1) {
            d2 += SQR(u0(m, I_Z4C_CHI, k + 1, j, i) -
                      u0(m, I_Z4C_CHI, k - 1, j, i));
          }
          if (sqrt(d2) == block_dchi(m)) {
            const int ordinal = ((k - ks) * nx2 + (j - js)) * nx1 + (i - is);
            Kokkos::atomic_min(&block_dchi_ordinal(m), ordinal);
          }
        });
  }

  // sync host and device
  refine_flag.template modify<DevExeSpace>();
  refine_flag.template sync<HostMemSpace>();

  // Honor the Z4c-specific cap as well as AthenaK's mesh-tree num_levels cap.
  for (int m = 0; m < nmb; ++m) {
    int level = pmesh->lloc_eachmb[m + mbs].level - root_lev;
    if (level > max_ref_lev) {
      refine_flag.h_view(m + mbs) = -1;
    } else if (level == max_ref_lev && refine_flag.h_view(m + mbs) == 1) {
      refine_flag.h_view(m + mbs) = 0;
    }
  }

  refine_flag.template modify<HostMemSpace>();
  refine_flag.template sync<DevExeSpace>();
  if (capture_replay_dchi) {
    const auto host_dchi =
        Kokkos::create_mirror_view_and_copy(HostMemSpace(), block_dchi);
    const auto host_ordinal =
        Kokkos::create_mirror_view_and_copy(HostMemSpace(), block_dchi_ordinal);
    last_dchi_max.assign(static_cast<std::size_t>(nmb), 0.0);
    last_dchi_ordinal.assign(static_cast<std::size_t>(nmb), -1);
    for (int m = 0; m < nmb; ++m) {
      last_dchi_max[m] = host_dchi(m);
      last_dchi_ordinal[m] = host_ordinal(m);
    }
  }
}

void Z4c_AMR::WriteDchiShadow(MeshBlockPack *pmbp) {
  Mesh *pmesh = pmbp->pmesh;
  const auto bounds = pmbp->pz4c->layout;
  const int nmb = pmbp->nmb_thispack;
  const int mbs = pmesh->gids_eachrank[global_variable::my_rank];
  constexpr int kCategories = 3;  // interior, active-edge, coarse-fine-adjacent block
  DvceArray3D<Real> maximum("Z4c dchi shadow maxima", nmb, kCategories, Z4c::nz4c);
  DvceArray1D<int> coarse_fine("Z4c dchi shadow coarse-fine", nmb);
  Kokkos::deep_copy(maximum, 0.0);
  auto coarse_fine_host = Kokkos::create_mirror_view(coarse_fine);
  auto &neighbors = pmbp->pmb->nghbr.h_view;
  auto &levels = pmbp->pmb->mb_lev.h_view;
  for (int m = 0; m < nmb; ++m) {
    bool adjacent = false;
    for (int ox2 = -1; ox2 <= 1; ++ox2) {
      for (int ox1 = -1; ox1 <= 1; ++ox1) {
        if (ox1 == 0 && ox2 == 0) continue;
        for (int child = 0; child < 2; ++child) {
          const auto neighbor = neighbors(m, NeighborIndex(ox1, ox2, 0, child, 0));
          adjacent = adjacent || (neighbor.gid >= 0 && neighbor.lev != levels(m));
        }
      }
    }
    coarse_fine_host(m) = adjacent ? 1 : 0;
  }
  Kokkos::deep_copy(coarse_fine, coarse_fine_host);
  const auto u0 = pmbp->pz4c->u0;
  par_for("Z4c_AMR::DchiShadow", DevExeSpace(), 0, nmb - 1, bounds.ks,
          bounds.ke, bounds.js, bounds.je, bounds.is, bounds.ie,
          KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
            const bool edge = (i - bounds.is < 2 || bounds.ie - i < 2 ||
                               j - bounds.js < 2 || bounds.je - j < 2);
            for (int v = 0; v < Z4c::nz4c; ++v) {
              const Real value = FourthDifferenceShadow2D(
                  u0(m, v, k, j, i - 2), u0(m, v, k, j, i - 1),
                  u0(m, v, k, j, i), u0(m, v, k, j, i + 1),
                  u0(m, v, k, j, i + 2), u0(m, v, k, j - 2, i),
                  u0(m, v, k, j - 1, i), u0(m, v, k, j + 1, i),
                  u0(m, v, k, j + 2, i));
              if (!edge) Kokkos::atomic_max(&maximum(m, 0, v), value);
              if (edge) Kokkos::atomic_max(&maximum(m, 1, v), value);
              if (coarse_fine(m) != 0) Kokkos::atomic_max(&maximum(m, 2, v), value);
            }
          });
  const auto host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), maximum);
  std::ostringstream name;
  name << "z4c_amr_shadow.rank" << std::setw(4) << std::setfill('0')
       << global_variable::my_rank << ".csv";
  const bool first = !std::ifstream(name.str()).good();
  std::ofstream output(name.str(), std::ios::app);
  if (!output) Kokkos::abort("cannot open Z4c AMR shadow diagnostic");
  if (first) output << "cycle,time,gid,level,region,component,eta4\n";
  constexpr std::array<const char *, kCategories> labels = {
      "clean_interior", "block_edge", "coarse_fine_adjacent_block"};
  for (int m = 0; m < nmb; ++m) {
    const auto &location = pmesh->lloc_eachmb[m + mbs];
    for (int category = 0; category < kCategories; ++category) {
      for (int v = 0; v < Z4c::nz4c; ++v) {
        output << pmesh->ncycle << ',' << std::setprecision(17) << pmesh->time << ','
               << (m + mbs) << ',' << location.level << ',' << labels[category] << ','
               << Z4c::Z4c_names[v] << ',' << host(m, category, v) << '\n';
      }
    }
  }
  if (!output) Kokkos::abort("cannot write Z4c AMR shadow diagnostic");
}

// Enforce some minimum resolution within a certain spherical region
void Z4c_AMR::RefineRadii(MeshBlockPack *pmbp) {
  Mesh *pmesh       = pmbp->pmesh;
  auto &refine_flag = pmesh->pmr->refine_flag;
  auto &size        = pmbp->pmb->mb_size;
  int nmb           = pmbp->nmb_thispack;
  int mbs           = pmesh->gids_eachrank[global_variable::my_rank];

  for (int m = 0; m < nmb; ++m) {
    // current refinement level
    int level = pmesh->lloc_eachmb[m + mbs].level - pmesh->root_level;

    // extract MeshBlock bounds
    Real &x1min = size.h_view(m).x1min;
    Real &x1max = size.h_view(m).x1max;
    Real &x2min = size.h_view(m).x2min;
    Real &x2max = size.h_view(m).x2max;
    Real &x3min = size.h_view(m).x3min;
    Real &x3max = size.h_view(m).x3max;

    const Real rmin2 = SquaredDistanceToAABB(
        0.0, 0.0, 0.0, x1min, x1max, x2min, x2max, x3min, x3max);

    for (size_t ir = 0; ir < radius.size(); ++ir) {
      if (rmin2 < SQ(radius[ir])) {
        if (level < reflevel[ir]) {
          refine_flag.h_view(m + mbs) = 1;
        } else if (level == reflevel[ir] && refine_flag.h_view(m + mbs) == -1) {
          refine_flag.h_view(m + mbs) = 0;
        }
      }
    }
  }

  // sync host and device
  refine_flag.template modify<HostMemSpace>();
  refine_flag.template sync<DevExeSpace>();
}

} // namespace z4c
