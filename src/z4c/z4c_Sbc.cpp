//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_Sbc.cpp
//! \brief placeholder for Sommerfeld boundary condition

#include <algorithm>
#include <cinttypes>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "z4c/cartoon_derivatives.hpp"
#include "z4c/full_constraint_bjorhus.hpp"
#include "z4c/sommerfeld_derivatives.hpp"
#include "z4c/z4c.hpp"
#include "z4c/boundary_rhs.hpp"
#include "z4c/z4c_symmetry.hpp"
#include "z4c/z4c_vertex_topology.hpp"
#include "coordinates/cell_locations.hpp"

namespace z4c {

namespace {

//! Write one default-off, read-only snapshot of the complete production RHS.
//!
//! A separate file is used for every rank so the diagnostic cannot perturb the
//! production communication order.  Every active local copy is retained: this is
//! important at shared and coarse/fine vertices, where choosing a canonical owner
//! before analysis would hide the very interface discrepancy being measured.
void DumpVertexRHSFieldDiagnostic(Z4c *z4c, MeshBlockPack *pack,
                                  Driver *pdriver, const int stage,
                                  const char *path_prefix) {
  (void)pdriver;
  // Driver initialization invokes BoundaryRHS with stage zero before CalcRHS;
  // that storage is intentionally zero and is not a semidiscrete operator sample.
  if (stage <= 0 || path_prefix == nullptr || path_prefix[0] == '\0') return;
  int diagnostic_stride = 1;
  if (const char *stride_text =
          std::getenv("ATHENA_Z4C_VC_RHS_FIELD_DIAGNOSTIC_STRIDE")) {
    char *end = nullptr;
    const long parsed = std::strtol(stride_text, &end, 10);
    if (end == stride_text || *end != '\0' || parsed <= 0 ||
        parsed > std::numeric_limits<int>::max()) {
      std::cerr << "### FATAL ERROR: ATHENA_Z4C_VC_RHS_FIELD_DIAGNOSTIC_STRIDE "
                   "must be a positive integer"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    diagnostic_stride = static_cast<int>(parsed);
  }
  std::ostringstream path;
  path << path_prefix << ".rank" << std::setfill('0') << std::setw(6)
       << global_variable::my_rank << ".csv";
  std::ifstream prior(path.str());
  if (prior.good()) return;
  prior.close();

  auto *mesh = pack->pmesh;
  auto *blocks = pack->pmb;
  auto &topology = *z4c->vertex_topology_plan;
  blocks->mb_gid.sync_host();
  blocks->mb_lev.sync_host();
  blocks->mb_size.sync_host();
  topology.records.sync_host();
  const auto host_rhs =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), z4c->u_rhs);
  const auto host_state =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), z4c->u0);
  const auto host_constraints =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), z4c->u_con);

  std::ofstream output(path.str(), std::ios::trunc);
  if (!output) {
    std::cerr << "### FATAL ERROR: cannot open native VC RHS field diagnostic "
              << path.str() << std::endl;
    std::exit(EXIT_FAILURE);
  }
  output << "schema,rank,nranks,time,cycle,rk_stage,topology_generation,"
            "diagnostic_stride,local_m,gid,level,relative_level,lx1,lx2,lx3,k,j,i,"
            "key1,key2,key3,role,canonical_owner,topological_multiplicity,"
            "local_edge_codimension,local_edge_distance,rho,x2,x3";
  output << ",nx1_intervals,nx2_intervals,nx3_intervals,"
            "x1min,x1max,x2min,x2max,x3min,x3max";
  for (int variable = 0; variable < Z4c::nz4c; ++variable) {
    output << ",state_" << Z4c::Z4c_names[variable];
  }
  for (int variable = 0; variable < Z4c::nz4c; ++variable) {
    output << ",rhs_" << Z4c::Z4c_names[variable];
  }
  for (int variable = 0; variable < Z4c::ncon; ++variable) {
    output << ',' << Z4c::Constraint_names[variable];
  }
  output << '\n' << std::setprecision(std::numeric_limits<Real>::max_digits10);

  const auto &layout = z4c->layout;
  for (int m = 0; m < pack->nmb_thispack; ++m) {
    const int gid = blocks->mb_gid.h_view(m);
    const auto &location = mesh->lloc_eachmb[gid];
    const auto &size = blocks->mb_size.h_view(m);
    for (int k = layout.ks; k <= layout.ke; ++k) {
      for (int j = layout.js; j <= layout.je; ++j) {
        for (int i = layout.is; i <= layout.ie; ++i) {
          if ((i - layout.is) % diagnostic_stride != 0 ||
              (layout.nx2 > 1 &&
               (j - layout.js) % diagnostic_stride != 0) ||
              (layout.nx3 > 1 &&
               (k - layout.ks) % diagnostic_stride != 0)) {
            continue;
          }
          const auto &record = topology.records.h_view(m, k, j, i);
          const int on_x1_edge = layout.nx1 > 1 &&
              (i == layout.is || i == layout.ie);
          const int on_x2_edge = layout.nx2 > 1 &&
              (j == layout.js || j == layout.je);
          const int on_x3_edge = layout.nx3 > 1 &&
              (k == layout.ks || k == layout.ke);
          const int edge_codimension = on_x1_edge + on_x2_edge + on_x3_edge;
          int edge_distance = std::min(i - layout.is, layout.ie - i);
          if (layout.nx2 > 1) {
            edge_distance = std::min(
                edge_distance, std::min(j - layout.js, layout.je - j));
          }
          if (layout.nx3 > 1) {
            edge_distance = std::min(
                edge_distance, std::min(k - layout.ks, layout.ke - k));
          }
          const Real rho = Z4cPointX<VertexCenteredZ4c>(
              i - layout.is, layout.nx1, size.x1min, size.x1max);
          const Real x2 = Z4cPointX<VertexCenteredZ4c>(
              j - layout.js, layout.nx2, size.x2min, size.x2max);
          const Real x3 = layout.nx3 > 1
              ? Z4cPointX<VertexCenteredZ4c>(
                    k - layout.ks, layout.nx3, size.x3min, size.x3max)
              : 0.0;
          output << "z4c_vc_rhs_field_v2," << global_variable::my_rank << ','
                 << global_variable::nranks << ',' << mesh->time << ','
                 << mesh->ncycle << ',' << stage << ',' << topology.generation
                 << ',' << diagnostic_stride << ',' << m << ',' << gid << ','
                 << location.level << ','
                 << location.level - mesh->root_level << ',' << location.lx1
                 << ',' << location.lx2 << ',' << location.lx3 << ',' << k
                 << ',' << j << ',' << i << ',' << record.key.i1 << ','
                 << record.key.i2 << ',' << record.key.i3 << ','
                 << vertex_topology::VertexNodeRoleName(record.role) << ','
                 << static_cast<int>(record.canonical_diagnostic_owner) << ','
                 << static_cast<int>(record.topological_multiplicity) << ','
                 << edge_codimension << ',' << edge_distance << ',' << rho
                 << ',' << x2 << ',' << x3 << ',' << layout.nx1 << ','
                 << layout.nx2 << ',' << layout.nx3 << ',' << size.x1min
                 << ',' << size.x1max << ',' << size.x2min << ','
                 << size.x2max << ',' << size.x3min << ',' << size.x3max;
          for (int variable = 0; variable < Z4c::nz4c; ++variable) {
            output << ',' << host_state(m, variable, k, j, i);
          }
          for (int variable = 0; variable < Z4c::nz4c; ++variable) {
            output << ',' << host_rhs(m, variable, k, j, i);
          }
          for (int variable = 0; variable < Z4c::ncon; ++variable) {
            output << ',' << host_constraints(m, variable, k, j, i);
          }
          output << '\n';
        }
      }
    }
  }
  if (!output) {
    std::cerr << "### FATAL ERROR: failed while writing native VC RHS field "
                 "diagnostic "
              << path.str() << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

}  // namespace

//---------------------------------------------------------------------------------------
//! \fn TaskStatus Z4c::Z4cBoundaryRHS
//! \brief placeholder for the Sommerfield Boundary conditions for z4c
template <typename Centering, typename Symmetry>
TaskStatus Z4c::Z4cBoundaryRHSImpl(Driver *pdriver, int stage) {
  auto &pm = pmy_pack->pmesh;
  auto &mb_bcs = pmy_pack->pmb->mb_bcs;
  const auto layout = pmy_pack->pz4c->layout;
  auto &size = pmy_pack->pmb->mb_size;

  int nmb = pmy_pack->nmb_thispack;
  int is = layout.is;
  int ie = layout.ie;
  int js = layout.js;
  int je = layout.je;
  int ks = layout.ks;
  int ke = layout.ke;

  auto &z4c_ = z4c;
  auto &rhs_ = rhs;
  bool &user_Sbc = opt.user_Sbc;
  const int fd_stencil = opt.fd_stencil;
  auto bcs = mb_bcs.d_view;
  const bool compact_work = layout.centering == Z4cGridCentering::vertex &&
                            opt.lean_runtime;

  // We only need to apply this condition for outflow boundaries
  if (pm->mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::outflow
      || pm->mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::diode
      || pm->mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::vacuum
      || pm->mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::user
      || pm->mesh_bcs[BoundaryFace::outer_x1] == BoundaryFlag::outflow
      || pm->mesh_bcs[BoundaryFace::outer_x1] == BoundaryFlag::diode
      || pm->mesh_bcs[BoundaryFace::outer_x1] == BoundaryFlag::vacuum
      || pm->mesh_bcs[BoundaryFace::outer_x1] == BoundaryFlag::user) {
    const auto boundary_work = physical_boundary_x1_work.d_view;
    const int work_count = compact_work ? nphysical_boundary_x1_work : nmb;
    par_for("z4crhs_bc_x1", DevExeSpace(), 0, (work_count-1), ks, ke, js, je,
    KOKKOS_LAMBDA(int row, int k, int j) {
      const int m = compact_work ? boundary_work(row) : row;
      // Inner boundary
      switch(mb_bcs.d_view(m,BoundaryFace::inner_x1)) {
        case BoundaryFlag::diode:
        case BoundaryFlag::vacuum:
        case BoundaryFlag::outflow:
            Z4cSommerfeldConfigured<Centering, Symmetry>(
                z4c_, rhs_, layout, size, bcs, user_Sbc, fd_stencil, m, k, j, is);
          break;
        case BoundaryFlag::user:
            if (user_Sbc) {
              Z4cSommerfeldConfigured<Centering, Symmetry>(
                  z4c_, rhs_, layout, size, bcs, user_Sbc, fd_stencil, m, k, j, is);
            }
          break;
        default:
          break;
      }
      // Outer boundary
      switch (mb_bcs.d_view(m,BoundaryFace::outer_x1)) {
        case BoundaryFlag::diode:
        case BoundaryFlag::vacuum:
        case BoundaryFlag::outflow:
            Z4cSommerfeldConfigured<Centering, Symmetry>(
                z4c_, rhs_, layout, size, bcs, user_Sbc, fd_stencil, m, k, j, ie);
          break;
        case BoundaryFlag::user:
            if (user_Sbc) {
              Z4cSommerfeldConfigured<Centering, Symmetry>(
                  z4c_, rhs_, layout, size, bcs, user_Sbc, fd_stencil, m, k, j, ie);
            }
          break;
        default:
          break;
      }
    });
  }
  if (pm->mesh_bcs[BoundaryFace::inner_x2] == BoundaryFlag::outflow
      || pm->mesh_bcs[BoundaryFace::inner_x2] == BoundaryFlag::diode
      || pm->mesh_bcs[BoundaryFace::inner_x2] == BoundaryFlag::vacuum
      || pm->mesh_bcs[BoundaryFace::inner_x2] == BoundaryFlag::user
      || pm->mesh_bcs[BoundaryFace::outer_x2] == BoundaryFlag::outflow
      || pm->mesh_bcs[BoundaryFace::outer_x2] == BoundaryFlag::diode
      || pm->mesh_bcs[BoundaryFace::outer_x2] == BoundaryFlag::vacuum
      || pm->mesh_bcs[BoundaryFace::outer_x2] == BoundaryFlag::user) {
    const auto boundary_work = physical_boundary_x2_work.d_view;
    const int work_count = compact_work ? nphysical_boundary_x2_work : nmb;
    par_for("z4crhs_bc_x2", DevExeSpace(), 0, (work_count-1), ks, ke, is, ie,
    KOKKOS_LAMBDA(int row, int k, int i) {
      const int m = compact_work ? boundary_work(row) : row;
      // Inner boundary
      switch(mb_bcs.d_view(m,BoundaryFace::inner_x2)) {
        case BoundaryFlag::diode:
        case BoundaryFlag::vacuum:
        case BoundaryFlag::outflow:
            Z4cSommerfeldConfigured<Centering, Symmetry>(
                z4c_, rhs_, layout, size, bcs, user_Sbc, fd_stencil, m, k, js, i);
          break;
        case BoundaryFlag::user:
            if (user_Sbc) {
              Z4cSommerfeldConfigured<Centering, Symmetry>(
                  z4c_, rhs_, layout, size, bcs, user_Sbc, fd_stencil, m, k, js, i);
            }
          break;
        default:
          break;
      }
      // Outer boundary
      switch (mb_bcs.d_view(m,BoundaryFace::outer_x2)) {
        case BoundaryFlag::diode:
        case BoundaryFlag::vacuum:
        case BoundaryFlag::outflow:
            Z4cSommerfeldConfigured<Centering, Symmetry>(
                z4c_, rhs_, layout, size, bcs, user_Sbc, fd_stencil, m, k, je, i);
          break;
        case BoundaryFlag::user:
            if (user_Sbc) {
              Z4cSommerfeldConfigured<Centering, Symmetry>(
                  z4c_, rhs_, layout, size, bcs, user_Sbc, fd_stencil, m, k, je, i);
            }
          break;
        default:
          break;
      }
    });
  }
  if constexpr (std::is_same_v<Symmetry, CartoonSO2>) {
    return TaskStatus::complete;
  }
  if (pm->mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::outflow
      || pm->mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::diode
      || pm->mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::vacuum
      || pm->mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::user
      || pm->mesh_bcs[BoundaryFace::outer_x3] == BoundaryFlag::outflow
      || pm->mesh_bcs[BoundaryFace::outer_x3] == BoundaryFlag::diode
      || pm->mesh_bcs[BoundaryFace::outer_x3] == BoundaryFlag::vacuum
      || pm->mesh_bcs[BoundaryFace::outer_x3] == BoundaryFlag::user) {
    const auto boundary_work = physical_boundary_x3_work.d_view;
    const int work_count = compact_work ? nphysical_boundary_x3_work : nmb;
    par_for("z4crhs_bc_x3", DevExeSpace(), 0, (work_count-1), js, je, is, ie,
    KOKKOS_LAMBDA(int row, int j, int i) {
      const int m = compact_work ? boundary_work(row) : row;
      // Inner boundary
      switch(mb_bcs.d_view(m,BoundaryFace::inner_x3)) {
        case BoundaryFlag::diode:
        case BoundaryFlag::vacuum:
        case BoundaryFlag::outflow:
            Z4cSommerfeldConfigured<Centering, Symmetry>(
                z4c_, rhs_, layout, size, bcs, user_Sbc, fd_stencil, m, ks, j, i);
          break;
        case BoundaryFlag::user:
            if (user_Sbc) {
              Z4cSommerfeldConfigured<Centering, Symmetry>(
                  z4c_, rhs_, layout, size, bcs, user_Sbc, fd_stencil, m, ks, j, i);
            }
          break;
        default:
          break;
      }
      // Outer boundary
      switch (mb_bcs.d_view(m,BoundaryFace::outer_x3)) {
        case BoundaryFlag::diode:
        case BoundaryFlag::vacuum:
        case BoundaryFlag::outflow:
            Z4cSommerfeldConfigured<Centering, Symmetry>(
                z4c_, rhs_, layout, size, bcs, user_Sbc, fd_stencil, m, ke, j, i);
          break;
        case BoundaryFlag::user:
            if (user_Sbc) {
              Z4cSommerfeldConfigured<Centering, Symmetry>(
                  z4c_, rhs_, layout, size, bcs, user_Sbc, fd_stencil, m, ke, j, i);
            }
          break;
        default:
          break;
      }
    });
  }


  return TaskStatus::complete;
}

template <typename Centering, typename Symmetry>
TaskStatus Z4c::Z4cFullConstraintBjorhusRHSImpl(Driver *pdriver, int stage) {
  (void)pdriver;
  // Driver initialization invokes the boundary task once before a volume RHS exists.
  if (stage <= 0) return TaskStatus::complete;

  const auto layout_ = layout;
  const int nmb = pmy_pack->nmb_thispack;
  const int is = layout_.is, ie = layout_.ie;
  const int js = layout_.js, je = layout_.je;
  const int ks = layout_.ks, ke = layout_.ke;
  const auto &size = pmy_pack->pmb->mb_size;
  const auto bcs = pmy_pack->pmb->mb_bcs.d_view;
  const bool user_sbc = opt.user_Sbc;
  const int fd_stencil = opt.fd_stencil;
  const auto &z4c_ = z4c;
  const auto &rhs_ = rhs;

  // Orientation ownership is disjoint: x1 owns x1-x2/x1-x3 corners, x2 owns the
  // remaining x2-x3 corners, and x3 owns only points not incident on x1/x2.  Each
  // owning thread constructs one composite normal from every incident physical face.
  par_for("z4c full constraint Bjorhus x1", DevExeSpace(), 0, nmb - 1, ks, ke,
          js, je,
      KOKKOS_LAMBDA(const int m, const int k, const int j) {
        ApplyFullConstraintBjorhusConfigured<Centering, Symmetry>(
            z4c_, rhs_, layout_, size, bcs, user_sbc, fd_stencil, 0, m, k, j,
            is);
        if (ie != is) {
          ApplyFullConstraintBjorhusConfigured<Centering, Symmetry>(
              z4c_, rhs_, layout_, size, bcs, user_sbc, fd_stencil, 0, m, k, j,
              ie);
        }
      });

  if (layout_.nx2 > 1) {
    par_for("z4c full constraint Bjorhus x2", DevExeSpace(), 0, nmb - 1, ks,
            ke, is, ie,
        KOKKOS_LAMBDA(const int m, const int k, const int i) {
          ApplyFullConstraintBjorhusConfigured<Centering, Symmetry>(
              z4c_, rhs_, layout_, size, bcs, user_sbc, fd_stencil, 1, m, k,
              js, i);
          if (je != js) {
            ApplyFullConstraintBjorhusConfigured<Centering, Symmetry>(
                z4c_, rhs_, layout_, size, bcs, user_sbc, fd_stencil, 1, m, k,
                je, i);
          }
        });
  }

  if constexpr (std::is_same_v<Symmetry, Cartesian3D>) {
    if (layout_.nx3 > 1) {
      par_for("z4c full constraint Bjorhus x3", DevExeSpace(), 0, nmb - 1, js,
              je, is, ie,
          KOKKOS_LAMBDA(const int m, const int j, const int i) {
            ApplyFullConstraintBjorhusConfigured<Centering, Symmetry>(
                z4c_, rhs_, layout_, size, bcs, user_sbc, fd_stencil, 2, m,
                ks, j, i);
            if (ke != ks) {
              ApplyFullConstraintBjorhusConfigured<Centering, Symmetry>(
                  z4c_, rhs_, layout_, size, bcs, user_sbc, fd_stencil, 2, m,
                  ke, j, i);
            }
          });
    }
  }

  return TaskStatus::complete;
}

TaskStatus Z4c::Z4cBoundaryRHS(Driver *pdriver, int stage) {
  TaskStatus status;
  if (pmy_pack->z4c_symmetry.mode == Z4cSymmetryMode::cartoon_so2) {
    if (opt.boundary_rhs_mode == Z4cBoundaryRHSMode::full_constraint_bjorhus) {
      status = layout.centering == Z4cGridCentering::vertex
                   ? Z4cFullConstraintBjorhusRHSImpl<VertexCenteredZ4c,
                                                     CartoonSO2>(pdriver, stage)
                   : Z4cFullConstraintBjorhusRHSImpl<CellCenteredZ4c,
                                                     CartoonSO2>(pdriver, stage);
    } else {
      status = layout.centering == Z4cGridCentering::vertex
                   ? Z4cBoundaryRHSImpl<VertexCenteredZ4c, CartoonSO2>(pdriver,
                                                                      stage)
                   : Z4cBoundaryRHSImpl<CellCenteredZ4c, CartoonSO2>(pdriver,
                                                                    stage);
    }
  } else {
    if (opt.boundary_rhs_mode == Z4cBoundaryRHSMode::full_constraint_bjorhus) {
      status = layout.centering == Z4cGridCentering::vertex
                   ? Z4cFullConstraintBjorhusRHSImpl<VertexCenteredZ4c,
                                                     Cartesian3D>(pdriver, stage)
                   : Z4cFullConstraintBjorhusRHSImpl<CellCenteredZ4c,
                                                     Cartesian3D>(pdriver, stage);
    } else {
      status = layout.centering == Z4cGridCentering::vertex
                   ? Z4cBoundaryRHSImpl<VertexCenteredZ4c, Cartesian3D>(pdriver,
                                                                       stage)
                   : Z4cBoundaryRHSImpl<CellCenteredZ4c, Cartesian3D>(pdriver,
                                                                     stage);
    }
  }
  if (status == TaskStatus::complete && chi_parent_provenance != nullptr) {
    chi_parent_provenance->AnalyzePreUpdate(pdriver, stage);
  }
  if (status == TaskStatus::complete &&
      layout.centering == Z4cGridCentering::vertex &&
      std::getenv("ATHENA_Z4C_VC_RHS_SYNC_DIAGNOSTIC") != nullptr) {
    // Observe the canonical shared-node RHS mismatch without changing the production
    // update.  Synchronization is applied only to this disposable diagnostic copy.
    DvceArray5D<Real> rhs_copy(
        "native VC shared RHS diagnostic", u_rhs.extent_int(0),
        u_rhs.extent_int(1), u_rhs.extent_int(2), u_rhs.extent_int(3),
        u_rhs.extent_int(4));
    Kokkos::deep_copy(rhs_copy, u_rhs);
    vertex_topology_plan->SynchronizeSharedNodes(
        rhs_copy, "ATHENA_Z4C_VC_RHS_SYNC_DIAGNOSTIC");
  }
  if (status == TaskStatus::complete &&
      layout.centering == Z4cGridCentering::vertex) {
    DumpVertexRHSFieldDiagnostic(
        this, pmy_pack, pdriver, stage,
        std::getenv("ATHENA_Z4C_VC_RHS_FIELD_DIAGNOSTIC"));
  }
  return status;
}

template TaskStatus Z4c::Z4cBoundaryRHSImpl<CellCenteredZ4c, Cartesian3D>(Driver *, int);
template TaskStatus Z4c::Z4cBoundaryRHSImpl<CellCenteredZ4c, CartoonSO2>(Driver *, int);
template TaskStatus Z4c::Z4cBoundaryRHSImpl<VertexCenteredZ4c, Cartesian3D>(Driver *, int);
template TaskStatus Z4c::Z4cBoundaryRHSImpl<VertexCenteredZ4c, CartoonSO2>(Driver *, int);
template TaskStatus
Z4c::Z4cFullConstraintBjorhusRHSImpl<CellCenteredZ4c, Cartesian3D>(Driver *, int);
template TaskStatus
Z4c::Z4cFullConstraintBjorhusRHSImpl<CellCenteredZ4c, CartoonSO2>(Driver *, int);
template TaskStatus
Z4c::Z4cFullConstraintBjorhusRHSImpl<VertexCenteredZ4c, Cartesian3D>(Driver *, int);
template TaskStatus
Z4c::Z4cFullConstraintBjorhusRHSImpl<VertexCenteredZ4c, CartoonSO2>(Driver *, int);

} // end namespace z4c
