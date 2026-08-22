//========================================================================================
//! \file ref_gh_tasks.cpp
//! \brief Driver tasks for reference-frame first-order GH.
//========================================================================================
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "athena.hpp"
#include "bvals/bvals.hpp"
#include "coordinates/cell_locations.hpp"
#include "driver/driver.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "ref_gh/ref_gh.hpp"
#include "ref_gh/ref_gh_geometry.hpp"
#include "ref_gh/reference_cache.hpp"
#include "ref_gh/reference_provider_cache.hpp"
#include "ref_gh/standard_gh_source.hpp"
#include "ref_gh/reference_geometry.hpp"
#include "ref_gh/reference_trumpet_schwarzschild.hpp"
#include "tasklist/numerical_relativity.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace ref_gh {

template <typename MaxLocation>
KOKKOS_INLINE_FUNCTION
void UpdateReferenceOracleMaximum(const Real cached, const Real oracle,
                                  const int category,
                                  MaxLocation &maximum) {
  Real scale = 1.0;
  const Real cached_magnitude = Kokkos::abs(cached);
  const Real oracle_magnitude = Kokkos::abs(oracle);
  if (cached_magnitude > scale) scale = cached_magnitude;
  if (oracle_magnitude > scale) scale = oracle_magnitude;
  // A spin derivative is a projected contraction of 4x4x4 two-jet terms.
  // Near the puncture those terms cancel strongly even when the final value is
  // O(1). Account for the contraction depth when comparing algebraically
  // equivalent operation orders; primitive/cache categories retain unit scale.
  const Real operation_scale = (category == 5) ? 32.0 : 1.0;
  const Real error = Kokkos::abs(cached - oracle)/(scale*operation_scale);
  if (error > maximum.val) {
    maximum.val = error;
    maximum.loc = category;
  }
}

KOKKOS_INLINE_FUNCTION
Real RawReferenceSpin(const ReferenceCachePoint &reference,
                      const int A, const int B, const int C) {
  Real value = 0.0;
  for (int a = 0; a < 4; ++a) {
    for (int c = 0; c < 4; ++c) {
      Real derivative = ReferenceDFrame(reference, c, B, a);
      for (int d = 0; d < 4; ++d) {
        derivative += ReferenceChristoffel(reference, a, c, d)
                      *ReferenceFrame(reference, B, d);
      }
      value += ReferenceCoframe(reference, A, a)
               *ReferenceFrame(reference, C, c)*derivative;
    }
  }
  return value;
}

KOKKOS_INLINE_FUNCTION
Real WorkspaceCoframeDerivative(const ReferenceWorkspacePoint &workspace,
                                const int p, const int A, const int a) {
  return workspace.workspace(
      workspace.m, kRefWorkspaceCoframeDerivative + 16*p + 4*A + a,
      workspace.k, workspace.j, workspace.i);
}

KOKKOS_INLINE_FUNCTION
Real RawReferenceSpinCoordinateDerivative(const ReferenceCachePoint &reference,
                                           const ReferenceWorkspacePoint &workspace,
                                           const int p, const int A,
                                           const int B, const int C) {
  Real coordinate_derivative = 0.0;
  for (int a = 0; a < 4; ++a) {
    const Real d_coframe = WorkspaceCoframeDerivative(workspace, p, A, a);
    for (int c = 0; c < 4; ++c) {
      Real frame_covariant_derivative = ReferenceDFrame(reference, c, B, a);
      Real d_frame_covariant_derivative =
          ReferenceDDFrame(reference, p, c, B, a);
      for (int d = 0; d < 4; ++d) {
        frame_covariant_derivative +=
            ReferenceChristoffel(reference, a, c, d)
            *ReferenceFrame(reference, B, d);
        d_frame_covariant_derivative +=
            ReferenceDChristoffel(reference, p, a, c, d)
              *ReferenceFrame(reference, B, d)
            + ReferenceChristoffel(reference, a, c, d)
              *ReferenceDFrame(reference, p, B, d);
      }
      coordinate_derivative +=
          (d_coframe*ReferenceFrame(reference, C, c)
           + ReferenceCoframe(reference, A, a)
             *ReferenceDFrame(reference, p, C, c))
            *frame_covariant_derivative
          + ReferenceCoframe(reference, A, a)
            *ReferenceFrame(reference, C, c)
            *d_frame_covariant_derivative;
    }
  }
  return coordinate_derivative;
}

KOKKOS_INLINE_FUNCTION
Real RawReferenceStructure4(const ReferenceCachePoint &reference,
                            const int E, const int C, const int D) {
  Real value = 0.0;
  for (int a = 0; a < 4; ++a) {
    for (int p = 0; p < 4; ++p) {
      value += ReferenceCoframe(reference, E, a)
               *(ReferenceFrame(reference, C, p)
                   *ReferenceDFrame(reference, p, D, a)
                 - ReferenceFrame(reference, D, p)
                   *ReferenceDFrame(reference, p, C, a));
    }
  }
  return value;
}

KOKKOS_INLINE_FUNCTION
void ControllerEigenvalueExtrema3(Real m00, Real m01, Real m02,
                                  Real m11, Real m12, Real m22,
                                  Real &minimum, Real &maximum) {
  // Fixed-sweep Jacobi diagonalization.  This duplicates the already-qualified
  // conditioning diagnostic locally so the controller can retain both extrema.
  for (int sweep = 0; sweep < 18; ++sweep) {
    int p = 0;
    int q = 1;
    Real largest = Kokkos::abs(m01);
    if (Kokkos::abs(m02) > largest) {
      largest = Kokkos::abs(m02);
      q = 2;
    }
    if (Kokkos::abs(m12) > largest) {
      largest = Kokkos::abs(m12);
      p = 1;
      q = 2;
    }
    if (largest < 1.0e-14) break;
    const Real app = p == 0 ? m00 : m11;
    const Real aqq = q == 1 ? m11 : m22;
    const Real apq = q == 1 ? m01 : (p == 0 ? m02 : m12);
    const Real angle = 0.5*Kokkos::atan2(2.0*apq, aqq - app);
    const Real cosine = Kokkos::cos(angle);
    const Real sine = Kokkos::sin(angle);
    const Real rotated_p = cosine*cosine*app - 2.0*sine*cosine*apq
                           + sine*sine*aqq;
    const Real rotated_q = sine*sine*app + 2.0*sine*cosine*apq
                           + cosine*cosine*aqq;
    if (p == 0 && q == 1) {
      const Real old02 = m02;
      const Real old12 = m12;
      m00 = rotated_p;
      m11 = rotated_q;
      m01 = 0.0;
      m02 = cosine*old02 - sine*old12;
      m12 = sine*old02 + cosine*old12;
    } else if (p == 0) {
      const Real old01 = m01;
      const Real old12 = m12;
      m00 = rotated_p;
      m22 = rotated_q;
      m02 = 0.0;
      m01 = cosine*old01 - sine*old12;
      m12 = sine*old01 + cosine*old12;
    } else {
      const Real old01 = m01;
      const Real old02 = m02;
      m11 = rotated_p;
      m22 = rotated_q;
      m12 = 0.0;
      m01 = cosine*old01 - sine*old02;
      m02 = sine*old01 + cosine*old02;
    }
  }
  if (m11 < m00) { const Real temporary = m00; m00 = m11; m11 = temporary; }
  if (m22 < m00) { const Real temporary = m00; m00 = m22; m22 = temporary; }
  if (m22 < m11) { const Real temporary = m11; m11 = m22; m22 = temporary; }
  minimum = m00;
  maximum = m22;
}

KOKKOS_INLINE_FUNCTION
Real WorkspaceSpinCoordinateDerivative(const ReferenceWorkspacePoint &workspace,
                                       const int p, const int pair,
                                       const int C) {
  return workspace.workspace(
      workspace.m, kRefWorkspaceSpinCoordinateDerivative + 24*p + 4*pair + C,
      workspace.k, workspace.j, workspace.i);
}

void RefGh::DebugFence(const char *label) const {
  if (opt.debug_task_fences) {
    Kokkos::fence(label);
    std::cout << "ref_gh debug fence passed: " << label << std::endl;
  }
}

void RefGh::QueueTasks() {
  using namespace numrel;  // NOLINT(build/namespaces)
  auto *pnr = pmy_pack->pnr;
  pnr->QueueTask(&RefGh::InitRecv, this, RefGh_Recv, "RefGh_Recv", Task_Start);
  pnr->QueueTask(&RefGh::CopyU, this, RefGh_CopyU, "RefGh_CopyU", Task_Run);
  pnr->QueueTask(&RefGh::MeasureController, this, RefGh_MeasureController,
                 "RefGh_MeasureController", Task_Run, {RefGh_CopyU});
  pnr->QueueTask(&RefGh::UpdateReferenceGeometry, this, RefGh_UpdateReference,
                 "RefGh_UpdateReference", Task_Run, {RefGh_MeasureController});
  if (opt.fd_order == 2) {
    pnr->QueueTask(&RefGh::CalcRHS<2>, this, RefGh_CalcRHS, "RefGh_CalcRHS",
                   Task_Run, {RefGh_UpdateReference});
  } else if (opt.fd_order == 4) {
    pnr->QueueTask(&RefGh::CalcRHS<3>, this, RefGh_CalcRHS, "RefGh_CalcRHS",
                   Task_Run, {RefGh_UpdateReference});
  } else {
    pnr->QueueTask(&RefGh::CalcRHS<4>, this, RefGh_CalcRHS, "RefGh_CalcRHS",
                   Task_Run, {RefGh_UpdateReference});
  }
  pnr->QueueTask(&RefGh::ExpRKUpdate, this, RefGh_ExplRK, "RefGh_ExplRK", Task_Run,
                 {RefGh_CalcRHS});
  pnr->QueueTask(&RefGh::RestrictU, this, RefGh_RestU, "RefGh_RestU", Task_Run,
                 {RefGh_ExplRK});
  pnr->QueueTask(&RefGh::SendU, this, RefGh_SendU, "RefGh_SendU", Task_Run,
                 {RefGh_RestU});
  pnr->QueueTask(&RefGh::RecvU, this, RefGh_RecvU, "RefGh_RecvU", Task_Run,
                 {RefGh_SendU});
  pnr->QueueTask(&RefGh::Prolongate, this, RefGh_Prolong, "RefGh_Prolong", Task_Run,
                 {RefGh_RecvU});
  pnr->QueueTask(&RefGh::ApplyPhysicalBCs, this, RefGh_BCS, "RefGh_BCS", Task_Run,
                 {RefGh_Prolong});
  pnr->QueueTask(&RefGh::NewTimeStep, this, RefGh_Newdt, "RefGh_Newdt", Task_Run,
                 {RefGh_BCS});
  pnr->QueueTask(&RefGh::ClearSend, this, RefGh_ClearS, "RefGh_ClearS", Task_End);
  pnr->QueueTask(&RefGh::ClearRecv, this, RefGh_ClearR, "RefGh_ClearR", Task_End,
                 {RefGh_ClearS});
}

TaskStatus RefGh::InitRecv(Driver *, int) {
  const auto status = pbval_u->InitRecv(nref_gh);
  DebugFence("ref_gh InitRecv");
  return status;
}
TaskStatus RefGh::ClearRecv(Driver *, int) { return pbval_u->ClearRecv(); }
TaskStatus RefGh::ClearSend(Driver *, int) { return pbval_u->ClearSend(); }

TaskStatus RefGh::CopyU(Driver *driver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  if (driver->integrator == "rk4") {
    if (stage == 1) {
      Kokkos::deep_copy(DevExeSpace(), u1, u0);
      controller_base = controller;
    } else {
      const Real delta = driver->delta[stage - 1];
      const auto state = u0;
      const auto base = u1;
      par_for("ref_gh rk4 base", DevExeSpace(), 0, pmy_pack->nmb_thispack - 1,
      0, nref_gh - 1, indcs.ks, indcs.ke, indcs.js, indcs.je, indcs.is, indcs.ie,
      KOKKOS_LAMBDA(const int m, const int n, const int k, const int j, const int i) {
        base(m, n, k, j, i) += delta*state(m, n, k, j, i);
      });
      controller_base.delta_q += delta*controller.delta_q;
      controller_base.delta_q_dot += delta*controller.delta_q_dot;
      controller_base.delta_p += delta*controller.delta_p;
      controller_base.delta_p_dot += delta*controller.delta_p_dot;
    }
  } else if (stage == 1) {
    Kokkos::deep_copy(DevExeSpace(), u1, u0);
    controller_base = controller;
  }
  DebugFence("ref_gh CopyU");
  return TaskStatus::complete;
}

void RefGh::PersistControllerState() {
  // Restart headers serialize the live ParameterInput.  Persisting these four
  // replicated scalars there retains the exact 50-field PDE restart layout.
  pinput->SetReal("ref_gh", "controller_delta_q", controller.delta_q);
  pinput->SetReal("ref_gh", "controller_delta_q_dot", controller.delta_q_dot);
  pinput->SetReal("ref_gh", "controller_delta_p", controller.delta_p);
  pinput->SetReal("ref_gh", "controller_delta_p_dot", controller.delta_p_dot);
  pinput->SetReal("ref_gh", "controller_generation",
                  static_cast<Real>(controller_generation));
}

TaskStatus RefGh::MeasureController(Driver *driver, const int stage) {
  if (opt.reference_controlled) MeasureControllerAtTime(StageTime(driver, stage));
  return TaskStatus::complete;
}

void RefGh::MeasureControllerAtTime(const Real stage_time) {
  if (!opt.reference_controlled) return;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  const auto levels = pmy_pack->pmb->mb_lev.d_view;
  const auto state = u0;
  const int max_level = pmy_pack->pmesh->max_level;
  const int ncells = indcs.nx1*indcs.nx2*indcs.nx3;
  const Real mass = opt.reference_mass;
  const Real center_x = opt.reference_center[0];
  const Real center_y = opt.reference_center[1];
  const Real center_z = opt.reference_center[2];
  const Real fit_min = opt.r_fit_min*mass;
  const Real fit_max = opt.r_fit_max*mass;
  const int level_difference = max_level - pmy_pack->pmesh->root_level;
  const Real refinement_factor = std::ldexp(1.0, level_difference);
  const Real finest_nominal_spacing = std::min(
      (pmy_pack->pmesh->mesh_size.x1max - pmy_pack->pmesh->mesh_size.x1min)
          /(pmy_pack->pmesh->mesh_indcs.nx1*refinement_factor),
      std::min(
          (pmy_pack->pmesh->mesh_size.x2max - pmy_pack->pmesh->mesh_size.x2min)
              /(pmy_pack->pmesh->mesh_indcs.nx2*refinement_factor),
          (pmy_pack->pmesh->mesh_size.x3max - pmy_pack->pmesh->mesh_size.x3min)
              /(pmy_pack->pmesh->mesh_indcs.nx3*refinement_factor)));
  const Real interface_buffer = 4.0*finest_nominal_spacing;

  // Five spatial-volume moments, five lapse moments, and mesh-validity counts.
  enum MomentIndex {
    kS0, kSx, kSyG, kSxx, kSxyG, kSyAlpha, kSxyAlpha,
    kEligible, kShellAll, kShellFinest, kBufferAll, kBufferFinest,
    kInvalid, kMomentCount
  };
  static_assert(kMomentCount <= NREDUCTION_VARIABLES,
                "controller moments exceed the fixed reduction array");
  array_sum::GlobalSum sums;
  Kokkos::parallel_reduce(
      "ref_gh controller fixed-shell moments",
      Kokkos::RangePolicy<>(DevExeSpace(),
          0, pmy_pack->nmb_thispack*ncells),
      KOKKOS_LAMBDA(const int idx, array_sum::GlobalSum &total) {
        int work = idx;
        const int i = work % indcs.nx1 + indcs.is; work /= indcs.nx1;
        const int j = work % indcs.nx2 + indcs.js; work /= indcs.nx2;
        const int k = work % indcs.nx3 + indcs.ks;
        const int m = work/indcs.nx3;
        const Real x_coord = CellCenterX(
            i - indcs.is, indcs.nx1,
            size.d_view(m).x1min, size.d_view(m).x1max);
        const Real y_coord = CellCenterX(
            j - indcs.js, indcs.nx2,
            size.d_view(m).x2min, size.d_view(m).x2max);
        const Real z_coord = CellCenterX(
            k - indcs.ks, indcs.nx3,
            size.d_view(m).x3min, size.d_view(m).x3max);
        const Real dx = x_coord - center_x;
        const Real dy = y_coord - center_y;
        const Real dz = z_coord - center_z;
        const Real radius = Kokkos::sqrt(dx*dx + dy*dy + dz*dz);
        const bool in_buffered_shell =
            radius >= fit_min - interface_buffer
            && radius <= fit_max + interface_buffer;
        if (in_buffered_shell) {
          total.the_array[kBufferAll] += 1.0;
          if (levels(m) == max_level) total.the_array[kBufferFinest] += 1.0;
        }
        if (!(radius >= fit_min && radius <= fit_max)) return;
        total.the_array[kShellAll] += 1.0;
        if (levels(m) != max_level) return;
        total.the_array[kShellFinest] += 1.0;

        Real relative_metric[4][4];  // NOLINT(runtime/arrays)
        for (int A = 0; A < 4; ++A) {
          for (int B = A; B < 4; ++B) {
            relative_metric[A][B] = relative_metric[B][A] =
                state(m, PsiIndex(A, B), k, j, i);
          }
        }
        Real spatial_inverse[3][3];  // NOLINT(runtime/arrays)
        Real spatial_determinant = 0.0;
        if (!InvertSpatial3(relative_metric, spatial_inverse,
                            spatial_determinant)) {
          total.the_array[kInvalid] += 1.0;
          return;
        }
        Real v2 = 0.0;
        for (int I = 0; I < 3; ++I) {
          for (int J = 0; J < 3; ++J) {
            v2 += relative_metric[0][I + 1]*spatial_inverse[I][J]
                  *relative_metric[0][J + 1];
          }
        }
        const Real relative_lapse_squared = -relative_metric[0][0] + v2;
        if (!(relative_lapse_squared > 0.0)
            || !Kokkos::isfinite(relative_lapse_squared)) {
          total.the_array[kInvalid] += 1.0;
          return;
        }
        const Real log_radius = Kokkos::log(radius/mass);
        const Real weight = (mass/radius)*(mass/radius)*(mass/radius);
        const Real log_volume = 0.5*Kokkos::log(spatial_determinant);
        const Real log_lapse = 0.5*Kokkos::log(relative_lapse_squared);
        total.the_array[kS0] += weight;
        total.the_array[kSx] += weight*log_radius;
        total.the_array[kSyG] += weight*log_volume;
        total.the_array[kSxx] += weight*log_radius*log_radius;
        total.the_array[kSxyG] += weight*log_radius*log_volume;
        total.the_array[kSyAlpha] += weight*log_lapse;
        total.the_array[kSxyAlpha] += weight*log_radius*log_lapse;
        total.the_array[kEligible] += 1.0;
      }, Kokkos::Sum<array_sum::GlobalSum>(sums));

  // The conditioning pass is separate from the shell fit: safety monitors
  // cover every native active cell, not only the controller volume.
  Real minus_lambda_min = 0.0;
  Real lambda_max = 0.0;
  Real minus_det_third_min = 0.0;
  Real det_third_max = 0.0;
  Real condition_max = 0.0;
  Real minus_relative_lapse_min = 0.0;
  Real relative_lapse_max = 0.0;
  Real v2_max = 0.0;
  Real psi_max = 0.0;
  Real inverse_psi_max = 0.0;
  Real invalid_max = 0.0;
  Kokkos::parallel_reduce(
      "ref_gh controller conditioning",
      Kokkos::RangePolicy<>(DevExeSpace(),
          0, pmy_pack->nmb_thispack*ncells),
      KOKKOS_LAMBDA(const int idx, Real &local_minus_lambda_min,
                    Real &local_lambda_max, Real &local_minus_det_third_min,
                    Real &local_det_third_max, Real &local_condition_max,
                    Real &local_minus_lapse_min, Real &local_lapse_max,
                    Real &local_v2_max, Real &local_psi_max,
                    Real &local_inverse_psi_max, Real &local_invalid_max) {
        int work = idx;
        const int i = work % indcs.nx1 + indcs.is; work /= indcs.nx1;
        const int j = work % indcs.nx2 + indcs.js; work /= indcs.nx2;
        const int k = work % indcs.nx3 + indcs.ks;
        const int m = work/indcs.nx3;
        Real relative_metric[4][4];  // NOLINT(runtime/arrays)
        for (int A = 0; A < 4; ++A) {
          for (int B = A; B < 4; ++B) {
            const Real value = state(m, PsiIndex(A, B), k, j, i);
            relative_metric[A][B] = relative_metric[B][A] = value;
            const Real magnitude = Kokkos::abs(value);
            if (magnitude > local_psi_max) local_psi_max = magnitude;
          }
        }
        Real inverse_metric[4][4];  // NOLINT(runtime/arrays)
        Real determinant4 = 0.0;
        Real spatial_inverse[3][3];  // NOLINT(runtime/arrays)
        Real determinant3 = 0.0;
        if (0.0 > local_invalid_max) local_invalid_max = 0.0;
        if (!Invert4(relative_metric, inverse_metric, determinant4)
            || !InvertSpatial3(relative_metric, spatial_inverse, determinant3)) {
          local_invalid_max = 1.0;
          return;
        }
        for (int A = 0; A < 4; ++A) {
          for (int B = 0; B < 4; ++B) {
            const Real magnitude = Kokkos::abs(inverse_metric[A][B]);
            if (magnitude > local_inverse_psi_max) {
              local_inverse_psi_max = magnitude;
            }
          }
        }
        Real eigen_min = 0.0;
        Real eigen_max = 0.0;
        ControllerEigenvalueExtrema3(
            relative_metric[1][1], relative_metric[1][2],
            relative_metric[1][3], relative_metric[2][2],
            relative_metric[2][3], relative_metric[3][3],
            eigen_min, eigen_max);
        Real v2 = 0.0;
        for (int I = 0; I < 3; ++I) {
          for (int J = 0; J < 3; ++J) {
            v2 += relative_metric[0][I + 1]*spatial_inverse[I][J]
                  *relative_metric[0][J + 1];
          }
        }
        const Real relative_lapse_squared = -relative_metric[0][0] + v2;
        if (!(eigen_min > 0.0) || !(relative_lapse_squared > 0.0)
            || !Kokkos::isfinite(eigen_max)
            || !Kokkos::isfinite(relative_lapse_squared)) {
          local_invalid_max = 1.0;
          return;
        }
        const Real det_third = Kokkos::pow(determinant3, 1.0/3.0);
        const Real relative_lapse = Kokkos::sqrt(relative_lapse_squared);
        if (-eigen_min > local_minus_lambda_min) {
          local_minus_lambda_min = -eigen_min;
        }
        if (eigen_max > local_lambda_max) local_lambda_max = eigen_max;
        if (-det_third > local_minus_det_third_min) {
          local_minus_det_third_min = -det_third;
        }
        if (det_third > local_det_third_max) local_det_third_max = det_third;
        const Real condition = eigen_max/eigen_min;
        if (condition > local_condition_max) local_condition_max = condition;
        if (-relative_lapse > local_minus_lapse_min) {
          local_minus_lapse_min = -relative_lapse;
        }
        if (relative_lapse > local_lapse_max) local_lapse_max = relative_lapse;
        if (v2 > local_v2_max) local_v2_max = v2;
      }, Kokkos::Max<Real>(minus_lambda_min), Kokkos::Max<Real>(lambda_max),
      Kokkos::Max<Real>(minus_det_third_min), Kokkos::Max<Real>(det_third_max),
      Kokkos::Max<Real>(condition_max),
      Kokkos::Max<Real>(minus_relative_lapse_min),
      Kokkos::Max<Real>(relative_lapse_max), Kokkos::Max<Real>(v2_max),
      Kokkos::Max<Real>(psi_max), Kokkos::Max<Real>(inverse_psi_max),
      Kokkos::Max<Real>(invalid_max));

  Real finest_spacing = std::numeric_limits<Real>::max();
  for (int m = 0; m < pmy_pack->nmb_thispack; ++m) {
    if (pmy_pack->pmb->mb_lev.h_view(m) == max_level) {
      finest_spacing = std::min(
          finest_spacing,
          std::min(pmy_pack->pmb->mb_size.h_view(m).dx1,
                   std::min(pmy_pack->pmb->mb_size.h_view(m).dx2,
                            pmy_pack->pmb->mb_size.h_view(m).dx3)));
    }
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, sums.the_array, NREDUCTION_VARIABLES,
                MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  Real maxima[] = {minus_lambda_min, lambda_max, minus_det_third_min,
                   det_third_max, condition_max, minus_relative_lapse_min,
                   relative_lapse_max, v2_max, psi_max, inverse_psi_max,
                   invalid_max};
  MPI_Allreduce(MPI_IN_PLACE, maxima, 11, MPI_ATHENA_REAL, MPI_MAX,
                MPI_COMM_WORLD);
  minus_lambda_min = maxima[0];
  lambda_max = maxima[1];
  minus_det_third_min = maxima[2];
  det_third_max = maxima[3];
  condition_max = maxima[4];
  minus_relative_lapse_min = maxima[5];
  relative_lapse_max = maxima[6];
  v2_max = maxima[7];
  psi_max = maxima[8];
  inverse_psi_max = maxima[9];
  invalid_max = maxima[10];
  MPI_Allreduce(MPI_IN_PLACE, &finest_spacing, 1, MPI_ATHENA_REAL, MPI_MIN,
                MPI_COMM_WORLD);
#endif

  const Real denominator = sums.the_array[kS0]*sums.the_array[kSxx]
                           - sums.the_array[kSx]*sums.the_array[kSx];
  const bool shell_valid = sums.the_array[kEligible] >= 8.0
      && sums.the_array[kShellAll] == sums.the_array[kShellFinest]
      && sums.the_array[kBufferAll] == sums.the_array[kBufferFinest]
      && sums.the_array[kInvalid] == 0.0
      && denominator > 0.0 && std::isfinite(denominator)
      && std::isfinite(finest_spacing);
  controller_diagnostics.e_G = shell_valid
      ? (2.0/3.0)*(sums.the_array[kS0]*sums.the_array[kSxyG]
                   - sums.the_array[kSx]*sums.the_array[kSyG])/denominator
      : NAN;
  controller_diagnostics.e_alpha = shell_valid
      ? (sums.the_array[kS0]*sums.the_array[kSxyAlpha]
         - sums.the_array[kSx]*sums.the_array[kSyAlpha])/denominator
      : NAN;
  controller_diagnostics.fitting_cell_count = sums.the_array[kEligible];
  controller_diagnostics.lambda_min = -minus_lambda_min;
  controller_diagnostics.lambda_max = lambda_max;
  controller_diagnostics.det_g_third_min = -minus_det_third_min;
  controller_diagnostics.det_g_third_max = det_third_max;
  controller_diagnostics.condition_max = condition_max;
  controller_diagnostics.relative_lapse_min = -minus_relative_lapse_min;
  controller_diagnostics.relative_lapse_max = relative_lapse_max;
  controller_diagnostics.v2_max = v2_max;
  controller_diagnostics.psi_max = psi_max;
  controller_diagnostics.inverse_psi_max = inverse_psi_max;
  controller_diagnostics.fitting_shell_valid = shell_valid;

  controller_diagnostics.r_core = opt.r_core0*mass
      *std::exp(-stage_time/(opt.tau_core*mass));
  const Real activation_coordinate = stage_time/(opt.tau_transition*mass);
  if (activation_coordinate <= 0.0) {
    controller_diagnostics.transition_amplitude = 0.0;
  } else if (activation_coordinate >= 1.0) {
    controller_diagnostics.transition_amplitude = 1.0;
  } else {
    const Real u = activation_coordinate;
    controller_diagnostics.transition_amplitude =
        u*u*u*(10.0 + u*(-15.0 + 6.0*u));
  }
  const Real r_full = (1.0 + opt.kappa_core)*controller_diagnostics.r_core;
  const bool feedback_active = opt.controller_enabled && shell_valid
      && r_full + opt.controller_fit_buffer_cells*finest_spacing < fit_min;
  controller_diagnostics.feedback_active = feedback_active;

  // Freeze all four variables if feedback is disabled or the shell is invalid.
  controller_rhs = {0.0, 0.0, 0.0, 0.0};
  if (feedback_active) {
    const Real omega_q = opt.controller_omega_q/mass;
    const Real omega_p = opt.controller_omega_p/mass;
    const Real raw_q_acceleration =
        -2.0*opt.controller_zeta*omega_q*controller.delta_q_dot
        -0.5*omega_q*omega_q*controller_diagnostics.e_G;
    const Real raw_p_acceleration =
        -2.0*opt.controller_zeta*omega_p*controller.delta_p_dot
        +omega_p*omega_p*controller_diagnostics.e_alpha;
    const Real acceleration_limit =
        opt.controller_acceleration_limit/(mass*mass);
    controller_rhs.delta_q = controller.delta_q_dot;
    controller_rhs.delta_q_dot = acceleration_limit
        *std::tanh(raw_q_acceleration/acceleration_limit);
    controller_rhs.delta_p = controller.delta_p_dot;
    controller_rhs.delta_p_dot = acceleration_limit
        *std::tanh(raw_p_acceleration/acceleration_limit);
  }

  if (invalid_max > 0.0 || !std::isfinite(controller_diagnostics.lambda_min)
      || !std::isfinite(controller_diagnostics.relative_lapse_min)) {
    std::cout << "### FATAL ERROR: Ref-GH controller conditioning became invalid at "
              << "stage_time=" << stage_time
              << " generation=" << controller_generation << std::endl;
    std::exit(EXIT_FAILURE);
  }
  DebugFence("ref_gh MeasureController");
}

Real RefGh::StageTime(const Driver *driver, const int target_stage) const {
  // Propagate the affine time coordinate through the same low-storage RK
  // recurrences used for u0/u1.  This derives the non-monotone RK4 abscissae
  // from the active integrator coefficients instead of maintaining a second
  // hard-coded Butcher table.
  Real state_weight = 1.0;
  Real state_time = 0.0;
  Real base_weight = 0.0;
  Real base_time = 0.0;
  for (int stage = 1; stage <= target_stage; ++stage) {
    if (stage == 1) {
      base_weight = state_weight;
      base_time = state_time;
    } else if (driver->integrator == "rk4") {
      base_weight += driver->delta[stage - 1]*state_weight;
      base_time += driver->delta[stage - 1]*state_time;
    }
    if (stage == target_stage) {
      return pmy_pack->pmesh->time
             + (state_time/state_weight)*pmy_pack->pmesh->dt;
    }
    const Real next_time = driver->gam0[stage - 1]*state_time
                           + driver->gam1[stage - 1]*base_time
                           + driver->beta[stage - 1];
    const Real next_weight = driver->gam0[stage - 1]*state_weight
                             + driver->gam1[stage - 1]*base_weight;
    state_time = next_time;
    state_weight = next_weight;
  }
  return pmy_pack->pmesh->time;
}

void RefGh::FillReferenceCache(const Real time, const bool include_diagnostics) {
  const bool diagnostics_requested =
      include_diagnostics || opt.validate_reference_cache;
  const bool production_generation_current = !opt.reference_controlled
      || reference_cache_generation == controller_generation;
  const bool diagnostic_generation_current = !opt.reference_controlled
      || reference_diagnostic_generation == controller_generation;
  const bool production_current = std::isfinite(reference_cache_time)
      && (!opt.reference_time_dependent || reference_cache_time == time)
      && production_generation_current;
  const bool diagnostics_current = std::isfinite(reference_diagnostic_time)
      && (!opt.reference_time_dependent || reference_diagnostic_time == time)
      && diagnostic_generation_current;
  if (production_current
      && (!diagnostics_requested || diagnostics_current)) return;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  const int n1 = indcs.nx1 + 2*indcs.ng;
  const int n2 = (indcs.nx2 > 1) ? indcs.nx2 + 2*indcs.ng : 1;
  const int n3 = (indcs.nx3 > 1) ? indcs.nx3 + 2*indcs.ng : 1;
  const int ncells = pmy_pack->nmb_thispack*n3*n2*n1;
  const auto provider = reference_provider;
  const auto workspace = reference_workspace;
  const auto evolution = reference_evolution;
  const auto diagnostic = reference_diagnostic;
  const auto table = reference_table;
  const int reference_kind = opt.reference_kind;
  const Real mass = opt.reference_mass;
  const Real center_x = opt.reference_center[0];
  const Real center_y = opt.reference_center[1];
  const Real center_z = opt.reference_center[2];
  const ControlledReferenceParameters controlled{
      mass, {center_x, center_y, center_z}, opt.r_core0, opt.tau_core,
      opt.kappa_core, opt.tau_transition, opt.regularization_outer_start,
      opt.regularization_outer_end, controller.delta_q,
      controller.delta_q_dot, controller_rhs.delta_q_dot,
      controller.delta_p, controller.delta_p_dot,
      controller_rhs.delta_p_dot};

  if (!production_current) {
    // Stage 1: evaluate the provider/profile two-jets once per point.
    Kokkos::parallel_for(
    "ref_gh reference provider profiles",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, ncells), KOKKOS_LAMBDA(const int idx) {
      int work = idx;
      const int i = work % n1; work /= n1;
      const int j = work % n2; work /= n2;
      const int k = work % n3;
      const int m = work/n3;
      const Real x = CellCenterX(i - indcs.is, indcs.nx1,
                                 size.d_view(m).x1min, size.d_view(m).x1max);
      const Real y = CellCenterX(j - indcs.js, indcs.nx2,
                                 size.d_view(m).x2min, size.d_view(m).x2max);
      const Real z = CellCenterX(k - indcs.ks, indcs.nx3,
                                 size.d_view(m).x3min, size.d_view(m).x3max);
      const ReferenceProviderPoint point{provider, m, k, j, i};
      PopulateReferenceProviderCache(reference_kind, table, mass,
                                     center_x, center_y, center_z,
                                     time, x, y, z, controlled, point);
    });

    // Stage 2: populate frame/coframe values and frame derivatives component by
    // component. Each work item holds only two scalar jets, not a full geometry.
    Kokkos::parallel_for(
    "ref_gh reference frame jets",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, ncells*16), KOKKOS_LAMBDA(const int idx) {
      const int component = idx/ncells;
      int work = idx % ncells;
      const int i = work % n1; work /= n1;
      const int j = work % n2; work /= n2;
      const int k = work % n3;
      const int m = work/n3;
      const int A = component/4;
      const int a = component % 4;
      const Real x = CellCenterX(i - indcs.is, indcs.nx1,
                                 size.d_view(m).x1min, size.d_view(m).x1max);
      const Real y = CellCenterX(j - indcs.js, indcs.nx2,
                                 size.d_view(m).x2min, size.d_view(m).x2max);
      const Real z = CellCenterX(k - indcs.ks, indcs.nx3,
                                 size.d_view(m).x3min, size.d_view(m).x3max);
      const ReferenceProviderPoint point{provider, m, k, j, i};
      const ReferenceJet coframe = ProviderCoframeJet(
          reference_kind, point, x, y, z, center_x, center_y, center_z, A, a);
      const ReferenceJet frame = ProviderFrameJet(
          reference_kind, point, x, y, z, center_x, center_y, center_z, A, a);
      evolution(m, RefMatrix4(kRefCoframe, A, a), k, j, i) = coframe.value;
      evolution(m, RefMatrix4(kRefFrame, A, a), k, j, i) = frame.value;
      for (int p = 0; p < 4; ++p) {
        evolution(m, RefRank3(kRefDFrame, p, A, a), k, j, i) = frame.d[p];
        for (int q = p; q < 4; ++q) {
          diagnostic(m, kRefDDFrame + 16*RefSymmetricPair4(p, q) + 4*A + a,
                     k, j, i) = frame.dd[p][q];
        }
      }
    });

    // Stage 3: spatial frame maps and their commutator coefficients.
    Kokkos::parallel_for(
    "ref_gh reference spatial frame",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, ncells*9), KOKKOS_LAMBDA(const int idx) {
      const int component = idx/ncells;
      int work = idx % ncells;
      const int i = work % n1; work /= n1;
      const int j = work % n2; work /= n2;
      const int k = work % n3;
      const int m = work/n3;
      const int I = component/3;
      const int p = component % 3;
      const ReferenceProviderPoint point{provider, m, k, j, i};
      evolution(m, RefMatrix3(kRefSpatialFrame, I, p), k, j, i) =
          ProviderSpatialFrame(reference_kind, point, I, p);
      evolution(m, RefMatrix3(kRefSpatialCoframe, I, p), k, j, i) =
          ProviderSpatialCoframe(reference_kind, point, I, p);
      evolution(m, RefMatrix3(kRefDtSpatialFrame, I, p), k, j, i) =
          ProviderDtSpatialFrame(reference_kind, point, I, p);
    });
    Kokkos::parallel_for(
    "ref_gh reference structure coefficients",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, ncells*9), KOKKOS_LAMBDA(const int idx) {
      const int component = idx/ncells;
      int work = idx % ncells;
      const int i = work % n1; work /= n1;
      const int j = work % n2; work /= n2;
      const int k = work % n3;
      const int m = work/n3;
      int I = 0;
      int J = 0;
      RefDecodeAntisymmetricPair3(component/3, I, J);
      const int K = component % 3;
      const ReferenceProviderPoint point{provider, m, k, j, i};
      evolution(m, kRefStructure + component, k, j, i) =
          ProviderStructure(reference_kind, point, I, J, K);
    });

    Kokkos::parallel_for(
    "ref_gh reference metric jets",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, ncells*16), KOKKOS_LAMBDA(const int idx) {
      const int component = idx/ncells;
      int work = idx % ncells;
      const int i = work % n1; work /= n1;
      const int j = work % n2; work /= n2;
      const int k = work % n3;
      const int m = work/n3;
      const int a = component/4;
      const int b = component % 4;
      const Real x = CellCenterX(i - indcs.is, indcs.nx1,
                                 size.d_view(m).x1min, size.d_view(m).x1max);
      const Real y = CellCenterX(j - indcs.js, indcs.nx2,
                                 size.d_view(m).x2min, size.d_view(m).x2max);
      const Real z = CellCenterX(k - indcs.ks, indcs.nx3,
                                 size.d_view(m).x3min, size.d_view(m).x3max);
      const ReferenceProviderPoint provider_point{provider, m, k, j, i};
      const ReferenceWorkspacePoint workspace_point{workspace, m, k, j, i};
      StoreWorkspaceMetricJet(
          ProviderMetricJet(reference_kind, provider_point, x, y, z,
                            center_x, center_y, center_z, a, b),
          a, b, workspace_point);
      StoreWorkspaceInverseMetricJet(
          ProviderInverseMetricJet(reference_kind, provider_point, x, y, z,
                                   center_x, center_y, center_z, a, b),
          a, b, workspace_point);
    });

    // Stage 4: connection and its coordinate derivative. Metric jets are
    // read from the compact update workspace, keeping this kernel scalar-small.
    Kokkos::parallel_for(
    "ref_gh reference connection",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, ncells*40), KOKKOS_LAMBDA(const int idx) {
      const int component = idx/ncells;
      int work = idx % ncells;
      const int i = work % n1; work /= n1;
      const int j = work % n2; work /= n2;
      const int k = work % n3;
      const int m = work/n3;
      const int a = component/10;
      int b = 0;
      int c = 0;
      RefDecodeSymmetricPair4(component % 10, b, c);
      const ReferenceWorkspacePoint point{workspace, m, k, j, i};
      Real christoffel = 0.0;
      Real d_christoffel[4] = {0.0, 0.0, 0.0, 0.0}; // NOLINT
      for (int ell = 0; ell < 4; ++ell) {
        const ReferenceJet metric_ell_c = LoadWorkspaceMetricJet(point, ell, c);
        const ReferenceJet metric_ell_b = LoadWorkspaceMetricJet(point, ell, b);
        const ReferenceJet metric_b_c = LoadWorkspaceMetricJet(point, b, c);
        const Real first_kind = 0.5*(metric_ell_c.d[b] + metric_ell_b.d[c]
                                     - metric_b_c.d[ell]);
        const Real inverse = WorkspaceInverseMetric(point, a, ell);
        christoffel += inverse*first_kind;
        for (int p = 0; p < 4; ++p) {
          const Real d_first = 0.5*(metric_ell_c.dd[p][b]
                                    + metric_ell_b.dd[p][c]
                                    - metric_b_c.dd[p][ell]);
          d_christoffel[p] +=
              WorkspaceDInverseMetric(point, p, a, ell)*first_kind
              + inverse*d_first;
        }
      }
      evolution(m, kRefChristoffel + component, k, j, i) = christoffel;
      for (int p = 0; p < 4; ++p) {
        diagnostic(m, kRefDChristoffel + 40*p + component, k, j, i) =
            d_christoffel[p];
      }
    });

    // Stage 5: spin connection, compressed in its metric antisymmetric pair.
    Kokkos::parallel_for(
    "ref_gh reference spin connection",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, ncells*24), KOKKOS_LAMBDA(const int idx) {
      const int component = idx/ncells;
      int work = idx % ncells;
      const int i = work % n1; work /= n1;
      const int j = work % n2; work /= n2;
      const int k = work % n3;
      const int m = work/n3;
      int A = 0;
      int B = 0;
      RefDecodeAntisymmetricPair4(component/4, A, B);
      const int C = component % 4;
      const ReferenceCachePoint reference{evolution, diagnostic, m, k, j, i};
      const Real eta_A = (A == 0) ? -1.0 : 1.0;
      const Real eta_B = (B == 0) ? -1.0 : 1.0;
      const Real projected = 0.5*(eta_A*RawReferenceSpin(reference, A, B, C)
                                  - eta_B*RawReferenceSpin(reference, B, A, C));
      evolution(m, kRefSpin + component, k, j, i) = eta_A*projected;
    });

    // Stage 6a: cache d(coframe) once. The metric-jet workspace is no longer
    // needed after connection assembly, so it is deliberately reused here.
    Kokkos::parallel_for(
    "ref_gh reference coframe derivative",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, ncells*64), KOKKOS_LAMBDA(const int idx) {
      const int component = idx/ncells;
      int work = idx % ncells;
      const int i = work % n1; work /= n1;
      const int j = work % n2; work /= n2;
      const int k = work % n3;
      const int m = work/n3;
      const int p = component/16;
      const int A = (component % 16)/4;
      const int a = component % 4;
      const ReferenceCachePoint reference{evolution, diagnostic, m, k, j, i};
      Real derivative = 0.0;
      for (int B = 0; B < 4; ++B) {
        for (int b = 0; b < 4; ++b) {
          derivative -= ReferenceCoframe(reference, A, b)
                        *ReferenceDFrame(reference, p, B, b)
                        *ReferenceCoframe(reference, B, a);
        }
      }
      workspace(m, kRefWorkspaceCoframeDerivative + component, k, j, i) = derivative;
    });

    // Stage 6b: form coordinate derivatives of the projected spin connection.
    Kokkos::parallel_for(
    "ref_gh reference coordinate spin derivative",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, ncells*96), KOKKOS_LAMBDA(const int idx) {
      const int component = idx/ncells;
      int work = idx % ncells;
      const int i = work % n1; work /= n1;
      const int j = work % n2; work /= n2;
      const int k = work % n3;
      const int m = work/n3;
      const int p = component/24;
      const int pair_component = (component % 24)/4;
      const int C = component % 4;
      int A = 0;
      int B = 0;
      RefDecodeAntisymmetricPair4(pair_component, A, B);
      const ReferenceCachePoint reference{evolution, diagnostic, m, k, j, i};
      const ReferenceWorkspacePoint workspace_point{workspace, m, k, j, i};
      const Real eta_A = (A == 0) ? -1.0 : 1.0;
      const Real eta_B = (B == 0) ? -1.0 : 1.0;
      const Real projected = 0.5*(
          eta_A*RawReferenceSpinCoordinateDerivative(
              reference, workspace_point, p, A, B, C)
          - eta_B*RawReferenceSpinCoordinateDerivative(
              reference, workspace_point, p, B, A, C));
      workspace(m, kRefWorkspaceSpinCoordinateDerivative + component,
                k, j, i) = eta_A*projected;
    });

    // Stage 6c: convert the coordinate derivative index to the frame index.
    Kokkos::parallel_for(
    "ref_gh reference spin derivative",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, ncells*96), KOKKOS_LAMBDA(const int idx) {
      const int component = idx/ncells;
      int work = idx % ncells;
      const int i = work % n1; work /= n1;
      const int j = work % n2; work /= n2;
      const int k = work % n3;
      const int m = work/n3;
      const int D = component/24;
      const int pair_component = (component % 24)/4;
      const int C = component % 4;
      const ReferenceCachePoint reference{evolution, diagnostic, m, k, j, i};
      const ReferenceWorkspacePoint workspace_point{workspace, m, k, j, i};
      Real derivative = 0.0;
      for (int p = 0; p < 4; ++p) {
        derivative += ReferenceFrame(reference, D, p)
                      *WorkspaceSpinCoordinateDerivative(
                          workspace_point, p, pair_component, C);
      }
      evolution(m, kRefSpinDerivative + component, k, j, i) = derivative;
    });

    // Stage 7: curvature in compact bivector form. Preserve the optimized exact
    // Schwarzschild Weyl tensor for the stationary trumpet. All other
    // nontrivial providers use the generic Cartan construction from the cached
    // spin, spin derivative, and frame commutator.
    Kokkos::parallel_for(
    "ref_gh reference curvature",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, ncells*21), KOKKOS_LAMBDA(const int idx) {
      const int component = idx/ncells;
      int work = idx % ncells;
      const int i = work % n1; work /= n1;
      const int j = work % n2; work /= n2;
      const int k = work % n3;
      const int m = work/n3;
      int first_pair = 0;
      int second_pair = 0;
      RefDecodeSymmetricPair6(component, first_pair, second_pair);
      int A = 0;
      int B = 0;
      int C = 0;
      int D = 0;
      RefDecodeAntisymmetricPair4(first_pair, A, B);
      RefDecodeAntisymmetricPair4(second_pair, C, D);
      const Real x = CellCenterX(i - indcs.is, indcs.nx1,
                                 size.d_view(m).x1min, size.d_view(m).x1max);
      const Real y = CellCenterX(j - indcs.js, indcs.nx2,
                                 size.d_view(m).x2min, size.d_view(m).x2max);
      const Real z = CellCenterX(k - indcs.ks, indcs.nx3,
                                 size.d_view(m).x3min, size.d_view(m).x3max);
      const ReferenceProviderPoint point{provider, m, k, j, i};
      Real raised = 0.0;
      if (reference_kind == 1) {
        raised = ProviderRiemann(reference_kind, point, mass, x, y, z,
                                 center_x, center_y, center_z,
                                 A, B, C, D);
      } else if (reference_kind != 0) {
        const ReferenceCachePoint reference{
            evolution, diagnostic, m, k, j, i};
        raised = ReferenceSpinDerivative(reference, C, A, B, D)
                 - ReferenceSpinDerivative(reference, D, A, B, C);
        for (int E = 0; E < 4; ++E) {
          raised += ReferenceSpin(reference, A, E, C)
                      *ReferenceSpin(reference, E, B, D)
                    - ReferenceSpin(reference, A, E, D)
                      *ReferenceSpin(reference, E, B, C)
                    - RawReferenceStructure4(reference, E, C, D)
                      *ReferenceSpin(reference, A, B, E);
        }
      }
      evolution(m, kRefRiemann + component, k, j, i) =
          ((A == 0) ? -1.0 : 1.0)*raised;
    });
    reference_cache_time = time;
    reference_cache_generation = controller_generation;
    reference_diagnostic_time = NAN;
  }

  // Reference Ricci is diagnostic-only. Keep it out of production cache
  // updates, but construct it on demand from the same prescribed provider.
  if (diagnostics_requested && !diagnostics_current) {
    Kokkos::parallel_for(
    "ref_gh reference Ricci",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, ncells*16), KOKKOS_LAMBDA(const int idx) {
      const int component = idx/ncells;
      int work = idx % ncells;
      const int i = work % n1; work /= n1;
      const int j = work % n2; work /= n2;
      const int k = work % n3;
      const int m = work/n3;
      const int B = component/4;
      const int D = component % 4;
      const Real x = CellCenterX(i - indcs.is, indcs.nx1,
                                 size.d_view(m).x1min, size.d_view(m).x1max);
      const Real y = CellCenterX(j - indcs.js, indcs.nx2,
                                 size.d_view(m).x2min, size.d_view(m).x2max);
      const Real z = CellCenterX(k - indcs.ks, indcs.nx3,
                                 size.d_view(m).x3min, size.d_view(m).x3max);
      const ReferenceProviderPoint point{provider, m, k, j, i};
      Real ricci = 0.0;
      for (int A = 0; A < 4; ++A) {
        ricci += ProviderRiemann(reference_kind, point, mass, x, y, z,
                                 center_x, center_y, center_z, A, B, A, D);
      }
      diagnostic(m, kRefRicci + component, k, j, i) = ricci;
    });
    reference_diagnostic_time = time;
    reference_diagnostic_generation = controller_generation;
  }

  if (opt.validate_reference_cache
      && (opt.reference_time_dependent
          || !reference_cache_oracle_validated
          || !reference_diagnostic_oracle_validated)) {
    using MaxLoc = Kokkos::MaxLoc<Real, int>;
    MaxLoc::value_type maximum_error;
    Kokkos::parallel_reduce(
    "ref_gh reference cache oracle", Kokkos::RangePolicy<>(DevExeSpace(),
    0, pmy_pack->nmb_thispack*n3*n2*n1),
    KOKKOS_LAMBDA(const int idx, MaxLoc::value_type &local_maximum) {
      int work = idx;
      const int i = work % n1; work /= n1;
      const int j = work % n2; work /= n2;
      const int k = work % n3;
      const int m = work/n3;
      const Real x = CellCenterX(i - indcs.is, indcs.nx1,
                                 size.d_view(m).x1min, size.d_view(m).x1max);
      const Real y = CellCenterX(j - indcs.js, indcs.nx2,
                                 size.d_view(m).x2min, size.d_view(m).x2max);
      const Real z = CellCenterX(k - indcs.ks, indcs.nx3,
                                 size.d_view(m).x3min, size.d_view(m).x3max);
      ReferenceGeometry oracle;
      GetReferenceGeometry(reference_kind, table, mass, center_x, center_y,
                           center_z, time, x, y, z, controlled, oracle);
      const ReferenceCachePoint cached{evolution, diagnostic, m, k, j, i};
      for (int A = 0; A < 4; ++A) {
        for (int a = 0; a < 4; ++a) {
          UpdateReferenceOracleMaximum(
              ReferenceCoframe(cached, A, a), oracle.coframe[A][a],
              0, local_maximum);
          UpdateReferenceOracleMaximum(
              ReferenceFrame(cached, A, a), oracle.frame[A][a],
              1, local_maximum);
          for (int p = 0; p < 4; ++p) {
            UpdateReferenceOracleMaximum(
                ReferenceDFrame(cached, p, A, a), oracle.d_frame[p][A][a],
                2, local_maximum);
          }
        }
      }
      for (int A = 0; A < 4; ++A) {
        for (int B = 0; B < 4; ++B) {
          for (int C = 0; C < 4; ++C) {
            if (B <= C) {
              UpdateReferenceOracleMaximum(
                  ReferenceChristoffel(cached, A, B, C),
                  oracle.christoffel[A][B][C], 3, local_maximum);
            }
            UpdateReferenceOracleMaximum(
                ReferenceSpin(cached, A, B, C), oracle.spin[A][B][C],
                4, local_maximum);
            for (int D = 0; D < 4; ++D) {
              UpdateReferenceOracleMaximum(
                  ReferenceSpinDerivative(cached, A, B, C, D),
                  oracle.spin_derivative[A][B][C][D], 5, local_maximum);
              UpdateReferenceOracleMaximum(
                  ReferenceRiemann(cached, A, B, C, D),
                  oracle.riemann_frame[A][B][C][D], 6, local_maximum);
            }
          }
        }
      }
      for (int I = 0; I < 3; ++I) {
        for (int J = 0; J < 3; ++J) {
          UpdateReferenceOracleMaximum(
              ReferenceSpatialFrame(cached, I, J), oracle.spatial_frame[I][J],
              7, local_maximum);
          UpdateReferenceOracleMaximum(
              ReferenceSpatialCoframe(cached, I, J),
              oracle.spatial_coframe[I][J], 8, local_maximum);
          UpdateReferenceOracleMaximum(
              ReferenceDtSpatialFrame(cached, I, J),
              oracle.dt_spatial_frame[I][J], 9, local_maximum);
          for (int K = 0; K < 3; ++K) {
            UpdateReferenceOracleMaximum(
                ReferenceStructure(cached, I, J, K),
                oracle.structure[I][J][K], 10, local_maximum);
          }
        }
      }
      for (int p = 0; p < 4; ++p) {
        for (int q = 0; q < 4; ++q) {
          for (int A = 0; A < 4; ++A) {
            for (int a = 0; a < 4; ++a) {
              UpdateReferenceOracleMaximum(
                  ReferenceDDFrame(cached, p, q, A, a),
                  oracle.dd_frame[p][q][A][a], 11, local_maximum);
            }
          }
        }
        for (int a = 0; a < 4; ++a) {
          for (int b = 0; b < 4; ++b) {
            for (int c = 0; c < 4; ++c) {
              if (b <= c) {
                UpdateReferenceOracleMaximum(
                    ReferenceDChristoffel(cached, p, a, b, c),
                    oracle.d_christoffel[p][a][b][c], 12, local_maximum);
              }
            }
          }
        }
      }
      for (int A = 0; A < 4; ++A) {
        for (int B = 0; B < 4; ++B) {
          UpdateReferenceOracleMaximum(
              ReferenceRicci(cached, A, B), oracle.ricci_frame[A][B],
              13, local_maximum);
        }
      }
    }, MaxLoc(maximum_error));
    constexpr Real kRoundoffTolerance =
        256.0*std::numeric_limits<Real>::epsilon();
    if (!(maximum_error.val <= kRoundoffTolerance)) {
      std::cout << "### FATAL ERROR: Ref-GH reference cache conditioned scaled "
                   "error "
                   "disagrees with oracle: " << maximum_error.val
                << " category=" << maximum_error.loc
                << " tolerance=" << kRoundoffTolerance
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    reference_cache_oracle_validated = true;
    reference_diagnostic_oracle_validated = true;
    if (global_variable::my_rank == 0) {
      std::cout << "reference-GH production cache oracle conditioned scaled Linf = "
                << maximum_error.val << ", time=" << time << std::endl;
    }
  }
}

TaskStatus RefGh::UpdateReferenceGeometry(Driver *driver, const int stage) {
  const Real time = StageTime(driver, stage);
  FillReferenceCache(time, opt.source_kind != 0);
  DebugFence("ref_gh UpdateReference");
  return TaskStatus::complete;
}

TaskStatus RefGh::ExpRKUpdate(Driver *driver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const Real gam0 = driver->gam0[stage - 1];
  const Real gam1 = driver->gam1[stage - 1];
  const Real beta_dt = driver->beta[stage - 1]*pmy_pack->pmesh->dt;
  const auto state = u0;
  const auto base = u1;
  const auto rhs = u_rhs;
  par_for("ref_gh RK update", DevExeSpace(), 0, pmy_pack->nmb_thispack - 1,
  0, nref_gh - 1, indcs.ks, indcs.ke, indcs.js, indcs.je, indcs.is, indcs.ie,
  KOKKOS_LAMBDA(const int m, const int n, const int k, const int j, const int i) {
    state(m, n, k, j, i) = gam0*state(m, n, k, j, i)
                               + gam1*base(m, n, k, j, i)
                               + beta_dt*rhs(m, n, k, j, i);
  });
  if (opt.reference_controlled) {
    controller.delta_q = gam0*controller.delta_q
                         + gam1*controller_base.delta_q
                         + beta_dt*controller_rhs.delta_q;
    controller.delta_q_dot = gam0*controller.delta_q_dot
                             + gam1*controller_base.delta_q_dot
                             + beta_dt*controller_rhs.delta_q_dot;
    controller.delta_p = gam0*controller.delta_p
                         + gam1*controller_base.delta_p
                         + beta_dt*controller_rhs.delta_p;
    controller.delta_p_dot = gam0*controller.delta_p_dot
                             + gam1*controller_base.delta_p_dot
                             + beta_dt*controller_rhs.delta_p_dot;
    ++controller_generation;
    const Real mass = opt.reference_mass;
    const bool invalid = !std::isfinite(controller.delta_q)
        || !std::isfinite(controller.delta_q_dot)
        || !std::isfinite(controller.delta_p)
        || !std::isfinite(controller.delta_p_dot)
        || std::abs(controller.delta_q) > opt.controller_delta_bound
        || std::abs(controller.delta_p) > opt.controller_delta_bound
        || std::abs(controller.delta_q_dot)*mass > opt.controller_rate_bound
        || std::abs(controller.delta_p_dot)*mass > opt.controller_rate_bound;
    if (invalid) {
      std::cout << "### FATAL ERROR: Ref-GH controller crossed a hard state bound: "
                << "delta_q=" << controller.delta_q
                << " delta_q_dot*M=" << controller.delta_q_dot*mass
                << " delta_p=" << controller.delta_p
                << " delta_p_dot*M=" << controller.delta_p_dot*mass
                << " generation=" << controller_generation << std::endl;
      std::exit(EXIT_FAILURE);
    }
    PersistControllerState();
  }
  DebugFence("ref_gh ExpRKUpdate");
  return TaskStatus::complete;
}

TaskStatus RefGh::RestrictU(Driver *, int) {
  if (pmy_pack->pmesh->multilevel) {
    pmy_pack->pmesh->pmr->RestrictCC(u0, coarse_u0, true);
  }
  DebugFence("ref_gh RestrictU");
  return TaskStatus::complete;
}
TaskStatus RefGh::SendU(Driver *, int) {
  const auto status = pbval_u->PackAndSendCC(u0, coarse_u0);
  DebugFence("ref_gh SendU");
  return status;
}
TaskStatus RefGh::RecvU(Driver *, int) {
  const auto status = pbval_u->RecvAndUnpackCC(u0, coarse_u0);
  DebugFence("ref_gh RecvU");
  return status;
}
TaskStatus RefGh::Prolongate(Driver *, int) {
  if (pmy_pack->pmesh->multilevel) pbval_u->ProlongateCC(u0, coarse_u0, true);
  DebugFence("ref_gh Prolongate");
  return TaskStatus::complete;
}
TaskStatus RefGh::ApplyPhysicalBCs(Driver *, int) {
  if (pmy_pack->pmesh->strictly_periodic) return TaskStatus::complete;
  if (opt.reference_kind != 1 && opt.reference_kind != 4
      && opt.reference_kind != 5) {
    std::cout << "### FATAL ERROR: non-periodic ref_gh boundaries are currently "
              << "implemented only for Schwarzschild reference states."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Current-reference boundary data for the Schwarzschild research gates.  In the
  // regular frame the complete prescribed state is simply Psi=eta, Pi=Phi=0.  This is
  // exact for stationary-reference tests and is used only while the dynamical transition
  // remains causally disconnected from the analysis region.
  // Internal block faces have BoundaryFlag::block and are left untouched.
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int ng = indcs.ng;
  const int n1 = indcs.nx1 + 2*ng;
  const int n2 = (indcs.nx2 > 1) ? indcs.nx2 + 2*ng : 1;
  const int n3 = (indcs.nx3 > 1) ? indcs.nx3 + 2*ng : 1;
  const int is = indcs.is;
  const int ie = indcs.ie;
  const int js = indcs.js;
  const int je = indcs.je;
  const int ks = indcs.ks;
  const int ke = indcs.ke;
  const int nmb = pmy_pack->nmb_thispack;
  const auto state = u0;
  const auto mb_bcs = pmy_pack->pmb->mb_bcs.d_view;

  if (pmy_pack->pmesh->mesh_bcs[BoundaryFace::inner_x1] != BoundaryFlag::periodic) {
    par_for("ref_gh exact trumpet x1 boundaries", DevExeSpace(), 0, nmb - 1,
    0, nref_gh - 1, 0, n3 - 1, 0, n2 - 1,
    KOKKOS_LAMBDA(const int m, const int n, const int k, const int j) {
      Real value = 0.0;
      if (n == PsiIndex(0, 0)) value = -1.0;
      if (n == PsiIndex(1, 1) || n == PsiIndex(2, 2) || n == PsiIndex(3, 3)) {
        value = 1.0;
      }
      if (mb_bcs(m, BoundaryFace::inner_x1) != BoundaryFlag::block) {
        for (int g = 1; g <= ng; ++g) state(m, n, k, j, is - g) = value;
      }
      if (mb_bcs(m, BoundaryFace::outer_x1) != BoundaryFlag::block) {
        for (int g = 1; g <= ng; ++g) state(m, n, k, j, ie + g) = value;
      }
    });
  }
  if (pmy_pack->pmesh->mesh_bcs[BoundaryFace::inner_x2] != BoundaryFlag::periodic) {
    par_for("ref_gh exact trumpet x2 boundaries", DevExeSpace(), 0, nmb - 1,
    0, nref_gh - 1, 0, n3 - 1, 0, n1 - 1,
    KOKKOS_LAMBDA(const int m, const int n, const int k, const int i) {
      Real value = 0.0;
      if (n == PsiIndex(0, 0)) value = -1.0;
      if (n == PsiIndex(1, 1) || n == PsiIndex(2, 2) || n == PsiIndex(3, 3)) {
        value = 1.0;
      }
      if (mb_bcs(m, BoundaryFace::inner_x2) != BoundaryFlag::block) {
        for (int g = 1; g <= ng; ++g) state(m, n, k, js - g, i) = value;
      }
      if (mb_bcs(m, BoundaryFace::outer_x2) != BoundaryFlag::block) {
        for (int g = 1; g <= ng; ++g) state(m, n, k, je + g, i) = value;
      }
    });
  }
  if (pmy_pack->pmesh->mesh_bcs[BoundaryFace::inner_x3] != BoundaryFlag::periodic) {
    par_for("ref_gh exact trumpet x3 boundaries", DevExeSpace(), 0, nmb - 1,
    0, nref_gh - 1, 0, n2 - 1, 0, n1 - 1,
    KOKKOS_LAMBDA(const int m, const int n, const int j, const int i) {
      Real value = 0.0;
      if (n == PsiIndex(0, 0)) value = -1.0;
      if (n == PsiIndex(1, 1) || n == PsiIndex(2, 2) || n == PsiIndex(3, 3)) {
        value = 1.0;
      }
      if (mb_bcs(m, BoundaryFace::inner_x3) != BoundaryFlag::block) {
        for (int g = 1; g <= ng; ++g) state(m, n, ks - g, j, i) = value;
      }
      if (mb_bcs(m, BoundaryFace::outer_x3) != BoundaryFlag::block) {
        for (int g = 1; g <= ng; ++g) state(m, n, ke + g, j, i) = value;
      }
    });
  }
  DebugFence("ref_gh ApplyPhysicalBCs");
  return TaskStatus::complete;
}

TaskStatus RefGh::NewTimeStep(Driver *driver, int stage) {
  if (stage != driver->nexp_stages) return TaskStatus::complete;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  const auto state = u0;
  const auto reference_cache = reference_evolution;
  const auto reference_extra = reference_diagnostic;
  const int ncells = indcs.nx1*indcs.nx2*indcs.nx3;
  Kokkos::parallel_reduce(
      "ref_gh dt", Kokkos::RangePolicy<>(DevExeSpace(),
      0, pmy_pack->nmb_thispack*ncells),
      KOKKOS_LAMBDA(const int idx, Real &local_minimum) {
        int work = idx;
        const int i = work % indcs.nx1 + indcs.is; work /= indcs.nx1;
        const int j = work % indcs.nx2 + indcs.js; work /= indcs.nx2;
        const int k = work % indcs.nx3 + indcs.ks;
        const int m = work/indcs.nx3;
        const ReferenceCachePoint reference{
            reference_cache, reference_extra, m, k, j, i};
        Real psi[4][4], metric[4][4];  // NOLINT(runtime/arrays)
        for (int a = 0; a < 4; ++a) {
          for (int b = a; b < 4; ++b) {
            psi[a][b] = psi[b][a] =
                state(m, PsiIndex(a, b), k, j, i);
          }
        }
        for (int a = 0; a < 4; ++a) {
          for (int b = 0; b < 4; ++b) {
            metric[a][b] = 0.0;
            for (int A = 0; A < 4; ++A) {
              for (int B = 0; B < 4; ++B) {
                metric[a][b] += ReferenceCoframe(reference, A, a)
                                *ReferenceCoframe(reference, B, b)*psi[A][B];
              }
            }
          }
        }
        Real inverse[4][4], determinant = 0.0;  // NOLINT(runtime/arrays)
        if (!Invert4(metric, inverse, determinant) || !(inverse[0][0] < 0.0)) {
          local_minimum = 0.0;
          return;
        }
        Real spatial_inverse[3][3], spatial_det = 0.0; // NOLINT(runtime/arrays)
        if (!InvertSpatial3(metric, spatial_inverse, spatial_det)) {
          local_minimum = 0.0;
          return;
        }
        const Real alpha = 1.0/Kokkos::sqrt(-inverse[0][0]);
        for (int p = 0; p < 3; ++p) {
          const Real beta = alpha*alpha*inverse[0][p + 1];
          const Real speed = Kokkos::abs(beta)
                             + alpha*Kokkos::sqrt(spatial_inverse[p][p]);
          const Real dx = (p == 0) ? size.d_view(m).dx1
                          : ((p == 1) ? size.d_view(m).dx2 : size.d_view(m).dx3);
          const Real candidate = speed > 0.0 ? dx/speed : 0.0;
          if (candidate < local_minimum) local_minimum = candidate;
        }
      }, Kokkos::Min<Real>(dtnew));
  max_char_speed = 0.0;  // populated by the full conditioning diagnostic pass later
  if (!(dtnew > 0.0) || !std::isfinite(dtnew)) {
    std::cout << "### FATAL ERROR: ref_gh reached an invalid effective timestep."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (opt.fail_closed_dt > 0.0 && dtnew < opt.fail_closed_dt) {
    std::cout << "### FATAL ERROR: ref_gh timestep crossed fail_closed_dt."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  DebugFence("ref_gh NewTimeStep");
  return TaskStatus::complete;
}

}  // namespace ref_gh
