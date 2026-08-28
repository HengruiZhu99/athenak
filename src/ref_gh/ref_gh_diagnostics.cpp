//========================================================================================
//! \file ref_gh_diagnostics.cpp
//! \brief ADM reconstruction and constraint refresh for reference-frame GH.
//========================================================================================
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "parameter_input.hpp"
#include "ref_gh/analytic_radial_q_production.hpp"
#include "ref_gh/ref_gh.hpp"
#include "ref_gh/ref_gh_geometry.hpp"
#include "ref_gh/reference_cache.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace ref_gh {

void RefGh::SetADMVariables(MeshBlockPack *pack) { pack->prefgh->RefGhToADM(); }

void RefGh::RefGhToADM() {
  if (pmy_pack->padm == nullptr) return;
  FillReferenceCache(pmy_pack->pmesh->time, false);
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int n1 = indcs.nx1 + 2*indcs.ng;
  const int n2 = (indcs.nx2 > 1) ? indcs.nx2 + 2*indcs.ng : 1;
  const int n3 = (indcs.nx3 > 1) ? indcs.nx3 + 2*indcs.ng : 1;
  const auto state = u0;
  const auto reference_cache = reference_evolution;
  const auto reference_extra = reference_diagnostic;
  const auto analytic_static = reference_static;
  const auto analytic_stage = reference_stage;
  const int reference_backend = opt.reference_backend;
  const Real center_x = opt.reference_center[0];
  const Real center_y = opt.reference_center[1];
  const Real center_z = opt.reference_center[2];
  auto &size = pmy_pack->pmb->mb_size;
  const auto adm_vars = pmy_pack->padm->adm;
  par_for("ref_gh to ADM", DevExeSpace(), 0, pmy_pack->nmb_thispack - 1,
  0, n3 - 1, 0, n2 - 1, 0, n1 - 1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real x = CellCenterX(i - indcs.is, indcs.nx1,
                               size.d_view(m).x1min, size.d_view(m).x1max);
    const Real y = CellCenterX(j - indcs.js, indcs.nx2,
                               size.d_view(m).x2min, size.d_view(m).x2max);
    const Real z = CellCenterX(k - indcs.ks, indcs.nx3,
                               size.d_view(m).x3min, size.d_view(m).x3max);
    const ProductionReferencePoint reference = MakeProductionReferencePoint(
        reference_backend, reference_cache, reference_extra, analytic_static,
        analytic_stage, m, k, j, i, x, y, z, center_x, center_y, center_z);
    Real psi[4][4], pi[4][4], phi[3][4][4], d_psi[4][4][4]; // NOLINT
    Real metric[4][4], d_metric[4][4][4]; // NOLINT
    CoordinateGhGeometry geometry;
    Real determinant = 0.0;
    if (!LoadProductionPointGeometry(state, reference, m, k, j, i, psi, pi,
                                     phi, d_psi, metric, d_metric, geometry,
                                     determinant)) {
      adm_vars.alpha(m, k, j, i) = NAN;
      adm_vars.psi4(m, k, j, i) = NAN;
      for (int a = 0; a < 3; ++a) {
        adm_vars.beta_u(m, a, k, j, i) = NAN;
        for (int b = a; b < 3; ++b) {
          adm_vars.g_dd(m, a, b, k, j, i) = NAN;
          adm_vars.vK_dd(m, a, b, k, j, i) = NAN;
        }
      }
      return;
    }
    adm_vars.alpha(m, k, j, i) = geometry.lapse;
    for (int a = 0; a < 3; ++a) {
      adm_vars.beta_u(m, a, k, j, i) = geometry.shift[a];
      for (int b = a; b < 3; ++b) {
        adm_vars.g_dd(m, a, b, k, j, i) = metric[a + 1][b + 1];
        adm_vars.vK_dd(m, a, b, k, j, i) =
            -geometry.lapse*geometry.christoffel[0][a + 1][b + 1];
      }
    }
    const Real det_spatial = adm::SpatialDet(
        metric[1][1], metric[1][2], metric[1][3], metric[2][2],
        metric[2][3], metric[3][3]);
    adm_vars.psi4(m, k, j, i) = Kokkos::pow(det_spatial, 1.0/3.0);
  });
}

void RefGh::CacheMetricCondition() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int ncells = indcs.nx1*indcs.nx2*indcs.nx3;
  const auto constraints = u_con;
  const auto adm_vars = pmy_pack->padm->adm;
  Kokkos::parallel_for(
      "ref_gh cache metric condition", Kokkos::RangePolicy<>(DevExeSpace(),
      0, pmy_pack->nmb_thispack*ncells), KOKKOS_LAMBDA(const int idx) {
        int work = idx;
        const int i = work % indcs.nx1 + indcs.is; work /= indcs.nx1;
        const int j = work % indcs.nx2 + indcs.js; work /= indcs.nx2;
        const int k = work % indcs.nx3 + indcs.ks;
        const int m = work/indcs.nx3;
        const Real frame_scale = constraints(
            m, kMetricConditionDiagnostic, k, j, i);
        const Real scale2 = frame_scale*frame_scale;
        constraints(m, kMetricConditionDiagnostic, k, j, i) =
            SymmetricConditionNumber3(
                scale2*adm_vars.g_dd(m, 0, 0, k, j, i),
                scale2*adm_vars.g_dd(m, 0, 1, k, j, i),
                scale2*adm_vars.g_dd(m, 0, 2, k, j, i),
                scale2*adm_vars.g_dd(m, 1, 1, k, j, i),
                scale2*adm_vars.g_dd(m, 1, 2, k, j, i),
                scale2*adm_vars.g_dd(m, 2, 2, k, j, i));
      });
}

void RefGh::UpdateDiagnostics() {
  FillReferenceCache(pmy_pack->pmesh->time, true);
  DebugFence("ref_gh diagnostics reference");
  RefGhToADM();
  DebugFence("ref_gh diagnostics ADM reconstruction");
  switch (opt.fd_order) {
    case 2: CalcConstraints<2>(); break;
    case 4: CalcConstraints<3>(); break;
    case 6: CalcConstraints<4>(); break;
  }
  DebugFence("ref_gh diagnostics constraints");
  CacheMetricCondition();
  DebugFence("ref_gh diagnostics metric condition");
}

void RefGh::AppendMaxLocationDiagnostics() {
  if (!opt.max_location_diagnostics) return;
  if (max_location_diagnostic_time == pmy_pack->pmesh->time
      && max_location_diagnostic_cycle == pmy_pack->pmesh->ncycle) return;
  max_location_diagnostic_time = pmy_pack->pmesh->time;
  max_location_diagnostic_cycle = pmy_pack->pmesh->ncycle;

  enum Diagnostic : int {
    kReferenceRicci, kReferenceRiemann, kSpin, kSpinDerivative,
    kPsi, kQ, kDelta, kPi, kPhi, kGhConstraint, kReductionConstraint,
    kCurlConstraint, kSourceCurvature, kSourceQq, kSourceDeltaDelta,
    kSourceDamping, kSourceFrameCorrection, kShiftLapseRatio,
    kDiagnosticCount
  };
  constexpr const char *names[kDiagnosticCount] = {
    "reference_Ricci", "reference_Riemann", "spin_connection",
    "spin_derivative", "Psi", "Q", "Delta", "Pi", "Phi",
    "GH_constraint", "reduction_constraint", "curl_constraint",
    "source_curvature", "source_QQ", "source_DeltaDelta",
    "source_damping", "source_frame_correction", "shift_lapse_ratio"
  };
  constexpr int kRecordFields = 12;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  const auto state = u0;
  const auto constraints = u_con;
  const auto reference_cache = reference_evolution;
  const auto reference_extra = reference_diagnostic;
  const auto adm_vars = pmy_pack->padm->adm;
  const int ncells = indcs.nx1*indcs.nx2*indcs.nx3;
  Real local_records[kDiagnosticCount*kRecordFields] = {};  // NOLINT

  using MaxLoc = Kokkos::MaxLoc<Real, int>;
  for (int diagnostic_index = 0; diagnostic_index < kDiagnosticCount;
       ++diagnostic_index) {
    MaxLoc::value_type maximum;
    Kokkos::parallel_reduce(
        "ref_gh diagnostic maximum location",
        Kokkos::RangePolicy<>(DevExeSpace(),
            0, pmy_pack->nmb_thispack*ncells),
        KOKKOS_LAMBDA(const int idx, MaxLoc::value_type &local_maximum) {
          int work = idx;
          const int i = work % indcs.nx1 + indcs.is; work /= indcs.nx1;
          const int j = work % indcs.nx2 + indcs.js; work /= indcs.nx2;
          const int k = work % indcs.nx3 + indcs.ks;
          const int m = work/indcs.nx3;
          const ReferenceCachePoint reference{
              reference_cache, reference_extra, m, k, j, i};
          Real value2 = 0.0;
          if (diagnostic_index == kReferenceRicci) {
            const Real value = constraints(
                m, kDiagnosticOffset + 2, k, j, i);
            value2 = value*value;
          } else if (diagnostic_index == kReferenceRiemann) {
            for (int A = 0; A < 4; ++A) {
              for (int B = 0; B < 4; ++B) {
                for (int C = 0; C < 4; ++C) {
                  for (int D = 0; D < 4; ++D) {
                    const Real value = ReferenceRiemann(reference, A, B, C, D);
                    value2 += value*value;
                  }
                }
              }
            }
          } else if (diagnostic_index == kSpin) {
            for (int A = 0; A < 4; ++A) {
              for (int B = 0; B < 4; ++B) {
                for (int C = 0; C < 4; ++C) {
                  const Real value = ReferenceSpin(reference, A, B, C);
                  value2 += value*value;
                }
              }
            }
          } else if (diagnostic_index == kSpinDerivative) {
            for (int D = 0; D < 4; ++D) {
              for (int A = 0; A < 4; ++A) {
                for (int B = 0; B < 4; ++B) {
                  for (int C = 0; C < 4; ++C) {
                    const Real value =
                        ReferenceSpinDerivative(reference, D, A, B, C);
                    value2 += value*value;
                  }
                }
              }
            }
          } else if (diagnostic_index == kPsi) {
            for (int n = kPsiOffset; n < kPiOffset; ++n) {
              const Real value = state(m, n, k, j, i);
              value2 += value*value;
            }
          } else if (diagnostic_index == kQ) {
            const Real value = constraints(
                m, kDiagnosticOffset + 0, k, j, i);
            value2 = value*value;
          } else if (diagnostic_index == kDelta) {
            const Real value = constraints(
                m, kDiagnosticOffset + 1, k, j, i);
            value2 = value*value;
          } else if (diagnostic_index == kPi) {
            for (int n = kPiOffset; n < kPhiOffset; ++n) {
              const Real value = state(m, n, k, j, i);
              value2 += value*value;
            }
          } else if (diagnostic_index == kPhi) {
            for (int n = kPhiOffset; n < nvar; ++n) {
              const Real value = state(m, n, k, j, i);
              value2 += value*value;
            }
          } else if (diagnostic_index == kGhConstraint) {
            for (int A = 0; A < 4; ++A) {
              const Real value = constraints(m, A, k, j, i);
              value2 += value*value;
            }
          } else if (diagnostic_index == kReductionConstraint) {
            const Real value = constraints(m, 4, k, j, i);
            value2 = value*value;
          } else if (diagnostic_index == kCurlConstraint) {
            const Real value = constraints(m, 5, k, j, i);
            value2 = value*value;
          } else if (diagnostic_index >= kSourceCurvature
                     && diagnostic_index <= kSourceFrameCorrection) {
            const int source_slot = kDiagnosticOffset + 4
                                    + diagnostic_index - kSourceCurvature;
            const Real value = constraints(m, source_slot, k, j, i);
            value2 = value*value;
          } else {
            const Real alpha = adm_vars.alpha(m, k, j, i);
            Real shift2 = 0.0;
            for (int a = 0; a < 3; ++a) {
              for (int b = 0; b < 3; ++b) {
                shift2 += adm_vars.g_dd(m, a, b, k, j, i)
                          *adm_vars.beta_u(m, a, k, j, i)
                          *adm_vars.beta_u(m, b, k, j, i);
              }
            }
            const Real ratio = shift2/(alpha*alpha);
            value2 = ratio*ratio;
          }
          const Real value = Kokkos::sqrt(value2);
          const Real comparable = Kokkos::isfinite(value)
              ? value : std::numeric_limits<Real>::max();
          if (comparable >= local_maximum.val) {
            local_maximum.val = comparable;
            local_maximum.loc = idx;
          }
        }, MaxLoc(maximum));

    int work = maximum.loc;
    const int ii = work % indcs.nx1; work /= indcs.nx1;
    const int jj = work % indcs.nx2; work /= indcs.nx2;
    const int kk = work % indcs.nx3;
    const int m = work/indcs.nx3;
    const Real x = CellCenterX(ii, indcs.nx1,
                               size.h_view(m).x1min, size.h_view(m).x1max);
    const Real y = CellCenterX(jj, indcs.nx2,
                               size.h_view(m).x2min, size.h_view(m).x2max);
    const Real z = CellCenterX(kk, indcs.nx3,
                               size.h_view(m).x3min, size.h_view(m).x3max);
    const Real dx = x - opt.reference_center[0];
    const Real dy = y - opt.reference_center[1];
    const Real dz = z - opt.reference_center[2];
    const Real radius = std::sqrt(dx*dx + dy*dy + dz*dz);
    const int offset = diagnostic_index*kRecordFields;
    local_records[offset + 0] = maximum.val;
    local_records[offset + 1] = radius;
    local_records[offset + 2] = 0.0;
    local_records[offset + 3] = pmy_pack->pmb->mb_lev.h_view(m);
    local_records[offset + 4] = global_variable::my_rank;
    local_records[offset + 5] = pmy_pack->pmb->mb_gid.h_view(m);
    local_records[offset + 6] = x;
    local_records[offset + 7] = y;
    local_records[offset + 8] = z;
    local_records[offset + 9] = ii;
    local_records[offset + 10] = jj;
    local_records[offset + 11] = kk;
  }

  const Real time = pmy_pack->pmesh->time;
  const Real r_core = opt.transition_path == kFixedCorePath
      ? opt.r_core0*opt.reference_mass
      : opt.r_core0*opt.reference_mass
          *std::exp(-time/(opt.tau_core*opt.reference_mass));
  for (int n = 0; n < kDiagnosticCount; ++n) {
    const int offset = n*kRecordFields;
    local_records[offset + 2] = local_records[offset + 1]/r_core;
  }

  std::vector<Real> gathered;
#if MPI_PARALLEL_ENABLED
  if (global_variable::my_rank == 0) {
    gathered.resize(global_variable::nranks*kDiagnosticCount*kRecordFields);
  }
  MPI_Gather(local_records, kDiagnosticCount*kRecordFields, MPI_ATHENA_REAL,
             gathered.data(), kDiagnosticCount*kRecordFields, MPI_ATHENA_REAL,
             0, MPI_COMM_WORLD);
#else
  gathered.assign(local_records,
                  local_records + kDiagnosticCount*kRecordFields);
#endif
  if (global_variable::my_rank != 0) return;

  const std::string filename =
      pinput->GetString("job", "basename") + ".ref_gh_maxloc.tsv";
  FILE *file = std::fopen(filename.c_str(), "a+");
  if (file == nullptr) {
    std::cout << "### FATAL ERROR: unable to open Ref-GH max-location file "
              << filename << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::fseek(file, 0, SEEK_END);
  if (std::ftell(file) == 0) {
    std::fprintf(file, "time\tcycle\tdiagnostic\tmaximum\tradius\t"
                       "r_over_r_core\tlevel\trank\tgid\tx\ty\tz\ti\tj\tk\n");
  }
  for (int n = 0; n < kDiagnosticCount; ++n) {
    const Real *best = nullptr;
    for (int rank = 0; rank < global_variable::nranks; ++rank) {
      const Real *candidate = gathered.data()
          + (rank*kDiagnosticCount + n)*kRecordFields;
      if (best == nullptr || candidate[0] > best[0]) best = candidate;
    }
    std::fprintf(file,
        "%.17e\t%d\t%s\t%.17e\t%.17e\t%.17e\t%d\t%d\t%d\t"
        "%.17e\t%.17e\t%.17e\t%d\t%d\t%d\n",
        time, pmy_pack->pmesh->ncycle, names[n], best[0], best[1], best[2],
        static_cast<int>(best[3]), static_cast<int>(best[4]),
        static_cast<int>(best[5]), best[6], best[7], best[8],
        static_cast<int>(best[9]), static_cast<int>(best[10]),
        static_cast<int>(best[11]));
  }
  std::fclose(file);
}

}  // namespace ref_gh
