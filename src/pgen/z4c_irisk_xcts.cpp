//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE for details
//========================================================================================
//! \file z4c_irisk_xcts.cpp
//! \brief Spectrally interpolate IrisK XCTS data onto an arbitrary AMR mesh.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "globals.hpp"
#include "irisk_athenak_spectral_interpolator.h"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "pgen/z4c_irisk_coordinate_map.hpp"
#include "z4c/cartoon_axis_boundary.hpp"
#include "z4c/z4c.hpp"
#include "z4c/z4c_amr.hpp"
#include "z4c/curvature_diagnostics.hpp"
#include "z4c/fastflow.hpp"
#include "z4c/stored_domain_bounds.hpp"

namespace {

struct ConstraintRegionSummary {
  Real c_rms = 0.0;
  Real h_rms = 0.0;
  Real m_rms = 0.0;
  Real z_rms = 0.0;
  Real volume = 0.0;
  Real h_linf = 0.0;
  Real m_linf = 0.0;
  std::uint64_t cell_count = 0;
};

struct ConstraintSummary {
  // Coordinate-volume RMS is the direct discrete comparison to IrisK's q=10
  // element quadrature.  Proper-volume RMS is retained separately for the
  // existing AthenaK convention.
  ConstraintRegionSummary coordinate_box;
  ConstraintRegionSummary proper_box;
  ConstraintRegionSummary coordinate_support;
  ConstraintRegionSummary proper_support;
};

constexpr std::array<char, 24> kVolumeOutputMagic{
    'A', 'T', 'H', 'E', 'N', 'A', '_', 'I', 'R', 'I', 'S', 'K',
    '_', 'V', 'O', 'L', 'U', 'M', 'E', '1', '\r', '\n', '\0', '\0'};
constexpr std::uint32_t kVolumeOutputVersion = 1;
constexpr std::uint32_t kVolumeOutputEndianTag = 0x01020304U;
constexpr std::uint32_t kVolumeOutputIntegerCount = 5;
constexpr std::uint32_t kVolumeOutputRealCount = 51;

[[noreturn]] void Fail(const std::string &message) {
  std::cout << "### FATAL ERROR in " << __FILE__ << std::endl
            << message << std::endl;
  std::exit(EXIT_FAILURE);
}

template <typename T>
void WriteVolumeScalar(std::ofstream &output, const T &value) {
  output.write(reinterpret_cast<const char *>(&value),
               static_cast<std::streamsize>(sizeof(T)));
  if (!output) {
    throw std::runtime_error("failed while writing IrisK active-cell volume");
  }
}

std::filesystem::path ResolveSpectralInputPath(const std::string &filename) {
  std::error_code error;
  std::filesystem::path resolved =
      std::filesystem::absolute(filename, error).lexically_normal();
  if (error) {
    Fail("cannot resolve IrisK spectral data path '" + filename +
         "' from AthenaK's launch directory: " + error.message());
  }
  resolved = std::filesystem::weakly_canonical(resolved, error);
  if (error || !std::filesystem::is_regular_file(resolved)) {
    Fail("IrisK spectral data file does not exist or is not a regular file: " +
         resolved.string() +
         " (relative problem/irisk_adm_spectral_file paths are resolved "
         "before -d, against AthenaK's launch directory)");
  }
  return resolved;
}

template <z4c_irisk::AdmMap Map>
void FillAdmFromIrisSpectralMapped(
    MeshBlockPack *pmbp, IrisAthenakSpectralInterpolator *interpolator) {
  auto &u_adm = pmbp->padm->u_adm;
  HostArray5D<Real>::HostMirror host_u_adm = create_mirror(u_adm);
  HostArray5D<Real>::HostMirror host_u_z4c = create_mirror(pmbp->pz4c->u0);
  // The spectral export contains ADM fields plus lapse and shift, but not Theta or the
  // telegraph flux B_i.  Initialize every Z4c-only field deterministically before
  // ADMToZ4c overwrites the conformal geometry and curvature variables.
  Kokkos::deep_copy(host_u_adm, 0.0);
  Kokkos::deep_copy(host_u_z4c, 0.0);
  adm::ADM::ADMhost_vars host_adm;
  host_adm.g_dd.InitWithShallowSlice(host_u_adm, adm::ADM::I_ADM_GXX,
                                     adm::ADM::I_ADM_GZZ);
  host_adm.vK_dd.InitWithShallowSlice(host_u_adm, adm::ADM::I_ADM_KXX,
                                      adm::ADM::I_ADM_KZZ);
  host_adm.psi4.InitWithShallowSlice(host_u_adm, adm::ADM::I_ADM_PSI4);
  host_adm.alpha.InitWithShallowSlice(host_u_z4c, z4c::Z4c::I_Z4C_ALPHA);
  host_adm.beta_u.InitWithShallowSlice(host_u_z4c, z4c::Z4c::I_Z4C_BETAX,
                                       z4c::Z4c::I_Z4C_BETAZ);

  auto &indcs = pmbp->pmesh->mb_indcs;
  pmbp->pmb->mb_size.sync_host();
  pmbp->pmb->mb_bcs.sync_host();
  auto size = pmbp->pmb->mb_size.h_view;
  auto bcs = pmbp->pmb->mb_bcs.h_view;
  const auto bounds = z4c::MakeStoredDomainBounds(indcs);
  const int isg = bounds.is;
  const int ieg = bounds.ie;
  const int jsg = bounds.js;
  const int jeg = bounds.je;
  const int ksg = bounds.ks;
  const int keg = bounds.ke;
  const std::size_t ny = static_cast<std::size_t>(bounds.n2);
  const std::size_t nz = static_cast<std::size_t>(bounds.n3);
  Real minimum_psi4 = std::numeric_limits<Real>::infinity();
  Real minimum_lapse = std::numeric_limits<Real>::infinity();
  int invalid_fields = 0;

  for (int m = 0; m < pmbp->nmb_thispack; ++m) {
    constexpr bool cartoon =
        Map == z4c_irisk::AdmMap::half_rho_z_suppressed_y_v2;
    const bool axis_block =
        cartoon && bcs(m, BoundaryFace::inner_x1) == BoundaryFlag::axis;
    const int interpolation_is = axis_block ? indcs.is : isg;
    const std::size_t interpolation_nx =
        static_cast<std::size_t>(ieg - interpolation_is + 1);
    const auto iris_dimensions =
        z4c_irisk::IrisTensorProductDimensions<Map>(interpolation_nx, ny, nz);
    std::vector<double> x(iris_dimensions[0]);
    std::vector<double> y(iris_dimensions[1]);
    std::vector<double> z(iris_dimensions[2]);
    for (int i = interpolation_is; i <= ieg; ++i) {
      x[static_cast<std::size_t>(i - interpolation_is)] =
          CellCenterX(i - indcs.is, indcs.nx1, size(m).x1min, size(m).x1max);
    }
    if constexpr (cartoon) {
      y[0] = z4c_irisk::CartoonIrisInterpolationCoordinates(0.0, 0.0, 0.0)[1];
      for (int j = jsg; j <= jeg; ++j) {
        z[static_cast<std::size_t>(j - jsg)] =
            CellCenterX(j - indcs.js, indcs.nx2, size(m).x2min, size(m).x2max);
      }
    } else {
      for (int j = jsg; j <= jeg; ++j) {
        y[static_cast<std::size_t>(j - jsg)] =
            CellCenterX(j - indcs.js, indcs.nx2, size(m).x2min, size(m).x2max);
      }
      for (int k = ksg; k <= keg; ++k) {
        z[static_cast<std::size_t>(k - ksg)] =
            CellCenterX(k - indcs.ks, indcs.nx3, size(m).x3min, size(m).x3max);
      }
    }
    std::vector<double> values(iris_dimensions[0] * iris_dimensions[1] *
                               iris_dimensions[2] *
                               IRISK_ATHENAK_ADM_VARIABLE_COUNT);
    std::array<char, 1024> error{};
    if (IrisAthenakSpectralInterpolateCartesian(
            interpolator, iris_dimensions[0], iris_dimensions[1],
            iris_dimensions[2], x.data(), y.data(), z.data(), values.data(),
            error.data(), error.size()) != 0) {
      Fail(std::string("IrisK spectral interpolation failed: ") + error.data());
    }

    for (int k = ksg; k <= keg; ++k)
      for (int j = jsg; j <= jeg; ++j)
        for (int i = interpolation_is; i <= ieg; ++i) {
          const std::size_t point = z4c_irisk::IrisPointIndex<Map>(
              static_cast<std::size_t>(i - interpolation_is),
              static_cast<std::size_t>(j - jsg),
              static_cast<std::size_t>(k - ksg), interpolation_nx, ny);
          const double *value =
              values.data() + point * IRISK_ATHENAK_ADM_VARIABLE_COUNT;
          for (int variable = 0; variable < IRISK_ATHENAK_ADM_VARIABLE_COUNT;
               ++variable) {
            invalid_fields |= !std::isfinite(value[variable]);
          }
          if (std::isfinite(value[12])) {
            minimum_psi4 = std::min(minimum_psi4,
                                    static_cast<Real>(value[12]));
          }
          if (std::isfinite(value[13])) {
            minimum_lapse = std::min(minimum_lapse,
                                     static_cast<Real>(value[13]));
          }
          invalid_fields |= !(value[12] > 0.0) || !(value[13] > 0.0);
          const auto metric =
              z4c_irisk::SymmetricTensorFromPhysicalCartesian<Map>(
                  std::array<double, 6>{value[0], value[1], value[2], value[3],
                                        value[4], value[5]});
          const auto curvature =
              z4c_irisk::SymmetricTensorFromPhysicalCartesian<Map>(
                  std::array<double, 6>{value[6], value[7], value[8], value[9],
                                        value[10], value[11]});
          const auto shift = z4c_irisk::VectorFromPhysicalCartesian<Map>(
              std::array<double, 3>{value[14], value[15], value[16]});
          host_adm.g_dd(m, 0, 0, k, j, i) = metric[0];
          host_adm.g_dd(m, 0, 1, k, j, i) = metric[1];
          host_adm.g_dd(m, 0, 2, k, j, i) = metric[2];
          host_adm.g_dd(m, 1, 1, k, j, i) = metric[3];
          host_adm.g_dd(m, 1, 2, k, j, i) = metric[4];
          host_adm.g_dd(m, 2, 2, k, j, i) = metric[5];
          host_adm.vK_dd(m, 0, 0, k, j, i) = curvature[0];
          host_adm.vK_dd(m, 0, 1, k, j, i) = curvature[1];
          host_adm.vK_dd(m, 0, 2, k, j, i) = curvature[2];
          host_adm.vK_dd(m, 1, 1, k, j, i) = curvature[3];
          host_adm.vK_dd(m, 1, 2, k, j, i) = curvature[4];
          host_adm.vK_dd(m, 2, 2, k, j, i) = curvature[5];
          host_adm.psi4(m, k, j, i) =
              z4c_irisk::ScalarFromPhysicalCartesian<Map>(value[12]);
          host_adm.alpha(m, k, j, i) =
              z4c_irisk::ScalarFromPhysicalCartesian<Map>(value[13]);
          for (int component = 0; component < 3; ++component) {
            host_adm.beta_u(m, component, k, j, i) = shift[component];
          }
        }
    if constexpr (cartoon) {
      if (axis_block) {
        for (int k = ksg; k <= keg; ++k) {
          for (int j = jsg; j <= jeg; ++j) {
            for (int n = 0; n <= adm::ADM::I_ADM_PSI4; ++n) {
              if (!z4c::FillAdmAxisGhostLine(
                      host_u_adm, m, n, k, j, indcs.is, indcs.ng)) {
                Fail("invalid ADM component in IrisK axis parity fill");
              }
            }
            for (int n = z4c::Z4c::I_Z4C_ALPHA;
                 n <= z4c::Z4c::I_Z4C_BETAZ; ++n) {
              if (!z4c::FillZ4cAxisGhostLine(
                      host_u_z4c, m, n, k, j, indcs.is, indcs.ng)) {
                Fail("invalid gauge component in IrisK axis parity fill");
              }
            }
          }
        }
      }
    }
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &invalid_fields, 1, MPI_INT, MPI_MAX,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &minimum_psi4, 1, MPI_ATHENA_REAL, MPI_MIN,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &minimum_lapse, 1, MPI_ATHENA_REAL, MPI_MIN,
                MPI_COMM_WORLD);
#endif
  if (invalid_fields != 0 || !std::isfinite(minimum_psi4) ||
      !std::isfinite(minimum_lapse) || minimum_psi4 <= 0.0 ||
      minimum_lapse <= 0.0) {
    Fail("IrisK spectral import contains nonfinite fields or a nonpositive "
         "conformal factor/lapse");
  }
  if (global_variable::my_rank == 0) {
    std::cout << std::setprecision(17)
              << "IrisK import field gates: finite=true min_psi4="
              << minimum_psi4 << " min_lapse=" << minimum_lapse
              << std::endl;
  }
  Kokkos::deep_copy(u_adm, host_u_adm);
  Kokkos::deep_copy(pmbp->pz4c->u0, host_u_z4c);
}

void FillAdmFromIrisSpectral(MeshBlockPack *pmbp,
                             IrisAthenakSpectralInterpolator *interpolator) {
  // Resolve cartesian3d versus cartoon_so2 once on the host; mapped field loops
  // below are compile-time specializations and add no evolution-time branch.
  z4c_irisk::AdmMap map;
  try {
    map = z4c_irisk::SelectAdmMap(pmbp->z4c_symmetry);
  } catch (const std::invalid_argument &error) {
    Fail(error.what());
  }
  switch (map) {
    case z4c_irisk::AdmMap::cartesian_xyz:
      FillAdmFromIrisSpectralMapped<z4c_irisk::AdmMap::cartesian_xyz>(
          pmbp, interpolator);
      return;
    case z4c_irisk::AdmMap::half_rho_z_suppressed_y_v2:
      FillAdmFromIrisSpectralMapped<
          z4c_irisk::AdmMap::half_rho_z_suppressed_y_v2>(pmbp, interpolator);
      return;
  }
  Fail("invalid IrisK ADM coordinate map");
}

void RecomputeAdmConstraints(MeshBlockPack *pmbp) {
  switch (pmbp->pz4c->opt.fd_stencil) {
    case 2:
      pmbp->pz4c->ADMConstraints<2>(pmbp);
      break;
    case 3:
      pmbp->pz4c->ADMConstraints<3>(pmbp);
      break;
    case 4:
      pmbp->pz4c->ADMConstraints<4>(pmbp);
      break;
    default:
      Fail("z4c_irisk_xcts supports Z4c stencil widths 2, 3, or 4");
  }
}

ConstraintRegionSummary ComputeConstraintRegionSummary(Mesh *pm,
                                                        const Real radius,
                                                        const bool proper_volume) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  const int nx1 = indcs.nx1;
  const int nx2 = indcs.nx2;
  const int nx3 = indcs.nx3;
  const int is = indcs.is;
  const int js = indcs.js;
  const int ks = indcs.ks;
  const int nmkji = pmbp->nmb_thispack * nx3 * nx2 * nx1;
  const int nkji = nx3 * nx2 * nx1;
  const int nji = nx2 * nx1;
  auto &u_con = pmbp->pz4c->u_con;
  auto &adm_vars = pmbp->padm->adm;
  auto &size = pmbp->pmb->mb_size;

  array_sum::GlobalSum local_sum;
  Kokkos::parallel_reduce(
      "irisk_xcts_constraint_region_summary",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int idx, array_sum::GlobalSum &sum) {
        const int m = idx / nkji;
        const int k0 = (idx - m * nkji) / nji;
        const int j0 = (idx - m * nkji - k0 * nji) / nx1;
        const int i = (idx - m * nkji - k0 * nji - j0 * nx1) + is;
        const int j = j0 + js;
        const int k = k0 + ks;
        const Real x = size.d_view(m).x1min +
                       (static_cast<Real>(i - is) + 0.5) *
                           size.d_view(m).dx1;
        const Real y = size.d_view(m).x2min +
                       (static_cast<Real>(j - js) + 0.5) *
                           size.d_view(m).dx2;
        const Real z = size.d_view(m).x3min +
                       (static_cast<Real>(k - ks) + 0.5) *
                           size.d_view(m).dx3;
        const bool in_region = x * x + y * y + z * z <= radius * radius;
        const Real detg = adm::SpatialDet(
            adm_vars.g_dd(m, 0, 0, k, j, i),
            adm_vars.g_dd(m, 0, 1, k, j, i),
            adm_vars.g_dd(m, 0, 2, k, j, i),
            adm_vars.g_dd(m, 1, 1, k, j, i),
            adm_vars.g_dd(m, 1, 2, k, j, i),
            adm_vars.g_dd(m, 2, 2, k, j, i));
        const Real coordinate_vol = size.d_view(m).dx1 * size.d_view(m).dx2 *
                                    size.d_view(m).dx3;
        const Real vol = coordinate_vol *
                         (proper_volume ? Kokkos::sqrt(Kokkos::abs(detg))
                                        : Real{1.0});
        array_sum::GlobalSum cell_sum;
        cell_sum.the_array[0] = in_region
                                    ? vol * u_con(m, z4c::Z4c::I_CON_C, k, j, i)
                                    : 0.0;
        cell_sum.the_array[1] = in_region
                                    ? vol * SQR(u_con(m, z4c::Z4c::I_CON_H, k, j, i))
                                    : 0.0;
        cell_sum.the_array[2] = in_region
                                    ? vol * u_con(m, z4c::Z4c::I_CON_M, k, j, i)
                                    : 0.0;
        cell_sum.the_array[3] = in_region
                                    ? vol * u_con(m, z4c::Z4c::I_CON_Z, k, j, i)
                                    : 0.0;
        cell_sum.the_array[4] = in_region ? vol : 0.0;
        cell_sum.the_array[5] = in_region ? 1.0 : 0.0;
        for (int n = 6; n < NREDUCTION_VARIABLES; ++n) {
          cell_sum.the_array[n] = 0.0;
        }
        sum += cell_sum;
      },
      Kokkos::Sum<array_sum::GlobalSum>(local_sum));

  Real totals[6];
  for (int n = 0; n < 6; ++n) {
    totals[n] = local_sum.the_array[n];
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, totals, 6, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
#endif

  Real h_linf = 0.0;
  Real m_linf = 0.0;
  Kokkos::parallel_reduce(
      "irisk_xcts_constraint_region_h_linf",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int idx, Real &maximum) {
        const int m = idx / nkji;
        const int k0 = (idx - m * nkji) / nji;
        const int j0 = (idx - m * nkji - k0 * nji) / nx1;
        const int i = (idx - m * nkji - k0 * nji - j0 * nx1) + is;
        const int j = j0 + js;
        const int k = k0 + ks;
        const Real x = size.d_view(m).x1min +
                       (static_cast<Real>(i - is) + 0.5) * size.d_view(m).dx1;
        const Real y = size.d_view(m).x2min +
                       (static_cast<Real>(j - js) + 0.5) * size.d_view(m).dx2;
        const Real z = size.d_view(m).x3min +
                       (static_cast<Real>(k - ks) + 0.5) * size.d_view(m).dx3;
        if (x * x + y * y + z * z <= radius * radius) {
          maximum = Kokkos::max(maximum,
                                Kokkos::abs(u_con(m, z4c::Z4c::I_CON_H, k, j, i)));
        }
      },
      Kokkos::Max<Real>(h_linf));
  Kokkos::parallel_reduce(
      "irisk_xcts_constraint_region_m_linf",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int idx, Real &maximum) {
        const int m = idx / nkji;
        const int k0 = (idx - m * nkji) / nji;
        const int j0 = (idx - m * nkji - k0 * nji) / nx1;
        const int i = (idx - m * nkji - k0 * nji - j0 * nx1) + is;
        const int j = j0 + js;
        const int k = k0 + ks;
        const Real x = size.d_view(m).x1min +
                       (static_cast<Real>(i - is) + 0.5) * size.d_view(m).dx1;
        const Real y = size.d_view(m).x2min +
                       (static_cast<Real>(j - js) + 0.5) * size.d_view(m).dx2;
        const Real z = size.d_view(m).x3min +
                       (static_cast<Real>(k - ks) + 0.5) * size.d_view(m).dx3;
        if (x * x + y * y + z * z <= radius * radius) {
          maximum = Kokkos::max(
              maximum,
              Kokkos::sqrt(Kokkos::max(Real{0.0},
                                        u_con(m, z4c::Z4c::I_CON_M, k, j, i))));
        }
      },
      Kokkos::Max<Real>(m_linf));
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &h_linf, 1, MPI_ATHENA_REAL, MPI_MAX,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &m_linf, 1, MPI_ATHENA_REAL, MPI_MAX,
                MPI_COMM_WORLD);
#endif

  ConstraintRegionSummary summary;
  summary.volume = totals[4];
  summary.cell_count = static_cast<std::uint64_t>(totals[5]);
  summary.h_linf = h_linf;
  summary.m_linf = m_linf;
  if (summary.volume > 0.0) {
    summary.c_rms = std::sqrt(totals[0] / summary.volume);
    summary.h_rms = std::sqrt(totals[1] / summary.volume);
    summary.m_rms = std::sqrt(totals[2] / summary.volume);
    summary.z_rms = std::sqrt(totals[3] / summary.volume);
  }
  return summary;
}

ConstraintSummary ComputeConstraintSummary(Mesh *pm) {
  constexpr Real kSupportRadius = 1.0;
  constexpr Real kWholeBoxRadius = std::numeric_limits<Real>::infinity();
  return {.coordinate_box =
              ComputeConstraintRegionSummary(pm, kWholeBoxRadius, false),
          .proper_box =
              ComputeConstraintRegionSummary(pm, kWholeBoxRadius, true),
          .coordinate_support =
              ComputeConstraintRegionSummary(pm, kSupportRadius, false),
          .proper_support =
              ComputeConstraintRegionSummary(pm, kSupportRadius, true)};
}

void WriteConstraintSummary(ParameterInput *pin, Mesh *pm,
                            const ConstraintSummary &summary) {
  if (global_variable::my_rank != 0) {
    return;
  }
  std::string filename =
      pin->GetOrAddString("problem", "constraint_summary_file", "AUTO");
  if (filename == "AUTO") {
    filename = pin->GetString("job", "basename");
    filename.append("-irisk-xcts-constraints.dat");
  }
  FILE *file = std::fopen(filename.c_str(), "r");
  if (file != nullptr) {
    file = std::freopen(filename.c_str(), "a", file);
  } else {
    file = std::fopen(filename.c_str(), "w");
    if (file != nullptr) {
      std::fprintf(
          file,
          "# Nx1 Nx2 Nx3 Ncycle region weighting sampled_volume cell_count "
          "C_rms H_rms M_rms Z_rms H_linf M_linf\n");
    }
  }
  if (file == nullptr) {
    Fail("IrisK constraint output file could not be opened: " + filename);
  }
  const auto write_region = [&](const char *region, const char *weighting,
                                const ConstraintRegionSummary &values) {
    std::fprintf(file,
                 "%04d %04d %04d %05d %s %s %.16e %llu %.16e %.16e %.16e "
                 "%.16e %.16e %.16e\n",
                 pm->mesh_indcs.nx1, pm->mesh_indcs.nx2,
                 pm->mesh_indcs.nx3, pm->ncycle, region, weighting,
                 values.volume,
                 static_cast<unsigned long long>(values.cell_count),
                 values.c_rms, values.h_rms, values.m_rms, values.z_rms,
                 values.h_linf, values.m_linf);
  };
  write_region("box", "coordinate", summary.coordinate_box);
  write_region("box", "proper", summary.proper_box);
  write_region("r<=1", "coordinate", summary.coordinate_support);
  write_region("r<=1", "proper", summary.proper_support);
  std::fclose(file);
}

void EnforceConstraintThresholds(ParameterInput *pin,
                                 const ConstraintSummary &summary) {
  const Real c_threshold = pin->GetOrAddReal(
      "problem", "fail_if_c_rms_above", std::numeric_limits<Real>::infinity());
  const Real h_threshold = pin->GetOrAddReal(
      "problem", "fail_if_h_rms_above", std::numeric_limits<Real>::infinity());
  const Real m_threshold = pin->GetOrAddReal(
      "problem", "fail_if_m_rms_above", std::numeric_limits<Real>::infinity());
  const Real z_threshold = pin->GetOrAddReal(
      "problem", "fail_if_z_rms_above", std::numeric_limits<Real>::infinity());
  const auto &box = summary.proper_box;
  if (box.c_rms > c_threshold || box.h_rms > h_threshold ||
      box.m_rms > m_threshold || box.z_rms > z_threshold) {
    Fail("IrisK imported proper-volume box constraints exceeded threshold: C=" +
         std::to_string(box.c_rms) + " H=" + std::to_string(box.h_rms) +
         " M=" + std::to_string(box.m_rms) + " Z=" +
         std::to_string(box.z_rms));
  }
}

bool FiniteConstraintRegionSummary(const ConstraintRegionSummary &summary) {
  return std::isfinite(summary.c_rms) && std::isfinite(summary.h_rms) &&
         std::isfinite(summary.m_rms) && std::isfinite(summary.z_rms) &&
         std::isfinite(summary.volume) && std::isfinite(summary.h_linf) &&
         std::isfinite(summary.m_linf) && summary.volume > 0.0 &&
         summary.cell_count > 0;
}

bool FiniteConstraintSummary(const ConstraintSummary &summary) {
  return FiniteConstraintRegionSummary(summary.coordinate_box) &&
         FiniteConstraintRegionSummary(summary.proper_box) &&
         FiniteConstraintRegionSummary(summary.coordinate_support) &&
         FiniteConstraintRegionSummary(summary.proper_support);
}

class CollapseTerminationMonitor {
 public:
  explicit CollapseTerminationMonitor(ParameterInput *pin)
      : pin_(pin),
        stop_on_horizon_(
            pin->GetOrAddBoolean("problem", "stop_on_horizon", false)),
        stop_on_dispersion_(
            pin->GetOrAddBoolean("problem", "stop_on_dispersion", false)),
        max_meshblocks_per_rank_stop_(pin->GetOrAddInteger(
            "problem", "max_meshblocks_per_rank_stop", 0)),
        check_interval_(
            pin->GetOrAddInteger("problem", "termination_check_interval", 8)),
        minimum_time_(
            pin->GetOrAddReal("problem", "dispersion_min_time", 10.0)),
        global_decay_(
            pin->GetOrAddReal("problem", "dispersion_global_decay", 0.05)),
        window_decay_(
            pin->GetOrAddReal("problem", "dispersion_window_decay", 0.5)),
        peak_abs_kretschmann_(pin->GetOrAddReal(
            "problem", "termination_peak_maxAbsKret", 0.0)),
        peak_abs_k_(pin->GetOrAddReal(
            "problem", "termination_peak_max_abs_K", 0.0)) {
    // Consume the superseded sample-count key so older decks remain parseable;
    // dispersal authority below uses the fixed coordinate-time windows [8,9]
    // and [9,10] instead.
    [[maybe_unused]] const int legacy_window_size =
        pin_->GetOrAddInteger("problem", "dispersion_window", 16);
    if (check_interval_ < 1) {
      Fail("problem.termination_check_interval must be positive");
    }
    if (max_meshblocks_per_rank_stop_ < 0) {
      Fail("problem.max_meshblocks_per_rank_stop must be nonnegative");
    }
    if (minimum_time_ < 10.0 || !std::isfinite(minimum_time_)) {
      Fail("problem.dispersion_min_time must be finite and at least 10");
    }
    if (!(global_decay_ > 0.0 && global_decay_ < 1.0) ||
        !(window_decay_ > 0.0 && window_decay_ < 1.0)) {
      Fail("problem dispersion decay factors must lie strictly between zero and one");
    }
  }

  std::string Check(Mesh *pm) {
    bool horizon_found = false;
    Real horizon_time = -1.0;
    if (stop_on_horizon_) {
      for (const auto &finder : pm->pmb_pack->pz4c->pfastflow) {
        if (finder->time_first_found >= 0.0) {
          horizon_found = true;
          horizon_time = finder->time_first_found;
          break;
        }
      }
    }
    int max_meshblocks_per_rank = pm->pmb_pack->nmb_thispack;
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &max_meshblocks_per_rank, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
#endif
    const bool meshblock_capacity_reached =
        max_meshblocks_per_rank_stop_ > 0 &&
        max_meshblocks_per_rank >= max_meshblocks_per_rank_stop_;
    // The resource guard is checked every cycle so the configured 180-block
    // clean stop cannot be skipped on the way to AthenaK's hard 200-block cap.
    if (!horizon_found && !meshblock_capacity_reached &&
        pm->ncycle % check_interval_ != 0) {
      return {};
    }

    const Z4cGlobalCurvatureMaxima maxima =
        ComputeZ4cGlobalCurvatureMaxima(pm);
    const ConstraintSummary constraints = ComputeConstraintSummary(pm);
    if (!maxima.finite || !FiniteConstraintSummary(constraints)) {
      Fail("nonfinite curvature or constraint diagnostic in collapse termination monitor");
    }
    if (horizon_found) {
      WriteTermination(pm, "collapse", maxima, constraints,
                       max_meshblocks_per_rank);
      std::ostringstream reason;
      reason << "confirmed apparent horizon at t=" << horizon_time;
      return reason.str();
    }
    if (meshblock_capacity_reached) {
      WriteTermination(pm, "meshblock_capacity", maxima, constraints,
                       max_meshblocks_per_rank);
      std::ostringstream reason;
      reason << "maximum per-rank MeshBlock occupancy reached "
             << max_meshblocks_per_rank;
      return reason.str();
    }
    if (!stop_on_dispersion_) {
      return {};
    }
    peak_abs_kretschmann_ =
        std::max(peak_abs_kretschmann_, maxima.max_abs_kretschmann);
    peak_abs_k_ = std::max(peak_abs_k_, maxima.max_abs_k);
    pin_->SetReal("problem", "termination_peak_maxAbsKret",
                  peak_abs_kretschmann_);
    pin_->SetReal("problem", "termination_peak_max_abs_K", peak_abs_k_);
    observations_.push_back(
        {pm->time, maxima.max_abs_kretschmann, maxima.max_abs_k});

    if (pm->time < minimum_time_) {
      return {};
    }
    Real first_abs_kretschmann = 0.0;
    Real second_abs_kretschmann = 0.0;
    Real first_abs_k = 0.0;
    Real second_abs_k = 0.0;
    if (!MaxWindow(8.0, 9.0, &first_abs_kretschmann, &first_abs_k) ||
        !MaxWindow(9.0, 10.0, &second_abs_kretschmann, &second_abs_k)) {
      return {};
    }
    const bool curvature_decayed =
        peak_abs_kretschmann_ > 0.0 &&
        maxima.max_abs_kretschmann <=
            global_decay_ * peak_abs_kretschmann_ &&
        second_abs_kretschmann <=
            window_decay_ * first_abs_kretschmann;
    const bool extrinsic_curvature_decayed =
        peak_abs_k_ > 0.0 &&
        maxima.max_abs_k <= global_decay_ * peak_abs_k_ &&
        second_abs_k <= window_decay_ * first_abs_k;
    if (!curvature_decayed || !extrinsic_curvature_decayed) {
      return {};
    }

    WriteTermination(pm, "dispersal", maxima, constraints,
                     max_meshblocks_per_rank);
    std::ostringstream reason;
    reason << "sustained dispersal at t=" << pm->time
           << " maxAbsKret/peak="
           << maxima.max_abs_kretschmann / peak_abs_kretschmann_
           << " max_abs_K/peak=" << maxima.max_abs_k / peak_abs_k_;
    return reason.str();
  }

  bool enabled() const {
    return stop_on_horizon_ || stop_on_dispersion_ ||
           max_meshblocks_per_rank_stop_ > 0;
  }

 private:
  struct CurvatureObservation {
    Real time;
    Real max_abs_kretschmann;
    Real max_abs_k;
  };

  bool MaxWindow(Real begin, Real end, Real *max_abs_kretschmann,
                 Real *max_abs_k) const {
    bool found = false;
    *max_abs_kretschmann = 0.0;
    *max_abs_k = 0.0;
    for (const auto &observation : observations_) {
      if (observation.time < begin || observation.time > end) {
        continue;
      }
      found = true;
      *max_abs_kretschmann =
          std::max(*max_abs_kretschmann, observation.max_abs_kretschmann);
      *max_abs_k = std::max(*max_abs_k, observation.max_abs_k);
    }
    return found;
  }

  void WriteTermination(
      Mesh *pm, const std::string &outcome,
      const Z4cGlobalCurvatureMaxima &maxima,
      const ConstraintSummary &constraints,
      int max_meshblocks_per_rank) const {
    if (global_variable::my_rank != 0) {
      return;
    }
    const std::string filename =
        pin_->GetString("job", "basename") + ".termination.json";
    std::ofstream output(filename);
    if (!output) {
      Fail("could not write collapse termination record: " + filename);
    }
    const auto &proper_box = constraints.proper_box;
    output << std::setprecision(17)
           << "{\n"
           << "  \"schema_version\": 2,\n"
           << "  \"outcome\": \"" << outcome << "\",\n"
           << "  \"time\": " << pm->time << ",\n"
           << "  \"cycle\": " << pm->ncycle << ",\n"
           << "  \"max_meshblocks_per_rank\": "
           << max_meshblocks_per_rank << ",\n"
           << "  \"max_abs_Kretschmann\": "
           << maxima.max_abs_kretschmann << ",\n"
           << "  \"max_abs_K\": " << maxima.max_abs_k << ",\n"
           << "  \"constraint_weighting\": \"proper_volume\",\n"
           << "  \"C_rms\": " << proper_box.c_rms << ",\n"
           << "  \"H_rms\": " << proper_box.h_rms << ",\n"
           << "  \"M_rms\": " << proper_box.m_rms << ",\n"
           << "  \"Z_rms\": " << proper_box.z_rms << ",\n"
           << "  \"volume\": " << proper_box.volume << "\n"
           << "}\n";
  }

  ParameterInput *pin_;
  bool stop_on_horizon_;
  bool stop_on_dispersion_;
  int max_meshblocks_per_rank_stop_;
  int check_interval_;
  Real minimum_time_;
  Real global_decay_;
  Real window_decay_;
  Real peak_abs_kretschmann_;
  Real peak_abs_k_;
  std::vector<CurvatureObservation> observations_;
};

void ConfigureCollapseTermination(ProblemGenerator *problem,
                                  ParameterInput *pin, Mesh *pm) {
  auto monitor = std::make_shared<CollapseTerminationMonitor>(pin);
  if (!monitor->enabled()) {
    return;
  }
  if (pm->pmb_pack->pz4c == nullptr || pm->pmb_pack->padm == nullptr) {
    Fail("collapse termination requires Z4c and ADM");
  }
  if (pin->GetOrAddBoolean("problem", "stop_on_horizon", false) &&
      pm->pmb_pack->pz4c->pfastflow.empty()) {
    Fail("problem.stop_on_horizon=true requires at least one FastFlow horizon");
  }
  if (pin->GetOrAddBoolean("problem", "stop_on_dispersion", false) &&
      pm->pmb_pack->pz4c->opt.fd_stencil != 4) {
    Fail("problem.stop_on_dispersion=true requires sixth-order Z4c with four ghosts");
  }
  problem->user_stopping_condition =
      [monitor](Mesh *mesh) { return monitor->Check(mesh); };
}

void IrisXctsConstraintReport(ParameterInput *pin, Mesh *pm) {
  if (pm->pmb_pack->pz4c == nullptr) {
    return;
  }
  RecomputeAdmConstraints(pm->pmb_pack);
  const ConstraintSummary summary = ComputeConstraintSummary(pm);
  if (pin->GetOrAddBoolean("problem", "write_constraint_summary", true)) {
    WriteConstraintSummary(pin, pm, summary);
  }
  EnforceConstraintThresholds(pin, summary);
}

void WriteImportedActiveCellVolume(ParameterInput *pin, MeshBlockPack *pmbp) {
  const std::string path =
      pin->GetOrAddString("problem", "volume_output", "EMPTY");
  if (path == "EMPTY" || path.empty()) {
    return;
  }
  if (global_variable::nranks != 1) {
    throw std::runtime_error(
        "problem.volume_output currently requires exactly one MPI rank; "
        "rank-local volume files are not a complete active-cell box");
  }

  const std::filesystem::path output_path(path);
  if (!output_path.parent_path().empty()) {
    std::filesystem::create_directories(output_path.parent_path());
  }
  std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("failed to open active-cell volume output: " +
                             output_path.string());
  }

  auto host_adm =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmbp->padm->u_adm);
  auto host_z4c =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmbp->pz4c->u0);
  auto host_constraints =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmbp->pz4c->u_con);
  pmbp->pmb->mb_size.sync_host();
  pmbp->pmb->mb_gid.sync_host();
  auto size = pmbp->pmb->mb_size.h_view;
  auto gids = pmbp->pmb->mb_gid.h_view;
  auto &indcs = pmbp->pmesh->mb_indcs;

  const std::uint64_t block_count =
      static_cast<std::uint64_t>(pmbp->nmb_thispack);
  const std::uint64_t nx1 = static_cast<std::uint64_t>(indcs.nx1);
  const std::uint64_t nx2 = static_cast<std::uint64_t>(indcs.nx2);
  const std::uint64_t nx3 = static_cast<std::uint64_t>(indcs.nx3);
  const std::uint64_t cell_count = block_count * nx1 * nx2 * nx3;

  output.write(kVolumeOutputMagic.data(),
               static_cast<std::streamsize>(kVolumeOutputMagic.size()));
  WriteVolumeScalar(output, kVolumeOutputVersion);
  WriteVolumeScalar(output, kVolumeOutputEndianTag);
  WriteVolumeScalar(output, kVolumeOutputIntegerCount);
  WriteVolumeScalar(output, kVolumeOutputRealCount);
  WriteVolumeScalar(output, block_count);
  WriteVolumeScalar(output, nx1);
  WriteVolumeScalar(output, nx2);
  WriteVolumeScalar(output, nx3);
  WriteVolumeScalar(output, cell_count);
  const std::string labels =
      "int32:gid,block,i,j,k\n"
      "float64:x,y,z,gxx,gxy,gxz,gyy,gyz,gzz,Kxx,Kxy,Kxz,Kyy,Kyz,Kzz,"
      "psi,alpha,betax,betay,betaz,"
      "z4c_chi,z4c_gxx,z4c_gxy,z4c_gxz,z4c_gyy,z4c_gyz,z4c_gzz,"
      "z4c_Khat,z4c_Axx,z4c_Axy,z4c_Axz,z4c_Ayy,z4c_Ayz,z4c_Azz,"
      "z4c_Gamx,z4c_Gamy,z4c_Gamz,z4c_Theta,z4c_alpha,"
      "z4c_betax,z4c_betay,z4c_betaz,z4c_Bx,z4c_By,z4c_Bz,"
      "H,M_norm,C_norm,Z_norm,sqrt_gamma,coordinate_cell_volume\n";
  const std::uint64_t label_bytes =
      static_cast<std::uint64_t>(labels.size());
  WriteVolumeScalar(output, label_bytes);
  output.write(labels.data(), static_cast<std::streamsize>(labels.size()));
  if (!output) {
    throw std::runtime_error("failed while writing active-cell volume header");
  }

  std::array<double, kVolumeOutputRealCount> record{};
  for (int m = 0; m < pmbp->nmb_thispack; ++m) {
    const double coordinate_cell_volume =
        static_cast<double>(size(m).dx1 * size(m).dx2 * size(m).dx3);
    for (int k = indcs.ks; k <= indcs.ke; ++k) {
      const std::int32_t local_k = static_cast<std::int32_t>(k - indcs.ks);
      const double z =
          CellCenterX(local_k, indcs.nx3, size(m).x3min, size(m).x3max);
      for (int j = indcs.js; j <= indcs.je; ++j) {
        const std::int32_t local_j = static_cast<std::int32_t>(j - indcs.js);
        const double y =
            CellCenterX(local_j, indcs.nx2, size(m).x2min, size(m).x2max);
        for (int i = indcs.is; i <= indcs.ie; ++i) {
          const std::int32_t local_i = static_cast<std::int32_t>(i - indcs.is);
          const double x =
              CellCenterX(local_i, indcs.nx1, size(m).x1min, size(m).x1max);
          const double gxx = host_adm(m, adm::ADM::I_ADM_GXX, k, j, i);
          const double gxy = host_adm(m, adm::ADM::I_ADM_GXY, k, j, i);
          const double gxz = host_adm(m, adm::ADM::I_ADM_GXZ, k, j, i);
          const double gyy = host_adm(m, adm::ADM::I_ADM_GYY, k, j, i);
          const double gyz = host_adm(m, adm::ADM::I_ADM_GYZ, k, j, i);
          const double gzz = host_adm(m, adm::ADM::I_ADM_GZZ, k, j, i);
          const double determinant =
              adm::SpatialDet(gxx, gxy, gxz, gyy, gyz, gzz);
          const double psi4 = host_adm(m, adm::ADM::I_ADM_PSI4, k, j, i);
          if (!(std::isfinite(determinant) && determinant > 0.0 &&
                std::isfinite(psi4) && psi4 > 0.0)) {
            throw std::runtime_error(
                "active-cell volume encountered invalid ADM metric or psi4");
          }

          std::size_t field = 0;
          record[field++] = x;
          record[field++] = y;
          record[field++] = z;
          for (int variable = adm::ADM::I_ADM_GXX;
               variable <= adm::ADM::I_ADM_GZZ; ++variable) {
            record[field++] = host_adm(m, variable, k, j, i);
          }
          for (int variable = adm::ADM::I_ADM_KXX;
               variable <= adm::ADM::I_ADM_KZZ; ++variable) {
            record[field++] = host_adm(m, variable, k, j, i);
          }
          record[field++] = std::pow(psi4, 0.25);
          record[field++] =
              host_z4c(m, z4c::Z4c::I_Z4C_ALPHA, k, j, i);
          for (int variable = z4c::Z4c::I_Z4C_BETAX;
               variable <= z4c::Z4c::I_Z4C_BETAZ; ++variable) {
            record[field++] = host_z4c(m, variable, k, j, i);
          }
          for (int variable = 0; variable < z4c::Z4c::nz4c; ++variable) {
            record[field++] = host_z4c(m, variable, k, j, i);
          }
          record[field++] =
              host_constraints(m, z4c::Z4c::I_CON_H, k, j, i);
          record[field++] = std::sqrt(std::max(
              0.0, static_cast<double>(
                       host_constraints(m, z4c::Z4c::I_CON_M, k, j, i))));
          record[field++] = std::sqrt(std::max(
              0.0, static_cast<double>(
                       host_constraints(m, z4c::Z4c::I_CON_C, k, j, i))));
          record[field++] = std::sqrt(std::max(
              0.0, static_cast<double>(
                       host_constraints(m, z4c::Z4c::I_CON_Z, k, j, i))));
          record[field++] = std::sqrt(determinant);
          record[field++] = coordinate_cell_volume;
          if (field != record.size() ||
              !std::all_of(record.begin(), record.end(),
                           [](const double value) {
                             return std::isfinite(value);
                           })) {
            throw std::runtime_error(
                "active-cell volume encountered invalid or incomplete data");
          }

          const std::array<std::int32_t, kVolumeOutputIntegerCount> indices{
              static_cast<std::int32_t>(gids(m)),
              static_cast<std::int32_t>(m),
              local_i,
              local_j,
              local_k};
          output.write(reinterpret_cast<const char *>(indices.data()),
                       static_cast<std::streamsize>(
                           indices.size() * sizeof(indices.front())));
          output.write(reinterpret_cast<const char *>(record.data()),
                       static_cast<std::streamsize>(
                           record.size() * sizeof(record.front())));
          if (!output) {
            throw std::runtime_error(
                "failed while writing active-cell volume payload");
          }
        }
      }
    }
  }
  output.close();
  if (!output) {
    throw std::runtime_error("failed to finalize active-cell volume output");
  }
}

void WriteImportedAdmPlane(ParameterInput *pin, MeshBlockPack *pmbp,
                           const bool xy_plane, const bool negative_side) {
  const char *path_key = xy_plane
                             ? (negative_side ? "xy_plane_output_minus"
                                              : "xy_plane_output")
                             : (negative_side ? "xz_plane_output_minus"
                                              : "xz_plane_output");
  const char *coordinate_key = xy_plane
                                   ? (negative_side ? "xy_plane_z_minus"
                                                    : "xy_plane_z")
                                   : (negative_side ? "xz_plane_y_minus"
                                                    : "xz_plane_y");
  const std::string path =
      pin->GetOrAddString("problem", path_key, "EMPTY");
  if (path == "EMPTY" || path.empty() || global_variable::my_rank != 0) {
    return;
  }
  const Real plane_coordinate =
      pin->GetOrAddReal("problem", coordinate_key, 0.0);
  const std::filesystem::path output_path(path);
  if (!output_path.parent_path().empty()) {
    std::filesystem::create_directories(output_path.parent_path());
  }
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("failed to open imported-ADM plane output: " +
                             path);
  }
  auto host_adm =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmbp->padm->u_adm);
  auto host_z4c =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmbp->pz4c->u0);
  auto host_constraints =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmbp->pz4c->u_con);
  pmbp->pmb->mb_size.sync_host();
  pmbp->pmb->mb_gid.sync_host();
  pmbp->pmb->mb_lev.sync_host();
  auto size = pmbp->pmb->mb_size.h_view;
  auto gids = pmbp->pmb->mb_gid.h_view;
  auto levels = pmbp->pmb->mb_lev.h_view;
  auto &indcs = pmbp->pmesh->mb_indcs;
  output << std::setprecision(17);
  output << "# x y z psi alpha beta_norm trK gxx_minus_one gxy Kxx Kxy "
            "chi Khat Theta Gam_norm A_norm B_norm H M_norm C_norm Z_norm "
            "level gid block i j k\n";
  for (int m = 0; m < pmbp->nmb_thispack; ++m) {
    const Real block_min = xy_plane ? size(m).x3min : size(m).x2min;
    const Real block_max = xy_plane ? size(m).x3max : size(m).x2max;
    const Real block_dx = xy_plane ? size(m).dx3 : size(m).dx2;
    if (plane_coordinate < block_min || plane_coordinate > block_max) {
      continue;
    }
    const int normal_begin = xy_plane ? indcs.ks : indcs.js;
    const int normal_n = xy_plane ? indcs.nx3 : indcs.nx2;
    const int samples = std::min(8, normal_n);
    const Real fractional_index =
        (plane_coordinate - block_min) / block_dx - 0.5;
    int sample_start =
        static_cast<int>(std::floor(fractional_index)) - samples / 2 + 1;
    sample_start = std::max(0, std::min(normal_n - samples, sample_start));
    std::vector<Real> normal_weights(static_cast<std::size_t>(samples), 1.0);
    for (int a = 0; a < samples; ++a) {
      const Real node_a = static_cast<Real>(sample_start + a);
      for (int b = 0; b < samples; ++b) {
        if (a == b) {
          continue;
        }
        const Real node_b = static_cast<Real>(sample_start + b);
        normal_weights[static_cast<std::size_t>(a)] *=
            (fractional_index - node_b) / (node_a - node_b);
      }
    }
    const int normal_index =
        normal_begin +
        std::clamp(static_cast<int>(std::llround(fractional_index)), 0,
                   normal_n - 1);
    const auto interpolate =
        [&](const auto &view, const int variable, const int k, const int j,
            const int i) {
          Real value = 0.0;
          for (int sample = 0; sample < samples; ++sample) {
            const int index = normal_begin + sample_start + sample;
            const int sample_k = xy_plane ? index : k;
            const int sample_j = xy_plane ? j : index;
            value += normal_weights[static_cast<std::size_t>(sample)] *
                     view(m, variable, sample_k, sample_j, i);
          }
          return value;
        };
    const int k_begin = xy_plane ? normal_index : indcs.ks;
    const int k_end = xy_plane ? normal_index : indcs.ke;
    const int j_begin = xy_plane ? indcs.js : normal_index;
    const int j_end = xy_plane ? indcs.je : normal_index;
    for (int k = k_begin; k <= k_end; ++k) {
      const Real z =
          xy_plane ? plane_coordinate
                   : CellCenterX(k - indcs.ks, indcs.nx3, size(m).x3min,
                                 size(m).x3max);
      for (int j = j_begin; j <= j_end; ++j) {
        const Real y =
            xy_plane ? CellCenterX(j - indcs.js, indcs.nx2, size(m).x2min,
                                   size(m).x2max)
                     : plane_coordinate;
        for (int i = indcs.is; i <= indcs.ie; ++i) {
          const Real x =
              CellCenterX(i - indcs.is, indcs.nx1, size(m).x1min,
                          size(m).x1max);
          const Real psi4 =
              interpolate(host_adm, adm::ADM::I_ADM_PSI4, k, j, i);
          const Real alpha =
              interpolate(host_z4c, z4c::Z4c::I_Z4C_ALPHA, k, j, i);
          const Real bx =
              interpolate(host_z4c, z4c::Z4c::I_Z4C_BETAX, k, j, i);
          const Real by =
              interpolate(host_z4c, z4c::Z4c::I_Z4C_BETAY, k, j, i);
          const Real bz =
              interpolate(host_z4c, z4c::Z4c::I_Z4C_BETAZ, k, j, i);
          const Real gxx =
              interpolate(host_adm, adm::ADM::I_ADM_GXX, k, j, i);
          const Real gxy =
              interpolate(host_adm, adm::ADM::I_ADM_GXY, k, j, i);
          const Real gxz =
              interpolate(host_adm, adm::ADM::I_ADM_GXZ, k, j, i);
          const Real gyy =
              interpolate(host_adm, adm::ADM::I_ADM_GYY, k, j, i);
          const Real gyz =
              interpolate(host_adm, adm::ADM::I_ADM_GYZ, k, j, i);
          const Real gzz =
              interpolate(host_adm, adm::ADM::I_ADM_GZZ, k, j, i);
          const Real beta_squared =
              gxx * bx * bx + gyy * by * by + gzz * bz * bz +
              2.0 * gxy * bx * by + 2.0 * gxz * bx * bz +
              2.0 * gyz * by * bz;
          const Real determinant =
              gxx * (gyy * gzz - gyz * gyz) -
              gxy * (gxy * gzz - gxz * gyz) +
              gxz * (gxy * gyz - gxz * gyy);
          const Real kxx =
              interpolate(host_adm, adm::ADM::I_ADM_KXX, k, j, i);
          const Real kxy =
              interpolate(host_adm, adm::ADM::I_ADM_KXY, k, j, i);
          const Real kxz =
              interpolate(host_adm, adm::ADM::I_ADM_KXZ, k, j, i);
          const Real kyy =
              interpolate(host_adm, adm::ADM::I_ADM_KYY, k, j, i);
          const Real kyz =
              interpolate(host_adm, adm::ADM::I_ADM_KYZ, k, j, i);
          const Real kzz =
              interpolate(host_adm, adm::ADM::I_ADM_KZZ, k, j, i);
          const Real trace_k =
              ((gyy * gzz - gyz * gyz) * kxx +
               (gxx * gzz - gxz * gxz) * kyy +
               (gxx * gyy - gxy * gxy) * kzz +
               2.0 * (gxz * gyz - gxy * gzz) * kxy +
               2.0 * (gxy * gyz - gxz * gyy) * kxz +
               2.0 * (gxy * gxz - gxx * gyz) * kyz) /
              determinant;
          const Real gamx =
              interpolate(host_z4c, z4c::Z4c::I_Z4C_GAMX, k, j, i);
          const Real gamy =
              interpolate(host_z4c, z4c::Z4c::I_Z4C_GAMY, k, j, i);
          const Real gamz =
              interpolate(host_z4c, z4c::Z4c::I_Z4C_GAMZ, k, j, i);
          const Real axx =
              interpolate(host_z4c, z4c::Z4c::I_Z4C_AXX, k, j, i);
          const Real axy =
              interpolate(host_z4c, z4c::Z4c::I_Z4C_AXY, k, j, i);
          const Real axz =
              interpolate(host_z4c, z4c::Z4c::I_Z4C_AXZ, k, j, i);
          const Real ayy =
              interpolate(host_z4c, z4c::Z4c::I_Z4C_AYY, k, j, i);
          const Real ayz =
              interpolate(host_z4c, z4c::Z4c::I_Z4C_AYZ, k, j, i);
          const Real azz =
              interpolate(host_z4c, z4c::Z4c::I_Z4C_AZZ, k, j, i);
          const Real bxi =
              interpolate(host_z4c, z4c::Z4c::I_Z4C_BX, k, j, i);
          const Real byi =
              interpolate(host_z4c, z4c::Z4c::I_Z4C_BY, k, j, i);
          const Real bzi =
              interpolate(host_z4c, z4c::Z4c::I_Z4C_BZ, k, j, i);
          output
              << x << ' ' << y << ' ' << z << ' ' << std::pow(psi4, 0.25)
              << ' ' << alpha << ' '
              << std::sqrt(std::max(Real{0.0}, beta_squared)) << ' '
              << trace_k << ' ' << gxx - 1.0 << ' ' << gxy << ' ' << kxx
              << ' ' << kxy << ' '
              << interpolate(host_z4c, z4c::Z4c::I_Z4C_CHI, k, j, i) << ' '
              << interpolate(host_z4c, z4c::Z4c::I_Z4C_KHAT, k, j, i) << ' '
              << interpolate(host_z4c, z4c::Z4c::I_Z4C_THETA, k, j, i) << ' '
              << std::sqrt(gamx * gamx + gamy * gamy + gamz * gamz) << ' '
              << std::sqrt(axx * axx + ayy * ayy + azz * azz +
                           2.0 * (axy * axy + axz * axz + ayz * ayz)) << ' '
              << std::sqrt(bxi * bxi + byi * byi + bzi * bzi) << ' '
              << interpolate(host_constraints, z4c::Z4c::I_CON_H, k, j, i)
              << ' '
              << std::sqrt(std::max(
                     Real{0.0}, interpolate(host_constraints,
                                            z4c::Z4c::I_CON_M, k, j, i)))
              << ' '
              << std::sqrt(std::max(
                     Real{0.0}, interpolate(host_constraints,
                                            z4c::Z4c::I_CON_C, k, j, i)))
              << ' '
              << std::sqrt(std::max(
                     Real{0.0}, interpolate(host_constraints,
                                            z4c::Z4c::I_CON_Z, k, j, i)))
              << ' ' << levels(m) - pmbp->pmesh->root_level << ' ' << gids(m)
              << ' ' << m << ' ' << i << ' ' << j << ' ' << k << '\n';
        }
      }
    }
  }
}

} // namespace

void IrisXctsRefinementCondition(MeshBlockPack *pmbp) {
  pmbp->pz4c->pamr->Refine(pmbp);
}

void ProblemGenerator::Z4cFinalizeImportedAdm(ParameterInput *pin) {
  pgen_final_func = IrisXctsConstraintReport;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  switch (pmbp->pz4c->opt.fd_stencil) {
    case 2:
      pmbp->pz4c->ADMToZ4c<2>(pmbp, pin);
      break;
    case 3:
      pmbp->pz4c->ADMToZ4c<3>(pmbp, pin);
      break;
    case 4:
      pmbp->pz4c->ADMToZ4c<4>(pmbp, pin);
      break;
    default:
      Fail("z4c_irisk_xcts supports Z4c stencil widths 2, 3, or 4");
  }
  pmbp->pz4c->ReconstructAxisParityGhosts();
  pmbp->pz4c->Z4cToADM(pmbp);
  RecomputeAdmConstraints(pmbp);
  const ConstraintSummary summary = ComputeConstraintSummary(pmy_mesh_);
  EnforceConstraintThresholds(pin, summary);
  try {
    WriteImportedActiveCellVolume(pin, pmbp);
    WriteImportedAdmPlane(pin, pmbp, true, false);
    WriteImportedAdmPlane(pin, pmbp, true, true);
    WriteImportedAdmPlane(pin, pmbp, false, false);
    WriteImportedAdmPlane(pin, pmbp, false, true);
  } catch (const std::exception &error) {
    Fail(std::string("failed to write imported-ADM plane: ") + error.what());
  }
  if (global_variable::my_rank == 0) {
    const auto &proper_box = summary.proper_box;
    const auto &coordinate_support = summary.coordinate_support;
    std::cout << "Converted imported ADM data to Z4c"
              << " proper_box_C_rms=" << proper_box.c_rms
              << " proper_box_H_rms=" << proper_box.h_rms
              << " proper_box_M_rms=" << proper_box.m_rms
              << " proper_box_Z_rms=" << proper_box.z_rms
              << " coordinate_support_H_rms=" << coordinate_support.h_rms
              << " coordinate_support_M_rms=" << coordinate_support.m_rms
              << std::endl;
  }
}

void ProblemGenerator::Z4cIrisXcts(ParameterInput *pin, const bool restart) {
  // Enroll on both fresh starts and restarts so adaptive runs retain their criterion.
  user_ref_func = IrisXctsRefinementCondition;
  ConfigureCollapseTermination(this, pin, pmy_mesh_);
  if (restart)
    return;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pz4c == nullptr || pmbp->padm == nullptr) {
    Fail("z4c_irisk_xcts requires both <z4c> and <adm> blocks");
  }
  const std::string filename =
      pin->GetOrAddString("problem", "irisk_adm_spectral_file", "EMPTY");
  if (filename == "EMPTY" || filename.empty()) {
    Fail("z4c_irisk_xcts requires problem.irisk_adm_spectral_file");
  }
  const std::filesystem::path resolved_filename =
      ResolveSpectralInputPath(filename);
  const std::string resolved_filename_string = resolved_filename.string();
  const std::uintmax_t spectral_file_bytes =
      std::filesystem::file_size(resolved_filename);
  IrisAthenakSpectralInterpolator *interpolator = nullptr;
  std::array<char, 1024> error{};
  if (IrisAthenakSpectralOpen(resolved_filename_string.c_str(), &interpolator,
                              error.data(), error.size()) != 0) {
    Fail("failed to open IrisK spectral data '" + resolved_filename_string +
         "': " + error.data());
  }
  FillAdmFromIrisSpectral(pmbp, interpolator);
  IrisAthenakSpectralClose(interpolator);

  // Match the established puncture import sequence, while preserving the
  // elliptically solved XCTS lapse and shift rather than imposing a puncture
  // pre-collapsed lapse.
  Z4cFinalizeImportedAdm(pin);
  if (global_variable::my_rank == 0) {
    std::cout << "Initialized Z4c from IrisK spectral XCTS data: "
              << resolved_filename_string
              << " bytes=" << spectral_file_bytes << std::endl;
  }
}
