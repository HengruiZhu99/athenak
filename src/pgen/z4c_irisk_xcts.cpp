//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE for details
//========================================================================================
//! \file z4c_irisk_xcts.cpp
//! \brief Spectrally interpolate IrisK XCTS data onto an arbitrary AMR mesh.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
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
#include "z4c/z4c.hpp"

namespace {

struct ConstraintSummary {
  Real c_rms = 0.0;
  Real h_rms = 0.0;
  Real m_rms = 0.0;
  Real z_rms = 0.0;
  Real volume = 0.0;
};

[[noreturn]] void Fail(const std::string &message) {
  std::cout << "### FATAL ERROR in " << __FILE__ << std::endl
            << message << std::endl;
  std::exit(EXIT_FAILURE);
}

void FillAdmFromIrisSpectral(MeshBlockPack *pmbp,
                             IrisAthenakSpectralInterpolator *interpolator) {
  auto &u_adm = pmbp->padm->u_adm;
  HostArray5D<Real>::HostMirror host_u_adm = create_mirror(u_adm);
  HostArray5D<Real>::HostMirror host_u_z4c = create_mirror(pmbp->pz4c->u0);
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
  auto size = pmbp->pmb->mb_size.h_view;
  const int isg = indcs.is - indcs.ng;
  const int ieg = indcs.ie + indcs.ng;
  const int jsg = indcs.js - indcs.ng;
  const int jeg = indcs.je + indcs.ng;
  const int ksg = indcs.ks - indcs.ng;
  const int keg = indcs.ke + indcs.ng;
  const std::size_t nx = static_cast<std::size_t>(ieg - isg + 1);
  const std::size_t ny = static_cast<std::size_t>(jeg - jsg + 1);
  const std::size_t nz = static_cast<std::size_t>(keg - ksg + 1);

  for (int m = 0; m < pmbp->nmb_thispack; ++m) {
    std::vector<double> x(nx), y(ny), z(nz);
    for (int i = isg; i <= ieg; ++i) {
      x[static_cast<std::size_t>(i - isg)] =
          CellCenterX(i - indcs.is, indcs.nx1, size(m).x1min, size(m).x1max);
    }
    for (int j = jsg; j <= jeg; ++j) {
      y[static_cast<std::size_t>(j - jsg)] =
          CellCenterX(j - indcs.js, indcs.nx2, size(m).x2min, size(m).x2max);
    }
    for (int k = ksg; k <= keg; ++k) {
      z[static_cast<std::size_t>(k - ksg)] =
          CellCenterX(k - indcs.ks, indcs.nx3, size(m).x3min, size(m).x3max);
    }
    std::vector<double> values(nx * ny * nz * IRISK_ATHENAK_ADM_VARIABLE_COUNT);
    std::array<char, 1024> error{};
    if (IrisAthenakSpectralInterpolateCartesian(
            interpolator, nx, ny, nz, x.data(), y.data(), z.data(),
            values.data(), error.data(), error.size()) != 0) {
      Fail(std::string("IrisK spectral interpolation failed: ") + error.data());
    }

    for (int k = ksg; k <= keg; ++k)
      for (int j = jsg; j <= jeg; ++j)
        for (int i = isg; i <= ieg; ++i) {
          const std::size_t point =
              static_cast<std::size_t>(i - isg) +
              nx * (static_cast<std::size_t>(j - jsg) +
                    ny * static_cast<std::size_t>(k - ksg));
          const double *value =
              values.data() + point * IRISK_ATHENAK_ADM_VARIABLE_COUNT;
          host_adm.g_dd(m, 0, 0, k, j, i) = value[0];
          host_adm.g_dd(m, 0, 1, k, j, i) = value[1];
          host_adm.g_dd(m, 0, 2, k, j, i) = value[2];
          host_adm.g_dd(m, 1, 1, k, j, i) = value[3];
          host_adm.g_dd(m, 1, 2, k, j, i) = value[4];
          host_adm.g_dd(m, 2, 2, k, j, i) = value[5];
          host_adm.vK_dd(m, 0, 0, k, j, i) = value[6];
          host_adm.vK_dd(m, 0, 1, k, j, i) = value[7];
          host_adm.vK_dd(m, 0, 2, k, j, i) = value[8];
          host_adm.vK_dd(m, 1, 1, k, j, i) = value[9];
          host_adm.vK_dd(m, 1, 2, k, j, i) = value[10];
          host_adm.vK_dd(m, 2, 2, k, j, i) = value[11];
          host_adm.psi4(m, k, j, i) = value[12];
          host_adm.alpha(m, k, j, i) = value[13];
          for (int component = 0; component < 3; ++component) {
            host_adm.beta_u(m, component, k, j, i) = value[14 + component];
          }
        }
  }
  Kokkos::deep_copy(u_adm, host_u_adm);
  Kokkos::deep_copy(pmbp->pz4c->u0, host_u_z4c);
}

void RecomputeAdmConstraints(MeshBlockPack *pmbp) {
  switch (pmbp->pmesh->mb_indcs.ng) {
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
      Fail("z4c_irisk_xcts supports nghost = 2, 3, or 4");
  }
}

ConstraintSummary ComputeConstraintSummary(Mesh *pm) {
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
      "irisk_xcts_constraint_summary",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int idx, array_sum::GlobalSum &sum) {
        const int m = idx / nkji;
        const int k0 = (idx - m * nkji) / nji;
        const int j0 = (idx - m * nkji - k0 * nji) / nx1;
        const int i = (idx - m * nkji - k0 * nji - j0 * nx1) + is;
        const int j = j0 + js;
        const int k = k0 + ks;
        const Real detg = adm::SpatialDet(
            adm_vars.g_dd(m, 0, 0, k, j, i),
            adm_vars.g_dd(m, 0, 1, k, j, i),
            adm_vars.g_dd(m, 0, 2, k, j, i),
            adm_vars.g_dd(m, 1, 1, k, j, i),
            adm_vars.g_dd(m, 1, 2, k, j, i),
            adm_vars.g_dd(m, 2, 2, k, j, i));
        const Real vol = size.d_view(m).dx1 * size.d_view(m).dx2 *
                         size.d_view(m).dx3 *
                         Kokkos::sqrt(Kokkos::abs(detg));
        array_sum::GlobalSum cell_sum;
        cell_sum.the_array[0] =
            vol * u_con(m, z4c::Z4c::I_CON_C, k, j, i);
        cell_sum.the_array[1] =
            vol * SQR(u_con(m, z4c::Z4c::I_CON_H, k, j, i));
        cell_sum.the_array[2] =
            vol * u_con(m, z4c::Z4c::I_CON_M, k, j, i);
        cell_sum.the_array[3] =
            vol * u_con(m, z4c::Z4c::I_CON_Z, k, j, i);
        cell_sum.the_array[4] = vol;
        for (int n = 5; n < NREDUCTION_VARIABLES; ++n) {
          cell_sum.the_array[n] = 0.0;
        }
        sum += cell_sum;
      },
      Kokkos::Sum<array_sum::GlobalSum>(local_sum));

  Real totals[5];
  for (int n = 0; n < 5; ++n) {
    totals[n] = local_sum.the_array[n];
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, totals, 5, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
#endif

  ConstraintSummary summary;
  summary.volume = totals[4];
  if (summary.volume > 0.0) {
    summary.c_rms = std::sqrt(totals[0] / summary.volume);
    summary.h_rms = std::sqrt(totals[1] / summary.volume);
    summary.m_rms = std::sqrt(totals[2] / summary.volume);
    summary.z_rms = std::sqrt(totals[3] / summary.volume);
  }
  return summary;
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
      std::fprintf(file,
                   "# Nx1  Nx2  Nx3   Ncycle  C_rms  H_rms  M_rms  Z_rms  "
                   "Volume\n");
    }
  }
  if (file == nullptr) {
    Fail("IrisK constraint output file could not be opened: " + filename);
  }
  std::fprintf(file,
               "%04d  %04d  %04d  %05d  %.16e  %.16e  %.16e  %.16e  "
               "%.16e\n",
               pm->mesh_indcs.nx1, pm->mesh_indcs.nx2,
               pm->mesh_indcs.nx3, pm->ncycle, summary.c_rms,
               summary.h_rms, summary.m_rms, summary.z_rms, summary.volume);
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
  if (summary.c_rms > c_threshold || summary.h_rms > h_threshold ||
      summary.m_rms > m_threshold || summary.z_rms > z_threshold) {
    Fail("IrisK imported constraints exceeded threshold: C=" +
         std::to_string(summary.c_rms) +
         " H=" + std::to_string(summary.h_rms) +
         " M=" + std::to_string(summary.m_rms) +
         " Z=" + std::to_string(summary.z_rms));
  }
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

void WriteImportedAdmPlane(ParameterInput *pin, MeshBlockPack *pmbp,
                           const bool xy_plane) {
  const char *path_key = xy_plane ? "xy_plane_output" : "xz_plane_output";
  const char *coordinate_key = xy_plane ? "xy_plane_z" : "xz_plane_y";
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
  output << "# x y z psi alpha beta_norm H M_norm C_norm Z_norm "
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
          output
              << x << ' ' << y << ' ' << z << ' ' << std::pow(psi4, 0.25)
              << ' ' << alpha << ' '
              << std::sqrt(std::max(Real{0.0}, beta_squared)) << ' '
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

void ProblemGenerator::Z4cFinalizeImportedAdm(ParameterInput *pin) {
  pgen_final_func = IrisXctsConstraintReport;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  switch (pmy_mesh_->mb_indcs.ng) {
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
      Fail("z4c_irisk_xcts supports nghost = 2, 3, or 4");
  }
  pmbp->pz4c->Z4cToADM(pmbp);
  RecomputeAdmConstraints(pmbp);
  const ConstraintSummary summary = ComputeConstraintSummary(pmy_mesh_);
  EnforceConstraintThresholds(pin, summary);
  try {
    WriteImportedAdmPlane(pin, pmbp, true);
    WriteImportedAdmPlane(pin, pmbp, false);
  } catch (const std::exception &error) {
    Fail(std::string("failed to write imported-ADM plane: ") + error.what());
  }
  if (global_variable::my_rank == 0) {
    std::cout << "Converted imported ADM data to Z4c"
              << " C_rms=" << summary.c_rms << " H_rms=" << summary.h_rms
              << " M_rms=" << summary.m_rms << " Z_rms=" << summary.z_rms
              << std::endl;
  }
}

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
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
  IrisAthenakSpectralInterpolator *interpolator = nullptr;
  std::array<char, 1024> error{};
  if (IrisAthenakSpectralOpen(filename.c_str(), &interpolator, error.data(),
                              error.size()) != 0) {
    Fail(std::string("failed to open IrisK spectral data: ") + error.data());
  }
  FillAdmFromIrisSpectral(pmbp, interpolator);
  IrisAthenakSpectralClose(interpolator);

  // Match the established puncture import sequence, while preserving the
  // elliptically solved XCTS lapse and shift rather than imposing a puncture
  // pre-collapsed lapse.
  Z4cFinalizeImportedAdm(pin);
  std::cout << "Initialized Z4c from IrisK spectral XCTS data: " << filename
            << std::endl;
}
