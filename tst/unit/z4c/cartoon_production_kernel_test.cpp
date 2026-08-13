//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE for details.
//========================================================================================
//! \file cartoon_production_kernel_test.cpp
//! \brief Production-linked collapsed-storage exercise for migrated Cartoon Z4c kernels.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>

#include <Kokkos_Core.hpp>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "z4c/cartoon_derivatives.hpp"
#include "z4c/stored_domain_bounds.hpp"
#include "z4c/z4c.hpp"
#include "z4c/z4c_symmetry.hpp"

#include "cartoon_production_kernel_test.hpp"

namespace z4c {
// This is a production template with explicit instantiations in z4c_adm.cpp.  The
// declaration is kept test-local so it does not enlarge the public application API.
template <typename Symmetry, int FD_STENCIL>
void ADMConstraintsImpl(MeshBlockPack *pmbp);
}  // namespace z4c

namespace {

constexpr Real kSentinel = 9.87654321e37;

std::string InputForStencil(const int stencil) {
  const int spatial_order = 2 * (stencil - 1);
  std::ostringstream input;
  input << R"(<mesh>
nx1 = 16
x1min = -2.0
x1max = 2.0
ix1_bc = user
ox1_bc = user
nx2 = 14
x2min = -1.75
x2max = 1.75
ix2_bc = user
ox2_bc = user
nx3 = 1
x3min = -0.5
x3max = 0.5
ix3_bc = user
ox3_bc = user
nghost = )" << stencil << R"(

<meshblock>
nx1 = 16
nx2 = 14
nx3 = 1

<mesh_refinement>
refinement = none

<time>
evolution = dynamic
cfl_number = 0.1

<coord>
minkowski = true

<z4c>
spatial_order = )" << spatial_order << R"(
diss = 0.02
user_Sbc = true
telegraph_lapse = true
telegraph_tau = 0.2
telegraph_kappa = 0.15
shift_advect = 1.0
lapse_advect = 1.0
nrad_wave_extraction = 0

<cce>
num_radii = 0

<fastflow>
num_horizons = 0
)";
  return input.str();
}

struct Tensor6 {
  Real xx, xz, xy, zz, zy, yy;
};

Tensor6 MetricField(const Real rho, const Real z, const bool minkowski) {
  if (minkowski) return {1.0, 0.0, 0.0, 1.0, 0.0, 1.0};
  const Real s = rho * rho;
  const Real p = 1.0 + 0.004 * s + 0.002 * z * z;
  const Real q = 1.0 + 0.003 * s + 0.004 * z * z;
  const Real v = 0.0010 + 0.0003 * z;
  const Real r = 0.0015 - 0.0002 * s;
  const Real u = -0.0012 + 0.0001 * z;
  const Real w = 0.0008 + 0.0001 * z;
  return {p + v * s, r * rho, w * s, q, u * rho, p - v * s};
}

Tensor6 ExtrinsicField(const Real rho, const Real z, const bool minkowski) {
  if (minkowski) return {};
  const Real s = rho * rho;
  const Real p = 0.0012 + 0.0003 * s - 0.0002 * z;
  const Real q = -0.0010 + 0.0002 * s + 0.0001 * z * z;
  const Real v = 0.00035 - 0.00004 * z;
  const Real r = -0.00045 + 0.00003 * s;
  const Real u = 0.00030 + 0.00002 * z;
  const Real w = -0.00025 + 0.00002 * z;
  return {p + v * s, r * rho, w * s, q, u * rho, p - v * s};
}

template <typename View>
void StoreTensor(const Tensor6 &value, const int offset, View &array,
                 const int j, const int i) {
  array(0, offset + 0, 0, j, i) = value.xx;
  array(0, offset + 1, 0, j, i) = value.xz;
  array(0, offset + 2, 0, j, i) = value.xy;
  array(0, offset + 3, 0, j, i) = value.zz;
  array(0, offset + 4, 0, j, i) = value.zy;
  array(0, offset + 5, 0, j, i) = value.yy;
}

void FillFields(MeshBlockPack *pack, const bool minkowski) {
  auto *z4c = pack->pz4c;
  auto *adm = pack->padm;
  auto z4c_host = Kokkos::create_mirror_view(z4c->u0);
  auto adm_host = Kokkos::create_mirror_view(adm->u_adm);
  const auto bounds = z4c::MakeStoredDomainBounds(pack->pmesh->mb_indcs);
  const auto &indcs = pack->pmesh->mb_indcs;
  const auto &size = pack->pmb->mb_size.h_view(0);
  for (int j = bounds.js; j <= bounds.je; ++j) {
    const Real z = CellCenterX(j - indcs.js, indcs.nx2, size.x2min, size.x2max);
    for (int i = bounds.is; i <= bounds.ie; ++i) {
      const Real rho = CellCenterX(i - indcs.is, indcs.nx1, size.x1min, size.x1max);
      const Real s = rho * rho;
      for (int n = 0; n < z4c::Z4c::nz4c; ++n) z4c_host(0, n, 0, j, i) = 0.0;
      for (int n = 0; n < adm::ADM::nadm - 4; ++n) adm_host(0, n, 0, j, i) = 0.0;

      const Tensor6 metric = MetricField(rho, z, minkowski);
      const Tensor6 extrinsic = ExtrinsicField(rho, z, minkowski);
      StoreTensor(metric, z4c::Z4c::I_Z4C_GXX, z4c_host, j, i);
      StoreTensor(extrinsic, z4c::Z4c::I_Z4C_AXX, z4c_host, j, i);
      StoreTensor(metric, adm::ADM::I_ADM_GXX, adm_host, j, i);
      StoreTensor(extrinsic, adm::ADM::I_ADM_KXX, adm_host, j, i);

      const Real common = s + z * z;
      z4c_host(0, z4c::Z4c::I_Z4C_CHI, 0, j, i) =
          minkowski ? 1.0 : 1.0 + 0.002 * common;
      z4c_host(0, z4c::Z4c::I_Z4C_KHAT, 0, j, i) =
          minkowski ? 0.0 : extrinsic.xx + extrinsic.zz + extrinsic.yy;
      z4c_host(0, z4c::Z4c::I_Z4C_THETA, 0, j, i) =
          minkowski ? 0.0 : 0.0001 * (s - 0.5 * z * z);
      z4c_host(0, z4c::Z4c::I_Z4C_ALPHA, 0, j, i) =
          minkowski ? 1.0 : 1.0 + 0.003 * common + 0.0002 * z;

      if (!minkowski) {
        const Real beta_a = 0.010 + 0.002 * z;
        const Real beta_b = -0.007 + 0.001 * s;
        z4c_host(0, z4c::Z4c::I_Z4C_BETAX, 0, j, i) = beta_a * rho;
        z4c_host(0, z4c::Z4c::I_Z4C_BETAY, 0, j, i) =
            0.005 + 0.002 * s + 0.001 * z;
        z4c_host(0, z4c::Z4c::I_Z4C_BETAZ, 0, j, i) = beta_b * rho;
        z4c_host(0, z4c::Z4c::I_Z4C_GAMX, 0, j, i) =
            (-0.006 + 0.001 * z) * rho;
        z4c_host(0, z4c::Z4c::I_Z4C_GAMY, 0, j, i) =
            0.004 - 0.001 * s + 0.0005 * z;
        z4c_host(0, z4c::Z4c::I_Z4C_GAMZ, 0, j, i) =
            (0.003 + 0.0007 * s) * rho;
        z4c_host(0, z4c::Z4c::I_Z4C_BX, 0, j, i) =
            (0.008 - 0.001 * z) * rho;
        z4c_host(0, z4c::Z4c::I_Z4C_BY, 0, j, i) =
            -0.003 + 0.0008 * s + 0.0004 * z;
        z4c_host(0, z4c::Z4c::I_Z4C_BZ, 0, j, i) =
            (-0.005 + 0.0006 * s) * rho;
      }
      adm_host(0, adm::ADM::I_ADM_PSI4, 0, j, i) =
          minkowski ? 1.0 : 1.0 + 0.0015 * common;
    }
  }
  Kokkos::deep_copy(z4c->u0, z4c_host);
  Kokkos::deep_copy(adm->u_adm, adm_host);
}

bool Finite(const Real value) { return std::isfinite(value); }

bool NearlyEqual(const Real left, const Real right, const Real tolerance) {
  const Real scale = std::max({Real(1.0), std::abs(left), std::abs(right)});
  return std::abs(left - right) <= tolerance * scale;
}

template <int STENCIL>
bool CheckRhs(MeshBlockPack *pack, const bool minkowski) {
  auto *z4c = pack->pz4c;
  Kokkos::deep_copy(z4c->u_rhs, 0.0);
  if (z4c->CalcRHSImpl<z4c::CartoonSO2, STENCIL>(nullptr, 0) !=
      TaskStatus::complete) return false;
  Kokkos::fence();
  auto rhs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), z4c->u_rhs);
  const auto &indcs = pack->pmesh->mb_indcs;
  Real maximum = 0.0;
  for (int j = indcs.js; j <= indcs.je; ++j) {
    for (int i = indcs.is; i <= indcs.ie; ++i) {
      for (int n = 0; n < z4c::Z4c::nz4c; ++n) {
        const Real value = rhs(0, n, 0, j, i);
        if (!Finite(value)) return false;
        maximum = std::max(maximum, std::abs(value));
      }
    }
  }
  if (minkowski) return maximum <= 2.0e-13;
  if (!(maximum > 1.0e-8)) return false;

  const int left = indcs.is + 2;
  const int right = indcs.ie - 2;
  const int j = indcs.js + indcs.nx2 / 2 + 1;
  const Real tolerance = 3.0e-10;
  if (!NearlyEqual(rhs(0, z4c::Z4c::I_Z4C_CHI, 0, j, left),
                   rhs(0, z4c::Z4c::I_Z4C_CHI, 0, j, right), tolerance)) return false;
  for (const int component : {0, 2}) {
    if (!NearlyEqual(rhs(0, z4c::Z4c::I_Z4C_BETAX + component, 0, j, left),
                     -rhs(0, z4c::Z4c::I_Z4C_BETAX + component, 0, j, right),
                     tolerance)) return false;
  }
  return NearlyEqual(rhs(0, z4c::Z4c::I_Z4C_BETAY, 0, j, left),
                     rhs(0, z4c::Z4c::I_Z4C_BETAY, 0, j, right), tolerance);
}

template <int STENCIL>
bool CheckAdmToZ4c(MeshBlockPack *pack, ParameterInput *input) {
  pack->pz4c->ADMToZ4c<STENCIL>(pack, input);
  Kokkos::fence();
  auto converted = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), pack->pz4c->u0);
  const auto &indcs = pack->pmesh->mb_indcs;
  Real maximum_gamma = 0.0;
  for (int j = indcs.js; j <= indcs.je; ++j) {
    for (int i = indcs.is; i <= indcs.ie; ++i) {
      for (int n = 0; n < z4c::Z4c::nz4c; ++n) {
        if (!Finite(converted(0, n, 0, j, i))) return false;
      }
      for (int component = 0; component < 3; ++component) {
        maximum_gamma = std::max(
            maximum_gamma,
            std::abs(converted(0, z4c::Z4c::I_Z4C_GAMX + component, 0, j, i)));
      }
    }
  }
  if (!(maximum_gamma > 1.0e-8)) return false;

  const int left = indcs.is + 2;
  const int right = indcs.ie - 2;
  const int j = indcs.js + indcs.nx2 / 2 + 1;
  const Real tolerance = 3.0e-9;
  for (const int component : {0, 2}) {
    if (!NearlyEqual(
            converted(0, z4c::Z4c::I_Z4C_GAMX + component, 0, j, left),
            -converted(0, z4c::Z4c::I_Z4C_GAMX + component, 0, j, right),
            tolerance)) return false;
  }
  return NearlyEqual(converted(0, z4c::Z4c::I_Z4C_GAMY, 0, j, left),
                     converted(0, z4c::Z4c::I_Z4C_GAMY, 0, j, right), tolerance);
}

template <int STENCIL>
bool CheckConstraints(MeshBlockPack *pack, const bool minkowski) {
  z4c::ADMConstraintsImpl<z4c::CartoonSO2, STENCIL>(pack);
  Kokkos::fence();
  auto constraints = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), pack->pz4c->u_con);
  const auto &indcs = pack->pmesh->mb_indcs;
  Real maximum_h = 0.0;
  Real maximum_m = 0.0;
  for (int j = indcs.js; j <= indcs.je; ++j) {
    for (int i = indcs.is; i <= indcs.ie; ++i) {
      const Real h = constraints(0, z4c::Z4c::I_CON_H, 0, j, i);
      const Real m = constraints(0, z4c::Z4c::I_CON_M, 0, j, i);
      if (!Finite(h) || !Finite(m)) return false;
      maximum_h = std::max(maximum_h, std::abs(h));
      maximum_m = std::max(maximum_m, std::abs(m));
    }
  }
  if (minkowski) return maximum_h <= 2.0e-13 && maximum_m <= 2.0e-25;
  if (!(maximum_h > 1.0e-8) || !(maximum_m > 1.0e-14)) return false;

  const int left = indcs.is + 2;
  const int right = indcs.ie - 2;
  const int j = indcs.js + indcs.nx2 / 2 + 1;
  const Real tolerance = 2.0e-9;
  if (!NearlyEqual(constraints(0, z4c::Z4c::I_CON_H, 0, j, left),
                   constraints(0, z4c::Z4c::I_CON_H, 0, j, right), tolerance)) return false;
  for (const int component : {0, 2}) {
    if (!NearlyEqual(constraints(0, z4c::Z4c::I_CON_MX + component, 0, j, left),
                     -constraints(0, z4c::Z4c::I_CON_MX + component, 0, j, right),
                     tolerance)) return false;
  }
  return NearlyEqual(constraints(0, z4c::Z4c::I_CON_MY, 0, j, left),
                     constraints(0, z4c::Z4c::I_CON_MY, 0, j, right), tolerance);
}

template <int STENCIL>
bool CheckWeyl(MeshBlockPack *pack, const bool minkowski) {
  pack->pz4c->Z4cWeylImpl<z4c::CartoonSO2, STENCIL>(pack);
  Kokkos::fence();
  auto weyl = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), pack->pz4c->u_weyl);
  const auto &indcs = pack->pmesh->mb_indcs;
  Real maximum_real = 0.0;
  Real maximum_imaginary = 0.0;
  for (int j = indcs.js; j <= indcs.je; ++j) {
    for (int i = indcs.is; i <= indcs.ie; ++i) {
      const Real real = weyl(0, 0, 0, j, i);
      const Real imaginary = weyl(0, 1, 0, j, i);
      if (!Finite(real) || !Finite(imaginary)) return false;
      maximum_real = std::max(maximum_real, std::abs(real));
      maximum_imaginary = std::max(maximum_imaginary, std::abs(imaginary));
    }
  }
  if (minkowski) return maximum_real <= 2.0e-13 && maximum_imaginary <= 2.0e-13;
  if (!(maximum_real > 1.0e-9) || !(maximum_imaginary > 1.0e-10)) return false;

  const int left = indcs.is + 2;
  const int right = indcs.ie - 2;
  const int j = indcs.js + indcs.nx2 / 2 + 1;
  return NearlyEqual(weyl(0, 0, 0, j, left), weyl(0, 0, 0, j, right), 3.0e-9) &&
         NearlyEqual(weyl(0, 1, 0, j, left), weyl(0, 1, 0, j, right), 3.0e-9);
}

bool CheckSommerfeldFaces(MeshBlockPack *pack) {
  auto *z4c = pack->pz4c;
  Kokkos::deep_copy(z4c->u_rhs, kSentinel);
  if (z4c->Z4cBoundaryRHSImpl<z4c::CartoonSO2>(nullptr, 0) !=
      TaskStatus::complete) return false;
  Kokkos::fence();
  auto rhs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), z4c->u_rhs);
  const auto &indcs = pack->pmesh->mb_indcs;
  const auto &size = pack->pmb->mb_size.h_view(0);
  const int middle_i = indcs.is + indcs.nx1 / 2;
  const int middle_j = indcs.js + indcs.nx2 / 2;
  if (rhs.extent_int(2) != 1) return false;
  constexpr int updated_components[] = {
      z4c::Z4c::I_Z4C_KHAT, z4c::Z4c::I_Z4C_AXX,
      z4c::Z4c::I_Z4C_AXY, z4c::Z4c::I_Z4C_AXZ,
      z4c::Z4c::I_Z4C_AYY, z4c::Z4c::I_Z4C_AYZ,
      z4c::Z4c::I_Z4C_AZZ, z4c::Z4c::I_Z4C_GAMX,
      z4c::Z4c::I_Z4C_GAMY, z4c::Z4c::I_Z4C_GAMZ,
      z4c::Z4c::I_Z4C_THETA};
  const auto updated = [&](const int j, const int i) {
    for (const int component : updated_components) {
      const Real value = rhs(0, component, 0, j, i);
      if (!Finite(value) || value == kSentinel) return false;
    }
    return true;
  };
  // Cover every cell of both signed-rho and both axial faces.  Corners belong to
  // both ranges and are deliberately checked twice.  The production x2 kernels
  // are enqueued after x1, so their final corner value has deterministic ownership.
  for (int j = indcs.js; j <= indcs.je; ++j) {
    if (!updated(j, indcs.is) || !updated(j, indcs.ie)) return false;
  }
  for (int i = indcs.is; i <= indcs.ie; ++i) {
    if (!updated(indcs.js, i) || !updated(indcs.je, i)) return false;
  }

  // Retain independent analytic samples on the midpoint of every active face.
  for (const auto &point : {std::pair<int, int>{middle_j, indcs.is},
                            {middle_j, indcs.ie}, {indcs.js, middle_i},
                            {indcs.je, middle_i}}) {
    // Independent fixed-order-2 oracle for Theta=c*(rho^2-z^2/2):
    // Theta_t = -Theta/r - (rho/r)*d_rho Theta - (z/r)*d_z Theta.
    const Real rho = CellCenterX(point.second - indcs.is, indcs.nx1,
                                 size.x1min, size.x1max);
    const Real z = CellCenterX(point.first - indcs.js, indcs.nx2,
                               size.x2min, size.x2max);
    const Real radius = std::sqrt(rho * rho + z * z);
    const Real theta = 0.0001 * (rho * rho - 0.5 * z * z);
    const Real expected = -theta / radius - (rho / radius) * (0.0002 * rho) -
                          (z / radius) * (-0.0001 * z);
    if (!NearlyEqual(rhs(0, z4c::Z4c::I_Z4C_THETA, 0, point.first,
                         point.second), expected, 2.0e-12)) return false;
  }
  const auto value = [&](const int component, const int j, const int i) {
    return rhs(0, component, 0, j, i);
  };
  if (!NearlyEqual(value(z4c::Z4c::I_Z4C_THETA, middle_j, indcs.is),
                   value(z4c::Z4c::I_Z4C_THETA, middle_j, indcs.ie), 2.0e-12) ||
      !NearlyEqual(value(z4c::Z4c::I_Z4C_THETA, indcs.js, middle_i),
                   value(z4c::Z4c::I_Z4C_THETA, indcs.je, middle_i), 2.0e-12))
    return false;
  // The two transverse vector components reverse across the signed-rho plane,
  // while the axial component does not. This catches using |rho| or the wrong map.
  for (const int component : {z4c::Z4c::I_Z4C_GAMX,
                              z4c::Z4c::I_Z4C_GAMZ}) {
    if (!NearlyEqual(value(component, middle_j, indcs.is),
                     -value(component, middle_j, indcs.ie), 2.0e-11)) return false;
  }
  if (!NearlyEqual(value(z4c::Z4c::I_Z4C_GAMY, middle_j, indcs.is),
                   value(z4c::Z4c::I_Z4C_GAMY, middle_j, indcs.ie), 2.0e-11))
    return false;
  // A suppressed-x3 launch would touch every cell in the sole stored plane.  The
  // strict interior remains untouched because Cartoon returns before that launch.
  for (int n = 0; n < z4c::Z4c::nz4c; ++n) {
    if (rhs(0, n, 0, middle_j, middle_i) != kSentinel) {
      std::cerr << "Sbc touched interior component=" << n << " value="
                << rhs(0, n, 0, middle_j, middle_i) << "\n";
      return false;
    }
  }
  return true;
}

template <int STENCIL>
bool RunStencil() {
  ParameterInput input;
  std::istringstream stream(InputForStencil(STENCIL));
  input.LoadFromStream(stream);
  Mesh mesh(&input);
  mesh.BuildTreeFromScratch(&input);
  MeshBlockPack *pack = mesh.pmb_pack;
  pack->AddCoordinates(&input);
  pack->z4c_symmetry = {z4c::Z4cSymmetryMode::cartoon_so2,
                         z4c::Z4cCoordinateMap::signed_rho_z_suppressed_y_v1,
                         1, STENCIL};
  pack->pz4c = new z4c::Z4c(pack, &input);
  pack->padm = new adm::ADM(pack, &input);
  pack->ptmunu = nullptr;
  if (pack->pz4c->u0.extent_int(2) != 1 || pack->padm->u_adm.extent_int(2) != 1 ||
      pack->pmesh->mb_indcs.nx3 != 1) return false;

  FillFields(pack, true);
  if (!CheckRhs<STENCIL>(pack, true)) {
    std::cerr << "STENCIL=" << STENCIL << " Minkowski RHS failed\n";
    return false;
  }
  if (!CheckConstraints<STENCIL>(pack, true)) {
    std::cerr << "STENCIL=" << STENCIL << " Minkowski constraints failed\n";
    return false;
  }
  if (!CheckWeyl<STENCIL>(pack, true)) {
    std::cerr << "STENCIL=" << STENCIL << " Minkowski Weyl failed\n";
    return false;
  }

  FillFields(pack, false);
  if (!CheckAdmToZ4c<STENCIL>(pack, &input)) {
    std::cerr << "STENCIL=" << STENCIL << " ADM-to-Z4c conversion failed\n";
    return false;
  }
  FillFields(pack, false);
  if (!CheckRhs<STENCIL>(pack, false)) {
    std::cerr << "STENCIL=" << STENCIL << " nontrivial RHS failed\n";
    return false;
  }
  if (!CheckConstraints<STENCIL>(pack, false)) {
    std::cerr << "STENCIL=" << STENCIL << " nontrivial constraints failed\n";
    return false;
  }
  if (!CheckWeyl<STENCIL>(pack, false)) {
    std::cerr << "STENCIL=" << STENCIL << " nontrivial Weyl failed\n";
    return false;
  }
  if (!CheckSommerfeldFaces(pack)) {
    std::cerr << "STENCIL=" << STENCIL << " Sommerfeld faces failed\n";
    return false;
  }
  return true;
}

bool MeetsCudaRequirement(const bool require_cuda) {
  if (!require_cuda) return true;
#if defined(KOKKOS_ENABLE_CUDA)
  return std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::Cuda> &&
         std::string(Kokkos::DefaultExecutionSpace::name()) == "Cuda";
#else
  return false;
#endif
}

}  // namespace

int RunCartoonProductionKernelTest(const bool require_cuda) {
  if (!MeetsCudaRequirement(require_cuda)) {
    std::cerr << "Cartoon production kernel test requires compile-time and runtime Cuda; "
              << "observed " << Kokkos::DefaultExecutionSpace::name() << "\n";
    return EXIT_FAILURE;
  }
  const bool passed = RunStencil<2>() && RunStencil<3>() && RunStencil<4>();
  if (!passed) {
    std::cerr << "Cartoon production kernel test failed on "
              << Kokkos::DefaultExecutionSpace::name() << "\n";
    return EXIT_FAILURE;
  }
  std::cout << "Cartoon CalcRHS/constraints/Sommerfeld/Weyl production kernels passed "
            << "for collapsed NGHOST 2/3/4 on "
            << Kokkos::DefaultExecutionSpace::name() << "\n";
  return EXIT_SUCCESS;
}
