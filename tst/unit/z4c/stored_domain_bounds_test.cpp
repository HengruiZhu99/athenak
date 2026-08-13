//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file stored_domain_bounds_test.cpp
//! \brief Bounds-checked coverage for active and collapsed stored extents.

#include <cstdlib>
#include <iostream>
#include <type_traits>
#include <vector>

#include <Kokkos_Core.hpp>

#include "pgen/z4c_irisk_coordinate_map.hpp"
#include "z4c/stored_domain_bounds.hpp"

namespace {

struct Indices {
  int ng;
  int nx1, nx2, nx3;
  int is, ie, js, je, ks, ke;
  int cnx1, cnx2, cnx3;
  int cis, cie, cjs, cje, cks, cke;
};

bool CheckCase(const int ng, const int nx2, const int nx3) {
  const int cnx2 = nx2 > 1 ? nx2 / 2 : 1;
  const int cnx3 = nx3 > 1 ? nx3 / 2 : 1;
  const Indices indcs{
      ng, 8, nx2, nx3, ng, ng + 7,
      nx2 > 1 ? ng : 0, nx2 > 1 ? ng + nx2 - 1 : 0,
      nx3 > 1 ? ng : 0, nx3 > 1 ? ng + nx3 - 1 : 0,
      4, cnx2, cnx3, ng, ng + 3,
      cnx2 > 1 ? ng : 0, cnx2 > 1 ? ng + cnx2 - 1 : 0,
      cnx3 > 1 ? ng : 0, cnx3 > 1 ? ng + cnx3 - 1 : 0};
  const auto bounds = z4c::MakeStoredDomainBounds(indcs);
  const auto coarse = z4c::MakeCoarseStoredDomainBounds(indcs);
  if (bounds.is != 0 || bounds.ie != 7 + 2 * ng || bounds.n1 != 8 + 2 * ng) {
    return false;
  }
  const int expected_n2 = nx2 > 1 ? nx2 + 2 * ng : 1;
  const int expected_n3 = nx3 > 1 ? nx3 + 2 * ng : 1;
  if (bounds.n2 != expected_n2 || bounds.n3 != expected_n3) return false;
  if (nx2 == 1 && (bounds.js != 0 || bounds.je != 0)) return false;
  if (nx3 == 1 && (bounds.ks != 0 || bounds.ke != 0)) return false;
  if (coarse.n1 != 4 + 2 * ng ||
      coarse.n2 != (cnx2 > 1 ? cnx2 + 2 * ng : 1) ||
      coarse.n3 != (cnx3 > 1 ? cnx3 + 2 * ng : 1)) return false;
  if (cnx2 == 1 && (coarse.js != 0 || coarse.je != 0)) return false;
  if (cnx3 == 1 && (coarse.ks != 0 || coarse.ke != 0)) return false;

  Kokkos::View<int***> storage("stored bounds", bounds.n3, bounds.n2, bounds.n1);
  Kokkos::parallel_for(
      "touch stored bounds",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
          {bounds.ks, bounds.js, bounds.is},
          {bounds.ke + 1, bounds.je + 1, bounds.ie + 1}),
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        storage(k, j, i) = 1;
      });
  Kokkos::fence();
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), storage);
  for (std::size_t k = 0; k < host.extent(0); ++k) {
    for (std::size_t j = 0; j < host.extent(1); ++j) {
      for (std::size_t i = 0; i < host.extent(2); ++i) {
        if (host(k, j, i) != 1) return false;
      }
    }
  }
  return true;
}

template <z4c_irisk::AdmMap Map>
bool CheckIrisImportCoverage(const int ng, const int nx2, const int nx3) {
  const Indices indcs{
      ng, 8, nx2, nx3, ng, ng + 7,
      nx2 > 1 ? ng : 0, nx2 > 1 ? ng + nx2 - 1 : 0,
      nx3 > 1 ? ng : 0, nx3 > 1 ? ng + nx3 - 1 : 0,
      4, nx2 > 1 ? nx2 / 2 : 1, nx3 > 1 ? nx3 / 2 : 1,
      ng, ng + 3, nx2 > 1 ? ng : 0,
      nx2 > 1 ? ng + nx2 / 2 - 1 : 0, nx3 > 1 ? ng : 0,
      nx3 > 1 ? ng + nx3 / 2 - 1 : 0};
  const auto bounds = z4c::MakeStoredDomainBounds(indcs);
  const auto dims = z4c_irisk::IrisTensorProductDimensions<Map>(
      static_cast<std::size_t>(bounds.n1),
      static_cast<std::size_t>(bounds.n2),
      static_cast<std::size_t>(bounds.n3));
  const std::size_t points = dims[0] * dims[1] * dims[2];
  std::vector<int> visits(points, 0);
  std::size_t writes = 0;
  for (int k = bounds.ks; k <= bounds.ke; ++k) {
    for (int j = bounds.js; j <= bounds.je; ++j) {
      for (int i = bounds.is; i <= bounds.ie; ++i) {
        const std::size_t point = z4c_irisk::IrisPointIndex<Map>(
            static_cast<std::size_t>(i - bounds.is),
            static_cast<std::size_t>(j - bounds.js),
            static_cast<std::size_t>(k - bounds.ks),
            static_cast<std::size_t>(bounds.n1),
            static_cast<std::size_t>(bounds.n2));
        if (point >= visits.size()) return false;
        ++visits[point];
        ++writes;
      }
    }
  }
  if (writes != static_cast<std::size_t>(bounds.n1 * bounds.n2 * bounds.n3)) {
    return false;
  }
  for (const int visit_count : visits) {
    if (visit_count != 1) return false;
  }
  if constexpr (Map == z4c_irisk::AdmMap::half_rho_z_suppressed_y_v2) {
    if (bounds.ks != bounds.ke || bounds.n3 != 1 || dims[1] != 1) return false;
    // The two physical-Z rows must remain distinct; the collapsed direction
    // must not duplicate or alias them.
    if (bounds.n2 > 1 &&
        z4c_irisk::IrisPointIndex<Map>(0, 0, 0, bounds.n1, bounds.n2) ==
            z4c_irisk::IrisPointIndex<Map>(0, 1, 0, bounds.n1, bounds.n2)) {
      return false;
    }
  }
  return true;
}

}  // namespace

int main(int argc, char *argv[]) {
  static_assert(std::is_trivially_copyable_v<z4c::StoredDomainBounds>);
  Kokkos::initialize(argc, argv);
  bool passed = true;
  for (const int ng : {2, 3, 4}) {
    passed = passed && CheckCase(ng, 8, 8);
    passed = passed && CheckCase(ng, 8, 1);
    passed = passed && CheckCase(ng, 1, 1);
    passed = passed && CheckIrisImportCoverage<
                           z4c_irisk::AdmMap::cartesian_xyz>(ng, 8, 8);
    passed = passed && CheckIrisImportCoverage<
                           z4c_irisk::AdmMap::half_rho_z_suppressed_y_v2>(
                           ng, 8, 1);
    passed = passed && CheckIrisImportCoverage<
                           z4c_irisk::AdmMap::half_rho_z_suppressed_y_v2>(
                           ng, 1, 1);
  }
  Kokkos::finalize();
  if (!passed) return EXIT_FAILURE;
  std::cout << "Stored-domain bounds tests passed\n";
  return EXIT_SUCCESS;
}
