//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file z4c_grid_layout_test.cpp
//! \brief Exact tests for Z4c cell/vertex layouts and physical coordinates.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>

#include "coordinates/cell_locations.hpp"
#include "z4c/stored_domain_bounds.hpp"
#include "z4c/z4c_grid.hpp"

namespace {

struct Indices {
  int ng;
  int nx1, nx2, nx3;
};

struct LegacyIndices {
  int ng;
  int nx1, nx2, nx3;
  int is, ie, js, je, ks, ke;
  int cnx1, cnx2, cnx3;
  int cis, cie, cjs, cje, cks, cke;
};

bool Near(const Real lhs, const Real rhs) {
  return std::abs(lhs - rhs) <=
         8.0 * std::numeric_limits<Real>::epsilon() *
             std::max<Real>(1.0, std::max(std::abs(lhs), std::abs(rhs)));
}

bool CheckLayout() {
  constexpr Indices full{4, 8, 10, 12};
  constexpr auto cell =
      z4c::MakeZ4cGridLayout(z4c::Z4cGridCentering::cell, full, 3);
  constexpr auto cell_same_ng =
      z4c::MakeZ4cGridLayout(z4c::Z4cGridCentering::cell, full);
  constexpr auto vertex =
      z4c::MakeZ4cGridLayout(z4c::Z4cGridCentering::vertex, full, 3);
  static_assert(cell.is == 4 && cell.ie == 11 && cell.n1 == 16);
  static_assert(cell.js == 4 && cell.je == 13 && cell.n2 == 18);
  static_assert(cell.ks == 4 && cell.ke == 15 && cell.n3 == 20);
  static_assert(cell.cis == 3 && cell.cie == 6 && cell.cn1 == 10);
  static_assert(cell.cjs == 3 && cell.cje == 7 && cell.cn2 == 11);
  static_assert(cell.cks == 3 && cell.cke == 8 && cell.cn3 == 12);
  static_assert(vertex.is == 4 && vertex.ie == 12 && vertex.n1 == 17);
  static_assert(vertex.js == 4 && vertex.je == 14 && vertex.n2 == 19);
  static_assert(vertex.ks == 4 && vertex.ke == 16 && vertex.n3 == 21);
  static_assert(vertex.cis == 3 && vertex.cie == 7 && vertex.cn1 == 11);
  static_assert(vertex.cjs == 3 && vertex.cje == 8 && vertex.cn2 == 12);
  static_assert(vertex.cks == 3 && vertex.cke == 9 && vertex.cn3 == 13);
  constexpr LegacyIndices legacy_indices{
      4, 8, 10, 12, 4, 11, 4, 13, 4, 15,
      4, 5, 6, 4, 7, 4, 8, 4, 9};
  const auto legacy = z4c::MakeStoredDomainBounds(legacy_indices);
  const auto legacy_coarse = z4c::MakeCoarseStoredDomainBounds(legacy_indices);
  return vertex.centering_schema == z4c::Z4cGridLayout::kCenteringSchema &&
         vertex.ng == 4 && vertex.coarse_ng == 3 &&
         cell_same_ng.is == legacy_indices.is &&
         cell_same_ng.ie == legacy_indices.ie &&
         cell_same_ng.js == legacy_indices.js &&
         cell_same_ng.je == legacy_indices.je &&
         cell_same_ng.ks == legacy_indices.ks &&
         cell_same_ng.ke == legacy_indices.ke &&
         cell_same_ng.n1 == legacy.n1 && cell_same_ng.n2 == legacy.n2 &&
         cell_same_ng.n3 == legacy.n3 &&
         cell_same_ng.cis == legacy_indices.cis &&
         cell_same_ng.cie == legacy_indices.cie &&
         cell_same_ng.cjs == legacy_indices.cjs &&
         cell_same_ng.cje == legacy_indices.cje &&
         cell_same_ng.cks == legacy_indices.cks &&
         cell_same_ng.cke == legacy_indices.cke &&
         cell_same_ng.cn1 == legacy_coarse.n1 &&
         cell_same_ng.cn2 == legacy_coarse.n2 &&
         cell_same_ng.cn3 == legacy_coarse.n3;
}

bool CheckCoordinates() {
  const Real xmin = -1.0;
  const Real xmax = 3.0;
  const int n = 8;
  if (!Near(VertexX(0, n, xmin, xmax), xmin) ||
      !Near(VertexX(n, n, xmin, xmax), xmax) ||
      !Near(VertexX(-2, n, xmin, xmax), -2.0) ||
      !Near(VertexX(n + 2, n, xmin, xmax), 4.0)) {
    return false;
  }
  if (!Near(CellCenterX(0, n, xmin, xmax), -0.75) ||
      !Near(CellCenterX(-1, n, xmin, xmax), -1.25) ||
      !Near(CellCenterX(n, n, xmin, xmax), 3.25)) {
    return false;
  }

  // Same-level blocks share an endpoint; dyadic coarse/fine vertices coincide.
  if (VertexX(8, 8, 0.0, 1.0) != VertexX(0, 8, 1.0, 2.0)) return false;
  for (int coarse_i = 0; coarse_i <= 8; ++coarse_i) {
    if (!Near(VertexX(coarse_i, 8, 0.0, 1.0),
              VertexX(2 * coarse_i, 16, 0.0, 1.0))) {
      return false;
    }
  }
  // The half-plane axis and an equatorial vertex are represented exactly.
  return VertexX(0, 32, 0.0, 2.0) == 0.0 &&
         VertexX(16, 32, -2.0, 2.0) == 0.0;
}

bool CheckCollapsed() {
  for (const auto centering : {z4c::Z4cGridCentering::cell,
                               z4c::Z4cGridCentering::vertex}) {
    for (const Indices indcs : {Indices{2, 8, 8, 1}, Indices{3, 8, 1, 1},
                                Indices{4, 8, 10, 12}}) {
      const auto layout = z4c::MakeZ4cGridLayout(centering, indcs);
      if (indcs.nx2 == 1 &&
          (layout.js != 0 || layout.je != 0 || layout.n2 != 1 ||
           layout.cjs != 0 || layout.cje != 0 || layout.cn2 != 1)) {
        return false;
      }
      if (indcs.nx3 == 1 &&
          (layout.ks != 0 || layout.ke != 0 || layout.n3 != 1 ||
           layout.cks != 0 || layout.cke != 0 || layout.cn3 != 1)) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

int main(int argc, char **argv) {
  static_assert(std::is_trivially_copyable_v<z4c::Z4cGridLayout>);
  if (argc != 2) return EXIT_FAILURE;
  const std::string test = argv[1];
  const bool passed = test == "layout"       ? CheckLayout()
                      : test == "coordinates" ? CheckCoordinates()
                      : test == "collapsed"   ? CheckCollapsed()
                                               : false;
  if (!passed) return EXIT_FAILURE;
  std::cout << "Z4c grid " << test << " test passed\n";
  return EXIT_SUCCESS;
}
