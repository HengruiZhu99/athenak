//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file weyl_tetrad.hpp
//! \brief Coordinate-map-aware initial vectors for the Z4c Weyl tetrad.

#ifndef Z4C_WEYL_TETRAD_HPP_
#define Z4C_WEYL_TETRAD_HPP_

#include <type_traits>

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "z4c/cartoon_derivatives.hpp"

namespace z4c {

template <typename Centering, typename Symmetry>
KOKKOS_INLINE_FUNCTION Real WeylX3Coordinate(const int k, const int ks,
                                              const int nx3,
                                              const Real x3min,
                                              const Real x3max) {
  if constexpr (std::is_same_v<Symmetry, Cartesian3D>) {
    if constexpr (std::is_same_v<Centering, VertexCenteredZ4c>) {
      return VertexX(k - ks, nx3, x3min, x3max);
    } else {
      static_assert(std::is_same_v<Centering, CellCenteredZ4c>,
                    "Unknown Z4c Weyl centering policy");
      return CellCenterX(k - ks, nx3, x3min, x3max);
    }
  } else {
    static_assert(std::is_same_v<Symmetry, CartoonSO2>,
                  "Unknown Z4c Weyl coordinate symmetry policy");
    return 0.0;
  }
}

template <typename Symmetry, typename Vector>
KOKKOS_INLINE_FUNCTION void InitializeWeylTetradSeed(
    const Real x1, const Real x2, const Real x3, Vector &radial,
    Vector &polar, Vector &azimuthal) {
  Real xx = x1;
  if (SQR(x1) + SQR(x2) < 1.0e-10) xx += 1.0e-8;
  if constexpr (std::is_same_v<Symmetry, CartoonSO2>) {
    // Cartoon component order is (X,Z,Y)=(rho,z,suppressed).  At Y=0 the
    // Cartesian radial, theta, and phi seeds must therefore be permuted from
    // the legacy (X,Y,Z) ordering.
    radial(0) = xx;
    radial(1) = x2;
    radial(2) = 0.0;
    polar(0) = xx * x2;
    polar(1) = -SQR(xx);
    polar(2) = 0.0;
    azimuthal(0) = 0.0;
    azimuthal(1) = 0.0;
    azimuthal(2) = xx;
  } else {
    static_assert(std::is_same_v<Symmetry, Cartesian3D>,
                  "Unknown Z4c Weyl tetrad symmetry policy");
    radial(0) = xx;
    radial(1) = x2;
    radial(2) = x3;
    polar(0) = xx * x3;
    polar(1) = x2 * x3;
    polar(2) = -SQR(xx) - SQR(x2);
    azimuthal(0) = -x2;
    azimuthal(1) = xx;
    azimuthal(2) = 0.0;
  }
}

}  // namespace z4c

#endif  // Z4C_WEYL_TETRAD_HPP_
