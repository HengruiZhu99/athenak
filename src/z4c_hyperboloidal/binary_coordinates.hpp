// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_BINARY_COORDINATES_HPP_
#define Z4C_HYPERBOLOIDAL_BINARY_COORDINATES_HPP_
#include <algorithm>
#include <cmath>
#include <limits>
namespace z4c_hyperboloidal {
// Host import only. Atlas transcendental evaluations can round differently on
// CPU and GPU. This tolerance is far below any supported mesh spacing and does
// not permit reordering, interpolation, or a different coordinate chart.
inline bool BinaryCoordinateMatches(double supplied,double expected) {
  return std::isfinite(supplied) && std::isfinite(expected) &&
      std::abs(supplied-expected)<=32*std::numeric_limits<double>::epsilon()*
          std::max(1.,std::max(std::abs(supplied),std::abs(expected)));
}
}
#endif
