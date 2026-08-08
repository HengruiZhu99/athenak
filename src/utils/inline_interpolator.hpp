#ifndef UTILS_INLINE_INTERPOLATOR_HPP_
#define UTILS_INLINE_INTERPOLATOR_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file inline_interpolator.hpp
//! \brief The following is a wrapper of the Lagrange interpolator
//!             to be used inside a Kokkos kernel.
#include <cmath>
#include <iostream>
#include <list>

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"

template <int NGHOST>
struct IndAndWghts {
  int ii0, ii1, ii2, ii3;
  Real wght1[2 * NGHOST];
  Real wght2[2 * NGHOST];
  Real wght3[2 * NGHOST];
  Real dwght1[2 * NGHOST];
  Real dwght2[2 * NGHOST];
  Real dwght3[2 * NGHOST];
  bool point_exist;
};

template <int NGHOST>
KOKKOS_INLINE_FUNCTION
IndAndWghts<NGHOST> IndicesAndWeights(
  const RegionIndcs &indcs,
  const DualArray1D<RegionSize> &size,
  Real rcoords[3],
  int nmb
) {
  IndAndWghts<NGHOST> result;

  // **INTERPOLATION INDICES**
  result.ii0 = result.ii1 = result.ii2 = result.ii3 = -1;
  result.point_exist = false;

  for (int m = 0; m < nmb; ++m) {
    // extract MeshBlock bounds
    auto mb = size.d_view(m);

    // save MeshBlock and zone indicies for nearest position to spherical patch
    // center if this angle position resides in this MeshBlock
    if (
      (rcoords[0] >= mb.x1min && rcoords[0] < mb.x1max)
      && (rcoords[1] >= mb.x2min && rcoords[1] < mb.x2max)
      && (rcoords[2] >= mb.x3min && rcoords[2] < mb.x3max)) {
      result.point_exist = true;
      result.ii0 = m;
      result.ii1 = static_cast<int>(Kokkos::floor((rcoords[0]
                    - (mb.x1min + mb.dx1 / 2.0)) / mb.dx1));
      result.ii2 = static_cast<int>(Kokkos::floor((rcoords[1]
                    - (mb.x2min + mb.dx2 / 2.0)) / mb.dx2));
      result.ii3 = static_cast<int>(Kokkos::floor((rcoords[2]
                    - (mb.x3min + mb.dx3 / 2.0)) / mb.dx3));

      break;
    }
  }

  // **INTERPOLATION WEIGHTS**
  if (result.ii0 == -1) {
    for (int i = 0; i < 2 * NGHOST; ++i) {
      result.wght1[i] = 0.;
      result.wght2[i] = 0.;
      result.wght3[i] = 0.;
      result.dwght1[i] = 0.;
      result.dwght2[i] = 0.;
      result.dwght3[i] = 0.;
    }
  } else {
    // extract MeshBlock bounds
    auto mb0 = size.d_view(result.ii0);

    // set interpolation weights
    for (int i = 0; i < 2 * NGHOST; ++i) {
      result.wght1[i] = 1.;
      result.wght2[i] = 1.;
      result.wght3[i] = 1.;
      result.dwght1[i] = 0.;
      result.dwght2[i] = 0.;
      result.dwght3[i] = 0.;

      Real x1vpi1 = CellCenterX(result.ii1 - NGHOST + i + 1,
                                indcs.nx1, mb0.x1min, mb0.x1max);
      Real x2vpi1 = CellCenterX(result.ii2 - NGHOST + i + 1,
                                indcs.nx2, mb0.x2min, mb0.x2max);
      Real x3vpi1 = CellCenterX(result.ii3 - NGHOST + i + 1,
                                indcs.nx3, mb0.x3min, mb0.x3max);

      for (int j = 0; j < 2 * NGHOST; ++j) {
        if (j != i) {
          Real x1vpj1 = CellCenterX(result.ii1 - NGHOST + j + 1, indcs.nx1,
                                    mb0.x1min, mb0.x1max);
          result.wght1[i] *= (rcoords[0] - x1vpj1) / (x1vpi1 - x1vpj1);
          Real x2vpj1 = CellCenterX(result.ii2 - NGHOST + j + 1, indcs.nx2,
                                    mb0.x2min, mb0.x2max);
          result.wght2[i] *= (rcoords[1] - x2vpj1) / (x2vpi1 - x2vpj1);
          Real x3vpj1 = CellCenterX(result.ii3 - NGHOST + j + 1, indcs.nx3,
                                    mb0.x3min, mb0.x3max);
          result.wght3[i] *= (rcoords[2] - x3vpj1) / (x3vpi1 - x3vpj1);
        }
      }

      // Analytic derivatives of the one-dimensional Lagrange basis.
      for (int k = 0; k < 2 * NGHOST; ++k) {
        if (k == i) continue;
        Real dterm1 = 1.;
        Real dterm2 = 1.;
        Real dterm3 = 1.;
        Real x1vpk1 = CellCenterX(result.ii1 - NGHOST + k + 1,
                                  indcs.nx1, mb0.x1min, mb0.x1max);
        Real x2vpk1 = CellCenterX(result.ii2 - NGHOST + k + 1,
                                  indcs.nx2, mb0.x2min, mb0.x2max);
        Real x3vpk1 = CellCenterX(result.ii3 - NGHOST + k + 1,
                                  indcs.nx3, mb0.x3min, mb0.x3max);
        for (int j = 0; j < 2 * NGHOST; ++j) {
          if (j == i || j == k) continue;
          Real x1vpj1 = CellCenterX(result.ii1 - NGHOST + j + 1,
                                    indcs.nx1, mb0.x1min, mb0.x1max);
          Real x2vpj1 = CellCenterX(result.ii2 - NGHOST + j + 1,
                                    indcs.nx2, mb0.x2min, mb0.x2max);
          Real x3vpj1 = CellCenterX(result.ii3 - NGHOST + j + 1,
                                    indcs.nx3, mb0.x3min, mb0.x3max);
          dterm1 *= (rcoords[0] - x1vpj1) / (x1vpi1 - x1vpj1);
          dterm2 *= (rcoords[1] - x2vpj1) / (x2vpi1 - x2vpj1);
          dterm3 *= (rcoords[2] - x3vpj1) / (x3vpi1 - x3vpj1);
        }
        result.dwght1[i] += dterm1 / (x1vpi1 - x1vpk1);
        result.dwght2[i] += dterm2 / (x2vpi1 - x2vpk1);
        result.dwght3[i] += dterm3 / (x3vpi1 - x3vpk1);
      }
    }
  }

  return result;
}

template <int NGHOST>
KOKKOS_INLINE_FUNCTION
Real InterpolateLagrange(
  const DvceArray5D<Real> &val,
  int var,
  const RegionIndcs &indcs,
  IndAndWghts<NGHOST> indcs_wghts
) {
  Real ival = 0.;
  int is = indcs.is;
  int js = indcs.js;
  int ks = indcs.ks;

  if (indcs_wghts.ii0 == -1) { // point not on this rank
    ival = 0.0;
  } else {
    for (int i = 0; i < 2 * NGHOST; i++) {
      for (int j = 0; j < 2 * NGHOST; j++) {
        for (int k = 0; k < 2 * NGHOST; k++) {
          Real iwght = indcs_wghts.wght1[i] * indcs_wghts.wght2[j] * indcs_wghts.wght3[k];
          ival += iwght * val(indcs_wghts.ii0, var,
                              indcs_wghts.ii3 - (NGHOST - k - ks) + 1,
                              indcs_wghts.ii2 - (NGHOST - j - js) + 1,
                              indcs_wghts.ii1 - (NGHOST - i - is) + 1);
        }
      }
    }
  }

  return ival;
}

template <int NGHOST>
KOKKOS_INLINE_FUNCTION
Real InterpolateLagrangeDerivative(
  const DvceArray5D<Real> &val,
  int var,
  int derivative_direction,
  const RegionIndcs &indcs,
  const IndAndWghts<NGHOST> &indcs_wghts
) {
  if (indcs_wghts.ii0 == -1) return 0.0;

  Real derivative = 0.;
  const int is = indcs.is;
  const int js = indcs.js;
  const int ks = indcs.ks;
  for (int i = 0; i < 2 * NGHOST; ++i) {
    for (int j = 0; j < 2 * NGHOST; ++j) {
      for (int k = 0; k < 2 * NGHOST; ++k) {
        Real weight;
        if (derivative_direction == 0) {
          weight = indcs_wghts.dwght1[i] * indcs_wghts.wght2[j]
                   * indcs_wghts.wght3[k];
        } else if (derivative_direction == 1) {
          weight = indcs_wghts.wght1[i] * indcs_wghts.dwght2[j]
                   * indcs_wghts.wght3[k];
        } else {
          weight = indcs_wghts.wght1[i] * indcs_wghts.wght2[j]
                   * indcs_wghts.dwght3[k];
        }
        derivative += weight * val(indcs_wghts.ii0, var,
                                    indcs_wghts.ii3 - (NGHOST - k - ks) + 1,
                                    indcs_wghts.ii2 - (NGHOST - j - js) + 1,
                                    indcs_wghts.ii1 - (NGHOST - i - is) + 1);
      }
    }
  }
  return derivative;
}

#endif  // UTILS_INLINE_INTERPOLATOR_HPP_
