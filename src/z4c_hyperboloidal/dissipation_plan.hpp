// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_DISSIPATION_PLAN_HPP_
#define Z4C_HYPERBOLOIDAL_DISSIPATION_PLAN_HPP_

#include "z4c_hyperboloidal/dissipation.hpp"
#include "z4c_hyperboloidal/point_state.hpp"

class MeshBlockPack;
namespace z4c_hyperboloidal {
class Z4cHyperboloidal;

// The five nodes of a difference stay in the target block's storage. Cartesian
// coarse/fine ghost nodes contain interpolation of the physical stored fields,
// separately from the residual interpolation used by the derivative operator.
class DissipationPlan {
 public:
  struct Kernel {
    DvceArray1D<unsigned> support;
    DvceArray2D<Real> coefficients;
    DvceArray1D<Real> volumes;

    KOKKOS_INLINE_FUNCTION Real Evaluate(int p, int f,
        const BlockPointReader &physical, Real epsilon) const {
      if (epsilon==0) return 0;
      const Real center = physical(f, 0, 0, 0);
      Real rate = 0;
      const unsigned mask = support(p);
      for (int axis = 0; axis < 3; ++axis)
        for (int k = 0; k < 8; ++k) {
          const int i = axis * 8 + k;
          if (mask & (1u << i)) {
            const Real neighbor = OffsetValue(physical, f, axis, k < 4 ? k - 4 : k - 3);
            rate += coefficients(i, p) * (neighbor - center);
          }
        }
      return epsilon * rate;
    }
  };

  void Rebuild(const MeshBlockPack &pack, const Z4cHyperboloidal &module);
  Kernel Bind() const { return {support_, coefficients_, volumes_}; }

 private:
  DvceArray1D<unsigned> support_;
  DvceArray2D<Real> coefficients_;
  DvceArray1D<Real> volumes_;
};
}  // namespace z4c_hyperboloidal
#endif
