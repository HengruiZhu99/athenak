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
struct KOLine {int axis, target; Real weight;};

class DissipationPlan {
 public:
  struct Kernel {
    DvceArray1D<int> offsets;
    DvceArray1D<KOLine> lines;
    DvceArray1D<Real> volumes;
    int n;
    KOKKOS_INLINE_FUNCTION Real Evaluate(int p, int f,
        const BlockPointReader &physical, Real epsilon) const {
      if (epsilon==0) return 0;
      Real rate=0;
      for (int t=offsets(p);t<offsets(p+1);++t) {
        const auto line=lines(t);
        Real values[PhysicalKO8::points];
        for (int a=0;a<PhysicalKO8::points;++a)
          values[a]=OffsetValue(physical,f,line.axis,a-line.target);
        rate+=PhysicalKO8::Line(values,line.target,line.weight,volumes(p),n,epsilon);
      }
      return rate;
    }
  };

  void Rebuild(const MeshBlockPack &pack, const Z4cHyperboloidal &module);
  Kernel Bind() const {return {offsets_,lines_,volumes_,n_};}

 private:
  DvceArray1D<int> offsets_;
  DvceArray1D<KOLine> lines_;
  DvceArray1D<Real> volumes_;
  int n_=0;
};
}  // namespace z4c_hyperboloidal
#endif
