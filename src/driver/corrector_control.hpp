#ifndef DRIVER_CORRECTOR_CONTROL_HPP_
#define DRIVER_CORRECTOR_CONTROL_HPP_
#include <cmath>
#include <limits>
#include <stdexcept>
#include "athena.hpp"
#include "z4c/z4c_grid.hpp"
namespace subcycling {
struct CorrectorControl {
  int minimum_passes=3,maximum_passes=8;
  Real absolute_tolerance=1e-12,relative_tolerance=1e-10;
  void Validate() const {
    if(minimum_passes<3 || maximum_passes<minimum_passes ||
       !std::isfinite(absolute_tolerance) || absolute_tolerance<=0 ||
       !std::isfinite(relative_tolerance) || relative_tolerance<0)
      throw std::invalid_argument("invalid hierarchy corrector control");
  }
};
struct CorrectorReport {
  int passes=0;
  Real endpoint_change=std::numeric_limits<Real>::infinity();
  Real history_change=std::numeric_limits<Real>::infinity();
  Real feedback_change=std::numeric_limits<Real>::infinity();
  bool converged=false;
};
class CorrectorFailure : public std::runtime_error {
 public:
  CorrectorFailure():std::runtime_error("hierarchy corrector failed to converge within its pass budget") {}
};
struct IntervalRetryControl {
  int maximum_halvings=8;
  double minimum_dt=0;
  void Validate() const {
    if(maximum_halvings<0 || maximum_halvings>20 ||
       !std::isfinite(minimum_dt) || minimum_dt<0)
      throw std::invalid_argument("invalid hierarchy interval retry control");
  }
};
struct AcceptedInterval {
  double dt=0;
  int attempts=0, total_passes=0;
  CorrectorReport corrector;
};
// Max scaled difference on explicitly selected active bounds. Also used on
// packed active RK histories, with dt weighting for RHS (increment units).
inline Real CorrectorDifference(const DvceArray5D<Real> &a,const DvceArray5D<Real> &b,
    const z4c::Z4cGridLayout &l,const CorrectorControl &control,Real scale=1) {
  control.Validate();
  for(int d=0;d<5;++d) if(a.extent(d)!=b.extent(d))
    throw std::invalid_argument("corrector comparison shape mismatch");
  if(l.is<0 || l.js<0 || l.ks<0 || l.ie<l.is || l.je<l.js || l.ke<l.ks ||
     l.ie>=a.extent_int(4) || l.je>=a.extent_int(3) || l.ke>=a.extent_int(2) ||
     !std::isfinite(scale) || scale<=0)
    throw std::invalid_argument("invalid corrector comparison bounds/scale");
  const int ni=l.ie-l.is+1,nj=l.je-l.js+1,nk=l.ke-l.ks+1,nv=a.extent_int(1);
  const int is=l.is,js=l.js,ks=l.ks;
  const std::size_t count=a.extent(0)*nv*nk*nj*ni;
  const Real absolute=control.absolute_tolerance,relative=control.relative_tolerance;
  Real error=0;
  Kokkos::parallel_reduce("hierarchy corrector difference",
    Kokkos::RangePolicy<DevExeSpace,Kokkos::IndexType<std::size_t>>(0,count),
    KOKKOS_LAMBDA(std::size_t q,Real &maximum) {
      const int i=q%ni+is;q/=ni;const int j=q%nj+js;q/=nj;
      const int k=q%nk+ks;q/=nk;const int v=q%nv;const int m=q/nv;
      const Real x=scale*a(m,v,k,j,i),y=scale*b(m,v,k,j,i);
      const Real denominator=absolute+relative*fmax(fabs(x),fabs(y));
      const Real value=Kokkos::isfinite(x) && Kokkos::isfinite(y) ?
        fabs(x-y)/denominator : INFINITY;
      if(value>maximum) maximum=value;
    },Kokkos::Max<Real>(error));
  return error;
}
}
#endif
