#ifndef DRIVER_CLASSICAL_RK4_UPDATE_HPP_
#define DRIVER_CLASSICAL_RK4_UPDATE_HPP_
#include <cmath>
#include <stdexcept>
#include "driver/block_batches.hpp"
#include "driver/classical_rk4.hpp"
#include "z4c/z4c_grid.hpp"

namespace classical_rk4 {
// Numerical update only. Callers own stage boundary conditions/projection and
// copy their selected beginning-of-step state before stage1. Neither global
// mesh time nor global dt is read, so the same kernel can advance child levels.
inline void Update(const subcycling::BlockBatches &blocks,
                   const z4c::Z4cGridLayout &l, double dt, int stage,
                   const DvceArray5D<Real> &state,
                   const DvceArray5D<Real> &initial,
                   const DvceArray5D<Real> &rhs, DvceArray5D<Real> &sum,
                   int zero_first=1, int zero_last=0) {
  if (stage<1 || stage>4 || !std::isfinite(dt) || dt<=0)
    throw std::invalid_argument("invalid local RK stage");
  const int count=blocks.Count(), nv=state.extent_int(1);
  if (count>state.extent_int(0) || nv<1 || l.is<0 || l.js<0 || l.ks<0 ||
      l.ie<l.is || l.je<l.js || l.ke<l.ks || l.ie>=state.extent_int(4) ||
      l.je>=state.extent_int(3) || l.ke>=state.extent_int(2))
    throw std::invalid_argument("invalid local RK active storage");
  for(int d=0;d<5;++d) if (initial.extent(d)!=state.extent(d) || rhs.extent(d)!=state.extent(d))
    throw std::invalid_argument("incompatible local RK field arrays");
  bool resize=sum.extent_int(0)!=count;
  for(int d=1;d<5;++d) resize=resize || sum.extent(d)!=state.extent(d);
  if(resize) {
    if(stage!=1) throw std::logic_error("RK storage changed inside a step");
    Kokkos::realloc(sum,count,nv,state.extent(2),state.extent(3),state.extent(4));
  }
  const Real weight=Weight(stage),next=NextFraction(stage);
  blocks.For5("z4c classical RK4 update",0,nv-1,l.ks,l.ke,l.js,l.je,l.is,l.ie,
      KOKKOS_LAMBDA(int m,int n,int k,int j,int i) {
    const Real value=rhs(m,n,k,j,i);
    sum(m,n,k,j,i)=(stage==1 ? 0.0 : sum(m,n,k,j,i))+weight*value;
    state(m,n,k,j,i)=n>=zero_first && n<=zero_last ? 0.0 :
        initial(m,n,k,j,i)+dt*(stage==4 ? sum(m,n,k,j,i) : next*value);
  });
}
}  // namespace classical_rk4
#endif  // DRIVER_CLASSICAL_RK4_UPDATE_HPP_
