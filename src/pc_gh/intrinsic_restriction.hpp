// AthenaK astrophysical plasma code, 3-clause BSD License (LICENSE).
#ifndef PC_GH_INTRINSIC_RESTRICTION_HPP_
#define PC_GH_INTRINSIC_RESTRICTION_HPP_
#include "athena.hpp"
namespace pc_gh::intrinsic {
// Point-value restriction, degree five per axis. Shift support inside the
// active source block: restriction precedes the new-stage ghost exchange.
KOKKOS_INLINE_FUNCTION
void PointRestrictionWeights(int fine, int first, int last, int &start, Real w[6]) {
  start=fine-2;
  if (start<first) start=first;
  if (start>last-5) start=last-5;
  const Real x=fine+0.5-start;
  for (int a=0;a<6;++a) {
    w[a]=1;
    for (int b=0;b<6;++b) if (a!=b) w[a]*=(x-b)/(a-b);
  }
}
template<typename State>
KOKKOS_INLINE_FUNCTION
Real PointRestrict2D(State u,int m,int n,int k,int fj,int fi,
                     int is,int ie,int js,int je) {
  int si,sj;Real wi[6],wj[6];
  PointRestrictionWeights(fi,is,ie,si,wi);
  PointRestrictionWeights(fj,js,je,sj,wj);
  Real result=0;
  for (int b=0;b<6;++b) for (int a=0;a<6;++a)
    result+=wj[b]*wi[a]*u(m,n,k,sj+b,si+a);
  return result;
}
}  // namespace pc_gh::intrinsic
#endif
