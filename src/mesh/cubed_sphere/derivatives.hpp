#ifndef MESH_CUBED_SPHERE_DERIVATIVES_HPP_
#define MESH_CUBED_SPHERE_DERIVATIVES_HPP_

#include "mesh/cubed_sphere/map.hpp"
#include "utils/finite_diff.hpp"

namespace cubed_sphere {
struct Stencil {
  Real idx[3];
  bool mapped = false;
  Point p;
};
// RegionSize and RegionIndcs are defined by mesh.hpp before this header is used.
KOKKOS_INLINE_FUNCTION Stencil MakeStencil(const GeometryView &geometry,
    const RegionSize &size, const RegionIndcs &indcs, int m,int k,int j,int i) {
  Stencil s;
  s.idx[0]=1/size.dx1; s.idx[1]=1/size.dx2; s.idx[2]=1/size.dx3;
  s.mapped=geometry.extent(0)!=0;
  if (s.mapped) s.p=geometry(m,k,j,i);
  return s;
}
struct UnitVelocity {
  Real sign;
  KOKKOS_INLINE_FUNCTION Real operator()(int,int,int,int,int) const {return sign;}
};
}  // namespace cubed_sphere

template<int N, typename T, typename... I>
KOKKOS_INLINE_FUNCTION Real Dx(int a, const cubed_sphere::Stencil &s, T &q,
                              int m, I... indices) {
  if (!s.mapped) return ::Dx<N>(a,s.idx,q,m,indices...);
  Real r=0;
  for (int A=0;A<3;++A) r+=s.p.J[A][a]*::Dx<N>(A,s.idx,q,m,indices...);
  return r;
}
template<int N, typename T, typename... I>
KOKKOS_INLINE_FUNCTION Real Dxy(int a,int b,const cubed_sphere::Stencil &s,T &q,
                               int m,I... indices) {
  if (!s.mapped) return a==b ? ::Dxx<N>(a,s.idx,q,m,indices...)
                            : ::Dxy<N>(a,b,s.idx,q,m,indices...);
  Real r=0;
  for (int A=0;A<3;++A) {
    r+=s.p.H[A][a][b]*::Dx<N>(A,s.idx,q,m,indices...);
    for (int B=0;B<3;++B) {
      Real dd=A==B ? ::Dxx<N>(A,s.idx,q,m,indices...)
                   : ::Dxy<N>(A,B,s.idx,q,m,indices...);
      r+=s.p.J[A][a]*s.p.J[B][b]*dd;
    }
  }
  return r;
}
template<int N, typename T, typename... I>
KOKKOS_INLINE_FUNCTION Real Dxx(int a,const cubed_sphere::Stencil &s,T &q,
                               int m,I... indices) {
  return ::Dxy<N>(a,a,s,q,m,indices...);
}
template<int N, typename V, typename T, typename... I>
KOKKOS_INLINE_FUNCTION Real Lx(int dir,const cubed_sphere::Stencil &s,
    const V &v,const T &q,int m,int a,I... indices) {
  if (!s.mapped) return ::Lx<N>(dir,s.idx,v,q,m,a,indices...);
  const int ii[]={indices...};
  constexpr int n=sizeof...(indices);
  int k=ii[n-3],j=ii[n-2],i=ii[n-1];
  Real r=0;
  for (int A=0;A<3;++A) {
    Real vq=0;
    for (int b=0;b<3;++b) vq+=s.p.J[A][b]*v(m,b,k,j,i);
    cubed_sphere::UnitVelocity unit{vq<0 ? Real(-1) : Real(1)};
    r+=s.p.J[A][dir]*v(m,a,k,j,i)*unit.sign*
        ::Lx<N>(A,s.idx,unit,q,m,a,indices...);
  }
  return r;
}
template<int N,typename T>
KOKKOS_INLINE_FUNCTION Real Diss(int A,const cubed_sphere::Stencil &s,T &q,
                                 int m,int n,int k,int j,int i) {
  Real scale=1;
  if (s.mapped) {
    scale=0;
    for (int a=0;a<3;++a) scale+=s.p.J[A][a]*s.p.J[A][a];
    scale=sqrt(scale);
  }
  return scale*::Diss<N>(A,s.idx,q,m,n,k,j,i);
}
#endif
