// SCRATCH ONLY. Mechanical tensor Appendix-B C_Z4c additions in physical-P storage.
// No scri closure: arbitrary finite physical Theta produces a genuine double pole.
#ifndef SCRATCH_C1_ADDITIONS_HPP_
#define SCRATCH_C1_ADDITIONS_HPP_
#include "z4c/hyperboloidal/conformal_rhs.hpp"
namespace z4c { namespace hyperboloidal {
template <typename T> struct C1Parts {
  Z4cRHS<T> regular{},pole{},double_pole{};
};
template <typename T>
C1Parts<T> TensorC1Additions(const Z4cJet<T>&u,const OmegaJet<T>&o,
                           T coefficient,bool covariant_connection=false) {
  C1Parts<T> out;
  if(coefficient==T(0))return out;
  const auto gt=Geometry(u.metric);
  const T a=u.alpha.value,c=u.chi.value,t=u.theta.value,O=o.omega;
  T zt[3]{};for(int i=0;i<3;++i)zt[i]=(u.lambda.value[i]-gt.contracted[i])/2;
  out.pole.theta=-coefficient*a*t*(u.trace.value+2*t-3*o.normal);
  for(int i=0;i<3;++i) {
    out.regular.trace+=coefficient*2*c*zt[i]*(O*u.alpha.d[i]-a*o.gradient[i]);
    out.regular.theta-=coefficient*(O*c*zt[i]*u.alpha.d[i]+a*O*zt[i]*u.chi.d[i]/2);
    for(int j=0;j<3;++j) {
      out.pole.a[i][j]=-coefficient*2*a*u.a.k[i][j]*t;
      out.pole.lambda[i]-=coefficient*2*t*gt.inverse[i][j]*u.alpha.d[j];
      out.double_pole.lambda[i]+=coefficient*2*a*t*gt.inverse[i][j]*o.gradient[j];
      // Separate derived covector completion; NOT part of Appendix-B C terms.
      if(covariant_connection)
        out.regular.lambda[i]-=coefficient*2*zt[j]*u.beta.d[j][i];
    }
  }
  return out;
}
template <typename T>
bool AssembleC1Interior(const C1Parts<T>&p,T O,Z4cRHS<T>&f) {
  if(!(O>0)||!Kokkos::isfinite(O))return false;
  f=p.regular;
  f.chi+=p.pole.chi/O+p.double_pole.chi/(O*O);
  f.trace+=p.pole.trace/O+p.double_pole.trace/(O*O);
  f.theta+=p.pole.theta/O+p.double_pole.theta/(O*O);
  bool valid=Kokkos::isfinite(f.chi)&&Kokkos::isfinite(f.trace)&&Kokkos::isfinite(f.theta);
  for(int i=0;i<3;++i){f.lambda[i]+=p.pole.lambda[i]/O+p.double_pole.lambda[i]/(O*O);
    valid=valid&&Kokkos::isfinite(f.lambda[i]);
    for(int j=0;j<3;++j){f.metric[i][j]+=p.pole.metric[i][j]/O+p.double_pole.metric[i][j]/(O*O);
      f.a[i][j]+=p.pole.a[i][j]/O+p.double_pole.a[i][j]/(O*O);
      valid=valid&&Kokkos::isfinite(f.metric[i][j])&&Kokkos::isfinite(f.a[i][j]);}}
  return valid;
}
}}
#endif
