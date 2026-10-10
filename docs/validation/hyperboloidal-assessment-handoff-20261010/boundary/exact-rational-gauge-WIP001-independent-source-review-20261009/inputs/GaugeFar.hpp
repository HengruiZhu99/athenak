// SOURCE-ONLY CPU arithmetic prototype; new identity, no native adoption.
#ifndef RESEARCH_REFERENCE_WAVE_MAP_ROBUST_OUTER_HPP_
#define RESEARCH_REFERENCE_WAVE_MAP_ROBUST_OUTER_HPP_
#include "arithmetic_traits.hpp"
#include "reference_wave_map_legacy.hpp"
namespace rwm {
// Joint positive-primal field closeness only. This is not a universal
// conditioning certificate for arbitrary metric/gradient contrast.
template<class T> inline bool UsesLegacyNear(
    const hyp::LayerPoint<T>&p,const hyp::Z4cJet<T>&u) {
  return inner::NearFieldValue(inner::Number<T>::Value(u.alpha.value),
                              inner::Number<T>::Value(p.alpha)) &&
         inner::NearFieldValue(inner::Number<T>::Value(u.chi.value),
                              inner::Number<T>::Value(p.state.chi.value));
}

// Real-arithmetic equivalent complete physical-P RWM rows, grouped far from
// reference. Complete factors reach Product before any scalar A0 underflow.
// ReferenceConnection, ScaledSource and Assemble remain the legacy definitions.
template<class T> inline hyp::GaugeRHSParts<T> GaugeFar(
    const hyp::LayerPoint<T>&p,const hyp::Z4cJet<T>&u,const Connection<T>&c) {
  hyp::GaugeRHSParts<T> q{};
  const T a=u.alpha.value,h=p.alpha,x=u.chi.value,y=p.state.chi.value;
  if(!c.valid||!(a>0)||!(x>0)||!(h>0)||!(y>0)||
     !Kokkos::isfinite(a)||!Kokkos::isfinite(x)||
     !Kokkos::isfinite(h)||!Kokkos::isfinite(y))return q;
  const auto g=hyp::Geometry(u.metric),gh=hyp::Geometry(p.state.metric);
  if(!g.valid||!gh.valid)return q;
  T db[3]{},dV[3][3]{},Lh[3][3]{},dL[3][3]{};
  for(int i=0;i<3;++i)db[i]=u.beta.value[i]-p.beta[i];
  for(int i=0;i<3;++i)for(int j=0;j<3;++j){
    dV[i][j]=inner::Product<T>({a,a,x,g.inverse[i][j]})-
             inner::Product<T>({h,h,y,gh.inverse[i][j]});
    Lh[i][j]=inner::Product<T>({h,h,y,gh.inverse[i][j]})-
             inner::Product<T>({p.beta[i],p.beta[j]});
    dL[i][j]=dV[i][j]-inner::Product<T>({db[i],u.beta.value[j]})-
                       inner::Product<T>({p.beta[i],db[j]});
  }
  q.pole.alpha=-inner::Product<T>({a,a,u.trace.value})+
                inner::Product<T>({a,h,p.k_physical});
  for(int i=0;i<3;++i){
    q.regular.alpha+=inner::Product<T>({u.beta.value[i],u.alpha.d[i]})-
                     inner::Product<T>({a,T(1)/h,p.beta[i],p.dalpha[i]});
    q.pole.alpha-=inner::Product<T>({a,db[i],p.domega[i]});
    for(int j=0;j<3;++j)
      q.pole.alpha-=inner::Product<T>({a,dL[i][j],c.scaled[0][i][j]});
  }
  for(int i=0;i<3;++i){
    q.regular.beta[i]=inner::Product<T>({a,a,x,u.lambda.value[i]})-
                      inner::Product<T>({h,h,y,p.state.lambda.value[i]});
    for(int j=0;j<3;++j){
      q.regular.beta[i]+=inner::Product<T>({u.beta.value[j],u.beta.d[j][i]-p.state.beta.d[j][i]})+
                        inner::Product<T>({db[j],p.state.beta.d[j][i]});
      q.regular.beta[i]+=inner::Product<T>({T(.5),a,a,g.inverse[i][j],u.chi.d[j]})-
                        inner::Product<T>({T(.5),h,h,gh.inverse[i][j],p.state.chi.d[j]})-
                        inner::Product<T>({a,x,g.inverse[i][j],u.alpha.d[j]})+
                        inner::Product<T>({h,y,gh.inverse[i][j],p.dalpha[j]});
      q.pole.beta[i]+=inner::Product<T>({T(2),dV[i][j],p.domega[j]});
      for(int l=0;l<3;++l)
        q.pole.beta[i]-=inner::Product<T>({dL[j][l],c.scaled[i+1][j][l]})+
                        inner::Product<T>({dL[j][l],u.beta.value[i],c.scaled[0][j][l]})+
                        inner::Product<T>({Lh[j][l],db[i],c.scaled[0][j][l]});
    }
  }
  q.valid=Kokkos::isfinite(q.regular.alpha)&&Kokkos::isfinite(q.pole.alpha);
  for(int i=0;i<3;++i)q.valid=q.valid&&Kokkos::isfinite(q.regular.beta[i])&&Kokkos::isfinite(q.pole.beta[i]);
  return q;
}

template<class T> inline hyp::GaugeRHSParts<T> Gauge(
    const hyp::LayerPoint<T>&p,const hyp::Z4cJet<T>&u,const Connection<T>&c) {
  if(UsesLegacyNear(p,u))return LegacyGauge(p,u,c);
  return GaugeFar(p,u,c);
}
} // namespace rwm
#endif
