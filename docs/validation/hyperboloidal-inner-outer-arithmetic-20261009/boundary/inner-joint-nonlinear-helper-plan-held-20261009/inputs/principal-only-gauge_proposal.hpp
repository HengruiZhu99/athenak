// SOURCE-ONLY proposal, not compiled or admitted. No production edits.
#include "reference_wave_map.hpp"
namespace ijp {
namespace hyp=z4c::hyperboloidal;
struct Parameters {
  bool candidate=false;
  double W=0,G0=.375,f=3,mu=.375;
};
template<class T> hyp::GaugeRHSParts<T> Gauge(
    const hyp::LayerPoint<T>&p,const hyp::Z4cJet<T>&u,const Parameters&v) {
  const T a=u.alpha.value,chi=u.chi.value;
  if(!(a>0)||!(chi>0)||!Kokkos::isfinite(a)||!Kokkos::isfinite(chi))return {};
  const T A=a*a*chi;
  const auto geo=hyp::Geometry(u.metric);
  if(!geo.valid)return {};
  if(v.candidate){
    // Frozen constant reference for principal extraction only. Nonflat source
    // identity/ReferenceConnection gates are explicitly outside this proposal.
    rwm::Connection<T> connection{};connection.valid=true;
    auto out=rwm::Gauge(p,u,connection);
    const T c=1-T(v.W),B=c*T(v.G0)+T(v.W)*A,k=B/(A+B);
    out.pole.alpha-=2*c*a*(u.trace.value-p.k_physical);
    for(int i=0;i<3;++i){
      out.regular.beta[i]+=(B-A)*(u.lambda.value[i]-p.state.lambda.value[i]);
      for(int j=0;j<3;++j)out.regular.beta[i]+=a*a*(2*k*k-T(.5))*geo.inverse[i][j]*
        (u.chi.d[j]-chi*p.state.chi.d[j]/p.state.chi.value);
    }
    out.valid=out.valid&&Kokkos::isfinite(out.regular.alpha)&&Kokkos::isfinite(out.pole.alpha);
    for(int i=0;i<3;++i)out.valid=out.valid&&Kokkos::isfinite(out.regular.beta[i])&&Kokkos::isfinite(out.pole.beta[i]);
    return out;
  }
  hyp::GaugeRHSParts<T> out{};
  const T mu=T(v.mu),ec=2*mu*mu/((1+mu)*(1+mu));
  out.pole.alpha=-a*a*T(v.f)*(u.trace.value-p.k_physical);
  for(int i=0;i<3;++i){
    out.regular.beta[i]=A*mu*(u.lambda.value[i]-p.state.lambda.value[i]);
    for(int j=0;j<3;++j)out.regular.beta[i]+=geo.inverse[i][j]*(
      a*a*ec*(u.chi.d[j]-chi*p.state.chi.d[j]/p.state.chi.value)-
      a*chi*(u.alpha.d[j]-a*p.dalpha[j]/p.alpha));
  }
  out.valid=Kokkos::isfinite(out.pole.alpha);
  for(int i=0;i<3;++i)out.valid=out.valid&&Kokkos::isfinite(out.regular.beta[i]);
  return out;
}
}
