// Actual C0 subsidiary plus the physical isotropic damping change.
// Reference linearization only; all coefficient derivatives are retained.
struct ConstraintJet {C q[8]{},d[3][8]{},dd[3][3][8]{};};
#include "subsidiary_c0.hpp"
std::array<C,8> SubsidiaryProfile(const hyp::LayerPoint<double>&p,double kap,
                                const ConstraintJet&q,bool keep_gradient=true) {
  auto out=SubsidiaryC0(p,kap,q);
  if(!g_profile)return out;
  const double o=p.omega,a=.5,S=1,critical=2*S/(a*a),d=kap-critical;
  const double k2=hyp::ResearchKappa2Profile(o,S,a,kap),m=kap*k2;
  // Delta Sij=-m/Omega*Theta*gammaij. Consequently H changes by
  // -4 K m Theta/Omega and M by +2 partial_i(m Theta/Omega).
  out[0]-=4*p.state.trace.value*m*q.q[7]/o;
  out[7]-=m*q.q[7]/o;
  for(int i=0;i<3;++i){
    out[1+i]+=2*m*q.d[i][7]/o-2*m*p.domega[i]*q.q[7]/(o*o);
    if(keep_gradient)out[1+i]+=2*d*p.domega[i]*q.q[7]/o;
  }
  return out;
}
std::array<C,8> Subsidiary(const hyp::LayerPoint<double>&p,double kap,
                         const ConstraintJet&q) {
  return SubsidiaryProfile(p,kap,q,true);
}
