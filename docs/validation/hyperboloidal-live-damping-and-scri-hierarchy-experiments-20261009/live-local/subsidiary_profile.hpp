// Exact C0 ADM correction: Delta M_i=2 partial_i(m Theta/Omega).
struct ConstraintJet {C q[8]{},d[3][8]{},dd[3][3][8]{};};
#include "subsidiary_c0.hpp"
std::array<C,8> SubsidiaryProfile(const hyp::LayerPoint<double>&p,double kap,
                                const ConstraintJet&q,bool keep_gradient=true) {
 auto out=SubsidiaryC0(p,kap,q);if(!g_liveprofile)return out;
 hyp::OmegaJet<double>o{};o.omega=p.omega;
 for(int i=0;i<3;++i){o.gradient[i]=p.domega[i];for(int j=0;j<3;++j)o.hessian[i][j]=p.omega_hessian[i][j];}
 const double n[3]={1,0,0}; // audit coefficient points lie on positive x axis
 const auto k2=hyp::ResearchLiveKappa2SpatialJet(p.state,o,p.radius,n,kap);
 const double m=kap*k2.value,om=p.omega;
 out[0]-=4*p.state.trace.value*m*q.q[7]/om;out[7]-=m*q.q[7]/om;
 for(int i=0;i<3;++i){out[1+i]+=2*m*q.d[i][7]/om-2*m*p.domega[i]*q.q[7]/(om*om);
  if(keep_gradient)out[1+i]+=2*kap*k2.d[i]*q.q[7]/om;}
 return out;
}
std::array<C,8> Subsidiary(const hyp::LayerPoint<double>&p,double kap,
                         const ConstraintJet&q){return SubsidiaryProfile(p,kap,q,true);}
