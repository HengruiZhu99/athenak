#include "profile_helpers.hpp"
int main(){double difference=0,einstein=0,reference=0,lo=0,hi=-1,finite=0;int rows=0;
 for(double a:{.5,.75,1.,2.}){hyp::LayerReference<double>ref(1,a,{true,.05,.95});
 for(int ri=0;ri<=1000;++ri){double r=ri/1000.;auto p=ref.At(r,0.,0.);auto u=Background(p,.01);auto o=Omega(u,p);
 const D k2=hyp::ResearchKappa2Profile(o.omega,D(1),D(a),D(10));lo=std::min(lo,k2.v);hi=std::max(hi,k2.v);if(!std::isfinite(k2.v))return 2;
 auto base=hyp::ConformalRHS(u,o,D(10)/u.alpha.value,D(0)),change=hyp::ConformalRHS(u,o,D(10)/u.alpha.value,k2);
 hyp::GaugeRHS<D>empty{};auto b=Fields(base.regular,empty),c=Fields(change.regular,empty);
 for(int i=0;i<20;++i)difference=std::max(difference,std::abs(c[i].v-b[i].v));
 b=Fields(base.pole,empty);c=Fields(change.pole,empty);
 for(int i=0;i<20;++i){D expected=(i==2||i==3)?-D(10)*k2*u.theta.value:D(0);difference=std::max(difference,std::abs((c[i]-b[i]-expected).v)/(1+std::abs(c[i].v)+std::abs(b[i].v)));}
 u.theta.value=0;u.theta.d[0]=u.theta.d[1]=u.theta.d[2]=0;
 base=hyp::ConformalRHS(u,o,D(10)/u.alpha.value,D(0));change=hyp::ConformalRHS(u,o,D(10)/u.alpha.value,k2);
 b=Fields(base.pole,empty);c=Fields(change.pole,empty);for(int i=0;i<20;++i)einstein=std::max(einstein,std::abs((c[i]-b[i]).v));
 if(p.omega>0)for(auto v:Evaluate(p,Background(p,0),a,10,1,true)){reference=std::max(reference,std::abs(v.v));if(!std::isfinite(v.v))return 3;}
 if(hyp::ResearchKappa2Profile(1.,1.,a,10.)!=0)return 4;++rows;
 }}
 std::cout<<std::setprecision(17)<<"{\"rows\":"<<rows<<",\"exact_core_kappa2_zero\":true,\"Einstein_addition_max\":"<<einstein<<",\"nonlinear_parts_identity_error\":"<<difference<<",\"reference_fixedpoint\":"<<reference<<",\"kappa2_min\":"<<lo<<",\"kappa2_max\":"<<hi<<",\"no_new_double_pole\":true}\n";
}
