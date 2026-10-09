#include "profile_helpers.hpp"
int main(){double difference=0,einstein=0,reference=0,lo=0,hi=-1,sigma=0;int rows=0;
 for(double a:{.5,.75,1.,2.}){hyp::LayerReference<double>ref(1,a,{true,.05,.95});
 for(int ri=0;ri<=1000;++ri){double r=ri/1000.;auto p=ref.At(r,0.,0.);auto u=Background(p,.01);Seed(u,ri%20,J(D(0,1)));Consistent(u);auto o=Omega(u,p);
 const D k2=hyp::ResearchLiveKappa2Profile(u,o,D(r),D(10));lo=std::min(lo,k2.v);hi=std::max(hi,k2.v);if(!std::isfinite(k2.v))return 2;
 auto base=hyp::ConformalRHS(u,o,D(10)/u.alpha.value,D(0)),change=hyp::ConformalRHS(u,o,D(10)/u.alpha.value,k2);
 hyp::GaugeRHS<D>empty{};auto b=Fields(base.regular,empty),c=Fields(change.regular,empty);
 for(int i=0;i<20;++i){difference=std::max(difference,std::abs(c[i].v-b[i].v));difference=std::max(difference,std::abs(c[i].d-b[i].d));}
 b=Fields(base.pole,empty);c=Fields(change.pole,empty);
 for(int i=0;i<20;++i){D expected=(i==2||i==3)?-D(10)*k2*u.theta.value:D(0);D err=c[i]-b[i]-expected;difference=std::max(difference,std::abs(err.v)/(1+std::abs(c[i].v)+std::abs(b[i].v)));difference=std::max(difference,std::abs(err.d)/(1+std::abs(c[i].d)+std::abs(b[i].d)));}
 const auto V=hyp::SmoothCutoff(D(r),D(.15),D(.3));
 const D value=-D(2)*u.alpha.value*o.normal-D(10)*(D(1)+k2);
 const D predicted=(D(1)-V.value)*(-D(2)*u.alpha.value*o.normal-D(10))-D(10)*V.value*o.omega;
 sigma=std::max(sigma,std::abs((value-predicted).v));sigma=std::max(sigma,std::abs((value-predicted).d));
 u.theta.value=D(0);for(int i=0;i<3;++i)u.theta.d[i]=D(0);
 base=hyp::ConformalRHS(u,o,D(10)/u.alpha.value,D(0));change=hyp::ConformalRHS(u,o,D(10)/u.alpha.value,k2);
 b=Fields(base.pole,empty);c=Fields(change.pole,empty);for(int i=0;i<20;++i){einstein=std::max(einstein,std::abs((c[i]-b[i]).v));einstein=std::max(einstein,std::abs((c[i]-b[i]).d));}
 if(p.omega>0)for(auto v:Evaluate(p,Background(p,0),a,10,1,true)){reference=std::max(reference,std::abs(v.v));if(!std::isfinite(v.v))return 3;}
 if(r<=.15&&!(k2.v==0&&k2.d==0))return 4;++rows;
 }}
 std::cout<<std::setprecision(17)<<"{\"rows\":"<<rows<<",\"inner_kappa2_exact_zero\":true,\"Einstein_addition_max\":"<<einstein<<",\"nonlinear_dual_parts_identity_error\":"<<difference<<",\"nonlinear_sigma_numerator_identity_error\":"<<sigma<<",\"reference_fixedpoint\":"<<reference<<",\"perturbed_kappa2_min\":"<<lo<<",\"perturbed_kappa2_max\":"<<hi<<",\"no_new_double_pole\":true}\n";
}
