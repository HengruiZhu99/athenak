#include "inputs/audit_helpers.hpp"
#include "inputs/nonlinear_values.hpp"
#include "inputs/four_gamma.hpp"
#include "early_feedback.hpp"
int main(){double reference=0,source=0,outer=0,pole=0,minnorm=1e100;int rows=0,outerrows=0,polerows=0;
 for(double a:{.5,.75,1.,2.}){hyp::LayerReference<double>ref(1,a,{true,.05,.95});hyp::LayerGaugeParameters g;g.scri_lapse_damping=1/a;
 for(double r:{.1,.44,.45,.5,.64,.65,.7,.8,.85,.9,.95,.98})for(const auto&n:{std::array<double,3>{1,0,0},std::array<double,3>{.36,-.48,.8}})for(int mode=0;mode<2;++mode){auto p=ref.At(r*n[0],r*n[1],r*n[2]);const auto u=Off(p);auto q=earlynf::Gauge(p,u,g,{5,mode});hyp::GaugeRHS<double>f{},base{},rf{};
 if(!q.valid||!qnf::Assemble(q,p.omega,f)||!qnf::Assemble(qnf::Gauge(p,u,g,{.85,.95,0,true}),p.omega,base)||!qnf::Assemble(earlynf::Gauge(p,p.state,g,{5,mode}),p.omega,rf))throw std::runtime_error("gauge invalid");reference=std::max(reference,std::abs(rf.alpha));for(int i=0;i<3;++i)reference=std::max(reference,std::abs(rf.beta[i]));
 const double W=earlynf::Weight(p,u,g,{5,mode});if(W>0){double norm=0;for(int i=0;i<3;++i)norm+=p.domega[i]*p.domega[i];minnorm=std::min(minnorm,norm);}
 hyp::Z4cRHS<double>geom{};const auto o=DoubleOmega(u,p);if(!hyp::AssembleInterior(hyp::ConformalRHS(u,o,10/u.alpha.value,0.),p.omega,geom))throw std::runtime_error("geometry invalid");double gamma[4]{},gb[4]{};FourGamma(u,geom,f,gamma);FourGamma(u,geom,base,gb);
 double dbox=0;for(int i=0;i<3;++i)dbox-=p.domega[i]*(gamma[i+1]-gb[i+1]);source=std::max(source,std::abs(dbox-W*5*qnf::NullDifference(p,u)/p.omega));source=std::max(source,std::abs(gamma[0]-gb[0]));
 if(r>=.95){auto old=qnf::Gauge(p,u,g,{.85,.95,5,true});outer=std::max(outer,std::abs(q.regular.alpha-old.regular.alpha));outer=std::max(outer,std::abs(q.pole.alpha-old.pole.alpha));for(int i=0;i<3;++i){outer=std::max(outer,std::abs(q.regular.beta[i]-old.regular.beta[i]));outer=std::max(outer,std::abs(q.pole.beta[i]-old.pole.beta[i]));}++outerrows;}++rows;
 }
 const auto p=ref.At(1,0,0);for(int mode=0;mode<2;++mode)for(int c=0;c<20;++c){auto u=Lift(p.state);Seed(u,c,J(D(0,1)));Consistent(u);auto f=earlynf::Gauge(Reference(p),u,g,{5,mode});auto old=qnf::Gauge(Reference(p),u,g,{.85,.95,5,true});pole=std::max(pole,std::abs(f.pole.alpha.d-old.pole.alpha.d));for(int i=0;i<3;++i)pole=std::max(pole,std::abs(f.pole.beta[i].d-old.pole.beta[i].d));++polerows;}}
 std::cout<<std::setprecision(17)<<"{\"source_rows\":"<<rows<<",\"outer_rows\":"<<outerrows<<",\"pole_columns\":"<<polerows<<",\"reference_fixedpoint_error\":"<<reference<<",\"generic_4D_Box_feedback_delta_error\":"<<source<<",\"identical_outer_gauge_error\":"<<outer<<",\"identical_leading_gauge_pole_error\":"<<pole<<",\"min_sampled_gradient_norm2_on_support\":"<<minnorm<<"}\n";
}
