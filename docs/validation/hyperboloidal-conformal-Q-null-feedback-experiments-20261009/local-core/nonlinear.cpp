#include "audit_helpers.hpp"
#include "four_gamma.hpp"
#include "nonlinear_values.hpp"
double Maximum(const hyp::GaugeRHSParts<double>&q){double x=std::max(std::abs(q.regular.alpha),std::abs(q.pole.alpha));for(int i=0;i<3;++i){x=std::max(x,std::abs(q.regular.beta[i]));x=std::max(x,std::abs(q.pole.beta[i]));}return x;}
int main(){hyp::LayerGaugeParameters g;double referr=0,factorerr=0,nullerr=0,sourceerr=0,boxerr=0,blenderr=0,coreerr=0,weightederr=0,tiny_noncore_max=0,continuity=0;int rows=0,source_rows=0,tiny_rows=0,tiny_noncore_rows=0;
 for(double a:{.5,.75,1.,2.})for(double xi:{1.5,1./a}){g.scri_lapse_damping=xi;hyp::LayerReference<double>ref(1,a,{true,.05,.95});
 for(double r:{.01,.1,.2,.44,.45,.450001,.5,.65,.8,.849999,.85,.850001,.86,.9,.94,.949999,.95,.950001,.98,.9999})for(const auto&n:{std::array<double,3>{1,0,0},std::array<double,3>{.36,-.48,.8}}){auto p=ref.At(r*n[0],r*n[1],r*n[2]);auto u=Off(p);const auto c=hyp::LayerCoefficients(p.radius,u.alpha.value,g);
 for(bool inner:{false,true}){const auto qr=qnf::Gauge(p,p.state,g,{.85,.95,5,inner});hyp::GaugeRHS<double>rr{};if(!qnf::Assemble(qr,p.omega,rr))throw std::runtime_error("reference invalid");referr=std::max(referr,std::abs(rr.alpha));for(int i=0;i<3;++i)referr=std::max(referr,std::abs(rr.beta[i]));
 auto q=qnf::Gauge(p,u,g,{.85,.95,5,inner});if(!q.valid)throw std::runtime_error("offconstraint invalid");auto b=hyp::InteriorLayerGauge(p,u,g);auto pg=g;pg.physical_trace_lapse=true;pg.preferred_source=false;auto ph=hyp::InteriorLayerGauge(p,u,pg);const double W=c.weight;
 blenderr=std::max(blenderr,std::abs(q.regular.alpha-(inner?(1-W)*ph.regular.alpha+W*b.regular.alpha:b.regular.alpha)));
 blenderr=std::max(blenderr,std::abs(q.pole.alpha-(inner?(1-W)*ph.pole.alpha+W*b.pole.alpha:b.pole.alpha)));
 ++rows;}
 double Dm=0;for(int i=0;i<3;++i)Dm+=(p.beta[i]*(u.alpha.value-p.alpha)/p.alpha-(u.beta.value[i]-p.beta[i]))*p.domega[i];
 const auto qb=qnf::FactoredBaseGauge(p,u,g);factorerr=std::max(factorerr,std::abs(qb.pole.alpha-(-c.alpha2f*(u.trace.value-p.k_physical)+3*(u.alpha.value+2*(1-c.weight))*Dm)));
 const auto ol=DoubleOmega(u,p),oh=DoubleOmega(p.state,p);const auto gl=hyp::Geometry(u.metric),gh=hyp::Geometry(p.state.metric);double G=0,Gh=0;for(int i=0;i<3;++i)for(int j=0;j<3;++j){G+=u.chi.value*gl.inverse[i][j]*p.domega[i]*p.domega[j];Gh+=p.state.chi.value*gh.inverse[i][j]*p.domega[i]*p.domega[j];}
 weightederr=std::max(weightederr,std::abs(qnf::WeightedNullDifference(p,u)-u.alpha.value*u.alpha.value*qnf::NullDifference(p,u)));
 nullerr=std::max(nullerr,std::abs(qnf::NullDifference(p,u)-((G-ol.normal*ol.normal)-(Gh-oh.normal*oh.normal))));
 if(r>=.85){hyp::Z4cRHS<double>geom{};hyp::GaugeRHS<double>f{};if(!hyp::AssembleInterior(hyp::ConformalRHS(u,ol,10/u.alpha.value,0.),p.omega,geom)||!qnf::Assemble(qnf::Gauge(p,u,g,{.85,.95,5,true}),p.omega,f))throw std::runtime_error("identity invalid");double gamma[4]{};FourGamma(u,geom,f,gamma);double refadv=0;for(int i=0;i<3;++i)refadv+=u.beta.value[i]*p.dalpha[i]/p.alpha;
 const double f0=(refadv+c.nu*std::log(u.alpha.value/p.alpha))/(u.alpha.value*u.alpha.value)-p.k_bar/u.alpha.value;
 sourceerr=std::max(sourceerr,std::abs(gamma[0]+2*u.theta.value/(u.alpha.value*p.omega)-f0));
 double box=0,H4=0,z=0,contract=0;for(int i=0;i<3;++i){double perp=f.beta[i];for(int j=0;j<3;++j)perp-=u.beta.value[j]*u.beta.d[j][i];double Fi=u.chi.value*u.lambda.value[i]-perp/(u.alpha.value*u.alpha.value)-u.beta.value[i]*f0;
 for(int j=0;j<3;++j){Fi+=u.chi.value*gl.inverse[i][j]*(u.chi.d[j]/(2*u.chi.value)-u.alpha.d[j]/u.alpha.value);H4+=(u.chi.value*gl.inverse[i][j]-u.beta.value[i]*u.beta.value[j]/(u.alpha.value*u.alpha.value))*p.omega_hessian[i][j];}
 double Zi=.5*u.chi.value*(u.lambda.value[i]-gl.contracted[i])-u.beta.value[i]*u.theta.value/(u.alpha.value*p.omega);sourceerr=std::max(sourceerr,std::abs(gamma[i+1]+2*Zi-Fi));contract+=Fi*p.domega[i];z+=Zi*p.domega[i];box-=gamma[i+1]*p.domega[i];}
 const double v=hyp::SmoothCutoff(p.radius,.85,.95).value,target=p.omega*p.w_omega+2*z+v*5*qnf::NullDifference(p,u)/p.omega;
 boxerr=std::max(boxerr,std::abs(H4+box-target));boxerr=std::max(boxerr,std::abs(contract-(H4-p.omega*p.w_omega-v*5*qnf::NullDifference(p,u)/p.omega)));++source_rows;
 }
 }
 // Exact Cauchy core: arbitrary positive alpha including subnormal values.
 auto p=ref.At(.01,0,0);auto ph=g;ph.physical_trace_lapse=true;ph.preferred_source=false;
 for(double alpha:{1e-4,1e-30,1e-100,1e-200,1e-300,1e-320}){auto u=p.state;u.alpha.value=alpha;u.trace.value+=.03;auto q=qnf::Gauge(p,u,g,{.85,.95,5,true});auto target=qnf::FactoredBaseGauge(p,u,ph);if(!q.valid||!std::isfinite(Maximum(q)))throw std::runtime_error("tiny lapse core invalid");coreerr=std::max(coreerr,std::abs(q.pole.alpha-target.pole.alpha));++tiny_rows;}
 for(double r:{.5,.9})for(double alpha:{1e-4,1e-100,1e-200,1e-300,1e-320})for(bool zero_beta:{false,true})for(bool inner:{false,true}){
 auto p2=ref.At(r,0,0);auto u=Off(p2);u.alpha.value=alpha;if(zero_beta)for(int i=0;i<3;++i)u.beta.value[i]=0;
 const auto q=qnf::Gauge(p2,u,g,{.85,.95,5,inner});if(!q.valid||!std::isfinite(Maximum(q)))throw std::runtime_error("tiny noncore source invalid");tiny_noncore_max=std::max(tiny_noncore_max,Maximum(q));++tiny_noncore_rows;
 }
 for(double r:{.5,.9})for(double ratio:{.5-1e-8,.5,.5+1e-8,.9,1.,1.1}){auto p2=ref.At(r,0,0);auto u=Off(p2);u.alpha.value=ratio*p2.alpha;
 weightederr=std::max(weightederr,std::abs(qnf::WeightedNullDifference(p2,u)-u.alpha.value*u.alpha.value*qnf::NullDifference(p2,u)));
 auto q=qnf::Gauge(p2,u,g,{.85,.95,5,true});auto old=hyp::InteriorLayerGauge(p2,u,g);auto ph=g;ph.physical_trace_lapse=true;ph.preferred_source=false;auto phy=hyp::InteriorLayerGauge(p2,u,ph);auto W=hyp::LayerCoefficients(p2.radius,u.alpha.value,g).weight;
 blenderr=std::max(blenderr,std::abs(q.pole.alpha-((1-W)*phy.pole.alpha+W*old.pole.alpha)));
 }
 // Two-sided continuity near both gauge and source cutoffs; fixed positive lapse.
 for(double rb:{.45,.85,.95})for(bool inner:{false,true}){double jump=0;const double h=1e-8;hyp::GaugeRHSParts<double>q[2];for(int j=0;j<2;++j){auto pp=ref.At(rb+(j?1:-1)*h,0,0);q[j]=qnf::Gauge(pp,Off(pp),g,{.85,.95,5,inner});}jump=std::max(std::abs(q[0].regular.alpha-q[1].regular.alpha),std::abs(q[0].pole.alpha-q[1].pole.alpha));for(int i=0;i<3;++i){jump=std::max(jump,std::abs(q[0].regular.beta[i]-q[1].regular.beta[i]));jump=std::max(jump,std::abs(q[0].pole.beta[i]-q[1].pole.beta[i]));}continuity=std::max(continuity,jump);}

 }
 std::cout<<std::setprecision(17)<<"{\"reference_and_blend_rows\":"<<rows<<",\"source_rows\":"<<source_rows<<",\"tiny_positive_core_rows\":"<<tiny_rows<<",\"reference_fixedpoint_error\":"<<referr<<",\"production_blend_equivalence_error\":"<<blenderr<<",\"factored_Q_pole_error\":"<<factorerr<<",\"factored_null_difference_error\":"<<nullerr<<",\"independent_4D_source_error\":"<<sourceerr<<",\"preferred_Box_extension_error\":"<<boxerr<<",\"exact_physical_inner_tiny_lapse_error\":"<<coreerr<<",\"weighted_null_factor_error\":"<<weightederr<<",\"tiny_noncore_rows\":"<<tiny_noncore_rows<<",\"tiny_noncore_max_source\":"<<tiny_noncore_max<<",\"cutoff_jump_at_h1e-8\":"<<continuity<<"}\n";
}
