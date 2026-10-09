#include "audit_helpers.hpp"
#include "four_gamma.hpp"
hyp::Z4cJet<double> Values(const Jet&u){hyp::Z4cJet<double>v{};auto s=[](auto&a,const auto&b){a.value=b.value.v;for(int i=0;i<3;++i){a.d[i]=b.d[i].v;for(int j=0;j<3;++j)a.dd[i][j]=b.dd[i][j].v;}};
 s(v.alpha,u.alpha);s(v.chi,u.chi);s(v.trace,u.trace);s(v.theta,u.theta);
 for(int i=0;i<3;++i){v.beta.value[i]=u.beta.value[i].v;v.lambda.value[i]=u.lambda.value[i].v;for(int j=0;j<3;++j){v.beta.d[j][i]=u.beta.d[j][i].v;v.lambda.d[j][i]=u.lambda.d[j][i].v;v.metric.g[i][j]=u.metric.g[i][j].v;v.a.k[i][j]=u.a.k[i][j].v;for(int k=0;k<3;++k){v.beta.dd[k][j][i]=u.beta.dd[k][j][i].v;v.lambda.dd[k][j][i]=u.lambda.dd[k][j][i].v;v.metric.dg[k][i][j]=u.metric.dg[k][i][j].v;v.a.dk[k][i][j]=u.a.dk[k][i][j].v;for(int l=0;l<3;++l)v.metric.ddg[k][l][i][j]=u.metric.ddg[k][l][i][j].v;}}}return v;}
hyp::Z4cJet<double> Off(const hyp::LayerPoint<double>&p){auto u=Lift(p.state);
 for(int c=0;c<20;++c){J x(D(.006*std::sin(c+1.)));for(int i=0;i<3;++i){x.d[i]=.003*std::cos((i+1.)*(c+1.));for(int j=0;j<3;++j)x.dd[i][j]=.002*std::sin((i+j+2.)*(c+1.));}Seed(u,c,x);}Consistent(u);return Values(u);}
double Maximum(const hyp::GaugeRHSParts<double>&q){double x=std::max(std::abs(q.regular.alpha),std::abs(q.pole.alpha));for(int i=0;i<3;++i){x=std::max(x,std::abs(q.regular.beta[i]));x=std::max(x,std::abs(q.pole.beta[i]));}return x;}
int main(){hyp::LayerGaugeParameters g;double referr=0,factorerr=0,nullerr=0,sourceerr=0,boxerr=0,blenderr=0,coreerr=0;int rows=0,source_rows=0,tiny_rows=0;
 for(double a:{.5,.75,1.,2.}){hyp::LayerReference<double>ref(1,a,{true,.05,.95});
 for(double r:{.01,.1,.2,.44,.45,.450001,.5,.65,.8,.849999,.85,.850001,.86,.9,.94,.949999,.95,.950001,.98,.9999})for(const auto&n:{std::array<double,3>{1,0,0},std::array<double,3>{.36,-.48,.8}}){auto p=ref.At(r*n[0],r*n[1],r*n[2]);auto u=Off(p);const auto c=hyp::LayerCoefficients(p.radius,u.alpha.value,g);
 for(bool inner:{false,true}){const auto qr=qnf::Gauge(p,p.state,g,{.85,.95,5,inner});hyp::GaugeRHS<double>rr{};if(!qnf::Assemble(qr,p.omega,rr))throw std::runtime_error("reference invalid");referr=std::max(referr,std::abs(rr.alpha));for(int i=0;i<3;++i)referr=std::max(referr,std::abs(rr.beta[i]));
 auto q=qnf::Gauge(p,u,g,{.85,.95,5,inner});if(!q.valid)throw std::runtime_error("offconstraint invalid");auto b=hyp::InteriorLayerGauge(p,u,g);auto pg=g;pg.physical_trace_lapse=true;pg.preferred_source=false;auto ph=hyp::InteriorLayerGauge(p,u,pg);const double W=c.weight;
 blenderr=std::max(blenderror,std::abs(q.regular.alpha-(inner?(1-W)*ph.regular.alpha+W*b.regular.alpha:b.regular.alpha)));
 blenderr=std::max(blenderror,std::abs(q.pole.alpha-(inner?(1-W)*ph.pole.alpha+W*b.pole.alpha:b.pole.alpha)));
 ++rows;}
 double Dm=0;for(int i=0;i<3;++i)Dm+=(p.beta[i]*(u.alpha.value-p.alpha)/p.alpha-(u.beta.value[i]-p.beta[i]))*p.domega[i];
 const auto qb=qnf::FactoredBaseGauge(p,u,g);factorerr=std::max(factorerr,std::abs(qb.pole.alpha-(-c.alpha2f*(u.trace.value-p.k_physical)+3*(u.alpha.value+2*(1-c.weight))*Dm)));
 const auto ol=hyp::CartesianOmega(u,p),oh=hyp::CartesianOmega(p.state,p);const auto gl=hyp::Geometry(u.metric),gh=hyp::Geometry(p.state.metric);double G=0,Gh=0;for(int i=0;i<3;++i)for(int j=0;j<3;++j){G+=u.chi.value*gl.inverse[i][j]*p.domega[i]*p.domega[j];Gh+=p.state.chi.value*gh.inverse[i][j]*p.domega[i]*p.domega[j];}
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
 }
 std::cout<<std::setprecision(17)<<"{\"reference_and_blend_rows\":"<<rows<<",\"source_rows\":"<<source_rows<<",\"tiny_positive_core_rows\":"<<tiny_rows<<",\"reference_fixedpoint_error\":"<<referr<<",\"production_blend_equivalence_error\":"<<blenderror<<",\"factored_Q_pole_error\":"<<factorerr<<",\"factored_null_difference_error\":"<<nullerr<<",\"independent_4D_source_error\":"<<sourceerr<<",\"preferred_Box_extension_error\":"<<boxerr<<",\"exact_physical_inner_tiny_lapse_error\":"<<coreerr<<"}\n";
}
