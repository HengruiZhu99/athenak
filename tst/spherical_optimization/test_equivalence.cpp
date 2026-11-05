#include "athena.hpp"
#include "z4c/hyperboloidal/current_spherical.hpp"
#include "z4c/hyperboloidal/central_z4c_configuration.hpp"
#include "z4c/hyperboloidal/stationary_gauge_reference.hpp"
#include <iostream>
#include <iomanip>
namespace h=z4c::hyperboloidal;namespace s=h::current_spherical;
void check(bool b,const char*m){if(!b)throw std::runtime_error(m);}
double worst=0;
void near(double a,double b){double e=std::abs(a-b)/(1+std::max(std::abs(a),std::abs(b)));worst=std::max(worst,e);check(e<2e-12,"spherical/Cartesian RHS mismatch");}
// Independent Cartesian polynomial jets: scalar a+b*x.x, vector (a+b*x.x)x,
// tensor (a+b*x.x)delta + c*x*x. No Radial* construction on this path.
h::ScalarJet<double> scalar(double r,double a,double b){h::ScalarJet<double>v{};v.value=a+b*r*r;v.d[0]=2*b*r;for(int i=0;i<3;++i)v.dd[i][i]=2*b;return v;}
h::VectorJet<double> vector(double r,double a,double b){h::VectorJet<double>v{};double x[3]={r,0,0};for(int i=0;i<3;++i){v.value[i]=(a+b*r*r)*x[i];for(int d=0;d<3;++d){v.d[d][i]=2*b*x[d]*x[i]+(i==d?a+b*r*r:0);for(int e=0;e<3;++e)v.dd[d][e][i]=2*b*((d==e?x[i]:0)+(d==i?x[e]:0)+(e==i?x[d]:0));}}return v;}
h::MetricJet<double> tensor(double r,double a,double b,double c){h::MetricJet<double>v{};double x[3]={r,0,0};for(int i=0;i<3;++i)for(int j=0;j<3;++j){v.g[i][j]=(i==j?a+b*r*r:0)+c*x[i]*x[j];for(int d=0;d<3;++d){v.dg[d][i][j]=(i==j?2*b*x[d]:0)+c*((i==d?x[j]:0)+(j==d?x[i]:0));for(int e=0;e<3;++e)v.ddg[d][e][i][j]=(i==j&&d==e?2*b:0)+c*((i==d&&j==e)+(j==d&&i==e));}}return v;}
void compare(const h::Z4cRHS<double>&a,const h::Z4cRHS<double>&b){near(a.chi,b.chi);near(a.trace,b.trace);near(a.theta,b.theta);for(int i=0;i<3;++i){near(a.lambda[i],b.lambda[i]);for(int j=0;j<3;++j){near(a.metric[i][j],b.metric[i][j]);near(a.a[i][j],b.a[i][j]);}}}
int main(){try{
 h::CentralZ4cConfiguration c;const auto config=c.Resolve();auto p=config.gauge;h::StationaryGaugeReference reference(p);
 int cases=0;
 for(int i=1;i<100;++i){double r=i/100.,x[3]={r,0,0};s::Jets exact{};auto&u=exact.u;
  u.chi=scalar(r,.81,.07);u.alpha=scalar(r,.63,.16);u.trace=scalar(r,-.7,.02);u.theta=scalar(r,.023,.011);
  u.metric=tensor(r,1.07,.021,.033);const auto A=tensor(r,.012,.013,-.024);
  for(int a=0;a<3;++a)for(int b=0;b<3;++b){u.a.k[a][b]=A.g[a][b];for(int d=0;d<3;++d)u.a.dk[d][a][b]=A.dg[d][a][b];}
  u.lambda=vector(r,.12,.04);u.beta=vector(r,-.14,.03);exact.B=vector(r,.035,.012);exact.D=vector(r,.11,-.027);
  s::State q{},d{},dd{};
  auto scalar_slot=[&](int f,const auto&v){q[f]=v.value;d[f]=v.d[0];dd[f]=v.dd[0][0];};
  auto vector_slot=[&](int f,const auto&v){q[f]=v.value[0];d[f]=v.d[0][0];dd[f]=v.dd[0][0][0];};
  scalar_slot(s::chi,u.chi);scalar_slot(s::alpha,u.alpha);scalar_slot(s::khat,u.trace);scalar_slot(s::theta,u.theta);
  vector_slot(s::lambda,u.lambda);vector_slot(s::beta,u.beta);vector_slot(s::flux,exact.B);vector_slot(s::driver,exact.D);
  for(int a=0;a<2;++a){q[s::gr+a]=u.metric.g[a][a];d[s::gr+a]=u.metric.dg[0][a][a];dd[s::gr+a]=u.metric.ddg[0][0][a][a];
   q[s::ar+a]=A.g[a][a];d[s::ar+a]=A.dg[0][a][a];dd[s::ar+a]=A.ddg[0][0][a][a];}
  const auto lifted=s::Lift(r,q.data(),d.data(),dd.data());
  for(int separated:{0,1}){p.separated_shift_speeds=separated;
   const auto source=reference.At(r);const auto a=s::Evaluate(x,exact,p,source),b=s::Evaluate(x,lifted,p,source);
   compare(a.geometry.regular,b.geometry.regular);compare(a.geometry.pole,b.geometry.pole);
   for(int f=0;f<12;++f){near(a.regular[f],b.regular[f]);near(a.pole[f],b.pole[f]);}
   near(a.regular[s::driver]-a.regular[s::lambda],u.beta.value[0]*(exact.D.d[0][0]-u.lambda.d[0][0])-p.damping*exact.D.value[0]);
   check(std::abs(u.lambda.value[0]-h::Geometry(u.metric).contracted[0])>1e-5,"fixture accidentally on constraint");++cases;
  }
 }
 // Tangency of every continuum radial/tangential equation on the algebraic
 // manifold, with nonzero off-constraint Lambda/Theta and arbitrary jets.
 double tangent_error=0;
 for(int i=1;i<100;++i){const double r=i/100.,x[3]={r,0,0};
  double q[10]={.8+.1*r,1+.04*r*r,.02+.01*r,-.2,.013,.09*r,.6+.03*r,-.1*r,.03*r,.07*r};
  double d[10]={.1,.08*r,.01,.02,.003,.09,.03,-.1,.03,.07},dd[10]{};dd[1]=.08;
  s::State Q{},D{},DD{};s::ExpandChart(q,d,dd,Q,D,DD);
  const auto j=s::Lift(r,Q.data(),D.data(),DD.data());const auto v=s::Evaluate(x,j,p,reference.At(r));
  s::State rate{};for(int f=0;f<12;++f)rate[f]=v.regular[f]+v.pole[f]/v.omega.omega;
  const double expected_g=-.5*Q[s::gt]/Q[s::gr]*rate[s::gr];
  const double expected_a=-.5*Q[s::gt]/Q[s::gr]*rate[s::ar]+.75*Q[s::ar]*Q[s::gt]/(Q[s::gr]*Q[s::gr])*rate[s::gr];
  near(rate[s::gt],expected_g);near(rate[s::at],expected_a);
  tangent_error=std::max({tangent_error,std::abs(rate[s::gt]-expected_g),std::abs(rate[s::at]-expected_a)});
 }
 // Single-radius plateau, physical mapping, and explicit bad-input rejection.
 for(double R:{6.,10.,20.,30.}){c.inner_z4c_radius_M=R;const auto z=c.Resolve();
  h::NativeWormhole map(1./3.,2e-13,R,2*R,h::NativeWormhole::TransitionProfile::polynomial_c6);
  near(map.Point(z.compact_radius).areal*3,R);
  for(double f:{.01,.5,1.}){const auto om=z.gauge.profile.Radial(f*z.compact_radius);check(om.first==0&&om.second==0,"nonflat core");check(h::InnerTelegrapherWeight(f*z.compact_radius,z.gauge.inner_begin,z.gauge.inner_end)==1,"non-Cartesian inner gauge");}
 }
 for(int bad=0;bad<4;++bad){h::CentralZ4cConfiguration invalid;if(bad==0)invalid.inner_z4c_radius_M=2;if(bad==1)invalid.slicing_end_ratio=1;if(bad==2)invalid.omega_fraction=.8;if(bad==3)invalid.plateau=.01;bool rejected=false;try{invalid.Resolve();}catch(const std::invalid_argument&){rejected=true;}check(rejected,"invalid central configuration accepted");}
 // Nonstationary/nonzero-gradient scri fixture: differentiate nullness using
 // the actual geometric RHS, and distinguish it from the bulk auxiliary PDE.
 double r=1,x[3]={1,0,0};s::State q{1.08,1.04,1/std::sqrt(1.04),.02,-.01,-3,.001,.03,1,-std::sqrt(1.08/1.04),.02,.13};
 s::State d{},dd{},normal{};for(int f=0;f<12;++f){d[f]=.003*(f+1);dd[f]=-.001*(f+1);normal[f]=.005*(f-3);}
 const auto j=s::Lift(r,q.data(),d.data(),dd.data());const auto v=s::Evaluate(x,j,p,reference.At(1));double D,clock;
 const auto rate=s::ScriRate(j,v,normal,.01,.6510416666660156,p,D,clock);
 near(2*q[s::beta]*rate[s::beta],rate[s::chi]/q[s::gr]-q[s::chi]*rate[s::gr]/(q[s::gr]*q[s::gr]));
 const double mu=h::ShiftCoefficients(1.,1.,q[s::chi],p).mu;
 near(rate[s::beta],-1+q[s::beta]*d[s::beta]+p.restoring*(d[s::beta]+1)+mu*D);
 check(std::abs(rate[s::chi])+std::abs(rate[s::gr])>1e-3,"stationary boundary fixture");
 std::cout<<std::setprecision(17)<<"{\"passed\":true,\"off_constraint_cases\":"<<cases<<",\"max_scaled_error\":"<<worst<<",\"tangent_error\":"<<tangent_error<<",\"central20_compact\":"<<config.compact_radius<<",\"fits_legacy_box_half_04\":"<<(config.FitsCartesianBox(.4)?"true":"false")<<",\"differential_auxiliary_boundary_proven\":false}\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;} }
