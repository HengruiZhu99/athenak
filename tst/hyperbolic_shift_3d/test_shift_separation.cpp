// Nontrivial source-coefficient and all-direction speed-gap control.
#include <mpi.h>
#include <iostream>
#include <iomanip>
#include "globals.hpp"
#include "refined_configuration.hpp"
namespace h=z4c::hyperboloidal;
void Require(bool q,const char*m){if(!q)throw std::runtime_error(m);}
KOKKOS_INLINE_FUNCTION
void Evaluate(int index,h::HyperbolicShiftParameters<Real> p,Real*out){
 const Real r=Real(index%101)/100;
 const Real alpha=pow(Real(10),Real((index/101)%7)-4);
 const Real chi=pow(Real(10),Real((index/707)%9)-6);
 const Real x[3]={r,0,0};h::Z4cJet<Real> u{};h::VectorJet<Real>B{};Real D[3];
 u.alpha.value=alpha;u.chi.value=chi;u.trace.value=.07;
 for(int i=0;i<3;++i){u.metric.g[i][i]=1;u.beta.value[i]=.02*(i+1);u.lambda.value[i]=.13*(i+1);D[i]=.04*(i+1);B.value[i]=-.03*(i+1);u.alpha.d[i]=.05*(i+1);
  for(int j=0;j<3;++j){u.beta.d[i][j]=.01*(i+j+1);B.d[i][j]=.03*(2*i-j);}}
 auto base=p;base.separated_shift_speeds=false;
 const auto before=h::HyperbolicGauge(x,u,B,D,base),after=h::HyperbolicGauge(x,u,B,D,p);
 const auto c=h::ShiftCoefficients(r,alpha,chi,p);const Real s=c.inner,o=c.omega;
 const Real Q=1.7,guard=fmax(o*o*chi,Real(h::InnerTelegrapher::chi_guard));
 const Real formula=Q*(Real(4./3.)*s*p.inner_gamma+(1-s)*chi*c.F+s*(1-s)*guard*Real(h::InnerTelegrapher::kappa/h::InnerTelegrapher::tau));
 // Normalize roundoff by the subtracted operands, including near equal speeds.
 const Real gap=h::GaugeShiftSpeedSquared(c,Q)-h::GaugeLapseSpeedSquared(alpha,chi,c,Q,Real(1),p);
 out[0]=gap;out[1]=fabs(gap-formula)/(1+fabs(h::GaugeShiftSpeedSquared(c,Q))+fabs(h::GaugeLapseSpeedSquared(alpha,chi,c,Q,Real(1),p)));out[2]=0;out[3]=0;
 for(int i=0;i<3;++i){
  const Real expected=s*Real(.75)*(2*o*alpha*chi+guard*Real(h::InnerTelegrapher::kappa/h::InnerTelegrapher::tau))*u.lambda.value[i];
  out[2]=fmax(out[2],fabs(after.shift_regular[i]-before.shift_regular[i]-expected)/(1+fabs(expected)+fabs(after.shift_regular[i])+fabs(before.shift_regular[i])));
  out[3]=fmax(out[3],fabs(after.flux_rate[i]-before.flux_rate[i]));
  out[3]=fmax(out[3],fabs(after.shift_pole[i]-before.shift_pole[i]));
  if(s==0)out[3]=fmax(out[3],fabs(after.shift_regular[i]-before.shift_regular[i]));
 }
 out[3]=fmax(out[3],fabs(after.lapse_regular-before.lapse_regular));out[3]=fmax(out[3],fabs(after.lapse_pole-before.lapse_pole));
 out[4]=(before.valid&&after.valid)?1:0;
}
int main(int argc,char**argv){MPI_Init(&argc,&argv);Kokkos::initialize(argc,argv);int code=0;
 try{
  Require(argc==2,"supply separated configuration");hyperbolic_reproduction::RefinedConfiguration cfg;cfg.Load(argv[1],hyperbolic_reproduction::ContinuousCoreMesh(12));
  const auto p=cfg.gauge;Require(p.separated_shift_speeds,"control disabled");
  const int count=101*7*9;Kokkos::View<Real*[5]> got("gap oracle",count);
  Kokkos::parallel_for("separated shift oracle",count,KOKKOS_LAMBDA(int i){Real out[5];Evaluate(i,p,out);for(int j=0;j<5;++j)got(i,j)=out[j];});
  auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),got);Real gap_min=1e99,gap_error=0,source_error=0,unrelated=0,parity=0;
  for(int i=0;i<count;++i){Real cpu[5];Evaluate(i,p,cpu);for(int j=0;j<5;++j)parity=std::max(parity,std::abs(cpu[j]-host(i,j))/(1+std::abs(cpu[j])));
   Require(cpu[4]==1,"manufactured RHS invalid");gap_min=std::min(gap_min,cpu[0]);gap_error=std::max(gap_error,cpu[1]);source_error=std::max(source_error,cpu[2]);unrelated=std::max(unrelated,cpu[3]);}
  std::cerr<<std::setprecision(17)<<"gap_min "<<gap_min<<" gap_error "<<gap_error<<" source_error "<<source_error<<" unaffected "<<unrelated<<" parity "<<parity<<"\n";
  Require(gap_min>0&&gap_error<2e-13,"positive gap identity failed");Require(source_error<2e-13&&unrelated==0,"RHS coefficient or unaffected outer equation changed");Require(parity<2e-13,"CPU/device mismatch");
  auto bad=p;bad.ratio=1;bool rejected=false;try{bad.Validate();}catch(const std::invalid_argument&){rejected=true;}Require(rejected,"unsupported outer ratio admitted");
  std::cout<<std::setprecision(17)<<"{\"passed\":true,\"backend\":\""<<Kokkos::DefaultExecutionSpace::name()<<"\",\"cases\":"<<count<<",\"min_gap\":"<<gap_min<<",\"gap_identity_scaled_error\":"<<gap_error<<",\"actual_rhs_scaled_error\":"<<source_error<<",\"unaffected_max_change\":"<<unrelated<<",\"cpu_device_scaled_error\":"<<parity<<"}\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';code=1;}
 Kokkos::finalize();MPI_Finalize();return code;
}
