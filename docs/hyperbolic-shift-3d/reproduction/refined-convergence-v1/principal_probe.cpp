// Frozen-coefficient continuum diagnostic, not an evolution or well-posedness proof.
#include <mpi.h>
#include <iostream>
#include <iomanip>
#include "globals.hpp"
#include "refined_configuration.hpp"
#include "z4c/hyperboloidal/hyperbolic_composite.hpp"
#include "z4c/hyperboloidal/native_boosted_wormhole.hpp"
namespace h=z4c::hyperboloidal;
using V=std::vector<Real>;using D=std::vector<std::array<Real,3>>;
using DD=std::vector<std::array<std::array<Real,3>,3>>;
bool coordinate(int f){return f<=6||f==18||(f>=19&&f<25);}
V Rate(const Real*x,const V&v,const D&d,const DD&dd,const h::HyperbolicShiftParameters<Real>&p){
 const auto u=h::HyperbolicComposite::Jet(x,v,d,dd);auto o=p.profile.At(x);
 h::SetStationaryOmegaNormal(u.alpha.value,u.beta.value,u.alpha.d,u.beta.d,o);
 auto parts=h::ConformalRHS(u,o,Real(4),Real(-.99));h::Z4cRHS<Real> rate{};
 if(!h::HyperbolicDamping(x,u,o,parts,p)||!h::AssembleInterior(parts,o.omega,rate))throw std::runtime_error("invalid point RHS");
 V out(31);h::HyperbolicComposite::StoreGeometry(rate,out.data());
 const auto B=h::HyperbolicComposite::Vector(19,v,d),aux=h::HyperbolicComposite::Vector(25,v,d);
 const auto gauge=h::HyperbolicGauge(x,u,B,aux.value,p);
 if(!gauge.valid)throw std::runtime_error("invalid gauge");
 out[18]=gauge.lapse_regular+gauge.lapse_pole/o.omega;
 Real driver[3];if(!h::HyperbolicAuxiliaryRate(u,aux,rate.lambda,driver))throw std::runtime_error("invalid auxiliary rate");
 for(int a=0;a<3;++a){out[19+a]=gauge.flux_rate[a];out[22+a]=gauge.shift_regular[a]+gauge.shift_pole[a]/o.omega;out[25+a]=driver[a];}
 return out;
}
int main(int argc,char**argv){
 MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);
 MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);Kokkos::initialize(argc,argv);int code=0;
 try{
  if(argc!=3&&argc!=4&&argc!=5)throw std::invalid_argument("usage: principal_probe config fresh_jsonl [crossings|separated [amplitude]]");
  if(std::filesystem::exists(argv[2]))throw std::runtime_error("refusing overwrite");
  const Real amplitude=argc==5?std::stod(argv[4]):1;
  if(!(amplitude>0))throw std::invalid_argument("invalid perturbation amplitude");
  hyperbolic_reproduction::RefinedConfiguration cfg;
  cfg.Load(argv[1],hyperbolic_reproduction::ContinuousCoreMesh(16));
  h::NativeBoostedWormhole pgen(cfg.boost_velocity,1./3.,h::boost_free_initial_gauge,cfg.initial_start_M,cfg.initial_end_M);
  std::ofstream out(argv[2]);out<<std::setprecision(17);
  const bool separated=argc>=4&&std::string(argv[3])=="separated";
  for(int gcase=0;gcase<2;++gcase){
   std::vector<Real> radii;
   auto gap=[&](Real r){const Real x[3]={r,0,0};auto f=pgen.Fields(x);cfg.InitialFields(x,f);V v(f.begin(),f.end());D d(31);DD dd(31);auto p=cfg.gauge;if(gcase)p.inner_gamma=1;const auto u=h::HyperbolicComposite::Jet(x,v,d,dd);const auto c=h::ShiftCoefficients(r,u.alpha.value,u.chi.value,p);return h::GaugeShiftSpeedSquared(c,Real(1))-h::GaugeLapseSpeedSquared(u.alpha.value,u.chi.value,c,Real(1),Real(1),p);};
   if(argc==3||separated)for(int ir=0;ir<=360;++ir)radii.push_back(.05+.0025*ir);
   else {
    if(std::string(argv[3])!="crossings")throw std::invalid_argument("unknown mode");
    for(int ir=0;ir<360;++ir){Real lo=.05+.0025*ir,hi=lo+.0025,gl=gap(lo),gh=gap(hi);if(gl*gh>=0)continue;
     for(int it=0;it<50;++it){Real mid=(lo+hi)/2,gm=gap(mid);if(gm*gl>0){lo=mid;gl=gm;}else hi=mid;}
     const Real root=(lo+hi)/2;for(Real eps:{-1e-4,-1e-6,0.,1e-6,1e-4})radii.push_back(root+eps);
    }
   }
   for(Real r:radii){const Real x[3]={r,0,0};auto fields=pgen.Fields(x);cfg.InitialFields(x,fields);
   V v(fields.begin(),fields.end());D d(31);DD dd(31);auto parameters=cfg.gauge;
   if(gcase){if(separated)parameters.separated_shift_speeds=true;else parameters.inner_gamma=1;} // Explicit analysis-only coefficient control.
   const auto u=h::HyperbolicComposite::Jet(x,v,d,dd);const auto geometry=h::Geometry(u.metric);
   const auto c=h::ShiftCoefficients(r,u.alpha.value,u.chi.value,parameters);
   Real a[3][28][28]{};
   // Each retained principal coefficient is affine in its perturbed entry.
   // Symmetric unit perturbations avoid small-step cancellation; no numerical
   // differentiation of the native initial data or gauge coefficients is used.
   for(int order=0;order<3;++order)for(int j=0;j<28;++j){
    if(order==0&&coordinate(j))continue;
    if(order==2&&!coordinate(j))continue;
    auto plus=v,minus=v;auto dp=d,dm=d;auto hp=dd,hm=dd;
    if(order==0){plus[j]+=amplitude;minus[j]-=amplitude;}
    if(order==1){dp[j][0]+=amplitude;dm[j][0]-=amplitude;}
    if(order==2){hp[j][0][0]+=amplitude;hm[j][0][0]-=amplitude;}
    const auto fp=Rate(x,plus,dp,hp,parameters),fm=Rate(x,minus,dm,hm,parameters);
    for(int i=0;i<28;++i)a[order][i][j]=(fp[i]-fm[i])/(2*amplitude);
   }
   Real forbidden=0;
   for(int i=0;i<28;++i)for(int j=0;j<28;++j){
    if(coordinate(i)&&!coordinate(j))forbidden=std::max(forbidden,std::abs(a[1][i][j]));
    if(coordinate(i)||!coordinate(j))forbidden=std::max(forbidden,std::abs(a[2][i][j]));
   }
   if(forbidden>1e-9)throw std::runtime_error("unexpected differential-order coupling");
   out<<"{\"r\":"<<r<<",\"perturbation_amplitude\":"<<amplitude<<",\"inner_gamma\":"<<parameters.inner_gamma<<",\"override\":"<<gcase
      <<",\"effective_inner_gamma\":"<<c.inner_gamma<<",\"separated_shift_speeds\":"<<parameters.separated_shift_speeds
      <<",\"beta_x\":"<<u.beta.value[0]<<",\"alpha\":"<<u.alpha.value<<",\"chi\":"<<u.chi.value<<",\"inner_weight\":"<<c.inner
      <<",\"gap\":"<<h::AllDirectionShiftLapseGap(u.metric.g,u.chi.value,c,parameters)
      <<",\"lapse_speed2\":"<<h::GaugeLapseSpeedSquared(u.alpha.value,u.chi.value,c,geometry.inverse[0][0],Real(1),parameters)
      <<",\"shift_speed2\":"<<h::GaugeShiftSpeedSquared(c,geometry.inverse[0][0])
      <<",\"forbidden_order\":"<<forbidden<<",\"inverse_metric\":[";
   for(int i=0;i<3;++i)for(int j=0;j<3;++j){if(i||j)out<<',';out<<geometry.inverse[i][j];}
   out<<"],\"P\":[";
   for(int i=0;i<28;++i){if(i)out<<',';out<<'[';
    for(int j=0;j<28;++j){if(j)out<<',';
     const int order=coordinate(i)?(coordinate(j)?1:0):(coordinate(j)?2:1);out<<a[order][i][j];}
    out<<']';}
   out<<"]}\n";
  }
  }
  out.close();if(!out)throw std::runtime_error("probe write failed");
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';code=1;}
 Kokkos::finalize();MPI_Finalize();return code;
}
