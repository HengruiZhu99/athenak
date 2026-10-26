// Read-only chart/jet/velocity export and saved native fixture audit.
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "mesh/mesh.hpp"
#include "mesh/cubed_sphere/derivatives.hpp"
#include "z4c/hyperboloidal/cartesian_patch.hpp"
#include "mesh/cubed_sphere/interior_radial.hpp"
#include "mesh/cubed_sphere/research_radial_sbp.hpp"
#include "mesh/cubed_sphere/conformal_bridge.hpp"
#include "mesh/cubed_sphere/reference_wave_map.hpp"
#include "goal1_holonomic_chart.hpp"
namespace cubed_sphere {
namespace hyp=z4c::hyperboloidal;using Z=z4c::Z4c;
#include "native_v2_loader.hpp"
}
namespace cs=cubed_sphere;namespace h=z4c::hyperboloidal;namespace hc=cs::holonomic_chart;
using State=h::Z4cJet<Real>;
const int ia[6]={0,0,0,1,1,2},ib[6]={0,1,2,1,2,2};
void Emit(const cs::Jet&a){std::cout<<"["<<a.f;for(double v:a.d)std::cout<<","<<v;for(auto&r:a.h)for(double v:r)std::cout<<","<<v;std::cout<<"]";}
void Config(const hc::Config&q){std::cout<<"[";for(int i=0;i<10;++i){if(i)std::cout<<",";Emit(q[i]);}std::cout<<"]";}
void Metric(const State&u){std::cout<<"[";for(int i=0;i<6;++i){if(i)std::cout<<",";Emit(hc::Metric(u.metric,ia[i],ib[i]));}std::cout<<"]";}
void Audit(const State&u,const h::LayerPoint<Real>*ref,const Real*x,const std::string&kind,int index,const hc::Config*input=nullptr){
 hc::Config q;bool valid=hc::Encode(u,q);std::cout<<"{\"kind\":\""<<kind<<"\",\"case\":"<<index<<",\"valid\":"<<(valid?"true":"false");
 if(!valid){std::cout<<"}\n";return;}State back{};if(!hc::Decode(q,back))throw std::runtime_error("valid chart decode failed");
 double err[3]{};for(int n=0;n<6;++n){auto a=hc::Metric(u.metric,ia[n],ib[n]),b=hc::Metric(back.metric,ia[n],ib[n]);err[0]=std::max(err[0],std::abs(a.f-b.f)/std::max(1.,std::abs(a.f)));for(int i=0;i<3;++i){err[1]=std::max(err[1],std::abs(a.d[i]-b.d[i])/std::max(1.,std::abs(a.d[i])));for(int j=0;j<3;++j)err[2]=std::max(err[2],std::abs(a.h[i][j]-b.h[i][j])/std::max(1.,std::abs(a.h[i][j])));}}
 auto det=hc::Determinant(u.metric);double dg=0,dh=0;for(int i=0;i<3;++i){dg=std::max(dg,std::abs(det.d[i]));for(int j=0;j<3;++j)dh=std::max(dh,std::abs(det.h[i][j]));}
 h::Z4cRHS<Real> f{};h::GaugeRHS<Real> gauge{};bool native=ref&&ref->omega>0;
 if(native){if(!h::AssembleInterior(h::ConformalRHS(u,h::CartesianOmega(u,*ref),10/u.alpha.value,0.),ref->omega,f)||!rwm::Assemble(rwm::Gauge(*ref,u,rwm::ReferenceConnection(*ref,x)),ref->omega,gauge))throw std::runtime_error("actual native chart velocity failed");}
 else{f.chi=u.chi.d[0];gauge.alpha=u.alpha.d[0];for(int i=0;i<3;++i){gauge.beta[i]=u.beta.d[0][i];for(int j=0;j<3;++j)f.metric[i][j]=u.metric.dg[0][i][j];}}
 std::array<Real,10>qt;if(!hc::Velocity(u,f,gauge,qt))throw std::runtime_error("velocity map failed");hc::Config clock=q;for(int n=0;n<10;++n)for(int d=0;d<3;++d){clock[n].d[d]=d==0?qt[n]:0;for(int e=0;e<3;++e)clock[n].h[d][e]=0;}State reconstructed{};hc::Decode(clock,reconstructed);double terr=0;for(int i=0;i<3;++i)for(int j=0;j<3;++j)terr=std::max(terr,std::abs(reconstructed.metric.dg[0][i][j]-f.metric[i][j])/std::max(1.,std::abs(f.metric[i][j])));
 auto time=u.metric;for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int d=0;d<3;++d){time.dg[d][i][j]=d==0?f.metric[i][j]:0;for(int e=0;e<3;++e)time.ddg[d][e][i][j]=0;}
 std::cout<<",\"native_time_rhs\":"<<(native?"true":"false")<<",\"roundtrip_value\":"<<err[0]<<",\"roundtrip_first\":"<<err[1]<<",\"roundtrip_second\":"<<err[2]<<",\"det_value_error\":"<<std::abs(det.f-1)<<",\"det_first\":"<<dg<<",\"det_second\":"<<dh<<",\"time_roundtrip\":"<<terr<<",\"det_time\":"<<hc::Determinant(time).d[0]<<",\"metric\":";Metric(u);std::cout<<",\"metric_decoded\":";Metric(back);std::cout<<",\"chart\":";Config(q);
 std::cout<<",\"chart_input\":";if(input)Config(*input);else std::cout<<"null";std::cout<<",\"chart_velocity\":[";for(int i=0;i<10;++i)std::cout<<(i?",":"")<<qt[i];std::cout<<"],\"native_config_velocity\":["<<f.chi;for(int i=0;i<6;++i)std::cout<<","<<f.metric[ia[i]][ib[i]];std::cout<<","<<gauge.alpha;for(double v:gauge.beta)std::cout<<","<<v;std::cout<<"]}"<<std::endl;
}
void Audit(const State&u,const h::LayerPoint<Real>&ref,const Real*x,const Real*,const std::string&kind,int index){Audit(u,&ref,x,kind,index);}
// NativeCase is inserted verbatim except its documented stratified selection.
#include "chart_native_cases.hpp"
int main(int argc,char**argv){Kokkos::initialize(argc,argv);std::cout<<std::setprecision(17);int status=0;
try{if(argc==4)NativeCase(argv[1],argv[2],argv[3]);else if(argc==1){std::string mode;int index;while(std::cin>>mode>>index){State u{};u.chi.value=u.alpha.value=1;
 if(mode=="config"){hc::Config q;for(auto&a:q){std::cin>>a.f;for(auto&v:a.d)std::cin>>v;for(auto&r:a.h)for(auto&v:r)std::cin>>v;}if(!hc::Decode(q,u))throw std::runtime_error("input config invalid");Audit(u,nullptr,nullptr,"config",index,&q);}
 else if(mode=="metric"){for(int n=0;n<6;++n){int i=ia[n],j=ib[n];std::cin>>u.metric.g[i][j];u.metric.g[j][i]=u.metric.g[i][j];for(int d=0;d<3;++d){std::cin>>u.metric.dg[d][i][j];u.metric.dg[d][j][i]=u.metric.dg[d][i][j];}for(int d=0;d<3;++d)for(int e=0;e<3;++e){std::cin>>u.metric.ddg[d][e][i][j];u.metric.ddg[d][e][j][i]=u.metric.ddg[d][e][i][j];}}Audit(u,nullptr,nullptr,"metric-input",index);}
 else if(mode=="invalid"){int kind;std::cin>>kind;for(int i=0;i<3;++i)u.metric.g[i][i]=1;if(kind==0)u.metric.g[0][0]=-1;if(kind==1)u.metric.g[1][1]=-1;if(kind==2)u.metric.g[2][2]=1.1;if(kind==3)u.chi.value=0;if(kind==4)u.alpha.value=-1;Audit(u,nullptr,nullptr,"invalid-value",index);}
 else if(mode=="reference"){Real x[3];for(auto&v:x)std::cin>>v;h::LayerReference<Real>ref(1.,.5,{true,.05,.95});auto p=ref.At(x[0],x[1],x[2]);Audit(p.state,&p,x,"reference",index);}
 else throw std::runtime_error("unknown chart command");}}
else throw std::runtime_error("use stdin or checkpoint reference label");}
catch(const std::exception&e){std::cerr<<e.what()<<std::endl;status=2;}Kokkos::finalize();return status;}
