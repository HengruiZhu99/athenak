#include "goal1_velocity_momentum.hpp"
#include "goal1_holonomic_momentum.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include "mesh/mesh.hpp"
#include "mesh/cubed_sphere/derivatives.hpp"
#include "mesh/cubed_sphere/interior_radial.hpp"
#include "mesh/cubed_sphere/research_radial_sbp.hpp"
namespace cubed_sphere {namespace hyp=z4c::hyperboloidal;using Z=z4c::Z4c;
#include "native_v2_loader.hpp"
}
namespace cs=cubed_sphere;namespace h=z4c::hyperboloidal;namespace hc=cs::holonomic_chart;namespace vm=cs::velocity_momentum;namespace ts=cs::typed_support;using State=h::Z4cJet<Real>;
#include "velocity_fixture_tools.hpp"
template<class T>std::array<T,55>Full(const h::Z4cJet<T>&u){auto basic=Functional(u);std::array<T,55>a;for(int n=0;n<37;++n)a[n]=basic[n];for(int d=0;d<3;++d)for(int n=0;n<6;++n)a[37+6*d+n]=u.metric.dg[d][ia[n]][ib[n]];return a;}
template<class T>h::Z4cJet<T>FullState(const std::array<T,55>&a){std::array<T,37>b;for(int n=0;n<37;++n)b[n]=a[n];auto u=StateFrom(b);for(int d=0;d<3;++d)for(int n=0;n<6;++n)u.metric.dg[d][ia[n]][ib[n]]=u.metric.dg[d][ib[n]][ia[n]]=a[37+6*d+n];return u;}
std::array<Real,10>Check(const State&u,const h::LayerPoint<Real>&ref,const Real*x,const std::string&kind,int index){
 static int serial=0;int ordinal=serial++;auto c=rwm::ReferenceConnection(ref,x);auto Fparts=vm::SourceParts(u,ref,c),parts=vm::MomentumParts(u,ref,c);std::array<Real,10>W,F,oldPi,oldF,qt;if(!vm::AssembleInterior(parts,ref.omega,W)||!vm::AssembleInterior(Fparts,ref.omega,F)||!cs::holonomic_momentum::Momentum(u,ref.state,ref.omega,oldPi)||!cs::holonomic_momentum::Source(u,ref,x,oldF)||!cs::holonomic_momentum::NativeVelocity(u,ref,x,qt))throw std::runtime_error("velocity value map failed");
 Real source=0,momentum=0,native=0,inverse=0;for(int n=0;n<10;++n){source=std::max(source,std::abs(F[n]-oldF[n])/std::max(Real(1),std::abs(oldF[n])));momentum=std::max(momentum,std::abs(W[n]-oldPi[n]-oldF[n])/std::max({Real(1),std::abs(W[n]),std::abs(oldPi[n]+oldF[n])}));Real target=qt[n];for(int d=0;d<3;++d){std::array<Real,10>g;vm::GradientDeviation(u,ref.state,d,g);target-=u.beta.value[d]*g[n];}native=std::max(native,std::abs(W[n]-target)/std::max(Real(1),std::abs(target)));}
 auto rebuilt=u;if(!vm::ReconstructInterior(rebuilt,ref,c,W))throw std::runtime_error("velocity value inverse failed");auto v0=Functional(u),v1=Functional(rebuilt);for(int n=11;n<22;++n)inverse=std::max(inverse,std::abs(v0[n]-v1[n])/std::max(Real(1),std::abs(v0[n])));
 auto values=Full(u);std::array<cs::Jet,55>input;for(int n=0;n<55;++n)input[n]=Seed(n+ordinal*3,values[n]);auto uj=FullState(input);auto pj=ts::Constant<cs::Jet>(ref);pj.omega=Seed(900+ordinal,ref.omega,.02*ref.omega);for(int d=0;d<3;++d)pj.domega[d]=Seed(1200+d+ordinal,ref.domega[d]);auto cj=ts::Constant<cs::Jet>(c);auto forward=vm::MomentumParts(uj,pj,cj);auto fs=vm::SourceParts(uj,pj,cj);std::array<cs::Jet,10>Wj,target;vm::AssembleInterior(forward,pj.omega,Wj);for(int n=0;n<10;++n)target[n]=Seed(1600+n+ordinal,W[n]);auto restored=uj;if(!vm::ReconstructInterior(restored,pj,cj,target))throw std::runtime_error("AD velocity inverse failed");std::array<cs::Jet,10>back;vm::AssembleInterior(vm::MomentumParts(restored,pj,cj),pj.omega,back);Real round=0,unchanged=0;for(int n=0;n<10;++n)round=std::max(round,Scaled(back[n],target[n]));auto restoredfull=Full(restored);for(int n=0;n<55;++n)if(n<11||n>21)unchanged=std::max(unchanged,Scaled(input[n],restoredfull[n]));auto geometry=h::Geometry(restored.metric);cs::Jet trace;for(int i=0;i<3;++i)for(int j=0;j<3;++j)trace+=geometry.inverse[i][j]*restored.a.k[i][j];
 std::cout<<"{\"kind\":\""<<kind<<"\",\"case\":"<<index<<",\"ordinal\":"<<ordinal<<",\"source_parts_value_scaled\":"<<source<<",\"canonical_Pi_plus_F_scaled\":"<<momentum<<",\"actual_native_velocity_scaled\":"<<native<<",\"native_value_inverse_scaled\":"<<inverse<<",\"AD_inverse_roundtrip_scaled\":"<<round<<",\"AD_tracefree_A_scaled\":"<<Scaled(trace,cs::Jet(0))<<",\"configuration_change_scaled\":"<<unchanged;
 if((kind=="synthetic"&&index%30==0)||(kind!="synthetic"&&index==0)){
  std::cout<<",\"input_u\":";Emit(input);std::cout<<",\"input_ref\":[";auto rf=Full(ref.state);for(int n=0;n<55;++n)std::cout<<(n?",":"")<<rf[n];std::cout<<"],\"input_Omega\":";Emit(pj.omega);std::cout<<",\"input_Omega_gradient\":[";for(int d=0;d<3;++d){if(d)std::cout<<",";Emit(pj.domega[d]);}std::cout<<"],\"connection\":[";for(int k=0;k<4;++k)for(int i=0;i<3;++i)for(int j=0;j<3;++j)std::cout<<(k||i||j?",":"")<<c.scaled[k][i][j];std::cout<<"],\"target_W\":";Emit(target);std::cout<<",\"forward_W\":";Emit(Wj);std::cout<<",\"source_regular\":";Emit(fs.regular);std::cout<<",\"source_pole\":";Emit(fs.pole);std::cout<<",\"inverse_p\":[";auto inv=Functional(restored);for(int n=11;n<22;++n){if(n>11)std::cout<<",";Emit(inv[n]);}std::cout<<"]";
 }std::cout<<"}"<<std::endl;return W;
}
void Audit(const State&u,const h::LayerPoint<Real>&p,const Real*x,const Real*,const std::string&kind,int index){Check(u,p,x,kind,index);}
#include "chart_native_cases.hpp"
#include "ad_synthetic_cases.hpp"
int main(int argc,char**argv){Kokkos::initialize(argc,argv);std::cout<<std::setprecision(17);int status=0;try{if(argc==1)Synthetic();else if(argc==4)NativeCase(argv[1],argv[2],argv[3]);else throw std::runtime_error("use noargs or saved checkpoint reference label");}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;status=2;}Kokkos::finalize();return status;}
