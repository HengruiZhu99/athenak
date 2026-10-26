#include "goal1_native_ad.hpp"
#include "goal1_native_reduced_audit.hpp"
#include "goal1_holonomic_momentum.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include "mesh/mesh.hpp"
#include "mesh/cubed_sphere/derivatives.hpp"
#include "mesh/cubed_sphere/interior_radial.hpp"
#include "mesh/cubed_sphere/research_radial_sbp.hpp"
namespace cubed_sphere {namespace hyp=z4c::hyperboloidal;using Z=z4c::Z4c;
#include "native_v2_loader.hpp"
}
namespace cs=cubed_sphere;namespace h=z4c::hyperboloidal;namespace hc=cs::holonomic_chart;namespace hm=cs::holonomic_momentum;namespace reduced=cs::native_reduced_source;using State=h::Z4cJet<Real>;
std::array<Real,10>Check(const State&u,const h::LayerPoint<Real>&p,const Real*x,const std::string&kind,int index){
 static int next=0;int ordinal=next++;if(kind!="synthetic"&&index%23)return {};h::LayerReference<Real>descriptor(1.,.5,{true,.05,.95});hc::Config q,q0;hc::Encode(u,q);hc::Encode(p.state,q0);reduced::Vector qbar{},P{},pi{};reduced::Gradient g{},DP{};reduced::Hessian Dg{};hm::Momentum(u,p.state,p.omega,pi);
 for(int n=0;n<10;++n){qbar[n]=q[n].f-q0[n].f;P[n]=pi[n]+qbar[n];for(int i=0;i<3;++i){g[i][n]=q[n].d[i]-q0[n].d[i];DP[i][n]=.001*sin((ordinal+1)*(n+1)*(.13+.07*i));for(int j=i;j<3;++j)Dg[i][j][n]=Dg[j][i][n]=.002*cos((ordinal+1)*(n+1)*(.11+.03*(i+1)*(j+1)));}}
 auto source=reduced::Source(descriptor,p,x,qbar,g,P);auto e=reduced::Evaluate(descriptor,p,x,qbar,g,P,Dg,DP);auto native=cs::native_reduced_audit::Direct(descriptor,p,x,e);if(!source.valid||!e.valid||!native.valid)throw std::runtime_error("unforced native time chain failed");auto geometry=h::Geometry(e.native.metric);Real error=0,chain=0;
 for(int n=0;n<10;++n){Real expected=source.P[n];for(int i=0;i<3;++i){expected+=e.native.beta.value[i]*DP[i][n];for(int j=0;j<3;++j)expected+=e.native.alpha.value*e.native.alpha.value*e.native.chi.value*geometry.inverse[i][j]*Dg[i][j][n];}error=std::max(error,std::abs(native.Pdot[n]-expected)/std::max({Real(1),std::abs(native.Pdot[n]),std::abs(expected)}));chain=std::max(chain,std::abs(native.Pdot[n]-e.Pdot[n])/std::max({Real(1),std::abs(native.Pdot[n]),std::abs(e.Pdot[n])}));}
 std::cout<<"{\"kind\":\""<<kind<<"\",\"case\":"<<index<<",\"ordinal\":"<<ordinal<<",\"native_q_spatial_gradient_scaled\":"<<native.q_gradient_scaled<<",\"native_q_value_scaled\":"<<native.q_value_scaled<<",\"native_P_independent_time_chain_scaled\":"<<chain<<",\"native_P_full_source_equivalence_scaled\":"<<error<<"}"<<std::endl;return source.P;
}
void Audit(const State&u,const h::LayerPoint<Real>&p,const Real*x,const Real*,const std::string&kind,int index){Check(u,p,x,kind,index);}
#include "chart_native_cases.hpp"
#include "ad_synthetic_cases.hpp"
int main(int argc,char**argv){Kokkos::initialize(argc,argv);std::cout<<std::setprecision(17);int status=0;try{if(argc==1)Synthetic();else if(argc==4)NativeCase(argv[1],argv[2],argv[3]);else throw std::runtime_error("use noargs or checkpoint reference label");}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;status=2;}Kokkos::finalize();return status;}
