#include "goal1_native_ad.hpp"
#include "goal1_native_reduced_source.hpp"
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
 static int ordinal=0;h::LayerReference<Real>descriptor(1.,.5,{true,.05,.95});hc::Config q,q0;if(!hc::Encode(u,q)||!hc::Encode(p.state,q0))throw std::runtime_error("invalid native reduced fixture");reduced::Vector qbar{},P{},pi{};reduced::Gradient g{},DP{};reduced::Hessian Dg{},curl{};
 hm::Momentum(u,p.state,p.omega,pi);for(int n=0;n<10;++n){qbar[n]=q[n].f-q0[n].f;P[n]=pi[n]+qbar[n];for(int i=0;i<3;++i){g[i][n]=q[n].d[i]-q0[n].d[i];DP[i][n]=.001*sin((ordinal+1)*(n+1)*(.13+.07*i));for(int j=i;j<3;++j)Dg[i][j][n]=Dg[j][i][n]=.002*cos((ordinal+1)*(n+1)*(.11+.03*(i+1)*(j+1)));}}
 curl=Dg;for(int n=0;n<10;++n)for(int i=0;i<3;++i)for(int j=i+1;j<3;++j){Real a=.003*sin((ordinal+1)*(n+1)*(i+j+.3));curl[i][j][n]+=a;curl[j][i][n]-=a;}
 auto source=reduced::Source(descriptor,p,x,qbar,g,P);auto actual=reduced::Evaluate(descriptor,p,x,qbar,g,P,Dg,DP);auto incompatible=reduced::Evaluate(descriptor,p,x,qbar,g,P,curl,DP);if(!source.valid||!actual.valid||!incompatible.valid)throw std::runtime_error("full native reduced source failed");
 auto geo=h::Geometry(actual.native.metric);Real a[3][3];for(int i=0;i<3;++i)for(int j=0;j<3;++j)a[i][j]=actual.native.alpha.value*actual.native.alpha.value*actual.native.chi.value*geo.inverse[i][j];
 Real compatible_error=0,raw_curl=0,corrected_curl=0,qerror=0,gerror=0;Real Ae[3][10][10]{};
 for(int i=0;i<3;++i)for(int col=0;col<10;++col){if(col>=1&&col<=5)continue;auto var=cs::typed_support::Map<cs::Jet>(actual.native,[](int,Real v){return cs::Jet(v);});if(col==0)var.chi.d[i].d[0]=1;else if(col==6)var.alpha.d[i].d[0]=1;else var.beta.d[i][col-7].d[0]=1;auto ref=cs::typed_support::Constant<cs::Jet>(p);std::array<cs::Jet,10>mapped;if(!cs::typed_momentum::Momentum(var,ref.state,cs::Jet(p.omega),mapped))throw std::runtime_error("nonlinear native Ai map failed");for(int row=0;row<10;++row)Ae[i][row][col]=mapped[row].d[0];}
 for(int n=0;n<10;++n){Real expected=source.P[n],expectedCurl=source.P[n];for(int i=0;i<3;++i){expected+=actual.native.beta.value[i]*DP[i][n];expectedCurl+=actual.native.beta.value[i]*DP[i][n];for(int j=0;j<3;++j){expected+=a[i][j]*Dg[i][j][n];expectedCurl+=a[i][j]*curl[i][j][n];}}
  Real correction=0;for(int i=0;i<3;++i)for(int j=i+1;j<3;++j)for(int col=0;col<10;++col){Real comm=0;for(int k=0;k<10;++k)comm+=Ae[i][n][k]*Ae[j][k][col]-Ae[j][n][k]*Ae[i][k][col];correction+=.5*comm*(curl[i][j][col]-curl[j][i][col]);}
  compatible_error=std::max(compatible_error,std::abs(actual.Pdot[n]-expected)/std::max({Real(1),std::abs(actual.Pdot[n]),std::abs(expected)}));raw_curl=std::max(raw_curl,std::abs(incompatible.Pdot[n]-expectedCurl));corrected_curl=std::max(corrected_curl,std::abs(incompatible.Pdot[n]-expectedCurl-correction)/std::max({Real(1),std::abs(incompatible.Pdot[n]),std::abs(expectedCurl)}));qerror=std::max(qerror,std::abs(actual.qdot[n]-source.q[n]));
  for(int i=0;i<3;++i){Real gradient=source.g[i][n]+DP[i][n];for(int j=0;j<3;++j)gradient+=actual.native.beta.value[j]*Dg[j][i][n];gerror=std::max(gerror,std::abs(actual.gdot[i][n]-gradient));}
 }
 std::cout<<"{\"kind\":\""<<kind<<"\",\"case\":"<<index<<",\"ordinal\":"<<ordinal++<<",\"Omega\":"<<p.omega<<",\"nonlinear_native_P_source_scaled\":"<<compatible_error<<",\"raw_incompatible_curl_defect_absolute\":"<<raw_curl<<",\"explicit_curl_corrected_scaled\":"<<corrected_curl<<",\"q_principal_independence_absolute\":"<<qerror<<",\"g_source_principal_identity_absolute\":"<<gerror<<",\"native_det_time_absolute\":"<<std::abs(actual.det_time)<<",\"native_A_trace_time_absolute\":"<<std::abs(actual.A_trace_time)<<"}"<<std::endl;return source.P;
}
void Audit(const State&u,const h::LayerPoint<Real>&p,const Real*x,const Real*,const std::string&kind,int index){Check(u,p,x,kind,index);}
#include "chart_native_cases.hpp"
#include "ad_synthetic_cases.hpp"
int main(int argc,char**argv){Kokkos::initialize(argc,argv);std::cout<<std::setprecision(17);int status=0;try{if(argc==1)Synthetic();else if(argc==4)NativeCase(argv[1],argv[2],argv[3]);else throw std::runtime_error("use noargs or checkpoint reference label");}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;status=2;}Kokkos::finalize();return status;}
