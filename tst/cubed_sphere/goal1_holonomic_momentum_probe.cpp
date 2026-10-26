// Actual native nonlinear q rows and momentum inversion, zero evolution.
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "mesh/mesh.hpp"
#include "mesh/cubed_sphere/derivatives.hpp"
#include "mesh/cubed_sphere/interior_radial.hpp"
#include "mesh/cubed_sphere/research_radial_sbp.hpp"
#include "goal1_holonomic_momentum.hpp"
namespace cubed_sphere {namespace hyp=z4c::hyperboloidal;using Z=z4c::Z4c;
#include "native_v2_loader.hpp"
}
namespace cs=cubed_sphere;namespace h=z4c::hyperboloidal;namespace hc=cs::holonomic_chart;namespace hm=cs::holonomic_momentum;
using State=h::Z4cJet<Real>;
std::array<Real,10> Check(const State&u,const h::LayerPoint<Real>&ref,const Real*x,const std::string&kind,int index){
 hc::Config q,q0;if(!hc::Encode(u,q)||!hc::Encode(ref.state,q0))throw std::runtime_error("invalid holonomic fixture");std::array<Real,10>pi,velocity,F;
 if(!hm::Momentum(u,ref.state,ref.omega,pi)||!hm::NativeVelocity(u,ref,x,velocity)||!hm::Source(u,ref,x,F))throw std::runtime_error("nonlinear q/momentum evaluation failed");auto restored=u;if(!hm::Reconstruct(restored,ref.state,ref.omega,pi))throw std::runtime_error("nonlinear momentum inverse failed");std::array<Real,10>after;if(!hm::Momentum(restored,ref.state,ref.omega,after))throw std::runtime_error("inverse momentum invalid");
 Real source=0,reverse=0,pvalue=0,qchange=0;for(int n=0;n<10;++n){Real expected=pi[n]+F[n];for(int d=0;d<3;++d)expected+=u.beta.value[d]*(q[n].d[d]-q0[n].d[d]);source=std::max(source,std::abs(velocity[n]-expected)/std::max({Real(1),std::abs(velocity[n]),std::abs(expected)}));reverse=std::max(reverse,std::abs(pi[n]-after[n])/std::max(Real(1),std::abs(pi[n])));}
 pvalue=std::max(std::abs(u.trace.value-restored.trace.value)/std::max(Real(1),std::abs(u.trace.value)),std::abs(u.theta.value-restored.theta.value)/std::max(Real(1),std::abs(u.theta.value)));
 auto g=h::Geometry(u.metric);Real trace=0,naive=0;
 for(int i=0;i<3;++i){pvalue=std::max(pvalue,std::abs(u.lambda.value[i]-restored.lambda.value[i])/std::max(Real(1),std::abs(u.lambda.value[i])));for(int j=0;j<3;++j){pvalue=std::max(pvalue,std::abs(u.a.k[i][j]-restored.a.k[i][j])/std::max(Real(1),std::abs(u.a.k[i][j])));trace+=g.inverse[i][j]*restored.a.k[i][j];naive+=g.inverse[i][j]*ref.state.a.k[i][j];}}
 hc::Config back;if(!hc::Encode(restored,back))throw std::runtime_error("inverse changed configuration admission");for(int n=0;n<10;++n){qchange=std::max(qchange,std::abs(q[n].f-back[n].f));for(int d=0;d<3;++d){qchange=std::max(qchange,std::abs(q[n].d[d]-back[n].d[d]));for(int e=0;e<3;++e)qchange=std::max(qchange,std::abs(q[n].h[d][e]-back[n].h[d][e]));}}
 std::cout<<"{\"kind\":\""<<kind<<"\",\"case\":"<<index<<",\"Omega\":"<<ref.omega<<",\"native_q_decomposition_scaled\":"<<source<<",\"canonical_inverse_scaled\":"<<reverse<<",\"native_momentum_value_roundtrip_scaled\":"<<pvalue<<",\"tracefree_A_absolute\":"<<std::abs(trace)<<",\"configuration_jet_change_absolute\":"<<qchange<<",\"naive_reference_A_trace_defect\":"<<std::abs(naive)<<",\"F\":[";for(int i=0;i<10;++i)std::cout<<(i?",":"")<<F[i];std::cout<<"]}"<<std::endl;return F;
}
void Audit(const State&u,const h::LayerPoint<Real>&ref,const Real*x,const Real*,const std::string&kind,int index){Check(u,ref,x,kind,index);}
#include "chart_native_cases.hpp"
void Synthetic(){std::mt19937_64 rng(20261018);std::normal_distribution<Real>normal(0,1);h::LayerReference<Real>reference(1.,.5,{true,.05,.95});int index=0;Real normals[3][3]={{1,0,0},{.8,-.48,.36},{1/sqrt(Real(3)),1/sqrt(Real(3)),1/sqrt(Real(3))}};
 for(Real radius:{0.,.05,.2,.5,.7,.9,.95,.98})for(auto&n:normals){Real x[3];for(int i=0;i<3;++i)x[i]=radius*n[i];auto p=reference.At(x[0],x[1],x[2]);hc::Config initial,same;hc::Encode(p.state,initial);std::array<Real,10>previousF{};
  for(int variant=0;variant<5;++variant){hc::Config q=variant==3?same:initial;auto u=p.state;if(variant){for(int i=0;i<10;++i){if(variant!=3){if(i==0||i==6)q[i].f*=1+.005*normal(rng);else q[i].f+=.005*normal(rng);}for(int d=0;d<3;++d){q[i].d[d]+=.01*normal(rng);for(int e=d;e<3;++e){q[i].h[d][e]+=.01*normal(rng);q[i].h[e][d]=q[i].h[d][e];}}}if(variant==2)same=q;if(!hc::Decode(q,u))throw std::runtime_error("synthetic config decode failed");auto g=h::Geometry(u.metric);Real tr=0;for(int i=0;i<3;++i){u.lambda.value[i]+=.005*normal(rng);for(int j=i;j<3;++j){u.a.k[i][j]+=.005*normal(rng);u.a.k[j][i]=u.a.k[i][j];}}for(int i=0;i<3;++i)for(int j=0;j<3;++j)tr+=g.inverse[i][j]*u.a.k[i][j];for(int i=0;i<3;++i)for(int j=0;j<3;++j)u.a.k[i][j]-=u.metric.g[i][j]*tr/3;u.trace.value+=.005*normal(rng);u.theta.value+=.005*normal(rng);}
   auto F=Check(u,p,x,"synthetic",index++);if(variant==2)previousF=F;if(variant==3){Real error=0;for(int i=0;i<10;++i)error=std::max(error,std::abs(F[i]-previousF[i]));std::cout<<"{\"kind\":\"source-invariance\",\"case\":"<<index-1<<",\"gradient_momentum_independence_absolute\":"<<error<<"}"<<std::endl;}
  }
 }
}
int main(int argc,char**argv){Kokkos::initialize(argc,argv);std::cout<<std::setprecision(17);int status=0;try{if(argc==1)Synthetic();else if(argc==4)NativeCase(argv[1],argv[2],argv[3]);else throw std::runtime_error("use noargs or checkpoint reference label");}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;status=2;}Kokkos::finalize();return status;}
