#include "goal1_native_ad.hpp"
#include "goal1_native_source_ad.hpp"
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
namespace cs=cubed_sphere;namespace h=z4c::hyperboloidal;namespace hc=cs::holonomic_chart;namespace hm=cs::holonomic_momentum;namespace sourcead=cs::native_source_ad;using State=h::Z4cJet<Real>;
std::array<Real,10>Check(const State&u,const h::LayerPoint<Real>&p,const Real*x,const std::string&kind,int index){
 static int ordinal=0;h::LayerReference<Real>descriptor(1.,.5,{true,.05,.95});hc::Config q;if(!hc::Encode(u,q))throw std::runtime_error("invalid source input");auto source=sourcead::Source(q,descriptor,p,x);if(!source.valid)throw std::runtime_error("invalid AD q source");std::array<Real,10>native;if(!hm::Source(u,p,x,native))throw std::runtime_error("invalid native q source");Real value=0;for(int n=0;n<10;++n)value=std::max(value,std::abs(source.value[n]-native[n])/std::max(Real(1),std::abs(native[n])));
 std::cout<<"{\"kind\":\""<<kind<<"\",\"case\":"<<index<<",\"ordinal\":"<<ordinal++<<",\"x\":["<<x[0]<<","<<x[1]<<","<<x[2]<<"],\"Omega\":"<<p.omega<<",\"native_source_value_scaled\":"<<value;
 if((kind=="synthetic"&&index%12==0)||(kind!="synthetic"&&index%23==0)){
  auto changed=sourcead::Source(q,descriptor,p,x,.123);if(!changed.valid)throw std::runtime_error("unused dependency fixture invalid");Real unused=0,defect=0,uncertainty=0;for(int n=0;n<10;++n){unused=std::max(unused,std::abs(source.value[n]-changed.value[n]));for(int d=0;d<3;++d)unused=std::max(unused,std::abs(source.gradient[d][n]-changed.gradient[d][n]));}
  for(int d=0;d<3;++d){Real fd[2][10]{};for(int level=0;level<2;++level){const Real step=level?5e-5:1e-4;Real samples[5][10];for(int k=-2;k<=2;++k){const Real t=k*step;Real xx[3]={x[0],x[1],x[2]};xx[d]+=t;auto pp=descriptor.At(xx[0],xx[1],xx[2]);auto path=q;for(int n=0;n<10;++n){path[n].f=q[n].f+t*q[n].d[d]+.5*t*t*q[n].h[d][d];for(int i=0;i<3;++i){path[n].d[i]=0;for(int j=0;j<3;++j)path[n].h[i][j]=0;}}State shifted{};hc::Decode(path,shifted);std::array<Real,10>ff;if(!hm::Source(shifted,pp,xx,ff))throw std::runtime_error("FD q source failed");for(int n=0;n<10;++n)samples[k+2][n]=ff[n];}for(int n=0;n<10;++n)fd[level][n]=(samples[0][n]-8*samples[1][n]+8*samples[3][n]-samples[4][n])/(12*step);}
   for(int n=0;n<10;++n){const Real scale=std::max(Real(1),std::abs(source.gradient[d][n]));defect=std::max(defect,std::abs(fd[1][n]-source.gradient[d][n])/scale);uncertainty=std::max(uncertainty,std::abs(fd[1][n]-fd[0][n])/scale);}
  }
  std::cout<<",\"source_gradient_FD_scaled\":"<<defect<<",\"FD_refinement_uncertainty_scaled\":"<<uncertainty<<",\"unused_higherjet_invariance_absolute\":"<<unused;
 }
 std::cout<<"}"<<std::endl;return native;
}
void Audit(const State&u,const h::LayerPoint<Real>&p,const Real*x,const Real*,const std::string&kind,int index){Check(u,p,x,kind,index);}
#include "chart_native_cases.hpp"
#include "ad_synthetic_cases.hpp"
int main(int argc,char**argv){Kokkos::initialize(argc,argv);std::cout<<std::setprecision(17);int status=0;try{if(argc==1)Synthetic();else if(argc==4)NativeCase(argv[1],argv[2],argv[3]);else throw std::runtime_error("use noargs or checkpoint reference label");}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;status=2;}Kokkos::finalize();return status;}
