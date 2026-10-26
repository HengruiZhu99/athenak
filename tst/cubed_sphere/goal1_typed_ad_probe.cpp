// Private typed-AD adapter must precede every native template definition.
#include "goal1_native_ad.hpp"
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
#include "goal1_typed_support.hpp"
#include "goal1_typed_momentum.hpp"
#include "goal1_holonomic_momentum.hpp"
namespace cubed_sphere {namespace hyp=z4c::hyperboloidal;using Z=z4c::Z4c;
#include "native_v2_loader.hpp"
}
namespace cs=cubed_sphere;namespace h=z4c::hyperboloidal;namespace ts=cs::typed_support;namespace typedmom=cs::typed_momentum;namespace hm=cs::holonomic_momentum;namespace hc=cs::holonomic_chart;using State=h::Z4cJet<Real>;
const int ia[6]={0,0,0,1,1,2},ib[6]={0,1,2,1,2,2};
template<class T>std::array<T,37> Functional(const h::Z4cJet<T>&u){std::array<T,37>a{};a[0]=u.chi.value;a[1]=u.alpha.value;for(int i=0;i<6;++i){a[2+i]=u.metric.g[ia[i]][ib[i]];a[13+i]=u.a.k[ia[i]][ib[i]];}for(int i=0;i<3;++i){a[8+i]=u.beta.value[i];a[19+i]=u.lambda.value[i];a[22+i]=u.chi.d[i];a[25+i]=u.alpha.d[i];for(int j=0;j<3;++j)a[28+3*i+j]=u.beta.d[i][j];}a[11]=u.trace.value;a[12]=u.theta.value;return a;}
template<class T>h::Z4cJet<T> StateFrom(const std::array<T,37>&a){h::Z4cJet<T>u{};u.chi.value=a[0];u.alpha.value=a[1];for(int i=0;i<6;++i){u.metric.g[ia[i]][ib[i]]=u.metric.g[ib[i]][ia[i]]=a[2+i];u.a.k[ia[i]][ib[i]]=u.a.k[ib[i]][ia[i]]=a[13+i];}for(int i=0;i<3;++i){u.beta.value[i]=a[8+i];u.lambda.value[i]=a[19+i];u.chi.d[i]=a[22+i];u.alpha.d[i]=a[25+i];for(int j=0;j<3;++j)u.beta.d[i][j]=a[28+3*i+j];}u.trace.value=a[11];u.theta.value=a[12];return u;}
cs::Jet Seed(int id,Real v,Real relative=.02){cs::Jet j(v);Real scale=relative*std::max(Real(1),std::abs(v));for(int d=0;d<3;++d){j.d[d]=scale*sin((id+1)*(.31+.17*d));for(int e=0;e<3;++e)j.h[d][e]=scale*.2*cos((id+1)*(.13+.07*(d+1)*(e+1)));}return j;}
void Emit(const cs::Jet&a){std::cout<<"["<<a.f;for(double v:a.d)std::cout<<","<<v;for(auto&r:a.h)for(double v:r)std::cout<<","<<v;std::cout<<"]";}
template<size_t N>void Emit(const std::array<cs::Jet,N>&a){std::cout<<"[";for(size_t i=0;i<N;++i){if(i)std::cout<<",";Emit(a[i]);}std::cout<<"]";}
Real Scaled(const cs::Jet&a,const cs::Jet&b){Real m=std::abs(a.f-b.f)/std::max({Real(1),std::abs(a.f),std::abs(b.f)});for(int d=0;d<3;++d){m=std::max(m,std::abs(a.d[d]-b.d[d])/std::max({Real(1),std::abs(a.d[d]),std::abs(b.d[d])}));for(int e=0;e<3;++e)m=std::max(m,std::abs(a.h[d][e]-b.h[d][e])/std::max({Real(1),std::abs(a.h[d][e]),std::abs(b.h[d][e])}));}return m;}
void CoreAudit(const State&u,const h::LayerPoint<Real>&p,const Real*x,const std::string&kind,int index){
 auto native=ts::Map<cs::Jet>(u,[](int id,Real v){return Seed(id+1700,v,.01);});auto reference=ts::Constant<cs::Jet>(p);auto c0=rwm::ReferenceConnection(p,x);auto connection=ts::Constant<cs::Jet>(c0);auto f=ts::Core(native,reference,connection);auto exact=ts::Core(u,p,c0);Real value=0;for(int i=0;i<22;++i)value=std::max(value,std::abs(f[i].f-exact[i])/std::max(Real(1),std::abs(exact[i])));
 std::array<std::array<Real,22>,2>first{},second{};const Real steps[2]={.003,.0015};
 for(int level=0;level<2;++level){Real step=steps[level];std::array<std::array<Real,22>,5>sample{};for(int k=-2;k<=2;++k){Real t=k*step;auto shifted=ts::Map<Real>(native,[&](int,const cs::Jet&j){return j.f+t*j.d[0]+.5*t*t*j.h[0][0];});sample[k+2]=ts::Core(shifted,p,c0);}for(int i=0;i<22;++i){first[level][i]=(sample[0][i]-8*sample[1][i]+8*sample[3][i]-sample[4][i])/(12*step);second[level][i]=(-sample[4][i]+16*sample[3][i]-30*sample[2][i]+16*sample[1][i]-sample[0][i])/(12*step*step);}}
 Real d=0,dd=0,uncertainty=0;for(int i=0;i<22;++i){d=std::max(d,std::abs(first[1][i]-f[i].d[0])/std::max(Real(1),std::abs(f[i].d[0])));dd=std::max(dd,std::abs(second[1][i]-f[i].h[0][0])/std::max(Real(1),std::abs(f[i].h[0][0])));uncertainty=std::max({uncertainty,std::abs(first[1][i]-first[0][i])/std::max(Real(1),std::abs(f[i].d[0])),std::abs(second[1][i]-second[0][i])/std::max(Real(1),std::abs(f[i].h[0][0]))});}
 std::cout<<"{\"kind\":\"native-core-AD\",\"fixture_kind\":\""<<kind<<"\",\"case\":"<<index<<",\"native22_value_scaled\":"<<value<<",\"native22_FD_first_scaled\":"<<d<<",\"native22_FD_second_scaled\":"<<dd<<",\"FD_refinement_uncertainty_scaled\":"<<uncertainty<<"}"<<std::endl;
}
std::array<Real,10>Check(const State&u,const h::LayerPoint<Real>&ref,const Real*x,const std::string&kind,int index){
 std::array<Real,10>old,newer;hm::Momentum(u,ref.state,ref.omega,old);if(!typedmom::Momentum(u,ref.state,ref.omega,newer))throw std::runtime_error("new value momentum invalid");auto oldp=u,newp=u;hm::Reconstruct(oldp,ref.state,ref.omega,old);if(!typedmom::Reconstruct(newp,ref.state,ref.omega,old))throw std::runtime_error("new value inverse invalid");auto v1=Functional(oldp),v2=Functional(newp);Real ve=0;for(int i=0;i<10;++i)ve=std::max(ve,std::abs(old[i]-newer[i])/std::max(Real(1),std::abs(old[i])));for(int i=11;i<22;++i)ve=std::max(ve,std::abs(v1[i]-v2[i])/std::max(Real(1),std::abs(v1[i])));
 auto a=Functional(u),r=Functional(ref.state);std::array<cs::Jet,37>aj,rj;for(int i=0;i<37;++i){aj[i]=Seed(i+index*3,a[i]);rj[i]=Seed(i+400+index*3,r[i]);}auto omega=Seed(1200+index,ref.omega,1);for(int i=0;i<3;++i){omega.d[i]*=.02*ref.omega;for(int j=0;j<3;++j)omega.h[i][j]*=.02*ref.omega;}auto ua=StateFrom(aj),ra=StateFrom(rj);std::array<cs::Jet,10>pi,target;for(int i=0;i<10;++i)target[i]=Seed(i+900+index,old[i]);if(!typedmom::Momentum(ua,ra,omega,pi))throw std::runtime_error("AD momentum invalid");auto reconstructed=ua;if(!typedmom::Reconstruct(reconstructed,ra,omega,target))throw std::runtime_error("AD inverse invalid");std::array<cs::Jet,10>back;typedmom::Momentum(reconstructed,ra,omega,back);Real round=0;for(int i=0;i<10;++i)round=std::max(round,Scaled(back[i],target[i]));auto inverse=Functional(reconstructed);auto geometry=h::Geometry(reconstructed.metric);cs::Jet tr;for(int i=0;i<3;++i)for(int j=0;j<3;++j)tr+=geometry.inverse[i][j]*reconstructed.a.k[i][j];Real trace=Scaled(tr,cs::Jet(0)),qchange=0;for(int i=0;i<37;++i)if(i<11||i>21)qchange=std::max(qchange,Scaled(aj[i],inverse[i]));
 std::cout<<"{\"kind\":\""<<kind<<"\",\"case\":"<<index<<",\"Omega\":"<<ref.omega<<",\"preserved_Real_value_scaled\":"<<ve<<",\"AD_inverse_momentum_jet_roundtrip_scaled\":"<<round<<",\"AD_tracefree_A_jet_scaled\":"<<trace<<",\"AD_configuration_change_scaled\":"<<qchange;
 if((kind=="synthetic"&&index%12==0)||(kind!="synthetic"&&index%23==0)){std::cout<<",\"input_u\":";Emit(aj);std::cout<<",\"input_ref\":";Emit(rj);std::cout<<",\"input_Omega\":";Emit(omega);std::cout<<",\"target_Pi\":";Emit(target);std::cout<<",\"forward_Pi\":";Emit(pi);std::cout<<",\"inverse_p\":[";for(int i=11;i<22;++i){if(i>11)std::cout<<",";Emit(inverse[i]);}std::cout<<"]";}std::cout<<"}"<<std::endl;
 if((kind=="synthetic"&&index%30==0)||(kind!="synthetic"&&(index==0||index==169)))CoreAudit(u,ref,x,kind,index);return {};
}
void Audit(const State&u,const h::LayerPoint<Real>&ref,const Real*x,const Real*,const std::string&kind,int index){Check(u,ref,x,kind,index);}
#include "chart_native_cases.hpp"
// The fixed synthetic/native fixture construction is copied into this build.
#include "ad_synthetic_cases.hpp"
int main(int argc,char**argv){Kokkos::initialize(argc,argv);std::cout<<std::setprecision(17);int status=0;try{if(argc==1)Synthetic();else if(argc==4)NativeCase(argv[1],argv[2],argv[3]);else throw std::runtime_error("use noargs or checkpoint reference label");}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;status=2;}Kokkos::finalize();return status;}
