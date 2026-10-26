#include "goal1_endpoint_point.hpp"
#include <iostream>
#include <iomanip>
int main(int argc,char**argv){Kokkos::initialize(argc,argv);std::cout<<std::setprecision(17);int index,status=0;Real eps,x[3];try{while(std::cin>>index>>eps>>x[0]>>x[1]>>x[2]){
 namespace cs=cubed_sphere;namespace h=z4c::hyperboloidal;h::Z4cJet<Real>z{};auto u=cs::typed_support::Map<Real>(z,[](int,Real){Real v;if(!(std::cin>>v))throw std::runtime_error("short fixture");return v;});h::LayerReference<Real>descriptor(1.,.5,{true,.05,.95});auto p=descriptor.At(x[0],x[1],x[2]);auto f=cs::endpoint_point::Evaluate(u,p,x);auto perturbed=cs::endpoint_point::Evaluate(u,p,x,.123);Real rhs=0,invariance=0;for(int n=0;n<22;++n){rhs=std::max(rhs,std::abs(f.rhs[n]));invariance=std::max(invariance,std::abs(f.rhs[n]-perturbed.rhs[n]));}auto bad=u;bad.trace.value+=.001;auto rejected=cs::endpoint_point::Evaluate(bad,p,x);
 std::cout<<"{\"case\":"<<index<<",\"epsilon\":"<<eps<<",\"valid\":"<<(f.valid?"true":"false")<<",\"pole_max\":"<<f.pole_max<<",\"tangent_max\":"<<f.tangent_max<<",\"endpoint_rhs_max\":"<<rhs<<",\"unused_jet_invariance\":"<<invariance<<",\"invalid_trace_rejected\":"<<(!rejected.valid?"true":"false")<<",\"rhs\":[";for(int n=0;n<22;++n)std::cout<<(n?",":"")<<f.rhs[n];std::cout<<"]}"<<std::endl;
 }}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;status=2;}Kokkos::finalize();return status;}
