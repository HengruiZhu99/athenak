// No repair: strictly interior SPD determinant-one donor perturbation may extrapolate invalid ghosts.
#include <cmath>
#include <iostream>
#include "z4c/hyperboloidal/cartesian_patch.hpp"
namespace hyp=z4c::hyperboloidal;
int main(int argc,char**argv){
 Kokkos::ScopeGuard guard(argc,argv);hyp::SphericalGhostGrid g;g.radius=1;
 for(int d=0;d<3;++d){g.n[d]=30;g.h[d]=2.1/24;g.first[d]=-.5*29*g.h[d];}
 hyp::LayerParameters lp;lp.enabled=true;lp.r0=.05;lp.r1=.95;
 hyp::LayerGaugeParameters lg;lg.physical_trace_lapse=true;lg.preferred_source=false;
 hyp::CartesianConformalPatch p(g,.5,2,lp,lg,true);auto q=p.Allocate("SPD donor rejection");p.InitializeReference(q);
 auto plans=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),p.ghosts);
 int donor=-1;double weight=0;
 for(size_t a=0;a<plans.extent(0);++a)for(int b=0;b<plans(a).count;++b)if(plans(a).weights[b]<weight){weight=plans(a).weights[b];donor=plans(a).donors[b];}
 if(donor<0)throw std::runtime_error("expected negative extrapolation weight");
 const int i=donor%30,j=donor/30%30,k=donor/900;
 if(!g.Interior(i,j,k))throw std::runtime_error("donor outside physical sphere");
 auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),q);
 for(int c=0;c<6;++c)host(0,z4c::Z4c::I_Z4C_GXX+c,k,j,i)=0;
 host(0,z4c::Z4c::I_Z4C_GXX,k,j,i)=1000;
 host(0,z4c::Z4c::I_Z4C_GYY,k,j,i)=host(0,z4c::Z4c::I_Z4C_GZZ,k,j,i)=1/std::sqrt(1000.);
 Kokkos::deep_copy(q,host);bool caught=false;
 try{p.Prepare(q);}catch(const std::runtime_error&e){caught=std::string(e.what()).find("invalid full ghost matrix/trace")!=std::string::npos;}
 if(!caught)throw std::runtime_error("invalid ghost did not fail explicitly");
 std::cout<<"{\"strict_interior_donor\":"<<donor<<",\"donor_metric_spd_det_one\":true,\"negative_weight\":"<<weight<<",\"invalid_extrapolated_ghost_rejected\":true}\n";
}
