// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include <algorithm>
#include <iomanip>
#include <iostream>
#include "z4c/hyperboloidal/puncture_position.hpp"
#include "z4c/hyperboloidal/puncture_neighborhood.hpp"
#include "z4c/hyperboloidal/wormhole_chart.hpp"
namespace h=z4c::hyperboloidal;
struct TrackerProbe {
  Real x[3]={.037,.013,-.021};int publications=0;
  Real GetPos(int a)const{return x[a];}
  void SetPos(Real *p){for(int a=0;a<3;++a)x[a]=p[a];++publications;}
};
template<typename F>void Rejected(F action,const char *message){
  bool rejected=false;try{action();}catch(const std::exception&){rejected=true;}
  if(!rejected)throw std::runtime_error(message);
}
int main(int argc,char **argv){Kokkos::initialize(argc,argv);int status=0;
try{
 TrackerProbe tracker;const Real beta[3]={.13,-.07,.04};
 auto point=h::PuncturePosition::FromTracker(tracker,beta,.3,2);
 for(int a=0;a<3;++a)if(point.position[a]!=tracker.x[a]||point.velocity[a]!=-beta[a])
   throw std::runtime_error("tracker position/transport sign");
 Rejected([&]{point.PublishAccepted(tracker);},"intermediate position publication accepted");
 if(tracker.publications)throw std::runtime_error("tracker mutated during sampling");
 const auto stamp=point.Capture();if(!point.Matches(stamp))throw std::runtime_error("fresh puncture stamp");
 point.SetKinematics(point.position,point.velocity,point.time,3);
 if(point.Matches(stamp)||point.revision!=stamp.revision+1)throw std::runtime_error("puncture revision invalidation");
 const auto direct=point.Capture();point.position[0]+=.001;
 if(point.Matches(direct))throw std::runtime_error("direct position edit bypassed stamp");
 point.SetKinematics({.037,.013,-.021},{-.13,.07,-.04},.3,-1);
 point.PublishAccepted(tracker);if(tracker.publications!=1)throw std::runtime_error("accepted tracker publication");
 const Real endpoint[3]={point.position[0],point.position[1],point.position[2]};
 if(!point.AtEndpoint(endpoint)||point.RadiusSquared(endpoint)!=0)throw std::runtime_error("explicit endpoint detection");
 const auto q0=point.SquaredRadiusJet(endpoint);
 if(q0.value!=0||q0.time_derivative!=0||q0.hessian[0][0]!=2)throw std::runtime_error("squared-radius endpoint jet");
 Rejected([&]{point.RadialPowerJet(endpoint,2.);},"power endpoint was implicitly regularized");
 Rejected([&]{point.FixedChartRate(endpoint,2.,0.,1.);},"chart endpoint was implicitly divided");
 Real time_error=0,space_error=0,rate_error=0,chart_time_error=0;int cases=0;
 for(const std::array<Real,3> x:{std::array<Real,3>{.17,-.11,.09},std::array<Real,3>{-.12,.08,.2}})
 for(Real power:{-1.3,0.,1.09,2.,4.,6.}){
  const auto jet=point.RadialPowerJet(x.data(),power);const Real eps=1e-5;
  auto plus=point,minus=point;for(int a=0;a<3;++a){plus.position[a]+=eps*point.velocity[a];minus.position[a]-=eps*point.velocity[a];}
  const Real difference=(plus.RadialPowerJet(x.data(),power).value-minus.RadialPowerJet(x.data(),power).value)/(2*eps);
  time_error=std::max(time_error,std::abs(difference-jet.time_derivative)/std::max(Real(1),std::abs(jet.time_derivative)));
  for(int a=0;a<3;++a){auto xp=x,xm=x;xp[a]+=eps;xm[a]-=eps;
   const auto jp=point.RadialPowerJet(xp.data(),power),jm=point.RadialPowerJet(xm.data(),power);
   space_error=std::max(space_error,std::abs((jp.value-jm.value)/(2*eps)-jet.gradient[a])/std::max(Real(1),std::abs(jet.gradient[a])));
   for(int b=0;b<3;++b)space_error=std::max(space_error,std::abs((jp.gradient[b]-jm.gradient[b])/(2*eps)-jet.hessian[a][b])/std::max(Real(1),std::abs(jet.hessian[a][b])));}
  const Real coefficient=1.2+.3*point.time+.2*x[0]-.1*x[1]*x[1];
  const Real physical_rate=jet.time_derivative*coefficient+jet.value*.3;
  rate_error=std::max(rate_error,std::abs(point.FixedChartRate(x.data(),power,physical_rate,coefficient)-.3));++cases;
  const h::WormholeChart chart(x.data(),point),cp(x.data(),plus),cm(x.data(),minus);
  const Real derivatives[3]={chart.chi_t,chart.weight_t,chart.alpha_t};
  const Real forwards[3]={cp.chi.f,cp.weight.f,cp.alpha.f},backwards[3]={cm.chi.f,cm.weight.f,cm.alpha.f};
  for(int f=0;f<3;++f)chart_time_error=std::max(chart_time_error,
    std::abs((forwards[f]-backwards[f])/(2*eps)-derivatives[f]));
 }
 const Real gradient[3]={.2,-.4,.1};Real transport=0;for(int a=0;a<3;++a)transport+=beta[a]*gradient[a];
 if(std::abs(point.ComovingRate(transport,gradient))>1e-15)throw std::runtime_error("nonzero constant beta transport was pinned/mis-signed");
 auto invalid=point;invalid.velocity[1]=std::numeric_limits<Real>::quiet_NaN();
 Rejected([&]{invalid.Relative(endpoint);},"nonfinite kinematics accepted");
 cubed_sphere::CompositeDifference grid(cubed_sphere::CompositeMesh(.25,{.65,1},1,4,2,2));
 const auto original_active=grid.active;const size_t original_halos=grid.halos.size();
 h::PunctureNeighborhood neighborhood(grid,.12);
 if(!neighborhood.Refresh(point)||!neighborhood.Matches(point)||neighborhood.Refresh(point))
   throw std::runtime_error("tracked neighborhood cache lifecycle");
 if(neighborhood.ActivePoints(point).empty()||neighborhood.HaloTargets(point).empty())
   throw std::runtime_error("tracked active/halo neighborhood empty");
 const auto scales=neighborhood.BoundedEquilibration(point);
 if(scales.size()!=grid.active.size())throw std::runtime_error("equilibration omitted active cells");
 for(size_t p=0;p<scales.size();++p){if(scales[p]<.5||scales[p]>1)
     throw std::runtime_error("equilibration not uniformly positive/bounded");
   if(!point.InNeighborhood(grid.geometry[p].x,.12)&&scales[p]!=1)
     throw std::runtime_error("equilibration changed an exterior cell");}
 const auto builds=neighborhood.RebuildCount();auto translated=point;translated.position[0]+=.013;
 if(neighborhood.Matches(translated)||!neighborhood.Refresh(translated)||neighborhood.RebuildCount()!=builds+1)
   throw std::runtime_error("translated neighborhood did not invalidate");
 const auto shifted_scales=neighborhood.BoundedEquilibration(translated);Real scaling_motion=0;
 for(size_t p=0;p<scales.size();++p)scaling_motion=std::max(scaling_motion,std::abs(scales[p]-shifted_scales[p]));
 if(!(scaling_motion>0))throw std::runtime_error("equilibration failed to follow moving position");
 translated.SetKinematics(translated.position,translated.velocity,translated.time+.001,1);
 if(neighborhood.Matches(translated)||!neighborhood.Refresh(translated))
   throw std::runtime_error("time/stage change retained stale neighborhood");
 auto coincident=point;const auto &cell=grid.geometry[0].x;
 coincident.SetKinematics({cell[0],cell[1],cell[2]},point.velocity,point.time);
 const auto at_cell=neighborhood.BoundedEquilibration(coincident);
 if(at_cell.size()!=grid.active.size()||at_cell[0]!=.5)
   throw std::runtime_error("puncture-coincident cell was excluded/floored");
 auto near_scri=point;near_scri.position={.95,0,0};
 Rejected([&]{neighborhood.BoundedEquilibration(near_scri);},"equilibration support reached scri");
 if(grid.active!=original_active||grid.halos.size()!=original_halos)
   throw std::runtime_error("puncture metadata mutated grid/halo ownership");
 if(time_error>2e-8||space_error>3e-7||rate_error>3e-14||chart_time_error>2e-9)
   throw std::runtime_error("moving puncture factor/chart chain rule");
 std::cout<<std::setprecision(17)<<"{\"cases\":"<<cases<<",\"time_relative_error\":"<<time_error
   <<",\"spatial_relative_error\":"<<space_error<<",\"coefficient_rate_error\":"<<rate_error
   <<",\"actual_chart_time_error\":"<<chart_time_error<<",\"all_active_cells\":"<<scales.size()
   <<",\"scaling_motion\":"<<scaling_motion<<",\"beta_transport_sign\":-1,\"revision_invalidation\":true,\"explicit_endpoint\":true,\"tracker_stage_publication_rejected\":true}\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';status=1;}
Kokkos::finalize();return status;}
