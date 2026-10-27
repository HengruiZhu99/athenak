// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include <iomanip>
#include <iostream>
#include "z4c/hyperboloidal/composite_shift.hpp"
namespace h=z4c::hyperboloidal;
int main(int argc,char **argv){Kokkos::initialize(argc,argv);int result=0;try{
  auto require=[](bool ok,const char *why){if(!ok)throw std::runtime_error(why);};
  std::vector<Real> logs{0,-1000},a(6,0),b(6,0);a[3]=b[3]=std::exp(250.);
  const Real rescued=h::LogWeightedDot(a,b,logs,h::LogSumPositive(logs));
  const Real relative=std::abs(rescued/std::exp(-500.)-1);
  require(relative<1e-12&&rescued>0&&std::exp(logs[1])==0,"lost small-measure cell with large residual");
  h::CompositeShiftStage stage(cubed_sphere::CompositeMesh(.25,{1},1,6,2),4);
  std::vector<h::MetricJet<Real>> metric(stage.grid.active.size());
  for(auto &g:metric)for(int i=0;i<3;++i)g.g[i][i]=1;
  std::vector<Real> chi(metric.size(),1e-250);stage.SetGeometry(metric,chi);
  multigrid::Vector constant(3*metric.size());for(size_t p=0;p<metric.size();++p){constant[3*p]=1;constant[3*p+1]=-2;constant[3*p+2]=3;}
  const Real uniform_error=std::abs(stage.Dot(constant,constant)-14);
  require(uniform_error<2e-11,"uniform tiny chi normalization");
  chi.assign(metric.size(),1);chi[0]=1e-250;stage.SetGeometry(metric,chi);
  const Real mixed_error=std::abs(stage.Dot(constant,constant)-14);
  require(mixed_error<2e-11,"mixed physical volume normalization");
  multigrid::Vector zero(constant.size(),0),solution=zero;
  const auto report=stage.Solve(zero,{},solution,1e-10,0,2);
  require(report.converged&&report.physical_residual_rms==0,"zero solve with extreme finite chi");
  require(!report.physical_integrals_representable&&std::isinf(report.physical_volume)
      &&std::isfinite(report.physical_log_volume),"unrepresentable volume must remain disclosed");
  bool rejected=false;try{chi[0]=0;stage.SetGeometry(metric,chi);}catch(const std::exception&){rejected=true;}
  require(rejected,"zero chi must require endpoint measure");
  std::cout<<std::setprecision(17)<<"{\"uniform_tiny_chi_error\":"<<uniform_error<<",\"mixed_tiny_chi_error\":"<<mixed_error
    <<",\"rescued_weight_underflow_relative_error\":"<<relative<<",\"physical_log_volume\":"<<report.physical_log_volume
    <<",\"unrepresentable_volume_disclosed\":true,\"all_cells_retained_in_log_dot\":true,\"zero_chi_rejected\":true}\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';result=1;}Kokkos::finalize();return result;}
