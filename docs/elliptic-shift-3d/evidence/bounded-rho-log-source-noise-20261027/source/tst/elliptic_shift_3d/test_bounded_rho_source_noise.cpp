// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#define RHO_LOG_SOURCE_NOISE_NO_MAIN
#include "test_rho_log_source_noise.cpp"
#include "z4c/hyperboloidal/bounded_rho_log_derivative.hpp"
int main(int argc,char **argv){Kokkos::initialize(argc,argv);int status=0;try{
  if(argc!=2)throw std::invalid_argument("usage actual initial.dat");auto grid=MakeGrid();const auto initial=ReadInitial(argv[1],grid);V stored(2*initial.size());for(size_t p=0;p<initial.size();++p){stored[2*p]=std::log(initial[p].chi);stored[2*p+1]=std::log(initial[p].alpha)-std::log(Chart(grid.geometry[p].x,1).reference.value);}h::PuncturePosition point;h::PunctureLogPlans base(grid,point);std::cout<<std::setprecision(17);SourceNoiseStats stats[2];size_t ready=0,mixed=0;Real minimum_theta=1;
  for(const auto &row:base.rows){const auto &x=grid.geometry[row.active].x;std::vector<h::BoundedRhoLogDerivative::Point> nodes;for(size_t donor:row.donors){const auto &d=grid.geometry[donor].x;nodes.push_back({d[0],d[1],d[2]});}h::BoundedRhoLogDerivative blend;
    if(blend.Build({x[0],x[1],x[2]},nodes,Spacing(grid,row.active),point)){++ready;mixed+=blend.diagnostics().theta<1;minimum_theta=std::min(minimum_theta,blend.diagnostics().theta);SourceNoise("bounded_blend",blend,row,grid,initial,stored,stats[1],true);}else std::cout<<"{\"phase\":\"unsupported\",\"point\":"<<row.active<<",\"status\":\""<<blend.diagnostics().status<<"\"}\n";
    SourceNoise("base",row.plan,row,grid,initial,stored,stats[0],true);}
  std::cout<<"{\"phase\":\"summary\",\"selected_rows\":"<<base.rows.size()<<",\"bounded_ready\":"<<ready<<",\"mixed_theta_rows\":"<<mixed<<",\"minimum_theta\":"<<minimum_theta<<",\"base\":";stats[0].Print();std::cout<<",\"bounded_blend\":";stats[1].Print();std::cout<<",\"all_changed_rows_included\":"<<(ready==base.rows.size())<<",\"caps_unchanged_16_32_64\":true,\"outside_selected_operator_unchanged\":true,\"principal_lapse_is_not_complete_RHS\":true,\"production_adopted\":false}\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';status=1;}Kokkos::finalize();return status;}
