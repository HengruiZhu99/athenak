// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#define ELLIPTIC_SHIFT_3D_AMR_CORE_NO_MAIN
#include "test_amr_core_derivatives.cpp"
#include "z4c/hyperboloidal/enriched_log_value.hpp"
int main(int argc,char **argv){Kokkos::initialize(argc,argv);int status=0;try{
  auto grid=MakeGrid();const int index=argc>1?std::stoi(argv[1]):48768;const auto x=PaddedPoint(grid,index);const auto &receiving=grid.mesh.blocks[index/(grid.padded*grid.padded*grid.padded)];const Real spacing=(receiving.hi[0]-receiving.lo[0])/grid.n;
  std::vector<std::pair<Real,size_t>> near;for(size_t p=0;p<grid.active.size();++p)if(grid.mesh.blocks[p/(grid.n*grid.n*grid.n)].face<0){Real d=0;for(int a=0;a<3;++a)d+=std::pow(grid.geometry[p].x[a]-x[a],2);near.emplace_back(d,p);}std::partial_sort(near.begin(),near.begin()+256,near.end());
  std::cout<<std::setprecision(17);h::PuncturePosition point;
  for(int count:{125,126,128,135,150,175,200,256}){std::vector<Point> nodes;std::set<Real> axes[3];for(int i=0;i<count;++i){const auto &q=grid.geometry[near[i].second].x;nodes.push_back({q[0],q[1],q[2]});for(int a=0;a<3;++a)axes[a].insert(q[a]);}
    h::EnrichedLogValue plan;const bool ready=plan.Build(x,nodes,spacing,point,grid.mesh.revision,8);const auto &d=plan.diagnostics();std::cout<<"{\"query\":"<<index<<",\"xyz\":["<<x[0]<<','<<x[1]<<','<<x[2]<<"],\"rho_over_h\":"<<Radius(x.data())/spacing<<",\"receiving_refinement\":"<<receiving.refinement<<",\"donors\":"<<count<<",\"axis_levels\":["<<axes[0].size()<<','<<axes[1].size()<<','<<axes[2].size()<<"],\"ready\":"<<ready<<",\"status\":\""<<d.status<<"\",\"rank\":"<<d.rank<<",\"condition\":"<<d.basis_condition<<",\"value_l1\":"<<d.value_weight_l1<<",\"value_reproduction\":"<<d.value_reproduction<<"}\n";
  }
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';status=1;}Kokkos::finalize();return status;}
