// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include <iostream>
#include "z4c/hyperboloidal/native_core_map.hpp"
#include "multigrid/composite_topology_transfer.hpp"
namespace h=z4c::hyperboloidal;
void Require(bool q,const char *s){if(!q)throw std::runtime_error(s);}
template<class F>void Reject(F f,int &n){bool caught=false;try{f();}catch(const std::exception &){caught=true;}Require(caught,"invalid native static leaves accepted");++n;}
std::vector<h::NativeCartesianBlock> LeafRecords(const cubed_sphere::CompositeMesh &mesh){std::vector<h::NativeCartesianBlock> out;
 for(const auto &b:mesh.blocks)if(b.face<0){h::NativeCartesianBlock n;n.native=out.size();n.refinement=b.refinement;for(int a=0;a<3;++a){n.lo[a]=b.lo[a];n.hi[a]=b.hi[a];}out.push_back(n);}return out;}
int main(int argc,char **argv){Kokkos::initialize(argc,argv);int code=0;try{int rejected=0;size_t cases=0,leaves=0,largest=0;
 for(int root:{1,2,4})for(int layers:{1,2,7}){std::vector<Real> radii=layers==7?std::vector<Real>{.45,.55,.60,.65,.70,.75,1}:(layers==2?std::vector<Real>{.65,1}:std::vector<Real>{1});
  auto original=cubed_sphere::CompositeMesh(.25,radii,1,6,4,root);original.RefineCartesian(0);auto records=LeafRecords(original);
  // Native pack slots can differ from the order in a composite source descriptor.
  std::reverse(records.begin(),records.end());for(size_t p=0;p<records.size();++p)records[p].native=p;
  const auto result=h::NativeCartesianCoreMap::FromNativeLeaves(.25,radii,1,6,4,root,records);
  h::NativeCartesianCoreMap map(result,records,6,4);Require(map.matches.size()==records.size(),"leaf omitted");
  for(const auto &m:map.matches){Require(m.native==m.composite,"factory native prefix identity");for(int k=0;k<6;++k)for(int j=0;j<6;++j)for(int i=0;i<6;++i){const Real q[3]={(i+.5)/6,(j+.5)/6,(k+.5)/6};Real x[3];result.Forward(m.composite,q,x);for(int a=0;a<3;++a)Require(x[a]==records[m.native].lo[a]+q[a]*(records[m.native].hi[a]-records[m.native].lo[a]),"actual native center changed");}}
  const auto plan=multigrid::CompositeCoarseningPlan::Build(result);Require(plan.dense_supported&&plan.TerminalCells()<=1024,"coarse cap unsupported");largest=std::max(largest,plan.TerminalCells());leaves+=records.size();++cases;
  auto changed=records;changed.front().refinement++;Reject([&]{h::NativeCartesianCoreMap::FromNativeLeaves(.25,radii,1,6,4,root,changed);},rejected);
  changed=records;changed.front().lo[0]+=.0001;Reject([&]{h::NativeCartesianCoreMap::FromNativeLeaves(.25,radii,1,6,4,root,changed);},rejected);
  changed=records;changed.pop_back();Reject([&]{h::NativeCartesianCoreMap::FromNativeLeaves(.25,radii,1,6,4,root,changed);},rejected);
  changed=records;changed.front().lo[0]=NAN;Reject([&]{h::NativeCartesianCoreMap::FromNativeLeaves(.25,radii,1,6,4,root,changed);},rejected);
  Reject([&]{h::NativeCartesianCoreMap::FromNativeLeaves(.25,radii,1,6,4,root,records,100);},rejected);
 }
 // Even complete native geometry must not evade the existing hierarchy cap.
 auto many=cubed_sphere::CompositeMesh(.25,{1},1,6,4,5);const auto too_many=h::NativeCartesianCoreMap::FromNativeLeaves(.25,{1},1,6,4,5,LeafRecords(many));
 Require(!multigrid::CompositeCoarseningPlan::Build(too_many).dense_supported,"coarse cap silently raised");
 std::cout<<"{\"cases\":"<<cases<<",\"matched_leaves\":"<<leaves<<",\"invalid_rejections\":"<<rejected<<",\"largest_terminal_cells\":"<<largest<<",\"coarse_cap\":1024,\"actual_Driver_run\":false,\"status\":\"passed\"}\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';code=1;}Kokkos::finalize();return code;}
