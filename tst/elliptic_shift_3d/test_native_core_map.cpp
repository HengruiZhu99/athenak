// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include <iostream>
#include "z4c/hyperboloidal/native_core_map.hpp"
namespace h=z4c::hyperboloidal;
void Check(bool okay,const char *message){if(!okay)throw std::runtime_error(message);}
template<class F>void Reject(F action){bool caught=false;try{action();}catch(const std::exception&){caught=true;}Check(caught,"invalid/stale native core map accepted");}
std::vector<h::NativeCartesianBlock> Records(const cubed_sphere::CompositeMesh &mesh){
  std::vector<h::NativeCartesianBlock> out;
  for(const auto &b:mesh.blocks)if(b.face<0){h::NativeCartesianBlock q;q.refinement=b.refinement;
    for(int a=0;a<3;++a){q.lo[a]=b.lo[a];q.hi[a]=b.hi[a];}out.push_back(q);}
  // Native pack order deliberately differs from composite block order. Refined
  // children also follow shell blocks in the CompositeMesh descriptor.
  for(size_t m=0;m<out.size();++m)out[m].native=out.size()-1-m;
  std::rotate(out.begin(),out.begin()+out.size()/2,out.end());return out;
}
int main(int argc,char **argv){Kokkos::initialize(argc,argv);int status=0;try{
  size_t matched=0,refined_nonprefix=0;int rejected=0;
  for(int bulk:{1,2})for(bool refined:{false,true}){
    cubed_sphere::CompositeMesh mesh(.25,{.65,1},1,6,3,bulk);
    if(refined)mesh.RefineCartesian(0);
    const auto records=Records(mesh);h::NativeCartesianCoreMap map(mesh,records,6,3);
    auto reordered=records;std::reverse(reordered.begin(),reordered.end());map.ValidateCurrent(mesh,reordered);
    for(const auto &entry:map.matches){const auto &b=mesh.blocks.at(entry.composite);
      const auto found=std::find_if(records.begin(),records.end(),[&](const auto &x){return x.native==entry.native;});
      Check(found!=records.end()&&b.face<0&&found->refinement==b.refinement,"field map selects wrong patch");
      for(int a=0;a<3;++a)Check(found->lo[a]==b.lo[a]&&found->hi[a]==b.hi[a],"field map changes physical bounds");
      for(int k=0;k<6;++k)for(int j=0;j<6;++j)for(int i=0;i<6;++i){
        const Real local[3]={(i+.5)/6,(j+.5)/6,(k+.5)/6};Real x[3];mesh.Forward(entry.composite,local,x);
        for(int a=0;a<3;++a)Check(std::abs(x[a]-(found->lo[a]+local[a]*(found->hi[a]-found->lo[a])))<1e-15,"native/composite active centers mismatch");
      }
      ++matched;if(refined&&entry.composite>=records.size())++refined_nonprefix;
    }
    auto reject=[&](auto f){Reject(f);++rejected;};
    reject([&]{h::NativeCartesianCoreMap bad(mesh,records,8,3);});
    reject([&]{h::NativeCartesianCoreMap bad(mesh,records,6,4);});
    auto changed=records;changed[0].hi[0]+=.001;
    reject([&]{h::NativeCartesianCoreMap bad(mesh,changed,6,3);});
    reject([&]{map.ValidateCurrent(mesh,changed);});
    changed=records;changed[0].refinement++;
    reject([&]{h::NativeCartesianCoreMap bad(mesh,changed,6,3);});
    changed=records;changed[0].lo[0]=std::numeric_limits<Real>::quiet_NaN();
    reject([&]{h::NativeCartesianCoreMap bad(mesh,changed,6,3);});
    changed=records;changed.pop_back();
    reject([&]{h::NativeCartesianCoreMap bad(mesh,changed,6,3);});
    auto revision=mesh;++revision.revision;
    reject([&]{map.ValidateCurrent(revision,records);});
    auto shell=mesh;shell.blocks.back().hi[1]-=.001;
    reject([&]{map.ValidateCurrent(shell,records);});
    if(records.size()>1){changed=records;changed[1].native=changed[0].native;
      reject([&]{h::NativeCartesianCoreMap bad(mesh,changed,6,3);});
      changed=records;std::swap(changed[0].native,changed[1].native);
      reject([&]{map.ValidateCurrent(mesh,changed);});}
    auto overlap=mesh;const auto first=map.matches.front().composite;overlap.blocks.push_back(overlap.blocks[first]);
    reject([&]{h::NativeCartesianCoreMap bad(overlap,Records(overlap),6,3);});
    auto hole=mesh;hole.blocks[first].hi[0]-=.001;
    reject([&]{h::NativeCartesianCoreMap bad(hole,Records(hole),6,3);});
  }
  Check(refined_nonprefix>0,"refined non-prefix Cartesian mapping untested");
  std::cout<<"{\"matched_leaves\":"<<matched<<",\"refined_nonprefix_leaves\":"<<refined_nonprefix
           <<",\"rejections\":"<<rejected<<",\"native_runtime_integrated\":false,\"status\":\"passed\"}\n";
}catch(const std::exception &error){std::cerr<<error.what()<<'\n';status=1;}Kokkos::finalize();return status;}
