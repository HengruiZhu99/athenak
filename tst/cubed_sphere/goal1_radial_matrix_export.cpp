// Export actual native candidate coefficient matrices, with no evolution.
#include <iomanip>
#include <iostream>
#include "mesh/cubed_sphere/research_radial_sbp.hpp"
using namespace cubed_sphere;
struct Geo {Map map;int n;Point operator()(int i)const{Real q[3]={(i+.5)/n,1.,.4};return map.Geometry(0,q);}};
struct Basis {int j;Real operator()(int i,int,int)const{return i==j?1:0;}};
int main(int argc,char**argv){Kokkos::initialize(argc,argv);std::cout<<std::setprecision(17);
 for(int n:{24,48,96})for(Real stretch:{0.,2.8592697712828787}){
  auto op=MakeResearchRadialSBP(n,4,1./n);auto d=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),op.d1);
  auto h=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),op.norm);Geo geo{{.25,.98,stretch},n};
  std::cout<<"{\"nodes\":"<<n<<",\"stretch\":"<<stretch<<",\"H\":[";
  for(int i=0;i<n;++i)std::cout<<(i?",":"")<<h(i)*ResearchLogicalVolume(geo(i))/n;
  std::cout<<"],\"D\":[";
  // Export the effective anchored sampling operator, not just stored rows.
  for(int i=0;i<n;++i){std::cout<<(i?",":"")<<"[";Real sum=0;for(int j=0;j<n;++j)sum+=d(i,j);
   for(int j=0;j<n;++j)std::cout<<(j?",":"")<<(d(i,j)-(i==j?sum:0));std::cout<<"]";}
  std::cout<<"],\"Q\":[";
  for(int i=0;i<n;++i){std::cout<<(i?",":"")<<"[";for(int j=0;j<n;++j)std::cout<<(j?",":"")<<ResearchRadialDissipation(Basis{j},geo,i,op,1./n,.1);std::cout<<"]";}
  std::cout<<"]}\n";
 }
 Kokkos::finalize();
}
