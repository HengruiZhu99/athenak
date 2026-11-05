// Analysis mapping oracle. Uses native geometry; no field allocation/evolution.
#include <iostream>
#include <iomanip>
#include "refined_configuration.hpp"
int main(int argc,char**argv){
 try{
  if(argc!=5)throw std::invalid_argument("N config leaves points required");
  const int n=std::stoi(argv[1]);
  hyperbolic_reproduction::RefinedConfiguration cfg;
  cfg.Load(argv[2],hyperbolic_reproduction::ContinuousCoreMesh(n));
  auto mesh=cfg.Mesh(n);const auto original=mesh.blocks;mesh.blocks.clear();
  std::ifstream leaves(argv[3]);int lx,ly,lz,level;
  const int root_level=int(std::log2(cfg.core_subdivisions));
  while(leaves>>lx>>ly>>lz>>level){
   cubed_sphere::CompositeBlock b;b.refinement=level-root_level;
   const int loc[3]={lx,ly,lz};
   const Real width=2*cfg.core_half_width/(cfg.core_subdivisions*(1<<b.refinement));
   for(int a=0;a<3;++a){b.lo[a]=-cfg.core_half_width+loc[a]*width;b.hi[a]=-cfg.core_half_width+(loc[a]+1)*width;}
   mesh.blocks.push_back(b);
  }
  for(const auto&b:original)if(b.face>=0)mesh.blocks.push_back(b);
  mesh.amr_cartesian_core=true;
  std::cout<<std::setprecision(17);
  for(size_t m=0;m<mesh.blocks.size();++m){const auto&b=mesh.blocks[m];
   std::cout<<"B "<<m<<' '<<b.face<<' '<<b.layer<<' '<<b.refinement;
   for(auto x:b.lo)std::cout<<' '<<x;for(auto x:b.hi)std::cout<<' '<<x;std::cout<<'\n';}
  std::ifstream points(argv[4]);Real x[3];int index=0;
  while(points>>x[0]>>x[1]>>x[2]){
   const int owner=mesh.Owner(x);if(owner<0)throw std::runtime_error("outside mesh");
   Real q[3],back[3];mesh.Inverse(owner,x,q);mesh.Forward(owner,q,back);
   std::cout<<"P "<<index++<<' '<<owner;
   for(auto v:q)std::cout<<' '<<v;for(auto v:back)std::cout<<' '<<v;std::cout<<'\n';
  }
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}
}
