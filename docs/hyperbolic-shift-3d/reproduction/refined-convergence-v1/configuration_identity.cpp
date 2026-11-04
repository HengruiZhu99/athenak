// Host configuration only: no MPI initialization, Kokkos execution or evolution.
#include <iostream>
#include <iomanip>
#include "refined_configuration.hpp"
int main(int argc,char**argv){
 try{
  if(argc!=2)throw std::invalid_argument("configuration file required");
  for(int n:{12,14,16}){
   hyperbolic_reproduction::RefinedConfiguration c;
   c.Load(argv[1],hyperbolic_reproduction::ContinuousCoreMesh(n));
   std::cout<<std::setprecision(17)<<"{\"n\":"<<n<<",\"identity\":"
     <<std::quoted(c.identity+";amr_policy=smr")<<",\"k\":"<<c.inner_map_scale
     <<",\"inner_gamma\":"<<c.gauge.inner_gamma<<"}\n";
  }
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}
}
