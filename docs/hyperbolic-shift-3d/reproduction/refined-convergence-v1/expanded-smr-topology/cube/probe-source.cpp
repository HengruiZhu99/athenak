// Actual AthenaK refinement lifecycle; no evolution, no analytic field installation.
#include <fstream>
#include <iostream>
#include <iomanip>
#include "refined_configuration.hpp"
#include "z4c/hyperboloidal/cartesian_amr.hpp"
namespace h=z4c::hyperboloidal;
int main(int argc,char**argv){
 MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);Kokkos::initialize(argc,argv);int code=0;
 try{
  if(argc!=4)throw std::runtime_error("config radius output");
  const Real radius=std::stod(argv[2]);if(!(radius>.025&&radius<=.1))throw std::runtime_error("outside bounded radius probe");
  hyperbolic_reproduction::RefinedConfiguration cfg;cfg.Load(argv[1],hyperbolic_reproduction::ContinuousCoreMesh(16));
  std::ifstream exists(argv[3]);if(exists.good())throw std::runtime_error("existing output");
  h::CartesianAMR amr(16,cfg.core_subdivisions,std::max(32,2048/global_variable::nranks+8),cfg.smr_levels+1,cfg.shell_radii,cfg.core_half_width,cfg.shell_inner_radius);
  for(int level=0;level<cfg.smr_levels;++level){
   for(int m=0;m<amr.mesh->nmb_total;++m)amr.mesh->pmr->ncyc_since_ref(m)=amr.mesh->pmr->refinement_interval;
   if(!amr.RefineAround({0,0,0},radius,0))throw std::runtime_error("refinement did not advance");
   if(global_variable::my_rank==0)std::cout<<"level "<<level+1<<" leaves "<<amr.mesh->nmb_total<<std::endl;
  }
  const auto d=amr.Descriptor();
  if(global_variable::my_rank==0){std::ofstream out(argv[3]);out<<std::setprecision(17)<<"{\"n\":16,\"smr_levels\":"<<cfg.smr_levels<<",\"probe_radius\":"<<radius<<",\"leaves\":"<<amr.mesh->nmb_total<<",\"k\":"<<cfg.inner_map_scale<<",\"blocks\":[";
   bool comma=false;for(const auto&b:d.blocks){if(b.face>=0)continue;if(comma)out<<',';comma=true;out<<"{\"level\":"<<b.refinement<<",\"lo\":["<<b.lo[0]<<','<<b.lo[1]<<','<<b.lo[2]<<"],\"hi\":["<<b.hi[0]<<','<<b.hi[1]<<','<<b.hi[2]<<"]}";}
   out<<"]}\n";}
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';code=2;}
 Kokkos::finalize();MPI_Finalize();return code;
}
