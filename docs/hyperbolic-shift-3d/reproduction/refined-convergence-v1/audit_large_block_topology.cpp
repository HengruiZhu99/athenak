// Construct the actual AthenaK SMR tree, but no coupled fields or evolution.
#include <mpi.h>
#include <iostream>
#include <iomanip>
#include "refined_configuration.hpp"
#include "z4c/hyperboloidal/cartesian_amr.hpp"
int main(int argc,char**argv){
 MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);
 MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);Kokkos::initialize(argc,argv);int status=0;
 try{
  if(argc!=4)throw std::invalid_argument("N config output.json");
  int n=std::stoi(argv[1]);hyperbolic_reproduction::RefinedConfiguration cfg;
  cfg.Load(argv[2],hyperbolic_reproduction::ContinuousCoreMesh(n));cfg.CheckCollective();
  z4c::hyperboloidal::CartesianAMR amr(n,cfg.core_subdivisions,
    std::max(32,512/global_variable::nranks+8),cfg.smr_levels+1,
    cfg.shell_radii,cfg.core_half_width,cfg.shell_inner_radius,cfg.shell_angular_subdivisions);
  for(int level=0;level<cfg.smr_levels;++level){
    for(int m=0;m<amr.mesh->nmb_total;++m)amr.mesh->pmr->ncyc_since_ref(m)=amr.mesh->pmr->refinement_interval;
    if(!amr.RefineBox(cfg.smr_half_widths))throw std::runtime_error("SMR level not refined");
  }
  const size_t retired=amr.SealTopology();const auto mesh=amr.Descriptor();
  size_t active=0;std::map<int,size_t> levels;
  for(const auto&b:mesh.blocks){int widths[3];const int overlap=b.layer<=0?3:0;
    for(int a=0;a<3;++a){
      const int lo=(b.face<0&&b.lo[a]>-mesh.half_width+1e-14)||(mesh.spherical_inner_boundary&&b.layer==0&&a==0)?0:-overlap;
      const int hi=b.face<0&&b.hi[a]<mesh.half_width-1e-14?n:n+overlap;widths[a]=hi-lo;}
    active+=size_t(widths[0])*widths[1]*widths[2];if(b.face<0)++levels[b.refinement];}
  if(global_variable::my_rank==0){std::ofstream out(argv[3]);out<<std::setprecision(17)
   <<"{\"n\":"<<n<<",\"core_leaves\":"<<amr.mesh->nmb_total<<",\"blocks\":"<<mesh.blocks.size()
   <<",\"active_cells\":"<<active<<",\"nominal_cells\":"<<mesh.blocks.size()*n*n*n
   <<",\"retired_bytes_per_rank\":"<<retired<<",\"identity\":"<<std::quoted(cfg.identity)
   <<",\"core_level_counts\":{";bool first=true;
   for(auto e:levels){if(!first)out<<',';first=false;out<<'"'<<e.first<<"\":"<<e.second;}
   out<<"},\"supports\":[";first=true;
   for(const std::array<Real,3>&x:std::vector<std::array<Real,3>>{{0,0,0},{.0337229,0,0},{-.0337229,0,0},{.09,0,0},{0,.045,0}}){
    const auto s=amr.SupportAt(x);if(!first)out<<',';first=false;
    out<<"{\"x\":["<<x[0]<<','<<x[1]<<','<<x[2]<<"],\"finest_spacing\":"<<s.spacing<<",\"support_radius\":"<<s.radius<<",\"level\":"<<s.level<<'}';}
   out<<"],\"core_blocks\":[";first=true;
   for(const auto&b:mesh.blocks)if(b.face<0){if(!first)out<<',';first=false;out<<"{\"level\":"<<b.refinement<<",\"lo\":["<<b.lo[0]<<','<<b.lo[1]<<','<<b.lo[2]<<"],\"hi\":["<<b.hi[0]<<','<<b.hi[1]<<','<<b.hi[2]<<"]}";}
   out<<"]}\n";if(!out)throw std::runtime_error("write failed");
  }
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';status=2;}
 Kokkos::finalize();MPI_Finalize();return status;
}
