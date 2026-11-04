// Native static octree policy and read-only finest-support checks.
#include <mpi.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include "refined_configuration.hpp"
#include "z4c/hyperboloidal/cartesian_amr.hpp"
namespace h=z4c::hyperboloidal;
void Check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
int main(int argc,char**argv){
 MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);
 MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);Kokkos::initialize(argc,argv);int code=0;
 try{
  if(argc!=3 && argc!=4)throw std::invalid_argument("config output [config-only]");
  hyperbolic_reproduction::RefinedConfiguration cfg;
  cfg.Load(argv[1],hyperbolic_reproduction::ContinuousCoreMesh(12));cfg.CheckCollective();
  if(argc==4){Check(std::string(argv[3])=="config-only","unknown option");}
  else{
   Check(cfg.smr_box && cfg.smr_levels==3,"expected qualified tube input");
   h::CartesianAMR amr(12,cfg.core_subdivisions,std::max(32,1024/global_variable::nranks+8),
      cfg.smr_levels+1,cfg.shell_radii,cfg.core_half_width,cfg.shell_inner_radius);
   for(int level=0;level<cfg.smr_levels;++level){
    for(int m=0;m<amr.mesh->nmb_total;++m)
      amr.mesh->pmr->ncyc_since_ref(m)=amr.mesh->pmr->refinement_interval;
    Check(amr.RefineBox(cfg.smr_half_widths),"static box did not refine");
   }
   Check(amr.mesh->nmb_total==400,"qualified symmetric tube changed topology");
   const std::array<Real,3> points[]={{0,0,0},{.0337229,0,0},{-.0337229,0,0},
                                    {.09,0,0},{0,.045,0},{.15,0,0},{.41,0,0}};
   const Real radii[]={.05,.05,.05,.01,.005,0,0};
   std::vector<h::CartesianAMR::FinestSupport> supports;
   for(int k=0;k<7;++k){const auto s=amr.SupportAt(points[k]);
    Check(std::abs(s.radius-radii[k])<1e-14,"finest-region distance incorrect");
    Check(std::abs(s.spacing-(.025/12))<1e-14 && s.level==3,"finest scale incorrect");
    Check(std::abs(s.core_margin-(.4-std::max({std::abs(points[k][0]),std::abs(points[k][1]),std::abs(points[k][2])})))<1e-14,"core margin incorrect");
    supports.push_back(s);
   }
   bool rejected=false;try{amr.RefineBox({0,.025,.025});}catch(const std::invalid_argument&){rejected=true;}
   Check(rejected,"zero-width refinement accepted");
   const auto mesh=amr.Descriptor();
   if(global_variable::my_rank==0){std::ofstream out(argv[2]);out<<std::setprecision(17)
    <<"{\"passed\":true,\"leaves\":"<<amr.mesh->nmb_total<<",\"points\":[";
    for(int k=0;k<7;++k){if(k)out<<',';out<<"{\"x\":["<<points[k][0]<<','<<points[k][1]<<','<<points[k][2]
      <<"],\"radius\":"<<supports[k].radius<<",\"spacing\":"<<supports[k].spacing<<'}';}
    out<<"],\"blocks\":[";bool comma=false;
    for(const auto&b:mesh.blocks)if(b.face<0){if(comma)out<<',';comma=true;
      out<<"{\"level\":"<<b.refinement<<",\"lo\":["<<b.lo[0]<<','<<b.lo[1]<<','<<b.lo[2]
         <<"],\"hi\":["<<b.hi[0]<<','<<b.hi[1]<<','<<b.hi[2]<<"]}";}
    out<<"]}\n";
   }
  }
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';code=2;}
 Kokkos::finalize();MPI_Finalize();return code;
}
