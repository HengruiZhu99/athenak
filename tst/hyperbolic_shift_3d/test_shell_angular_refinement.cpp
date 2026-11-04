// Optional interior shell subdivision reuses the existing physical donor maps.
#include <mpi.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include "refined_configuration.hpp"
#include "mesh/cubed_sphere/composite_difference.hpp"
#include "z4c/hyperboloidal/cartesian_amr.hpp"
void Check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
int main(int argc,char**argv){
 MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);
 MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);Kokkos::initialize(argc,argv);int code=0;
 try{
  if(argc!=3 && argc!=4)throw std::invalid_argument("config output [config-only]");
  hyperbolic_reproduction::RefinedConfiguration cfg;
  cfg.Load(argv[1],hyperbolic_reproduction::ContinuousCoreMesh(12));cfg.CheckCollective();
  if(argc==4)Check(std::string(argv[3])=="config-only","unknown mode");
  else{
   const auto refined=cfg.Mesh(12);auto original=refined;
   cubed_sphere::CompositeMesh base(cfg.core_half_width,cfg.shell_radii,1,12,3,cfg.core_subdivisions,cfg.shell_inner_radius);
   original=base;auto ones=base;ones.SubdivideShellAngles(std::vector<int>(cfg.shell_radii.size(),1));
   Check(ones.blocks.size()==base.blocks.size(),"all-one topology changed");
   for(size_t m=0;m<base.blocks.size();++m)for(int a=0;a<3;++a)
    Check(ones.blocks[m].lo[a]==base.blocks[m].lo[a] && ones.blocks[m].hi[a]==base.blocks[m].hi[a],"all-one bounds changed");
   size_t core=0,shell=0,surface=0;double maxmap=0,maxinverse=0,maxjac=0,maxhess=0;
   for(size_t m=0;m<refined.blocks.size();++m){const auto&b=refined.blocks[m];
    if(b.face<0){++core;for(int a=0;a<3;++a)Check(b.lo[a]==base.blocks[m].lo[a] && b.hi[a]==base.blocks[m].hi[a],"core changed");continue;}
    ++shell;if(b.layer==int(refined.radii.size())-1)++surface;
    const int parent=cfg.core_subdivisions*cfg.core_subdivisions*cfg.core_subdivisions+6*b.layer+b.face;
    for(double t:{.13,.51,.87}){double q[3]={t,.29,.63},x[3],oldq[3],back[3];refined.Forward(m,q,x);base.Inverse(parent,x,oldq);refined.Inverse(m,x,back);
     auto a=refined.Geometry(m,q),z=base.Geometry(parent,oldq);
     for(int i=0;i<3;++i){maxmap=std::max(maxmap,std::abs(a.x[i]-z.x[i]));maxinverse=std::max(maxinverse,std::abs(back[i]-q[i]));
      for(int A=0;A<3;++A){const double scale=A==0?1:cfg.shell_angular_subdivisions[b.layer];
       maxjac=std::max(maxjac,std::abs(a.J[A][i]-scale*z.J[A][i])/(1+std::abs(a.J[A][i])));
       for(int j=0;j<3;++j)maxhess=std::max(maxhess,std::abs(a.H[A][i][j]-scale*z.H[A][i][j])/(1+std::abs(a.H[A][i][j])));
      }
     }
     if(b.layer>=7)Check(refined.Owner(x)==int(m),"subdivided donor ownership wrong");
    }
   }
   Check(core==64 && shell==288 && surface==6,"candidate block counts changed");
   Check(maxmap<1e-14 && maxinverse<2e-12 && maxjac<2e-12 && maxhess<2e-12,"subdivided map identity failed");
   bool rejected=false;try{auto twice=refined;twice.SubdivideShellAngles(cfg.shell_angular_subdivisions);}catch(const std::invalid_argument&){rejected=true;}Check(rejected,"double subdivision accepted");
   double polynomial_error=0,radial_distance=0;
   {cubed_sphere::CompositeMesh probe(.1,{.4,.7,1},1,12,3,1,.08);
    probe.SubdivideShellAngles({1,2,1});cubed_sphere::CompositeDifference grid(probe,4,true);
    Check(grid.mixed_shell_angles,"mixed angular interfaces not detected");
    for(double r:{.399999,.400001,.699999,.700001}){
      const double norm=std::sqrt(1+.3*.3+.2*.2),x[3]={r/norm,.3*r/norm,.2*r/norm};
      auto terms=grid.Interpolate(x,6);bool left=false,right=false;
      auto value=[](double radius,double u,double v){return std::pow(radius,5)+std::pow(u,5)+.3*std::pow(v,5)+std::pow(radius,3)*u*u*v*v;};
      const double target=value(r,4*std::atan(.3)/std::acos(-1.),4*std::atan(.2)/std::acos(-1.));
      double sum=0;
      for(const auto&e:terms){int p=grid.inverse_active.at(e.point);Check(p>=0,"unevolved angular-interface donor");
        const auto& node=grid.geometry[p];double radius=std::sqrt(node.x[0]*node.x[0]+node.x[1]*node.x[1]+node.x[2]*node.x[2]);
        sum+=e.weight*value(radius,4*std::atan(node.x[1]/node.x[0])/std::acos(-1.),4*std::atan(node.x[2]/node.x[0])/std::acos(-1.));
        const int layer=probe.blocks[grid.cells[p].block].layer;
        left=left||(layer==(r<.5?0:1));right=right||(layer==(r<.5?1:2));
        radial_distance=std::max(radial_distance,std::abs(radius-r));
      }
      Check(left&&right,"radial donor chain skipped an adjacent angular coarse/fine layer");
      polynomial_error=std::max(polynomial_error,std::abs(sum-target));
    }
    Check(polynomial_error<2e-13 && radial_distance<.07,"mixed angular interpolation polynomial/radius check failed");
   }
   z4c::hyperboloidal::CartesianAMR amr(12,4,72,1,cfg.shell_radii,cfg.core_half_width,cfg.shell_inner_radius,cfg.shell_angular_subdivisions);
   const auto descriptor=amr.Descriptor();Check(descriptor.blocks.size()==refined.blocks.size(),"AMR descriptor drops angular blocks");
   for(size_t m=0;m<refined.blocks.size();++m)if(refined.blocks[m].face>=0)for(int a=0;a<3;++a)
    Check(descriptor.blocks[m].lo[a]==refined.blocks[m].lo[a] && descriptor.blocks[m].hi[a]==refined.blocks[m].hi[a],"AMR angular bounds wrong");
   if(global_variable::my_rank==0){std::ofstream out(argv[2]);out<<std::setprecision(17)<<"{\"passed\":true,\"core_blocks\":"<<core<<",\"shell_blocks\":"<<shell<<",\"surface_blocks\":"<<surface<<",\"map_error\":"<<maxmap<<",\"inverse_error\":"<<maxinverse<<",\"jacobian_error\":"<<maxjac<<",\"hessian_error\":"<<maxhess<<",\"mixed_polynomial_error\":"<<polynomial_error<<",\"max_radial_donor_distance\":"<<radial_distance<<"}\n";}
  }
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';code=2;}
 Kokkos::finalize();MPI_Finalize();return code;
}
