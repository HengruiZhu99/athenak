#include <mpi.h>
#include <iomanip>
#include <iostream>
#include "globals.hpp"
#include "z4c/hyperboloidal/cartesian_amr.hpp"
#include "z4c/hyperboloidal/composite_dissipation.hpp"
namespace h=z4c::hyperboloidal;
int main(int argc,char**argv){
  MPI_Init(&argc,&argv);Kokkos::initialize(argc,argv);int result=0;
  MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);
  try {
    const int n=argc>1?std::stoi(argv[1]):12;h::CartesianAMR amr(n);
    std::vector<int> flags(8,0);flags[0]=1;amr.Regrid(flags);
    auto mesh=amr.Descriptor();
    // Numerical stencil audit uses the production six shells and overlap.
    cubed_sphere::CompositeDifference g(mesh,4,true);g.SetInnerTransitionLogical(true);
    for(int mode=0;mode<2;++mode){
      auto fun=[&](const Real*x){return mode?std::sin(17*x[0]+11*x[1]-13*x[2]):1+x[0]+2*x[1]+3*x[2]+x[0]*x[1]*x[2]+x[0]*x[0]*x[1]*x[1];};
      std::vector<Real> u(g.count,0);for(size_t p=0;p<g.active.size();++p)u[g.active[p]]=fun(g.geometry[p].x);
      g.Fill(u,1,[&](int,const auto&x){return fun(x.data());});
      Real first=0,second=0,mixed=0,advection=0,ko=0,constant=0;int counts[3]={};
      for(size_t p=0;p<g.nominal_count;++p){const auto c=g.cells[p];const auto&b=mesh.blocks[c.block];if(b.face>=0)continue;
        // Report the entire Cartesian region, including all internal faces,
        // edges and corners; the fixed spherical interpolation is also active.
        const auto*x=g.geometry[p].x;Real d[3],dd[3][3];g.Derivatives(u,0,p,d,dd);
        const Real k[3]={17,11,-13};Real ed[3]={1+x[1]*x[2]+2*x[0]*x[1]*x[1],2+x[0]*x[2]+2*x[1]*x[0]*x[0],3+x[0]*x[1]};
        Real edd[3][3]={{2*x[1]*x[1],x[2]+4*x[0]*x[1],x[1]},{x[2]+4*x[0]*x[1],2*x[0]*x[0],x[0]},{x[1],x[0],0}};
        if(mode)for(int a=0;a<3;++a){ed[a]=k[a]*std::cos(17*x[0]+11*x[1]-13*x[2]);for(int b=0;b<3;++b)edd[a][b]=-k[a]*k[b]*fun(x);}
        const int ijk[3]={c.i,c.j,c.k};int seam=0;for(int a=0;a<3;++a)if(ijk[a]<4||ijk[a]>=n-4)++seam;if(seam)++counts[seam-1];
        for(int a=0;a<3;++a){first=std::max(first,std::abs(d[a]-ed[a]));second=std::max(second,std::abs(dd[a][a]-edd[a][a]));
          for(int b=a+1;b<3;++b)mixed=std::max(mixed,std::abs(dd[a][b]-edd[a][b]));
          const int at=g.active[p],stride=g.Offset(a,1);const Real v=u[at];
          const Real bias=(-3*(u[at-stride]-v)+18*(u[at+stride]-v)-6*(u[at+2*stride]-v)+(u[at+3*stride]-v))/(12*((mesh.blocks[c.block].hi[a]-mesh.blocks[c.block].lo[a])/n));
          advection=std::max(advection,std::abs(bias-ed[a]));}
        using S=h::CompositeDissipation;Real numerical=0,exact=0;
        for(const auto&t:S::CartesianTerms(g,p)){Real sum=0,esum=0;
          for(int a=0;a<S::points;++a){const int index=t.point[a];const int block=index/(g.padded*g.padded*g.padded);
            const int flat=index%(g.padded*g.padded*g.padded);const int i=flat%g.padded-g.storage_halo,j=(flat/g.padded)%g.padded-g.storage_halo,kk=flat/(g.padded*g.padded)-g.storage_halo;
            const Real q[3]={(i+.5)/n,(j+.5)/n,(kk+.5)/n};Real xx[3];mesh.Forward(block,q,xx);
            sum+=S::Coefficient(a)*(u[index]-u[g.active[p]]);esum+=S::Coefficient(a)*(fun(xx)-fun(x));}
          numerical-=.5*n*t.target_coefficient*t.weight*sum/(S::normalization*g.volumes[p]);
          exact-=.5*n*t.target_coefficient*t.weight*esum/(S::normalization*g.volumes[p]);}
        ko=std::max(ko,std::abs(numerical-exact));
      }
      for(const auto&r:g.halos){Real sum=r.boundary_weight;for(const auto&e:r.entries)sum+=e.weight;constant=std::max(constant,std::abs(sum-1));}
      if(global_variable::my_rank==0)std::cout<<std::setprecision(17)<<"{\"n\":"<<n<<",\"mode\":"<<mode<<",\"first\":"<<first<<",\"second\":"<<second<<",\"mixed\":"<<mixed<<",\"advection\":"<<advection<<",\"ko_interpolation\":"<<ko<<",\"constant\":"<<constant<<",\"face_cells\":"<<counts[0]<<",\"edge_cells\":"<<counts[1]<<",\"corner_cells\":"<<counts[2]<<"}\n";
      if(!std::isfinite(first+second+mixed+advection+ko)||(!mode&&std::max({first,second,mixed,advection})>1e-7))throw std::runtime_error("mixed operator qualification failed");
    }
  }catch(const std::exception&e){std::cerr<<e.what()<<'\n';result=1;}
  MPI_Allreduce(MPI_IN_PLACE,&result,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);Kokkos::finalize();MPI_Finalize();return result;
}
