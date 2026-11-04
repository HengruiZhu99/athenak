#include <mpi.h>
#include <iostream>
#include <iomanip>
#include "z4c/hyperboloidal/cartesian_amr.hpp"
#include "z4c/hyperboloidal/hyperbolic_composite.hpp"
namespace h=z4c::hyperboloidal;
void require(bool ok,const char*message){if(!ok)throw std::runtime_error(message);}
template<class Rows>void halo_equal(const Rows&a,const Rows&b){
  require(a.size()==b.size(),"halo row count");
  for(size_t r=0;r<a.size();++r){
    require(a[r].target==b[r].target&&a[r].boundary_weight==b[r].boundary_weight&&a[r].boundary==b[r].boundary,"halo target/boundary");
    require(a[r].entries.size()==b[r].entries.size(),"halo donor count");
    for(size_t j=0;j<a[r].entries.size();++j)require(a[r].entries[j].point==b[r].entries[j].point&&a[r].entries[j].weight==b[r].entries[j].weight,"halo donor/order/weight");
  }
}
template<class Terms>void terms_equal(const Terms&a,const Terms&b){
  require(a.size()==b.size(),"scri/radial term count");
  for(size_t i=0;i<a.size();++i){require(a[i].point==b[i].point&&a[i].boundary==b[i].boundary,"scri/radial donor");
    for(int j=0;j<3;++j){require(a[i].d[j]==b[i].d[j],"scri/radial first derivative");
      for(int k=0;k<3;++k)require(a[i].dd[j][k]==b[i].dd[j][k],"scri/radial second derivative");}}
}
int main(int argc,char**argv){
  MPI_Init(&argc,&argv);Kokkos::initialize(argc,argv);int bad=0;
  MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);
  try{
    h::CartesianAMR amr(12,2,80);
    auto check=[&](int leaves){
      require(amr.mesh->nmb_total==leaves,"unexpected topology fixture");auto mesh=amr.Descriptor();
      using M=h::HyperbolicComposite;
      const double start=MPI_Wtime();
      M old(mesh,false,true,2,M::NormalDerivative::sampled_numerator,M::RadialClosure::donor_seven);
      old.grid.SetInnerTransitionLogical(true);const double old_end=MPI_Wtime();
      M next(mesh,false,true,2,M::NormalDerivative::sampled_numerator,M::RadialClosure::donor_seven,true);
      const double new_end=MPI_Wtime();
      const auto&a=old.grid;const auto&b=next.grid;
      require(a.active==b.active&&a.inverse_active==b.inverse_active&&a.volumes==b.volumes,"active ordering/volume");
      for(size_t p=0;p<a.cells.size();++p){const auto&c=a.cells[p];const auto&d=b.cells[p];
        require(c.block==d.block&&c.i==d.i&&c.j==d.j&&c.k==d.k,"logical cell ordering");
        for(int i=0;i<3;++i){require(a.geometry[p].x[i]==b.geometry[p].x[i],"coordinate");
          for(int j=0;j<3;++j){require(a.geometry[p].J[i][j]==b.geometry[p].J[i][j],"Jacobian");
            for(int k=0;k<3;++k)require(a.geometry[p].H[i][j][k]==b.geometry[p].H[i][j][k],"Hessian");}}}
      halo_equal(a.halos,b.halos);halo_equal(a.extrapolated_halos,b.extrapolated_halos);
      require(old.scri.base==next.scri.base&&old.scri.values==next.scri.values,"scri ordering/state");
      require(old.scri.rows.size()==next.scri.rows.size(),"scri rows");
      for(size_t i=0;i<old.scri.rows.size();++i){require(old.scri.rows[i].radial_spacing==next.scri.rows[i].radial_spacing,"scri spacing");terms_equal(old.scri.rows[i].terms,next.scri.rows[i].terms);}
      require(old.radial_plan->row_index==next.radial_plan->row_index&&old.radial_plan->rows.size()==next.radial_plan->rows.size(),"radial rows");
      for(size_t i=0;i<old.radial_plan->rows.size();++i)terms_equal(old.radial_plan->rows[i],next.radial_plan->rows[i]);
      if(!global_variable::my_rank)std::cout<<std::setprecision(17)<<"{\"leaves\":"<<leaves<<",\"passed\":true,\"old_seconds\":"<<old_end-start<<",\"new_seconds\":"<<new_end-old_end<<",\"halo_rows\":"<<a.halos.size()<<"}\n";
    };
    check(8);std::vector<int> flags(8);flags[0]=1;amr.Regrid(flags);check(15);
    flags.assign(15,0);flags[7]=1;amr.Regrid(flags);check(71);
    flags.assign(amr.mesh->nmb_total,0);for(int m=0;m<amr.mesh->nmb_total;++m)if(amr.mesh->lloc_eachmb[m].level>amr.mesh->root_level)flags[m]=-1;
    amr.Regrid(flags);require(amr.mesh->nmb_total==15,"coarsening fixture");
    flags.assign(15,0);for(int m=0;m<15;++m)if(amr.mesh->lloc_eachmb[m].level==amr.mesh->root_level){flags[m]=1;break;}
    amr.Regrid(flags);check(22);
  }catch(const std::exception&e){std::cerr<<e.what()<<'\n';bad=1;}
  MPI_Allreduce(MPI_IN_PLACE,&bad,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);Kokkos::finalize();MPI_Finalize();return bad;
}
