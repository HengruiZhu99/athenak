// Fixed topology can retire native transfer buffers without changing its map.
#include <mpi.h>
#include <fstream>
#include <iostream>
#include "z4c/hyperboloidal/cartesian_amr.hpp"
int main(int argc,char**argv){
 MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);
 MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);Kokkos::initialize(argc,argv);
 int code=0;
 try{
  if(argc!=2)throw std::invalid_argument("output.json");
  auto check=[](bool ok,const char*why){if(!ok)throw std::runtime_error(why);};
  z4c::hyperboloidal::CartesianAMR a(12,2,96,2);
  Kokkos::deep_copy(a.state,Real(1.25));Kokkos::deep_copy(a.compensation,Real(-.125));
  check(a.RefineBox({.025,.025,.025}),"native tree did not refine before sealing");
  for(Real x:a.Gather())check(std::abs(x-1.25)<1e-13,"native transfer changed constant state");
  for(Real x:a.Gather(true))check(std::abs(x+.125)<1e-13,"native transfer changed compensation");
  const auto before=a.Descriptor();const auto support=a.SupportAt({0,0,0});
  const size_t bytes=(a.state.span()+a.coarse.span()+a.compensation.span()+a.coarse_compensation.span())*sizeof(Real);
  check(bytes>0 && a.SealTopology()==bytes,"wrong retired allocation count");
  check(a.state.span()==0 && a.compensation.span()==0 && a.coarse.span()==0 && a.coarse_compensation.span()==0,"transfer allocations retained");
  check(a.SealTopology()==0,"sealing is not idempotent");
  const auto after=a.Descriptor();const auto now=a.SupportAt({0,0,0});
  check(before.blocks.size()==after.blocks.size() && before.revision==after.revision,"sealed topology changed");
  for(size_t i=0;i<before.blocks.size();++i)for(int j=0;j<3;++j)
    check(before.blocks[i].lo[j]==after.blocks[i].lo[j] && before.blocks[i].hi[j]==after.blocks[i].hi[j],"sealed block bounds changed");
  check(support.radius==now.radius && support.spacing==now.spacing && support.level==now.level,"support changed");
  int rejected=0;
  try{a.Regrid(std::vector<int>(a.mesh->nmb_total,0));}catch(const std::logic_error&){++rejected;}
  try{a.RefineBox({.025,.025,.025});}catch(const std::logic_error&){++rejected;}
  try{a.RefineAround({0,0,0},.025,2);}catch(const std::logic_error&){++rejected;}
  try{a.Gather();}catch(const std::logic_error&){++rejected;}
  check(rejected==4,"sealed transfer entry point accepted");
  if(global_variable::my_rank==0){std::ofstream out(argv[1]);out<<"{\"passed\":true,\"retired_bytes_per_rank\":"<<bytes<<",\"leaves\":"<<a.mesh->nmb_total<<",\"rejected_transfer_calls\":"<<rejected<<"}\n";}
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';code=2;}
 Kokkos::finalize();MPI_Finalize();return code;
}
