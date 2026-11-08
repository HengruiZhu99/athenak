// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Same-object restart invalidation complements the actual-process restart gate.
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
#include "globals.hpp"
#include "driver/driver.hpp"
#include "mesh/mesh.hpp"
#include "outputs/outputs.hpp"
#include "pgen/pgen.hpp"
#include "z4c_hyperboloidal/diagnostics.hpp"
#include "z4c_hyperboloidal/restart.hpp"
#include "z4c_hyperboloidal/z4c_hyperboloidal.hpp"

namespace h=z4c_hyperboloidal;
void Require(bool yes,const char *message) {if (!yes)throw std::runtime_error(message);}
std::vector<Real> ConstraintValues(h::Diagnostics &diagnostics) {
  const auto host=Kokkos::create_mirror_view_and_copy(HostMemSpace(),diagnostics.Points());
  std::vector<Real> values;values.reserve(8*host.size());
  for (std::size_t q=0;q<host.size();++q) {
    const auto &p=host(q);Require(p.status==h::PointStatus::success,"invalid constraint point");
    for (Real v:{p.chi,p.lapse,p.theta,p.constraints.hamiltonian,
                 p.constraints.momentum_physical_norm2,p.constraints.z4.z_physical_norm2,
                 p.constraints.null_residual,p.constraints.physical_trace})values.push_back(v);
  }
  return values;
}
int main(int argc,char **argv) {
  MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);Kokkos::initialize(argc,argv);
  try {{
    Require(argc==2,"usage: history_restart_cache fixture");
    ParameterInput pin;IOWrapper file;file.Open(argv[1],IOWrapper::FileMode::read);
    pin.LoadFromFile(file);file.Close();
    pin.SetString("time","evolution","dynamic");
    pin.SetString("time","integrator","rk4_classical");
    pin.SetInteger("time","nlim",0);pin.SetReal("time","tlim",1);
    Mesh mesh(&pin);mesh.BuildTreeFromScratch(&pin);mesh.AddCoordinatesAndPhysics(&pin);
    mesh.pgen=std::make_unique<ProblemGenerator>(&pin,&mesh);
    Kokkos::Timer timer;Driver driver(&pin,&mesh,3600,&timer);Outputs outputs(&pin,&mesh);
    Require(outputs.pout_list.empty(),"cache fixture unexpectedly enrolled output");
    driver.Initialize(&mesh,&pin,&outputs,false);
    auto &p=*mesh.pmb_pack->pz4c_hyperboloidal;auto &diagnostics=p.GetDiagnostics();
    HostArray2D<Real> original;h::restart::Pack(p,original);
    diagnostics.Refresh();const auto reference=ConstraintValues(diagnostics);
    const auto first=p.points.h_view(0);const auto state=p.u0;
    if (global_variable::my_rank==0) {
      Kokkos::parallel_for("restart cache test mutate theta",Kokkos::RangePolicy<>(0,1),
        KOKKOS_LAMBDA(int) {state(first.m,h::Fields::theta,first.k,first.j,first.i)+=1e-8;});
      Kokkos::fence();p.InvalidateDiagnostics();
    }
    diagnostics.Refresh();const auto modified=ConstraintValues(diagnostics);
    if (global_variable::my_rank==0)
      Require(modified[2]==reference[2]+1e-8,"mutation did not enter cached constraints");
    const auto before=diagnostics.GetStatistics();const auto version=p.StateVersion();
    h::restart::Unpack(p,original);
    Require(p.StateVersion()==version+1,"Unpack failed to invalidate diagnostics");
    diagnostics.Refresh();diagnostics.Refresh();
    Require(diagnostics.GetStatistics().refreshes==before.refreshes+1,
            "Unpack failed to refresh exactly once on every rank");
    Require(diagnostics.GetStatistics().searches==0,"constraint refresh unexpectedly searched");
    const auto restored=ConstraintValues(diagnostics);
    Require(reference.size()==restored.size() &&
            std::memcmp(reference.data(),restored.data(),reference.size()*sizeof(Real))==0,
            "Unpack left stale constraint values");
    HostArray2D<Real> after;h::restart::Pack(p,after);
    Require(original.size()==after.size() &&
            std::memcmp(original.data(),after.data(),original.size()*sizeof(Real))==0,
            "restart diagnostic refresh changed checkpoint state");
    if (!global_variable::my_rank)std::cout<<"PASS same-object restart diagnostic cache ranks="
        <<global_variable::nranks<<'\n';
  }}catch (const std::exception &error) {
    std::cerr<<"restart diagnostic cache rank "<<global_variable::my_rank<<": "
             <<error.what()<<'\n';MPI_Abort(MPI_COMM_WORLD,1);return 1;
  }
  Kokkos::finalize();MPI_Finalize();return 0;
}
