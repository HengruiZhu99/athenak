// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Test-only initial checkpoint writer. Links the production Athena objects;
// does not add a pulse hook or alter any production evolution equation.
#include <cmath>
#include <iostream>
#include <memory>
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "outputs/outputs.hpp"
#include "pgen/pgen.hpp"
#include "z4c_hyperboloidal/z4c_hyperboloidal.hpp"
#include "z4c_hyperboloidal/spherical_policy.hpp"
namespace h=z4c_hyperboloidal;
int main(int argc,char **argv) {
  MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);Kokkos::initialize(argc,argv);
  int failed=0;
  try {{
    if(argc!=3 && argc!=5 && argc!=6)throw std::runtime_error(
      "usage: nonradiative_seed input.athinput amplitude [center half_width [shape]]");
    const Real center=argc>=5 ? std::stod(argv[3]) : .9555;
    const Real width=argc>=5 ? std::stod(argv[4]) : .009;
    const Real shape=argc==6 ? std::stod(argv[5]) : 1;
    if(!std::isfinite(shape)||!(shape>0))throw std::runtime_error("shape must be positive");
    if(!std::isfinite(center)||!std::isfinite(width)||!(width>0 && center-width>0 && center+width<1))
      throw std::runtime_error("pulse support must lie strictly inside (0,1)");
    const Real amplitude=std::stod(argv[2]);
    if(!std::isfinite(amplitude)||fabs(amplitude)>.1)
      throw std::runtime_error("pulse amplitude must be finite and <=.1");
    ParameterInput pin;IOWrapper file;file.Open(argv[1],IOWrapper::FileMode::read);
    pin.LoadFromFile(file);file.Close();
    pin.SetInteger("time","nlim",0);
    Mesh mesh(&pin);mesh.BuildTreeFromScratch(&pin);mesh.AddCoordinatesAndPhysics(&pin);
    if(mesh.multilevel)mesh.pmr=new MeshRefinement(&mesh,&pin);
    mesh.pgen=std::make_unique<ProblemGenerator>(&pin,&mesh);
    auto &p=*mesh.pmb_pack->pz4c_hyperboloidal;
    const auto state=p.u0;const auto points=p.points.d_view;
    const auto maps=p.geometry;const auto reference=p.reference;
    const auto parameters=p.parameters;
    Real peak=0;int count=0;
    // Pure gauge: leave physical spatial metric, K_phys, A, Lambda, Theta,
    // shift, B and D unchanged. B and D are dynamical gauge auxiliaries,
    // not lapse-gradient constraints. A wider pulse can cross the gauge blend:
    // there B=0 is free initial gauge data, not a claim of a stationary driver.
    Kokkos::parallel_reduce("seed nonradiative lapse",p.npoints,
      KOKKOS_LAMBDA(int q,Real &maximum,int &nonzero) {
        const auto c=points(q);const auto &x=maps(q).x;
        const Real r=h::SphericalPolicy::Radius(x),z=(r-center)/width;
        if(fabs(z)>=1 || amplitude==0)return;
        const Real angular=1+.5*(x[0]*x[0]-x[1]*x[1])/(r*r);
        const Real bump=amplitude*exp(shape*(1-1/(1-z*z)))*angular;
        state(c.m,h::Fields::lapse,c.k,c.j,c.i)+=bump;
        maximum=fmax(maximum,fabs(bump));++nonzero;
      },Kokkos::Max<Real>(peak),Kokkos::Sum<int>(count));
    MPI_Allreduce(MPI_IN_PLACE,&peak,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&count,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    if(amplitude!=0 && !(peak>.9*fabs(amplitude) && count>0))
      throw std::runtime_error("pulse inadequately sampled");
    p.FormResidual();p.InvalidateDiagnostics();
    Kokkos::Timer timer;Driver driver(&pin,&mesh,1800,&timer);Outputs outputs(&pin,&mesh);
    driver.Initialize(&mesh,&pin,&outputs,false);
    if(!global_variable::my_rank)std::cout<<"NONRADIATIVE_SEED amplitude="<<amplitude
      <<" peak="<<peak<<" points="<<count<<" center="<<center<<" width="<<width
      <<" shape="<<shape<<" gauge_blend="<<parameters.inner_begin<<","<<parameters.inner_end
      <<" angular=1+.5(nx^2-ny^2)\n";
  }}catch(const std::exception &e){std::cerr<<"rank "<<global_variable::my_rank<<" FAIL "<<e.what()<<'\n';failed=1;}
  MPI_Allreduce(MPI_IN_PLACE,&failed,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
  Kokkos::finalize();MPI_Finalize();return failed;
}
