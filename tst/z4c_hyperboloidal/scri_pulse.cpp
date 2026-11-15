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
    if(argc!=3)throw std::runtime_error("usage: scri_pulse input.athinput amplitude");
    const Real amplitude=std::stod(argv[2]);
    if(!std::isfinite(amplitude)||fabs(amplitude)>1e-4)
      throw std::runtime_error("pulse amplitude must be finite and <=1e-4");
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
    // C-infinity radial bump: r in (.88,.96). Leading lapse/trace gauge
    // polarization travels outward: delta P=Omega sqrt(chi Q/F) d_r delta alpha.
    // This is NOT a full coupled, constraint-free eigenmode. Matched zero-pulse
    // controls and amplitude scaling are required before interpreting reflections.
    Kokkos::parallel_reduce("seed smooth outgoing gauge probe",p.npoints,
      KOKKOS_LAMBDA(int q,Real &maximum,int &nonzero) {
        const auto c=points(q);const auto &x=maps(q).x;
        const Real r=h::SphericalPolicy::Radius(x),z=(r-.92)/.04;
        if(fabs(z)>=1 || amplitude==0)return;
        const Real bump=amplitude*exp(1-1/(1-z*z));
        const Real deriv=-2*z*bump/(.04*(1-z*z)*(1-z*z));
        const auto u=h::SphericalPolicy::Physical(x,reference(q));
        const auto g=h::Geometry(u.metric);
        Real Q=0;for(int a=0;a<3;++a)for(int b=0;b<3;++b)
          Q+=x[a]*g.inverse[a][b]*x[b]/(r*r);
        const auto coefficient=h::ShiftCoefficients(r,u.alpha.value,u.chi.value,parameters);
        state(c.m,h::Fields::lapse,c.k,c.j,c.i)+=bump;
        state(c.m,h::Fields::trace,c.k,c.j,c.i)+=
          parameters.profile.At(x).omega*sqrt(u.chi.value*Q/coefficient.F)*deriv;
        maximum=fmax(maximum,fabs(bump));++nonzero;
      },Kokkos::Max<Real>(peak),Kokkos::Sum<int>(count));
    MPI_Allreduce(MPI_IN_PLACE,&peak,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&count,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    if(amplitude!=0 && !(peak>.9*fabs(amplitude) && count>0))
      throw std::runtime_error("pulse inadequately sampled");
    p.FormResidual();p.InvalidateDiagnostics();
    Kokkos::Timer timer;Driver driver(&pin,&mesh,1800,&timer);Outputs outputs(&pin,&mesh);
    driver.Initialize(&mesh,&pin,&outputs,false);
    if(!global_variable::my_rank)std::cout<<"PULSE_CHECKPOINT amplitude="<<amplitude
      <<" peak="<<peak<<" points="<<count<<" center=.92 width=.04\n";
  }}catch(const std::exception &e){std::cerr<<"rank "<<global_variable::my_rank<<" FAIL "<<e.what()<<'\n';failed=1;}
  MPI_Allreduce(MPI_IN_PLACE,&failed,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
  Kokkos::finalize();MPI_Finalize();return failed;
}
