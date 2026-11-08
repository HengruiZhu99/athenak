// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Full local block/scri initialization versus the independent legacy initializer.
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "pgen/pgen.hpp"
#include "driver/driver.hpp"
#include "z4c_hyperboloidal/z4c_hyperboloidal.hpp"
#include "z4c_hyperboloidal/spherical_policy.hpp"
#include "z4c/hyperboloidal/central_z4c_configuration.hpp"
#include "z4c/hyperboloidal/spherical_reference_3d.hpp"
#include "tst/hyperbolic_shift_3d/endpoint_policy/chi_policy.hpp"

namespace native = z4c_hyperboloidal;
namespace legacy = z4c::hyperboloidal;
void Require(bool yes,const char *what) {if (!yes) throw std::runtime_error(what);}

int main(int argc,char **argv) {
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);
  const std::string input=argv[1];
  Kokkos::initialize(argc,argv);
  try {
    {
      ParameterInput pin;
      IOWrapper file;file.Open(input.c_str(),IOWrapper::FileMode::read);
      pin.LoadFromFile(file);file.Close();
      Mesh mesh(&pin);mesh.BuildTreeFromScratch(&pin);mesh.AddCoordinatesAndPhysics(&pin);
      mesh.pgen=std::make_unique<ProblemGenerator>(&pin,&mesh);
      auto &p=*mesh.pmb_pack->pz4c_hyperboloidal;
      Require(p.IsInitialized(),"native problem generator did not initialize module");
      Require(p.nblocks==mesh.nmb_thisrank && p.u0.extent_int(0)==p.nblocks,
              "bulk state is not local native block storage");
      Require(p.s0.extent_int(0)==p.nfaces,"scri state is not local face storage");
      Require(p.origin==7 && p.overlap==3 && p.nghost==4,"N12 overlap/storage contract");
      auto state=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.u0);
      auto residual=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.residual);
      auto geom=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.geometry);
      auto edge=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.s0);
      auto edge_geom=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.surface_geometry);
      legacy::CentralZ4cConfiguration config;
      config.inner_z4c_radius_M=20;config.compact_slicing_end=.97;
      auto parameters=config.Resolve().gauge;parameters.outer_lapse_damping=.25;
      legacy::SphericalReference3D oracle(parameters);
      double error=0;std::uint64_t comparisons=0,nominal=0;
      auto compare=[&](double a,double b) {
        Require(std::isfinite(a)&&std::isfinite(b),"nonfinite initial state");
        error=std::max(error,std::abs(a-b)/std::max({1.,std::abs(a),std::abs(b)}));
        ++comparisons;
      };
      for (int i=0;i<p.npoints;++i) {
        auto c=p.points.h_view(i);
        Require(c.m>=0&&c.m<p.nblocks,"point escaped owning pack");
        const auto q=oracle.At(geom(i).x);
        for (int f=0;f<31;++f) {
          compare(state(c.m,f,c.k,c.j,c.i),q[f].f);
          Require(residual(c.m,f,c.k,c.j,c.i)==0,"initial residual not zero");
        }
        if (c.i>=p.origin&&c.i<p.origin+p.n[0]&&c.j>=p.origin&&c.j<p.origin+p.n[1]
            &&c.k>=p.origin&&c.k<p.origin+p.n[2]) ++nominal;
      }
      for (int i=0;i<p.nsurface_points;++i) {
        auto c=p.surface_points.h_view(i);
        Require(p.face_blocks.h_view(c.face)>=0 &&
                p.face_blocks.h_view(c.face)<p.nblocks,"scri owner not local");
        auto q=oracle.At(edge_geom(i).x,true);
        for (int f=0;f<31;++f) compare(edge(c.face,f,c.k,c.j),q[f].f);
      }
      for (int m=0;m<p.nblocks;++m) {
        Require(std::isnan(state(m,0,0,0,0)) && std::isnan(residual(m,0,0,0,0)),
                "unused ghosts were silently assigned analytic data");
      }
      std::uint64_t totals[4]={std::uint64_t(p.npoints),std::uint64_t(p.nsurface_points),
                               nominal,comparisons};
      MPI_Allreduce(MPI_IN_PLACE,totals,4,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE,&error,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
      Require(totals[0]==229392&&totals[1]==864&&totals[2]==183168,
              "N12 complete evolved workload changed");
      Require(error<1e-12,"initial-state oracle exceeds predeclared 1e-12 tolerance");

      // Exercise actual native module stage methods. Only the P+3 rate is nonzero;
      // poisoned dependent/dormant RHS values must never enter an update.
      Kokkos::Timer timer;Driver driver(&pin,&mesh,1000,&timer);
      driver.integrator="rk4_classical";driver.nexp_stages=4;mesh.dt=p.requested_dt;
      auto cells=p.points.d_view;auto faces=p.surface_points.d_view;
      auto rhs=p.u_rhs;auto srhs=p.s_rhs;
      const double poison=std::numeric_limits<double>::quiet_NaN();
      Kokkos::deep_copy(rhs,poison);Kokkos::deep_copy(srhs,poison);
      Kokkos::deep_copy(p.u_accum,poison);Kokkos::deep_copy(p.s_accum,poison);
      Kokkos::parallel_for("test bulk rates",Kokkos::RangePolicy<>(0,p.npoints),
        KOKKOS_LAMBDA(int i) {auto c=cells(i);for(int f=0;f<31;++f)
          if(native::Fields::Independent(f))rhs(c.m,f,c.k,c.j,c.i)=f==13?.25:0;});
      Kokkos::parallel_for("test scri rates",Kokkos::RangePolicy<>(0,p.nsurface_points),
        KOKKOS_LAMBDA(int i) {auto c=faces(i);for(int f=0;f<31;++f)
          if(native::Fields::Independent(f))srhs(c.face,f,c.k,c.j)=f==13?.25:0;});
      for (int stage=1;stage<=4;++stage) {
        driver.stage_time=time_integration::ClassicalRK4::RHSTime(mesh.time,mesh.dt,stage);
        driver.stage_state_time=time_integration::ClassicalRK4::StateTime(mesh.time,mesh.dt,stage);
        p.CopyU(&driver,stage);p.ExpRKUpdate(&driver,stage);
      }
      auto after=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.u0);
      auto surface_after=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.s0);
      // HostSpace mirror may alias a Serial device view: immutable RK base is authoritative.
      auto base=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.u1);
      auto surface_base=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.s1);
      for (int i=0;i<p.npoints;++i) {
        auto c=p.points.h_view(i);
        for (int f=0;f<31;++f) {
          double wanted=base(c.m,f,c.k,c.j,c.i)+(f==13?mesh.dt*.25:0);
          Require(std::abs(after(c.m,f,c.k,c.j,c.i)-wanted)<1e-12,
                  "native bulk RK storage/update mismatch");
        }
      }
      for (int i=0;i<p.nsurface_points;++i) {
        auto c=p.surface_points.h_view(i);
        for (int f=0;f<31;++f) {
          double wanted=surface_base(c.face,f,c.k,c.j)+(f==13?mesh.dt*.25:0);
          Require(std::abs(surface_after(c.face,f,c.k,c.j)-wanted)<1e-12,
                  "native scri RK storage/update mismatch");
        }
      }
      native::ChiRecord nr;EndpointChiRecord lr;
      const double values[5]={1.,-1.,1e-12,poison,2.};
      for(int i=0;i<5;++i) {
        double a=native::FloorChiCandidate(values[i],1e-10,.5*i,i==4,nr);
        double b=EndpointChiCandidate(values[i],1e-10,.5*i,i==4,lr);
        Require(a==b||(std::isnan(a)&&std::isnan(b)),"chi-floor candidate policy changed");
      }
      Require(nr.minimum==lr.minimum&&nr.accepted_minimum==lr.accepted_minimum&&
              nr.floor_count==lr.floor_count&&nr.nonpositive_count==lr.nonpositive_count&&
              nr.nonfinite_count==lr.nonfinite_count&&nr.max_correction==lr.max_correction&&
              nr.sum_correction==lr.sum_correction&&nr.first_floor==lr.first_floor,
              "chi-floor diagnostics changed");
      // A valid stage-four input can produce an inadmissible accepted state.
      // Reject it immediately, rather than waiting for the next cycle's RHS.
      const auto initial=p.reference,initial_edge=p.surface_reference;
      const auto current=p.u0;
      const auto current_edge=p.s0;
      const auto coordinates=p.geometry;
      const auto reset=[&]() {
        Kokkos::parallel_for("reset acceptance test bulk",Kokkos::RangePolicy<>(0,p.npoints),
          KOKKOS_LAMBDA(int i) {auto c=cells(i);for(int f=0;f<31;++f)
            current(c.m,f,c.k,c.j,c.i)=initial(i)[f].f;});
        Kokkos::parallel_for("reset acceptance test scri",Kokkos::RangePolicy<>(0,p.nsurface_points),
          KOKKOS_LAMBDA(int i) {auto c=faces(i);for(int f=0;f<31;++f)
            current_edge(c.face,f,c.k,c.j)=initial_edge(i)[f].f;});
        Kokkos::deep_copy(rhs,0.);Kokkos::deep_copy(srhs,0.);
        p.CopyU(&driver,1);
      };
      for (int scenario=0;scenario<2;++scenario) {
        reset();const Real dt=mesh.dt;
        Kokkos::parallel_for("inject invalid accepted state",Kokkos::RangePolicy<>(0,1),
          KOKKOS_LAMBDA(int) {
            const auto c=cells(0);
            if (scenario==0) {
              const Real r=native::SphericalPolicy::Radius(coordinates(0).x);
              const Real alpha=(1+r*r)/2+current(c.m,18,c.k,c.j,c.i);
              rhs(c.m,18,c.k,c.j,c.i)=-6*(alpha+1)/dt;
            } else rhs(c.m,0,c.k,c.j,c.i)=poison;
          });
        bool rejected=false;
        try {p.ExpRKUpdate(&driver,4);}
        catch (const std::runtime_error &error) {
          rejected=std::string(error.what()).find("Invalid accepted hyperboloidal bulk geometry")
              !=std::string::npos;
        }
        Require(rejected,"invalid accepted bulk lapse/chi escaped the stage-four guard");
      }
      reset();
      if(global_variable::my_rank==0)std::cout<<std::setprecision(17)
        <<"PASS native module: bulk="<<totals[0]<<" scri="<<totals[1]
        <<" nominal="<<totals[2]<<" comparisons="<<totals[3]
        <<" max_scaled_error="<<error
        <<" RK_storage=passed chi_policy=passed accepted_state_guard=passed\n";
    }
    Kokkos::finalize();MPI_Finalize();return 0;
  } catch(const std::exception &e) {
    std::cerr<<"module oracle rank "<<global_variable::my_rank<<": "<<e.what()<<'\n';
    MPI_Abort(MPI_COMM_WORLD,1);return 1;
  }
}
