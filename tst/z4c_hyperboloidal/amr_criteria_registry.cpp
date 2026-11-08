// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Targeted native criterion/registration checks; this does not evolve or regrid.
#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/refinement_criteria.hpp"
#include "z4c_hyperboloidal/z4c_hyperboloidal.hpp"

void Require(bool condition,const char *message) {
  if (!condition) throw std::runtime_error(message);
}

int main(int argc,char **argv) {
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);
  Kokkos::initialize(argc,argv);
  try {
    {
      ParameterInput pin;
      IOWrapper file;file.Open(argv[1],IOWrapper::FileMode::read);
      pin.LoadFromFile(file);file.Close();
      pin.SetString("amr_criterion0","method","min_max");
      pin.SetString("amr_criterion0","variable","hyperboloidal_chi");
      pin.SetReal("amr_criterion0","value_max",3);
      Mesh mesh(&pin);mesh.BuildTreeFromScratch(&pin);mesh.AddCoordinatesAndPhysics(&pin);
      mesh.pmr=new MeshRefinement(&mesh,&pin);
      auto &ref=*mesh.pmr;
      auto &p=*mesh.pmb_pack->pz4c_hyperboloidal;
      RefinementCriteria criteria(&mesh,&pin);
      auto &criterion=criteria.rcrit.front();
      Require(criterion.index_offset==p.overlap && criterion.hyper_field==0,
              "mapped criterion lost field or origin");
      for (int f=0;f<z4c_hyperboloidal::Fields::count;++f) {
        criterion.rvariable="hyperboloidal_"+
            std::string(z4c_hyperboloidal::Fields::names[f]);
        criteria.SetRefinementData(mesh.pmb_pack,false,false);
        Require(criterion.hyper_field==f,"stored field-name binding changed");
      }
      criterion.rvariable="hyperboloidal_chi";
      criteria.SetRefinementData(mesh.pmb_pack,false,false);
      Require(!criteria.NeedsAcceptedHalos(),"min/max spuriously requires halos");
      const int start=mesh.gids_eachrank[global_variable::my_rank];
      auto flags=ref.refine_flag.d_view;
      auto reset=[&]() {Kokkos::deep_copy(flags,0);};
      auto check=[&](bool surface_only,int value) {
        for (int m=0;m<p.nblocks;++m) {
          int expected=surface_only && p.block_faces.h_view(m)<0 ? -1 : value;
          Require(ref.refine_flag.h_view(start+m)==expected,"criterion flag mismatch");
        }
      };

      // If native origin ng is used instead of ng+overlap, padding forces refinement.
      auto u=p.u0;const int o=p.origin,n=p.n[0];
      Kokkos::deep_copy(u,1e6);
      par_for("criterion nominal data",DevExeSpace(),0,p.nblocks-1,
        o,o+n-1,o,o+n-1,o,o+n-1,
        KOKKOS_LAMBDA(int m,int k,int j,int i) {u(m,0,k,j,i)=2;});
      reset();criteria.CheckMinMax(mesh.pmb_pack,criterion);check(false,-1);
      auto old=p.u0;
      p.u0=DvceArray5D<Real>("replacement criterion state",p.nblocks,31,
                             p.storage[2],p.storage[1],p.storage[0]);
      Kokkos::deep_copy(p.u0,4);
      criteria.SetRefinementData(mesh.pmb_pack,false,false);
      Require(criterion.rdata.data()!=old.data(),"criterion still bound to old allocation");
      reset();criteria.CheckMinMax(mesh.pmb_pack,criterion);check(false,1);

      // Signed offsets and zero values have defined normalized derivative behavior.
      // Poison the absent outer radial ghost: its neighbor must come from s0.
      auto faces=p.block_faces.d_view;u=p.u0;
      const Real poison=std::numeric_limits<Real>::quiet_NaN();
      for (auto method:{RefCritMethod::slope,RefCritMethod::second_deriv}) {
        criterion.rmethod=method;criterion.rvalue_max=.1;
        Require(criteria.NeedsAcceptedHalos(),"derivative criterion omits halo refresh");
        for (Real base:{-2.0,0.0,2.0}) {
          Kokkos::deep_copy(u,base);Kokkos::deep_copy(p.s0,base);
          par_for("poison absent exterior",DevExeSpace(),0,p.nblocks-1,
            o,o+n-1,o,o+n-1,KOKKOS_LAMBDA(int m,int k,int j) {
              if (faces(m)>=0) u(m,0,k,j,o+n)=poison;
            });
          auto evaluate=[&]() {
            reset();
            if (method==RefCritMethod::slope)
              criteria.CheckSlope(mesh.pmb_pack,criterion);
            else criteria.CheckSecondDeriv(mesh.pmb_pack,criterion);
          };
          evaluate();check(false,-1);
          Kokkos::deep_copy(p.s0,base+1);
          evaluate();check(true,1);
        }
      }
      criterion.rmethod=RefCritMethod::user;
      Require(criteria.NeedsAcceptedHalos(),"user criterion lacks accepted halos");

      // Collective schema/shape validation occurs before peer offsets are used.
      const int cap=mesh.nmb_maxperrank,ng=mesh.mb_indcs.ng;
      DvceArray5D<Real> a("registered a",cap,2,n+2*ng,n+2*ng,n+2*ng);
      DvceArray5D<Real> ca("registered coarse a",cap,2,n/2+2*ng,n/2+2*ng,n/2+2*ng);
      DvceArray5D<Real> b("registered b",cap,1,n+2*ng,n+2*ng,n+2*ng);
      DvceArray5D<Real> cb("registered coarse b",cap,1,n/2+2*ng,n/2+2*ng,n/2+2*ng);
      auto reject=[&](auto action) {
        bool caught=false;try {action();} catch(const std::invalid_argument &) {caught=true;}
        Require(caught,"invalid registry operation was accepted");
      };
      ref.RegisterCC("first",a,ca);ref.RegisterCC("second",b,cb);
      Require(ref.RegisteredComponents()==3,"registered component count is wrong");
      Require(ref.send_data.size()>=a.size()+b.size() &&
              ref.recv_data.size()>=a.size()+b.size(),"registered MPI buffers too small");
      reject([&]() {ref.RegisterCC("first",b,cb);});
      reject([&]() {ref.UnregisterCC("missing");});
      ref.UnregisterCC("second");ref.UnregisterCC("first");
      const auto retained=ref.send_data.size();
      ref.RegisterCC("again",a,ca);ref.UnregisterCC("again");
      Require(ref.RegisteredComponents()==0 && ref.send_data.size()==retained,
              "temporary registry failed to release views or retain capacity");
      if (global_variable::nranks>1) {
        reject([&]() {ref.RegisterCC(global_variable::my_rank==0 ? "left":"right",b,cb);});
        reject([&]() {ref.RegisterCC("component mismatch",
            global_variable::my_rank==0 ? a:b,global_variable::my_rank==0 ? ca:cb);});
        DvceArray5D<Real> wrong("wrong shape",cap,1,n+2*ng,n+2*ng,n+2*ng-1);
        reject([&]() {ref.RegisterCC("one rank invalid",
            global_variable::my_rank==0 ? wrong:b,cb);});
      }
      Require(ref.RegisteredComponents()==0,"failed registration retained a view");
      if (global_variable::my_rank==0)
        std::cout<<"PASS criteria origin/rebinding/scri/signed-zero; registry schema/shape/lifetime"
                 <<" ranks="<<global_variable::nranks<<"\n";
    }
    Kokkos::finalize();MPI_Finalize();return 0;
  } catch(const std::exception &error) {
    std::cerr<<"rank "<<global_variable::my_rank<<": "<<error.what()<<"\n";
    MPI_Abort(MPI_COMM_WORLD,1);return 1;
  }
}
