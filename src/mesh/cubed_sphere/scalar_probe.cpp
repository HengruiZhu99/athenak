#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include "mesh/mesh.hpp"
#include "mesh/cubed_sphere/extension.hpp"
#include "mesh/cubed_sphere/derivatives.hpp"
#include "z4c/z4c.hpp"

namespace cubed_sphere {
namespace {
struct Flow {
  GeometryView geometry;
  DualArray1D<RegionSize> sizes;
  RegionIndcs ind;
  KOKKOS_INLINE_FUNCTION Real operator()(int m,int a,int k,int j,int i) const {
    if (geometry.extent(0)) return -geometry(m,k,j,i).x[a];
    auto s=sizes.d_view(m);
    Real x[3]={s.x1min+(i-ind.is+0.5)*s.dx1,s.x2min+(j-ind.js+0.5)*s.dx2,
               s.x3min+(k-ind.ks+0.5)*s.dx3};
    return -x[a];
  }
};
KOKKOS_INLINE_FUNCTION Real Pulse(const Real x[3],Real t) {
  const Real scale=exp(-t),width=0.25;
  return exp(-(SQR(scale*x[0]-0.5)+SQR(scale*x[1]-0.2)+SQR(scale*x[2]+0.1))/SQR(width));
}
}
// Exercises the actual leaf/patch interpolation and mapped upwind derivative with
// u_t+x.grad(u)=0. This has an exact solution and a maximum bound on the sphere;
// it is a diagnostic of the transport discretization, not a coupled-Z4c stability proof.
// The original spacetime state is restored before the ordinary driver starts.
void Extension::ScalarProbe(Real tlim,Real cfl) {
  if (!std::isfinite(tlim) || tlim<=0 || tlim>12 || initial_data_!=0 ||
      !std::isfinite(cfl) || cfl<=0 || cfl>1)
    throw std::invalid_argument(
        "scalar probe requires Minkowski initial data, 0<tlim<=12 and 0<CFL<=1");
  auto *bp=bulk_, *sp=shell_->pmb_pack;
  auto bu=bp->pz4c->u0,su=sp->pz4c->u0;
  // create_mirror_view may alias on Serial: snapshots must own distinct storage.
  DvceArray5D<Real> original_b("scalar original bulk",bu.extent(0),25,bu.extent(2),bu.extent(3),bu.extent(4));
  DvceArray5D<Real> original_s("scalar original shell",su.extent(0),25,su.extent(2),su.extent(3),su.extent(4));
  Kokkos::deep_copy(original_b,bu); Kokkos::deep_copy(original_s,su);
  bool outer=analytic_outer_; analytic_outer_=false;
  // The probe bypasses the driver's ordinary bulk neighbor exchange. Give its
  // uniform bulk halos the same active-leaf interpolation used for adaptive runs.
  scalar_probe_active_=true; mesh_sequence_=-1;
  Real step=std::numeric_limits<Real>::max();
  for (auto *pack:{bp,sp}) {
    auto ind=pack->pmesh->mb_indcs; auto u=pack->pz4c->u0;
    Flow flow{pack->pcoord->z4c_geometry,pack->pmb->mb_size,ind};
    auto geometry=flow.geometry; auto sizes=flow.sizes;
    par_for("cubed sphere scalar initial",DevExeSpace(),0,pack->nmb_thispack-1,
        ind.ks,ind.ke,ind.js,ind.je,ind.is,ind.ie,
        KOKKOS_LAMBDA(int m,int k,int j,int i) {
      Real x[3]; for (int a=0;a<3;++a) x[a]=-flow(m,a,k,j,i);
      u(m,0,k,j,i)=Pulse(x,0);
    });
    Real dt=std::numeric_limits<Real>::max();
    int cells=pack->nmb_thispack*ind.nx1*ind.nx2*ind.nx3;
    Kokkos::parallel_reduce("scalar CFL",Kokkos::RangePolicy<>(DevExeSpace(),0,cells),
        KOKKOS_LAMBDA(int l,Real &min_dt) {
      int i=l%ind.nx1+ind.is,j=(l/ind.nx1)%ind.nx2+ind.js;
      int k=(l/(ind.nx1*ind.nx2))%ind.nx3+ind.ks,m=l/(ind.nx1*ind.nx2*ind.nx3);
      auto s=MakeStencil(geometry,sizes.d_view(m),ind,m,k,j,i);
      Real inverse_dt=0;
      for (int A=0;A<3;++A) {
        Real speed=0;
        if (!s.mapped) speed=flow(m,A,k,j,i);
        else for (int a=0;a<3;++a) speed+=s.p.J[A][a]*flow(m,a,k,j,i);
        inverse_dt+=fabs(speed)*s.idx[A];
      }
      min_dt=fmin(min_dt,cfl/inverse_dt);
    },Kokkos::Min<Real>(dt));
    step=std::min(step,dt);
  }
  std::ofstream out("cubed_sphere_scalar.csv"); out<<std::setprecision(17);
  out<<"time,region,minimum,maximum,linf_error\n";
  auto measure=[&](Real time) {
    for (auto *pack:{bp,sp}) {
      auto ind=pack->pmesh->mb_indcs; auto u=pack->pz4c->u0;
      Flow flow{pack->pcoord->z4c_geometry,pack->pmb->mb_size,ind};
      Real min=0,max=0,error=0;
      int cells=pack->nmb_thispack*ind.nx1*ind.nx2*ind.nx3;
      Kokkos::parallel_reduce("scalar bounds",Kokkos::RangePolicy<>(DevExeSpace(),0,cells),
          KOKKOS_LAMBDA(int l,Real &mn,Real &mx,Real &err) {
        int i=l%ind.nx1+ind.is,j=(l/ind.nx1)%ind.nx2+ind.js;
        int k=(l/(ind.nx1*ind.nx2))%ind.nx3+ind.ks,m=l/(ind.nx1*ind.nx2*ind.nx3);
        Real value=u(m,0,k,j,i),x[3];
        for (int a=0;a<3;++a) x[a]=-flow(m,a,k,j,i);
        mn=fmin(mn,value); mx=fmax(mx,value); err=fmax(err,fabs(value-Pulse(x,time)));
        if (!Kokkos::isfinite(value)) err=INFINITY;
      },Kokkos::Min<Real>(min),Kokkos::Max<Real>(max),Kokkos::Max<Real>(error));
      out<<time<<","<<(pack==bp ? "bulk" : "shell")<<","<<min<<","<<max<<","<<error<<"\n";
      if (!std::isfinite(error) || max>2 || min<-1) {
        out.flush(); throw std::runtime_error("cubed sphere scalar transport amplification failed");
      }
    }
    out.flush();
  };
  Real time=0; int cycle=0; measure(time);
  while (time<tlim) {
    Real dt=std::min(step,tlim-time);
    Kokkos::deep_copy(bp->pz4c->u1,bu); Kokkos::deep_copy(sp->pz4c->u1,su);
    for (int stage=0;stage<3;++stage) {
      Fill(time+(stage==1 ? dt : (stage==2 ? dt/2 : 0)));
      for (auto *pack:{bp,sp}) {
        auto ind=pack->pmesh->mb_indcs;
        auto u=pack->pz4c->u0,reg=pack->pz4c->u1,rhs=pack->pz4c->u_rhs;
        Flow flow{pack->pcoord->z4c_geometry,pack->pmb->mb_size,ind};
        auto geometry=flow.geometry; auto sizes=flow.sizes;
        Real diss=0.1/64;
        par_for("scalar RHS",DevExeSpace(),0,pack->nmb_thispack-1,
            ind.ks,ind.ke,ind.js,ind.je,ind.is,ind.ie,
            KOKKOS_LAMBDA(int m,int k,int j,int i) {
          auto s=MakeStencil(geometry,sizes.d_view(m),ind,m,k,j,i);
          // A scalar subview gives the scalar overload of the ordinary upwind operator.
          auto scalar=Kokkos::subview(u,Kokkos::ALL,0,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
          Real rate=0;
          for (int a=0;a<3;++a) rate+=::Lx<3>(a,s,flow,scalar,m,a,k,j,i)
              +diss*::Diss<3>(a,s,u,m,0,k,j,i);
          rhs(m,0,k,j,i)=rate;
        });
      }
      // Both RHSs must be evaluated before either state changes.
      for (auto *pack:{bp,sp}) {
        auto ind=pack->pmesh->mb_indcs;
        auto u=pack->pz4c->u0,reg=pack->pz4c->u1,rhs=pack->pz4c->u_rhs;
        Real g0=stage==0 ? 0 : (stage==1 ? 0.25 : 2.0/3);
        Real g1=1-g0,beta=stage==1 ? 0.25 : (stage==2 ? 2.0/3 : 1);
        par_for("scalar RK",DevExeSpace(),0,pack->nmb_thispack-1,
            ind.ks,ind.ke,ind.js,ind.je,ind.is,ind.ie,
            KOKKOS_LAMBDA(int m,int k,int j,int i) {
          u(m,0,k,j,i)=g0*u(m,0,k,j,i)+g1*reg(m,0,k,j,i)+beta*dt*rhs(m,0,k,j,i);
        });
      }
    }
    time+=dt; ++cycle;
    if (cycle%50==0 || time>=tlim) measure(time);
  }
  Kokkos::deep_copy(bu,original_b); Kokkos::deep_copy(su,original_s);
  analytic_outer_=outer; scalar_probe_active_=false; mesh_sequence_=-1;
  Fill(bulk_->pmesh->time);
  std::cout<<"Cubed sphere scalar outward transport: t="<<time<<", cycles="<<cycle
           <<", dt="<<step<<std::endl;
}
}  // namespace cubed_sphere
