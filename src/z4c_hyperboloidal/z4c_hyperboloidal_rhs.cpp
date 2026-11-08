// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
//! \file z4c_hyperboloidal_rhs.cpp
//! \brief Rank-batched mapped RHS on disjoint native block work lists.

#include <stdexcept>

#include "driver/driver.hpp"
#include "mesh/mesh.hpp"
#include "z4c_hyperboloidal/z4c_hyperboloidal.hpp"
#include "z4c_hyperboloidal/point_rhs.hpp"
#include "z4c_hyperboloidal/endpoint_state.hpp"
#include "z4c_hyperboloidal/dissipation_plan.hpp"
#include "z4c_hyperboloidal/surface_exchange.hpp"

namespace z4c_hyperboloidal {

TaskStatus Z4cHyperboloidal::CalcRHS(Driver *driver,int stage,RHSSubset subset) {
  if (!plans_ready_) throw std::logic_error("Hyperboloidal RHS has no stage plans");
  if (!driver->UsesClassicalRK4() || stage<1 || stage>4)
    throw std::invalid_argument("Hyperboloidal RHS requires classical RK4 stage");
  const auto selected=rhs_points_[static_cast<int>(subset)];
  const auto cells=points.d_view;const auto maps=geometry;const auto refs=reference;
  const auto sources=stationary_source;const auto params=parameters;
  const auto physical=u0,delta=residual,output=u_rhs;
  const auto statuses=rhs_status;
  const auto gaps=rhs_gap,cfls=rhs_cfl;
  const auto dissipation=dissipation_plan_->Bind();
  const auto surface=surface_exchange_->BindKernel();
  const Real epsilon=ko,dt=pmy_pack->pmesh->dt;
  const int count=n[0];
  Kokkos::parallel_for("hyperboloidal block RHS",Kokkos::RangePolicy<>(0,selected.extent_int(0)),
    KOKKOS_LAMBDA(int index) {
      const int p=selected(index);const auto c=cells(p);const auto map=maps(p);
      const Real inv_h[3]={Real(count),Real(count),Real(count)};
      const BlockPointReader reader{delta,c.m,c.k,c.j,c.i};
      const auto state=surface.HasRadialRow(p)
          ? EndpointState(p,reader,map,inv_h,refs(p),params,surface)
          : BuildPointState(reader,map,inv_h,refs(p),params);
      statuses(p)=static_cast<int>(state.status);
      if (state.status!=PointStatus::success) return;
      PointForcing forcing;
      for (int f=0;f<nvar;++f) {
        forcing.transport[f]=state.transport[f];
        forcing.dissipation[f]=dissipation.Evaluate(p,f,{physical,c.m,c.k,c.j,c.i},epsilon);
      }
      const auto result=EvaluateBulkPoint(map,state.jets,params,sources(p),forcing);
      statuses(p)=static_cast<int>(result.status);
      gaps(p)=result.gap;cfls(p)=dt*count*(result.speed[0]+result.speed[1]+result.speed[2]);
      if (dt/fmin(params.tau,Real(InnerTelegrapher::tau))>.25)
        statuses(p)=static_cast<int>(PointStatus::cfl_exceeded);
      for (int f=0;f<nvar;++f) output(c.m,f,c.k,c.j,c.i)=result.rate[f];
    });
  int invalid=0;
  Kokkos::parallel_reduce("hyperboloidal RHS validity",
    Kokkos::RangePolicy<>(0,selected.extent_int(0)),KOKKOS_LAMBDA(int i,int &bad) {
      if (statuses(selected(i))!=static_cast<int>(PointStatus::success)) ++bad;
    },invalid);
  if (invalid) throw std::runtime_error("Invalid native hyperboloidal bulk RHS");
  return TaskStatus::complete;
}
}  // namespace z4c_hyperboloidal
