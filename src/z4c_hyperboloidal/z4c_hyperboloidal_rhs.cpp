// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
//! \file z4c_hyperboloidal_rhs.cpp
//! \brief Rank-batched mapped RHS on disjoint native block work lists.

#include <stdexcept>

#include "driver/driver.hpp"
#include "mesh/mesh.hpp"
#include "z4c_hyperboloidal/z4c_hyperboloidal.hpp"
#include "z4c_hyperboloidal/point_rhs.hpp"
#include "z4c_hyperboloidal/dissipation_plan.hpp"
#include "z4c_hyperboloidal/surface_exchange.hpp"

namespace z4c_hyperboloidal {
namespace {
KOKKOS_INLINE_FUNCTION PointState EndpointState(int p,const BlockPointReader &reader,
    const mapped::Point &map,const Real inv_h[3],const CartesianChartJets &reference,
    const HyperbolicShiftParameters<Real> &parameters,const SurfaceExchange::Kernel &surface) {
  PointState out;
  Real beta[3]{};
  for (int a=0;a<3;++a) beta[a]=reader(22+a)+reference[22+a].f-map.x[a];
  for (int f=0;f<Fields::count;++f) if (Fields::Independent(f)) {
    out.jets[f]=surface.ResidualJet(p,f,reader.state,f<=6||f==18||(f>=22&&f<25));
    // Radial transport already belongs to the endpoint reconstruction. Only
    // the ordinary angular centered-to-biased corrections are added here.
    for (int a=1;a<3;++a) {
      Real b=0;for (int d=0;d<3;++d) b+=map.J[a][d]*beta[d];
      const int sign=b>=0?1:-1;
      const Real center=reader(f);
      const Real vm=OffsetValue(reader,f,a,-sign)-center;
      const Real v1=OffsetValue(reader,f,a,sign)-center;
      const Real v2=OffsetValue(reader,f,a,2*sign)-center;
      const Real v3=OffsetValue(reader,f,a,3*sign)-center;
      const Real biased=sign*inv_h[a]*(-3*vm+18*v1-6*v2+v3)/12;
      out.transport[f]+=b*(biased-FirstDerivative(reader,f,a,inv_h[a]));
    }
  }
  mapped::Jet lift[3];
  for (int a=0;a<3;++a) lift[a]=surface.NormalCorrection(p,map,a);
  RestorePointState(out,reader,map,inv_h,reference,parameters,lift);
  return out;
}
}  // namespace

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
