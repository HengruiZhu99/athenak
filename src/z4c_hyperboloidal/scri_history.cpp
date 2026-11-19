// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include "mesh/mesh.hpp"
#include "z4c_hyperboloidal/surface_exchange.hpp"
#include "z4c_hyperboloidal/scri_reconstruction.hpp"
#include "z4c_hyperboloidal/scri_geometry.hpp"
namespace z4c_hyperboloidal {
std::array<Real,ScriHistory::count> SurfaceExchange::MeasureInvestigation() {
  using H=ScriHistory;
  auto &p=*module_;
  const auto samples=sampled_,refs=p.surface_reference;
  const auto maps=p.surface_geometry,bulk_maps=p.geometry;
  const auto points=p.surface_points.d_view;
  const auto active=p.active_blocks;
  const auto owners=p.face_blocks.d_view;
  const auto metrics=metric_geometry_,penrose=penrose_geometry_;
  const auto rows=rows_;const auto params=p.parameters;const auto source=p.boundary_source;
  const Real penalty=p.penalty;
  auto kernel=BindKernel();if(correction_jets_ready_)kernel.correction_jets=correction_jets_;
  if(investigation_values_.extent_int(0)!=p.nsurface_points)
    Kokkos::realloc(investigation_values_,p.nsurface_points,H::count);
  const auto values=investigation_values_;
  int invalid=0;
  Kokkos::parallel_reduce("scri pointwise signed diagnostics",
    Kokkos::RangePolicy<>(0,p.nsurface_points),KOKKOS_LAMBDA(int q,int &bad) {
      if(active.extent(0) && !active(owners(points(q).face)))return;
      auto j=samples(q);const auto map=maps(q);const auto c=points(q);
      if(!SphericalPolicy::Restore(map.x,j,refs(q))) {++bad;return;}
      const auto before=SphericalPolicy::Boundary(map,rows(q).spacing,j,params,source,penalty,&metrics(q),&penrose(q));
      const auto reconstruction = ReconstructRegularScri(
          map.x, SphericalPolicy::Physical(map.x,j), metrics(q), before, params);
      mapped::Jet lift[3];
      for(int a=0;a<3;++a) {lift[a]=kernel.Correction(c.face,c.j,c.k,map,a);j[22+a]=j[22+a]+lift[a];}
      const auto u=SphericalPolicy::Physical(map.x,j);
      const auto rate=SphericalPolicy::Boundary(map,rows(q).spacing,j,params,source,penalty,&metrics(q),&penrose(q));
      const auto no_penalty=SphericalPolicy::Boundary(map,rows(q).spacing,j,params,source,0,&metrics(q),&penrose(q));
      auto omega=params.profile.At(map.x);omega.omega=0;
      SetStationaryOmegaNormal(u.alpha.value,u.beta.value,u.alpha.d,u.beta.d,omega);
      const auto D=SphericalPolicy::Vector(j,25);
      const auto geometry=EvaluateScriGeometry(map.x,u,D.value,omega,params,&metrics(q),&penrose(q));
      auto parts=DampedConformalRHS(map.x,u,omega,params,&metrics(q),&penrose(q));
      const bool damping=parts.valid;
      const auto trace=ScriRegularityTraces(u,omega,&metrics(q),&penrose(q));
      const auto pole=TensorPoleNormalJet(u,omega,map.x,map.x,params.kappa1,params.kappa2,Real(1),params.outer_lapse_damping,true,true);
      if(!reconstruction.valid || !rate.valid || !no_penalty.valid || !geometry.valid || !damping) {++bad;return;}
      Real signed_values[H::signed_count]{},absolute[H::absolute_count]{};
      signed_values[H::s_error]=sqrt(u.chi.value*geometry.Q)-1;
      signed_values[H::z_before]=before.speed_t;signed_values[H::z_after]=rate.speed_t;
      signed_values[H::clock_original] = rate.clock;
      signed_values[H::clock_modified] = rate.clock;
      for(int a=0;a<3;++a) {
        signed_values[H::Bx+a]=rate.defect[a];
        signed_values[H::qx+a]=kernel.correction(c.face,a,c.k,c.j);
      }
      signed_values[H::invariant]=reconstruction.invariant;
      signed_values[H::I_geom]=geometry.obstruction;
      signed_values[H::I_difference]=reconstruction.invariant-geometry.obstruction;
      Real theta_n=0,p_n=0;
      for(int a=0;a<3;++a) {theta_n+=map.x[a]*u.theta.d[a];p_n+=map.x[a]*u.trace.d[a];}
      signed_values[H::theta_n]=theta_n;
      signed_values[H::scalar_normal]=2*p_n+theta_n+6+3*geometry.N+2*geometry.L;
      signed_values[H::N]=geometry.N;signed_values[H::L]=geometry.L;
      signed_values[H::p_t]=rate.rate[13];signed_values[H::theta_t]=rate.rate[14];
      signed_values[H::pen_s_t]=rate.speed_t-no_penalty.speed_t;
      signed_values[H::pen_chi]=rate.rate[0]-no_penalty.rate[0];
      signed_values[H::pen_p]=rate.rate[13]-no_penalty.rate[13];
      signed_values[H::pen_theta]=rate.rate[14]-no_penalty.rate[14];
      signed_values[H::clock_before]=before.clock;
      const Real invh=1/rows(q).spacing;
      absolute[H::pole_chi_h]=fabs(parts.pole.chi)*invh;
      absolute[H::pole_p_h]=fabs(parts.pole.trace)*invh;
      absolute[H::pole_theta_h]=fabs(parts.pole.theta)*invh;
      absolute[H::pole_alpha_h]=fabs(pole.lapse.value)*invh;
      for(int a=0;a<3;++a) {
        absolute[H::reg_lambda]=fmax(absolute[H::reg_lambda],fabs(trace.lambda[a]));
        absolute[H::pole_lambda_h]=fmax(absolute[H::pole_lambda_h],fabs(parts.pole.lambda[a])*invh);
        absolute[H::pole_beta_h]=fmax(absolute[H::pole_beta_h],fabs(params.restoring*(u.beta.value[a]+map.x[a]))*invh);
        absolute[H::pen_lambda]=fmax(absolute[H::pen_lambda],fabs(rate.rate[15+a]-no_penalty.rate[15+a]));
        for(int b=0;b<3;++b) {
          absolute[H::reg_a]=fmax(absolute[H::reg_a],fabs(trace.a[a][b]));
          absolute[H::pole_a_h]=fmax(absolute[H::pole_a_h],fabs(parts.pole.a[a][b])*invh);
          for(int k=0;k<3;++k)absolute[H::lift_beta_hessian]=fmax(absolute[H::lift_beta_hessian],fabs(lift[k].h[a][b]));
        }
      }
      for(int f=7;f<=12;++f)absolute[H::pen_a]=fmax(absolute[H::pen_a],fabs(rate.rate[f]-no_penalty.rate[f]));
      for(int i=0;i<H::signed_count;++i) {values(q,2*i)=-signed_values[i];values(q,2*i+1)=signed_values[i];}
      for(int i=0;i<H::absolute_count;++i)values(q,2*H::signed_count+i)=absolute[i];
      for(int i=0;i<H::count;++i)if(!Kokkos::isfinite(values(q,i)))++bad;
    },invalid);
  if(invalid)throw std::runtime_error("Invalid scri investigation diagnostic");
  std::array<Real,H::count> maxima;maxima.fill(-std::numeric_limits<Real>::infinity());
  const auto host=Kokkos::create_mirror_view_and_copy(HostMemSpace(),values);
  const auto host_active=Kokkos::create_mirror_view_and_copy(HostMemSpace(),active);
  for(int q=0;q<p.nsurface_points;++q) {
    if(host_active.extent(0) && !host_active(p.face_blocks.h_view(p.surface_points.h_view(q).face)))continue;
    for(int c=0;c<H::count;++c)maxima[c]=std::max(maxima[c],host(q,c));
  }
  Real shift_hessian=0;
  Kokkos::parallel_reduce("scri lift Hessians in adjacent bulk support",
    Kokkos::RangePolicy<>(0,p.npoints),KOKKOS_LAMBDA(int point,Real &b) {
      const auto c=kernel.points(point);const int face=kernel.block_faces(c.m);
      if(active.extent(0) && !active(c.m))return;
      if(face<0 || c.i-kernel.origin<kernel.n-2)return;
      const auto map=bulk_maps(point);
      for (int f = 0; f < 3; ++f) {
        const auto jet=kernel.NormalCorrection(point,map,f);
        for(int i=0;i<3;++i)for(int j=0;j<3;++j)
          b=fmax(b,fabs(jet.h[i][j]));
      }
    },Kokkos::Max<Real>(shift_hessian));
  maxima[2*H::signed_count+H::lift_beta_hessian]=std::max(maxima[2*H::signed_count+H::lift_beta_hessian],std::max(Real(0),shift_hessian));
  return maxima;
}
void Z4cHyperboloidal::AccumulateScriInvestigation(const std::array<Real,ScriHistory::count>&v) {
  if(!scri_investigation_ready_)scri_investigation_maxima_=v;
  else for(int i=0;i<ScriHistory::count;++i)scri_investigation_maxima_[i]=std::max(scri_investigation_maxima_[i],v[i]);
  scri_investigation_ready_=true;
}
std::array<Real,ScriHistory::count> Z4cHyperboloidal::TakeScriInvestigation() {
  if(!scri_investigation_enabled_ || !plans_ready_)throw std::logic_error("Scri investigation requires initialized plans");
  const int cycle=pmy_pack->pmesh->ncycle;
  if(!scri_investigation_ready_ && scri_investigation_written_cycle_==cycle)return scri_investigation_last_;
  if(!scri_investigation_ready_)surface_exchange_->Evaluate();
  scri_investigation_last_=scri_investigation_maxima_;
  scri_investigation_written_cycle_=cycle;scri_investigation_ready_=false;
  return scri_investigation_last_;
}
}  // namespace z4c_hyperboloidal
