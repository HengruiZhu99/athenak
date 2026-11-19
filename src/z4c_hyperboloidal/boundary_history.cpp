// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Optional diagnostics: no owned state, stage cache, or RHS values are changed.
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include "mesh/mesh.hpp"
#include "z4c_hyperboloidal/surface_exchange.hpp"
#include "z4c_hyperboloidal/point_rhs.hpp"
namespace z4c_hyperboloidal {
constexpr int columns=20;
std::array<Real,20> SurfaceExchange::MeasureHistory() {
  auto &p=*module_;
  const auto samples=SampledJets();const auto maps=p.surface_geometry;
  const auto refs=p.surface_reference;const auto points=p.surface_points.d_view;
  const auto metrics=MetricGeometry();const auto penrose=PenroseGeometry();
  const auto params=p.parameters;
  auto kernel=BindKernel();
  if(correction_jets_ready_)kernel.correction_jets=correction_jets_;
  const auto rhs=p.s_rhs;const auto owners=p.face_blocks.d_view;const auto bulk=p.u0;const auto edge=p.s0;
  const auto active=p.active_blocks;
  const int n=p.n[0],origin=p.origin;
  if(history_values_.extent(0)!=static_cast<std::size_t>(p.nsurface_points))
    Kokkos::realloc(history_values_,p.nsurface_points,columns);
  const auto measured=history_values_;
  Kokkos::deep_copy(measured,0.0);
  Kokkos::parallel_for("scri history stage diagnostics",Kokkos::RangePolicy<>(0,p.nsurface_points),
    KOKKOS_LAMBDA(int q) {
      if(active.extent(0) && !active(owners(points(q).face)))return;
      auto j=samples(q);const auto map=maps(q);const auto c=points(q);
      if(!SphericalPolicy::Restore(map.x,j,refs(q))) {
        measured(q,0)=NAN;return;
      }
      for(int a=0;a<3;++a)j[22+a]=j[22+a]+kernel.Correction(c.face,c.j,c.k,map,a);
      const auto u=SphericalPolicy::Physical(map.x,j);
      auto omega=params.profile.At(map.x);omega.omega=0;
      SetStationaryOmegaNormal(u.alpha.value,u.beta.value,u.alpha.d,u.beta.d,omega);
      auto parts=DampedConformalRHS(map.x,u,omega,params,&metrics(q),&penrose(q));
      if(!parts.valid) {
        measured(q,0)=NAN;return;
      }
      const auto traces=ScriRegularityTraces(u,omega,&metrics(q),&penrose(q));
      const auto pole=TensorPoleNormalJet(u,omega,map.x,map.x,params.kappa1,params.kappa2,
                                             Real(1),params.outer_lapse_damping,true,true);
      Real Q=0,traceA=0;
      for(int a=0;a<3;++a)for(int b=0;b<3;++b) {
        Q+=metrics(q).inverse[a][b]*map.x[a]*map.x[b];
        traceA+=metrics(q).inverse[a][b]*u.a.k[a][b];
      }
      const auto &g=u.metric.g;
      const Real det=g[0][0]*(g[1][1]*g[2][2]-g[1][2]*g[2][1])
        -g[0][1]*(g[1][0]*g[2][2]-g[1][2]*g[2][0])
        +g[0][2]*(g[1][0]*g[2][1]-g[1][1]*g[2][0]);
      measured(q,2)=fabs(u.alpha.value-1);
      measured(q,4)=fabs(u.chi.value*Q-1);measured(q,5)=fabs(u.trace.value+3);
      measured(q,6)=fabs(u.theta.value);measured(q,7)=fabs(det-1);measured(q,8)=fabs(traceA);
      measured(q,11)=fabs(parts.pole.chi);measured(q,12)=fabs(parts.pole.trace);
      measured(q,13)=fabs(parts.pole.theta);measured(q,16)=fabs(pole.lapse.value);
      for(int a=0;a<3;++a) {
        measured(q,3)=fmax(measured(q,3),fabs(u.beta.value[a]+sqrt(u.chi.value*Q)*map.x[a]));
        measured(q,9)=fmax(measured(q,9),fabs(traces.lambda[a]));
        measured(q,14)=fmax(measured(q,14),fabs(parts.pole.lambda[a]));
        const Real C=edge(c.face,25+a,c.k,c.j)-edge(c.face,15+a,c.k,c.j);
        measured(q,17)=fmax(measured(q,17),fabs(C));
        Real extrapolated=0;
        for(int k=0;k<5;++k) {
          Real w=1;for(int l=0;l<5;++l)if(k!=l)
            w*=(4+kernel.face_gap(c.face)-l)/Real(k-l);
          const int m=owners(c.face);
          extrapolated+=w*(bulk(m,25+a,origin+c.k,origin+c.j,origin+n-5+k)
                           -bulk(m,15+a,origin+c.k,origin+c.j,origin+n-5+k));
        }
        measured(q,18)=fmax(measured(q,18),fabs(C-extrapolated));
        for(int b=0;b<3;++b) {
          measured(q,10)=fmax(measured(q,10),fabs(traces.a[a][b]));
          measured(q,15)=fmax(measured(q,15),fabs(parts.pole.a[a][b]));
        }
      }
      for(int f=0;f<Fields::count;++f)measured(q,19)=fmax(measured(q,19),fabs(rhs(c.face,f,c.k,c.j)));
    });
  const auto host=Kokkos::create_mirror_view_and_copy(HostMemSpace(),measured);
  const auto host_active=Kokkos::create_mirror_view_and_copy(HostMemSpace(),active);
  std::array<Real,columns> maxima{};int invalid=0,count=p.nsurface_points;
  for(int q=0;q<count;++q) {
    if(host_active.extent(0) && !host_active(p.face_blocks.h_view(p.surface_points.h_view(q).face)))continue;
    for(int c=0;c<columns;++c) {
      invalid+=!std::isfinite(host(q,c));maxima[c]=std::max(maxima[c],host(q,c));
    }
  }
  if(invalid)throw std::runtime_error("Nonfinite scri history observation");
  maxima[0]=max_compatibility_;maxima[1]=max_clock_;
  return maxima;
}
void Z4cHyperboloidal::AccumulateBoundaryHistory(const std::array<Real,20> &values) {
  for(int i=0;i<20;++i)boundary_history_maxima_[i]=std::max(boundary_history_maxima_[i],values[i]);
  boundary_history_ready_=true;
}
std::array<Real,20> Z4cHyperboloidal::TakeBoundaryHistory() {
  if(!boundary_history_enabled_ || !plans_ready_)
    throw std::logic_error("Scri history requires enabled, initialized stage plans");
  const int cycle=pmy_pack->pmesh->ncycle;
  if(!boundary_history_ready_ && boundary_history_written_cycle_==cycle)
    return boundary_history_last_;
  // Fresh/zero-step output has no RK samples. Native initialization has already
  // sampled and solved the boundary; only its RHS scratch is evaluated here.
  if(!boundary_history_ready_)surface_exchange_->Evaluate();
  boundary_history_last_=boundary_history_maxima_;
  boundary_history_written_cycle_=cycle;
  boundary_history_maxima_.fill(0);boundary_history_ready_=false;
  return boundary_history_last_;
}
}  // namespace z4c_hyperboloidal
