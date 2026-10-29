// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_HYPERBOLIC_SCRI_DEVICE_HPP_
#define Z4C_HYPERBOLOIDAL_HYPERBOLIC_SCRI_DEVICE_HPP_

#include "z4c/hyperboloidal/hyperbolic_composite.hpp"
#include "z4c/hyperboloidal/hyperbolic_composite_device.hpp"

namespace z4c::hyperboloidal {

// Distributed native scri atlas operator. Project() supplies the owned rows of
// BoundaryValues; gather those rows before Evaluate(). Evaluate reads immutable
// global active/surface states and produces the owned RHS and dependent D trace.
// Gather LocalSurface() again before off-node null projection in the halo fill.
// Every buffer is point-major. No volume ghosts or angular reduction are used.
class HyperbolicScriDevice {
 public:
  using Device=HyperbolicCompositeDevice;
  using ExecutionSpace=Device::ExecutionSpace;
  using MemorySpace=Device::MemorySpace;
  using Buffer=Device::Buffer;
  using ConstBuffer=Device::ConstBuffer;
  template<class T>using View=Kokkos::View<T*,MemorySpace>;
  static constexpr int nf=Device::nf;
  enum Status : int {
    success=0,invalid_input=1,invalid_geometry=2,invalid_source=3,
    invalid_gauge=4,cfl_exceeded=5,nonfinite_output=6
  };
  struct Summary {
    Real null_before=0,null_after=0,pole=0,clock=0,compatibility=0,max_cfl=0;
    int bad_rows=0,first_bad=-1,reason=success;
  };

 private:
  struct Row {Real x[3],h;int begin,end;};
  struct Term {int point;bool boundary;Real w,d[3],dd[3][3];};
  struct RowDiagnostic {
    Real null_before=0,null_after=0,pole=0,clock=0,compatibility=0,cfl=0;
    int status=success;
  };
  using Rows=View<Row>;
  using Terms=View<Term>;
  using Diagnostics=View<RowDiagnostic>;
  using Jets=View<ScalarJet<Real>>;
  // Only fields 0..24 enter the native scri equations. D is dependent and J
  // vanishes in the outer collar; their six stencils were never sampled.
  static constexpr int sampled_fields=25;
  int first_,local_,total_,active_total_;
  Rows rows_;
  Terms terms_;
  Jets samples_;
  Buffer local_surface_,local_rhs_;
  Diagnostics diagnostics_;

  struct Sample {
    ConstBuffer active,surface;
    Rows rows;Terms terms;
    KOKKOS_INLINE_FUNCTION Real Value(const Term&t,int f) const {
      return t.boundary?surface(nf*t.point+f):active(nf*t.point+f);
    }
    KOKKOS_INLINE_FUNCTION ScalarJet<Real> operator()(int local,int f) const {
      const auto row=rows(local);ScalarJet<Real> result{};
      const Real anchor=Value(terms(row.begin),f);result.value=anchor;
      for(int i=row.begin;i<row.end;++i) {
        const auto term=terms(i);const Real difference=Value(term,f)-anchor;
        result.value+=term.w*difference;
        for(int a=0;a<3;++a) {
          result.d[a]+=term.d[a]*difference;
          for(int b=0;b<3;++b)result.dd[a][b]+=term.dd[a][b]*difference;
        }
      }
      return result;
    }
  };
  // Each work item performs one field stencil in precisely the original term
  // order. This exposes field-level parallelism and keeps the stencil state
  // out of the substantially larger native tensor/normal-AD kernel.
  struct CacheSamples {
    Sample sample;Jets output;
    KOKKOS_INLINE_FUNCTION void operator()(int index) const {
      output(index)=sample(index/sampled_fields,index%sampled_fields);
    }
  };
  struct CachedSample {
    ConstBuffer surface;Rows rows;Jets samples;
    KOKKOS_INLINE_FUNCTION ScalarJet<Real> operator()(int local,int f) const {
      return samples(sampled_fields*local+f);
    }
  };
  template<class Sampling>
  KOKKOS_INLINE_FUNCTION static void Jet(const Sampling&sample,int local,
                                        Z4cJet<Real>&u,VectorJet<Real>&B,
                                        Real shift_derivative[3][3]) {
    const auto row=sample.rows(local);const auto*x=row.x;
    const Real r=sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
    u.chi=sample(local,0);u.trace=sample(local,13);u.trace.value-=3;
    u.theta=sample(local,14);u.alpha=sample(local,18);u.alpha.value+=(1+r*r)/2;
    for(int a=0;a<3;++a){u.alpha.d[a]+=x[a];u.alpha.dd[a][a]+=1;}
    constexpr int ti[6]={0,0,0,1,1,2},tj[6]={0,1,2,1,2,2};
    for(int t=0;t<6;++t) {
      const int i=ti[t],j=tj[t];
      const auto metric=sample(local,1+t),curvature=sample(local,7+t);
      u.metric.g[i][j]=u.metric.g[j][i]=metric.value+(i==j);
      u.a.k[i][j]=u.a.k[j][i]=curvature.value;
      for(int a=0;a<3;++a) {
        u.metric.dg[a][i][j]=u.metric.dg[a][j][i]=metric.d[a];
        u.a.dk[a][i][j]=u.a.dk[a][j][i]=curvature.d[a];
        for(int b=0;b<3;++b)u.metric.ddg[a][b][i][j]=u.metric.ddg[a][b][j][i]=metric.dd[a][b];
      }
    }
    for(int i=0;i<3;++i) {
      const auto shift=sample(local,22+i),connection=sample(local,15+i),flux=sample(local,19+i);
      u.beta.value[i]=shift.value-x[i];u.lambda.value[i]=connection.value;B.value[i]=flux.value;
      for(int a=0;a<3;++a) {
        u.beta.d[a][i]=shift.d[a]-(a==i);u.lambda.d[a][i]=connection.d[a];
        B.d[a][i]=flux.d[a];shift_derivative[a][i]=shift.d[a];
        for(int b=0;b<3;++b) {
          u.beta.dd[a][b][i]=shift.dd[a][b];u.lambda.dd[a][b][i]=connection.dd[a][b];
        }
      }
    }
  }
  struct ProjectRows {
    ConstBuffer input;Buffer output;Rows rows;Diagnostics diagnostics;int first;
    KOKKOS_INLINE_FUNCTION void operator()(int local) const {
      RowDiagnostic info{};const auto row=rows(local);const auto*x=row.x;Real v[nf];bool finite=true;
      for(int f=0;f<nf;++f) {
        v[f]=input(nf*(first+local)+f);output(nf*local+f)=v[f];
        finite=finite&&Kokkos::isfinite(v[f]);
      }
      if(!finite||!(v[0]>0)){info.status=invalid_input;diagnostics(local)=info;return;}
      MetricJet<Real> metric{};constexpr int ti[6]={0,0,0,1,1,2},tj[6]={0,1,2,1,2,2};
      for(int t=0;t<6;++t)metric.g[ti[t]][tj[t]]=metric.g[tj[t]][ti[t]]=v[1+t]+(ti[t]==tj[t]);
      const auto g=Geometry(metric);
      if(!g.valid){info.status=invalid_geometry;diagnostics(local)=info;return;}
      Real q=0,normal=0;
      for(int i=0;i<3;++i) {
        normal+=x[i]*(v[22+i]-x[i]);
        for(int j=0;j<3;++j)q+=g.inverse[i][j]*x[i]*x[j];
      }
      if(!(q>0)||!Kokkos::isfinite(v[0]*q)){info.status=invalid_geometry;diagnostics(local)=info;return;}
      info.null_before=fabs(normal*normal-v[0]*q)/(v[0]*q);
      const Real speed=sqrt(v[0]*q);v[18]=0;
      for(int i=0;i<3;++i){v[22+i]=(1-speed)*x[i];v[28+i]=0;}
      normal=0;for(int i=0;i<3;++i)normal+=x[i]*(v[22+i]-x[i]);
      info.null_after=fabs(normal*normal-v[0]*q)/(v[0]*q);
      for(int f=0;f<nf;++f)output(nf*local+f)=v[f];
      diagnostics(local)=info;
    }
  };
  template<class Sampling>struct EvaluateRows {
    Sampling sample;Buffer surface,rhs;Diagnostics diagnostics;int first;Real dt;
    ConstBuffer additive_rates;
    KOKKOS_INLINE_FUNCTION void operator()(int local) const {
      RowDiagnostic info{};const auto row=sample.rows(local);const auto*x=row.x;
      bool finite=true;
      for(int f=0;f<nf;++f) {
        rhs(nf*local+f)=0;surface(nf*local+f)=sample.surface(nf*(first+local)+f);
        finite=finite&&Kokkos::isfinite(surface(nf*local+f));
      }
      if(!finite){info.status=invalid_input;diagnostics(local)=info;return;}
      if(additive_rates.extent(0)!=0) {
        for(int f=0;f<nf;++f) {
          const Real value=additive_rates(nf*local+f);
          const bool free=f<18||(f>=19&&f<=21);
          if(!Kokkos::isfinite(value)||(!free&&value!=0)) {
            info.status=invalid_input;diagnostics(local)=info;return;
          }
        }
      }
      Z4cJet<Real> u{};VectorJet<Real>B{};Real shift_derivative[3][3]{};
      Jet(sample,local,u,B,shift_derivative);
      if(!(u.chi.value>0&&u.alpha.value>0)){info.status=invalid_input;diagnostics(local)=info;return;}
      const auto g=Geometry(u.metric);
      if(!g.valid){info.status=invalid_geometry;diagnostics(local)=info;return;}
      OmegaJet<Real> omega{};
      for(int i=0;i<3;++i){omega.gradient[i]=-x[i];omega.hessian[i][i]=-1;}
      SetStationaryOmegaNormal(u.alpha.value,u.beta.value,u.alpha.d,u.beta.d,omega);
      auto parts=ConformalRHS(u,omega,Real(4),Real(-.99));
      if(!HyperbolicDamping(x,u,omega,parts)){info.status=invalid_source;diagnostics(local)=info;return;}
      const auto pole=TensorPoleNormalJet(u,omega,x,x,Real(4),Real(-.99),Real(1),Real(1),true,true);
      auto rate=TensorScriLimit(parts.regular,pole,Real(-1));
      const auto trace=ScriRegularityTraces(u,omega);const Real h=row.h;
      info.cfl=dt*12/h;
      if(info.cfl>.5){info.status=cfl_exceeded;diagnostics(local)=info;return;}
      Real q=0;for(int i=0;i<3;++i)for(int j=0;j<3;++j)q+=g.inverse[i][j]*x[i]*x[j];
      rate.chi+=4/h*parts.pole.chi-8/h*(u.chi.value-1/q);
      rate.trace+=4/h*parts.pole.trace-8/h*(u.trace.value+3);
      rate.theta+=4/h*parts.pole.theta-8/h*u.theta.value;
      info.pole=fmax(fabs(parts.pole.chi),fmax(fabs(parts.pole.trace),fabs(parts.pole.theta)));
      for(int i=0;i<3;++i) {
        rate.lambda[i]+=4/h*parts.pole.lambda[i]-8/h*trace.lambda[i];
        info.pole=fmax(info.pole,fabs(parts.pole.lambda[i]));
        for(int j=0;j<3;++j) {
          rate.a[i][j]+=4/h*parts.pole.a[i][j]-8/h*trace.a[i][j];
          info.pole=fmax(info.pole,fabs(parts.pole.a[i][j]));
        }
      }
      // Optional angular dissipation is an additive rate of the 21 free
      // surface fields. Geometry must enter before the moving-normal D trace
      // below; lapse, shift, D and J are never independently filtered.
      if(additive_rates.extent(0)!=0) {
        rate.chi+=additive_rates(nf*local);
        rate.trace+=additive_rates(nf*local+13);
        rate.theta+=additive_rates(nf*local+14);
        constexpr int qi[6]={0,0,0,1,1,2},qj[6]={0,1,2,1,2,2};
        for(int t=0;t<6;++t) {
          const int i=qi[t],j=qj[t];
          rate.metric[i][j]+=additive_rates(nf*local+1+t);
          rate.a[i][j]+=additive_rates(nf*local+7+t);
          if(i!=j){rate.metric[j][i]+=additive_rates(nf*local+1+t);
            rate.a[j][i]+=additive_rates(nf*local+7+t);}
        }
        for(int i=0;i<3;++i)rate.lambda[i]+=additive_rates(nf*local+15+i);
      }
      rhs(nf*local)=rate.chi;rhs(nf*local+13)=rate.trace;rhs(nf*local+14)=rate.theta;
      constexpr int ti[6]={0,0,0,1,1,2},tj[6]={0,1,2,1,2,2};
      for(int t=0;t<6;++t) {
        rhs(nf*local+1+t)=rate.metric[ti[t]][tj[t]];rhs(nf*local+7+t)=rate.a[ti[t]][tj[t]];
      }
      for(int i=0;i<3;++i)rhs(nf*local+15+i)=rate.lambda[i];
      // Only gauge lapse/flux outputs enter BoundaryRates. Its shift_regular
      // contains D, but is discarded. Supplying zero D removes all cross-row
      // reads of the dependent trace and preserves these outputs exactly.
      const Real unused_D[3]{};const auto gauge=HyperbolicGauge(x,u,B,unused_D);
      if(!gauge.valid){info.status=invalid_gauge;diagnostics(local)=info;return;}
      info.clock=fabs(gauge.lapse_regular-pole.lapse.derivative);
      info.pole=fmax(info.pole,fabs(gauge.lapse_pole));
      for(int a=0;a<3;++a)rhs(nf*local+19+a)=gauge.flux_rate[a];
      if(additive_rates.extent(0)!=0)for(int a=0;a<3;++a)
        rhs(nf*local+19+a)+=additive_rates(nf*local+19+a);
      Real qdot=0,invn[3]{};
      for(int i=0;i<3;++i)for(int j=0;j<3;++j)invn[i]+=g.inverse[i][j]*x[j];
      for(int i=0;i<3;++i)for(int j=0;j<3;++j)qdot-=invn[i]*rate.metric[i][j]*invn[j];
      const Real speed=sqrt(u.chi.value*q),speed_t=(rate.chi*q+u.chi.value*qdot)/(2*speed);
      const Real mu=ShiftCoefficients(Real(1),Real(1),u.chi.value).mu;
      for(int a=0;a<3;++a) {
        Real regular=-x[a],restoring_limit=0;
        for(int b=0;b<3;++b) {
          regular+=u.beta.value[b]*u.beta.d[b][a];restoring_limit+=4*x[b]*shift_derivative[b][a];
        }
        const Real target=-speed_t*x[a];
        surface(nf*local+25+a)=(target-regular-restoring_limit)/mu;
        info.compatibility=fmax(info.compatibility,
            fabs(regular+restoring_limit+mu*surface(nf*local+25+a)-target));
      }
      for(int f=0;f<nf;++f)if(!Kokkos::isfinite(rhs(nf*local+f))||!Kokkos::isfinite(surface(nf*local+f)))
        info.status=nonfinite_output;
      if(!Kokkos::isfinite(info.pole)||!Kokkos::isfinite(info.clock)||!Kokkos::isfinite(info.compatibility))
        info.status=nonfinite_output;
      diagnostics(local)=info;
    }
  };
  struct ReduceRows {
    using value_type=Summary;Diagnostics diagnostics;int first;
    KOKKOS_INLINE_FUNCTION void init(Summary&out) const {
      out=Summary{};out.first_bad=INT_MAX;
    }
    KOKKOS_INLINE_FUNCTION void join(Summary&a,const Summary&b) const {
      a.null_before=fmax(a.null_before,b.null_before);a.null_after=fmax(a.null_after,b.null_after);
      a.pole=fmax(a.pole,b.pole);a.clock=fmax(a.clock,b.clock);
      a.compatibility=fmax(a.compatibility,b.compatibility);a.max_cfl=fmax(a.max_cfl,b.max_cfl);
      a.bad_rows+=b.bad_rows;
      if(b.first_bad<a.first_bad){a.first_bad=b.first_bad;a.reason=b.reason;}
    }
    KOKKOS_INLINE_FUNCTION void operator()(int local,Summary&out) const {
      const auto info=diagnostics(local);
      out.null_before=fmax(out.null_before,info.null_before);out.null_after=fmax(out.null_after,info.null_after);
      out.pole=fmax(out.pole,info.pole);out.clock=fmax(out.clock,info.clock);
      out.compatibility=fmax(out.compatibility,info.compatibility);out.max_cfl=fmax(out.max_cfl,info.cfl);
      if(info.status!=success) {
        ++out.bad_rows;
        if(first+local<out.first_bad){out.first_bad=first+local;out.reason=info.status;}
      }
    }
  };
  Summary Reduce() const {
    Summary out;
    Kokkos::parallel_reduce("scri owned diagnostics",Kokkos::RangePolicy<ExecutionSpace>(0,local_),
                           ReduceRows{diagnostics_,first_},out);
    if(out.first_bad==INT_MAX)out.first_bad=-1;
    return out;
  }
  void CheckSurface(ConstBuffer surface) const {
    if(surface.extent(0)!=size_t(nf)*total_)
      throw std::invalid_argument("device scri global atlas size mismatch");
  }

 public:
  HyperbolicScriDevice(const HyperbolicComposite&model,int first_surface,int local_count)
      :first_(first_surface),local_(local_count),total_(model.scri.rows.size()),
       active_total_(model.grid.active.size()) {
    if(first_<0||local_<0||size_t(first_)+size_t(local_)>model.scri.rows.size())
      throw std::invalid_argument("invalid device scri owned row range");
    rows_=Rows("scri owned rows",local_);diagnostics_=Diagnostics("scri owned diagnostics",local_);
    local_surface_=Buffer("scri owned surface",size_t(nf)*local_);
    local_rhs_=Buffer("scri owned RHS",size_t(nf)*local_);
#if !defined(HYPERBOLIC_SCRI_INLINE_SAMPLING) || !HYPERBOLIC_SCRI_INLINE_SAMPLING
    samples_=Jets("scri owned field jets",size_t(sampled_fields)*local_);
#endif
    auto rows=Kokkos::create_mirror_view(rows_);std::vector<Term>terms;
    std::vector<int>active(model.grid.count,-1);
    for(int p=0;p<active_total_;++p)active[model.grid.active[p]]=p;
    for(int local=0;local<local_;++local) {
      const auto&source=model.scri.rows[first_+local];Row row{};
      for(int a=0;a<3;++a)row.x[a]=source.geometry.x[a];
      row.h=source.radial_spacing;row.begin=terms.size();
      if(!(row.h>0)||!std::isfinite(row.h)||source.terms.empty())
        throw std::invalid_argument("invalid device scri row geometry/stencil");
      for(const auto&t:source.terms) {
        Term term{};term.boundary=t.boundary;term.w=t.w;
        if(t.boundary) {
          if(t.point<0||t.point>=total_)throw std::invalid_argument("invalid scri surface donor");
          term.point=t.point;
        }else {
          if(t.point<0||t.point>=model.grid.count||active[t.point]<0)
            throw std::invalid_argument("scri volume donor is not active");
          term.point=active[t.point];
        }
        for(int a=0;a<3;++a) {
          term.d[a]=t.d[a];for(int b=0;b<3;++b)term.dd[a][b]=t.dd[a][b];
        }
        terms.push_back(term);
      }
      row.end=terms.size();rows(local)=row;
    }
    terms_=Terms("scri owned sample terms",terms.size());auto host_terms=Kokkos::create_mirror_view(terms_);
    for(size_t i=0;i<terms.size();++i)host_terms(i)=terms[i];
    Kokkos::deep_copy(rows_,rows);Kokkos::deep_copy(terms_,host_terms);
  }
  Summary Project(ConstBuffer global_surface) {
    CheckSurface(global_surface);
    Kokkos::parallel_for("scri owned null projection",Kokkos::RangePolicy<ExecutionSpace>(0,local_),
                        ProjectRows{global_surface,local_surface_,rows_,diagnostics_,first_});
    return Reduce();
  }
  Summary Evaluate(ConstBuffer global_active,ConstBuffer projected_surface,Real dt,
                   ConstBuffer additive_rates={}) {
    CheckSurface(projected_surface);
    if(global_active.extent(0)!=size_t(nf)*active_total_||!(dt>0)||!std::isfinite(dt))
      throw std::invalid_argument("invalid device scri active state/timestep");
    if(projected_surface.data()==local_surface_.data())
      throw std::invalid_argument("device scri atlas input must not alias the owned output");
    if(additive_rates.extent(0)!=0) {
      if(additive_rates.extent(0)!=size_t(nf)*local_)
        throw std::invalid_argument("device scri additive-rate size mismatch");
      if(additive_rates.data()==local_surface_.data()||additive_rates.data()==local_rhs_.data())
        throw std::invalid_argument("device scri additive rates must not alias writable outputs");
    }
    const Sample source{global_active,projected_surface,rows_,terms_};
#if defined(HYPERBOLIC_SCRI_INLINE_SAMPLING) && HYPERBOLIC_SCRI_INLINE_SAMPLING
    // Controlled performance reference only: compile the same equations and
    // field summation order with the original in-kernel sampling organization.
    Kokkos::parallel_for("scri owned native rates inline sampling",
        Kokkos::RangePolicy<ExecutionSpace>(0,local_),
        EvaluateRows<Sample>{source,local_surface_,local_rhs_,diagnostics_,first_,dt,additive_rates});
#else
    Kokkos::parallel_for("scri owned field sampling",
        Kokkos::RangePolicy<ExecutionSpace>(0,sampled_fields*local_),CacheSamples{source,samples_});
    const CachedSample sample{projected_surface,rows_,samples_};
    Kokkos::parallel_for("scri owned native rates",Kokkos::RangePolicy<ExecutionSpace>(0,local_),
        EvaluateRows<CachedSample>{sample,local_surface_,local_rhs_,diagnostics_,first_,dt,additive_rates});
#endif
    return Reduce();
  }
  const Buffer&LocalSurface() const{return local_surface_;}
  const Buffer&LocalRHS() const{return local_rhs_;}
  int FirstSurface() const{return first_;}
  int LocalCount() const{return local_;}
  static const char*Reason(int code) {
    switch(code) {
      case success:return "success";
      case invalid_input:return "nonfinite or nonpositive scri input";
      case invalid_geometry:return "invalid scri metric";
      case invalid_source:return "invalid native scri source";
      case invalid_gauge:return "invalid scri gauge";
      case cfl_exceeded:return "scri explicit penalty CFL bound exceeded";
      case nonfinite_output:return "nonfinite scri RHS or dependent trace";
      default:return "unknown scri status";
    }
  }
};
}  // namespace z4c::hyperboloidal
#endif  // Z4C_HYPERBOLOIDAL_HYPERBOLIC_SCRI_DEVICE_HPP_
