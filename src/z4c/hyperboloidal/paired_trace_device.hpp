// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_PAIRED_TRACE_DEVICE_HPP_
#define Z4C_HYPERBOLOIDAL_PAIRED_TRACE_DEVICE_HPP_
#include <set>
#include "mesh/cubed_sphere/paired_trace_penalty.hpp"
#include "z4c/hyperboloidal/hyperbolic_composite_device.hpp"
namespace z4c::hyperboloidal {
// Deterministic row evaluation and transposed lift, batched across fields.
// Only rows contributing to an owned point are stored. No shared-node average.
class PairedTraceDevice {
 public:
  using Device=HyperbolicCompositeDevice;
  using Buffer=Device::Buffer;using ConstBuffer=Device::ConstBuffer;
  template<class T>using View=Kokkos::View<T*,Device::MemorySpace>;
  struct Row {int begin,end,normal,interface,diagnostic_owner;Real weight;};
  struct Term {int point;Real coefficient;};
  struct Lift {int row;Real coefficient;};
  struct Target {int point,begin,end;Real inverse_volume;};
 private:
  View<Row> rows_;View<Term> terms_;View<Target> targets_;View<Lift> lifts_;
  View<int> dependencies_;
  Buffer chart_,residual_,reference_;
  int first_;
  std::vector<int> dependencies;
  template<class T>static View<T> Upload(const char*name,const std::vector<T>&v){
    View<T> out(std::string(name),v.size());auto host=Kokkos::create_mirror_view(out);
    for(size_t i=0;i<v.size();++i)host(i)=v[i];Kokkos::deep_copy(out,host);return out;
  }
  KOKKOS_INLINE_FUNCTION static bool NormalField(int f){return f<=6||f==18||(f>=22&&f<=24);}
  void Residual(ConstBuffer state){
    const auto q=chart_,A=reference_;const auto ids=dependencies_;
    Kokkos::parallel_for("paired trace chart",Kokkos::RangePolicy<>(0,31*ids.extent(0)),KOKKOS_LAMBDA(int i){
      const int p=ids(i/31),f=i%31;const Real v=state(31*p+f);
      q(31*p+f)=f==0?log(v):f==18?log((v+A(p))/A(p)):v;
    });
    const auto rows=rows_;const auto terms=terms_;const auto residual=residual_;
    Kokkos::parallel_for("paired interface disagreement",Kokkos::RangePolicy<>(0,31*rows.extent(0)),KOKKOS_LAMBDA(int i){
      const auto r=rows(i/31);const int f=i%31;Real value=0;
      if(!r.normal||NormalField(f))for(int t=r.begin;t<r.end;++t)value+=terms(t).coefficient*q(31*terms(t).point+f);
      residual(i)=value;
    });
  }
 public:
  Real upper_value,upper_normal;
  PairedTraceDevice(const cubed_sphere::CompositeDifference&g,const cubed_sphere::PairedTracePenalty&p,int first,int local)
    :first_(first),upper_value(p.upper_value),upper_normal(p.upper_normal){
    std::vector<Row> rows;std::vector<Term> terms;std::vector<Target> targets;std::vector<Lift> lifts;
    std::vector<std::vector<Lift>> inverse(local);std::set<int> ids;
    for(const auto&r:p.rows){bool used=false;
      for(const auto&t:r.terms)if(t.point>=first&&t.point<first+local){used=true;break;}
      if(!used)continue;
      const int index=rows.size();Row row{int(terms.size()),0,r.normal,r.interface,
        int(r.terms.front().point>=first&&r.terms.front().point<first+local),r.weight};
      for(const auto&t:r.terms){terms.push_back({t.point,t.coefficient});ids.insert(t.point);
        if(t.point>=first&&t.point<first+local)inverse[t.point-first].push_back({index,t.coefficient});}
      row.end=terms.size();rows.push_back(row);
    }
    for(int i=0;i<local;++i)if(!inverse[i].empty()){
      Target target{i,int(lifts.size()),0,1/g.volumes[first+i]};
      lifts.insert(lifts.end(),inverse[i].begin(),inverse[i].end());target.end=lifts.size();targets.push_back(target);
    }
    dependencies.assign(ids.begin(),ids.end());dependencies_=Upload("trace donor ids",dependencies);
    rows_=Upload("trace rows",rows);terms_=Upload("trace terms",terms);
    targets_=Upload("trace lift targets",targets);lifts_=Upload("trace adjoint entries",lifts);
    chart_=Buffer("trace chart",31*g.active.size());residual_=Buffer("trace residual",31*rows.size());
    std::vector<Real>A;for(const auto&geo:g.geometry){Real r2=0;for(Real x:geo.x)r2+=x*x;A.push_back((1+r2)/2);}
    reference_=Upload("trace reference lapse",A);
  }
  const std::vector<int>& ActiveDependencies()const{return dependencies;}
  std::array<Real,6> Measure(ConstBuffer state){
    Residual(state);Buffer sums("trace diagnostic sums",6);Kokkos::deep_copy(sums,Real(0));
    const auto rows=rows_;const auto residual=residual_;
    Kokkos::parallel_for("paired trace norm",Kokkos::RangePolicy<>(0,rows.extent(0)),KOKKOS_LAMBDA(int i){
      const auto row=rows(i);if(!row.diagnostic_owner)return;Real value=0;
      for(int f=0;f<31;++f)value+=residual(31*i+f)*residual(31*i+f);
      Kokkos::atomic_add(&sums(3*row.interface+row.normal),row.weight*value);
      if(!row.normal)Kokkos::atomic_add(&sums(3*row.interface+2),row.weight);
    });
    auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),sums);
    std::array<Real,6> out;for(int i=0;i<6;++i)out[i]=host(i);return out;
  }
  void Add(ConstBuffer state,Buffer rhs,Real kappa,Real normal_ratio){
    if(kappa==0)return;Residual(state);
    const auto rows=rows_;const auto targets=targets_;const auto lifts=lifts_;const auto residual=residual_,A=reference_;const int first=first_;
    Kokkos::parallel_for("paired adjoint lift",Kokkos::RangePolicy<>(0,31*targets.extent(0)),KOKKOS_LAMBDA(int i){
      const auto target=targets(i/31);const int f=i%31,p=target.point;Real rate=0;
      for(int t=target.begin;t<target.end;++t){const auto lift=lifts(t);const auto row=rows(lift.row);
        rate-=lift.coefficient*row.weight*(row.normal?normal_ratio:1)*residual(31*lift.row+f);}
      rate*=kappa*target.inverse_volume;
      // P acts in log chi/log(alpha/A). Convert to physical RHS here; the
      // existing RK chart conversion divides by the same physical variable.
      if(f==0)rate*=state(31*(first+p));else if(f==18)rate*=state(31*(first+p)+18)+A(first+p);
      rhs(31*p+f)+=rate;
    });
    Kokkos::fence();
  }
};
}
#endif
