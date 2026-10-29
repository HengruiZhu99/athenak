// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_HYPERBOLIC_ANGULAR_DISSIPATION_DEVICE_HPP_
#define Z4C_HYPERBOLOIDAL_HYPERBOLIC_ANGULAR_DISSIPATION_DEVICE_HPP_

#include "mesh/cubed_sphere/angular_dissipation.hpp"

namespace z4c::hyperboloidal {

// Two distributed operations with no MPI inside: Residual reads the gathered
// native surface and computes owned B*u[21]. The caller gathers this residual;
// Apply then computes owned -epsilon*n*M^-1 B^T M*(B*u)[31]. Only free fields
// are populated. Native scri geometry rates must consume this source BEFORE D
// is reconstructed; prescribed lapse/shift and dependent D/J stay untouched.
class HyperbolicAngularDissipationDevice {
 public:
  using ExecutionSpace=Kokkos::DefaultExecutionSpace;
  using MemorySpace=typename ExecutionSpace::memory_space;
  using Buffer=Kokkos::View<Real*,MemorySpace>;
  using ConstBuffer=Kokkos::View<const Real*,MemorySpace>;
  using Plan=cubed_sphere::AngularDissipationPlan;
  static constexpr int nf=Plan::native_fields,free_fields=Plan::free_fields;
  struct Summary {int bad_values=0,first_bad=-1;};

 private:
  struct Term {int point;Real coefficient;};
  using Terms=Kokkos::View<Term*,MemorySpace>;
  using Indices=Kokkos::View<int*,MemorySpace>;
  int first_,local_,total_,n_;
  Real bound_;
  Indices offsets_,transpose_offsets_,status_;
  Terms terms_,transpose_terms_;
  Buffer areas_,local_residual_,local_rhs_;

  struct ResidualRows {
    ConstBuffer input;Buffer output;Indices offsets,status;Terms terms;int first;
    KOKKOS_INLINE_FUNCTION void operator()(int index)const {
      const int local=index/free_fields,f=index%free_fields,field=Plan::FreeField(f);
      const Real anchor=input(nf*(first+local)+field);Real value=0;
      bool valid=Kokkos::isfinite(anchor);
      for(int k=offsets(local);k<offsets(local+1);++k) {
        const auto t=terms(k);const Real donor=input(nf*t.point+field);
        valid=valid&&Kokkos::isfinite(donor);value+=t.coefficient*(donor-anchor);
      }
      valid=valid&&Kokkos::isfinite(value);status(index)=!valid;
      output(index)=valid?value:Real(0);
    }
  };
  struct ApplyRows {
    ConstBuffer input;Buffer output,areas;Indices offsets,status;Terms terms;
    int first,n;Real epsilon;
    KOKKOS_INLINE_FUNCTION void operator()(int index)const {
      const int local=index/free_fields,f=index%free_fields;Real value=0;bool valid=true;
      for(int k=offsets(local);k<offsets(local+1);++k) {
        const auto t=terms(k);const Real donor=input(free_fields*t.point+f);
        valid=valid&&Kokkos::isfinite(donor);value+=t.coefficient*areas(t.point)*donor;
      }
      value*=-epsilon*n/areas(first+local);
      valid=valid&&Kokkos::isfinite(value);status(index)=!valid;
      output(nf*local+Plan::FreeField(f))=valid?value:Real(0);
    }
  };
  struct ReduceValues {
    using value_type=Summary;Indices status;int first;
    KOKKOS_INLINE_FUNCTION void init(Summary&out)const{out.bad_values=0;out.first_bad=INT_MAX;}
    KOKKOS_INLINE_FUNCTION void join(Summary&a,const Summary&b)const {
      a.bad_values+=b.bad_values;if(b.first_bad<a.first_bad)a.first_bad=b.first_bad;
    }
    KOKKOS_INLINE_FUNCTION void operator()(int index,Summary&out)const {
      if(status(index)){++out.bad_values;const int row=first+index/free_fields;
        if(row<out.first_bad)out.first_bad=row;}
    }
  };
  Summary Reduce()const {
    Summary out;
    Kokkos::parallel_reduce("angular dissipation validity",Kokkos::RangePolicy<ExecutionSpace>(0,free_fields*local_),
                           ReduceValues{status_,first_},out);
    if(out.first_bad==INT_MAX)out.first_bad=-1;
    return out;
  }
  static void UploadPlan(const std::vector<int>&offsets,const std::vector<Plan::Term>&terms,
                         int first,int local,Indices&device_offsets,Terms&device_terms) {
    auto host_offsets=Kokkos::create_mirror_view(device_offsets);
    const int begin=offsets[first],end=offsets[first+local];
    device_terms=Terms("angular owned incidences",end-begin);
    auto host_terms=Kokkos::create_mirror_view(device_terms);
    for(int i=0;i<=local;++i)host_offsets(i)=offsets[first+i]-begin;
    for(int i=begin;i<end;++i)host_terms(i-begin)={terms[i].point,terms[i].coefficient};
    Kokkos::deep_copy(device_offsets,host_offsets);Kokkos::deep_copy(device_terms,host_terms);
  }

 public:
  HyperbolicAngularDissipationDevice(const Plan&plan,int first,int count)
      :first_(first),local_(count),total_(plan.Points()),n_(plan.N()),bound_(plan.SpectralUpperBound()) {
    if(first<0||count<0||size_t(first)+size_t(count)>size_t(total_))
      throw std::invalid_argument("invalid angular dissipation owned range");
    offsets_=Indices("angular owned residual offsets",local_+1);
    transpose_offsets_=Indices("angular owned transpose offsets",local_+1);
    status_=Indices("angular owned validity",free_fields*local_);
    areas_=Buffer("angular fixed global areas",total_);
    local_residual_=Buffer("angular owned residual",size_t(free_fields)*local_);
    local_rhs_=Buffer("angular owned native source",size_t(nf)*local_);
    auto areas=Kokkos::create_mirror_view(areas_);
    for(int i=0;i<total_;++i)areas(i)=plan.Areas()[i];Kokkos::deep_copy(areas_,areas);
    UploadPlan(plan.Offsets(),plan.Terms(),first_,local_,offsets_,terms_);
    UploadPlan(plan.TransposeOffsets(),plan.TransposeTerms(),first_,local_,transpose_offsets_,transpose_terms_);
  }
  Summary Residual(ConstBuffer global_surface) {
    if(global_surface.extent(0)!=size_t(nf)*total_)
      throw std::invalid_argument("device angular surface shape");
    if(global_surface.data()==local_residual_.data())
      throw std::invalid_argument("device angular residual input aliases output");
    Kokkos::parallel_for("angular owned polynomial residual",Kokkos::RangePolicy<ExecutionSpace>(0,free_fields*local_),
        ResidualRows{global_surface,local_residual_,offsets_,status_,terms_,first_});
    return Reduce();
  }
  Summary Apply(ConstBuffer global_residual,Real epsilon) {
    if(global_residual.extent(0)!=size_t(free_fields)*total_||!(epsilon>=0)||!std::isfinite(epsilon))
      throw std::invalid_argument("device angular residual shape/epsilon");
    if(global_residual.data()==local_rhs_.data())
      throw std::invalid_argument("device angular transpose input aliases output");
    Kokkos::deep_copy(local_rhs_,Real(0));
    Kokkos::parallel_for("angular owned weighted transpose",Kokkos::RangePolicy<ExecutionSpace>(0,free_fields*local_),
        ApplyRows{global_residual,local_rhs_,areas_,transpose_offsets_,status_,transpose_terms_,first_,n_,epsilon});
    return Reduce();
  }
  const Buffer&LocalResidual()const{return local_residual_;}
  const Buffer&LocalRHS()const{return local_rhs_;}
  int FirstSurface()const{return first_;}
  int LocalCount()const{return local_;}
  int TotalCount()const{return total_;}
  int N()const{return n_;}
  Real SpectralUpperBound()const{return bound_;}
};
}  // namespace z4c::hyperboloidal
#endif  // Z4C_HYPERBOLOIDAL_HYPERBOLIC_ANGULAR_DISSIPATION_DEVICE_HPP_
