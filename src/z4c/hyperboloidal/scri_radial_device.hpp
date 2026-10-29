// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_SCRI_RADIAL_DEVICE_HPP_
#define Z4C_HYPERBOLOIDAL_SCRI_RADIAL_DEVICE_HPP_
#include "mesh/cubed_sphere/scri_radial_difference.hpp"

namespace z4c::hyperboloidal {
// Owned-point device copy of the optional direct radial derivative plan.
// Points may be contiguous bulk ownership or noncontiguous numerator donors.
// This helper performs no exchange, field projection, or time integration.
class ScriRadialDevice {
 public:
  using ExecutionSpace=Kokkos::DefaultExecutionSpace;
  using MemorySpace=typename ExecutionSpace::memory_space;
  using Buffer=Kokkos::View<Real*,MemorySpace>;
  using ConstBuffer=Kokkos::View<const Real*,MemorySpace>;
  using IntBuffer=Kokkos::View<int*,MemorySpace>;
  struct Term {int point;bool boundary;Real d[3],dd[3][3];};
  using Terms=Kokkos::View<Term*,MemorySpace>;
  struct Kernel {
    ConstBuffer padded,surface;IntBuffer offsets,active;Terms terms;
    int fields=31,padded_count=0;
    KOKKOS_INLINE_FUNCTION bool HasRow(int local)const{
      return offsets.extent(0)>0&&offsets(local+1)>offsets(local);
    }
    KOKKOS_INLINE_FUNCTION void Derivatives(int f,int local,Real d[3],Real dd[3][3],bool second)const{
      const Real anchor=padded(f*padded_count+active(local));
      for(int a=0;a<3;++a){d[a]=0;for(int b=0;b<3;++b)dd[a][b]=0;}
      for(int i=offsets(local);i<offsets(local+1);++i){const auto t=terms(i);
        const Real value=(t.boundary?surface(fields*t.point+f):padded(f*padded_count+t.point))-anchor;
        for(int a=0;a<3;++a){d[a]+=t.d[a]*value;
          if(second)for(int b=0;b<3;++b)dd[a][b]+=t.dd[a][b]*value;}
      }
    }
  };
  // Optional execution variant: derivative values are stage data, not mesh
  // data. Prepare refreshes them on every invocation using the immutable plan.
  struct Jet {Real d[3],dd[3][3];};
  using Jets=Kokkos::View<Jet*,MemorySpace>;
  struct CachedKernel {
    IntBuffer lookup;Jets jets;int fields=31;
    KOKKOS_INLINE_FUNCTION bool HasRow(int local)const{
      return lookup.extent(0)>0&&lookup(local)>=0;
    }
    KOKKOS_INLINE_FUNCTION void Derivatives(int f,int local,Real d[3],Real dd[3][3],bool second)const{
      const auto value=jets(fields*lookup(local)+f);
      for(int a=0;a<3;++a){d[a]=value.d[a];for(int b=0;b<3;++b)dd[a][b]=second?value.dd[a][b]:0;}
    }
  };
#if defined(HYPERBOLIC_RADIAL_FIELD_BATCH) && HYPERBOLIC_RADIAL_FIELD_BATCH
  using BoundKernel=CachedKernel;
#else
  using BoundKernel=Kernel;
#endif
 private:
  IntBuffer offsets_,active_;Terms terms_;int fields_,padded_count_,surface_count_;
#if defined(HYPERBOLIC_RADIAL_FIELD_BATCH) && HYPERBOLIC_RADIAL_FIELD_BATCH
  IntBuffer cached_points_,cached_lookup_;Jets cached_jets_;
  struct SampleFields {
    Kernel source;IntBuffer points;Jets jets;
    KOKKOS_INLINE_FUNCTION void operator()(int index)const{
      Jet result{};source.Derivatives(index%source.fields,points(index/source.fields),result.d,result.dd,true);
      jets(index)=result;
    }
  };
#endif
 public:
  ScriRadialDevice(const cubed_sphere::ScriRadialDifference&plan,const std::vector<size_t>&points)
      :offsets_("owned direct radial offsets",points.size()+1),active_("owned direct radial anchors",points.size()),
       fields_(plan.scri.components),padded_count_(plan.grid.count),surface_count_(plan.scri.rows.size()) {
    auto offsets=Kokkos::create_mirror_view(offsets_),active=Kokkos::create_mirror_view(active_);
    std::vector<Term>terms;
    for(size_t i=0;i<points.size();++i){const auto p=points[i];
      if(p>=plan.grid.active.size())throw std::invalid_argument("invalid direct radial owned point");
      active(i)=plan.grid.active[p];offsets(i)=terms.size();
      if(plan.Contains(p))for(const auto&t:plan.rows[plan.row_index[p]]){
        Term value{};value.point=t.point;value.boundary=t.boundary;
        for(int a=0;a<3;++a){value.d[a]=t.d[a];for(int b=0;b<3;++b)value.dd[a][b]=t.dd[a][b];}
        terms.push_back(value);
      }
    }
    offsets(points.size())=terms.size();terms_=Terms("owned direct radial terms",terms.size());
    auto host_terms=Kokkos::create_mirror_view(terms_);for(size_t i=0;i<terms.size();++i)host_terms(i)=terms[i];
    Kokkos::deep_copy(offsets_,offsets);Kokkos::deep_copy(active_,active);Kokkos::deep_copy(terms_,host_terms);
#if defined(HYPERBOLIC_RADIAL_FIELD_BATCH) && HYPERBOLIC_RADIAL_FIELD_BATCH
    std::vector<int> selected;
    cached_lookup_=IntBuffer("radial stage lookup",points.size());
    auto lookup=Kokkos::create_mirror_view(cached_lookup_);
    for(size_t i=0;i<points.size();++i){lookup(i)=-1;
      if(offsets(i+1)>offsets(i)){lookup(i)=selected.size();selected.push_back(i);}}
    cached_points_=IntBuffer("radial stage sample rows",selected.size());
    auto samples=Kokkos::create_mirror_view(cached_points_);
    for(size_t i=0;i<selected.size();++i)samples(i)=selected[i];
    cached_jets_=Jets("current radial field derivatives",size_t(fields_)*selected.size());
    Kokkos::deep_copy(cached_lookup_,lookup);Kokkos::deep_copy(cached_points_,samples);
#endif
  }
  Kernel Bind(ConstBuffer padded,ConstBuffer surface)const{
    if(padded.extent(0)!=size_t(fields_)*padded_count_||surface.extent(0)!=size_t(fields_)*surface_count_)
      throw std::invalid_argument("direct radial device field shape mismatch");
    return Kernel{padded,surface,offsets_,active_,terms_,fields_,padded_count_};
  }
  BoundKernel Prepare(ConstBuffer padded,ConstBuffer surface)const{
    const auto source=Bind(padded,surface);
#if defined(HYPERBOLIC_RADIAL_FIELD_BATCH) && HYPERBOLIC_RADIAL_FIELD_BATCH
    Kokkos::parallel_for("radial derivatives across owned rows and fields",
      Kokkos::RangePolicy<ExecutionSpace>(0,cached_jets_.extent(0)),SampleFields{source,cached_points_,cached_jets_});
    return CachedKernel{cached_lookup_,cached_jets_,fields_};
#else
    return source;
#endif
  }
};
} // namespace z4c::hyperboloidal
#endif
