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
 private:
  IntBuffer offsets_,active_;Terms terms_;int fields_,padded_count_,surface_count_;
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
  }
  Kernel Bind(ConstBuffer padded,ConstBuffer surface)const{
    if(padded.extent(0)!=size_t(fields_)*padded_count_||surface.extent(0)!=size_t(fields_)*surface_count_)
      throw std::invalid_argument("direct radial device field shape mismatch");
    return Kernel{padded,surface,offsets_,active_,terms_,fields_,padded_count_};
  }
};
} // namespace z4c::hyperboloidal
#endif
