// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
//! \file dissipation_plan.cpp
//! \brief Physical KO8 differences on native block-owned evolved cells.

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "mesh/mesh.hpp"
#include "z4c_hyperboloidal/z4c_hyperboloidal.hpp"
#include "z4c_hyperboloidal/dissipation_plan.hpp"

namespace z4c_hyperboloidal {
namespace {
Real Volume(const mapped::Point &point, int n) {
  const auto &a=point.J;
  const Real determinant=a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])
      -a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])
      +a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]);
  if (!std::isfinite(determinant) || determinant==0)
    throw std::runtime_error("Degenerate hyperboloidal KO map");
  return 1/(std::abs(determinant)*n*n*n);
}
}  // namespace

void DissipationPlan::Rebuild(const MeshBlockPack &pack,
                              const Z4cHyperboloidal &module) {
  n_=module.n[0];
  const int n=n_,origin=module.origin;
  const auto &maps=pack.pmb->mb_map.h_view;
  std::vector<KOLine> lines;
  Kokkos::realloc(offsets_,module.npoints+1);
  Kokkos::realloc(volumes_,module.npoints);
  auto offsets=Kokkos::create_mirror_view(offsets_);
  auto volumes=Kokkos::create_mirror_view(volumes_);
  const auto geometry=Kokkos::create_mirror_view_and_copy(HostMemSpace(),module.geometry);
  const auto index_of=[&](const int m,const int i,const int j,const int k) {
    return ((m*module.storage[2]+k)*module.storage[1]+j)*module.storage[0]+i;
  };
  std::vector<int> point_indices(module.nblocks*module.storage[0]
      *module.storage[1]*module.storage[2],-1);
  for (int p=0;p<module.npoints;++p) {
    const auto cell=module.points.h_view(p);
    point_indices[index_of(cell.m,cell.i,cell.j,cell.k)]=p;
    volumes(p)=Volume(geometry(p),n)
        *(maps(cell.m).node_offset[0]==0 && cell.i==origin ? .5 : 1.);
  }
  for (int p=0;p<module.npoints;++p) {
    offsets(p)=lines.size();
    const auto cell=module.points.h_view(p);
    const auto map=maps(cell.m);
    const auto bounds=module.evolved_bounds.h_view(cell.m);
    const auto point=geometry(p);
    if (map.face<0) {
      // Continue every Cartesian difference through the native halo at the
      // target spacing, as in the N12 mixed-level Cartesian closure. Truncate
      // only at the exterior of the independently evolved core overlap.
      for (int axis=0;axis<3;++axis) for (int target=0;target<5;++target) {
        const Real h=(map.hi[axis]-map.lo[axis])/n;
        const Real start=point.x[axis]-target*h;
        const Real limit=pack.pmesh->atlas->HalfWidth()+(module.overlap-.5)*h;
        if (start < -limit-1e-13 || start+4*h > limit+1e-13) continue;
        KOLine line{axis,target,0};
        for (int a=0;a<5;++a) {
          Real r2=0;
          for (int d=0;d<3;++d) {
            const Real x=point.x[d]+(d==axis ? (a-target)*h : 0);
            r2+=x*x;
          }
          const Real r=std::sqrt(r2),taper=std::pow(std::max(Real(0),1-r*r/.64),6);
          line.weight+=volumes(p)/(n*h)*taper/5;
        }
        lines.push_back(line);
      }
      continue;
    }
    struct Candidate {int axis,target,key;};
    std::vector<Candidate> candidates;
    for (int axis=0;axis<3;++axis) for (int target=0;target<5;++target) {
      int index[3]={cell.i,cell.j,cell.k};
      index[axis]-=target;
      if (index[axis]<bounds.lo[axis] || index[axis]+4>bounds.hi[axis]) continue;
      bool nominal=true;
      for (int a=0;a<3;++a) nominal=nominal && index[a]>=origin && index[a]<origin+n;
      // The old scatter accumulated starts in nominal-first, then overlap
      // order. Preserve this arithmetic order without a global field index.
      const int key=(nominal ? 0 : module.storage[0]*module.storage[1]*module.storage[2])
          +(index[2]*module.storage[1]+index[1])*module.storage[0]+index[0];
      candidates.push_back({axis,target,key});
    }
    std::sort(candidates.begin(),candidates.end(),[](const auto &a,const auto &b) {
      return a.key<b.key || (a.key==b.key && a.axis<b.axis);
    });
    for (const auto &candidate:candidates) {
      KOLine line{candidate.axis,candidate.target,0};
      for (int a=0;a<5;++a) {
        int index[3]={cell.i,cell.j,cell.k};
        index[line.axis]+=a-line.target;
        const int donor_index=point_indices[index_of(cell.m,index[0],index[1],index[2])];
        if (donor_index<0) throw std::logic_error("KO shell line leaves evolved storage");
        const auto &donor=geometry(donor_index);
        Real inverse_length=0,r2=0;
        for (int d=0;d<3;++d) {
          inverse_length+=donor.J[line.axis][d]*donor.J[line.axis][d];
          r2+=donor.x[d]*donor.x[d];
        }
        const Real r=std::sqrt(r2),taper=std::pow(std::max(Real(0),1-r*r/.64),6);
        line.weight+=volumes(donor_index)*std::sqrt(inverse_length)*taper/5;
      }
      lines.push_back(line);
    }
  }
  offsets(module.npoints)=lines.size();
  Kokkos::realloc(lines_,lines.size());
  auto host_lines=Kokkos::create_mirror_view(lines_);
  for (size_t t=0;t<lines.size();++t) host_lines(t)=lines[t];
  Kokkos::deep_copy(offsets_,offsets);
  Kokkos::deep_copy(volumes_,volumes);
  Kokkos::deep_copy(lines_,host_lines);
}
}  // namespace z4c_hyperboloidal
