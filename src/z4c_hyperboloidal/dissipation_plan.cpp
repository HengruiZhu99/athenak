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
  const int n = module.n[0], origin = module.origin;
  const auto &maps=pack.pmb->mb_map.h_view;
  struct KOLine {
    int axis, target;
    Real weight;
  };
  std::vector<KOLine> lines;
  std::vector<int> offsets(module.npoints + 1);
  Kokkos::realloc(volumes_,module.npoints);
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
        *(maps(cell.m).node_offset[0]==0 && maps(cell.m).lo[0]==0
            && cell.i==origin ? .5 : 1.);
    // The nodal outer layer has a half cell at its physical inner endpoint.
    // Refinement introduces interior radial starts, which retain full weight.
  }
  for (int p=0;p<module.npoints;++p) {
    offsets[p] = lines.size();
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
      bool active=true;
      for (int a=0;a<5;++a) {
        int q[3]={index[0]-origin,index[1]-origin,index[2]-origin};q[axis]+=a;
        active=active && module.activity.Evolved(map,n,q[0],q[1],q[2]);
      }
      if (!active) continue;
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
        const Real r=std::sqrt(r2);
        Real taper=std::pow(std::max(Real(0),1-r*r/.64),6);
        // Angular lines alone receive the floor; it leaves radial lines unchanged.
        if (line.axis>0) taper=std::max(taper,module.ko_angular(r));
        line.weight+=volumes(donor_index)*std::sqrt(inverse_length)*taper/5;
      }
      lines.push_back(line);
    }
  }
  offsets[module.npoints] = lines.size();
  // Assemble the row of the weighted transpose-times-difference operator once.
  // Its zero row sum lets the kernel difference every donor against the center.
  Kokkos::realloc(support_, module.npoints);
  Kokkos::realloc(coefficients_, 24, module.npoints);
  auto support = Kokkos::create_mirror_view(support_);
  auto coefficients = Kokkos::create_mirror_view(coefficients_);
  Kokkos::deep_copy(coefficients, Real(0));
  for (int p = 0; p < module.npoints; ++p) {
    unsigned mask = 0;
    for (int t = offsets[p]; t < offsets[p + 1]; ++t) {
      const auto line = lines[t];
      const Real factor = -n * PhysicalKO8::Coefficient(line.target) * line.weight /
                          (PhysicalKO8::normalization * volumes(p));
      for (int a = 0; a < PhysicalKO8::points; ++a) {
        const int delta = a - line.target;
        if (delta == 0) continue;
        const int k = line.axis * 8 + (delta < 0 ? delta + 4 : delta + 3);
        mask |= 1u << k;
        coefficients(k, p) += factor * PhysicalKO8::Coefficient(a);
      }
    }
    support(p) = mask;
  }
  Kokkos::deep_copy(support_, support);
  Kokkos::deep_copy(coefficients_, coefficients);
  Kokkos::deep_copy(volumes_,volumes);
}
}  // namespace z4c_hyperboloidal
