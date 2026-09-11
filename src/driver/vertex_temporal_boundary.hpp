#ifndef DRIVER_VERTEX_TEMPORAL_BOUNDARY_HPP_
#define DRIVER_VERTEX_TEMPORAL_BOUNDARY_HPP_
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>
#include "driver/rk4_predictor_states.hpp"
#include "driver/subcycle_hierarchy.hpp"
#include "mesh/vertex_amr.hpp"

namespace subcycling {
// A target coordinate in fine-level global vertex units. All targets are one
// level finer than the supplied coarse block list. This plan is geometry-only;
// callers scatter the returned (target,component) values into their ghost arrays.
using FineVertex2D = std::array<std::int64_t,2>;
class VertexTemporalBoundary {
 public:
  template<int ORDER>
  void Build(const std::vector<BlockKey> &coarse_blocks,
             const std::vector<int> &source_blocks, int nx, int ny,
             const std::vector<FineVertex2D> &targets) {
    blocks_=-1;  // A failed rebuild invalidates the previous topology plan.
    static_assert(ORDER==4 || ORDER==6 || ORDER==8,"qualified VC transfer order");
    if (nx<2 || ny<2 || coarse_blocks.empty() || source_blocks.size()!=coarse_blocks.size()) {
      throw std::invalid_argument("invalid temporal boundary coarse layout");
    }
    std::set<int> unique;
    for (int id : source_blocks) if (id<0 || !unique.insert(id).second)
      throw std::invalid_argument("invalid temporal boundary source indices");
    std::map<std::pair<int,int>,int> donors;
    const int level=coarse_blocks.front()[0];
    if (level<0) throw std::invalid_argument("negative coarse level");
    for (int m=0;m<static_cast<int>(coarse_blocks.size());++m) {
      const auto &key=coarse_blocks[m];
      if (key[0]!=level || key[1]<0 || key[2]<0 || key[3]!=0 ||
          !donors.emplace(std::make_pair(key[1],key[2]),m).second) {
        throw std::invalid_argument("temporal boundary requires unique common-level Cartoon donors");
      }
    }
    // Resolve every stencil point to an ACTIVE donor, including across block
    // faces/corners. No ghost-to-ghost reads or order reduction at missing data.
    std::vector<std::array<int,3>> indices;
    std::vector<Real> weights;
    std::vector<int> offsets{0};
    for (const auto &target : targets) {
      std::int64_t left[2]; int count[2];
      for (int d=0;d<2;++d) {
        if (target[d]<0) throw std::invalid_argument("axis/physical boundary provider required");
        const bool odd=target[d]%2;
        left[d]=target[d]/2+(odd ? vertex_amr::MidpointRule<ORDER>::left_offset : 0);
        count[d]=odd ? ORDER : 1;
      }
      for (int j=0;j<count[1];++j) for (int i=0;i<count[0];++i) {
        const auto x=left[0]+i, y=left[1]+j;
        int donor=-1,di=0,dj=0;
        if (x>=0 && y>=0) {
          const auto bx=x/nx,by=y/ny;
          for (auto dx=bx-(x%nx==0);dx<=bx && donor<0;++dx) {
            for (auto dy=by-(y%ny==0);dy<=by && donor<0;++dy) {
              if (dx<0 || dy<0 || dx>std::numeric_limits<int>::max() ||
                  dy>std::numeric_limits<int>::max()) continue;
              auto found=donors.find({static_cast<int>(dx),static_cast<int>(dy)});
              if (found!=donors.end()) {donor=found->second;di=x-dx*nx;dj=y-dy*ny;}
            }
          }
        }
        if (donor<0) throw std::runtime_error("unavailable active temporal-boundary stencil");
        indices.push_back({donor,dj,di});
        weights.push_back((count[0]==1 ? 1 : vertex_amr::MidpointRule<ORDER>::weight(i))*
                          (count[1]==1 ? 1 : vertex_amr::MidpointRule<ORDER>::weight(j)));
      }
      offsets.push_back(indices.size());
    }
    Kokkos::realloc(indices_,indices.size(),3);
    Kokkos::realloc(weights_,weights.size());
    Kokkos::realloc(offsets_,offsets.size());
    auto hi=Kokkos::create_mirror_view(indices_);
    auto hw=Kokkos::create_mirror_view(weights_);
    auto ho=Kokkos::create_mirror_view(offsets_);
    for (int n=0;n<static_cast<int>(indices.size());++n) {
      for (int d=0;d<3;++d) hi(n,d)=indices[n][d];
      hw(n)=weights[n];
    }
    for (int n=0;n<static_cast<int>(offsets.size());++n) ho(n)=offsets[n];
    Kokkos::deep_copy(indices_,hi);Kokkos::deep_copy(weights_,hw);Kokkos::deep_copy(offsets_,ho);
    source_blocks_=source_blocks;
    nx_=nx;ny_=ny;blocks_=coarse_blocks.size();targets_=targets.size();
  }
  // Predictor block order must be identical to Build's coarse block order.
  // Temporal and spatial interpolation are linear here. Algebraic projection
  // and physical/axis boundary treatment belong to the consuming stage path.
  void Evaluate(const RK4PredictorStates &predictor, double fraction,
                double fine_dt, int stage, DvceArray2D<Real> &out) {
    if (blocks_<0) throw std::logic_error("uninitialized temporal boundary plan");
    if (predictor.SourceBlocks()!=source_blocks_)
      throw std::invalid_argument("temporal predictor donor order changed");
    predictor.EvaluateStage(fraction,fine_dt,stage,stage_values_);
    if (stage_values_.extent_int(0)!=blocks_ || stage_values_.extent_int(2)!=1 ||
        stage_values_.extent_int(3)!=ny_+1 || stage_values_.extent_int(4)!=nx_+1) {
      throw std::invalid_argument("temporal predictor does not match boundary layout");
    }
    const int nv=stage_values_.extent_int(1);
    if (out.extent_int(0)!=targets_ || out.extent_int(1)!=nv) Kokkos::realloc(out,targets_,nv);
    const auto values=stage_values_;const auto ids=indices_;
    const auto w=weights_;const auto offsets=offsets_;
    if (targets_>0) par_for("interpolate VC temporal boundary",DevExeSpace(),
        0,targets_-1,0,nv-1,KOKKOS_LAMBDA(int p,int v) {
      Real result=0;
      for (int n=offsets(p);n<offsets(p+1);++n)
        result+=w(n)*values(ids(n,0),v,0,ids(n,1),ids(n,2));
      out(p,v)=result;
    });
  }
 private:
  std::vector<int> source_blocks_;
  int nx_=0,ny_=0,blocks_=-1,targets_=0;
  DvceArray2D<int> indices_;
  DvceArray1D<int> offsets_;
  DvceArray1D<Real> weights_;
  DvceArray5D<Real> stage_values_;
};
}  // namespace subcycling
#endif  // DRIVER_VERTEX_TEMPORAL_BOUNDARY_HPP_
