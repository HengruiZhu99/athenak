#ifndef DRIVER_RK4_PREDICTOR_STATES_HPP_
#define DRIVER_RK4_PREDICTOR_STATES_HPP_
#include <array>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>
#include "athena.hpp"
#include "driver/rk4_dense_boundary.hpp"
#include "z4c/z4c_grid.hpp"

namespace subcycling {
namespace detail {
inline void CopyRKActiveValues(const DvceArray5D<Real> &src, const DvceArray5D<Real> &dst,
    const DvceArray1D<int> &blocks, const z4c::Z4cGridLayout &layout) {
    const auto ids=blocks; const int is=layout.is,js=layout.js,ks=layout.ks;
    if (dst.extent(0)>0) par_for("retain coarse RK active values",DevExeSpace(),
        0,dst.extent_int(0)-1,0,dst.extent_int(1)-1,0,dst.extent_int(2)-1,
        0,dst.extent_int(3)-1,0,dst.extent_int(4)-1,
        KOKKOS_LAMBDA(int p,int v,int k,int j,int i) {
      dst(p,v,k,j,i)=src(ids(p),v,k+ks,j+js,i+is);
    });
  }
}  // namespace detail

// Sparse coarse RK history. Stores active vertices only; ghost RHS values are
// not supplied by the evolution operator. Spatial interpolation must obtain
// its complete stencil from active donors. No physical diagnostic ownership.
class RK4PredictorStates {
 public:
  void Begin(const DvceArray5D<Real> &state, const z4c::Z4cGridLayout &layout,
             const std::vector<int> &blocks, double start, double dt) {
    if (!std::isfinite(start) || !std::isfinite(dt) || dt<=0 ||
        !std::isfinite(start+dt) || start+dt==start) {
      throw std::invalid_argument("invalid predictor interval");
    }
    if (layout.is<0 || layout.js<0 || layout.ks<0 ||
        layout.ie<layout.is || layout.je<layout.js || layout.ke<layout.ks ||
        state.extent_int(4)<=layout.ie || state.extent_int(3)<=layout.je ||
        state.extent_int(2)<=layout.ke || state.extent_int(1)<1) {
      throw std::invalid_argument("invalid predictor active bounds");
    }
    std::set<int> unique;
    for (int b : blocks) if (b<0 || b>=state.extent_int(0) || !unique.insert(b).second) {
      throw std::invalid_argument("invalid or duplicate predictor block");
    }
    source_blocks_=blocks;
    completed_=0; start_=start; dt_=dt; layout_=layout;
    for (int d=0; d<5; ++d) source_shape_[d]=state.extent_int(d);
    const int np=blocks.size(), nv=state.extent_int(1);
    const int nk=layout.ke-layout.ks+1, nj=layout.je-layout.js+1, ni=layout.ie-layout.is+1;
    for (auto &a : data_) {
      if (a.extent_int(0)!=np || a.extent_int(1)!=nv || a.extent_int(2)!=nk ||
          a.extent_int(3)!=nj || a.extent_int(4)!=ni) Kokkos::realloc(a,np,nv,nk,nj,ni);
    }
    Kokkos::realloc(blocks_,np);
    auto host=Kokkos::create_mirror_view(blocks_);
    for (int p=0; p<np; ++p) host(p)=blocks[p];
    Kokkos::deep_copy(blocks_,host);
    detail::CopyRKActiveValues(state,data_[0],blocks_,layout_);
  }
  void Capture(const DvceArray5D<Real> &rhs, int stage) {
    if (stage<1 || stage>4 || stage!=completed_+1 || dt_<=0) {
      throw std::logic_error("predictor RHS stages must be captured in order");
    }
    for (int d=0; d<5; ++d) if (rhs.extent_int(d)!=source_shape_[d]) {
      throw std::invalid_argument("predictor source storage changed within step");
    }
    detail::CopyRKActiveValues(rhs,data_[stage],blocks_,layout_);
    completed_=stage;
  }
  // Fine-start fraction is relative to THIS parent step. Reject requests outside
  // its interval, including nominally synchronous stages with an overlong dt.
  void EvaluateStage(double fraction, double fine_dt, int stage,
                     DvceArray5D<Real> &out) const {
    if (completed_!=4) throw std::logic_error("incomplete coarse RK predictor");
    if (stage<1 || stage>4 || !std::isfinite(fraction) || !std::isfinite(fine_dt) ||
        fine_dt<=0 || fraction<0 || fraction>1 || fine_dt/dt_>1-fraction) {
      throw std::invalid_argument("fine stage outside predictor interval");
    }
    const auto y=data_[0], a=data_[1], b=data_[2], c=data_[3], d=data_[4];
    bool resize=false;
    for (int d=0;d<5;++d) resize=resize || out.extent(d)!=y.extent(d);
    if (resize) Kokkos::realloc(out,y.extent(0),y.extent(1),y.extent(2),y.extent(3),y.extent(4));
    const double dt=dt_;
    if (y.extent(0)>0) par_for("evaluate coarse RK stage predictor",DevExeSpace(),
        0,y.extent_int(0)-1,0,y.extent_int(1)-1,0,y.extent_int(2)-1,
        0,y.extent_int(3)-1,0,y.extent_int(4)-1,
        KOKKOS_LAMBDA(int p,int v,int k,int j,int i) {
      RK4DenseBoundary predictor{y(p,v,k,j,i),dt,a(p,v,k,j,i),b(p,v,k,j,i),
                                c(p,v,k,j,i),d(p,v,k,j,i)};
      out(p,v,k,j,i)=predictor.StageBoundary(fraction,fine_dt,stage);
    });
  }
  const std::vector<int> &SourceBlocks() const { return source_blocks_; }
  int CompletedStages() const { return completed_; }
  double StartTime() const { return start_; }
  std::size_t Bytes() const { return 5*data_[0].size()*sizeof(Real); }
 private:
  std::array<DvceArray5D<Real>,5> data_;
  DvceArray1D<int> blocks_;
  std::array<int,5> source_shape_{};
  std::vector<int> source_blocks_;
  z4c::Z4cGridLayout layout_;
  double start_=0,dt_=0;
  int completed_=0;
};
}  // namespace subcycling
#endif  // DRIVER_RK4_PREDICTOR_STATES_HPP_
