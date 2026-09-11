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
#include "driver/corrector_control.hpp"
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
  // Physical common-time values for global coefficients/diagnostics. These
  // deliberately differ from intermediate RK stage vectors. The caller must
  // select each leaf's containing interval and exclude covered parent values.
  void EvaluateValue(double fraction,DvceArray5D<Real> &out) const {
    if(completed_!=4) throw std::logic_error("incomplete physical RK history");
    if(!std::isfinite(fraction) || fraction<0 || fraction>1)
      throw std::invalid_argument("physical time outside RK history");
    const auto y=data_[0],a=data_[1],b=data_[2],c=data_[3],d=data_[4];
    bool resize=false;for(int q=0;q<5;++q) resize|=out.extent(q)!=y.extent(q);
    if(resize) Kokkos::realloc(out,y.extent(0),y.extent(1),y.extent(2),y.extent(3),y.extent(4));
    const double dt=dt_;
    if(y.extent(0)>0) par_for("common-time physical RK reconstruction",DevExeSpace(),
        0,y.extent_int(0)-1,0,y.extent_int(1)-1,0,y.extent_int(2)-1,
        0,y.extent_int(3)-1,0,y.extent_int(4)-1,
        KOKKOS_LAMBDA(int p,int v,int k,int j,int i) {
      RK4DenseBoundary predictor{y(p,v,k,j,i),dt,a(p,v,k,j,i),b(p,v,k,j,i),
                                c(p,v,k,j,i),d(p,v,k,j,i)};
      out(p,v,k,j,i)=predictor.Value(fraction);
    });
  }
  // Fine-start fraction is relative to THIS parent step. Reject requests outside
  // its interval, including nominally synchronous stages with an overlong dt.
  void EvaluateStage(double fraction, double fine_dt, int stage,
                     DvceArray5D<Real> &out,bool rhs=false) const {
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
      out(p,v,k,j,i)=rhs ? predictor.StageRHS(fraction,fine_dt,stage) :
                                 predictor.StageBoundary(fraction,fine_dt,stage);
    });
  }
  // Two-way coupling: the first fine half-step supplies a local
  // Taylor reconstruction for its parent's RK stage vectors and dense history.
  // This is not evaluation of a physical solution outside the stored interval.
  void EvaluateParentStage(double parent_dt,int stage,bool rhs,DvceArray5D<Real> &out) const {
    if(completed_!=4 || stage<1 || stage>4 || dt_<=0 ||
       !std::isfinite(parent_dt) || std::abs(parent_dt/dt_-2)>1e-12)
      throw std::invalid_argument("invalid fine-to-parent stage reconstruction");
    const auto y=data_[0],a=data_[1],b=data_[2],c=data_[3],d=data_[4];
    bool resize=false;for(int q=0;q<5;++q) resize|=out.extent(q)!=y.extent(q);
    if(resize) Kokkos::realloc(out,y.extent(0),y.extent(1),y.extent(2),y.extent(3),y.extent(4));
    const double h=dt_;
    if(y.extent(0)>0) par_for("fine-to-parent RK reconstruction",DevExeSpace(),
        0,y.extent_int(0)-1,0,y.extent_int(1)-1,0,y.extent_int(2)-1,
        0,y.extent_int(3)-1,0,y.extent_int(4)-1,KOKKOS_LAMBDA(int p,int v,int k,int j,int i) {
      RK4DenseBoundary dense{y(p,v,k,j,i),h,a(p,v,k,j,i),b(p,v,k,j,i),c(p,v,k,j,i),d(p,v,k,j,i)};
      Real value=dense.StageBoundary(0,parent_dt,stage);
      if(rhs) {
        const double first=dense.First(0),second=dense.Second(0),third=dense.Third();
        const double jac=4*(c(p,v,k,j,i)-b(p,v,k,j,i))/(h*h),H=parent_dt;
        value=stage==1 ? first : stage==2 ? first+H*second/2+H*H*(third-jac)/8 :
              stage==3 ? first+H*second/2+H*H*(third+jac)/8 : first+H*second+H*H*third/2;
      }
      out(p,v,k,j,i)=value;
    });
  }

  Real Difference(const RK4PredictorStates &other,const CorrectorControl &control) const {
    if(completed_!=4 || other.completed_!=4 || source_blocks_!=other.source_blocks_ ||
       start_!=other.start_ || dt_!=other.dt_)
      throw std::invalid_argument("incompatible corrector RK histories");
    z4c::Z4cGridLayout l;
    l.is=l.js=l.ks=0;l.ie=data_[0].extent_int(4)-1;
    l.je=data_[0].extent_int(3)-1;l.ke=data_[0].extent_int(2)-1;
    Real error=0;
    for(int n=0;n<5;++n) error=std::max(error,CorrectorDifference(data_[n],other.data_[n],
                                                               l,control,n==0 ? 1 : dt_));
    return error;
  }
  const std::vector<int> &SourceBlocks() const { return source_blocks_; }
  int CompletedStages() const { return completed_; }
  double StartTime() const { return start_; }
  double Dt() const { return dt_; }
  std::size_t Bytes() const { return 5*data_[0].size()*sizeof(Real); }
 private:
  friend class RK4PhysicalMaximum;
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
