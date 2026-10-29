// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_HYPERBOLIC_SAMPLED_NORMAL_DEVICE_HPP_
#define Z4C_HYPERBOLOIDAL_HYPERBOLIC_SAMPLED_NORMAL_DEVICE_HPP_
#include "z4c/hyperboloidal/hyperbolic_composite.hpp"
#include "z4c/hyperboloidal/hyperbolic_composite_device.hpp"

namespace z4c::hyperboloidal {
// Five native interior numerator samples for each owned surface row. The caller
// must provision halos for these rows' blocks, independently of volume ownership.
// No MPI, stage integration or geometric projection is performed here.
class HyperbolicSampledNormalDevice {
 public:
  using Device=HyperbolicCompositeDevice;
  using Buffer=Device::Buffer;using ConstBuffer=Device::ConstBuffer;
  using ExecutionSpace=Device::ExecutionSpace;using MemorySpace=Device::MemorySpace;
  static constexpr int nf=31;
  enum Status:int {success=0,invalid_input=1,invalid_geometry=2,invalid_source=3,nonfinite_output=4};
  struct Summary {int bad_rows=0,first_bad=-1,reason=success;};
 private:
  int first_,local_,first_block_=0,end_block_=0,n_,width_,padded_count_,donors_;
  Device::IntBuffer active_,status_;
  Device::GeometryBuffer geometry_;
  Buffer poles_;
  std::unique_ptr<ScriRadialDevice> radial_plan_;
  struct Kernel {
    Device::PointKernel point;Device::GeometryBuffer geometry;
    Buffer poles;Device::IntBuffer status;
    KOKKOS_INLINE_FUNCTION void operator()(int sample)const {
      status(sample)=success;for(int f=0;f<nf;++f)poles(nf*sample+f)=0;
      const auto map=geometry(sample);const auto*x=map.x;
      const Real r=sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
      Z4cJet<Real> u{};VectorJet<Real>B{},D{};Real correction[nf]{};
      if(point.BuildJet(sample,map,(1+r*r)/2,u,B,D,correction)!=Device::success){status(sample)=invalid_input;return;}
      if(!Geometry(u.metric).valid){status(sample)=invalid_geometry;return;}
      OmegaJet<Real> omega{};omega.omega=(1-r*r)/2;
      for(int a=0;a<3;++a){omega.gradient[a]=-x[a];omega.hessian[a][a]=-1;}
      SetStationaryOmegaNormal(u.alpha.value,u.beta.value,u.alpha.d,u.beta.d,omega);
      auto parts=ConformalRHS(u,omega,Real(4),Real(-.99));
      if(!HyperbolicDamping(x,u,omega,parts)){status(sample)=invalid_source;return;}
      poles(nf*sample)=parts.pole.chi;poles(nf*sample+13)=parts.pole.trace;
      poles(nf*sample+14)=parts.pole.theta;
      constexpr int ti[6]={0,0,0,1,1,2},tj[6]={0,1,2,1,2,2};
      for(int t=0;t<6;++t)poles(nf*sample+7+t)=parts.pole.a[ti[t]][tj[t]];
      for(int a=0;a<3;++a)poles(nf*sample+15+a)=parts.pole.lambda[a];
      for(int f=0;f<nf;++f)if(!Kokkos::isfinite(poles(nf*sample+f)))status(sample)=nonfinite_output;
    }
  };
  struct Reduce {
    using value_type=Summary;Device::IntBuffer status;int first,donors;
    KOKKOS_INLINE_FUNCTION void init(Summary&out)const{out=Summary{};out.first_bad=INT_MAX;}
    KOKKOS_INLINE_FUNCTION void join(Summary&a,const Summary&b)const{
      a.bad_rows+=b.bad_rows;if(b.first_bad<a.first_bad){a.first_bad=b.first_bad;a.reason=b.reason;}}
    KOKKOS_INLINE_FUNCTION void operator()(int row,Summary&out)const{
      for(int s=0;s<donors;++s)if(status(donors*row+s)!=success){
        ++out.bad_rows;if(first+row<out.first_bad){out.first_bad=first+row;out.reason=status(donors*row+s);}break;}
    }
  };
 public:
  HyperbolicSampledNormalDevice(const HyperbolicComposite&model,int first_surface,int local_count)
      :first_(first_surface),local_(local_count),n_(model.grid.n),width_(model.grid.padded),padded_count_(model.grid.count),
       donors_(model.scri.radial_nodes-1){
    const bool radial=model.radial_closure==HyperbolicComposite::RadialClosure::donor_seven;
    if(first_<0||local_<0||size_t(first_)+size_t(local_)>model.scri.rows.size()||model.scri.radial_nodes!=(radial?7:6)||
       model.grid.order!=4||model.grid.ng<3)throw std::invalid_argument("invalid sampled-normal owned range/grid");
    active_=Device::IntBuffer("sampled normal donor indices",donors_*local_);
    status_=Device::IntBuffer("sampled normal source status",donors_*local_);
    geometry_=Device::GeometryBuffer("sampled normal donor maps",donors_*local_);
    poles_=Buffer("owned interior pole samples",size_t(nf)*donors_*local_);
    std::vector<size_t> points;
    auto active=Kokkos::create_mirror_view(active_);auto geometry=Kokkos::create_mirror_view(geometry_);
    if(local_){first_block_=model.scri.rows[first_].block;end_block_=first_block_+1;}
    for(int local=0;local<local_;++local){const auto&row=model.scri.rows[first_+local];
      first_block_=std::min(first_block_,row.block);end_block_=std::max(end_block_,row.block+1);
      for(int s=0;s<donors_;++s){const size_t p=size_t(row.block)*n_*n_*n_+(row.k*n_+row.j)*n_+n_-donors_+s;
        active(donors_*local+s)=model.grid.active[p];geometry(donors_*local+s)=model.grid.geometry[p];points.push_back(p);}}
    Kokkos::deep_copy(active_,active);Kokkos::deep_copy(geometry_,geometry);
    if(radial){
      if(!model.radial_plan)throw std::invalid_argument("missing direct radial plan");
      radial_plan_=std::make_unique<ScriRadialDevice>(*model.radial_plan,points);
    }
  }
  Summary Evaluate(ConstBuffer padded,ConstBuffer surface={}){
    if(padded.extent(0)!=size_t(nf)*padded_count_)throw std::invalid_argument("sampled-normal padded shape mismatch");
    Device::PointKernel point{};point.padded=padded;point.active=active_;point.n=n_;
    point.width=width_;point.padded_count=padded_count_;
    if(radial_plan_)point.radial=radial_plan_->Bind(padded,surface);
    Kokkos::parallel_for("owned scri interior pole samples",Kokkos::RangePolicy<ExecutionSpace>(0,donors_*local_),
                        Kernel{point,geometry_,poles_,status_});
    Summary result;Kokkos::parallel_reduce("owned sampled-normal diagnostics",Kokkos::RangePolicy<ExecutionSpace>(0,local_),
                                         Reduce{status_,first_,donors_},result);
    if(result.first_bad==INT_MAX)result.first_bad=-1;return result;
  }
  const Buffer&LocalPoleSamples()const{return poles_;}
  int FirstBlock()const{return first_block_;}int EndBlock()const{return end_block_;}
  int FirstSurface()const{return first_;}int LocalCount()const{return local_;}
  static const char*Reason(int code){switch(code){case success:return "success";
    case invalid_input:return "nonfinite or nonpositive sampled input";case invalid_geometry:return "invalid sampled metric";
    case invalid_source:return "invalid sampled native numerator";case nonfinite_output:return "nonfinite sampled numerator";
    default:return "unknown sampled-normal status";}}
};
}  // namespace z4c::hyperboloidal
#endif  // Z4C_HYPERBOLOIDAL_HYPERBOLIC_SAMPLED_NORMAL_DEVICE_HPP_
