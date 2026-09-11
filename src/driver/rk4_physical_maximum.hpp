#ifndef DRIVER_RK4_PHYSICAL_MAXIMUM_HPP_
#define DRIVER_RK4_PHYSICAL_MAXIMUM_HPP_
#include <map>
#include <set>
#include "driver/rk4_predictor_states.hpp"
namespace subcycling {
// Local maximum of |a*u_i+b*u_j| at one physical time. For Z4c use
// K = Khat + 2 Theta. Caller chooses ONLY physical leaves and combines level
// maxima (and rank maxima) evaluated at the same time, never asynchronous u0.
class RK4PhysicalMaximum {
 public:
  void Build(const std::vector<int> &sources,const std::vector<int> &leaves,
             int first,int second,Real a=1,Real b=2) {
    ready_=false;
    if(first<0 || second<0 || !std::isfinite(a) || !std::isfinite(b))
      throw std::invalid_argument("invalid physical maximum components");
    std::map<int,int> packed;
    for(int n=0;n<static_cast<int>(sources.size());++n)
      if(sources[n]<0 || !packed.emplace(sources[n],n).second)
        throw std::invalid_argument("invalid physical maximum sources");
    std::set<int> unique;
    ids_=DvceArray1D<int>("physical maximum leaf indices",leaves.size());
    auto host=Kokkos::create_mirror_view(ids_);
    for(int n=0;n<static_cast<int>(leaves.size());++n) {
      if(!packed.count(leaves[n]) || !unique.insert(leaves[n]).second)
        throw std::invalid_argument("invalid physical maximum leaves");
      host(n)=packed.at(leaves[n]);
    }
    Kokkos::deep_copy(ids_,host);sources_=sources;
    first_=first;second_=second;a_=a;b_=b;ready_=true;
  }
  Real Evaluate(const RK4PredictorStates &history,double fraction) {
    if(!ready_ || history.SourceBlocks()!=sources_)
      throw std::invalid_argument("physical maximum history mismatch");
    history.EvaluateValue(fraction,values_);
    if(first_>=values_.extent_int(1) || second_>=values_.extent_int(1))
      throw std::invalid_argument("physical maximum component out of range");
    const auto values=values_;const auto ids=ids_;
    const int ni=values.extent_int(4),nj=values.extent_int(3),nk=values.extent_int(2);
    const int first=first_,second=second_;const Real a=a_,b=b_;
    const std::size_t count=ids.extent(0)*ni*nj*nk;
    if(count==0) return 0;
    Real maximum=0;
    Kokkos::parallel_reduce("common-time physical leaf maximum",
      Kokkos::RangePolicy<DevExeSpace,Kokkos::IndexType<std::size_t>>(0,count),
      KOKKOS_LAMBDA(std::size_t q,Real &out) {
        const int i=q%ni;q/=ni;const int j=q%nj;q/=nj;const int k=q%nk;
        const int m=ids(q/nk);
        const Real x=a*values(m,first,k,j,i)+b*values(m,second,k,j,i);
        const Real v=Kokkos::isfinite(x)?fabs(x):INFINITY;
        if(v>out) out=v;
      },Kokkos::Max<Real>(maximum));
    if(!std::isfinite(maximum)) throw std::runtime_error("nonfinite common-time physical maximum");
    return maximum;
  }
 private:
  bool ready_=false;
  int first_=0,second_=0;
  Real a_=1,b_=2;
  std::vector<int> sources_;
  DvceArray1D<int> ids_;
  DvceArray5D<Real> values_;
};
}
#endif
