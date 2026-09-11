#ifndef DRIVER_BLOCK_BATCHES_HPP_
#define DRIVER_BLOCK_BATCHES_HPP_
#include <map>
#include <vector>
#include "athena.hpp"

namespace subcycling {
// Persistent device block lists. This first consumer batches synchronous RHS
// evaluation by level; it does not permit exchanging asynchronous states.
class BlockBatches {
 public:
  void Update(bool enabled, const DualArray1D<int> &levels, int count) {
    enabled_ = enabled; count_ = count;
    if (!enabled) return;
    bool changed = static_cast<int>(cached_.size()) != count;
    if (!changed) {
      for (int m=0; m<count; ++m) changed |= cached_[m] != levels.h_view(m);
    }
    if (!changed) return;
    cached_.resize(count);
    std::map<int, std::vector<int>> groups;
    for (int m=0; m<count; ++m) {
      cached_[m] = levels.h_view(m);
      groups[cached_[m]].push_back(m);
    }
    Kokkos::realloc(indices_, count);
    auto host = Kokkos::create_mirror_view(indices_);
    ranges_.clear();
    int offset=0;
    for (const auto &group : groups) {
      ranges_.push_back({offset, static_cast<int>(group.second.size())});
      for (int m : group.second) host(offset++) = m;
    }
    Kokkos::deep_copy(indices_, host);
  }
  template <typename Kernel>
  void For4(const std::string &name, int ks, int ke, int js, int je,
            int is, int ie, Kernel kernel) const {
    if (!enabled_) {
      par_for(name, DevExeSpace(), 0, count_-1, ks,ke,js,je,is,ie,kernel);
      return;
    }
    auto indices=indices_;
    for (const auto &range : ranges_) {
      const int start=range.first;
      par_for(name, DevExeSpace(), 0,range.second-1,ks,ke,js,je,is,ie,
          KOKKOS_LAMBDA(int b, int k, int j, int i) {
        kernel(indices(start+b),k,j,i);
      });
    }
  }
  template <typename Kernel>
  void For5(const std::string &name, int ns, int ne, int ks, int ke,
            int js, int je, int is, int ie, Kernel kernel) const {
    if (!enabled_) {
      par_for(name,DevExeSpace(),0,count_-1,ns,ne,ks,ke,js,je,is,ie,kernel);
      return;
    }
    auto indices=indices_;
    for (const auto &range : ranges_) {
      const int start=range.first;
      par_for(name,DevExeSpace(),0,range.second-1,ns,ne,ks,ke,js,je,is,ie,
          KOKKOS_LAMBDA(int b, int n, int k, int j, int i) {
        kernel(indices(start+b),n,k,j,i);
      });
    }
  }
 private:
  bool enabled_=false;
  int count_=0;
  std::vector<int> cached_;
  std::vector<std::pair<int,int>> ranges_;
  DvceArray1D<int> indices_;
};
}  // namespace subcycling
#endif  // DRIVER_BLOCK_BATCHES_HPP_
