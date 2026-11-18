#ifndef MESH_BLOCK_OWNERSHIP_HPP_
#define MESH_BLOCK_OWNERSHIP_HPP_

#include <stdexcept>
#include <utility>
#include <vector>

// Stable global identities are independent of dense local storage slots.
// The traditional consecutive-ID partition retains its arithmetic fast path.
class LocalBlockOwnership {
 public:
  void Reset(std::vector<int> ids, int global_count) {
    if (global_count<0) throw std::invalid_argument("Negative global block count");
    std::vector<int> inverse(global_count,-1);
    bool consecutive=true;
    for (std::size_t m=0;m<ids.size();++m) {
      const int gid=ids[m];
      if (gid<0 || gid>=global_count || inverse[gid]>=0)
        throw std::invalid_argument("Invalid or duplicate locally owned block ID");
      inverse[gid]=static_cast<int>(m);
      consecutive=consecutive && gid==ids.front()+static_cast<int>(m);
    }
    ids_=std::move(ids);
    consecutive_=consecutive;
    first_=ids_.empty()?0:ids_.front();
    if (consecutive_) inverse.clear();
    inverse_=std::move(inverse);
  }
  int GlobalId(int local) const {return ids_.at(local);}
  int LocalId(int gid) const {
    if (consecutive_)
      return gid>=first_ && gid-first_<static_cast<int>(ids_.size())?gid-first_:-1;
    return gid>=0 && gid<static_cast<int>(inverse_.size())?inverse_[gid]:-1;
  }
  bool Contiguous() const {return consecutive_;}
  const std::vector<int> &GlobalIds() const {return ids_;}

 private:
  std::vector<int> ids_,inverse_;
  int first_=0;
  bool consecutive_=true;
};
#endif
