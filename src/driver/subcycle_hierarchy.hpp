#ifndef DRIVER_SUBCYCLE_HIERARCHY_HPP_
#define DRIVER_SUBCYCLE_HIERARCHY_HPP_
#include <array>
#include <map>
#include <stdexcept>
#include <vector>

namespace subcycling {
// Keys are (logical level,x,y,z). Unlike the evolution leaf tree, this plan
// retains covered ancestors. Those nodes need real predictor states, not the
// existing transfer cache. SourceLeaf indexes the original leaf list only.
using BlockKey = std::array<int,4>;
struct HierarchyNode {
  BlockKey key;
  int source_leaf=-1, parent=-1;
  std::array<int,8> children{{-1,-1,-1,-1,-1,-1,-1,-1}};
  bool Covered() const { return source_leaf < 0; }
};
class Hierarchy {
 public:
  Hierarchy(const std::vector<BlockKey> &leaves, int root, int dimension)
      : dimension_(dimension) {
    if (root < 0 || (dimension != 2 && dimension != 3) || leaves.empty()) {
      throw std::invalid_argument("invalid subcycling hierarchy shape");
    }
    std::map<BlockKey,int> sources;
    for (int m=0; m<static_cast<int>(leaves.size()); ++m) {
      auto key=leaves[m];
      if (key[0]<root || key[1]<0 || key[2]<0 || key[3]<0 ||
          (dimension==2 && key[3]!=0) || sources.count(key)) {
        throw std::invalid_argument("invalid or duplicate subcycling leaf");
      }
      sources.emplace(key,m);
      index_.emplace(key,0);
      while (key[0]>root) { key=ParentKey(key); index_.emplace(key,0); }
    }
    for (auto &entry : index_) {
      entry.second=static_cast<int>(nodes_.size());
      HierarchyNode node; node.key=entry.first;
      auto found=sources.find(entry.first);
      if (found!=sources.end()) node.source_leaf=found->second;
      nodes_.push_back(node);
    }
    for (int m=0; m<static_cast<int>(nodes_.size()); ++m) {
      auto &node=nodes_[m];
      if (node.key[0]==root) continue;
      node.parent=index_.at(ParentKey(node.key));
      auto &parent=nodes_[node.parent];
      if (!parent.Covered()) throw std::invalid_argument("overlapping hierarchy leaves");
      const int slot=(node.key[1]&1)+2*(node.key[2]&1)+4*(node.key[3]&1);
      parent.children[slot]=m;
    }
    for (const auto &node : nodes_) {
      if (!node.Covered()) continue;
      for (int c=0; c<(1<<dimension); ++c) {
        if (node.children[c]<0) throw std::invalid_argument("incomplete refined parent");
      }
    }
  }
  int Dimension() const { return dimension_; }
  const std::vector<HierarchyNode> &Nodes() const { return nodes_; }
  int Find(const BlockKey &key) const {
    auto found=index_.find(key); return found==index_.end() ? -1 : found->second;
  }
 private:
  static BlockKey ParentKey(BlockKey key) {
    --key[0]; key[1]/=2; key[2]/=2; key[3]/=2; return key;
  }
  int dimension_;
  std::map<BlockKey,int> index_;
  std::vector<HierarchyNode> nodes_;
};
}  // namespace subcycling
#endif  // DRIVER_SUBCYCLE_HIERARCHY_HPP_
