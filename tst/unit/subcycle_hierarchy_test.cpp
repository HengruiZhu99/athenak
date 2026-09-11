#include <iostream>
#include <stdexcept>
#include "driver/subcycle_hierarchy.hpp"
void Check(bool ok) { if (!ok) throw std::runtime_error("hierarchy regression"); }
int main() {
  for (int dim : {2,3}) {
    std::vector<subcycling::BlockKey> leaves{{0,1,0,0}};
    // Refine the origin root, then refine its origin child once more.
    for (int c=1; c<(1<<dim); ++c) leaves.push_back({1,c&1,(c>>1)&1,(c>>2)&1});
    for (int c=0; c<(1<<dim); ++c) leaves.push_back({2,c&1,(c>>1)&1,(c>>2)&1});
    subcycling::Hierarchy h(leaves,0,dim);
    Check(h.Nodes().size()==leaves.size()+2);
    int parents=0;
    for (const auto &node : h.Nodes()) {
      if (node.Covered()) {
        ++parents;
        for (int c=0; c<(1<<dim); ++c) {
          const auto &child=h.Nodes()[node.children[c]];
          Check(h.Nodes()[child.parent].key==node.key);
        }
      } else { Check(leaves[node.source_leaf]==node.key); }
    }
    Check(parents==2);
    Check(h.Find({0,1,0,0})>=0 && h.Find({0,2,0,0})==-1);
    for (int kind=0; kind<3; ++kind) {
      auto bad=leaves;
      if (kind==0) bad.push_back(leaves[0]);
      if (kind==1) bad.push_back({0,0,0,0});
      if (kind==2) bad.pop_back();
      bool failed=false;
      try { subcycling::Hierarchy invalid(bad,0,dim); }
      catch (const std::invalid_argument &) { failed=true; }
      Check(failed);
    }
  }
  std::cout << "PASS: sparse covered parents, leaf identity, 2D/3D children, invalid trees\n";
}
