#include <cassert>
#include <stdexcept>
#include "mesh/block_ownership.hpp"

int main() {
  LocalBlockOwnership map;
  map.Reset({3,4,5},9);
  assert(map.Contiguous() && map.GlobalId(2)==5 && map.LocalId(4)==1);
  assert(map.LocalId(2)==-1 && map.LocalId(6)==-1 && map.LocalId(-1)==-1);
  map.Reset({7,1,5},9);
  assert(!map.Contiguous() && map.GlobalId(0)==7 && map.LocalId(1)==1);
  assert(map.LocalId(3)==-1 && map.LocalId(9)==-1);
  for (const auto &ids:{std::vector<int>{1,1},std::vector<int>{-1},std::vector<int>{9}}) {
    bool rejected=false;
    try {map.Reset(ids,9);} catch(const std::invalid_argument &) {rejected=true;}
    assert(rejected && map.GlobalId(0)==7 && map.LocalId(5)==2);
  }
  map.Reset({},9);
  assert(map.LocalId(0)==-1 && map.GlobalIds().empty());
}
