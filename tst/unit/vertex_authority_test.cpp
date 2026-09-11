#include <iostream>
#include <numeric>
#include <random>
#include "z4c/vertex_authority.hpp"
void Check(bool ok) { if (!ok) throw std::runtime_error("authority regression"); }
int main() {
  std::mt19937 generator(713);
  for (int groups : {0,1,7,100,1000}) {
    std::vector<int> group_for,levels;
    for (int g=0; g<groups; ++g) {
      for (int n=0, count=1+generator()%8; n<count; ++n) {
        group_for.push_back(g); levels.push_back(generator()%20);
      }
    }
    std::vector<int> order(group_for.size()); std::iota(order.begin(),order.end(),0);
    std::shuffle(order.begin(),order.end(),generator);
    std::stable_sort(order.begin(),order.end(),[&](int a,int b){return group_for[a]<group_for[b];});
    auto actual=z4c::BuildVertexAuthorities(order,group_for,groups,[&](int i){return levels[i];});
    std::vector<int> expected;
    for (int g=0; g<groups; ++g) {
      int maximum=-1;
      for (int i : order) if (group_for[i]==g) maximum=std::max(maximum,levels[i]);
      Check(actual.levels[g]==maximum && actual.begin[g]==static_cast<int>(expected.size()));
      for (int i : order) if (group_for[i]==g && levels[i]==maximum) expected.push_back(i);
      Check(actual.end[g]==static_cast<int>(expected.size()));
    }
    Check(actual.contributors==expected);
  }
  bool rejected=false;
  try { z4c::BuildVertexAuthorities({1,0},{0,1},2,[](int){return 0;}); }
  catch (const std::invalid_argument &) { rejected=true; }
  Check(rejected);
  std::cout << "PASS: exact authority membership and reduction order against quadratic reference\n";
}
