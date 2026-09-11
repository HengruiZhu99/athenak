#ifndef Z4C_VERTEX_AUTHORITY_HPP_
#define Z4C_VERTEX_AUTHORITY_HPP_
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace z4c {
struct VertexAuthorityGroups {
  std::vector<int> levels, begin, end, contributors;
};
// sorted_indices is contiguous by canonical vertex group. Preserve that exact
// order within each group's finest-level contributors (and thus reduction order).
template <typename Level>
VertexAuthorityGroups BuildVertexAuthorities(const std::vector<int> &sorted_indices,
    const std::vector<int> &group_for, int groups, Level level) {
  if (groups<0) throw std::invalid_argument("negative vertex group count");
  VertexAuthorityGroups result;
  result.levels.assign(groups,std::numeric_limits<int>::min());
  result.begin.resize(groups); result.end.resize(groups);
  result.contributors.reserve(sorted_indices.size());
  int prior=-1;
  for (int index : sorted_indices) {
    if (index<0 || index>=static_cast<int>(group_for.size())) {
      throw std::invalid_argument("invalid vertex contributor index");
    }
    const int group=group_for[index];
    if (group<0 || group>=groups || group<prior) {
      throw std::invalid_argument("vertex contributors not ordered by group");
    }
    prior=group;
    result.levels[group]=std::max(result.levels[group],level(index));
  }
  std::size_t cursor=0;
  for (int group=0; group<groups; ++group) {
    result.begin[group]=static_cast<int>(result.contributors.size());
    while (cursor<sorted_indices.size() && group_for[sorted_indices[cursor]]==group) {
      const int index=sorted_indices[cursor++];
      if (level(index)==result.levels[group]) result.contributors.push_back(index);
    }
    result.end[group]=static_cast<int>(result.contributors.size());
    if (result.begin[group]==result.end[group]) {
      throw std::invalid_argument("vertex group has no finest-level authority");
    }
  }
  return result;
}
}  // namespace z4c
#endif  // Z4C_VERTEX_AUTHORITY_HPP_
