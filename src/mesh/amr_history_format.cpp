#include "mesh/amr_history_format.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>

namespace amr_history {
namespace {

class Cursor {
 public:
  explicit Cursor(const std::string &text) : text_(text) {}
  bool literal(const std::string &value) {
    if (text_.compare(pos_, value.size(), value) != 0) return false;
    pos_ += value.size();
    return true;
  }
  bool integer(std::int64_t *value) {
    const std::size_t begin = pos_;
    if (pos_ < text_.size() && text_[pos_] == '-') ++pos_;
    const std::size_t digits = pos_;
    while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') ++pos_;
    if (digits == pos_) return false;
    try {
      *value = std::stoll(text_.substr(begin, pos_ - begin));
    } catch (...) { return false; }
    return true;
  }
  bool string(std::string *value) {
    if (!literal("\"")) return false;
    const std::size_t begin = pos_;
    while (pos_ < text_.size() && text_[pos_] != '"') {
      if (text_[pos_] == '\\' || static_cast<unsigned char>(text_[pos_]) < 0x20) return false;
      ++pos_;
    }
    if (pos_ == text_.size()) return false;
    *value = text_.substr(begin, pos_ - begin);
    return literal("\"");
  }
  bool boolean(bool *value) {
    if (literal("true")) { *value = true; return true; }
    if (literal("false")) { *value = false; return true; }
    return false;
  }
  bool end() const { return pos_ == text_.size(); }
 private:
  const std::string &text_;
  std::size_t pos_ = 0;
};

bool ParseInt(Cursor *cursor, int *value) {
  std::int64_t parsed = 0;
  if (!cursor->integer(&parsed) || parsed < std::numeric_limits<int>::min() ||
      parsed > std::numeric_limits<int>::max()) return false;
  *value = static_cast<int>(parsed);
  return true;
}

bool IsAncestor(const Location &ancestor, const Location &descendant) {
  if (ancestor.level >= descendant.level) return false;
  const int shift = descendant.level - ancestor.level;
  return (descendant.lx1 >> shift) == ancestor.lx1 &&
         (descendant.lx2 >> shift) == ancestor.lx2 &&
         (descendant.lx3 >> shift) == ancestor.lx3;
}

bool ParseLocation(Cursor *cursor, Location *location) {
  std::int64_t level, x1, x2, x3;
  if (!cursor->literal("[") || !cursor->integer(&level) || !cursor->literal(",") ||
      !cursor->integer(&x1) || !cursor->literal(",") || !cursor->integer(&x2) ||
      !cursor->literal(",") || !cursor->integer(&x3) || !cursor->literal("]")) return false;
  const auto valid32 = [](std::int64_t value) {
    return value >= std::numeric_limits<std::int32_t>::min() &&
           value <= std::numeric_limits<std::int32_t>::max();
  };
  if (!valid32(level) || !valid32(x1) || !valid32(x2) || !valid32(x3)) return false;
  *location = {static_cast<std::int32_t>(level), static_cast<std::int32_t>(x1),
               static_cast<std::int32_t>(x2), static_cast<std::int32_t>(x3)};
  return true;
}

std::string HeaderBase(const Header &h) {
  std::ostringstream out;
  out << "{\"type\":\"header\",\"schema\":" << h.schema
      << ",\"dimension\":" << h.dimension
      << ",\"symmetry\":\"" << h.symmetry
      << "\",\"coordinate_map\":\"" << h.coordinate_map
      << "\",\"root_level\":" << h.root_level
      << ",\"refinement_ratio\":" << h.refinement_ratio
      << ",\"root_blocks\":[" << h.root_blocks[0] << ',' << h.root_blocks[1] << ','
      << h.root_blocks[2] << "],\"domain_hex\":[";
  for (int i = 0; i < 6; ++i) out << (i ? ",\"" : "\"") << h.domain_hex[i] << '"';
  out << "],\"periodic\":[" << (h.periodic[0] ? "true" : "false") << ','
      << (h.periodic[1] ? "true" : "false") << ','
      << (h.periodic[2] ? "true" : "false")
      << "],\"max_level\":" << h.max_level << ",\"real_bytes\":" << h.real_bytes
      << ",\"cells_per_meshblock\":[" << h.cells_per_meshblock[0] << ','
      << h.cells_per_meshblock[1] << ',' << h.cells_per_meshblock[2]
      << "],\"source_id\":\"" << h.source_id << '"';
  return out.str();
}

std::string EventBase(const Event &e) {
  std::ostringstream out;
  out << "{\"type\":\"event\",\"event\":" << e.index
      << ",\"time\":\"" << e.time_decimal << "\",\"time_hex\":\"" << e.time_hex
      << "\",\"cycle\":" << e.cycle << ",\"leaves\":" << TreeCanonical(e.leaves)
      << ",\"leaf_count\":" << e.leaf_count << ",\"max_level\":" << e.max_level
      << ",\"requested_refine\":" << e.requested_refine
      << ",\"requested_derefine\":" << e.requested_derefine
      << ",\"created\":" << e.created << ",\"deleted\":" << e.deleted
      << ",\"balance_induced\":" << e.balance_induced
      << ",\"tree_checksum\":\"" << e.tree_checksum << '"';
  return out.str();
}

bool Overlap(std::int64_t a0, std::int64_t a1, std::int64_t b0, std::int64_t b1) {
  return std::max(a0, b0) < std::min(a1, b1);
}

}  // namespace

bool operator==(const Location &a, const Location &b) {
  return a.level == b.level && a.lx1 == b.lx1 && a.lx2 == b.lx2 && a.lx3 == b.lx3;
}
bool operator<(const Location &a, const Location &b) {
  if (a.level != b.level) return a.level < b.level;
  if (a.lx3 != b.lx3) return a.lx3 < b.lx3;
  if (a.lx2 != b.lx2) return a.lx2 < b.lx2;
  return a.lx1 < b.lx1;
}

std::string Checksum(const std::string &text) {
  std::uint64_t value = UINT64_C(14695981039346656037);
  for (unsigned char byte : text) {
    value ^= byte;
    value *= UINT64_C(1099511628211);
  }
  std::ostringstream out;
  out << std::hex << std::setfill('0') << std::setw(16) << value;
  return out.str();
}

std::string HexReal(double value) {
  std::ostringstream out;
  out << std::hexfloat << value;
  return out.str();
}
std::string DecimalReal(double value) {
  std::ostringstream out;
  out << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
  return out.str();
}
bool ParseReal(const std::string &text, double *value) {
  try {
    std::size_t used = 0;
    *value = std::stod(text, &used);
    return used == text.size() && std::isfinite(*value);
  } catch (...) { return false; }
}

std::string TreeCanonical(std::vector<Location> leaves) {
  std::sort(leaves.begin(), leaves.end());
  std::ostringstream out;
  out << '[';
  for (std::size_t i = 0; i < leaves.size(); ++i) {
    if (i) out << ',';
    out << '[' << leaves[i].level << ',' << leaves[i].lx1 << ',' << leaves[i].lx2 << ','
        << leaves[i].lx3 << ']';
  }
  out << ']';
  return out.str();
}
std::string TreeChecksum(std::vector<Location> leaves) { return Checksum(TreeCanonical(leaves)); }

std::string EncodeHeader(Header header) {
  header.checksum = Checksum(HeaderBase(header));
  return HeaderBase(header) + ",\"checksum\":\"" + header.checksum + "\"}";
}

bool DecodeHeader(const std::string &line, Header *h, std::string *error) {
  Cursor c(line);
  std::int64_t schema = 0;
  if (!c.literal("{\"type\":\"header\",\"schema\":") || !c.integer(&schema) ||
      !c.literal(",\"dimension\":") || !ParseInt(&c, &h->dimension) ||
      !c.literal(",\"symmetry\":") || !c.string(&h->symmetry) ||
      !c.literal(",\"coordinate_map\":") || !c.string(&h->coordinate_map) ||
      !c.literal(",\"root_level\":") || !ParseInt(&c, &h->root_level) ||
      !c.literal(",\"refinement_ratio\":") || !ParseInt(&c, &h->refinement_ratio) ||
      !c.literal(",\"root_blocks\":[")) {
    *error = "malformed history header"; return false;
  }
  h->schema = static_cast<int>(schema);
  for (int i = 0; i < 3; ++i) {
    if ((i && !c.literal(",")) || !ParseInt(&c, &h->root_blocks[i])) {
      *error = "malformed root_blocks"; return false;
    }
  }
  if (!c.literal("],\"domain_hex\":[")) { *error = "missing domain_hex"; return false; }
  for (int i = 0; i < 6; ++i) {
    if ((i && !c.literal(",")) || !c.string(&h->domain_hex[i])) {
      *error = "malformed domain_hex"; return false;
    }
  }
  if (!c.literal("],\"periodic\":[")) { *error = "missing periodic"; return false; }
  for (int i = 0; i < 3; ++i) {
    if ((i && !c.literal(",")) || !c.boolean(&h->periodic[i])) {
      *error = "malformed periodic"; return false;
    }
  }
  if (!c.literal("],\"max_level\":") || !ParseInt(&c, &h->max_level) ||
      !c.literal(",\"real_bytes\":") || !ParseInt(&c, &h->real_bytes) ||
      !c.literal(",\"cells_per_meshblock\":[")) {
    *error = "malformed header tail"; return false;
  }
  for (int i = 0; i < 3; ++i) {
    if ((i && !c.literal(",")) || !ParseInt(&c, &h->cells_per_meshblock[i])) {
      *error = "malformed cells_per_meshblock"; return false;
    }
  }
  if (!c.literal("],\"source_id\":") || !c.string(&h->source_id) ||
      !c.literal(",\"checksum\":") || !c.string(&h->checksum) || !c.literal("}") ||
      !c.end()) { *error = "malformed header checksum"; return false; }
  if (EncodeHeader(*h) != line) { *error = "noncanonical or checksum-invalid header"; return false; }
  return ValidateHeader(*h, error);
}

std::string EncodeEvent(Event event) {
  std::sort(event.leaves.begin(), event.leaves.end());
  event.leaf_count = static_cast<int>(event.leaves.size());
  event.max_level = 0;
  for (const auto &loc : event.leaves) event.max_level = std::max(event.max_level, int(loc.level));
  event.tree_checksum = TreeChecksum(event.leaves);
  event.checksum = Checksum(EventBase(event));
  return EventBase(event) + ",\"checksum\":\"" + event.checksum + "\"}";
}

bool DecodeEvent(const std::string &line, Event *e, std::string *error) {
  Cursor c(line);
  std::int64_t index = 0;
  if (!c.literal("{\"type\":\"event\",\"event\":") || !c.integer(&index) || index < 0 ||
      !c.literal(",\"time\":") || !c.string(&e->time_decimal) ||
      !c.literal(",\"time_hex\":") || !c.string(&e->time_hex) ||
      !c.literal(",\"cycle\":") || !c.integer(&e->cycle) ||
      !c.literal(",\"leaves\":[")) { *error = "malformed event prefix"; return false; }
  e->index = static_cast<std::uint64_t>(index);
  e->leaves.clear();
  if (!c.literal("]")) {
    while (true) {
      Location loc;
      if (!ParseLocation(&c, &loc)) { *error = "malformed event leaves"; return false; }
      e->leaves.push_back(loc);
      if (c.literal("]")) break;
      if (!c.literal(",")) { *error = "malformed event leaf separator"; return false; }
    }
  }
  if (!c.literal(",\"leaf_count\":") || !ParseInt(&c, &e->leaf_count) ||
      !c.literal(",\"max_level\":") || !ParseInt(&c, &e->max_level) ||
      !c.literal(",\"requested_refine\":") || !ParseInt(&c, &e->requested_refine) ||
      !c.literal(",\"requested_derefine\":") || !ParseInt(&c, &e->requested_derefine) ||
      !c.literal(",\"created\":") || !ParseInt(&c, &e->created) ||
      !c.literal(",\"deleted\":") || !ParseInt(&c, &e->deleted) ||
      !c.literal(",\"balance_induced\":") || !ParseInt(&c, &e->balance_induced) ||
      !c.literal(",\"tree_checksum\":") || !c.string(&e->tree_checksum) ||
      !c.literal(",\"checksum\":") || !c.string(&e->checksum) || !c.literal("}") ||
      !c.end()) { *error = "malformed event tail"; return false; }
  double dec = 0.0, hex = 0.0;
  if (!ParseReal(e->time_decimal, &dec) || !ParseReal(e->time_hex, &hex) || dec != hex) {
    *error = "event time encodings disagree"; return false;
  }
  if (EncodeEvent(*e) != line) { *error = "noncanonical or checksum-invalid event"; return false; }
  return true;
}

bool ValidateHeader(const Header &h, std::string *error) {
  if (h.schema != 1 || h.dimension < 1 || h.dimension > 3 || h.refinement_ratio != 2 ||
      h.root_level < 0 || h.max_level < h.root_level || h.max_level > 31 ||
      (h.real_bytes != 4 && h.real_bytes != 8)) {
    *error = "unsupported history header"; return false;
  }
  for (int d = 0; d < 3; ++d) {
    if (h.root_blocks[d] < 1 || h.cells_per_meshblock[d] < 1) {
      *error = "nonpositive grid extent in header"; return false;
    }
    if (d >= h.dimension && h.root_blocks[d] != 1) {
      *error = "nontrivial collapsed root dimension"; return false;
    }
  }
  for (const auto &bound : h.domain_hex) {
    double value;
    if (!ParseReal(bound, &value)) { *error = "invalid domain bound"; return false; }
  }
  return true;
}

bool Compatible(const Header &a, const Header &b, std::string *error) {
  if (a.schema != b.schema) { *error = "history schema mismatch"; return false; }
  if (a.dimension != b.dimension) { *error = "history dimension mismatch"; return false; }
  if (a.symmetry != b.symmetry) { *error = "history symmetry mismatch"; return false; }
  if (a.coordinate_map != b.coordinate_map) { *error = "history coordinate-map mismatch"; return false; }
  if (a.root_level != b.root_level) { *error = "history root-level mismatch"; return false; }
  if (a.refinement_ratio != b.refinement_ratio) { *error = "history refinement-ratio mismatch"; return false; }
  if (a.root_blocks != b.root_blocks) { *error = "history root-block mismatch"; return false; }
  if (a.domain_hex != b.domain_hex) { *error = "history physical-domain mismatch"; return false; }
  if (a.periodic != b.periodic) { *error = "history periodicity mismatch"; return false; }
  if (a.max_level != b.max_level) { *error = "history maximum-level mismatch"; return false; }
  if (a.real_bytes != b.real_bytes) { *error = "history Real-size mismatch"; return false; }
  if (a.source_id != b.source_id) { *error = "history source-id mismatch"; return false; }
  return true;
}

bool ValidateTree(const Header &h, std::vector<Location> leaves, std::string *error) {
  if (leaves.empty()) { *error = "empty leaf tree"; return false; }
  std::sort(leaves.begin(), leaves.end());
  if (std::adjacent_find(leaves.begin(), leaves.end()) != leaves.end()) {
    *error = "duplicate leaf"; return false;
  }
  for (std::size_t i = 0; i < leaves.size(); ++i) {
    const auto &a = leaves[i];
    if (a.level < h.root_level || a.level > h.max_level) {
      *error = "leaf level outside header bounds"; return false;
    }
    const int shift = a.level - h.root_level;
    const std::int64_t limits[3] = {
      std::int64_t(h.root_blocks[0]) << shift, std::int64_t(h.root_blocks[1]) << shift,
      std::int64_t(h.root_blocks[2]) << shift};
    if (a.lx1 < 0 || a.lx1 >= limits[0] || a.lx2 < 0 || a.lx2 >= limits[1] ||
        a.lx3 < 0 || a.lx3 >= limits[2]) { *error = "leaf outside root domain"; return false; }
    for (std::size_t j = i + 1; j < leaves.size(); ++j) {
      if (IsAncestor(a, leaves[j]) || IsAncestor(leaves[j], a)) {
        *error = "ancestor/descendant leaf overlap"; return false;
      }
    }
  }
  long double coverage = 0.0L;
  for (const auto &loc : leaves) coverage += std::ldexp(1.0L, h.dimension * (h.max_level - loc.level));
  const long double expected = static_cast<long double>(h.root_blocks[0]) *
      h.root_blocks[1] * h.root_blocks[2] *
      std::ldexp(1.0L, h.dimension * (h.max_level - h.root_level));
  if (coverage != expected) { *error = "leaf tree does not cover the root domain"; return false; }

  const int max_level = h.max_level;
  struct Box { std::int64_t lo[3], hi[3]; int level; };
  std::vector<Box> boxes;
  boxes.reserve(leaves.size());
  for (const auto &loc : leaves) {
    const int s = max_level - loc.level;
    Box b{{std::int64_t(loc.lx1) << s, std::int64_t(loc.lx2) << s,
           std::int64_t(loc.lx3) << s},
          {std::int64_t(loc.lx1 + 1) << s, std::int64_t(loc.lx2 + 1) << s,
           std::int64_t(loc.lx3 + 1) << s}, loc.level};
    boxes.push_back(b);
  }
  const std::int64_t domain[3] = {
      std::int64_t(h.root_blocks[0]) << (max_level - h.root_level),
      std::int64_t(h.root_blocks[1]) << (max_level - h.root_level),
      std::int64_t(h.root_blocks[2]) << (max_level - h.root_level)};
  for (std::size_t i = 0; i < boxes.size(); ++i) for (std::size_t j = i + 1; j < boxes.size(); ++j) {
    bool face_neighbor = false;
    for (int d = 0; d < h.dimension; ++d) {
      const bool touch = boxes[i].hi[d] == boxes[j].lo[d] || boxes[j].hi[d] == boxes[i].lo[d] ||
          (h.periodic[d] && ((boxes[i].hi[d] == domain[d] && boxes[j].lo[d] == 0) ||
                             (boxes[j].hi[d] == domain[d] && boxes[i].lo[d] == 0)));
      if (!touch) continue;
      bool overlaps = true;
      for (int q = 0; q < h.dimension; ++q) if (q != d) {
        overlaps = overlaps && Overlap(boxes[i].lo[q], boxes[i].hi[q], boxes[j].lo[q], boxes[j].hi[q]);
      }
      face_neighbor = face_neighbor || overlaps;
    }
    if (face_neighbor && std::abs(boxes[i].level - boxes[j].level) > 1) {
      *error = "leaf tree violates 2:1 face balance"; return false;
    }
  }
  return true;
}

bool ValidateEvents(const Header &h, const std::vector<Event> &events, std::string *error) {
  if (events.empty()) { *error = "history has no initial event"; return false; }
  double prior = -std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < events.size(); ++i) {
    if (events[i].index != i || !ValidateTree(h, events[i].leaves, error)) return false;
    double time;
    if (!ParseReal(events[i].time_hex, &time) || time < prior || (i > 0 && TimeEqual(time, prior))) {
      *error = "history event times are not strictly increasing"; return false;
    }
    if (i > 0 && events[i].leaves == events[i - 1].leaves) {
      *error = "consecutive events repeat the same tree"; return false;
    }
    prior = time;
  }
  return true;
}

bool AppendOnlyExtension(const std::vector<Event> &authority,
                         const std::vector<Event> &extension,
                         std::string *error) {
  if (extension.size() <= authority.size()) {
    *error = "AMR history extension does not append an event";
    return false;
  }
  for (std::size_t index = 0; index < authority.size(); ++index) {
    if (EncodeEvent(authority[index]) != EncodeEvent(extension[index])) {
      *error = "AMR history extension changes the authenticated authority prefix";
      return false;
    }
  }
  return true;
}

bool AuthenticatedBranch(const std::vector<Event> &authority,
                         const std::vector<Event> &branch,
                         std::size_t base_event, std::string *error) {
  if (base_event >= authority.size()) {
    *error = "AMR history branch base is outside the authenticated authority";
    return false;
  }
  if (branch.size() <= base_event + 1) {
    *error = "AMR history branch has no event after its authenticated base";
    return false;
  }
  for (std::size_t index = 0; index <= base_event; ++index) {
    if (EncodeEvent(authority[index]) != EncodeEvent(branch[index])) {
      *error = "AMR history branch changes its authenticated authority prefix";
      return false;
    }
  }
  if (base_event + 1 < authority.size() &&
      EncodeEvent(authority[base_event + 1]) == EncodeEvent(branch[base_event + 1])) {
    *error = "AMR history branch does not diverge after its declared base";
    return false;
  }
  return true;
}

bool DeriveTransition(const Header &h, std::vector<Location> current,
                      std::vector<Location> target, Transition *result, std::string *error) {
  if (!ValidateTree(h, current, error) || !ValidateTree(h, target, error)) return false;
  std::sort(current.begin(), current.end());
  std::sort(target.begin(), target.end());
  result->flags.assign(current.size(), 0);
  result->refine_parents = 0;
  result->derefine_leaves = 0;
  std::set<Location> target_set(target.begin(), target.end());
  const int nchild = 1 << h.dimension;
  for (std::size_t i = 0; i < current.size(); ++i) {
    const auto &loc = current[i];
    if (target_set.count(loc)) continue;
    int descendants = 0;
    for (const auto &candidate : target) if (IsAncestor(loc, candidate)) {
      if (candidate.level != loc.level + 1) { *error = "target requires multiple refinement generations"; return false; }
      ++descendants;
    }
    if (descendants) {
      if (descendants != nchild) { *error = "target has incomplete refined sibling group"; return false; }
      result->flags[i] = 1;
      ++result->refine_parents;
      continue;
    }
    bool has_parent = false;
    for (const auto &candidate : target) if (IsAncestor(candidate, loc)) {
      if (candidate.level != loc.level - 1) { *error = "target requires multiple derefinement generations"; return false; }
      has_parent = true;
    }
    if (!has_parent) { *error = "current leaf has no target representation"; return false; }
    result->flags[i] = -1;
    ++result->derefine_leaves;
  }
  std::vector<Location> shadow;
  for (std::size_t i = 0; i < current.size(); ++i) {
    const auto &loc = current[i];
    if (result->flags[i] == 0) shadow.push_back(loc);
    if (result->flags[i] == 1) {
      for (int child = 0; child < nchild; ++child) {
        shadow.push_back({loc.level + 1, 2 * loc.lx1 + (child & 1),
                          2 * loc.lx2 + ((child >> 1) & (h.dimension >= 2 ? 1 : 0)),
                          2 * loc.lx3 + ((child >> 2) & (h.dimension >= 3 ? 1 : 0))});
      }
    }
  }
  for (const auto &candidate : target) {
    bool parent_of_derefined = false;
    for (std::size_t i = 0; i < current.size(); ++i) {
      if (result->flags[i] == -1 && IsAncestor(candidate, current[i])) parent_of_derefined = true;
    }
    if (parent_of_derefined) shadow.push_back(candidate);
  }
  std::sort(shadow.begin(), shadow.end());
  shadow.erase(std::unique(shadow.begin(), shadow.end()), shadow.end());
  if (shadow != target) { *error = "derived one-transaction tree does not equal target"; return false; }
  return true;
}

bool TimeEqual(double a, double b) {
  const double scale = std::max({1.0, std::abs(a), std::abs(b)});
  // A replay at a different resolution reaches a scheduled physical time by a
  // different sequence of floating-point additions.  Treat the resulting
  // bounded accumulation error as the same coordinate time instead of taking
  // a near-zero PDE step to consume the last few ulps.  Even at the largest
  // campaign times this tolerance remains many orders of magnitude below the
  // smallest physical CFL timestep.
  return std::abs(a - b) <= 32.0 * std::numeric_limits<double>::epsilon() * scale;
}
bool LimitTimestep(double time, double next, double *dt, std::string *error) {
  if (!std::isfinite(time) || !std::isfinite(next) || !std::isfinite(*dt) || *dt <= 0.0) {
    *error = "invalid replay timestep input"; return false;
  }
  if (next < time && !TimeEqual(next, time)) { *error = "next replay event is in the past"; return false; }
  if (TimeEqual(next, time)) return true;
  // Preserve an unmodified production timestep when its rounded endpoint is already the
  // recorded event time.  Computing `next - time` can be one ulp smaller than that same
  // timestep, even though `time + dt` rounds exactly to `next`; clipping in that case
  // needlessly perturbs an otherwise identical same-resolution replay.
  if (time + *dt == next) return true;
  const double remaining = next - time;
  if (remaining > 0.0 && *dt > remaining) *dt = remaining;
  if (*dt <= 0.0 || time + *dt == time) { *error = "replay timestep would not advance"; return false; }
  return true;
}

}  // namespace amr_history
