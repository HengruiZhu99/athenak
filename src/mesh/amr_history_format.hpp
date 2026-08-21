#ifndef MESH_AMR_HISTORY_FORMAT_HPP_
#define MESH_AMR_HISTORY_FORMAT_HPP_

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace amr_history {

struct Location {
  std::int32_t level = 0;
  std::int32_t lx1 = 0;
  std::int32_t lx2 = 0;
  std::int32_t lx3 = 0;
};

bool operator==(const Location &a, const Location &b);
bool operator<(const Location &a, const Location &b);

struct Header {
  int schema = 2;
  int dimension = 1;
  std::string symmetry = "cartesian3d";
  std::string coordinate_map = "cartesian";
  std::string grid_centering = "cell";
  int centering_schema = 1;
  int root_level = 0;
  int refinement_ratio = 2;
  std::array<int, 3> root_blocks{{1, 1, 1}};
  std::array<std::string, 6> domain_hex;
  std::array<bool, 3> periodic{{false, false, false}};
  int max_level = 31;
  int real_bytes = 8;
  std::array<int, 3> cells_per_meshblock{{1, 1, 1}};
  std::string source_id = "athena";
  std::string checksum;
};

struct Event {
  std::uint64_t index = 0;
  std::string time_decimal;
  std::string time_hex;
  std::int64_t cycle = 0;
  std::vector<Location> leaves;
  int leaf_count = 0;
  int max_level = 0;
  int requested_refine = 0;
  int requested_derefine = 0;
  int created = 0;
  int deleted = 0;
  int balance_induced = 0;
  std::string tree_checksum;
  std::string checksum;
};

struct Transition {
  std::vector<int> flags;
  int refine_parents = 0;
  int derefine_leaves = 0;
};

std::string Checksum(const std::string &text);
std::string HexReal(double value);
std::string DecimalReal(double value);
bool ParseReal(const std::string &text, double *value);

std::string TreeCanonical(std::vector<Location> leaves);
std::string TreeChecksum(std::vector<Location> leaves);

std::string EncodeHeader(Header header);
bool DecodeHeader(const std::string &line, Header *header, std::string *error);
std::string EncodeEvent(Event event);
bool DecodeEvent(const std::string &line, Event *event, std::string *error);

bool ValidateHeader(const Header &header, std::string *error);
bool Compatible(const Header &recorded, const Header &candidate, std::string *error);
bool ValidateTree(const Header &header, std::vector<Location> leaves, std::string *error);
bool ValidateEvents(const Header &header, const std::vector<Event> &events,
                    std::string *error);
bool AppendOnlyExtension(const std::vector<Event> &authority,
                         const std::vector<Event> &extension,
                         std::string *error);
bool AuthenticatedBranch(const std::vector<Event> &authority,
                         const std::vector<Event> &branch,
                         std::size_t base_event, std::string *error);
bool DeriveTransition(const Header &header, std::vector<Location> current,
                      std::vector<Location> target, Transition *transition,
                      std::string *error);

bool TimeEqual(double a, double b);
bool LimitTimestep(double time, double next_event_time, double *dt,
                   std::string *error);

}  // namespace amr_history

#endif  // MESH_AMR_HISTORY_FORMAT_HPP_
