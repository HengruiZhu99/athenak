#include "mesh/amr_history.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <sys/stat.h>

#include "athena.hpp"
#include "config.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/meshblock_tree.hpp"
#include "z4c/z4c.hpp"
#include "z4c/z4c_symmetry.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {
constexpr const char *kRestartBlock = "amr_history_restart";

bool FileExists(const std::string &path) {
  struct stat info;
  return stat(path.c_str(), &info) == 0;
}

std::string ReadAll(const std::string &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) return {};
  return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

amr_history::Location Convert(const LogicalLocation &loc) {
  return {loc.level, loc.lx1, loc.lx2, loc.lx3};
}

}  // namespace

AMRHistory::AMRHistory(Mesh *mesh, ParameterInput *pin) : mesh_(mesh), pin_(pin) {
  if (!pin->DoesParameterExist("mesh_refinement", "amr_history_mode")) return;
  const std::string mode = pin->GetString("mesh_refinement", "amr_history_mode");
  if (mode == "off") return;
  if (mode == "record") mode_ = Mode::record;
  else if (mode == "replay") mode_ = Mode::replay;
  else Fatal("unknown <mesh_refinement>/amr_history_mode='" + mode + "'");
  if (!pin->DoesParameterExist("mesh_refinement", "amr_history_file")) {
    Fatal("active AMR history mode requires <mesh_refinement>/amr_history_file");
  }
  path_ = pin->GetString("mesh_refinement", "amr_history_file");
  if (path_.empty()) Fatal("amr_history_file must not be empty");
  ledger_path_ = record() ? path_ + ".ledger.jsonl"
                          : pin->GetString("job", "basename") + ".amr_history_replay.jsonl";
}

[[noreturn]] void AMRHistory::Fatal(const std::string &message) const {
  std::cerr << "### FATAL ERROR: AMR history: " << message << std::endl;
  std::exit(EXIT_FAILURE);
}

amr_history::Header AMRHistory::CurrentHeader() const {
  amr_history::Header h;
  h.dimension = mesh_->three_d ? 3 : (mesh_->two_d ? 2 : 1);
  if (mesh_->pmb_pack != nullptr && mesh_->pmb_pack->pz4c != nullptr) {
    h.symmetry = z4c::ToString(mesh_->pmb_pack->z4c_symmetry.mode);
    h.coordinate_map = z4c::ToString(mesh_->pmb_pack->z4c_symmetry.coordinate_map);
  } else {
    h.symmetry = pin_->DoesParameterExist("z4c", "symmetry")
        ? pin_->GetString("z4c", "symmetry") : "cartesian3d";
    h.coordinate_map = pin_->DoesParameterExist("z4c", "coordinate_map")
        ? pin_->GetString("z4c", "coordinate_map") : "cartesian_xyz";
  }
  h.root_level = mesh_->root_level;
  h.root_blocks = {{mesh_->nmb_rootx1, mesh_->nmb_rootx2, mesh_->nmb_rootx3}};
  h.domain_hex = {{amr_history::HexReal(mesh_->mesh_size.x1min),
                   amr_history::HexReal(mesh_->mesh_size.x1max),
                   amr_history::HexReal(mesh_->mesh_size.x2min),
                   amr_history::HexReal(mesh_->mesh_size.x2max),
                   amr_history::HexReal(mesh_->mesh_size.x3min),
                   amr_history::HexReal(mesh_->mesh_size.x3max)}};
  for (int d = 0; d < 3; ++d) h.periodic[d] = false;
  const char *inner[3] = {"ix1_bc", "ix2_bc", "ix3_bc"};
  const char *outer[3] = {"ox1_bc", "ox2_bc", "ox3_bc"};
  for (int d = 0; d < h.dimension; ++d) {
    h.periodic[d] = pin_->GetString("mesh", inner[d]) == "periodic" &&
                    pin_->GetString("mesh", outer[d]) == "periodic";
  }
  h.max_level = mesh_->max_level;
  h.real_bytes = sizeof(Real);
  h.cells_per_meshblock = {{mesh_->mb_indcs.nx1, mesh_->mb_indcs.nx2,
                            mesh_->mb_indcs.nx3}};
  h.source_id = std::string("athena-") + std::to_string(ATHENA_VERSION_MAJOR) + "." +
                std::to_string(ATHENA_VERSION_MINOR) + "-git-" + ATHENA_GIT_COMMIT_HASH;
  return h;
}

std::vector<amr_history::Location> AMRHistory::CurrentLeaves() const {
  std::vector<LogicalLocation> raw;
  mesh_->ptree->CollectLeafLocations(raw);
  std::vector<amr_history::Location> leaves;
  leaves.reserve(raw.size());
  for (const auto &loc : raw) leaves.push_back(Convert(loc));
  std::sort(leaves.begin(), leaves.end());
  return leaves;
}

void AMRHistory::LoadHistory() {
  std::ifstream input(path_);
  if (!input) Fatal("cannot open history file '" + path_ + "'");
  std::string line, error;
  if (!std::getline(input, line) || !amr_history::DecodeHeader(line, &header_, &error)) {
    Fatal("invalid history header: " + error);
  }
  events_.clear();
  while (std::getline(input, line)) {
    if (line.empty()) Fatal("blank or truncated history record");
    amr_history::Event event;
    if (!amr_history::DecodeEvent(line, &event, &error)) Fatal("invalid event: " + error);
    events_.push_back(std::move(event));
  }
  if (!input.eof()) Fatal("history read failed before EOF");
  if (!amr_history::ValidateEvents(header_, events_, &error)) Fatal(error);
  const auto candidate = CurrentHeader();
  if (!amr_history::Compatible(header_, candidate, &error)) Fatal(error);
  loaded_digest_ = FileDigest();
}

void AMRHistory::WriteFreshHistory() {
  if (FileExists(path_) || FileExists(ledger_path_)) {
    Fatal("fresh record output already exists");
  }
  header_ = CurrentHeader();
  std::string error;
  if (!amr_history::ValidateHeader(header_, &error)) Fatal(error);
  if (global_variable::my_rank == 0) {
    std::ofstream output(path_, std::ios::binary | std::ios::out);
    if (!output) Fatal("cannot create history file");
    output << amr_history::EncodeHeader(header_) << '\n';
    output.flush();
    if (!output) Fatal("failed to flush history header");
  }
#if MPI_PARALLEL_ENABLED
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  events_.clear();
  AppendEvent(0, 0);
}

bool AMRHistory::HasRestartCarrier() const {
  return pin_->DoesParameterExist(kRestartBlock, "schema");
}

void AMRHistory::LoadRestartCarrier() {
  if (pin_->GetInteger(kRestartBlock, "schema") != 1) Fatal("unsupported restart carrier");
  if (pin_->GetString(kRestartBlock, "mode") != (record() ? "record" : "replay")) {
    Fatal("restart carrier mode mismatch");
  }
  if (pin_->GetString(kRestartBlock, "history_digest") != loaded_digest_) {
    Fatal("restart history digest mismatch");
  }
  const std::string expected_bytes = pin_->GetString(kRestartBlock, "history_bytes");
  if (expected_bytes != std::to_string(FileSize())) Fatal("restart history length mismatch");
  const int last = pin_->GetInteger(kRestartBlock, "last_applied_event");
  const int next = pin_->GetInteger(kRestartBlock, "next_event");
  if (last < 0 || next < 0 || static_cast<std::size_t>(next) > events_.size()) {
    Fatal("restart cursor outside history");
  }
  last_applied_event_ = static_cast<std::size_t>(last);
  next_event_ = static_cast<std::size_t>(next);
  if (pin_->GetString(kRestartBlock, "tree_checksum") != CurrentTreeChecksum()) {
    Fatal("restart tree checksum mismatch");
  }
  if (!pin_->GetBoolean(kRestartBlock, "post_event")) Fatal("restart is not post-event");
}

void AMRHistory::Initialize(bool restart) {
  if (!active() || initialized_) return;
  if (!mesh_->adaptive) Fatal("record/replay requires adaptive refinement");
  if (replay() && !restart && FileExists(ledger_path_)) {
    Fatal("fresh replay ledger already exists");
  }
  if (record()) {
    if (!restart) {
      WriteFreshHistory();
      last_applied_event_ = 0;
      next_event_ = events_.size();
    } else {
      LoadHistory();
      if (!HasRestartCarrier()) Fatal("record restart lacks AMR history carrier");
      LoadRestartCarrier();
      if (next_event_ != events_.size()) Fatal("record restart cursor is not at file end");
    }
  } else {
    LoadHistory();
    if (restart && HasRestartCarrier()) {
      LoadRestartCarrier();
    } else {
      if (events_.empty()) Fatal("replay history has no initial event");
      double initial_time = 0.0;
      if (!amr_history::ParseReal(events_[0].time_hex, &initial_time) ||
          !amr_history::TimeEqual(mesh_->time, initial_time) ||
          CurrentLeaves() != events_[0].leaves) {
        Fatal("fresh/legacy replay origin does not equal event zero");
      }
      last_applied_event_ = 0;
      next_event_ = 1;
    }
  }
  initialized_ = true;
}

void AMRHistory::LimitTimestep() {
  if (!replay() || !initialized_ || next_event_ >= events_.size()) return;
  double next_time = 0.0;
  std::string error;
  if (!amr_history::ParseReal(events_[next_event_].time_hex, &next_time) ||
      !amr_history::LimitTimestep(mesh_->time, next_time, &mesh_->dt, &error)) Fatal(error);
}

bool AMRHistory::PrepareReplayFlags() {
  if (!replay() || !initialized_ || next_event_ >= events_.size()) return false;
  double event_time = 0.0;
  if (!amr_history::ParseReal(events_[next_event_].time_hex, &event_time)) Fatal("bad event time");
  if (!amr_history::TimeEqual(mesh_->time, event_time)) {
    if (mesh_->time > event_time) Fatal("replay advanced past next event");
    return false;
  }
  const auto current = CurrentLeaves();
  amr_history::Transition transition;
  std::string error;
  if (!amr_history::DeriveTransition(header_, current, events_[next_event_].leaves,
                                     &transition, &error)) Fatal(error);
  Kokkos::realloc(mesh_->pmr->refine_flag, mesh_->nmb_total);
  for (int gid = 0; gid < mesh_->nmb_total; ++gid) {
    const auto loc = Convert(mesh_->lloc_eachmb[gid]);
    const auto found = std::lower_bound(current.begin(), current.end(), loc);
    if (found == current.end() || !(*found == loc)) Fatal("GID leaf missing from sorted tree");
    const std::size_t index = static_cast<std::size_t>(found - current.begin());
    mesh_->pmr->refine_flag.h_view(gid) = transition.flags[index];
  }
  mesh_->pmr->refine_flag.template modify<HostMemSpace>();
  mesh_->pmr->refine_flag.template sync<DevExeSpace>();
  requested_refine_ = transition.refine_parents;
  requested_derefine_ = transition.derefine_leaves;
  replay_target_ = events_[next_event_].leaves;
  replay_event_pending_ = true;
  return true;
}

void AMRHistory::CaptureRequestedFlags() {
  requested_refine_ = 0;
  requested_derefine_ = 0;
  for (int gid = 0; gid < mesh_->nmb_total; ++gid) {
    requested_refine_ += mesh_->pmr->refine_flag.h_view(gid) > 0;
    requested_derefine_ += mesh_->pmr->refine_flag.h_view(gid) < 0;
  }
}

void AMRHistory::ValidateReplayProposedTree() {
  if (!replay_event_pending_) Fatal("replay tree validation without pending event");
  if (CurrentLeaves() != replay_target_) Fatal("production tree differs from replay target");
}

void AMRHistory::AppendEvent(int created, int deleted) {
  amr_history::Event event;
  event.index = events_.size();
  event.time_decimal = amr_history::DecimalReal(mesh_->time);
  event.time_hex = amr_history::HexReal(mesh_->time);
  event.cycle = mesh_->ncycle;
  event.leaves = CurrentLeaves();
  event.requested_refine = requested_refine_;
  event.requested_derefine = requested_derefine_;
  event.created = created;
  event.deleted = deleted;
  const int dimension_children = 1 << header_.dimension;
  const int explicit_created = requested_refine_ * (dimension_children - 1);
  event.balance_induced = std::max(0, created - explicit_created);
  const std::string encoded = amr_history::EncodeEvent(event);
  std::string error;
  if (!amr_history::DecodeEvent(encoded, &event, &error) ||
      !amr_history::ValidateTree(header_, event.leaves, &error)) Fatal(error);
  if (global_variable::my_rank == 0) {
    std::ofstream output(path_, std::ios::binary | std::ios::app);
    if (!output) Fatal("cannot append history event");
    output << encoded << '\n';
    output.flush();
    if (!output) Fatal("failed to flush history event");
  }
#if MPI_PARALLEL_ENABLED
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  events_.push_back(event);
  last_applied_event_ = event.index;
  next_event_ = events_.size();
  AppendLedger("record", event, true);
  if (global_variable::my_rank == 0) {
    std::cout << "AMR_HISTORY_RECORD event=" << event.index << " time_hex="
              << event.time_hex << " cycle=" << event.cycle << " leaves="
              << event.leaf_count << " max_level=" << event.max_level
              << " checksum=" << event.tree_checksum << std::endl;
  }
}

void AMRHistory::AfterAcceptedTransaction(int created, int deleted) {
  if (record()) {
    AppendEvent(created, deleted);
    return;
  }
  if (!replay_event_pending_) Fatal("accepted replay transaction without pending event");
  if (CurrentLeaves() != replay_target_) Fatal("accepted replay hierarchy mismatch");
  auto event = events_[next_event_];
  AppendLedger("replay", event, true);
  if (global_variable::my_rank == 0) {
    std::cout << "AMR_HISTORY_REPLAY event=" << event.index << " time_hex="
              << event.time_hex << " requested_leaves=" << event.leaf_count
              << " accepted_leaves=" << CurrentLeaves().size() << " max_level="
              << event.max_level << " checksum=" << event.tree_checksum
              << " exact_match=true" << std::endl;
  }
  last_applied_event_ = next_event_;
  ++next_event_;
  replay_event_pending_ = false;
  replay_target_.clear();
}

void AMRHistory::AppendLedger(const std::string &action, const amr_history::Event &event,
                              bool exact_match) const {
  if (global_variable::my_rank != 0) return;
  std::ofstream out(ledger_path_, std::ios::binary | std::ios::app);
  if (!out) Fatal("cannot append AMR history ledger");
  out << "{\"action\":\"" << action << "\",\"event\":" << event.index
      << ",\"time_hex\":\"" << event.time_hex << "\",\"leaves\":"
      << event.leaf_count << ",\"max_level\":" << event.max_level
      << ",\"tree_checksum\":\"" << event.tree_checksum
      << "\",\"ranks\":" << global_variable::nranks
      << ",\"exact_match\":" << (exact_match ? "true" : "false") << "}\n";
  out.flush();
  if (!out) Fatal("failed to flush AMR history ledger");
}

std::string AMRHistory::FileDigest() const {
  const std::string bytes = ReadAll(path_);
  if (bytes.empty()) Fatal("cannot read history for digest");
  return amr_history::Checksum(bytes);
}

std::uint64_t AMRHistory::FileSize() const {
  std::ifstream input(path_, std::ios::binary | std::ios::ate);
  if (!input) Fatal("cannot stat history file");
  const auto position = input.tellg();
  if (position < 0) Fatal("negative history file length");
  return static_cast<std::uint64_t>(position);
}

std::string AMRHistory::CurrentTreeChecksum() const {
  return amr_history::TreeChecksum(CurrentLeaves());
}

void AMRHistory::StoreRestartState(ParameterInput *pin) const {
  if (!active() || !initialized_) return;
  pin->SetInteger(kRestartBlock, "schema", 1);
  pin->SetString(kRestartBlock, "mode", record() ? "record" : "replay");
  pin->SetString(kRestartBlock, "history_digest", FileDigest());
  pin->SetString(kRestartBlock, "history_bytes", std::to_string(FileSize()));
  pin->SetInteger(kRestartBlock, "last_applied_event", static_cast<int>(last_applied_event_));
  pin->SetInteger(kRestartBlock, "next_event", static_cast<int>(next_event_));
  pin->SetBoolean(kRestartBlock, "post_event", true);
  pin->SetString(kRestartBlock, "tree_checksum", CurrentTreeChecksum());
}
