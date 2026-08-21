#include "mesh/amr_history.hpp"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
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
#include "mesh/meshblock_pack.hpp"
#include "mesh/meshblock.hpp"
#include "z4c/z4c.hpp"
#include "z4c/z4c_amr.hpp"
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
  const bool has_compatible_source_parameter =
      pin->DoesParameterExist("mesh_refinement", "amr_history_compatible_source_id");
  const char *compatible_source_environment =
      std::getenv("ATHENA_AMR_HISTORY_COMPATIBLE_SOURCE_ID");
  const bool has_compatible_source_environment =
      compatible_source_environment != nullptr;
  if (has_compatible_source_parameter || has_compatible_source_environment) {
    if (!replay()) Fatal("amr_history_compatible_source_id is replay-only");
    const std::string parameter_source_id = has_compatible_source_parameter
        ? pin->GetString("mesh_refinement", "amr_history_compatible_source_id") : "";
    const std::string environment_source_id = has_compatible_source_environment
        ? std::string(compatible_source_environment) : "";
    if (has_compatible_source_parameter && has_compatible_source_environment &&
        parameter_source_id != environment_source_id) {
      Fatal("amr_history_compatible_source_id parameter/environment mismatch");
    }
    compatible_source_id_ = has_compatible_source_parameter
        ? parameter_source_id : environment_source_id;
    if (compatible_source_id_.empty()) {
      Fatal("amr_history_compatible_source_id must not be empty");
    }
  }
  const bool has_extension_parameter =
      pin->DoesParameterExist("mesh_refinement", "amr_history_extension_file");
  const char *extension_environment =
      std::getenv("ATHENA_AMR_HISTORY_EXTENSION_FILE");
  const bool has_extension_environment = extension_environment != nullptr;
  if (has_extension_parameter || has_extension_environment) {
    if (!replay()) Fatal("amr_history_extension_file is replay-only");
    const std::string parameter_path = has_extension_parameter
        ? pin->GetString("mesh_refinement", "amr_history_extension_file") : "";
    const std::string environment_path = has_extension_environment
        ? std::string(extension_environment) : "";
    if (has_extension_parameter && has_extension_environment &&
        parameter_path != environment_path) {
      Fatal("amr_history_extension_file parameter/environment mismatch");
    }
    extension_path_ = has_extension_parameter ? parameter_path : environment_path;
    if (extension_path_.empty()) Fatal("amr_history_extension_file must not be empty");
  }
  if (const char *branch_base =
          std::getenv("ATHENA_AMR_HISTORY_BRANCH_BASE_EVENT")) {
    if (extension_path_.empty()) {
      Fatal("AMR history branch base requires an extension file");
    }
    char *end = nullptr;
    errno = 0;
    const long value = std::strtol(branch_base, &end, 10);
    if (errno != 0 || end == branch_base || *end != '\0' ||
        value < 0 || value > INT_MAX) {
      Fatal("ATHENA_AMR_HISTORY_BRANCH_BASE_EVENT is invalid");
    }
    extension_branch_base_event_ = static_cast<int>(value);
  }
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
  auto candidate = CurrentHeader();
  if (!compatible_source_id_.empty()) {
    if (header_.source_id != compatible_source_id_) {
      Fatal("history source-id does not match explicit compatible source-id");
    }
    std::cout << "AMR_HISTORY_SOURCE_COMPATIBILITY"
              << " recorded_source_id=" << header_.source_id
              << " current_source_id=" << candidate.source_id
              << " explicit_match=true" << std::endl;
    candidate.source_id = header_.source_id;
  }
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

void AMRHistory::LoadAppendOnlyExtension() {
  std::ifstream input(extension_path_);
  if (!input) Fatal("cannot open append-only history extension '" + extension_path_ + "'");
  std::string line, error;
  amr_history::Header extension_header;
  if (!std::getline(input, line) ||
      !amr_history::DecodeHeader(line, &extension_header, &error)) {
    Fatal("invalid history extension header: " + error);
  }
  if (amr_history::EncodeHeader(extension_header) !=
      amr_history::EncodeHeader(header_)) {
    Fatal("history extension header differs from authenticated authority header");
  }
  std::vector<amr_history::Event> extension;
  while (std::getline(input, line)) {
    if (line.empty()) Fatal("blank or truncated history extension record");
    amr_history::Event event;
    if (!amr_history::DecodeEvent(line, &event, &error)) {
      Fatal("invalid history extension event: " + error);
    }
    extension.push_back(std::move(event));
  }
  if (!input.eof()) Fatal("history extension read failed before EOF");
  if (!amr_history::ValidateEvents(header_, extension, &error)) {
    Fatal(error);
  }
  if (extension_branch_base_event_ >= 0) {
    const auto base = static_cast<std::size_t>(extension_branch_base_event_);
    if (base < last_applied_event_ || base + 1 < next_event_) {
      Fatal("AMR history branch would alter an already applied replay event");
    }
    if (!amr_history::AuthenticatedBranch(events_, extension, base, &error)) {
      Fatal(error);
    }
  } else if (!amr_history::AppendOnlyExtension(events_, extension, &error)) {
    Fatal(error);
  }
  events_ = std::move(extension);
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
    if (!extension_path_.empty()) {
      if (!restart || !HasRestartCarrier()) {
        Fatal("append-only history extension requires an authenticated replay restart");
      }
      LoadAppendOnlyExtension();
    }
  }
  initialized_ = true;
}

void AMRHistory::LimitTimestep() {
  if (!replay() || !initialized_ || next_event_ >= events_.size()) return;
  double next_time = 0.0;
  std::string error;
  const double candidate_dt = mesh_->dt;
  if (!amr_history::ParseReal(events_[next_event_].time_hex, &next_time) ||
      !amr_history::LimitTimestep(mesh_->time, next_time, &mesh_->dt, &error)) Fatal(error);
  last_candidate_dt_ = candidate_dt;
  last_applied_dt_ = mesh_->dt;
  last_timestep_clipped_ = mesh_->dt != candidate_dt;
  if (mesh_->dt != candidate_dt && global_variable::my_rank == 0) {
    std::cout << "AMR_HISTORY_TIMESTEP_CLIP event=" << next_event_
              << " time_hex=" << amr_history::HexReal(mesh_->time)
              << " candidate_dt_hex=" << amr_history::HexReal(candidate_dt)
              << " applied_dt_hex=" << amr_history::HexReal(mesh_->dt)
              << " target_time_hex=" << events_[next_event_].time_hex << std::endl;
  }
}

void AMRHistory::CaptureShadowFlags() {
  if (!replay()) Fatal("shadow refinement flags are replay-only");
  shadow_refine_ = 0;
  shadow_derefine_ = 0;
  for (int gid = 0; gid < mesh_->nmb_total; ++gid) {
    shadow_refine_ += mesh_->pmr->refine_flag.h_view(gid) > 0;
    shadow_derefine_ += mesh_->pmr->refine_flag.h_view(gid) < 0;
  }
  shadow_flags_captured_ = true;
  AppendShadowLedger();
  if (mesh_->pmb_pack != nullptr && mesh_->pmb_pack->pz4c != nullptr &&
      mesh_->pmb_pack->pz4c->chi_parent_provenance != nullptr &&
      next_event_ < events_.size()) {
    mesh_->pmb_pack->pz4c->chi_parent_provenance->RecordShadowAMRRequests(
        next_event_, events_[next_event_].time_hex, CurrentTreeChecksum());
  }
}

void AMRHistory::AppendShadowLedger() const {
  if (!replay() || mesh_->pmb_pack == nullptr || mesh_->pmb_pack->pz4c == nullptr) {
    return;
  }
  if (!mesh_->pmb_pack->pz4c->pamr->capture_replay_dchi) return;
  const auto &dchi = mesh_->pmb_pack->pz4c->pamr->last_dchi_max;
  const auto &ordinal = mesh_->pmb_pack->pz4c->pamr->last_dchi_ordinal;
  const int nmb = mesh_->pmb_pack->nmb_thispack;
  if (dchi.size() != static_cast<std::size_t>(nmb) ||
      ordinal.size() != static_cast<std::size_t>(nmb)) {
    Fatal("replay dchi shadow values do not match the local MeshBlock count");
  }
  const int begin = mesh_->gids_eachrank[global_variable::my_rank];
  std::vector<int> authority_flags(static_cast<std::size_t>(mesh_->nmb_total), 0);
  bool authority_event = false;
  std::string authority_time_hex;
  if (next_event_ < events_.size()) {
    authority_time_hex = events_[next_event_].time_hex;
    double event_time = 0.0;
    if (!amr_history::ParseReal(authority_time_hex, &event_time)) {
      Fatal("cannot decode next authority time for shadow ledger");
    }
    authority_event = amr_history::TimeEqual(mesh_->time, event_time);
    if (authority_event) {
      amr_history::Transition transition;
      std::string error;
      if (!amr_history::DeriveTransition(header_, CurrentLeaves(),
                                         events_[next_event_].leaves,
                                         &transition, &error)) {
        Fatal("cannot derive authority transition for shadow ledger: " + error);
      }
      const auto current = CurrentLeaves();
      if (transition.flags.size() != current.size()) {
        Fatal("authority transition flag count differs from current tree");
      }
      for (int gid = 0; gid < mesh_->nmb_total; ++gid) {
        const auto loc = Convert(mesh_->lloc_eachmb[gid]);
        const auto found = std::lower_bound(current.begin(), current.end(), loc);
        if (found == current.end() || !(*found == loc)) {
          Fatal("GID leaf missing while writing shadow authority action");
        }
        authority_flags[gid] = transition.flags[
            static_cast<std::size_t>(found - current.begin())];
      }
    }
  }
  const auto action_name = [](const int flag) {
    return flag > 0 ? "refine" : (flag < 0 ? "derefine" : "same");
  };
  const auto classification = [](const int native, const int authority) {
    if (native == authority) return "AGREES";
    if (native < 0) return "WOULD_DEREFINE";
    if (native > 0 && authority == 0) return "WOULD_REFINE_EARLIER";
    if (authority > 0 && native == 0) return "WOULD_NOT_REFINE";
    return "OTHER";
  };
  std::ostringstream name;
  name << pin_->GetString("job", "basename") << ".amr_native_shadow.rank"
       << std::setw(4) << std::setfill('0') << global_variable::my_rank
       << ".jsonl";
  std::ofstream output(name.str(), std::ios::app);
  if (!output) Fatal("cannot append native AMR shadow ledger");
  const double tau_c = mesh_->pmb_pack->z4c_restart_state.central.proper_time;
  mesh_->pmb_pack->pmb->mb_size.sync_host();
  int strongest = -1;
  for (int m = 0; m < nmb; ++m) {
    if (strongest < 0 || dchi[m] > dchi[strongest]) strongest = m;
  }
  for (int m = 0; m < nmb; ++m) {
    const int gid = begin + m;
    const auto &loc = mesh_->lloc_eachmb[gid];
    const auto &size = mesh_->pmb_pack->pmb->mb_size.h_view(m);
    const int native = mesh_->pmr->refine_flag.h_view(gid);
    const int authority = authority_flags[gid];
    if (m != strongest && native == 0 && authority == 0) continue;
    if (ordinal[m] < 0 || ordinal[m] >=
        mesh_->mb_indcs.nx1 * mesh_->mb_indcs.nx2 * mesh_->mb_indcs.nx3) {
      Fatal("native AMR shadow maximum has no valid active-cell ordinal");
    }
    const int nji = mesh_->mb_indcs.nx2 * mesh_->mb_indcs.nx1;
    const int ok = ordinal[m] / nji;
    const int remainder = ordinal[m] % nji;
    const int oj = remainder / mesh_->mb_indcs.nx1;
    const int oi = remainder % mesh_->mb_indcs.nx1;
    const Real strongest_rho =
        size.x1min + (static_cast<Real>(oi) + 0.5) * size.dx1;
    const Real strongest_z =
        size.x2min + (static_cast<Real>(oj) + 0.5) * size.dx2;
    output << std::setprecision(17)
           << "{\"schema\":\"athenak_amr_native_shadow_v1\",\"cycle\":"
           << mesh_->ncycle << ",\"time\":" << mesh_->time
           << ",\"time_hex\":\"" << amr_history::HexReal(mesh_->time)
           << "\",\"tau_c\":" << tau_c << ",\"root_nx1\":"
           << mesh_->mesh_indcs.nx1 << ",\"cells_per_meshblock\":["
           << mesh_->mb_indcs.nx1 << ',' << mesh_->mb_indcs.nx2 << ','
           << mesh_->mb_indcs.nx3 << "],\"gid\":" << gid
           << ",\"logical_location\":[" << loc.level << ',' << loc.lx1 << ','
           << loc.lx2 << ',' << loc.lx3 << "],\"relative_level\":"
           << loc.level - mesh_->root_level << ",\"dx\":" << size.dx1
           << ",\"raw_dchi\":" << dchi[m] << ",\"dchi_over_dx\":"
           << dchi[m] / size.dx1 << ",\"native_action\":\""
           << action_name(native) << "\",\"authority_event\":"
           << (authority_event ? "true" : "false")
           << ",\"authority_event_index\":"
           << (next_event_ < events_.size() ? std::to_string(next_event_) : "null")
           << ",\"authority_event_time_hex\":";
    if (authority_time_hex.empty()) output << "null";
    else output << '\"' << authority_time_hex << '\"';
    output << ",\"authority_action\":\"" << action_name(authority)
           << "\",\"classification\":\""
           << classification(native, authority) << "\",\"record_scope\":\""
           << (m == strongest ? "rank_strongest" : "requested_or_authority")
           << "\",\"strongest_cell_ordinal\":" << ordinal[m]
           << ",\"strongest_cell_offset\":[" << oi << ',' << oj << ',' << ok
           << "],\"strongest_physical_location\":[" << strongest_rho << ','
           << strongest_z << "],\"block_center\":["
           << 0.5 * (size.x1min + size.x1max) << ','
           << 0.5 * (size.x2min + size.x2max) << "]}\n";
  }
  output.flush();
  if (!output) Fatal("failed to flush native AMR shadow ledger");
}

bool AMRHistory::PrepareReplayFlags() {
  if (!replay() || !initialized_ || next_event_ >= events_.size()) return false;
  if (!shadow_flags_captured_) Fatal("replay criterion shadow was not evaluated");
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
  if (mesh_->pmr->refine_flag.extent(0) !=
      static_cast<std::size_t>(mesh_->nmb_total)) {
    Fatal("replay criterion shadow produced the wrong flag extent");
  }
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
              << " exact_match=true shadow_refine=" << shadow_refine_
              << " shadow_derefine=" << shadow_derefine_ << std::endl;
  }
  last_applied_event_ = next_event_;
  ++next_event_;
  replay_event_pending_ = false;
  shadow_flags_captured_ = false;
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
      << ",\"exact_match\":" << (exact_match ? "true" : "false");
  if (action == "replay") {
    double authority_time = 0.0;
    if (!amr_history::ParseReal(event.time_hex, &authority_time)) {
      Fatal("cannot decode authority time for diagnostic replay ledger");
    }
    std::uint64_t authority_bits = 0;
    std::uint64_t actual_bits = 0;
    static_assert(sizeof(authority_bits) == sizeof(authority_time),
                  "ULP diagnostic assumes IEEE binary64");
    std::memcpy(&authority_bits, &authority_time, sizeof(authority_bits));
    std::memcpy(&actual_bits, &mesh_->time, sizeof(actual_bits));
    const long long ulps = actual_bits >= authority_bits
        ? static_cast<long long>(actual_bits - authority_bits)
        : -static_cast<long long>(authority_bits - actual_bits);
    out << std::setprecision(17)
        << ",\"authority_time_hex\":\"" << event.time_hex << "\""
        << ",\"actual_mesh_time_hex\":\""
        << amr_history::HexReal(mesh_->time) << "\""
        << ",\"signed_time_difference\":" << mesh_->time - authority_time
        << ",\"ulp_difference\":" << ulps
        << ",\"candidate_dt_hex\":\""
        << amr_history::HexReal(last_candidate_dt_) << "\""
        << ",\"applied_dt_hex\":\""
        << amr_history::HexReal(last_applied_dt_) << "\""
        << ",\"preceding_timestep_clipped\":"
        << (last_timestep_clipped_ ? "true" : "false");
    if (mesh_->pmb_pack != nullptr && mesh_->pmb_pack->pz4c != nullptr &&
        mesh_->pmb_pack->pz4c->chi_parent_provenance != nullptr) {
      mesh_->pmb_pack->pz4c->chi_parent_provenance->RecordReplayAlignment(
          event.index, event.time_hex, amr_history::HexReal(mesh_->time),
          mesh_->time - authority_time, ulps, last_timestep_clipped_);
    }
  }
  out << "}\n";
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
  if (!extension_path_.empty()) {
    Fatal("restart output is disabled for a diagnostic append-only history extension");
  }
  pin->SetInteger(kRestartBlock, "schema", 1);
  pin->SetString(kRestartBlock, "mode", record() ? "record" : "replay");
  pin->SetString(kRestartBlock, "history_digest", FileDigest());
  pin->SetString(kRestartBlock, "history_bytes", std::to_string(FileSize()));
  pin->SetInteger(kRestartBlock, "last_applied_event", static_cast<int>(last_applied_event_));
  pin->SetInteger(kRestartBlock, "next_event", static_cast<int>(next_event_));
  pin->SetBoolean(kRestartBlock, "post_event", true);
  pin->SetString(kRestartBlock, "tree_checksum", CurrentTreeChecksum());
}
