#ifndef MESH_AMR_HISTORY_HPP_
#define MESH_AMR_HISTORY_HPP_

#include <cstdint>
#include <string>
#include <vector>

#include "mesh/amr_history_format.hpp"

class Mesh;
class ParameterInput;

class AMRHistory {
 public:
  enum class Mode { off, record, replay };

  AMRHistory(Mesh *mesh, ParameterInput *pin);
  ~AMRHistory() = default;

  bool active() const { return mode_ != Mode::off; }
  bool record() const { return mode_ == Mode::record; }
  bool replay() const { return mode_ == Mode::replay; }

  void Initialize(bool restart);
  void LimitTimestep();
  void CaptureShadowFlags();
  bool PrepareReplayFlags();
  void CaptureRequestedFlags();
  void ValidateReplayProposedTree();
  void AfterAcceptedTransaction(int created, int deleted);
  void StoreRestartState(ParameterInput *pin) const;

 private:
  [[noreturn]] void Fatal(const std::string &message) const;
  amr_history::Header CurrentHeader() const;
  std::vector<amr_history::Location> CurrentLeaves() const;
  void LoadHistory();
  void LoadAppendOnlyExtension();
  void WriteFreshHistory();
  void AppendEvent(int created, int deleted);
  void AppendLedger(const std::string &action, const amr_history::Event &event,
                    bool exact_match) const;
  std::string FileDigest() const;
  std::uint64_t FileSize() const;
  std::string CurrentTreeChecksum() const;
  bool HasRestartCarrier() const;
  void LoadRestartCarrier();

  Mesh *mesh_ = nullptr;
  ParameterInput *pin_ = nullptr;
  Mode mode_ = Mode::off;
  std::string path_;
  std::string extension_path_;
  int extension_branch_base_event_ = -1;
  std::string ledger_path_;
  amr_history::Header header_;
  std::vector<amr_history::Event> events_;
  std::size_t next_event_ = 0;
  std::size_t last_applied_event_ = 0;
  bool initialized_ = false;
  bool replay_event_pending_ = false;
  int requested_refine_ = 0;
  int requested_derefine_ = 0;
  int shadow_refine_ = 0;
  int shadow_derefine_ = 0;
  bool shadow_flags_captured_ = false;
  bool last_timestep_clipped_ = false;
  std::vector<amr_history::Location> replay_target_;
  std::string loaded_digest_;
};

#endif  // MESH_AMR_HISTORY_HPP_
