//========================================================================================
//! \file chi_parent_provenance.hpp
//! \brief Default-off, state-preserving provenance audit for invalid coarse Z4c chi.

#ifndef Z4C_CHI_PARENT_PROVENANCE_HPP_
#define Z4C_CHI_PARENT_PROVENANCE_HPP_

#include <cstddef>
#include <string>

#include "athena.hpp"

class MeshBlockPack;
class MeshBoundaryValuesCC;
class ParameterInput;
class Driver;

namespace z4c {

enum class ChiProvenanceCheckpoint : int {
  s0_after_rk = 0,
  s1_after_restriction = 1,
  s2_after_receive = 2,
  s3_after_boundary = 3,
  s4_before_parent_gate = 4,
};

enum class ChiWriterProvenance : int {
  local_restriction_centered = 0,
  local_restriction_radial_edge = 1,
  local_restriction_z_edge = 2,
  local_restriction_corner = 3,
  same_level_owner_receive = 4,
  coarser_neighbor_receive = 5,
  axis_boundary_fill = 6,
  outer_physical_boundary_fill = 7,
  preexisting_unchanged_cache = 8,
  unknown = 9,
};

const char *ChiWriterProvenanceName(ChiWriterProvenance writer);

struct ChiParentProvenanceConfig {
  bool enabled = false;
  Real start_time = 0.0;
  std::string output_basename = "chi_parent_provenance";
  bool control_target_trace = false;
};

// Exact, production-order chi RHS terms captured by the production kernels.  The
// array is allocated only when the provenance diagnostic is enabled.
enum ChiProvenanceTerm : int {
  chi_adv_rho = 0,
  chi_adv_z = 1,
  chi_adv_y = 2,
  chi_lie_divergence = 3,
  chi_curvature_source = 4,
  chi_rhs_before_ko = 5,
  chi_ko_rho = 6,
  chi_rhs_after_ko_rho = 7,
  chi_ko_z = 8,
  chi_rhs_after_ko_z = 9,
  chi_ko_y = 10,
  chi_rhs_after_ko_y = 11,
  chi_rhs_after_ko = 12,
  chi_adv_total_production = 13,
  n_chi_provenance_terms = 14,
};

struct ChiRKArithmetic {
  Real affine_base = 0.0;
  Real rhs_increment = 0.0;
  Real candidate = 0.0;
};

KOKKOS_INLINE_FUNCTION
ChiRKArithmetic EvaluateChiRKCandidate(const Real gamma0, const Real gamma1,
                                       const Real beta_dt, const Real chi_old,
                                       const Real chi_accumulator,
                                       const Real rhs) {
  ChiRKArithmetic result;
  result.affine_base = gamma0 * chi_old + gamma1 * chi_accumulator;
  result.rhs_increment = beta_dt * rhs;
  result.candidate = gamma0 * chi_old + gamma1 * chi_accumulator + beta_dt * rhs;
  return result;
}

ChiParentProvenanceConfig ReadChiParentProvenanceConfig(ParameterInput *pin);

class ChiParentProvenanceRuntime {
 public:
  struct HostSnapshot;

  ChiParentProvenanceRuntime(MeshBlockPack *pack,
                             const ChiParentProvenanceConfig &config);
  ~ChiParentProvenanceRuntime();

  void RecordBeforeCopy(Driver *driver, int stage);
  void RecordAfterCopy(Driver *driver, int stage);
  void AnalyzePreUpdate(Driver *driver, int stage);
  void RecordCheckpoint(ChiProvenanceCheckpoint checkpoint, int stage,
                        MeshBoundaryValuesCC *boundary);
  void AnalyzeBoundaryFailure(MeshBoundaryValuesCC *boundary,
                              unsigned long long invalid_parent_stencils,
                              unsigned long long first_rejected_key);
  void RecordShadowAMRRequests(std::size_t next_event,
                               const std::string &next_event_time_hex,
                               const std::string &tree_checksum);
  void RecordReplayAlignment(std::size_t event,
                             const std::string &authority_time_hex,
                             const std::string &actual_time_hex,
                             double signed_difference, long long ulp_difference,
                             bool preceding_timestep_clipped);

 private:
  MeshBlockPack *pack_ = nullptr;
  ChiParentProvenanceConfig config_;
  std::string output_root_;
  int cycle_ = -1;
  int stage_ = -1;
  HostSnapshot *fine_s0_ = nullptr;
  HostSnapshot *coarse_s1_ = nullptr;
  HostSnapshot *coarse_s2_ = nullptr;
  HostSnapshot *coarse_s3_ = nullptr;
  HostSnapshot *u0_before_copy_ = nullptr;
  HostSnapshot *u1_before_copy_ = nullptr;
  HostSnapshot *u1_after_copy_ = nullptr;
  int copy_cycle_ = -1;
  int copy_stage_ = -1;
};

}  // namespace z4c

#endif  // Z4C_CHI_PARENT_PROVENANCE_HPP_
