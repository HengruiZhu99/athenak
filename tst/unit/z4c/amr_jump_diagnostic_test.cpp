//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file amr_jump_diagnostic_test.cpp
//! \brief Unit tests for default-off AMR-jump configuration and ledger closure.

#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "z4c/amr_jump_diagnostic.hpp"
#include "z4c/cartoon_meridional_sampler.hpp"

namespace {

int failures = 0;

void Expect(const bool condition, const std::string &message) {
  if (condition) return;
  std::cerr << "FAIL: " << message << '\n';
  ++failures;
}

z4c::AMRJumpDiagnosticContext ValidContext() {
  z4c::AMRJumpDiagnosticContext context;
  context.cartoon = true;
  context.adaptive = true;
  context.multilevel = true;
  context.root_level = 0;
  context.maximum_level = 20;
  context.nranks = 4;
  return context;
}

z4c::AMRJumpDiagnosticConfig Parse(const std::string &body,
                                   const z4c::AMRJumpDiagnosticContext &context) {
  ParameterInput input;
  std::istringstream stream("<z4c>\n" + body);
  input.LoadFromStream(stream);
  return z4c::ReadAMRJumpDiagnosticConfig(&input, context);
}

bool ParseFails(const std::string &body,
                const z4c::AMRJumpDiagnosticContext &context) {
  try {
    (void)Parse(body, context);
  } catch (const std::invalid_argument &) {
    return true;
  }
  return false;
}

void TestConfiguration() {
  const auto context = ValidContext();
  const auto defaults = Parse("", context);
  Expect(!defaults.enabled, "diagnostic must default off");
  Expect(defaults.target_level_before == 2 && defaults.target_level_after == 3,
         "default target must be the 2-to-3 transition");
  Expect(defaults.target_cycle == -1,
         "default target cycle must select the first matching transition");
  Expect(defaults.post_cycles == 8, "default post-event window must be eight cycles");
  Expect(defaults.output_basename == "z4c_amr_jump",
         "default output basename must be stable");
  Expect(defaults.target_transfer.empty(),
         "target-transaction transfer override must default off");
  const auto explicit_none = Parse("amr_jump_target_transfer = none\n", context);
  Expect(explicit_none.target_transfer.empty(),
         "explicit none must preserve the default-off target transfer");
  Expect(defaults.hierarchy_control ==
             z4c::AMRJumpHierarchyControl::dynamic,
         "hierarchy control must default to dynamic");
  auto shallow_context = context;
  shallow_context.maximum_level = 1;
  Expect(!Parse("", shallow_context).enabled,
         "default-off parsing must not reject a mesh shallower than the default target");

  const auto enabled = Parse(
      "amr_jump_diagnostic = true\n"
      "amr_jump_target_level_before = 2\n"
      "amr_jump_target_level_after = 3\n"
      "amr_jump_target_cycle = 1724\n"
      "amr_jump_post_cycles = 8\n"
      "amr_jump_output_basename = event_1724\n"
      "amr_jump_target_transfer = limited_o2\n"
      "amr_jump_hierarchy_control = freeze_after_target\n",
      context);
  Expect(enabled.enabled && enabled.target_cycle == 1724 &&
             enabled.output_basename == "event_1724" &&
             enabled.target_transfer == "limited_o2" &&
             enabled.hierarchy_control ==
                 z4c::AMRJumpHierarchyControl::freeze_after_target,
         "valid enabled configuration must parse exactly");

  const auto buffered = Parse(
      "amr_jump_diagnostic = true\n"
      "amr_jump_target_cycle = 1724\n"
      "amr_jump_hierarchy_control = buffered_freeze_after_target\n",
      context);
  Expect(buffered.hierarchy_control ==
             z4c::AMRJumpHierarchyControl::buffered_freeze_after_target,
         "buffered frozen hierarchy configuration must parse exactly");
  Expect(ParseFails("amr_jump_diagnostic = true\n"
                    "amr_jump_hierarchy_control = freeze_after_target\n",
                    context),
         "freeze control without an exact target cycle must fail");
  Expect(ParseFails("amr_jump_diagnostic = true\n"
                    "amr_jump_target_cycle = 1724\n"
                    "amr_jump_hierarchy_control = approximate\n", context),
         "unknown hierarchy control must fail");
  Expect(ParseFails("amr_jump_diagnostic = false\n"
                    "amr_jump_target_cycle = 1724\n"
                    "amr_jump_hierarchy_control = freeze_after_target\n",
                    context),
         "hierarchy control must not silently disappear with diagnostics disabled");
  Expect(ParseFails("amr_jump_diagnostic = false\n"
                    "amr_jump_target_cycle = 1724\n"
                    "amr_jump_target_transfer = limited_o2\n", context),
         "target transfer must not silently disappear with diagnostics disabled");
  Expect(ParseFails("amr_jump_diagnostic = true\n"
                    "amr_jump_target_transfer = limited_o2\n", context),
         "target transfer without an exact target cycle must fail");
  Expect(ParseFails("amr_jump_diagnostic = true\n"
                    "amr_jump_target_cycle = 1724\n"
                    "amr_jump_target_transfer = approximate\n", context),
         "unknown target transfer must fail");

  Expect(ParseFails("amr_jump_diagnostic = true\n"
                    "amr_jump_unknown = 1\n", context),
         "unknown diagnostic-prefixed key must fail");
  Expect(ParseFails("amr_jump_diagnostic = true\n"
                    "amr_jump_target_level_before = 3\n"
                    "amr_jump_target_level_after = 3\n", context),
         "non-increasing target levels must fail");
  Expect(ParseFails("amr_jump_diagnostic = true\n"
                    "amr_jump_target_level_before = 2\n"
                    "amr_jump_target_level_after = 4\n", context),
         "multi-level target jump must fail");
  Expect(ParseFails("amr_jump_diagnostic = true\n"
                    "amr_jump_target_level_before = -1\n", context),
         "target below root must fail");
  Expect(ParseFails("amr_jump_diagnostic = true\n"
                    "amr_jump_target_level_after = 21\n"
                    "amr_jump_target_level_before = 20\n", context),
         "target above maximum level must fail");
  const auto zero_pde = Parse("amr_jump_diagnostic = true\n"
                              "amr_jump_post_cycles = 0\n", context);
  Expect(zero_pde.post_cycles == 0,
         "zero post-event cycles must select the stop-after-T5 probe");
  Expect(ParseFails("amr_jump_diagnostic = true\n"
                    "amr_jump_post_cycles = -1\n", context),
         "negative post-event window must fail");
  Expect(ParseFails("amr_jump_diagnostic = true\n"
                    "amr_jump_target_cycle = -2\n", context),
         "invalid negative target cycle must fail");
  Expect(ParseFails("amr_jump_diagnostic = true\n"
                    "amr_jump_output_basename = ../escape\n", context),
         "path-valued output basename must fail");

  auto cartesian = context;
  cartesian.cartoon = false;
  Expect(ParseFails("amr_jump_diagnostic = true\n", cartesian),
         "non-Cartoon use must fail");
  auto fixed = context;
  fixed.adaptive = false;
  Expect(ParseFails("amr_jump_diagnostic = true\n", fixed),
         "non-adaptive use must fail");
  auto single_level = context;
  single_level.multilevel = false;
  Expect(ParseFails("amr_jump_diagnostic = true\n", single_level),
         "single-level use must fail");
}

void TestMeasure() {
  const Real rho = 1.25;
  const Real dx1 = 0.125;
  const Real dx2 = 0.25;
  const Real detg = 1.44;
  const Real first = z4c::Z4cDiagnosticCellMeasure(
      z4c::Z4cSymmetryMode::cartoon_so2, rho, dx1, dx2, 0.5, detg);
  const Real second = z4c::Z4cDiagnosticCellMeasure(
      z4c::Z4cSymmetryMode::cartoon_so2, rho, dx1, dx2, 1.0e-9, detg);
  Expect(first == second, "Cartoon measure must be exactly independent of dx3");

  const Real parent_rho = 1.5 * dx1;
  const Real parent = z4c::kCartoonTwoPi * parent_rho * dx1 * dx2;
  Real children = 0.0;
  for (int j = 0; j < 2; ++j) {
    for (int i = 0; i < 2; ++i) {
      const Real child_rho = parent_rho - 0.25 * dx1 + 0.5 * i * dx1;
      children += z4c::kCartoonTwoPi * child_rho * (0.5 * dx1) *
                  (0.5 * dx2);
    }
  }
  Expect(parent == children,
         "flat ring coordinate volume must partition exactly into four children");
}

void TestLedger() {
  using z4c::AMRJumpLedgerIncrement;
  using z4c::AMRJumpPhase;
  using z4c::AMRJumpWriter;
  std::vector<AMRJumpLedgerIncrement> increments;
  Real value = 1.0;
  const std::vector<std::pair<AMRJumpPhase, AMRJumpWriter>> stages = {
      {AMRJumpPhase::t1_balanced_topology_proposal,
       AMRJumpWriter::topology_only},
      {AMRJumpPhase::t2_redistributed_refined_active,
       AMRJumpWriter::refine_or_derefine_transfer},
      {AMRJumpPhase::t3_boundary_reconstruction, AMRJumpWriter::restrict},
      {AMRJumpPhase::t3_boundary_reconstruction, AMRJumpWriter::mpi_receive},
      {AMRJumpPhase::t3_boundary_reconstruction,
       AMRJumpWriter::physical_or_axis_bc},
      {AMRJumpPhase::t3_boundary_reconstruction,
       AMRJumpWriter::same_level_coarse_refresh},
      {AMRJumpPhase::t3_boundary_reconstruction,
       AMRJumpWriter::coarse_to_fine_prolongation},
      {AMRJumpPhase::t4_projected_z4c,
       AMRJumpWriter::algebraic_projection},
      {AMRJumpPhase::t5_accepted_new_hierarchy,
       AMRJumpWriter::adm_or_constraint_recomputation}};
  for (std::size_t index = 0; index < stages.size(); ++index) {
    const Real next = value + 0.125 * static_cast<Real>(index + 1);
    increments.push_back({stages[index].first, stages[index].second,
                          static_cast<int>(index), 0, 17,
                          value, next});
    value = next;
  }
  const auto closed = z4c::CloseAMRJumpLedger(1.0, value, increments);
  Expect(closed.closed(), "complete ordered ledger must telescope");

  auto wrong_total = z4c::CloseAMRJumpLedger(1.0, value + 1.0, increments);
  Expect(!wrong_total.closed(), "accounting mismatch must fail closure");

  auto unordered = increments;
  std::swap(unordered.front(), unordered.back());
  Expect(!z4c::CloseAMRJumpLedger(1.0, value, unordered).closed(),
         "out-of-order phase inventory must fail closure");

  auto unordered_writer = increments;
  std::swap(unordered_writer[3], unordered_writer[4]);
  Expect(!z4c::CloseAMRJumpLedger(1.0, value, unordered_writer).closed(),
         "out-of-order T3 writer inventory must fail closure");

  auto duplicate_owner = increments;
  duplicate_owner.push_back(increments.back());
  duplicate_owner.back().owner_rank = 1;
  duplicate_owner.back().before = duplicate_owner.back().after;
  Expect(!z4c::CloseAMRJumpLedger(1.0, value, duplicate_owner).closed(),
         "duplicate ownership across ranks must fail closure");

  auto nonfinite = increments;
  nonfinite.back().after = std::numeric_limits<Real>::quiet_NaN();
  Expect(!z4c::CloseAMRJumpLedger(1.0, value, nonfinite).closed(),
         "nonfinite ledger value must fail closure");
}

}  // namespace

int main() {
  TestConfiguration();
  TestMeasure();
  TestLedger();
  if (failures != 0) {
    std::cerr << failures << " AMR jump diagnostic tests failed\n";
    return 1;
  }
  std::cout << "AMR jump diagnostic configuration, measure, and ledger tests passed\n";
  return 0;
}
