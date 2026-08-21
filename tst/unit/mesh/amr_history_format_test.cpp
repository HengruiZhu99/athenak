#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "mesh/amr_history_format.hpp"

namespace {
void Require(bool condition, const std::string &message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

amr_history::Header Header() {
  amr_history::Header h;
  h.dimension = 2;
  h.symmetry = "cartoon_so2";
  h.coordinate_map = "half_rho_z_suppressed_y_v2";
  h.root_level = 1;
  h.root_blocks = {{2, 2, 1}};
  h.domain_hex = {{"0x0p+0", "0x1p+3", "-0x1p+3", "0x1p+3", "-0x1p-1", "0x1p-1"}};
  h.max_level = 4;
  h.cells_per_meshblock = {{8, 8, 1}};
  h.source_id = "unit";
  return h;
}

std::vector<amr_history::Location> RootLeaves() {
  return {{1, 0, 0, 0}, {1, 1, 0, 0}, {1, 0, 1, 0}, {1, 1, 1, 0}};
}

std::vector<amr_history::Location> RefinedLeaves() {
  return {{2, 0, 0, 0}, {2, 1, 0, 0}, {2, 0, 1, 0}, {2, 1, 1, 0},
          {1, 1, 0, 0}, {1, 0, 1, 0}, {1, 1, 1, 0}};
}
}  // namespace

int main() {
  auto header = Header();
  std::string error;
  Require(amr_history::ValidateHeader(header, &error), error);
  const std::string encoded_header = amr_history::EncodeHeader(header);
  amr_history::Header decoded_header;
  Require(amr_history::DecodeHeader(encoded_header, &decoded_header, &error), error);
  Require(amr_history::EncodeHeader(decoded_header) == encoded_header, "header round trip");
  Require(decoded_header.schema == 2 &&
              decoded_header.grid_centering == "cell" &&
              decoded_header.centering_schema == 1,
          "schema-2 centering provenance round trip");
  auto legacy_header = header;
  legacy_header.schema = 1;
  legacy_header.grid_centering = "cell";
  legacy_header.centering_schema = 0;
  const std::string encoded_legacy = amr_history::EncodeHeader(legacy_header);
  amr_history::Header decoded_legacy;
  Require(amr_history::DecodeHeader(encoded_legacy, &decoded_legacy, &error), error);
  Require(amr_history::EncodeHeader(decoded_legacy) == encoded_legacy,
          "legacy schema-1 header round trip");
  Require(decoded_legacy.grid_centering == "cell" &&
              decoded_legacy.centering_schema == 0,
          "legacy schema-1 centering is explicit after decode");

  amr_history::Event e0;
  e0.time_decimal = "0";
  e0.time_hex = "0x0p+0";
  e0.leaves = RootLeaves();
  const std::string encoded_event = amr_history::EncodeEvent(e0);
  amr_history::Event decoded_event;
  Require(amr_history::DecodeEvent(encoded_event, &decoded_event, &error), error);
  Require(decoded_event.leaves == RootLeaves(), "event leaves round trip");

  amr_history::Transition transition;
  Require(amr_history::DeriveTransition(header, RootLeaves(), RefinedLeaves(),
                                        &transition, &error), error);
  Require(transition.refine_parents == 1 && transition.derefine_leaves == 0,
          "one parent refinement");
  Require(amr_history::DeriveTransition(header, RefinedLeaves(), RootLeaves(),
                                        &transition, &error), error);
  Require(transition.refine_parents == 0 && transition.derefine_leaves == 4,
          "four sibling derefinement");

  auto high = header;
  high.cells_per_meshblock = {{16, 16, 1}};
  Require(amr_history::Compatible(header, high, &error), "cells per block may differ");
  auto vertex = header;
  vertex.grid_centering = "vertex";
  Require(!amr_history::Compatible(header, vertex, &error),
          "grid centering must match by default");
  high.root_blocks[0] = 4;
  Require(!amr_history::Compatible(header, high, &error), "root blocks must match");

  std::vector<amr_history::Event> events{decoded_event};
  amr_history::Event e1;
  e1.index = 1;
  e1.time_decimal = "0.125";
  e1.time_hex = "0x1p-3";
  e1.leaves = RefinedLeaves();
  const auto e1_line = amr_history::EncodeEvent(e1);
  Require(amr_history::DecodeEvent(e1_line, &e1, &error), error);
  events.push_back(e1);
  Require(amr_history::ValidateEvents(header, events, &error), error);

  auto extension = events;
  auto e2 = e1;
  e2.index = 2;
  e2.time_decimal = "0.25";
  e2.time_hex = "0x1p-2";
  e2.leaves = RootLeaves();
  const auto e2_line = amr_history::EncodeEvent(e2);
  Require(amr_history::DecodeEvent(e2_line, &e2, &error), error);
  extension.push_back(e2);
  Require(amr_history::ValidateEvents(header, extension, &error), error);
  Require(amr_history::AppendOnlyExtension(events, extension, &error), error);
  auto changed_prefix = extension;
  changed_prefix[1] = changed_prefix[2];
  Require(!amr_history::AppendOnlyExtension(events, changed_prefix, &error),
          "changed extension prefix rejected");
  Require(!amr_history::AppendOnlyExtension(events, events, &error),
          "non-extending history rejected");

  auto authority = extension;
  auto authority_e3 = e2;
  authority_e3.index = 3;
  authority_e3.time_decimal = "0.375";
  authority_e3.time_hex = "0x1.8p-2";
  authority_e3.leaves = RefinedLeaves();
  const auto authority_e3_line = amr_history::EncodeEvent(authority_e3);
  Require(amr_history::DecodeEvent(authority_e3_line, &authority_e3, &error), error);
  authority.push_back(authority_e3);
  auto branch = extension;
  auto branch_e3 = authority_e3;
  branch_e3.time_decimal = "0.3125";
  branch_e3.time_hex = "0x1.4p-2";
  const auto branch_e3_line = amr_history::EncodeEvent(branch_e3);
  Require(amr_history::DecodeEvent(branch_e3_line, &branch_e3, &error), error);
  branch.push_back(branch_e3);
  Require(amr_history::AuthenticatedBranch(authority, branch, 2, &error), error);
  auto bad_branch = branch;
  bad_branch[2] = bad_branch[1];
  Require(!amr_history::AuthenticatedBranch(authority, bad_branch, 2, &error),
          "changed branch prefix rejected");
  Require(!amr_history::AuthenticatedBranch(authority, extension, 2, &error),
          "branch without a post-base event rejected");
  Require(!amr_history::AuthenticatedBranch(authority, authority, 2, &error),
          "branch without divergence rejected");

  auto nonmonotonic = events;
  nonmonotonic[1].time_decimal = "0";
  nonmonotonic[1].time_hex = "0x0p+0";
  Require(!amr_history::ValidateEvents(header, nonmonotonic, &error),
          "nonmonotonic event time rejected");
  auto repeated = events;
  repeated[1].leaves = repeated[0].leaves;
  Require(!amr_history::ValidateEvents(header, repeated, &error),
          "repeated accepted tree rejected");
  auto wrong_index = events;
  wrong_index[1].index = 3;
  Require(!amr_history::ValidateEvents(header, wrong_index, &error),
          "noncontiguous event index rejected");

  auto malformed = encoded_event;
  malformed[malformed.size() - 3] = malformed[malformed.size() - 3] == '0' ? '1' : '0';
  Require(!amr_history::DecodeEvent(malformed, &decoded_event, &error), "bad hash rejected");
  auto duplicate = RootLeaves(); duplicate.push_back(duplicate.front());
  Require(!amr_history::ValidateTree(header, duplicate, &error), "duplicate rejected");
  auto overlap = RootLeaves(); overlap.push_back({2, 0, 0, 0});
  Require(!amr_history::ValidateTree(header, overlap, &error), "overlap rejected");
  auto incomplete = RootLeaves(); incomplete.pop_back();
  Require(!amr_history::ValidateTree(header, incomplete, &error), "incomplete rejected");

  auto impossible = RefinedLeaves();
  impossible[0].level = 3;
  impossible[0].lx1 = 0;
  impossible[0].lx2 = 0;
  Require(!amr_history::ValidateTree(header, impossible, &error),
          "impossible incomplete multigeneration tree rejected");

  double dt = 0.2;
  Require(amr_history::LimitTimestep(0.0, 0.125, &dt, &error) && dt == 0.125,
          "timestep clips to event");
  const double recorded_time = std::strtod("0x1.3333333333333p-5", nullptr);
  const double production_dt = std::strtod("0x1.3333333333333p-6", nullptr);
  const double recorded_next = std::strtod("0x1.cccccccccccccp-5", nullptr);
  Require(recorded_time + production_dt == recorded_next,
          "fixture production timestep lands on recorded event");
  Require(production_dt > recorded_next - recorded_time,
          "fixture exposes subtractive one-ulp discrepancy");
  dt = production_dt;
  Require(amr_history::LimitTimestep(recorded_time, recorded_next, &dt, &error) &&
              dt == production_dt,
          "exact rounded endpoint preserves production timestep bitwise");
  const double n256_time = std::strtod("0x1.1c66666666643p-1", nullptr);
  const double n256_next = std::strtod("0x1.1c66666666666p-1", nullptr);
  const double n256_dt = std::strtod("0x1.3333333333333p-9", nullptr);
  Require(n256_next > n256_time && n256_next - n256_time < n256_dt,
          "fixture exposes accumulated cross-resolution roundoff remainder");
  Require(amr_history::TimeEqual(n256_time, n256_next),
          "cross-resolution event time is equal within bounded roundoff");
  dt = n256_dt;
  Require(amr_history::LimitTimestep(n256_time, n256_next, &dt, &error) &&
              dt == n256_dt,
          "cross-resolution event does not create a near-zero PDE step");
  const double aurora_high_time =
      std::strtod("0x1.6a09e658c4fdep-8", nullptr);
  const double aurora_high_dt =
      std::strtod("0x1.21a18513d404p-10", nullptr);
  const double aurora_event =
      std::strtod("0x1.b272479dba2b8p-8", nullptr);
  Require(aurora_high_time + aurora_high_dt < aurora_event &&
              amr_history::TimeEqual(aurora_high_time + aurora_high_dt,
                                     aurora_event),
          "fixture exposes the Aurora 2x replay roundoff undershoot");
  dt = aurora_high_dt;
  Require(amr_history::LimitTimestep(aurora_high_time, aurora_event, &dt,
                                     &error) &&
              dt == aurora_event - aurora_high_time &&
              aurora_high_time + dt == aurora_event,
          "near-event 2x timestep is adjusted to the exact authority time");
  Require(amr_history::TimeEqual(0.125, 0.125), "event time equality");

  std::cout << "AMR_HISTORY_FORMAT_TEST_PASS" << std::endl;
  return EXIT_SUCCESS;
}
