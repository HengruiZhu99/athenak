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
  Require(amr_history::TimeEqual(0.125, 0.125), "event time equality");

  std::cout << "AMR_HISTORY_FORMAT_TEST_PASS" << std::endl;
  return EXIT_SUCCESS;
}
