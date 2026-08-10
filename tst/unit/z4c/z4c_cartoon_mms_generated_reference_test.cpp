//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file z4c_cartoon_mms_generated_reference_test.cpp
//! \brief Compare compiled generated C++ against checked-in 90-digit references.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <regex>
#include <set>
#include <string>

#include "pgen/unit_tests/z4c_cartoon_derivatives_oracle.hpp"

namespace {

bool Lookup(const std::string &name, const z4c_mms::AnalyticOracle &oracle,
            Real *value) {
  if (name == "scalar") {
    *value = oracle.scalar;
    return true;
  }
  int a = 0, b = 0, c = 0, consumed = 0;
  if (std::sscanf(name.c_str(), "scalar_first[%d]%n", &a, &consumed) == 1 &&
      consumed == static_cast<int>(name.size()) && a >= 0 && a < 3) {
    *value = oracle.scalar_first[a]; return true;
  }
  if (std::sscanf(name.c_str(), "scalar_second[%d][%d]%n", &a, &b, &consumed) == 2 &&
      consumed == static_cast<int>(name.size()) && a >= 0 && a < 3 && b >= 0 && b < 3) {
    *value = oracle.scalar_second[a][b]; return true;
  }
  if (std::sscanf(name.c_str(), "vector[%d]%n", &a, &consumed) == 1 &&
      consumed == static_cast<int>(name.size()) && a >= 0 && a < 3) {
    *value = oracle.vector[a]; return true;
  }
  if (std::sscanf(name.c_str(), "vector_first[%d][%d]%n", &a, &b, &consumed) == 2 &&
      consumed == static_cast<int>(name.size()) && a >= 0 && a < 3 && b >= 0 && b < 3) {
    *value = oracle.vector_first[a][b]; return true;
  }
  if (std::sscanf(name.c_str(), "vector_second[%d][%d][%d]%n", &a, &b, &c,
                  &consumed) == 3 && consumed == static_cast<int>(name.size()) &&
      a >= 0 && a < 3 && b >= 0 && b < 3 && c >= 0 && c < 3) {
    *value = oracle.vector_second[a][b][c]; return true;
  }
  if (std::sscanf(name.c_str(), "tensor[%d]%n", &a, &consumed) == 1 &&
      consumed == static_cast<int>(name.size()) && a >= 0 && a < 6) {
    *value = oracle.tensor[a]; return true;
  }
  if (std::sscanf(name.c_str(), "tensor_first[%d][%d]%n", &a, &b, &consumed) == 2 &&
      consumed == static_cast<int>(name.size()) && a >= 0 && a < 6 && b >= 0 && b < 3) {
    *value = oracle.tensor_first[a][b]; return true;
  }
  if (std::sscanf(name.c_str(), "tensor_second[%d][%d][%d]%n", &a, &b, &c,
                  &consumed) == 3 && consumed == static_cast<int>(name.size()) &&
      a >= 0 && a < 6 && b >= 0 && b < 3 && c >= 0 && c < 3) {
    *value = oracle.tensor_second[a][b][c]; return true;
  }
  return false;
}

struct SplitOracle {
  z4c_mms::FieldValues fields;
  z4c_mms::ScalarOracle scalar;
  z4c_mms::VectorOracle vector[3];
  z4c_mms::VectorOracle vector_generic[3];
  z4c_mms::TensorOracle tensor[6];
  z4c_mms::TensorOracle tensor_generic[6];
};

void EvaluateSplitOracle(const Real x, const Real y, const Real z,
                         SplitOracle *oracle) {
  z4c_mms::EvaluateFieldValues(x, y, z, oracle->fields);
  z4c_mms::EvaluateScalarOracle(x, y, z, oracle->scalar);
  z4c_mms::EvaluateVectorOracle0(x, y, z, oracle->vector[0]);
  z4c_mms::EvaluateVectorOracle1(x, y, z, oracle->vector[1]);
  z4c_mms::EvaluateVectorOracle2(x, y, z, oracle->vector[2]);
  z4c_mms::EvaluateTensorOracle0(x, y, z, oracle->tensor[0]);
  z4c_mms::EvaluateTensorOracle1(x, y, z, oracle->tensor[1]);
  z4c_mms::EvaluateTensorOracle2(x, y, z, oracle->tensor[2]);
  z4c_mms::EvaluateTensorOracle3(x, y, z, oracle->tensor[3]);
  z4c_mms::EvaluateTensorOracle4(x, y, z, oracle->tensor[4]);
  z4c_mms::EvaluateTensorOracle5(x, y, z, oracle->tensor[5]);
  for (int component = 0; component < 3; ++component) {
    z4c_mms::EvaluateVectorOracle(component, x, y, z,
                                  oracle->vector_generic[component]);
  }
  for (int component = 0; component < 6; ++component) {
    z4c_mms::EvaluateTensorOracle(component, x, y, z,
                                  oracle->tensor_generic[component]);
  }
}

bool LookupSplit(const std::string &name, const SplitOracle &oracle,
                 const bool generic, Real *value) {
  if (name == "scalar") {
    *value = oracle.fields.scalar;
    return true;
  }
  int a = 0, b = 0, c = 0, consumed = 0;
  const auto *vector = generic ? oracle.vector_generic : oracle.vector;
  const auto *tensor = generic ? oracle.tensor_generic : oracle.tensor;
  if (std::sscanf(name.c_str(), "scalar_first[%d]%n", &a, &consumed) == 1 &&
      consumed == static_cast<int>(name.size()) && a >= 0 && a < 3) {
    *value = oracle.scalar.first[a]; return true;
  }
  if (std::sscanf(name.c_str(), "scalar_second[%d][%d]%n", &a, &b, &consumed) == 2 &&
      consumed == static_cast<int>(name.size()) && a >= 0 && a < 3 && b >= 0 && b < 3) {
    *value = oracle.scalar.second[a][b]; return true;
  }
  if (std::sscanf(name.c_str(), "vector[%d]%n", &a, &consumed) == 1 &&
      consumed == static_cast<int>(name.size()) && a >= 0 && a < 3) {
    *value = oracle.fields.vector[a]; return true;
  }
  if (std::sscanf(name.c_str(), "vector_first[%d][%d]%n", &a, &b, &consumed) == 2 &&
      consumed == static_cast<int>(name.size()) && a >= 0 && a < 3 && b >= 0 && b < 3) {
    *value = vector[a].first[b]; return true;
  }
  if (std::sscanf(name.c_str(), "vector_second[%d][%d][%d]%n", &a, &b, &c,
                  &consumed) == 3 && consumed == static_cast<int>(name.size()) &&
      a >= 0 && a < 3 && b >= 0 && b < 3 && c >= 0 && c < 3) {
    *value = vector[a].second[b][c]; return true;
  }
  if (std::sscanf(name.c_str(), "tensor[%d]%n", &a, &consumed) == 1 &&
      consumed == static_cast<int>(name.size()) && a >= 0 && a < 6) {
    *value = oracle.fields.tensor[a]; return true;
  }
  if (std::sscanf(name.c_str(), "tensor_first[%d][%d]%n", &a, &b, &consumed) == 2 &&
      consumed == static_cast<int>(name.size()) && a >= 0 && a < 6 && b >= 0 && b < 3) {
    *value = tensor[a].first[b]; return true;
  }
  if (std::sscanf(name.c_str(), "tensor_second[%d][%d][%d]%n", &a, &b, &c,
                  &consumed) == 3 && consumed == static_cast<int>(name.size()) &&
      a >= 0 && a < 6 && b >= 0 && b < 3 && c >= 0 && c < 3) {
    *value = tensor[a].second[b][c]; return true;
  }
  return false;
}

bool ParseStrictLongDouble(const std::string &token, long double *value) {
  try {
    std::size_t consumed = 0;
    *value = std::stold(token, &consumed);
    return consumed == token.size() && std::isfinite(*value);
  } catch (...) {
    return false;
  }
}

bool ParseCoordinate(const std::string &token, Real *value) {
  const auto slash = token.find('/');
  if (slash == std::string::npos) {
    long double parsed = 0.0L;
    if (!ParseStrictLongDouble(token, &parsed)) return false;
    *value = static_cast<Real>(parsed);
    return true;
  }
  if (slash == 0 || slash + 1 == token.size() || token.find('/', slash + 1) !=
      std::string::npos) return false;
  long double numerator = 0.0L, denominator = 0.0L;
  if (!ParseStrictLongDouble(token.substr(0, slash), &numerator) ||
      !ParseStrictLongDouble(token.substr(slash + 1), &denominator) ||
      denominator == 0.0L) return false;
  *value = static_cast<Real>(numerator / denominator);
  return std::isfinite(*value);
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 2) return EXIT_FAILURE;
  std::ifstream input(argv[1]);
  if (!input) return EXIT_FAILURE;
  const std::regex point_start(R"("point_xyz"\s*:\s*\[)");
  const std::regex coordinate_pattern("^\\s*\"([^\"]+)\"\\s*,?\\s*$");
  const std::regex values_start("\"values\"\\s*:\\s*\\{");
  const std::regex value_pattern(
      "\"([a-zA-Z_0-9\\[\\]]+)\"\\s*:\\s*\"([^\"]+)\"");
  std::string line;
  z4c_mms::AnalyticOracle oracle{};
  SplitOracle split{};
  bool have_point = false;
  bool in_values = false;
  int points = 0;
  int comparisons = 0;
  int values_this_point = 0;
  std::set<std::string> names_this_point;
  while (std::getline(input, line)) {
    std::smatch match;
    if (std::regex_search(line, point_start)) {
      if (have_point || in_values) return EXIT_FAILURE;
      std::array<Real, 3> point{};
      for (int coordinate = 0; coordinate < 3; ++coordinate) {
        if (!std::getline(input, line) ||
            !std::regex_match(line, match, coordinate_pattern) ||
            !ParseCoordinate(match[1].str(), &point[coordinate])) {
          std::cerr << "malformed/truncated rational point coordinate\n";
          return EXIT_FAILURE;
        }
      }
      if (!std::getline(input, line) || line.find(']') == std::string::npos) {
        std::cerr << "unterminated point_xyz\n";
        return EXIT_FAILURE;
      }
      z4c_mms::EvaluateAnalyticOracle(point[0], point[1], point[2], oracle);
      EvaluateSplitOracle(point[0], point[1], point[2], &split);
      have_point = true;
      ++points;
    } else if (have_point && std::regex_search(line, values_start)) {
      if (in_values) return EXIT_FAILURE;
      in_values = true;
      values_this_point = 0;
      names_this_point.clear();
    } else if (in_values && std::regex_search(line, match, value_pattern)) {
      Real observed = 0.0;
      Real split_observed = 0.0;
      Real generic_observed = 0.0;
      if (!Lookup(match[1].str(), oracle, &observed) ||
          !LookupSplit(match[1].str(), split, false, &split_observed) ||
          !LookupSplit(match[1].str(), split, true, &generic_observed) ||
          !names_this_point.insert(match[1].str()).second) {
        std::cerr << "unknown/duplicate generated reference key\n";
        return EXIT_FAILURE;
      }
      long double reference = 0.0L;
      if (!ParseStrictLongDouble(match[2].str(), &reference)) {
        std::cerr << "invalid or partially parsed reference value\n";
        return EXIT_FAILURE;
      }
      const long double scale = std::max({1.0L, std::abs(reference),
                                          std::abs(static_cast<long double>(observed))});
      const long double tolerance =
          256.0L * std::numeric_limits<Real>::epsilon() * scale;
      if (!std::isfinite(observed) || !std::isfinite(split_observed) ||
          !std::isfinite(generic_observed) ||
          std::abs(static_cast<long double>(observed) - reference) > tolerance ||
          std::abs(static_cast<long double>(split_observed) - reference) > tolerance ||
          std::abs(static_cast<long double>(generic_observed) - reference) > tolerance ||
          std::abs(static_cast<long double>(split_observed) -
                   static_cast<long double>(observed)) > tolerance ||
          std::abs(static_cast<long double>(generic_observed) -
                   static_cast<long double>(split_observed)) > tolerance) {
        std::cerr << "generated reference mismatch at " << match[1].str() << '\n';
        return EXIT_FAILURE;
      }
      ++comparisons;
      ++values_this_point;
    } else if (in_values && line.find('}') != std::string::npos) {
      if (values_this_point != 130) {
        std::cerr << "missing/extra value in generated reference record\n";
        return EXIT_FAILURE;
      }
      in_values = false;
      have_point = false;
    }
  }
  Real invalid = 0.0;
  if (in_values || have_point || ParseCoordinate("1/0", &invalid) ||
      ParseCoordinate("1/2junk", &invalid) || ParseCoordinate("/2", &invalid) ||
      Lookup("tensor_second[6][0][0]", oracle, &invalid) ||
      points != 13 || comparisons != 13 * 130) {
    std::cerr << "generated reference coverage mismatch: points=" << points
              << " comparisons=" << comparisons << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
