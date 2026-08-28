#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "z4c/state_admissibility.hpp"

namespace {

using z4c::EvaluateZ4cState;
using z4c::ProjectAdmissibleConformalState;
using z4c::SelectFirstZ4cFailureKey;
using z4c::Z4cStateFailureReason;

void Require(const bool condition, const char *message) {
  if (!condition) {
    std::cerr << "state admissibility test: " << message << '\n';
    std::exit(EXIT_FAILURE);
  }
}

std::array<Real, 25> ValidState() {
  std::array<Real, 25> values{};
  values[0] = 0.8;  // chi
  values[1] = 1.0;  // gxx
  values[4] = 1.0;  // gyy
  values[6] = 1.0;  // gzz
  values[18] = 1.0; // alpha
  return values;
}

}  // namespace

int main() {
  auto values = ValidState();
  auto state = EvaluateZ4cState(values.data(), static_cast<int>(values.size()));
  Require(state.reason == Z4cStateFailureReason::valid, "valid SPD state rejected");
  Require(state.metric.determinant == 1.0, "valid determinant mismatch");

  Real metric[6] = {2.0, 0.2, -0.1, 1.5, 0.3, 1.25};
  Real atracefree[6] = {0.3, -0.1, 0.05, -0.2, 0.04, 0.4};
  Require(ProjectAdmissibleConformalState(metric, atracefree),
          "valid SPD projection rejected");
  const auto projected = z4c::EvaluateConformalMetric(
      metric[0], metric[1], metric[2], metric[3], metric[4], metric[5]);
  Require(std::fabs(projected.determinant - 1.0) < 64.0 * std::numeric_limits<Real>::epsilon(),
          "projection did not normalize determinant");
  const Real trace = (metric[3] * metric[5] - metric[4] * metric[4]) * atracefree[0] +
      2.0 * (metric[2] * metric[4] - metric[1] * metric[5]) * atracefree[1] +
      2.0 * (metric[1] * metric[4] - metric[2] * metric[3]) * atracefree[2] +
      (metric[0] * metric[5] - metric[2] * metric[2]) * atracefree[3] +
      2.0 * (metric[1] * metric[2] - metric[0] * metric[4]) * atracefree[4] +
      (metric[0] * metric[3] - metric[1] * metric[1]) * atracefree[5];
  Require(std::fabs(trace) < 256.0 * std::numeric_limits<Real>::epsilon(),
          "projection did not make A trace-free");

  values = ValidState();
  values[6] = -1.0;
  state = EvaluateZ4cState(values.data(), static_cast<int>(values.size()));
  Require(state.reason == Z4cStateFailureReason::nonpositive_metric_pivot_2,
          "negative determinant rejected with wrong reason");

  values = ValidState();
  values[6] = 0.0;
  state = EvaluateZ4cState(values.data(), static_cast<int>(values.size()));
  Require(state.reason == Z4cStateFailureReason::nonpositive_metric_pivot_2,
          "zero determinant not rejected");

  values = ValidState();
  values[1] = std::numeric_limits<Real>::quiet_NaN();
  state = EvaluateZ4cState(values.data(), static_cast<int>(values.size()));
  Require(state.reason == Z4cStateFailureReason::nonfinite_component,
          "NaN metric component not rejected");

  values = ValidState();
  values[1] = -1.0;
  values[4] = -1.0;
  state = EvaluateZ4cState(values.data(), static_cast<int>(values.size()));
  Require(state.metric.determinant > 0.0 &&
              state.reason == Z4cStateFailureReason::nonpositive_metric_pivot_0,
          "positive-determinant indefinite metric accepted");
  Real invalid_metric[6] = {-1.0, 0.0, 0.0, -1.0, 0.0, 1.0};
  Real invalid_a[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  const auto metric_before = std::array<Real, 6>{-1.0, 0.0, 0.0, -1.0, 0.0, 1.0};
  const auto a_before = std::array<Real, 6>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  Require(!ProjectAdmissibleConformalState(invalid_metric, invalid_a),
          "indefinite metric projection accepted");
  for (int component = 0; component < 6; ++component) {
    Require(invalid_metric[component] == metric_before[component] &&
                invalid_a[component] == a_before[component],
            "invalid projection mutated source state");
  }

  for (const Real chi : {0.0, -1.0, std::numeric_limits<Real>::quiet_NaN()}) {
    values = ValidState();
    values[0] = chi;
    state = EvaluateZ4cState(values.data(), static_cast<int>(values.size()));
    Require(state.reason != Z4cStateFailureReason::valid, "invalid chi accepted");
  }
  values = ValidState();
  values[19] = std::numeric_limits<Real>::infinity();
  state = EvaluateZ4cState(values.data(), static_cast<int>(values.size()));
  Require(state.reason == Z4cStateFailureReason::nonfinite_component &&
              state.first_nonfinite_component == 19,
          "nonfinite nonmetric component not identified");

  values = ValidState();
  values[18] = 0.0;
  state = EvaluateZ4cState(values.data(), static_cast<int>(values.size()));
  Require(state.reason == Z4cStateFailureReason::nonpositive_lapse,
          "nonpositive lapse accepted");
  state = EvaluateZ4cState(values.data(), static_cast<int>(values.size()), false);
  Require(state.reason == Z4cStateFailureReason::valid,
          "shock-avoiding negative-lapse policy was not honored");

  constexpr unsigned long long rank0 = (41ULL << 32) | 9ULL;
  constexpr unsigned long long rank1 = (19ULL << 32) | 27ULL;
  constexpr unsigned long long rank2 = (19ULL << 32) | 11ULL;
  Require(SelectFirstZ4cFailureKey(rank0, rank1) == rank1 &&
              SelectFirstZ4cFailureKey(rank1, rank0) == rank1 &&
              SelectFirstZ4cFailureKey(SelectFirstZ4cFailureKey(rank0, rank1), rank2) ==
                  rank2,
          "first-failure key selection is not deterministic by GID then ordinal");

  std::cout << "Z4C_STATE_ADMISSIBILITY_UNIT_PASS\n";
  return EXIT_SUCCESS;
}
