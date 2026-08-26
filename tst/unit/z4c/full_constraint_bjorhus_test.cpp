//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE for details
//========================================================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>

#include <Kokkos_Core.hpp>

#include "athena.hpp"
#include "z4c/full_constraint_bjorhus.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {

bool NearlyEqual(const Real left, const Real right, const Real tolerance = 2.0e-13) {
  return std::abs(left - right) <=
         tolerance * std::max({Real(1.0), std::abs(left), std::abs(right)});
}

z4c::FullConstraintBjorhusFrame CartesianFrame(const int side0,
                                                const int side1 = 0) {
  const Real metric[3][3] = {{1.0, 0.0, 0.0},
                             {0.0, 1.0, 0.0},
                             {0.0, 0.0, 1.0}};
  const int side[3] = {side0, side1, 0};
  z4c::FullConstraintBjorhusFrame frame;
  if (z4c::MakeFullConstraintBjorhusFrame(metric, side, &frame) !=
      z4c::FullConstraintBjorhusStatus::valid) {
    std::cerr << "failed to construct Cartesian Bjorhus frame\n";
    std::exit(2);
  }
  return frame;
}

Real TangentialRateError(const z4c::FullConstraintBjorhusFrame &frame,
                         const z4c::FullConstraintBjorhusRates &rates) {
  Real normal = 0.0;
  for (int a = 0; a < 3; ++a) normal += frame.normal_u[a] * rates.vector_covector[a];
  Real maximum = 0.0;
  for (int a = 0; a < 3; ++a) {
    maximum = std::max(maximum,
                       std::abs(rates.vector_covector[a] -
                                frame.normal_d[a] * normal));
  }
  return maximum;
}

bool CheckOutgoingPulse() {
  const auto frame = CartesianFrame(1);
  z4c::FullConstraintBjorhusRates rates;
  // A manufactured purely outgoing pulse has zero projection on all four incoming
  // rows.  These nonzero values stand for its independently prescribed outgoing rows.
  const Real outgoing_before[4] = {0.8, -0.35, 0.42, -0.19};
  z4c::FullConstraintBjorhusCorrection correction;
  const auto status = z4c::SolveFullConstraintBjorhusCorrection(
      1.0, frame, rates, &correction);
  const auto outgoing_change =
      z4c::FullConstraintBjorhusInducedOutgoingRateChange(1.0, frame,
                                                          correction);
  return status == z4c::FullConstraintBjorhusStatus::valid &&
         NearlyEqual(correction.theta, 0.0) &&
         NearlyEqual(correction.gamma_u[0], 0.0) &&
         NearlyEqual(correction.gamma_u[1], 0.0) &&
         NearlyEqual(correction.gamma_u[2], 0.0) &&
         NearlyEqual(outgoing_change.theta, 0.0) &&
         NearlyEqual(outgoing_change.z_normal, 0.0) &&
         NearlyEqual(outgoing_change.vector_covector[0], 0.0) &&
         NearlyEqual(outgoing_change.vector_covector[1], 0.0) &&
         NearlyEqual(outgoing_change.vector_covector[2], 0.0) &&
         NearlyEqual(outgoing_before[0] + outgoing_change.theta,
                     outgoing_before[0]) &&
         NearlyEqual(outgoing_before[1] + outgoing_change.z_normal,
                     outgoing_before[1]) &&
         NearlyEqual(outgoing_before[2] + outgoing_change.vector_covector[1],
                     outgoing_before[2]) &&
         NearlyEqual(outgoing_before[3] + outgoing_change.vector_covector[2],
                     outgoing_before[3]);
}

bool CheckIncomingThetaZPulse(Real *maximum_induced_outgoing_rate) {
  constexpr Real chi = 1.44;
  const auto frame = CartesianFrame(1);
  z4c::FullConstraintBjorhusRates rates;
  rates.theta = 0.7;
  rates.z_normal = -0.4;
  rates.vector_covector[0] = 0.2;  // not an independent transverse row
  rates.vector_covector[1] = 0.25;
  rates.vector_covector[2] = -0.5;

  z4c::FullConstraintBjorhusCorrection correction;
  if (z4c::SolveFullConstraintBjorhusCorrection(chi, frame, rates,
                                                &correction) !=
      z4c::FullConstraintBjorhusStatus::valid) {
    return false;
  }
  const auto corrected = z4c::ApplyFullConstraintBjorhusCorrectionToRates(
      chi, frame, rates, correction);
  if (!NearlyEqual(corrected.theta, 0.0) ||
      !NearlyEqual(corrected.z_normal, 0.0) ||
      TangentialRateError(frame, corrected) > 2.0e-13) {
    return false;
  }

  const auto outgoing_change =
      z4c::FullConstraintBjorhusInducedOutgoingRateChange(chi, frame,
                                                          correction);
  *maximum_induced_outgoing_rate =
      std::max({std::abs(outgoing_change.theta),
                std::abs(outgoing_change.z_normal),
                std::abs(outgoing_change.vector_covector[1]),
                std::abs(outgoing_change.vector_covector[2])});
  // This nonzero result is a required limitation check: a Theta/Gamma-only sparse
  // correction cannot also preserve every outgoing characteristic-rate projection.
  return *maximum_induced_outgoing_rate > 0.1;
}

bool CheckCornerFrameAndAxisOwnership() {
  const auto corner = CartesianFrame(1, -1);
  if (!NearlyEqual(corner.normal_u[0], 1.0 / std::sqrt(2.0)) ||
      !NearlyEqual(corner.normal_u[1], -1.0 / std::sqrt(2.0))) {
    return false;
  }
  const int corner_side[3] = {1, -1, 0};
  const int z_only_side[3] = {0, -1, 0};
  return z4c::FullConstraintBjorhusOwnsPoint(0, corner_side, false) &&
         !z4c::FullConstraintBjorhusOwnsPoint(1, corner_side, false) &&
         z4c::FullConstraintBjorhusOwnsPoint(1, z_only_side, false) &&
         !z4c::FullConstraintBjorhusOwnsPoint(1, z_only_side, true);
}

bool CheckFailClosed() {
  auto frame = CartesianFrame(1);
  z4c::FullConstraintBjorhusRates rates;
  z4c::FullConstraintBjorhusCorrection correction;
  if (z4c::SolveFullConstraintBjorhusCorrection(0.0, frame, rates,
                                                &correction) !=
      z4c::FullConstraintBjorhusStatus::invalid_coefficient) {
    return false;
  }
  const Real indefinite[3][3] = {{1.0, 0.0, 0.0},
                                 {0.0, -1.0, 0.0},
                                 {0.0, 0.0, -1.0}};
  const int side[3] = {1, 0, 0};
  return z4c::MakeFullConstraintBjorhusFrame(indefinite, side, &frame) ==
         z4c::FullConstraintBjorhusStatus::invalid_metric;
}

std::uint64_t DeterministicChecksum() {
  const auto frame = CartesianFrame(1, -1);
  std::uint64_t hash = UINT64_C(1469598103934665603);
  for (int sample = 0; sample < 257; ++sample) {
    z4c::FullConstraintBjorhusRates rates;
    rates.theta = std::sin(0.13 * sample);
    rates.z_normal = std::cos(0.07 * sample);
    for (int a = 0; a < 3; ++a) {
      rates.vector_covector[a] = std::sin(0.11 * sample + 0.3 * a);
    }
    z4c::FullConstraintBjorhusCorrection correction;
    if (z4c::SolveFullConstraintBjorhusCorrection(
            0.9 + 0.001 * sample, frame, rates, &correction) !=
        z4c::FullConstraintBjorhusStatus::valid) {
      return 0;
    }
    const auto corrected = z4c::ApplyFullConstraintBjorhusCorrectionToRates(
        0.9 + 0.001 * sample, frame, rates, correction);
    if (!NearlyEqual(corrected.theta, 0.0, 2.0e-12) ||
        !NearlyEqual(corrected.z_normal, 0.0, 2.0e-12) ||
        TangentialRateError(frame, corrected) > 2.0e-12) {
      return 0;
    }
    const Real values[4] = {correction.theta, correction.gamma_u[0],
                            correction.gamma_u[1], correction.gamma_u[2]};
    for (const Real value : values) {
      std::uint64_t bits = 0;
      static_assert(sizeof(value) <= sizeof(bits));
      std::memcpy(&bits, &value, sizeof(value));
      hash ^= bits;
      hash *= UINT64_C(1099511628211);
    }
  }
  return hash;
}

}  // namespace

int main(int argc, char **argv) {
#if MPI_PARALLEL_ENABLED
  MPI_Init(&argc, &argv);
#endif
  Kokkos::initialize(argc, argv);
  Real induced_outgoing_rate = 0.0;
  bool passed = CheckOutgoingPulse() &&
                CheckIncomingThetaZPulse(&induced_outgoing_rate) &&
                CheckCornerFrameAndAxisOwnership() && CheckFailClosed();
  const std::uint64_t checksum = DeterministicChecksum();
  passed = passed && checksum != 0;
#if MPI_PARALLEL_ENABLED
  std::uint64_t minimum_checksum = checksum;
  std::uint64_t maximum_checksum = checksum;
  MPI_Allreduce(MPI_IN_PLACE, &minimum_checksum, 1, MPI_UINT64_T, MPI_MIN,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &maximum_checksum, 1, MPI_UINT64_T, MPI_MAX,
                MPI_COMM_WORLD);
  passed = passed && minimum_checksum == maximum_checksum;
#endif
  if (passed) {
    std::cout << "full constraint Bjorhus manufactured tests passed"
              << " checksum=" << checksum
              << " induced_outgoing_rate=" << induced_outgoing_rate << '\n';
  } else {
    std::cerr << "full constraint Bjorhus manufactured tests failed\n";
  }
  Kokkos::finalize();
#if MPI_PARALLEL_ENABLED
  MPI_Finalize();
#endif
  return passed ? 0 : 1;
}
