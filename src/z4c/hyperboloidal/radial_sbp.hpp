// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE").
#ifndef Z4C_HYPERBOLOIDAL_RADIAL_SBP_HPP_
#define Z4C_HYPERBOLOIDAL_RADIAL_SBP_HPP_

#include <vector>
#include "z4c/hyperboloidal/cmc_reference.hpp"

namespace z4c {
namespace hyperboloidal {

// HOST-ONLY model problem on a boundary-fitted spherical shell. Evolves the
// outgoing radial characteristic q_t+c_+(r) q_r=0. This is not a 3D Z4c boundary
// closure. D is the diagonal-norm second-order SBP derivative (first order at
// endpoints); SAT imposes only the inner inflow. Scri is an evolved endpoint.
// E=1/2 q^T H C^{-1} q obeys
// E'=-q_N^2/2-(q_0-g)^2/2+g^2/2, exactly at the semidiscrete level.
class RadialSBP {
 public:
  RadialSBP(CMCReference<double> ref, double inner, int intervals)
      : ref_(ref), inner_(inner), intervals_(intervals) {
    ref_.Validate();
    if (!std::isfinite(inner) || inner <= 0 || inner >= ref.scri_radius ||
        intervals < 2) {
      throw std::invalid_argument("Radial SBP requires 0<inner<S and intervals>=2");
    }
    dr_ = (ref.scri_radius-inner)/intervals;
  }
  int Size() const { return intervals_+1; }
  double Radius(int i) const {
    return i == intervals_ ? ref_.scri_radius : inner_+i*dr_;
  }
  double Weight(int i) const {
    return dr_*(i == 0 || i == intervals_ ? 0.5 : 1);
  }
  double MaxSpeed() const { return ref_.OutgoingSpeed(ref_.scri_radius); }
  double Spacing() const { return dr_; }
  std::vector<double> RHS(const std::vector<double> &q, double inflow) const {
    CheckSize(q);
    std::vector<double> rhs(Size());
    for (int i = 0; i < Size(); ++i) {
      const double derivative = i == 0 ? (q[1]-q[0])/dr_ :
          (i == intervals_ ? (q[i]-q[i-1])/dr_ : (q[i+1]-q[i-1])/(2*dr_));
      rhs[i] = -ref_.OutgoingSpeed(Radius(i))*derivative;
    }
    rhs[0] -= ref_.OutgoingSpeed(inner_)*(q[0]-inflow)/Weight(0);
    return rhs;
  }
  double EnergyProduct(const std::vector<double> &q,
                       const std::vector<double> &v) const {
    CheckSize(q);
    CheckSize(v);
    double sum = 0;
    for (int i = 0; i < Size(); ++i) {
      sum += Weight(i)*q[i]*v[i]/ref_.OutgoingSpeed(Radius(i));
    }
    return sum;
  }

 private:
  void CheckSize(const std::vector<double> &q) const {
    if (q.size() != static_cast<std::size_t>(Size())) {
      throw std::invalid_argument("Radial SBP vector size mismatch");
    }
  }
  CMCReference<double> ref_;
  double inner_, dr_;
  int intervals_;
};

}  // namespace hyperboloidal
}  // namespace z4c
#endif  // Z4C_HYPERBOLOIDAL_RADIAL_SBP_HPP_
