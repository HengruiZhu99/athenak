//========================================================================================
//! \file cartoon_m0_fastflow.hpp
//! \brief Axisymmetric m=0 adapter for the existing FastFlow horizon finder.
//========================================================================================
#ifndef Z4C_CARTOON_M0_FASTFLOW_HPP_
#define Z4C_CARTOON_M0_FASTFLOW_HPP_

#include <array>
#include <cstdio>
#include <string>
#include <vector>

#include "athena.hpp"
#include "z4c/z4c_restart.hpp"

class MeshBlockPack;
class ParameterInput;

namespace z4c {

struct M0AdmSample {
  bool valid = false;
  std::array<Real, 6> metric{};
  std::array<Real, 6> curvature{};
  std::array<Real, 18> metric_derivative{};
};

struct M0SurfacePoint {
  bool valid = false;
  Real expansion = 0.0;
  Real flow_residual = 0.0;
  Real area_factor = 0.0;  // dA/(dtheta dphi)
  Real spin_integrand_z = 0.0;
};

struct M0CandidateSummary {
  bool converged = false;
  std::string branch;
  std::string failure = "not_run";
  Real center_z = 0.0;
  Real axis_extremum_z = 0.0;
  Real center_lapse = 0.0;
  Real area = 0.0;
  Real irreducible_mass = 0.0;
  Real mass = 0.0;
  Real spin_z = 0.0;
  Real mean_radius = 0.0;
  Real minimum_radius = 0.0;
  Real direct_residual = 0.0;
  Real flow_residual = 0.0;
  std::vector<Real> coefficients;
};

struct M0AxisSample {
  bool valid = false;
  Real z = 0.0;
  Real lapse = 0.0;
};

//! Rotate physical-Cartesian covariant tensors from phi=0 to arbitrary phi.
M0AdmSample RotateM0AdmSample(const M0AdmSample &sample, Real phi);

//! Direct, flow-independent outgoing expansion evaluation for F=r-h(theta).
M0SurfacePoint EvaluateM0SurfacePoint(Real theta, Real radius,
                                      Real radius_theta,
                                      Real radius_theta_theta,
                                      const M0AdmSample &sample);
Real M0HorizonMass(Real area, Real spin_z);
bool SelectM0AxisLapseMinimum(const std::vector<M0AxisSample> &samples,
                              int sign, Real *center_z, Real *lapse);

//! Deterministic accepted-candidate selection. Returns -1 on failure.
int SelectM0Single(const std::vector<M0CandidateSummary> &candidates);
bool SelectM0MirrorPair(const std::vector<M0CandidateSummary> &candidates,
                        Real relative_tolerance, int *plus, int *minus);
bool ValidateM0RestartState(const Z4cM0FastFlowRestartState &state, int lmax,
                            std::string *reason);
std::vector<M0CandidateSummary> RestoreM0Candidates(
    const Z4cM0FastFlowRestartState &state, int lmax,
    const std::vector<Real> &weights, const std::vector<Real> &y0);
Real MinimumM0SelectedRadius(const std::vector<M0CandidateSummary> &candidates,
                             const std::vector<int> &selected);
Real M0SelectedCenterZ(const std::vector<M0CandidateSummary> &candidates,
                       const std::vector<int> &selected);

//! Cartoon-only implementation composed by FastFlow; it is not a second public finder.
class CartoonM0FastFlow {
 public:
  CartoonM0FastFlow(MeshBlockPack *pack, ParameterInput *pin, int horizon);
  ~CartoonM0FastFlow();
  bool ShouldSearch(int cycle, Real time) const;
  void Find(int cycle, Real time);
  void Write(int cycle, Real time);

  bool Found() const { return found_; }
  int LastSearchCycle() const { return last_search_cycle_; }
  Real InitialRadius() const { return initial_radius_; }
  Real MinimumRadius() const;
  Real SelectedCenterZ() const;
  //! Not restart-authoritative: schema 1 has no first-found-time field.
  Real TimeFirstFound() const { return time_first_found_; }
  int Lmax() const { return lmax_; }
  int Ntheta() const { return ntheta_; }
  int Iterations() const { return iterations_; }
  int FindInterval() const { return find_interval_; }
  Real StartTime() const { return start_time_; }
  Real StopTime() const { return stop_time_; }

 private:
  M0CandidateSummary SearchCandidate(const std::string &branch, Real center_z,
                                     const std::vector<Real> &warm_start);
  M0AdmSample SampleAdm(Real rho, Real z) const;
  M0AxisSample SampleAxisLapse(Real z) const;
  void Restore();
  void Capture();

  MeshBlockPack *pack_;
  ParameterInput *pin_;
  int horizon_;
  int lmax_;
  int ntheta_;
  int iterations_;
  int find_interval_;
  Real start_time_;
  Real stop_time_;
  Real initial_radius_;
  Real flow_scale_;
  Real hrms_tolerance_;
  Real mass_tolerance_;
  Real direct_tolerance_;
  Real pair_tolerance_;
  Real center_seed_;
  Real axis_search_bound_;
  int axis_search_samples_;
  std::string mode_;
  std::vector<Real> theta_;
  std::vector<Real> weights_;
  std::vector<Real> y0_;
  std::vector<Real> dy0_;
  std::vector<Real> ddy0_;
  std::vector<M0CandidateSummary> candidates_;
  std::vector<int> selected_;
  bool found_ = false;
  int last_search_cycle_ = -1;
  Real last_search_time_ = 0.0;
  Real time_first_found_ = -1.0;
  FILE *output_ = nullptr;
};

}  // namespace z4c

#endif  // Z4C_CARTOON_M0_FASTFLOW_HPP_
