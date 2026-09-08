//========================================================================================
//! \file cartoon_m0_fastflow.hpp
//! \brief Axisymmetric m=0 adapter for the existing FastFlow horizon finder.
//========================================================================================
#ifndef Z4C_CARTOON_M0_FASTFLOW_HPP_
#define Z4C_CARTOON_M0_FASTFLOW_HPP_

#include <array>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "athena.hpp"
#include "z4c/z4c_restart.hpp"

class MeshBlockPack;
class ParameterInput;

namespace z4c {
// Normalized Y_l0 and first/second theta derivatives; stable through high l.
std::array<Real, 3> M0Harmonic(int l, Real theta);


struct M0AdmSample {
  bool valid = false;
  Real spacing = 0.0;
  int error = 1;  // 0 valid, 1 unavailable stencil, 2 nonfinite fields
  std::array<Real, 6> metric{};
  std::array<Real, 6> curvature{};
  std::array<Real, 18> metric_derivative{};
};

struct M0SurfacePoint {
  bool valid = false;
  Real expansion = 0.0;
  Real ingoing_expansion = 0.0;
  Real flow_residual = 0.0;
  Real area_factor = 0.0;  // dA/(dtheta dphi)
  Real spin_integrand_z = 0.0;
};

struct M0AngularStage {
  int lmax, ntheta, iterations;
  Real epsilon2, epsilon_inf, area;
  bool verified;
  std::string failure;
};

struct M0CandidateSummary {
  std::vector<M0AngularStage> angular_stages;
  bool angular_candidate = false;
  int solved_lmax = 0;
  bool converged = false;
  bool verified = false;
  Real epsilon_inf = 0.0;
  Real ingoing_min = 0.0, ingoing_max = 0.0;
  Real spacing = 0.0;
  int iterations = 0;
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
  Real fresh_initial_radius = 0.0;
  std::vector<Real> coefficients;
};

struct M0AxisSample {
  bool valid = false;
  Real z = 0.0;
  Real lapse = 0.0;
};

//! Rotate physical-Cartesian covariant tensors from phi=0 to arbitrary phi.
M0AdmSample RotateM0AdmSample(const M0AdmSample& sample, Real phi);

//! Direct, flow-independent outgoing expansion evaluation for F=r-h(theta).
M0SurfacePoint EvaluateM0SurfacePoint(Real theta, Real radius, Real radius_theta,
                                      Real radius_theta_theta, const M0AdmSample& sample);
Real M0HorizonMass(Real area, Real spin_z);
bool SelectM0AxisLapseMinimum(const std::vector<M0AxisSample>& samples, int sign,
                              Real* center_z, Real* lapse);

//! Candidate-specific fresh-sphere radii.  The origin sphere encloses the
//! lapse minima with the requested margin, while equal-radius mirror spheres
//! remain strictly disjoint when 0 < pair_fraction < 1.
Real M0OriginInitialRadius(Real configured_radius, Real lapse_radius_factor,
                           Real plus_center_z, Real minus_center_z);
Real M0DisjointPairInitialRadius(Real configured_radius, Real pair_fraction,
                                 Real plus_center_z, Real minus_center_z);

//! Deterministic accepted-candidate selection. Returns -1 on failure.
int SelectM0Outermost(const std::vector<M0CandidateSummary>& candidates);
int SelectM0Single(const std::vector<M0CandidateSummary>& candidates);
bool SelectM0MirrorPair(const std::vector<M0CandidateSummary>& candidates,
                        Real relative_tolerance, int* plus, int* minus);
bool ValidateM0RestartState(const Z4cM0FastFlowRestartState& state, int lmax,
                            std::string* reason);
std::vector<M0CandidateSummary> RestoreM0Candidates(
    const Z4cM0FastFlowRestartState& state, int lmax, const std::vector<Real>& weights,
    const std::vector<Real>& y0);
Real MinimumM0SelectedRadius(const std::vector<M0CandidateSummary>& candidates,
                             const std::vector<int>& selected);
Real M0SelectedCenterZ(const std::vector<M0CandidateSummary>& candidates,
                       const std::vector<int>& selected);

// Geometry callbacks are independent of grid centering and surface optimization.
using M0GeometrySampler =
    std::function<std::vector<M0AdmSample>(const std::vector<std::array<Real, 2>>&)>;
struct M0SolveOptions {
  int lmax = 4, ntheta = 12, iterations = 300, backtracks = 24;
  bool batch_newton = true, candidate_policy = false;
  bool candidate_l32_window = false;
  int coarse_iterations = 96, candidate_points = 1061;
  Real promotion_max = 0.5, candidate_bound = 0.01, angular_ratio = 0.8;
  Real shape_change = 0.02, area_change = 0.01;
  Real flow_scale = 1.0, newton_switch = 0.1;
  Real epsilon2 = 1.e-6, epsilon_inf = 1.e-5, displacement = 0.1;
};
M0CandidateSummary AssessM0Surface(const M0GeometrySampler&, int ntheta,
                                   const M0CandidateSummary&);
M0CandidateSummary PreferM0Recentered(const M0CandidateSummary&,
                                      const M0CandidateSummary&);
M0CandidateSummary SolveM0Surface(const M0GeometrySampler& sample,
                                  const M0SolveOptions& options,
                                  const std::string& branch, Real center, Real radius,
                                  const std::vector<Real>& seed = {});

M0CandidateSummary SolveM0Refined(const M0GeometrySampler&, const M0SolveOptions&,
    int start_l, const std::string& branch, Real center, Real radius,
    const std::vector<Real>& seed = {});

std::vector<M0CandidateSummary> RestoreM0Seeds(const Z4cM0FastFlowRestartState&,
                                               int lmax);

//! Cartoon-only implementation composed by FastFlow; it is not a second public finder.
class CartoonM0FastFlow {
 public:
  CartoonM0FastFlow(MeshBlockPack* pack, ParameterInput* pin, int horizon);
  ~CartoonM0FastFlow();
  bool ShouldSearch(int cycle, Real time) const;
  void Find(int cycle, Real time, bool force = false);
  void Write(int cycle, Real time);

  // nvcc requires a member function enclosing an extended KOKKOS_LAMBDA to be
  // publicly accessible.  Keep only the two kernel-launching samplers public;
  // they remain implementation details of the composed Cartoon finder.
  M0AdmSample SampleAdm(Real rho, Real z) const;
  std::vector<M0AdmSample> SampleAdmBatch(
      const std::vector<std::array<Real, 2>>& points) const;
  M0AxisSample SampleAxisLapse(Real z) const;
  std::vector<M0AxisSample> SampleAxisLapseBatch(const std::vector<Real>& z) const;

  bool Found() const { return found_; }
  bool StrictlyVerified() const {
    for (int i : selected_) if (candidates_[i].verified) return true;
    return false;
  }
  int LastSearchCycle() const { return last_search_cycle_; }
  Real InitialRadius() const { return initial_radius_; }
  Real MinimumRadius() const;
  Real SelectedCenterZ() const;
  //! Restart-authoritative first accepted-surface time.
  Real TimeFirstFound() const { return time_first_found_; }
  int Lmax() const { return lmax_; }
  int Ntheta() const { return ntheta_; }
  int Iterations() const { return iterations_; }
  int FindInterval() const { return find_interval_; }
  Real StartTime() const { return start_time_; }
  Real StopTime() const { return stop_time_; }

 private:
  M0CandidateSummary SearchCandidate(const std::string& branch, Real center_z,
                                     Real fresh_initial_radius,
                                     const std::vector<Real>& warm_start);
  void Restore();
  void Capture();

  M0SolveOptions solve_options_;
  bool outermost_selection_ = true;
  int radius_count_ = 8;
  int l_start_ = 0, search_count_ = 0, discovery_interval_ = 8;
  Real tracking_residual_ = 0.01;
  std::vector<M0CandidateSummary> last_trial_;
  Real radius_min_ = 0.0;
  std::vector<M0CandidateSummary> last_good_;
  MeshBlockPack* pack_;
  ParameterInput* pin_;
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
  bool adaptive_initial_radius_;
  Real origin_lapse_radius_factor_;
  Real pair_disjoint_fraction_;
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
  FILE* output_ = nullptr;
};

}  // namespace z4c

#endif  // Z4C_CARTOON_M0_FASTFLOW_HPP_
