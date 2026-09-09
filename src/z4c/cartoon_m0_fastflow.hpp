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
  Real flow_weight = 1.0;  // 1/|grad F|; applies to Theta-c as well as Theta
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
  // CE convergence is never a horizon-found or restart-success flag.
  bool ce_converged = false;
  Real expansion_target = 0.0, reference_radius = 0.0;
  Real solve_epsilon2_tolerance = 1.e-6, solve_epsilon_inf_tolerance = 1.e-5;
  Real outgoing_min = 0.0, outgoing_max = 0.0;
  Real physical_epsilon2 = 0.0, physical_epsilon_inf = 0.0;
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
  // reference_radius=0 preserves the legacy area-radius normalization.
  Real expansion_target = 0.0, reference_radius = 0.0;
  int lmax = 4, ntheta = 12, iterations = 300, backtracks = 24;
  bool batch_newton = true, candidate_policy = false;
  bool candidate_l32_window = false;
  int coarse_iterations = 96, candidate_points = 1061;
  Real promotion_max = 0.5, candidate_bound = 0.01, angular_ratio = 0.8;
  Real shape_change = 0.02, area_change = 0.01;
  Real flow_scale = 1.0, newton_switch = 0.1;
  Real epsilon2 = 1.e-6, epsilon_inf = 1.e-5, displacement = 0.1;
};
struct M0BracketOptions {
  int continuation_steps = 4;
  int dense_points = 1061, separation_points = 65, radial_points = 16;
  Real max_width_fraction = 0.25;
};
struct M0BracketPoint {
  Real theta, rho, z, radius, outgoing, ingoing, spacing;
};
struct M0BracketResult {
  M0CandidateSummary central, inner, outer;
  std::vector<M0CandidateSummary> continuation;
  std::array<std::vector<M0BracketPoint>, 3> profiles;
  bool supported = false, nested = false, signs_resolved = false;
  bool narrow = false, valid_geometry = false;
  // These are deliberately not inferred from a numerical sandwich.
  bool stability_operator_verified = false, spatially_validated = false;
  Real q = 0.05, reference_radius = 0.0;
  Real inner_margin = 0.0, outer_margin = 0.0;
  Real angular_uncertainty_inner = 0.0, angular_uncertainty_outer = 0.0;
  Real radial_gap_inner = 0.0, radial_gap_outer = 0.0;
  Real proper_inner_min = 0.0, proper_inner_max = 0.0;
  Real proper_outer_min = 0.0, proper_outer_max = 0.0;
  Real proper_width_max = 0.0, separation_quadrature_change = 0.0;
  int geometry_points = 0;
  std::string failure = "not_run";
};
M0BracketResult FindM0Bracket(const M0GeometrySampler&, const M0SolveOptions&,
                              const M0BracketOptions&, const M0CandidateSummary&);
void WriteM0Bracket(const std::string&, int cycle, Real time, int candidate,
                    const M0BracketResult&);
// Exposed separately for regression tests of invalid/non-nested sandwiches.
M0BracketResult VerifyM0Bracket(const M0GeometrySampler&, const M0BracketOptions&,
                                const M0BracketResult&);

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
  bool BracketSupported() const { return bracket_selected_ >= 0; }
  Real TimeFirstBracket() const { return first_bracket_time_; }
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
  M0BracketOptions bracket_options_;
  bool bracket_enabled_ = false;
  int ce_iterations_ = 96, bracket_selected_ = -1;
  Real first_bracket_time_ = -1.0;
  std::vector<M0BracketResult> brackets_;
  std::vector<int> bracket_candidate_ids_;
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
