#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include "z4c/hyperboloidal/cartesian_patch.hpp"
namespace h = z4c::hyperboloidal;

// This utility evaluates stored states. It never calls RHS or an integrator.
int main(int argc, char **argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  if (argc != 3) return 2;
  std::ifstream input(argv[1], std::ios::binary);
  h::SphericalGhostGrid grid{};
  grid.radius = 1;
  input.read(reinterpret_cast<char *>(grid.n), sizeof(grid.n));
  input.read(reinterpret_cast<char *>(grid.first), sizeof(grid.first));
  input.read(reinterpret_cast<char *>(grid.h), sizeof(grid.h));
  if (!input || (grid.n[0] != 32 && grid.n[0] != 44)
      || grid.n[1] != grid.n[0] || grid.n[2] != grid.n[0]) return 3;
  h::LayerGaugeParameters gauge;
  gauge.physical_trace_lapse = true;
  gauge.preferred_source = false;
  gauge.scri_lapse_damping = 2;
  h::CartesianConformalPatch patch(grid, .5, 2, {true, .05, .95}, gauge, true);
  patch.kappa1 = 10;
  auto data = patch.Allocate("read-only restart snapshot");
  auto host = Kokkos::create_mirror_view(data);
  for (int f = 0; f < 25; ++f)
  for (int k = 0; k < grid.n[2]; ++k)
  for (int j = 0; j < grid.n[1]; ++j)
  for (int i = 0; i < grid.n[0]; ++i)
    input.read(reinterpret_cast<char *>(&host(0, f, k, j, i)), sizeof(Real));
  if (!input || input.peek() != std::char_traits<char>::eof()) return 4;
  Kokkos::deep_copy(data, host);
  const auto actual = patch.Diagnose(data);  // The sole private header difference.
  const auto q = h::BindCartesianFields(patch.deviations);
  const auto nodes = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), patch.active);
  const Real idx[3] = {1/grid.h[0], 1/grid.h[1], 1/grid.h[2]};
  double square[4]{}, delta_square = 0, prediction_error = 0;
  double m_difference = 0, z_difference = 0, theta_difference = 0;
  double first_jet_difference = 0, h_difference_max = 0;
  std::ofstream cells(argv[2], std::ios::binary);
  for (size_t point = 0; point < nodes.extent(0); ++point) {
    const int s = nodes(point), i = s % grid.n[0], j = s/grid.n[0] % grid.n[1];
    const int k = s/(grid.n[0]*grid.n[1]);
    const double x = grid.first[0]+i*grid.h[0], y = grid.first[1]+j*grid.h[1];
    const double z = grid.first[2]+k*grid.h[2];
    const auto p = patch.reference.At(x, y, z);
    auto original = h::LoadMeshJet<3>(q, idx, 0, k, j, i);
    auto composed = h::LoadComposedMeshJet<3>(q, idx, 0, k, j, i);
    h::AddReferenceJet(original, p, patch.reference);
    h::AddReferenceJet(composed, p, patch.reference);
    const auto c0 = h::EvolvedConstraints(original, h::CartesianOmega(original, p));
    const auto c1 = h::EvolvedConstraints(composed, h::CartesianOmega(composed, p));
    if (!c0.valid || !c1.valid || !c0.z4.valid || !c1.z4.valid) return 5;
    const auto g = h::Geometry(original.metric);
    // With fixed values and first jets, only the Ricci scalar changes:
    // dH = Omega^2 [chi*(g^di*g^dj-g^dd*g^ij)*d(g_ij,dd)
    //                    + 2*g^dd*d(chi,dd)], summed over d,i,j.
    double predicted = 0;
    for (int d = 0; d < 3; ++d) {
      predicted += 2*g.inverse[d][d]*(composed.chi.dd[d][d]-original.chi.dd[d][d]);
      first_jet_difference = std::max(first_jet_difference,
          std::abs(original.chi.d[d]-composed.chi.d[d]));
      for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) {
        predicted += original.chi.value*(g.inverse[d][a]*g.inverse[d][b]
            - g.inverse[d][d]*g.inverse[a][b])
            *(composed.metric.ddg[d][d][a][b]-original.metric.ddg[d][d][a][b]);
        first_jet_difference = std::max(first_jet_difference,
            std::abs(original.metric.dg[d][a][b]-composed.metric.dg[d][a][b]));
      }
    }
    predicted *= p.omega*p.omega;
    const double difference = c1.hamiltonian-c0.hamiltonian;
    prediction_error = std::max(prediction_error, std::abs(difference-predicted));
    h_difference_max = std::max(h_difference_max, std::abs(difference));
    delta_square += difference*difference;
    const double m = std::sqrt(c0.momentum_conformal_norm2);
    const double zn = std::sqrt(c0.z4.z_conformal_norm2);
    m_difference = std::max(m_difference,
        std::abs(m-std::sqrt(c1.momentum_conformal_norm2)));
    z_difference = std::max(z_difference,
        std::abs(zn-std::sqrt(c1.z4.z_conformal_norm2)));
    theta_difference = std::max(theta_difference,
        std::abs(original.theta.value-composed.theta.value));
    square[0] += c0.hamiltonian*c0.hamiltonian;
    square[1] += c1.hamiltonian*c1.hamiltonian;
    square[2] += m*m;
    square[3] += zn*zn;
    const double row[12] = {x,y,z,c0.hamiltonian,c1.hamiltonian,predicted,
        m,zn,original.theta.value,difference,
        m-std::sqrt(c1.momentum_conformal_norm2),
        zn-std::sqrt(c1.z4.z_conformal_norm2)};
    cells.write(reinterpret_cast<const char *>(row), sizeof(row));
  }
  if (!cells) return 6;
  const double n = nodes.extent(0);
  std::cout << std::setprecision(17)
      << "{\"diagnose_loader\":\"" << DIAGNOSTIC_LOADER << "\",\"active_cells\":" << n
      << ",\"diagnose\":{\"H\":" << actual.h_l2 << ",\"M\":" << actual.m_l2
      << ",\"Z\":" << actual.z_l2 << ",\"Theta\":" << actual.theta_l2
      << ",\"max_H\":" << actual.max_h << ",\"max_H_radius\":" << actual.max_h_radius
      << ",\"max_M\":" << actual.max_m << "},\"direct\":{\"H_original\":"
      << std::sqrt(square[0]/n) << ",\"H_composed\":" << std::sqrt(square[1]/n)
      << ",\"M\":" << std::sqrt(square[2]/n) << ",\"Z\":" << std::sqrt(square[3]/n)
      << "},\"H_difference_rms\":" << std::sqrt(delta_square/n)
      << ",\"H_difference_max\":" << h_difference_max
      << ",\"Ricci_H_difference_prediction_max_error\":" << prediction_error
      << ",\"momentum_difference_max\":" << m_difference
      << ",\"Z_difference_max\":" << z_difference
      << ",\"Theta_difference_max\":" << theta_difference
      << ",\"first_metric_chi_jet_difference_max\":" << first_jet_difference << "}\n";
  return prediction_error < 1e-11 && m_difference == 0 && z_difference == 0
      && theta_difference == 0 && first_jet_difference == 0 ? 0 : 1;
}
