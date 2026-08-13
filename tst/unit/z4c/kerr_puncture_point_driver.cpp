//========================================================================================
// Focused host driver for the reusable Kerr-puncture point evaluator.
//========================================================================================

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "pgen/z4c/kerr_puncture.hpp"

namespace {

template <typename Scalar>
void PrintTensor(const kerr_puncture::SymmetricTensor3<Scalar> &tensor) {
  std::cout << ' ' << tensor.xx << ' ' << tensor.xy << ' ' << tensor.xz
            << ' ' << tensor.yy << ' ' << tensor.yz << ' ' << tensor.zz;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 9) {
    std::cerr << "usage: driver MAP GAUGE M CHI Z_H X1 X2 X3\n";
    return EXIT_FAILURE;
  }
  const std::string map_name = argv[1];
  const std::string gauge_name = argv[2];
  const double mass = std::stod(argv[3]);
  const double chi = std::stod(argv[4]);
  const double center = std::stod(argv[5]);
  const double x1 = std::stod(argv[6]);
  const double x2 = std::stod(argv[7]);
  const double x3 = std::stod(argv[8]);
  const kerr_puncture::Parameters<double> parameters{mass, chi, center};
  kerr_puncture::PointData<double> data;
  if (map_name == "cartesian" && gauge_name == "precollapsed") {
    data = kerr_puncture::Evaluate<
        kerr_puncture::CoordinateMap::cartesian_xyz,
        kerr_puncture::GaugeChoice::pre_collapsed>(x1, x2, x3, parameters);
  } else if (map_name == "cartesian" && gauge_name == "stationary") {
    data = kerr_puncture::Evaluate<
        kerr_puncture::CoordinateMap::cartesian_xyz,
        kerr_puncture::GaugeChoice::stationary>(x1, x2, x3, parameters);
  } else if (map_name == "cartoon" && gauge_name == "precollapsed") {
    data = kerr_puncture::Evaluate<
        kerr_puncture::CoordinateMap::half_rho_z_suppressed_y_v2,
        kerr_puncture::GaugeChoice::pre_collapsed>(x1, x2, x3, parameters);
  } else if (map_name == "cartoon" && gauge_name == "stationary") {
    data = kerr_puncture::Evaluate<
        kerr_puncture::CoordinateMap::half_rho_z_suppressed_y_v2,
        kerr_puncture::GaugeChoice::stationary>(x1, x2, x3, parameters);
  } else {
    std::cerr << "invalid map/gauge\n";
    return EXIT_FAILURE;
  }

  std::cout << std::setprecision(17)
            << static_cast<int>(data.valid) << ' '
            << static_cast<int>(data.physical_adm_available) << ' '
            << static_cast<int>(data.at_puncture) << ' '
            << data.isotropic_radius << ' ' << data.boyer_lindquist_radius
            << ' ' << data.r_plus << ' ' << data.r_minus << ' '
            << data.horizon_radius << ' ' << data.lapse << ' '
            << data.shift[0] << ' ' << data.shift[1] << ' ' << data.shift[2]
            << ' ' << data.psi4;
  PrintTensor(data.spatial_metric);
  PrintTensor(data.extrinsic_curvature);
  std::cout << ' ' << data.conformal_chi;
  PrintTensor(data.conformal_metric);
  std::cout << ' ' << data.trace_extrinsic_curvature;
  PrintTensor(data.conformal_tracefree_curvature);
  std::cout << '\n';
  return EXIT_SUCCESS;
}
