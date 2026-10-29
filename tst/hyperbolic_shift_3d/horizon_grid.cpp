// Optional initial-data-only probe of the actual composite Cartesian/shell grid.
// No evolution, RHS evaluation, projection, or standard-Z4c test is performed.
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "globals.hpp"
#include "parameter_input.hpp"
#include "z4c/fastflow.hpp"
#include "z4c/hyperboloidal/horizon_geometry.hpp"
#include "z4c/hyperboloidal/hyperbolic_composite.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace h = z4c::hyperboloidal;

int main(int argc, char **argv) {
  #if MPI_PARALLEL_ENABLED
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &global_variable::my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &global_variable::nranks);
  #else
    global_variable::my_rank = 0; global_variable::nranks = 1;
  #endif
  Kokkos::initialize(argc, argv);
  int status = 0;
  try {
    if (argc < 4) throw std::invalid_argument(
        "usage: horizon_grid n initial.txt output-prefix [rms_tol=1e-8] [lmax=6] "
        "[ntheta=12] [iterations=500] [radius=.07] [flow_alpha=.02]");
    const int n = std::stoi(argv[1]);
    const std::string initial = argv[2], output = argv[3];
    const double tolerance = argc > 4 ? std::stod(argv[4]) : 1e-8;
    const int lmax = argc > 5 ? std::stoi(argv[5]) : 6;
    const int ntheta = argc > 6 ? std::stoi(argv[6]) : 12;
    const int iterations = argc > 7 ? std::stoi(argv[7]) : 500;
    const double radius = argc > 8 ? std::stod(argv[8]) : .07;
    const double alpha = argc > 9 ? std::stod(argv[9]) : .02;
    if (std::filesystem::exists(output+".result.txt")) {
      throw std::runtime_error("refusing to overwrite existing horizon-grid result");
    }
    h::HyperbolicComposite e(cubed_sphere::CompositeMesh(.125,{.45,.75,1},1,n,3),false);
    std::ifstream in(initial);
    size_t count = 0; in >> count;
    if (count != e.state.size()) throw std::runtime_error("initial field count mismatch");
    for (auto &v : e.state) {
      if (!(in >> v) || !std::isfinite(v)) throw std::runtime_error("invalid initial field");
    }
    std::string trailing;
    if (in >> trailing) throw std::runtime_error("trailing initial data");
    // Fill the existing interpolation halos once; no evolution operator runs.
    const auto padded = e.Padded(e.state);
    ParameterInput pin;
    std::ostringstream input;
    input << std::setprecision(17) << "<job>\nbasename=" << output
          << "\n<fastflow>\nntheta=" << ntheta << "\nlmax=" << lmax
          << "\nflow=standard\nflow_iterations_0=" << iterations
          << "\nflow_alpha_beta_const_0=" << alpha
          << "\ninitial_radius_0=" << radius << "\nmass_tol_0=1e-10\n"
          << "expansion_rms_tol_0=" << tolerance
          << "\nhmean_tol_0=1e8\nstart_time_0=0\nstop_time_0=1\n";
    std::istringstream stream(input.str()); pin.LoadFromStream(stream);
    FastFlow finder(nullptr,&pin,0,[&](const Real x[3],z4c::HorizonGeometry<Real>&q) {
      // This initial-data probe partitions by direction. The production sampler
      // must instead use its actual finest-active-block ownership.
      const int owner = x[0] < 0 ? 0 : global_variable::nranks-1;
      if (owner != global_variable::my_rank) return false;
      try {
        h::HyperbolicComposite::V v;
        std::vector<std::array<Real,3>> d;
        std::vector<std::array<std::array<Real,3>,3>> dd;
        e.grid.SampleJets(padded,e.nf,x,v,d,dd,true);
        const auto u=e.Jet(x,v,d,dd);
        q=h::PhysicalHorizonGeometry(u,e.Omega(x,u));
      } catch (const std::exception &) { q={}; }
      return true;
    });
    finder.Find(0,0);
    const auto r = finder.GetResult();
    std::ostringstream line;
    line << std::setprecision(17) << "n=" << n << " lmax=" << lmax
         << " ntheta=" << ntheta << " rms_tol=" << tolerance
         << " found=" << r.found << " complete=" << r.complete_surface
         << " iterations=" << r.iterations << " mass=" << r.mass
         << " areal_radius=" << r.areal_radius << " coordinate_radius=" << r.mean_radius
         << " expansion_rms=" << r.expansion_rms
         << " last_expansion_rms=" << r.last_expansion_rms
         << " last_mean_radius=" << r.last_mean_radius << '\n';
    if (global_variable::my_rank == 0) {
      std::ofstream result(output+".result.txt"); result << line.str();
      std::cout << line.str();
    }
    if (!r.found) status=2;
  } catch (const std::exception &ex) { std::cerr << ex.what() << '\n'; status=1; }
  Kokkos::finalize();
  #if MPI_PARALLEL_ENABLED
    MPI_Finalize();
  #endif
  return status;
}
