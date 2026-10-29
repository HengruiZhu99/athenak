// READ-ONLY horizon diagnostic of an explicitly UNADMITTED failed snapshot.
// No evolution, RHS evaluation, projection, or standard-Z4c test is performed.
#include <filesystem>
#include <cstring>
#include <cstdint>
#include <limits>
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
struct Snapshot { char magic[16]; int n,nf,kind,admitted; uint64_t cells,surface,steps; Real time,dt; };
static_assert(sizeof(Snapshot)==72,"native snapshot header");

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
    std::ifstream in(initial,std::ios::binary); Snapshot snapshot{};
    in.read(reinterpret_cast<char*>(&snapshot),sizeof(snapshot));
    if (!in || std::memcmp(snapshot.magic,"HYP3D_MPI_V1",12) || snapshot.n!=n || snapshot.nf!=31
        || snapshot.kind!=2 || snapshot.admitted!=0 || snapshot.cells*31!=e.state.size()
        || snapshot.surface!=e.scri.values.size()) throw std::runtime_error("expected unadmitted puncture V1 snapshot");
    if (std::filesystem::file_size(initial)!=72+16*(e.state.size()+e.scri.values.size()))
      throw std::runtime_error("snapshot byte count mismatch");
    in.read(reinterpret_cast<char*>(e.state.data()),e.state.size()*sizeof(Real));
    in.seekg(e.state.size()*sizeof(Real),std::ios::cur);
    in.read(reinterpret_cast<char*>(e.scri.values.data()),e.scri.values.size()*sizeof(Real));
    if (!in) throw std::runtime_error("snapshot payload truncated");
    for (Real value:e.state) if (!std::isfinite(value)) throw std::runtime_error("nonfinite volume data");
    // Do not reclose the failed outer boundary or advance the state. Every
    // accepted sample below is guarded to use only Cartesian-core active cells.
    auto padded=e.ActivePadded(e.state); std::vector<bool> active(e.grid.count,false);
    for (int point:e.grid.active) active[point]=true;
    for (int f=0;f<31;++f) for (int point=0;point<e.grid.count;++point)
      if (!active[point]) padded[f*e.grid.count+point]=std::numeric_limits<Real>::quiet_NaN();
    int rejected_stencil=0; size_t samples=0;
    ParameterInput pin;
    std::ostringstream input;
    input << std::setprecision(17) << "<job>\nbasename=" << output
          << "\n<fastflow>\nntheta=" << ntheta << "\nlmax=" << lmax
          << "\nverbose=true\nflow=standard\nflow_iterations_0=" << iterations
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
        ++samples;
        if (e.grid.mesh.Owner(x)!=0) { ++rejected_stencil; q={}; return true; }
        Real local[3];e.grid.mesh.Inverse(0,x,local);
        for(int a=0;a<3;++a) {
          const int first=int(std::floor(n*local[a]-.5))-2;
          if(first<0 || first+6>n) { ++rejected_stencil; q={}; return true; }
        }
        h::HyperbolicComposite::V v;
        std::vector<std::array<Real,3>> d;
        std::vector<std::array<std::array<Real,3>,3>> dd;
        e.grid.SampleJets(padded,e.nf,x,v,d,dd,true);
        const auto u=e.Jet(x,v,d,dd);
        q=h::PhysicalHorizonGeometry(u,e.Omega(x,u));
      } catch (const std::exception &) { q={}; }
      return true;
    });
    finder.Find(snapshot.steps,snapshot.time);
    const auto r = finder.GetResult();
    std::ostringstream line;
    line << std::setprecision(17) << "n=" << n << " lmax=" << lmax
         << " ntheta=" << ntheta << " rms_tol=" << tolerance
         << " found=" << r.found << " complete=" << r.complete_surface
         << " iterations=" << r.iterations << " mass=" << r.mass
         << " areal_radius=" << r.areal_radius << " coordinate_radius=" << r.mean_radius
         << " expansion_rms=" << r.expansion_rms
         << " last_expansion_rms=" << r.last_expansion_rms
         << " last_mean_radius=" << r.last_mean_radius
         << " snapshot_time=" << snapshot.time << " snapshot_admitted=" << snapshot.admitted
         << " seed_radius=" << radius << " samples=" << samples
         << " rejected_core_stencils=" << rejected_stencil << " active_core_only=1" << '\n';
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
