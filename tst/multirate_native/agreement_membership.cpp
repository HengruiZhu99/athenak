// Stream the production agreement plan for a real mesh without allocating the evolution.
#include <iostream>
#include <set>
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "z4c_hyperboloidal/overlap_agreement.hpp"
int main(int argc, char** argv) {
#if MPI_PARALLEL_ENABLED
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &global_variable::my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &global_variable::nranks);
#endif
  Kokkos::initialize(argc, argv);
  int status = 0;
  try {
    if (argc != 2) throw std::invalid_argument("usage: agreement_membership input.athinput");
    ParameterInput pin;
    IOWrapper file;
    file.Open(argv[1], IOWrapper::FileMode::read);
    pin.LoadFromFile(file);
    file.Close();
    Mesh mesh(&pin);
    mesh.BuildTreeFromScratch(&pin);
    namespace hyp = z4c_hyperboloidal;
    const int origin = mesh.mb_indcs.ng + pin.GetInteger("mesh_atlas", "evolved_overlap");
    hyp::NativeDonorSelector selector(mesh, origin);
    hyp::AgreementParameters parameters;
    parameters.inner = mesh.atlas->SphericalInnerRadius();
    parameters.taper_width = pin.GetOrAddReal("z4c_hyperboloidal", "agreement_taper_width", .015);
    std::set<int> group;
    std::uint64_t rows = 0;
    hyp::VisitAgreementStencils(mesh, selector, parameters,
                                [&](const auto& stencil, const auto&, Real, Real, bool) {
                                  ++rows;
                                  for (const auto& d : stencil)
                                    if (d.weight != 0) group.insert(d.point.gid);
                                });
    std::map<int, int> cartesian, spherical;
    for (int g : group) {
      const auto& map = selector.Map(g);
      if (map.face < 0)
        ++cartesian[mesh.lloc_eachmb[g].level - mesh.root_level];
      else
        ++spherical[map.layer];
    }
    std::cout << "AGREEMENT_MEMBERSHIP blocks=" << mesh.nmb_total << " rows=" << rows
              << " participants=" << group.size() << "\n";
    for (const auto& item : cartesian)
      std::cout << "cartesian spatial_level=" << item.first << " count=" << item.second << "\n";
    for (const auto& item : spherical)
      std::cout << "spherical region=" << item.first << " count=" << item.second << "\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    status = 1;
  }
  Kokkos::finalize();
#if MPI_PARALLEL_ENABLED
  MPI_Finalize();
#endif
  return status;
}
