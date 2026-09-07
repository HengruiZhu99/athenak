#include "z4c/frozen_mots.hpp"

#include <cstring>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <vector>

#include "driver/driver.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "z4c/cartoon_m0_fastflow.hpp"
#include "z4c/z4c.hpp"
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif
namespace z4c {
void RunFrozenMots(Mesh* mesh, Driver* driver, ParameterInput* pin) {
  auto* pack = mesh->pmb_pack;
  auto* z = pack->pz4c;
  if (!z || z->layout.centering != Z4cGridCentering::vertex ||
      pack->z4c_symmetry.mode != Z4cSymmetryMode::cartoon_so2 || pack->phydro ||
      pack->pmhd)
    throw std::runtime_error("horizon_only requires vacuum native VC Cartoon");
  const auto layout = z->layout;
  const Real time = mesh->time, dt = mesh->dt;
  const int cycle = mesh->ncycle, blocks = mesh->nmb_total;
  std::vector<LogicalLocation> topology(mesh->lloc_eachmb, mesh->lloc_eachmb + blocks);
  auto snapshot = [&]() {
    auto host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), z->u0);
    std::vector<Real> values;
    for (int m = 0; m < pack->nmb_thispack; ++m)
      for (int v = 0; v < Z4c::nz4c; ++v)
        for (int k = layout.ks; k <= layout.ke; ++k)
          for (int j = layout.js; j <= layout.je; ++j)
            for (int i = layout.is; i <= layout.ie; ++i)
              values.push_back(host(m, v, k, j, i));
    return values;
  };
  const auto before = snapshot();
  auto check = [&]() {
    auto after = snapshot();
    int changed =
        before.size() != after.size() ||
        std::memcmp(before.data(), after.data(), before.size() * sizeof(Real)) != 0 ||
        mesh->time != time || mesh->dt != dt || mesh->ncycle != cycle ||
        mesh->nmb_total != blocks;
    for (int n = 0; n < blocks; ++n) {
      const auto &a = topology[n], &b = mesh->lloc_eachmb[n];
      changed = changed || a.lx1 != b.lx1 || a.lx2 != b.lx2 || a.lx3 != b.lx3 ||
                a.level != b.level;
    }
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &changed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
    if (changed) throw std::runtime_error("frozen MOTS changed active state or mesh");
  };
  driver->InitBoundaryValuesAndPrimitives(mesh, true);
  check();
  z->Z4cToADM(pack);
  CartoonM0FastFlow finder(pack, pin, 0);
  finder.Find(cycle, time, true);
  check();
  finder.Write(cycle, time);
  if (global_variable::my_rank == 0) {
    std::ofstream out("frozen_mots.json");
    out << std::setprecision(17) << "{\"time\":" << time << ",\"cycle\":" << cycle
        << ",\"blocks\":" << blocks << ",\"active_state_unchanged\":true,"
        << "\"mesh_unchanged\":true,\"verified_candidate\":"
        << (finder.Found() ? "true" : "false") << ",\"spatially_validated\":false}\n";
    std::ofstream parameters("frozen_parameters.athinput");
    pin->ParameterDump(parameters);
  }
}
}  // namespace z4c
