#include "boltzmann/boltzmann.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "tasklist/numerical_relativity.hpp"

namespace boltzmann {
void Boltzmann::QueueTasks() {
  using namespace numrel;
  auto* nr = pack->pnr;
  nr->QueueTask(&Boltzmann::InitRecv, this, Boltz_Receive, "boltzmann receive", Task_Start);
  nr->QueueTask(&Boltzmann::GeometryAndStress, this, Boltz_Stress, "boltzmann stress",
                Task_Run);
  nr->QueueTask(&Boltzmann::Fluxes, this, Boltz_Flux, "boltzmann flux", Task_Run,
                {Boltz_Stress});
  nr->QueueTask(&Boltzmann::SendFlux, this, Boltz_SendFlux, "boltzmann send flux", Task_Run,
                {Boltz_Flux});
  nr->QueueTask(&Boltzmann::RecvFlux, this, Boltz_RecvFlux, "boltzmann receive flux", Task_Run,
                {Boltz_SendFlux});
  nr->QueueTask(&Boltzmann::Update, this, Boltz_Update, "boltzmann update", Task_Run,
                {Boltz_RecvFlux});
  nr->QueueTask(&Boltzmann::SendState, this, Boltz_Send, "boltzmann send", Task_Run,
                {Boltz_Update});
  nr->QueueTask(&Boltzmann::RecvState, this, Boltz_Boundary, "boltzmann boundaries", Task_Run,
                {Boltz_Send});
  nr->QueueTask(&Boltzmann::Clear, this, Boltz_Clear, "boltzmann clear", Task_End);
  nr->QueueTask(&Boltzmann::AcceptedStress, this, Boltz_FinalStress,
                "boltzmann accepted stress", Task_End, {Boltz_Clear});
}

TaskStatus Boltzmann::AcceptedStress(Driver* driver, int stage) {
  auto status = GeometryAndStress(driver, stage);
  // Store the CFL bound of the accepted metric, as on a checkpoint restart.
  // Its characteristic fields are reused by the next first-stage flux.
  if (transport_order == 4 && !fixed_metric && stage == driver->nexp_stages)
    return FluxesFourth(driver, 0);
  return status;
}

TaskStatus Boltzmann::InitRecv(Driver*, int stage) {
  auto status = boundary->InitRecv(nang);
  if (status == TaskStatus::complete && stage >= 0 && pack->pmesh->multilevel)
    status = boundary->InitFluxRecv(nang);
  return status;
}
TaskStatus Boltzmann::SendFlux(Driver*, int) {
  return pack->pmesh->multilevel ? boundary->PackAndSendFluxCC(flux) : TaskStatus::complete;
}
TaskStatus Boltzmann::RecvFlux(Driver*, int) {
  return pack->pmesh->multilevel ? boundary->RecvAndUnpackFluxCC(flux) : TaskStatus::complete;
}
TaskStatus Boltzmann::SendState(Driver*, int) {
  if (pack->pmesh->multilevel) pack->pmesh->pmr->RestrictCC(u, coarse);
  return boundary->PackAndSendCC(u, coarse);
}
TaskStatus Boltzmann::Clear(Driver*, int stage) {
  boundary->ClearSend();
  boundary->ClearRecv();
  if (stage >= 0 && pack->pmesh->multilevel) {
    boundary->ClearFluxSend();
    boundary->ClearFluxRecv();
  }
  return TaskStatus::complete;
}

void Boltzmann::Ghosts() {
  const auto flags = pack->pmb->mb_bcs.d_view;
  const auto b = pack->pmesh->mb_indcs;
  const int sectors = npsi;
  for (int level = 0; level < 1 + pack->pmesh->multilevel; ++level) {
    auto state = level == 0 ? u : coarse;
    int is = level == 0 ? b.is : b.cis, ie = level == 0 ? b.ie : b.cie;
    int js = level == 0 ? b.js : b.cjs, je = level == 0 ? b.je : b.cje;
    // Vacuum incoming radiation. Outgoing face reconstruction uses interior data.
    // Fill z first, then rotate through pi across the axis (including corners).
    par_for(
        "boltzmann axial boundaries", DevExeSpace(), 0, pack->nmb_thispack - 1, 0, nang - 1, 0,
        state.extent_int(3) - 1, 0, state.extent_int(4) - 1,
        KOKKOS_LAMBDA(int m, int n, int j, int i) {
          if ((j < js && flags(m, BoundaryFace::inner_x2) == BoundaryFlag::outflow) ||
              (j > je && flags(m, BoundaryFace::outer_x2) == BoundaryFlag::outflow) ||
              (i > ie && flags(m, BoundaryFace::outer_x1) == BoundaryFlag::outflow))
            state(m, n, 0, j, i) = 0;
        });
    par_for(
        "boltzmann axis continuation", DevExeSpace(), 0, pack->nmb_thispack - 1, 0, nang - 1, 0,
        state.extent_int(3) - 1, 0, is - 1, KOKKOS_LAMBDA(int m, int n, int j, int i) {
          if (flags(m, BoundaryFace::inner_x1) == BoundaryFlag::axis) {
            int other =
                n / sectors * sectors + (sectors + sectors / 2 - 1 - n % sectors) % sectors;
            state(m, n, 0, j, i) = state(m, other, 0, j, 2 * is - 1 - i);
          }
        });
  }
}
TaskStatus Boltzmann::RecvState(Driver*, int) {
  auto status = boundary->RecvAndUnpackCC(u, coarse);
  if (status != TaskStatus::complete) return status;
  Ghosts();
  if (pack->pmesh->multilevel) {
    boundary->FillCoarseInBndryCC(u, coarse);
    boundary->ProlongateCC(u, coarse);
    Ghosts();
  }
  return TaskStatus::complete;
}
void Boltzmann::Synchronize(Driver* driver) {
  speed_cache_valid = false;
  InitRecv(driver, -1);
  SendState(driver, -1);
  Clear(driver, -1);
  RecvState(driver, -1);
  GeometryAndStress(driver, 0);
  Fluxes(driver, 0);
}
}  // namespace boltzmann
