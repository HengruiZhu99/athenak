#ifndef BOLTZMANN_BOLTZMANN_HPP_
#define BOLTZMANN_BOLTZMANN_HPP_
#include "boltzmann/geometry.hpp"
#include "bvals/bvals.hpp"
#include "tasklist/task_list.hpp"

struct HistoryData;
namespace boltzmann {
struct FaceGeometry {
  Real minus_shift, triad[3], killing_norm;
  KOKKOS_INLINE_FUNCTION Real Speed(const Real n[3]) const {
    return minus_shift + triad[0] * n[0] + triad[1] * n[1] + triad[2] * n[2];
  }
};
void RingHistory(HistoryData*, Mesh*);
class Boltzmann {
 public:
  Boltzmann(MeshBlockPack* pack, ParameterInput* pin);
  ~Boltzmann();
  void QueueTasks();
  void InitializeRing(ParameterInput* pin);
  void Synchronize(Driver* driver);
  TaskStatus GeometryAndStress(Driver*, int);
  TaskStatus Fluxes(Driver*, int);
  TaskStatus Update(Driver*, int);
  TaskStatus InitRecv(Driver*, int);
  TaskStatus SendFlux(Driver*, int);
  TaskStatus RecvFlux(Driver*, int);
  TaskStatus SendState(Driver*, int);
  TaskStatus RecvState(Driver*, int);
  TaskStatus Clear(Driver*, int);

  MeshBlockPack* pack;
  const int npsi, nmu_half;
  int nmu, nang;
  Real mu_min, dmu, dpsi, dtnew;
  bool backreaction, fixed_metric;
  DvceArray5D<Real> u, coarse, moments;
  // Per-ordinate angle averages of (1, n_a, n_a n_b), in Tmunu order.
  DvceArray2D<Real> weights;
  Real escaped = 0, escaped_j = 0, work = 0;
  void Ghosts();

 private:
  MeshBoundaryValuesCC* boundary;
  DvceArray5D<Real> accumulator, angular_mu, angular_psi, source;
  DvceFaceFld5D<Real> flux;
  DvceArray3D<Geometry> geometry;
  DvceArray4D<FaceGeometry> face_geometry;
  Real budget_old[3]{};
};
}  // namespace boltzmann
#endif  // BOLTZMANN_BOLTZMANN_HPP_
