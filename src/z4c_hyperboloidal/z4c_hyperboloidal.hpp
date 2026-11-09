#ifndef Z4C_HYPERBOLOIDAL_Z4C_HYPERBOLOIDAL_HPP_
#define Z4C_HYPERBOLOIDAL_Z4C_HYPERBOLOIDAL_HPP_
//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================

#include <array>
#include <cstdint>
#include <memory>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "z4c_hyperboloidal/fields.hpp"
#include "z4c_hyperboloidal/spherical_reference_3d.hpp"
#include "z4c_hyperboloidal/rhs_subset.hpp"

class Driver;
class MeshBlockPack;
class MeshRefinement;

namespace z4c_hyperboloidal {
class MappedHalo;
class SurfaceExchange;
class DissipationPlan;
class RemeshState;
class Diagnostics;

// Local execution indices only. MeshBlockPack remains the ownership authority;
// these lists batch irregular evolved overlap regions into device kernels.
struct BlockPoint {int m, k, j, i;};
struct SurfacePoint {int face, k, j;};
struct EvolvedBounds {int lo[3], hi[3];};  // inclusive stored-array indices

struct ChiRecord {
  Real minimum=1e300, accepted_minimum=1e300;
  Real floor_count=0, nonpositive_count=0, nonfinite_count=0;
  Real max_correction=0, sum_correction=0, first_floor=1e300;
};

KOKKOS_INLINE_FUNCTION Real FloorChiCandidate(Real raw, Real floor, Real time,
                                              bool accepted, ChiRecord &record) {
  if (!Kokkos::isfinite(raw)) {record.nonfinite_count+=1; return raw;}
  record.minimum=fmin(record.minimum,raw);
  if (accepted) record.accepted_minimum=fmin(record.accepted_minimum,raw);
  if (raw<=0) record.nonpositive_count+=1;
  if (floor>0 && raw<floor) {
    record.floor_count+=1;
    record.first_floor=fmin(record.first_floor,time);
    record.max_correction=fmax(record.max_correction,floor-raw);
    record.sum_correction+=floor-raw;
    return floor;
  }
  return raw;
}

class Z4cHyperboloidal {
 public:
  Z4cHyperboloidal(MeshBlockPack *ppack, ParameterInput *pin);
  ~Z4cHyperboloidal();

  static constexpr int nvar=Fields::count;
  int n[3], storage[3], origin, overlap, nghost;
  int nblocks, nfaces, npoints, nsurface_points;
  Real ko, chi_floor, penalty, requested_dt, dtnew;
  HyperbolicShiftParameters<Real> parameters;

  // Only local native blocks are allocated. Nominal cells start at origin;
  // additional evolved overlap is separate from the four difference ghosts.
  DvceArray5D<Real> u0, u1, u_rhs, u_accum, coarse_u0, residual;
  DvceArray4D<Real> s0, s1, s_rhs, s_accum;
  DualArray1D<EvolvedBounds> evolved_bounds;
  DualArray1D<BlockPoint> points;
  DualArray1D<SurfacePoint> surface_points;
  DualArray1D<int> face_blocks, block_faces;
  DvceArray1D<mapped::Point> geometry, surface_geometry;
  DvceArray1D<CartesianChartJets> reference, surface_reference;
  DvceArray1D<StationaryGaugeSource<Real>> stationary_source;
  StationaryGaugeSource<Real> boundary_source;
  DvceArray1D<ChiRecord> chi_records, surface_chi_records;
  DvceArray1D<int> rhs_status;
  DvceArray1D<Real> rhs_gap, rhs_cfl;

  // State-changing entry points invalidate accepted-state diagnostics. Call this
  // explicitly after external seeding of the public state views.
  void InvalidateDiagnostics() {++state_version_;}
  std::uint64_t StateVersion() const {return state_version_;}
  Diagnostics &GetDiagnostics();
  std::array<Real,10> DiagnosticHistorySums();
  void InitializeReference();
  void InitializeBoundaryValues(Driver *driver);
  TaskStatus NewTimeStep(Driver *driver, int stage);
  Real StageMaxCompatibility() const;
  Real StageMaxClock() const;
  void PrepareRemesh(MeshRefinement &refinement);
  void FinishRemesh(MeshRefinement &refinement);
  DvceArray5D<Real> &RemeshNominal();
  DvceArray5D<Real> &RemeshCoarse();
  void FormResidual();
  void EnforceAlgebraicConstraints();
  TaskStatus CopyU(Driver *driver, int stage);
  TaskStatus ExpRKUpdate(Driver *driver, int stage);
  void BuildStagePlans();
  void QueueZ4cHyperboloidalTasks();
  TaskStatus InitRecv(Driver *driver, int stage);
  TaskStatus PrepareStage(Driver *driver, int stage);
  TaskStatus SendBulk(Driver *driver, int stage);
  TaskStatus FillLocalHalos(Driver *driver, int stage);
  TaskStatus RecvBulk(Driver *driver, int stage);
  TaskStatus FillRemoteHalos(Driver *driver, int stage);
  TaskStatus SendSurface(Driver *driver, int stage);
  TaskStatus RecvSurface(Driver *driver, int stage);
  TaskStatus SolveNormalCompatibility(Driver *driver, int stage);
  TaskStatus CalcSurfaceRHS(Driver *driver, int stage);
  TaskStatus ProjectSurface(Driver *driver, int stage);
  TaskStatus ClearSend(Driver *driver, int stage);
  TaskStatus ClearRecv(Driver *driver, int stage);
  TaskStatus CalcRHS(Driver *driver, int stage, RHSSubset subset);
  bool IsInitialized() const {return initialized_;}
  bool StagePlansReady() const {return plans_ready_;}
  const DvceArray1D<int> &RHSPoints(RHSSubset subset) const {
    return rhs_points_[static_cast<int>(subset)];
  }

 private:
  friend class Diagnostics;
  MeshBlockPack *pmy_pack;
  ParameterInput *pinput_;
  std::uint64_t state_version_=0;
  std::unique_ptr<Diagnostics> diagnostics_;
  bool initialized_=false;
  bool plans_ready_=false;
  std::unique_ptr<MappedHalo> mapped_halo_;
  std::unique_ptr<SurfaceExchange> surface_exchange_;
  std::unique_ptr<DissipationPlan> dissipation_plan_;
  std::unique_ptr<RemeshState> remesh_;
  std::array<DvceArray1D<int>,5> rhs_points_;
  void BuildLocalWorkLists();
  void AllocateStorage();
  void BuildReferenceCache();
  void ResetMeshStorage();
};
}  // namespace z4c_hyperboloidal
#endif  // Z4C_HYPERBOLOIDAL_Z4C_HYPERBOLOIDAL_HPP_
