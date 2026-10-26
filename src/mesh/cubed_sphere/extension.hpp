#ifndef MESH_CUBED_SPHERE_EXTENSION_HPP_
#define MESH_CUBED_SPHERE_EXTENSION_HPP_

#include <memory>
#include <iosfwd>
#include "athena.hpp"
#include "mesh/cubed_sphere/map.hpp"
class Mesh;
class MeshBlockPack;
class ParameterInput;
class Driver;
namespace z4c {class Z4c;}

namespace cubed_sphere {
// Fixed six-patch exterior, coupled to the ordinary Cartesian AMR hierarchy.
// The shell is a separate pack: Cartesian octree topology is never used at seams.
class Extension {
 public:
  Extension(MeshBlockPack *bulk, ParameterInput *pin);
  ~Extension();
  void BeginStage(Driver *driver, int stage);
  void Update(Driver *driver, int stage);
  void Fill(Real time);
  void FillCoarse();
  void FinishStage(Driver *driver, int stage);
  Real TimeStep(Driver *driver);
  void Report(ParameterInput *pin);
  Mesh *Shell() const {return shell_.get();}
 private:
  struct Row {
    int tm,tk,tj,ti,sm,sk,sj,si;
    int source_shell;
    int count[3];
    Real w[3][6];
  };
  using Plan = DvceArray1D<Row>;
  MeshBlockPack *bulk_;
  std::unique_ptr<ParameterInput> pin_;
  std::unique_ptr<Mesh> shell_;
  Map map_;
  Plan bulk_plan_,coarse_plan_,shell_plan_;
  int mesh_sequence_ = -1;
  int plan_builds_ = 0;
  bool analytic_outer_ = false;
  bool scalar_probe_active_ = false;
  Real amplitude_ = 0, wave_number_ = 1, pulse_width_ = 0.35;
  int initial_data_ = 0;
  DvceArray5D<Real> stage_bulk_,stage_shell_;
  void RebuildPlans();
  void Apply(const Plan &plan, const DvceArray5D<Real> &target);
  void CheckState(MeshBlockPack *pack, int stage, Real time, const char *where);
  void BoundaryRHS();
  void ScalarProbe(Real tlim, Real cfl);
  void ConformalProbe(ParameterInput *pin, Real tlim);
  void ReportRegions(MeshBlockPack *pack, const char *region, std::ostream &out) const;
};
}  // namespace cubed_sphere
#endif
