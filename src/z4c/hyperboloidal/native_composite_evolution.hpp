// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_NATIVE_COMPOSITE_EVOLUTION_HPP_
#define Z4C_HYPERBOLOIDAL_NATIVE_COMPOSITE_EVOLUTION_HPP_
#include <memory>
#include <string>
#include "athena.hpp"
class MeshBlockPack;
class ParameterInput;
class Driver;
struct LogicalLocation;
namespace z4c {class Z4c; namespace hyperboloidal {
// Explicit experimental host owner. Native u0 is a verified physical mirror of
// canonical global coefficients; outer shells/scri are owned here, not native MBs.
class NativeCompositeEvolution {
 public:
  NativeCompositeEvolution(Z4c &z,MeshBlockPack *pack,ParameterInput *pin);
  ~NativeCompositeEvolution();
  void Initialize(ParameterInput *pin,bool restart);
  void BeginStage(Driver *driver,int stage);
  void EvaluateRHS(Driver *driver,int stage);
  void AdvanceStage(Driver *driver,int stage);
  void PrepareNative();
  void ToADM();
  void Constraints();
  Real TimeStep() const;
  void WriteRestart(const std::string &path);
  // Accepted-step canonical transaction; no native physical-field transfer is
  // authoritative. Prepare may throw before native Views change.
  void PrepareRegrid(const LogicalLocation *leaves,int count);
  void CommitRegrid();
  int ValidationRefinementFlag(int native)const; // sentinel2: normal criteria

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};
}} // namespace z4c::hyperboloidal
#endif
