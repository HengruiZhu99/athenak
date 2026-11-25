// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_MULTIRATE_EVOLUTION_HPP_
#define Z4C_HYPERBOLOIDAL_MULTIRATE_EVOLUTION_HPP_
#include <memory>
class Driver;
class ParameterInput;
namespace z4c_hyperboloidal {
class Z4cHyperboloidal;
// Scheme A adapter. The numerical schedule and sparse record transport remain
// independent of this physics module. Only common endpoints are accepted states.
class MultirateEvolution {
 public:
  MultirateEvolution(Z4cHyperboloidal&, Driver*, bool rebuilt);
  static void ReadOptions(ParameterInput& input);
  ~MultirateEvolution();
  bool Poll(Driver*);

 private:
 public:  // nvcc requires public access for device-lambda types and functions
  struct Impl;
 private:
  std::unique_ptr<Impl> impl_;
};
}  // namespace z4c_hyperboloidal
#endif
