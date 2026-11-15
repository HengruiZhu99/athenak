// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef OUTPUTS_SCRI_HPP_
#define OUTPUTS_SCRI_HPP_
#include <memory>
#include "outputs/outputs.hpp"

// Native accepted-state surface archive; no evolution-stage hooks.
class ScriOutput : public BaseTypeOutput {
 public:
  ScriOutput(ParameterInput *,Mesh *,OutputParameters);
  ~ScriOutput() override;
  void LoadOutputData(Mesh *) override;
  void WriteOutputFile(Mesh *,ParameterInput *) override;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
#endif
