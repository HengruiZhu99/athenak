//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================

#ifndef UTILS_HORIZON_DUMP_HPP_
#define UTILS_HORIZON_DUMP_HPP_

#include <string>
#include <vector>

#include "athena.hpp"

class MeshBlockPack;
class ParameterInput;
class CartesianGrid;

//! \class HorizonDump
//! \brief Interpolate reconstructed ADM fields for the external AHFinderDirect adapter.
class HorizonDump {
 public:
  HorizonDump(MeshBlockPack *pmbp, ParameterInput *pin, int n, int common_horizon,
              const std::string &parameter_block = "z4c",
              bool gauge_from_adm = false);
  ~HorizonDump();

  void SetGridAndInterpolate(Real center[3]);
  void SetGridAndInterpolatePcGh(Real center[3], Real inner_radius);
  void ETK_setup_parfile();

  int horizon_nx;
  int common_horizon;
  int horizon_ind;
  int output_count;
  int regularize_order;

  Real horizon_dt;
  Real horizon_last_output_time;
  Real horizon_extent;
  CartesianGrid *pcat_grid = nullptr;
  Real pos[3];
  Real r_guess;
  bool is_cheb;

 private:
  MeshBlockPack const *pmbp;
  bool gauge_from_adm;
  std::vector<int> variable_to_dump;
  void WriteInterpolatedData(Real *data_out, int count, bool reduce);
};

#endif  // UTILS_HORIZON_DUMP_HPP_
