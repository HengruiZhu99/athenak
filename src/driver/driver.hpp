#ifndef DRIVER_DRIVER_HPP_
#define DRIVER_DRIVER_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file driver.hpp
//  \brief definitions for Driver class

#include <ctime>

#include "athena_arrays.hpp"
#include "parameter_input.hpp"
#include "outputs/outputs.hpp"

//----------------------------------------------------------------------------------------
//! \class Driver

class Driver {
 public:
  Driver(std::unique_ptr<ParameterInput> &pin, std::shared_ptr<Mesh> &pmesh,
         std::unique_ptr<Outputs> &pout);
  ~Driver() = default;

  // data
  bool time_evolution;

  // folowing data only relevant for runs involving time evolution
  Real tlim;      // stopping time
  int nlim;       // cycle-limit
  int ndiag;      // cycles between output of diagnostic information
  // variables for various SSP RK integrators
  std::string integrator;          // integrator name (rk1, rk2, rk3)
  int nstages;                     // total number of stages
  Real cfl_limit;                  // maximum CFL number for integrator
  Real gam0[3], gam1[3], beta[3];  // averaging weights and fractional timestep per stage

  // functions
  void Initialize(std::shared_ptr<Mesh> &pmesh, std::unique_ptr<Outputs> &pout);
  void Execute(std::shared_ptr<Mesh> &pmesh, std::unique_ptr<Outputs> &pout);
  void Finalize(std::shared_ptr<Mesh> &pmesh, std::unique_ptr<Outputs> &pout);

 private:
  clock_t tstart_, tstop_;  // variables to measure cpu time
  int nmb_updated_;         // running total of MB updated during run
  void OutputCycleDiagnostics(std::shared_ptr<Mesh> &pm);

};
#endif // DRIVER_DRIVER_HPP_
