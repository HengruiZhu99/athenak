//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file shu_osher.cpp
//  \brief Problem generator for Shu-Osher shocktube test, involving interaction of a
//   Mach 3 shock with a sine wave density distribution.
//
// REFERENCE: C.W. Shu & S. Osher, "Efficient implementation of essentially
//   non-oscillatory shock-capturing schemes, II", JCP, 83, 32 (1998)

// C++ headers
#include <cmath>  // sin()

// Athena++ headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "utils/grid_locations.hpp"
#include "pgen.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//  \brief Shu-Osher test problem generator

void ProblemGenerator::UserProblem(MeshBlockPack *pmbp, ParameterInput *pin)
{
  if (pmbp->phydro == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Shu-Osher test can only be run in Hydro, but no <hydro> block "
              << "in input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  // setup problem parameters
  Real dl = 3.857143;
  Real pl = 10.33333;
  Real ul = 2.629369;
  Real vl = 0.0;
  Real wl = 0.0;

  // capture variables for kernel
  Real gm1 = pmbp->phydro->peos->eos_data.gamma - 1.0;
  int &is = pmbp->mb_cells.is, &ie = pmbp->mb_cells.ie;
  int &js = pmbp->mb_cells.js, &je = pmbp->mb_cells.je;
  int &ks = pmbp->mb_cells.ks, &ke = pmbp->mb_cells.ke;
  int &nx1 = pmbp->mb_cells.nx1;
  auto &u0 = pmbp->phydro->u0;
  auto &size = pmbp->pmb->mbsize;

  par_for("pgen_shock1", DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m,int k, int j, int i)
    {
      Real x1v = CellCenterX(i-is, nx1, size.x1min.d_view(m), size.x1max.d_view(m));

      if (x1v < -0.8) {
        u0(m,IDN,k,j,i) = dl;
        u0(m,IM1,k,j,i) = ul*dl;
        u0(m,IM2,k,j,i) = vl*dl;
        u0(m,IM3,k,j,i) = wl*dl;
        u0(m,IEN,k,j,i) = pl/gm1 + 0.5*dl*(ul*ul + vl*vl + wl*wl);
      } else {
        u0(m,IDN,k,j,i) = 1.0 + 0.2*std::sin(5.0*M_PI*(x1v));
        u0(m,IM1,k,j,i) = 0.0;
        u0(m,IM2,k,j,i) = 0.0;
        u0(m,IM3,k,j,i) = 0.0;
        u0(m,IEN,k,j,i) = 1.0/gm1;
      }
    }
  );

  return;
}