//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_linear_wave.cpp
//! \brief z4c linear (gravitational) wave test


// C/C++ headers
#include <algorithm> // min, max
#include <cmath>     // sqrt()
#include <cstdio>    // fopen(), fprintf(), freopen()
#include <iostream>  // endl
#include <sstream>   // stringstream
#include <string>    // c_str()
#include <limits>

// Athena++ headers
#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "z4c/z4c.hpp"
#include "driver/driver.hpp"
#include "pgen/pgen.hpp"
#include "geodesic-grid/gauss_legendre.hpp"

// function to compute errors in solution at end of run
void GLInterpolationErrors(ParameterInput *pin, Mesh *pm);

namespace {
// global variable to control computation of initial conditions versus errors
bool set_initial_conditions = true;
} // end anonymous namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::Z4cLinearWave()
//! \brief Sets initial conditions for gw linear wave tests

void ProblemGenerator::GaussLegendreTest(ParameterInput *pin, const bool restart) {
  pgen_final_func = GLInterpolationErrors;
  if (restart)
    return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pz4c == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Z4c Wave test can only be run in Z4c, but no <z4c> block "
              << "in input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Prepare Initial Data

  // capture variables for the kernel
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &is = indcs.is;
  int &ie = indcs.ie;
  int &js = indcs.js;
  int &je = indcs.je;
  int &ks = indcs.ks;
  int &ke = indcs.ke;
  auto &pz4c = pmbp->pz4c;

  // Code below will automatically calculate wavevector along grid diagonal, imposing the
  // conditions of periodicity and exactly one wavelength along each grid direction
  Real x1size = pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min;
  Real x2size = pmy_mesh_->mesh_size.x2max - pmy_mesh_->mesh_size.x2min;
  Real x3size = pmy_mesh_->mesh_size.x3max - pmy_mesh_->mesh_size.x3min;

  // Wave amplitude
  Real amp = pin->GetOrAddReal("problem", "amp", 0.001);

  // compute solution in u1 register. For initial conditions, set u1 -> u0.
  auto &u1 = (set_initial_conditions)? pmbp->pz4c->u0 : pmbp->pz4c->u1;

  // Initialize wavevector
  Real kx1 = pin->GetOrAddReal("problem", "kx1", 1. / x1size);
  Real kx2 = pin->GetOrAddReal("problem", "kx2", 1. / x2size);
  Real kx3 = pin->GetOrAddReal("problem", "kx3", 1. / x3size);

  // Wavevector length
  Real knorm = sqrt(SQR(kx1) + SQR(kx2) + SQR(kx3));

  // Calculate angular offset of the wavevector from zhat
  Real theta = std::atan2(sqrt(kx2 * kx2 + kx1 * kx1), kx3);
  Real phi = std::atan2(kx1, kx2);

  // set new time limit in ParameterInput (to be read by Driver constructor) based on
  // wave speed of selected mode.
  // input tlim is interpreted asnumber of wave periods for evolution
  if (set_initial_conditions) {
    Real tlim = pin->GetReal("time", "tlim");
    pin->SetReal("time", "tlim", tlim*knorm);
  }

  // rotated weight for each tensor element
  Real axx, axy, axz, ayy, ayz, azz;
  axx = -SQR(cos(theta))*cos(2*phi)-SQR(cos(phi))*SQR(sin(theta));
  axy = -0.25*(3+cos(2*theta))*sin(2*phi);
  axz = -cos(theta)*sin(theta)*sin(phi);
  ayy = SQR(cos(theta))*cos(2*phi)-SQR(sin(theta))*SQR(sin(phi));
  ayz = cos(theta)*sin(theta)*cos(phi);
  azz = SQR(sin(theta));

  par_for("pgen_linwave1", DevExeSpace(), 0, (pmbp->nmb_thispack - 1),
      ks, ke, js, je, is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        Real &x2min = size.d_view(m).x2min;
        Real &x2max = size.d_view(m).x2max;
        Real &x3min = size.d_view(m).x3min;
        Real &x3max = size.d_view(m).x3max;

        int nx1 = indcs.nx1;
        int nx2 = indcs.nx2;
        int nx3 = indcs.nx3;

        Real x1v = CellCenterX(i - is, nx1, x1min, x1max);
        Real x2v = CellCenterX(j - js, nx2, x2min, x2max);
        Real x3v = CellCenterX(k - ks, nx3, x3min, x3max);
        Real sinkx = sin(2 * M_PI * (kx1 * x1v + kx2 * x2v + kx3 * x3v));
        Real coskx = knorm * M_PI * cos(2 * M_PI * (kx1 * x1v + kx2 * x2v + kx3 * x3v));

        u1(m,pz4c->I_Z4C_GXX,k,j,i) = 1 + axx * amp * sinkx;
        u1(m,pz4c->I_Z4C_GXY,k,j,i) = axy * amp * sinkx;
        u1(m,pz4c->I_Z4C_GXZ,k,j,i) = axz * amp * sinkx;
        u1(m,pz4c->I_Z4C_GYY,k,j,i) = 1 + ayy * amp * sinkx;
        u1(m,pz4c->I_Z4C_GYZ,k,j,i) = ayz * amp * sinkx;
        u1(m,pz4c->I_Z4C_GZZ,k,j,i) = 1 + azz * amp * sinkx;

        u1(m,pz4c->I_Z4C_AXX,k,j,i) = axx * amp * coskx;
        u1(m,pz4c->I_Z4C_AXY,k,j,i) = axy * amp * coskx;
        u1(m,pz4c->I_Z4C_AXZ,k,j,i) = axz * amp * coskx;
        u1(m,pz4c->I_Z4C_AYY,k,j,i) = ayy * amp * coskx;
        u1(m,pz4c->I_Z4C_AYZ,k,j,i) = ayz * amp * coskx;
        u1(m,pz4c->I_Z4C_AZZ,k,j,i) = azz * amp * coskx;

        u1(m,pz4c->I_Z4C_ALPHA,k,j,i) = 1;
        u1(m,pz4c->I_Z4C_CHI,k,j,i) = 1;
        u1(m,pz4c->I_Z4C_KHAT,k,j,i) = 0;
        u1(m,pz4c->I_Z4C_THETA,k,j,i) = 0;

        u1(m,pz4c->I_Z4C_GAMX,k,j,i) = 0;
        u1(m,pz4c->I_Z4C_GAMY,k,j,i) = 0;
        u1(m,pz4c->I_Z4C_GAMZ,k,j,i) = 0;
      });
  return;
}

void GLInterpolationErrors(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &z4c = pmbp->pz4c->z4c;
  // Code below will automatically calculate wavevector along grid diagonal, imposing the
  // conditions of periodicity and exactly one wavelength along each grid direction
  Real x1size = pm->mesh_size.x1max - pm->mesh_size.x1min;
  Real x2size = pm->mesh_size.x2max - pm->mesh_size.x2min;
  Real x3size = pm->mesh_size.x3max - pm->mesh_size.x3min;

  Real amp = pin->GetOrAddReal("problem", "amp", 0.001);
  Real kx1 = pin->GetOrAddReal("problem", "kx1", 1. / x1size);
  Real kx2 = pin->GetOrAddReal("problem", "kx2", 1. / x2size);
  Real kx3 = pin->GetOrAddReal("problem", "kx3", 1. / x3size);
  // Wavevector length
  Real knorm = sqrt(SQR(kx1) + SQR(kx2) + SQR(kx3));

  // Calculate angular offset of the wavevector from zhat
  Real theta = std::atan2(sqrt(kx2 * kx2 + kx1 * kx1), kx3);
  Real phi = std::atan2(kx1, kx2);

  // set new time limit in ParameterInput (to be read by Driver constructor) based on
  // wave speed of selected mode.
  // input tlim is interpreted asnumber of wave periods for evolution
  if (set_initial_conditions) {
    Real tlim = pin->GetReal("time", "tlim");
    pin->SetReal("time", "tlim", tlim*knorm);
  }
  int ntheta = pin->GetOrAddInteger("problem", "ntheta",20);
  Real rad = pin->GetOrAddInteger("problem", "radius",1);
  int lmax = pin->GetOrAddInteger("problem","lmax",8);
  int nfilt = (lmax+1)*(lmax+1);

  // Gauss Legendre Grid
  std::cout << "Starting Tests: Gauss-Legendre Grid" << std::endl;
  GaussLegendreGrid *S = new GaussLegendreGrid(pmbp, ntheta, rad, nfilt);
  int nangles = S->nangles;
  // First test integration
  DualArray1D<Real> ones;
  Kokkos::realloc(ones,nangles);
  for (int i = 0; i < nangles; ++i) {
    ones.h_view(i) = 1;
  }

  Real Integrate = S->Integrate(ones);
  std::cout << "Unit Sphere Area Error \t" << 
  abs(Integrate - 4 * M_PI * rad * rad)/Integrate << std::endl;

  // rotated weight for each tensor element
  Real axx, axy, axz, ayy, ayz, azz;
  axx = -SQR(cos(theta))*cos(2*phi)-SQR(cos(phi))*SQR(sin(theta));
  
  // Test Interpolation to Sphere
  auto g_dd_surf =  S->InterpolateToSphere(z4c.g_dd);

  int n = 8;

  Real x = S->cart_pos.h_view(n,0);
  Real y = S->cart_pos.h_view(n,1);
  Real z = S->cart_pos.h_view(n,2);
  Real sinkx = sin(2 * M_PI * (kx1 * x + kx2 * y + kx3 * z));
  Real true_value = 1 + axx * amp * sinkx;

  std::cout << "Interpolation Error \t" << g_dd_surf(0,0,n) << "\t" << true_value << "\t" << g_dd_surf(0,0,n) - true_value << std::endl;

  // Gauss Legendre Grid
  std::cout << "Starting Tests: Geodesic Grid" << std::endl;
  SphericalGrid *psph = new SphericalGrid(pmbp, 20, rad);
  Real area = 0;
  for (int n=0; n<psph->nangles; ++n) {
    area += psph->solid_angles.h_view(n);
  }
  std::cout << "Unit Sphere Area Error \t" << 
  abs(area - 4 * M_PI * rad * rad)/area << std::endl;

  return;
}
