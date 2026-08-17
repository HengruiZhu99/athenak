//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file adm_z4c.cpp
//! \brief implementation of functions in the Z4c class related to ADM decomposition

// C standard headers
#include <math.h> // pow

// C++ standard headers
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <type_traits>

// Athena++ headers
#include "parameter_input.hpp"
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "z4c/z4c.hpp"
#include "z4c/stored_domain_bounds.hpp"
#include "z4c/cartoon_derivatives.hpp"
#include "z4c/z4c_symmetry.hpp"
#include "z4c/tmunu.hpp"
#include "coordinates/cell_locations.hpp"

namespace z4c {

//! \fn void Z4c::ADMToZ4c(MeshBlockPack *pmbp, ParameterInput *pin)
//! \brief Compute Z4c variables from ADM variables
//
// p  = detgbar^(-1/3)
// p0 = psi^(-4)
//
// gtilde_ij = p gbar_ij
// Ktilde_ij = p p0 K_ij
//
// phi = - log(p) / 4
// K   = gtildeinv^ij Ktilde_ij
// Atilde_ij = Ktilde_ij - gtilde_ij K / 3
//
// G^i = - del_j gtildeinv^ji
//
// BAM: Z4c_init()
// https://git.tpi.uni-jena.de/bamdev/z4
// https://git.tpi.uni-jena.de/bamdev/z4/blob/master/z4_init.m
//
// The Z4c variables will be set on the whole MeshBlock with the exception of
// the Gamma's that can only be set in the interior of the MeshBlock.
template <typename Symmetry, int FD_STENCIL>
void ADMToZ4cImpl(MeshBlockPack *pmbp, ParameterInput *pin) {
  // capture variables for the kernel
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  const auto bounds = MakeStoredDomainBounds(indcs);
  int nmb = pmbp->nmb_thispack;

  auto &z4c = pmbp->pz4c->z4c;
  auto &adm = pmbp->padm->adm;
  auto &opt = pmbp->pz4c->opt;
  // 2 1D scratch array and 1 2D scratch array
  par_for("initialize z4c fields",DevExeSpace(),
  0,nmb-1,bounds.ks,bounds.ke,bounds.js,bounds.je,bounds.is,bounds.ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> Kt_dd;
    Real detg = adm::SpatialDet(adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i),
                                adm.g_dd(m,0,2,k,j,i), adm.g_dd(m,1,1,k,j,i),
                                adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i));
    Real oopsi4 = pow(detg, -1./3.);
    z4c.chi(m,k,j,i) = pow(detg, 1./12.*opt.chi_psi_power);

    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b) {
      z4c.g_dd(m,a,b,k,j,i) = oopsi4 * adm.g_dd(m,a,b,k,j,i);
      Kt_dd(a,b)            = oopsi4 * adm.vK_dd(m,a,b,k,j,i);
    }

    detg = adm::SpatialDet(z4c.g_dd(m,0,0,k,j,i), z4c.g_dd(m,0,1,k,j,i),
                           z4c.g_dd(m,0,2,k,j,i), z4c.g_dd(m,1,1,k,j,i),
                           z4c.g_dd(m,1,2,k,j,i), z4c.g_dd(m,2,2,k,j,i));
    z4c.vKhat(m,k,j,i) = adm::Trace(1.0/detg,
                              z4c.g_dd(m,0,0,k,j,i), z4c.g_dd(m,0,1,k,j,i),
                              z4c.g_dd(m,0,2,k,j,i), z4c.g_dd(m,1,1,k,j,i),
                              z4c.g_dd(m,1,2,k,j,i), z4c.g_dd(m,2,2,k,j,i),
                              Kt_dd(0,0), Kt_dd(0,1), Kt_dd(0,2),
                              Kt_dd(1,1), Kt_dd(1,2), Kt_dd(2,2));

    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b) {
      z4c.vA_dd(m,a,b,k,j,i) = Kt_dd(a,b) - (1./3.) *
                                z4c.vKhat(m,k,j,i) * z4c.g_dd(m,a,b,k,j,i);
    }
  });
  Kokkos::fence();

  DvceArray5D<Real> g_uu("g_uu", nmb, 6, bounds.n3, bounds.n2, bounds.n1);
  AthenaTensor<Real, TensorSymm::SYM2, 3, 2> g3u;
  g3u.InitWithShallowSlice(g_uu, 0, 5);
  // GLOOP
  par_for("invert z4c metric",DevExeSpace(),
  0,nmb-1,bounds.ks,bounds.ke,bounds.js,bounds.je,bounds.is,bounds.ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i){
    Real detg = adm::SpatialDet(z4c.g_dd(m,0,0,k,j,i), z4c.g_dd(m,0,1,k,j,i),
                                z4c.g_dd(m,0,2,k,j,i), z4c.g_dd(m,1,1,k,j,i),
                                z4c.g_dd(m,1,2,k,j,i), z4c.g_dd(m,2,2,k,j,i));
    adm::SpatialInv(1.0/detg,
              z4c.g_dd(m,0,0,k,j,i), z4c.g_dd(m,0,1,k,j,i), z4c.g_dd(m,0,2,k,j,i),
              z4c.g_dd(m,1,1,k,j,i), z4c.g_dd(m,1,2,k,j,i), z4c.g_dd(m,2,2,k,j,i),
              &g3u(m,0,0,k,j,i), &g3u(m,0,1,k,j,i), &g3u(m,0,2,k,j,i),
              &g3u(m,1,1,k,j,i), &g3u(m,1,2,k,j,i), &g3u(m,2,2,k,j,i));
  });
  Kokkos::fence();

  // Compute Gammas
  // Compute only for internal points
  // ILOOP
  /*int const &IZ4CGAMX = pmbp->pz4c->I_Z4C_GAMX;
  int const &IZ4CGAMY = pmbp->pz4c->I_Z4C_GAMY;
  int const &IZ4CGAMZ = pmbp->pz4c->I_Z4C_GAMZ;
  auto              &u0 = pmbp->pz4c->u0;
  sub_DvceArray5D_0D g_00 = Kokkos::subview(g_uu, Kokkos::ALL, 0,
                            Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  sub_DvceArray5D_0D g_01 = Kokkos::subview(g_uu, Kokkos::ALL, 1,
                            Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  sub_DvceArray5D_0D g_02 = Kokkos::subview(g_uu, Kokkos::ALL, 2,
                            Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  sub_DvceArray5D_0D g_11 = Kokkos::subview(g_uu, Kokkos::ALL, 3,
                            Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  sub_DvceArray5D_0D g_12 = Kokkos::subview(g_uu, Kokkos::ALL, 4,
                            Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  sub_DvceArray5D_0D g_22 = Kokkos::subview(g_uu, Kokkos::ALL, 5,
                            Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);*/
  par_for("initialize Gamma",DevExeSpace(),0,nmb-1,indcs.ks,indcs.ke,
          indcs.js,indcs.je,indcs.is,indcs.ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // Usage of Dx: pmbp->pz4c->Dx(blockn, posvar, k,j,i, dir, nghost, dx, quantity);
    Real idx[] = {1/size.d_view(m).dx1, 1/size.d_view(m).dx2, 1/size.d_view(m).dx3};
    /*AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> g_uu;
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> Gamma_udd;
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> dg_ddd;
    Real detg = adm::SpatialDet(z4c.g_dd(m,0,0,k,j,i), z4c.g_dd(m,0,1,k,j,i),
                                z4c.g_dd(m,0,2,k,j,i), z4c.g_dd(m,1,1,k,j,i),
                                z4c.g_dd(m,1,2,k,j,i), z4c.g_dd(m,2,2,k,j,i));
    adm::SpatialInv(1.0/detg,
              z4c.g_dd(m,0,0,k,j,i), z4c.g_dd(m,0,1,k,j,i), z4c.g_dd(m,0,2,k,j,i),
              z4c.g_dd(m,1,1,k,j,i), z4c.g_dd(m,1,2,k,j,i), z4c.g_dd(m,2,2,k,j,i),
              &g_uu(0,0), &g_uu(0,1), &g_uu(0,2),
              &g_uu(1,1), &g_uu(1,2), &g_uu(2,2));
    for (int a = 0; a < 3; ++a)
    for (int b = 0; b < a; ++b)
    for (int c = 0; c < 3; ++c) {
      dg_ddd(c,a,b) = Dx<NGHOST>(c, idx, z4c.g_dd, m, a, b, k, j, i);
    }*/
    /*u0(m,IZ4CGAMX,k,j,i) = -Dx<NGHOST>(0, idx, g_00, m, k, j, i)  // d/dx g00
                           -Dx<NGHOST>(1, idx, g_01, m, k, j, i)  // d/dy g01
                           -Dx<NGHOST>(2, idx, g_02, m, k, j, i); // d/dz g02
    u0(m,IZ4CGAMY,k,j,i) = -Dx<NGHOST>(0, idx, g_01, m, k, j, i)  // d/dx g01
                           -Dx<NGHOST>(1, idx, g_11, m, k, j, i)  // d/dy g11
                           -Dx<NGHOST>(2, idx, g_12, m, k, j, i); // d/dz g12
    u0(m,IZ4CGAMZ,k,j,i) = -Dx<NGHOST>(0, idx, g_02, m, k, j, i)  // d/dx g01
                           -Dx<NGHOST>(1, idx, g_12, m, k, j, i)  // d/dy g11
                           -Dx<NGHOST>(2, idx, g_22, m, k, j, i); // d/dz g12*/
    /*for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
    for (int c = 0; c < b; ++c) {
      Gamma_udd(a, b, c) = 0.0;
      for (int d = 0; d < 3; ++d) {
        Gamma_udd(a, b, c) += 0.5*g_uu(a, d)*
          (-dg_ddd(d, b, c) + dg_ddd(b, d, c) + dg_ddd(c, b, d));
      }
    }
    for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
    for (int c = 0; c < 3; ++c) {
      z4c.vGam_u(m, a, k, j, i) += g_uu(b, c)*Gamma_udd(a, b, c);
    }*/
    // Keep the compile-time symmetry branch in the named device helper.  The
    // Kokkos lambda references z4c and g3u unconditionally, avoiding nvcc's
    // extended-lambda first-capture restriction for if-constexpr bodies.
    auto derivatives = MakeCellCenteredDerivativeProvider<Symmetry, FD_STENCIL>(
        idx, size.d_view, indcs.nx1, indcs.is, m, k, j, i);
    for (int a = 0; a < 3; ++a) {
      z4c.vGam_u(m, a, k, j, i) = 0.0;
      for (int b = 0; b < 3; ++b) {
        z4c.vGam_u(m, a, k, j, i) -=
            derivatives.template TensorFirst<TensorVariance::all_upper>(
                b, b, a, g3u);
      }
    }
  });
  pmbp->pz4c->AlgConstr(pmbp);
  return;
}

template <int NGHOST>
void Z4c::ADMToZ4c(MeshBlockPack *pmbp, ParameterInput *pin) {
  if (pmbp->pz4c->opt.fd_stencil != NGHOST) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Z4c ADM conversion dispatch mismatch: requested "
              << pmbp->pz4c->opt.fd_stencil << " but called " << NGHOST << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmbp->z4c_symmetry.mode == Z4cSymmetryMode::cartoon_so2) {
    ADMToZ4cImpl<CartoonSO2, NGHOST>(pmbp, pin);
  } else {
    ADMToZ4cImpl<Cartesian3D, NGHOST>(pmbp, pin);
  }
}
template void Z4c::ADMToZ4c<2>(MeshBlockPack *pmbp, ParameterInput *pin);
template void Z4c::ADMToZ4c<3>(MeshBlockPack *pmbp, ParameterInput *pin);
template void Z4c::ADMToZ4c<4>(MeshBlockPack *pmbp, ParameterInput *pin);
//----------------------------------------------------------------------------------------
//! \fn void Z4c::Z4cToADM(MeshBlockPack *pmbp)
//! \brief Compute ADM Psi4, g_ij, and K_ij from Z4c variables
//
// This sets the ADM variables everywhere in the MeshBlock
void Z4cToADMViews(MeshBlockPack *pmbp, const Z4c::Z4c_vars z4c,
                   const adm::ADM::ADM_vars adm_fields,
                   const Real chi_psi_power) {
  // capture variables for the kernel
  auto &indcs = pmbp->pmesh->mb_indcs;
  const auto bounds = MakeStoredDomainBounds(indcs);

  int nmb = pmbp->nmb_thispack;

  auto adm = adm_fields;
  par_for("initialize z4c fields",DevExeSpace(),
  0,nmb-1,bounds.ks,bounds.ke,bounds.js,bounds.je,bounds.is,bounds.ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    adm.psi4(m,k,j,i) = pow(z4c.chi(m,k,j,i), 4./chi_psi_power);

    // g_ab
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b) {
      adm.g_dd(m,a,b,k,j,i) = adm.psi4(m,k,j,i) * z4c.g_dd(m,a,b,k,j,i);
    }

    // K_ab
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b) {
      adm.vK_dd(m,a,b,k,j,i) = adm.psi4(m,k,j,i) * z4c.vA_dd(m,a,b,k,j,i) +
        (1./3.) * (z4c.vKhat(m,k,j,i) + 2.*z4c.vTheta(m,k,j,i)) * adm.g_dd(m,a,b,k,j,i);
    }
  });
  return;
}

void Z4c::Z4cToADM(MeshBlockPack *pmbp) {
  Z4cToADMViews(pmbp, pmbp->pz4c->z4c, pmbp->padm->adm,
                pmbp->pz4c->opt.chi_psi_power);
}
//----------------------------------------------------------------------------------------
//! \fn void Z4c::ADMConstraints(AthenaArray<Real> & u_adm, AthenaArray<Real> & u_mat)
//! \brief compute constraints ADM vars
//
// Note: we are assuming that u_adm has been initialized with the correct
// metric and matter quantities
//
// BAM: adm_constraints_N()
// https://git.tpi.uni-jena.de/bamdev/adm
// https://git.tpi.uni-jena.de/bamdev/adm/blob/master/adm_constraints_N.m
//
// The constraints are set only in the MeshBlock interior, because derivatives
// of the ADM quantities are needed to compute them.
template <typename Symmetry, int FD_STENCIL>
void ADMConstraintsViewsImpl(MeshBlockPack *pmbp, const Z4c::Z4c_vars z4c,
                             const adm::ADM::ADM_vars adm_fields,
                             DvceArray5D<Real> u_con,
                             const Z4c::Constraint_vars con,
                             const bool is_vacuum,
                             const Tmunu::Tmunu_vars tmunu) {
  // capture variables for the kernel
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  //For GLOOPS

  int nmb = pmbp->nmb_thispack;

  auto adm = adm_fields;

  Kokkos::deep_copy(u_con, 0.);
  par_for("ADM Hamiltonian constraint loop",DevExeSpace(),
  0,nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> g_uu;
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> R_dd;
    AthenaPointTensor<Real, TensorSymm::NONE, 3, 2> K_ud;

    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> dg_ddd;
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> Gamma_ddd;
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> Gamma_udd;

    AthenaPointTensor<Real, TensorSymm::SYM22, 3, 4> ddg_dddd;

    Real idx[] = {1/size.d_view(m).dx1, 1/size.d_view(m).dx2, 1/size.d_view(m).dx3};
    auto derivatives = MakeCellCenteredDerivativeProvider<Symmetry, FD_STENCIL>(
        idx, size.d_view, indcs.nx1, is, m, k, j, i);

    // -----------------------------------------------------------------------------------
    // derivatives
    //
    // first derivatives of g and K
    for(int c = 0; c < 3; ++c)
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b) {
      dg_ddd(c,a,b) =
          derivatives.template TensorFirst<TensorVariance::all_lower>(
              c, a, b, adm.g_dd);
    }

    // second derivatives of g
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b)
    for(int c = 0; c < 3; ++c)
    for(int d = c; d < 3; ++d) {
      if(a == b) {
        ddg_dddd(a,a,c,d) =
            derivatives.template TensorSecond<TensorVariance::all_lower>(
                a, a, c, d, adm.g_dd);
      } else {
        ddg_dddd(a,b,c,d) =
            derivatives.template TensorSecond<TensorVariance::all_lower>(
                a, b, c, d, adm.g_dd);
      }
    }

    // -----------------------------------------------------------------------------------
    // inverse metric
    //
    Real detg = adm::SpatialDet(adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i),
                                adm.g_dd(m,0,2,k,j,i), adm.g_dd(m,1,1,k,j,i),
                                adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i));
    adm::SpatialInv(1./detg,
               adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i), adm.g_dd(m,0,2,k,j,i),
               adm.g_dd(m,1,1,k,j,i), adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i),
               &g_uu(0,0), &g_uu(0,1), &g_uu(0,2),
               &g_uu(1,1), &g_uu(1,2), &g_uu(2,2));

    // -----------------------------------------------------------------------------------
    // Christoffel symbols
    //
    for(int c = 0; c < 3; ++c)
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b) {
      Gamma_ddd(c,a,b) = 0.5*(dg_ddd(a,b,c) + dg_ddd(b,a,c) - dg_ddd(c,a,b));
      Gamma_udd(c,a,b) = 0.0;
    }

    for(int c = 0; c < 3; ++c)
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b)
    for(int d = 0; d < 3; ++d) {
      Gamma_udd(c,a,b) += g_uu(c,d)*Gamma_ddd(d,a,b);
    }

    // -----------------------------------------------------------------------------------
    // Ricci tensor and Ricci scalar
    //
    Real R = 0.0;
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b) {
      R_dd(a,b) = 0.0;
      for(int c = 0; c < 3; ++c)
      for(int d = 0; d < 3; ++d) {
        // Part with the Christoffel symbols
        for(int e = 0; e < 3; ++e) {
          R_dd(a,b) += g_uu(c,d) * Gamma_udd(e,a,c) * Gamma_ddd(e,b,d);
          R_dd(a,b) -= g_uu(c,d) * Gamma_udd(e,a,b) * Gamma_ddd(e,c,d);
        }
        // Wave operator part of the Ricci
        R_dd(a,b) += 0.5*g_uu(c,d)*(
            - ddg_dddd(c,d,a,b) - ddg_dddd(a,b,c,d) +
              ddg_dddd(a,c,b,d) + ddg_dddd(b,c,a,d));
      }
    }

    for(int a = 0; a < 3; ++a)
    for(int b = 0; b < 3; ++b) {
      R += g_uu(a,b) * R_dd(a,b);
    }

    // -----------------------------------------------------------------------------------
    // Extrinsic curvature: traces and derivatives
    //
    Real K = 0.0;
    for(int a = 0; a < 3; ++a) {
      for(int b = 0; b < 3; ++b) {
        K_ud(a,b) = 0.0;
        for(int c = 0; c < 3; ++c) {
          K_ud(a,b) += g_uu(a,c) * adm.vK_dd(m,c,b,k,j,i);
        }
      }
      K += K_ud(a,a);
    }

    // K^a_b K^b_a
    Real KK = 0.0;
    for(int a = 0; a < 3; ++a)
    for(int b = 0; b < 3; ++b) {
      KK += K_ud(a,b) * K_ud(b,a);
    }

    // -----------------------------------------------------------------------------------
    // Actual constraints
    //
    // Hamiltonian constraint
    //
    con.H(m,k,j,i) = R + SQR(K) - KK;
    if(!is_vacuum) {
      con.H(m,k,j,i) -= 16*M_PI * tmunu.E(m,k,j,i);
    }
  });

  par_for("ADM momentum constraint loop",DevExeSpace(),
  0,nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> Gamma_u;
    AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> Gamma_u_z4c;
    AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> M_u;
    AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> dpsi4_d;

    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> g_uu;

    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> dg_ddd;
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> dK_ddd;
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> Gamma_ddd;
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> Gamma_udd;
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> DK_ddd;
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> DK_udd;

    Real idx[] = {1/size.d_view(m).dx1, 1/size.d_view(m).dx2, 1/size.d_view(m).dx3};
    auto derivatives = MakeCellCenteredDerivativeProvider<Symmetry, FD_STENCIL>(
        idx, size.d_view, indcs.nx1, is, m, k, j, i);

    // -----------------------------------------------------------------------------------
    // derivatives
    //
    // first derivatives of g and K
    for(int c = 0; c < 3; ++c)
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b) {
      dg_ddd(c,a,b) =
          derivatives.template TensorFirst<TensorVariance::all_lower>(
              c, a, b, adm.g_dd);
      dK_ddd(c,a,b) =
          derivatives.template TensorFirst<TensorVariance::all_lower>(
              c, a, b, adm.vK_dd);
    }

    // first derivative of psi4
    for (int a =0; a < 3; ++a) {
      dpsi4_d(a) = derivatives.ScalarFirst(a, adm.psi4);
    }

    // -----------------------------------------------------------------------------------
    // inverse metric
    //
    Real detg = adm::SpatialDet(adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i),
                                adm.g_dd(m,0,2,k,j,i), adm.g_dd(m,1,1,k,j,i),
                                adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i));
    adm::SpatialInv(1./detg,
               adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i), adm.g_dd(m,0,2,k,j,i),
               adm.g_dd(m,1,1,k,j,i), adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i),
               &g_uu(0,0), &g_uu(0,1), &g_uu(0,2),
               &g_uu(1,1), &g_uu(1,2), &g_uu(2,2));

    // -----------------------------------------------------------------------------------
    // Christoffel symbols
    //
    for(int c = 0; c < 3; ++c)
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b) {
      Gamma_ddd(c,a,b) = 0.5*(dg_ddd(a,b,c) + dg_ddd(b,a,c) - dg_ddd(c,a,b));
      Gamma_udd(c,a,b) = 0.0;
    }

    for(int c = 0; c < 3; ++c)
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b)
    for(int d = 0; d < 3; ++d) {
      Gamma_udd(c,a,b) += g_uu(c,d)*Gamma_ddd(d,a,b);
    }

    for(int a = 0; a < 3; ++a) {
      Gamma_u(a) = 0.0;
      for(int b = 0; b < 3; ++b)
      for(int c = 0; c < 3; ++c) {
        Gamma_u(a) += g_uu(b,c)*Gamma_udd(a,b,c);
      }
    }

    // Find the contracted conformal Christoffel symbol
    for (int a = 0; a < 3; ++a) {
      Gamma_u_z4c(a) = adm.psi4(m,k,j,i)*Gamma_u(a);
      for (int b = 0; b < 3; ++b) {
        Gamma_u_z4c(a) += 0.5*g_uu(a,b)*dpsi4_d(b);
      }
    }

    // -----------------------------------------------------------------------------------
    // Extrinsic curvature derivatives
    //
    // Covariant derivative of K
    for(int a = 0; a < 3; ++a)
    for(int b = 0; b < 3; ++b)
    for(int c = b; c < 3; ++c) {
      DK_ddd(a,b,c) = dK_ddd(a,b,c);
      for(int d = 0; d < 3; ++d) {
        DK_ddd(a,b,c) -= Gamma_udd(d,a,b) * adm.vK_dd(m,d,c,k,j,i);
        DK_ddd(a,b,c) -= Gamma_udd(d,a,c) * adm.vK_dd(m,b,d,k,j,i);
      }
    }

    for(int a = 0; a < 3; ++a)
    for(int b = 0; b < 3; ++b)
    for(int c = b; c < 3; ++c) {
      DK_udd(a,b,c) = 0.0;
      for(int d = 0; d < 3; ++d) {
        DK_udd(a,b,c) += g_uu(a,d) * DK_ddd(d,b,c);
      }
    }

    // -----------------------------------------------------------------------------------
    // Actual constraints
    //
    // Momentum constraint (contravariant)
    //
    for(int a = 0; a < 3; ++a) {
      M_u(a) = 0.0;
      for(int b = 0; b < 3; ++b) {
        if(!is_vacuum) {
          M_u(a) -= 8*M_PI * g_uu(a,b) * tmunu.S_d(m,b,k,j,i);
        }
        for(int c = 0; c < 3; ++c) {
          M_u(a) += g_uu(a,b) * DK_udd(c,b,c);
          M_u(a) -= g_uu(b,c) * DK_udd(a,b,c);
        }
      }
    }

    // Momentum constraint (covariant)
    for(int a = 0; a < 3; ++a) {
      for(int b = 0; b < 3; ++b) {
        con.M_d(m,a,k,j,i) += adm.g_dd(m,a,b,k,j,i) * M_u(b);
      }
    }

    // Momentum constraint (norm squared)
    for(int a = 0; a < 3; ++a)
    for(int b = 0; b < 3; ++b) {
      con.M(m,k,j,i) += adm.g_dd(m,a,b,k,j,i) * M_u(a) * M_u(b);
    }

    // Constraint violation Z (norm squared)
    for(int a = 0; a < 3; ++a)
    for(int b = 0; b < 3; ++b) {
      con.Z(m,k,j,i) += 0.25*z4c.g_dd(m,a,b,k,j,i)
                        *(z4c.vGam_u(m,a,k,j,i) - Gamma_u_z4c(a))
                        *(z4c.vGam_u(m,b,k,j,i) - Gamma_u_z4c(b));
    }

    // Constraint violation monitor C^2
    con.C(m,k,j,i) = SQR(con.H(m,k,j,i)) + con.M(m,k,j,i) +
                     SQR(z4c.vTheta(m,k,j,i)) + 4.0*con.Z(m,k,j,i);
});
}

template <typename Symmetry, int FD_STENCIL>
void ADMConstraintsImpl(MeshBlockPack *pmbp) {
  const bool is_vacuum = pmbp->ptmunu == nullptr;
  Tmunu::Tmunu_vars tmunu;
  if (!is_vacuum) tmunu = pmbp->ptmunu->tmunu;
  ADMConstraintsViewsImpl<Symmetry, FD_STENCIL>(
      pmbp, pmbp->pz4c->z4c, pmbp->padm->adm, pmbp->pz4c->u_con,
      pmbp->pz4c->con, is_vacuum, tmunu);
}

template <int NGHOST>
void Z4c::ADMConstraints(MeshBlockPack *pmbp) {
  if (pmbp->pz4c->opt.fd_stencil != NGHOST) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Z4c constraint dispatch mismatch: requested "
              << pmbp->pz4c->opt.fd_stencil << " but called " << NGHOST << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmbp->z4c_symmetry.mode == Z4cSymmetryMode::cartoon_so2) {
    ADMConstraintsImpl<CartoonSO2, NGHOST>(pmbp);
  } else {
    ADMConstraintsImpl<Cartesian3D, NGHOST>(pmbp);
  }
  pmbp->pz4c->ReconstructConstraintAxisParityGhosts();
}

void Z4c::EvaluateDiagnosticConstraints(
    DvceArray5D<Real> &scratch_adm,
    DvceArray5D<Real> &scratch_constraints,
    const int diagnostic_stencil) {
  if (pmy_pack->ptmunu != nullptr) {
    std::cerr << "### FATAL ERROR in " << __FILE__
              << ": AMR jump scratch constraints are restricted to vacuum Z4c"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const auto bounds = MakeStoredDomainBounds(pmy_pack->pmesh->mb_indcs);
  const int stencil = diagnostic_stencil < 0 ? opt.fd_stencil : diagnostic_stencil;
  if (stencil < 2 || stencil > 4 || stencil > pmy_pack->pmesh->mb_indcs.ng) {
    std::cerr << "### FATAL ERROR in " << __FILE__
              << ": unsupported AMR jump scratch constraint stencil "
              << stencil << " with nghost=" << pmy_pack->pmesh->mb_indcs.ng
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const int nmb = pmy_pack->nmb_thispack;
  Kokkos::realloc(scratch_adm, nmb, adm::ADM::nadm - 4, bounds.n3,
                  bounds.n2, bounds.n1);
  Kokkos::realloc(scratch_constraints, nmb, ncon, bounds.n3, bounds.n2,
                  bounds.n1);

  adm::ADM::ADM_vars diagnostic_adm;
  diagnostic_adm.psi4.InitWithShallowSlice(scratch_adm, adm::ADM::I_ADM_PSI4);
  diagnostic_adm.g_dd.InitWithShallowSlice(
      scratch_adm, adm::ADM::I_ADM_GXX, adm::ADM::I_ADM_GZZ);
  diagnostic_adm.vK_dd.InitWithShallowSlice(
      scratch_adm, adm::ADM::I_ADM_KXX, adm::ADM::I_ADM_KZZ);

  Constraint_vars diagnostic_constraints;
  diagnostic_constraints.C.InitWithShallowSlice(scratch_constraints, I_CON_C);
  diagnostic_constraints.H.InitWithShallowSlice(scratch_constraints, I_CON_H);
  diagnostic_constraints.M.InitWithShallowSlice(scratch_constraints, I_CON_M);
  diagnostic_constraints.Z.InitWithShallowSlice(scratch_constraints, I_CON_Z);
  diagnostic_constraints.M_d.InitWithShallowSlice(
      scratch_constraints, I_CON_MX, I_CON_MZ);

  Z4cToADMViews(pmy_pack, z4c, diagnostic_adm, opt.chi_psi_power);
  const Tmunu::Tmunu_vars empty_matter;
  switch (stencil) {
    case 2:
      ADMConstraintsViewsImpl<CartoonSO2, 2>(
          pmy_pack, z4c, diagnostic_adm, scratch_constraints,
          diagnostic_constraints, true, empty_matter);
      break;
    case 3:
      ADMConstraintsViewsImpl<CartoonSO2, 3>(
          pmy_pack, z4c, diagnostic_adm, scratch_constraints,
          diagnostic_constraints, true, empty_matter);
      break;
    case 4:
      ADMConstraintsViewsImpl<CartoonSO2, 4>(
          pmy_pack, z4c, diagnostic_adm, scratch_constraints,
          diagnostic_constraints, true, empty_matter);
      break;
    default:
      std::cerr << "### FATAL ERROR in " << __FILE__
                << ": unsupported AMR jump scratch constraint stencil "
                << stencil << std::endl;
      std::exit(EXIT_FAILURE);
  }
  ReconstructConstraintAxisParityGhosts(scratch_constraints);
}

template void ADMConstraintsImpl<Cartesian3D, 2>(MeshBlockPack *);
template void ADMConstraintsImpl<Cartesian3D, 3>(MeshBlockPack *);
template void ADMConstraintsImpl<Cartesian3D, 4>(MeshBlockPack *);
template void ADMConstraintsImpl<CartoonSO2, 2>(MeshBlockPack *);
template void ADMConstraintsImpl<CartoonSO2, 3>(MeshBlockPack *);
template void ADMConstraintsImpl<CartoonSO2, 4>(MeshBlockPack *);
template void Z4c::ADMConstraints<2>(MeshBlockPack *pmbp);
template void Z4c::ADMConstraints<3>(MeshBlockPack *pmbp);
template void Z4c::ADMConstraints<4>(MeshBlockPack *pmbp);
} // namespace z4c
