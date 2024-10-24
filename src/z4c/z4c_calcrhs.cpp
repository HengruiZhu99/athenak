//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \fn TaskStatus Z4c::CalcRHS
//! \brief Computes the wave equation RHS

#include <math.h>

//#include <algorithm>
//#include <cinttypes>
#include <iostream>
//#include <limits>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "z4c/z4c.hpp"
#include "z4c/tmunu.hpp"
#include "coordinates/cell_locations.hpp"

namespace z4c {

template <int NGHOST>
//! \fn void Z4c::CalcRHS(Driver *pdriver, int stage)
//! \brief compute rhs of the z4c equations
TaskStatus Z4c::CalcRHS(Driver *pdriver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;

  int nmb = pmy_pack->nmb_thispack;

  auto &z4c = pmy_pack->pz4c->z4c;
  auto &rhs = pmy_pack->pz4c->rhs;
  auto &opt = pmy_pack->pz4c->opt;

  bool is_vacuum = (pmy_pack->ptmunu == nullptr) ? true : false;
  Tmunu::Tmunu_vars tmunu;
  if (!is_vacuum) tmunu = pmy_pack->ptmunu->tmunu;

  // set team_size
  int team_size = 128;
  // scratch size
  size_t scr_size = ScrArray1D<Real>::shmem_size(team_size)*15 // 0 tensors
                + ScrArray2D<Real>::shmem_size(3,team_size)*10 // vectors
                + ScrArray2D<Real>::shmem_size(6,team_size)*11 // rank 2 tensor with symm
                + ScrArray2D<Real>::shmem_size(9,team_size)*2  // rank 2 tensor with no symm
                + ScrArray2D<Real>::shmem_size(18,team_size)*4 // rank 3 tensor with symm
                + ScrArray2D<Real>::shmem_size(36,team_size);  // rank 4 tensor with symm
  // play with this later
  int scr_level = 1;
  // ===================================================================================
  // Main RHS calculation
  //
  const int nk = ke - ks + 1;
  const int nj = je - js + 1;
  const int ni = ie - is + 1;
  const int nji = ni * nj;
  const int nkji = nk * nji;
  const int nmkji = nmb * nkji;

  const int nteam = nmkji / team_size;

  par_for_outer("z4c rhs loop",DevExeSpace(), scr_size, scr_level, 0, nteam - 1,
  KOKKOS_LAMBDA(TeamMember_t member, const int team) {
    // Define scratch arrays to be used in the following calculations
    // inverse of conf. metric
    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 2> g_uu;
    // inverse of A
    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 2> A_uu;
    // g^cd A_ac A_db
    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 2> AA_dd;
    // Ricci tensor
    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 2> R_dd;
    // Ricci tensor, conformal contribution
    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 2> Rphi_dd;
    // 2nd differential of the lapse
    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 2> Ddalpha_dd;
    // 2nd differential of phi
    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 2> Ddphi_dd;
    // Christoffel symbols of 1st kind
    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 3> Gamma_ddd;
    // Christoffel symbols of 2nd kind
    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 3> Gamma_udd;
    // metric 1st drvts
    AthenaScratchTensor<Real, TensorSymm::SYM2,  3, 3> dg_ddd;
    // shift 2nd drvts
    AthenaScratchTensor<Real, TensorSymm::ISYM2, 3, 3> ddbeta_ddu;
    // metric 2nd drvts
    AthenaScratchTensor<Real, TensorSymm::SYM22, 3, 4> ddg_dddd;
    // lapse 2nd drvts
    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 2> ddalpha_dd;
    // shift 1st drvts
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 2> dbeta_du;
    // chi 2nd drvts
    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 2> ddchi_dd;
    // Gamma 1st drvts
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 2> dGam_du;
    // Lie derivative of conf. 3-metric
    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 2> Lg_dd;
    // Lie derivative of A
    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 2> LA_dd;
    
    // Gamma computed from the metric
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> Gamma_u;
    // Covariant derivative of A
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> DA_u;
    // lapse 1st drvts
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> dalpha_d;
    // 2nd "divergence" of beta
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> ddbeta_d;
    // chi 1st drvts
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> dchi_d;
    // phi 1st drvts
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> dphi_d;
    // Khat 1st drvts
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> dKhat_d;
    // Theta 1st drvts
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> dTheta_d;
    // Lie derivative of Gamma
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> LGam_u;
    // Lie derivative of the shift
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> Lbeta_u;

    g_uu.NewAthenaScratchTensor(member, scr_level, team_size);
    A_uu.NewAthenaScratchTensor(member, scr_level, team_size);
    AA_dd.NewAthenaScratchTensor(member, scr_level, team_size);
    R_dd.NewAthenaScratchTensor(member, scr_level, team_size);
    Rphi_dd.NewAthenaScratchTensor(member, scr_level, team_size);
    Ddalpha_dd.NewAthenaScratchTensor(member, scr_level, team_size);
    Ddphi_dd.NewAthenaScratchTensor(member, scr_level, team_size);
    Gamma_ddd.NewAthenaScratchTensor(member, scr_level, team_size);
    Gamma_udd.NewAthenaScratchTensor(member, scr_level, team_size);
    dg_ddd.NewAthenaScratchTensor(member, scr_level, team_size);
    ddbeta_ddu.NewAthenaScratchTensor(member, scr_level, team_size);
    ddg_dddd.NewAthenaScratchTensor(member, scr_level, team_size);
    ddalpha_dd.NewAthenaScratchTensor(member, scr_level, team_size);
    dbeta_du.NewAthenaScratchTensor(member, scr_level, team_size);
    ddchi_dd.NewAthenaScratchTensor(member, scr_level, team_size);
    dGam_du.NewAthenaScratchTensor(member, scr_level, team_size);
    Lg_dd.NewAthenaScratchTensor(member, scr_level, team_size);
    LA_dd.NewAthenaScratchTensor(member, scr_level, team_size);
    Gamma_u.NewAthenaScratchTensor(member, scr_level, team_size);
    DA_u.NewAthenaScratchTensor(member, scr_level, team_size);
    dalpha_d.NewAthenaScratchTensor(member, scr_level, team_size);
    ddbeta_d.NewAthenaScratchTensor(member, scr_level, team_size);
    dchi_d.NewAthenaScratchTensor(member, scr_level, team_size);
    dphi_d.NewAthenaScratchTensor(member, scr_level, team_size);
    dKhat_d.NewAthenaScratchTensor(member, scr_level, team_size);
    dTheta_d.NewAthenaScratchTensor(member, scr_level, team_size);
    LGam_u.NewAthenaScratchTensor(member, scr_level, team_size);
    Lbeta_u.NewAthenaScratchTensor(member, scr_level, team_size);

    par_for_inner(member, 0, team_size - 1, [&](const int t) {
      int index = team_size * team + t;
      int m = (index)/nkji;
      int k = (index - m*nkji)/nji;
      int j = (index - m*nkji - k*nji)/ni;
      int i = (index - m*nkji - k*nji - j*ni) + is;
      j += js;
      k += ks;

      Real idx[] = {1/size.d_view(m).dx1, 1/size.d_view(m).dx2, 1/size.d_view(m).dx3};

      // -----------------------------------------------------------------------------------
      // Initialize everything to zero
      //
      // Scalars

      // auxiliary Lie derivatives along the shift vector
      // Lie derivative of the lapse
      Real Lalpha = 0.0;
      // Lie derivative of chi
      Real Lchi = 0.0;
      // Lie derivative of Khat
      Real LKhat = 0.0;
      // Lie derivative of Theta
      Real LTheta = 0.0;

      // determinant of three metric
      Real detg = 0.0;
      // bounded version of chi
      Real chi_guarded = 0.0;
      // 1/psi4
      Real oopsi4 = 0.0;
      // trace of A
      Real AA = 0.0;
      // Ricci scalar
      Real R = 0.0;
      // tilde H
      Real Ht = 0.0;
      // trace of extrinsic curvature
      Real K = 0.0;
      // Trace of S_ik
      Real S = 0.0;
      // Trace of Ddalpha_dd
      Real Ddalpha = 0.0;
      // d_a beta^a
      Real dbeta = 0.0;

      //
      // Vectors
      for (int a = 0; a < 3; ++a) {
        Lbeta_u(a,t) = 0.0;
        LGam_u(a,t) = 0.0;
        Gamma_u(a,t) = 0.0;
        DA_u(a,t) = 0.0;
        ddbeta_d(a,t) = 0.0;
      }

      //
      // Symmetric tensors
      for (int a = 0; a < 3; ++a)
      for (int b = a; b < 3; ++b) {
        Lg_dd(a,b,t) = 0.0;
        LA_dd(a,b,t) = 0.0;
        AA_dd(a,b,t) = 0.0;
        R_dd(a,b,t) = 0.0;
        A_uu(a,b,t) = 0.0;
        for (int c = 0; c < 3; ++c) {
            Gamma_udd(c,a,b,t) = 0.0;
        }
      }

      // -----------------------------------------------------------------------------------
      // 1st derivatives
      //
      // Scalars
      for(int a = 0; a < 3; ++a) {
        dalpha_d(a,t) = Dx<NGHOST>(a, idx, z4c.alpha, m,k,j,i);
        dchi_d  (a,t) = Dx<NGHOST>(a, idx, z4c.chi,   m,k,j,i);
        dKhat_d (a,t) = Dx<NGHOST>(a, idx, z4c.vKhat,  m,k,j,i);
        dTheta_d(a,t) = Dx<NGHOST>(a, idx, z4c.vTheta, m,k,j,i);
      }

      // Vectors
      for(int a = 0; a < 3; ++a)
      for(int b = 0; b < 3; ++b) {
        dbeta_du(b,a,t) = Dx<NGHOST>(b, idx, z4c.beta_u, m,a,k,j,i);
        dGam_du(b,a,t) = Dx<NGHOST>(b, idx, z4c.vGam_u,  m,a,k,j,i);
      }

      // Tensors
      for(int a = 0; a < 3; ++a)
      for(int b = a; b < 3; ++b)
      for(int c = 0; c < 3; ++c) {
        dg_ddd(c,a,b,t) = Dx<NGHOST>(c, idx, z4c.g_dd, m,a,b,k,j,i);
      }

      // -----------------------------------------------------------------------------------
      // 2nd derivatives
      //
      // Scalars
      for(int a = 0; a < 3; ++a) {
        ddalpha_dd(a,a,t) = Dxx<NGHOST>(a, idx, z4c.alpha, m,k,j,i);
        ddchi_dd(a,a,t) = Dxx<NGHOST>(a, idx, z4c.chi,   m,k,j,i);

        for(int b = a + 1; b < 3; ++b) {
          ddalpha_dd(a,b,t) = Dxy<NGHOST>(a, b, idx, z4c.alpha, m,k,j,i);
          ddchi_dd(a,b,t) = Dxy<NGHOST>(a, b, idx, z4c.chi,   m,k,j,i);
        }
      }

      // Vectors
      for(int c = 0; c < 3; ++c)
      for(int a = 0; a < 3; ++a) {
        ddbeta_ddu(a,a,c,t) = Dxx<NGHOST>(a, idx, z4c.beta_u, m,c,k,j,i);
        for(int b = a + 1; b < 3; ++b) {
          ddbeta_ddu(a,b,c,t) = Dxy<NGHOST>(a, b, idx, z4c.beta_u, m,c,k,j,i);
        }
      }

      // Tensors
      for(int c = 0; c < 3; ++c)
      for(int d = c; d < 3; ++d)
      for(int a = 0; a < 3; ++a) {
        ddg_dddd(a,a,c,d,t) = Dxx<NGHOST>(a, idx, z4c.g_dd, m,c,d,k,j,i);
        for(int b = a + 1; b < 3; ++b) {
          ddg_dddd(a,b,c,d,t) = Dxy<NGHOST>(a, b, idx, z4c.g_dd, m,c,d,k,j,i);
        }
      }

      // -----------------------------------------------------------------------------------
      // Advective derivatives
      //

      //
      // Scalars
      for(int a = 0; a < 3; ++a) {
        Lalpha += Lx<NGHOST>(a, idx, z4c.beta_u, z4c.alpha, m,a,k,j,i);
        Lchi   += Lx<NGHOST>(a, idx, z4c.beta_u, z4c.chi,   m,a,k,j,i);
        LKhat  += Lx<NGHOST>(a, idx, z4c.beta_u, z4c.vKhat,  m,a,k,j,i);
        LTheta += Lx<NGHOST>(a, idx, z4c.beta_u, z4c.vTheta, m,a,k,j,i);
      }

      //
      // Vectors
      for(int a = 0; a < 3; ++a)
      for(int b = 0; b < 3; ++b) {
        Lbeta_u(b,t) += Lx<NGHOST>(a, idx, z4c.beta_u, z4c.beta_u, m,a,b,k,j,i);
        LGam_u(b,t)  += Lx<NGHOST>(a, idx, z4c.beta_u, z4c.vGam_u,  m,a,b,k,j,i);
      }

      //
      // Tensors
      for(int a = 0; a < 3; ++a)
      for(int b = a; b < 3; ++b)
      for(int c = 0; c < 3; ++c) {
        Lg_dd(a,b,t) += Lx<NGHOST>(c, idx, z4c.beta_u, z4c.g_dd, m,c,a,b,k,j,i);
        LA_dd(a,b,t) += Lx<NGHOST>(c, idx, z4c.beta_u, z4c.vA_dd, m,c,a,b,k,j,i);
      }

      // -----------------------------------------------------------------------------------
      // Get K from Khat
      //
      K = z4c.vKhat(m,k,j,i) + 2.*z4c.vTheta(m,k,j,i);

      // -----------------------------------------------------------------------------------
      // Inverse metric

      detg = adm::SpatialDet(z4c.g_dd(m,0,0,k,j,i), z4c.g_dd(m,0,1,k,j,i),
                                z4c.g_dd(m,0,2,k,j,i), z4c.g_dd(m,1,1,k,j,i),
                                z4c.g_dd(m,1,2,k,j,i), z4c.g_dd(m,2,2,k,j,i));
      adm::SpatialInv(1.0/detg,
                z4c.g_dd(m,0,0,k,j,i), z4c.g_dd(m,0,1,k,j,i), z4c.g_dd(m,0,2,k,j,i),
                z4c.g_dd(m,1,1,k,j,i), z4c.g_dd(m,1,2,k,j,i), z4c.g_dd(m,2,2,k,j,i),
                &g_uu(0,0,t), &g_uu(0,1,t), &g_uu(0,2,t),
                &g_uu(1,1,t), &g_uu(1,2,t), &g_uu(2,2,t));

      // -----------------------------------------------------------------------------------
      // Christoffel symbols

      for(int c = 0; c < 3; ++c)
      for(int a = 0; a < 3; ++a)
      for(int b = a; b < 3; ++b) {
        Gamma_ddd(c,a,b,t) = 0.5*(dg_ddd(a,b,c,t) + dg_ddd(b,a,c,t) - dg_ddd(c,a,b,t));
      }
      for(int c = 0; c < 3; ++c)
      for(int a = 0; a < 3; ++a)
      for(int b = a; b < 3; ++b)
      for(int d = 0; d < 3; ++d) {
        Gamma_udd(c,a,b,t) += g_uu(c,d,t)*Gamma_ddd(d,a,b,t);
      }
      // Gamma's computed from the conformal metric (not evolved)
      for(int a = 0; a < 3; ++a)
      for(int b = 0; b < 3; ++b)
      for(int c = 0; c < 3; ++c) {
        Gamma_u(a,t) += g_uu(b,c,t)*Gamma_udd(a,b,c,t);
      }

      // -----------------------------------------------------------------------------------
      // Curvature of conformal metric
      //
      for(int a = 0; a < 3; ++a)
      for(int b = a; b < 3; ++b) {
        for(int c = 0; c < 3; ++c) {
          R_dd(a,b,t) += 0.5*(z4c.g_dd(m,c,a,k,j,i)*dGam_du(b,c,t) +
                            z4c.g_dd(m,c,b,k,j,i)*dGam_du(a,c,t) +
                            Gamma_u(c,t)*(Gamma_ddd(a,b,c,t) + Gamma_ddd(b,a,c,t)));
        }
        for(int c = 0; c < 3; ++c)
        for(int d = 0; d < 3; ++d) {
          R_dd(a,b,t) -= 0.5*g_uu(c,d,t)*ddg_dddd(c,d,a,b,t);
        }
        for(int c = 0; c < 3; ++c)
        for(int d = 0; d < 3; ++d)
        for(int e = 0; e < 3; ++e) {
          R_dd(a,b,t) += g_uu(c,d,t)*(
              Gamma_udd(e,c,a,t)*Gamma_ddd(b,e,d,t) +
              Gamma_udd(e,c,b,t)*Gamma_ddd(a,e,d,t) +
              Gamma_udd(e,a,d,t)*Gamma_ddd(e,c,b,t));
        }
      }

      // -----------------------------------------------------------------------------------
      // Derivatives of conformal factor phi
      //
      chi_guarded = (z4c.chi(m,k,j,i)>opt.chi_div_floor)
                      ? z4c.chi(m,k,j,i) : opt.chi_div_floor;
      oopsi4 = pow(chi_guarded, -4./opt.chi_psi_power);
      for(int a = 0; a < 3; ++a) {
        dphi_d(a,t) = dchi_d(a,t)/(chi_guarded * opt.chi_psi_power);
      }
      for(int a = 0; a < 3; ++a)
      for(int b = a; b < 3; ++b) {
        Ddphi_dd(a,b,t) = ddchi_dd(a,b,t)/(chi_guarded * opt.chi_psi_power) -
          opt.chi_psi_power * dphi_d(a,t) * dphi_d(b,t);
        for(int c = 0; c < 3; ++c) {
          Ddphi_dd(a,b,t) -= Gamma_udd(c,a,b,t)*dphi_d(c,t);
        }
      }

      // -----------------------------------------------------------------------------------
      // Curvature contribution from conformal factor
      //
      for(int a = 0; a < 3; ++a)
      for(int b = a; b < 3; ++b) {
        Rphi_dd(a,b,t) = 4.*dphi_d(a,t)*dphi_d(b,t) - 2.*Ddphi_dd(a,b,t);
        for(int c = 0; c < 3; ++c)
        for(int d = 0; d < 3; ++d) {
          Rphi_dd(a,b,t) -= 2.*z4c.g_dd(m,a,b,k,j,i) * g_uu(c,d,t)*(Ddphi_dd(c,d,t) +
              2.*dphi_d(c,t)*dphi_d(d,t));
        }
      }

      // TODO(JMF): Update with Tmunu terms.
      // -----------------------------------------------------------------------------------
      // Trace of the matter stress tensor
      //
      // Matter commented out
      //S.ZeroClear();
      //member.team_barrier();
      //for(int a = 0; a < 3; ++a)
      //for(int b = 0; b < 3; ++b) {
      //  ILOOP1(1) {
      //    S(1) += oopsi4(1) * g_uu(a,b,i) * mat.S_dd(m,a,b,k,j,i);
      //  }
      //}
      if(!is_vacuum) {
        for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) {
          S += oopsi4 * g_uu(a,b,t) * tmunu.S_dd(m,a,b,k,j,i);
        }
      }

      // -----------------------------------------------------------------------------------
      // 2nd covariant derivative of the lapse
      // TODO(JMF): This could potentially be sped up by calculating d_i phi d^i alpha
      // beforehand.
      for(int a = 0; a < 3; ++a)
      for(int b = 0; b < 3; ++b) {
        Ddalpha_dd(a,b,t) = ddalpha_dd(a,b,t)
                        - 2.*(dphi_d(a,t)*dalpha_d(b,t) + dphi_d(b,t)*dalpha_d(a,t));
        for(int c = 0; c < 3; ++c) {
          Ddalpha_dd(a,b,t) -= Gamma_udd(c,a,b,t)*dalpha_d(c,t);
          for(int d = 0; d < 3; ++d) {
              Ddalpha_dd(a,b,t) += 2.*z4c.g_dd(m,a,b,k,j,i) * g_uu(c,d,t)
              * dphi_d(c,t) * dalpha_d(d,t);
          }
        }
      }

      for(int a = 0; a < 3; ++a)
      for(int b = 0; b < 3; ++b) {
        Ddalpha += oopsi4 * g_uu(a,b,t) * Ddalpha_dd(a,b,t);
      }

      // -----------------------------------------------------------------------------------
      // Contractions of A_ab, inverse, and derivatives
      //
      for(int a = 0; a < 3; ++a)
      for(int b = a; b < 3; ++b)
      for(int c = 0; c < 3; ++c)
      for(int d = 0; d < 3; ++d) {
        AA_dd(a,b,t) += g_uu(c,d,t) * z4c.vA_dd(m,a,c,k,j,i) * z4c.vA_dd(m,d,b,k,j,i);
      }
      for(int a = 0; a < 3; ++a)
      for(int b = 0; b < 3; ++b) {
        AA += g_uu(a,b,t) * AA_dd(a,b,t);
      }
      for(int a = 0; a < 3; ++a)
      for(int b = a; b < 3; ++b)
      for(int c = 0; c < 3; ++c)
      for(int d = 0; d < 3; ++d) {
        A_uu(a,b,t) += g_uu(a,c,t) * g_uu(b,d,t) * z4c.vA_dd(m,c,d,k,j,i);
      }
      // TODO(JMF): dchi_d/chi_guarded is opt.chi_psi_power * dphi_d.
      for(int a = 0; a < 3; ++a) {
        for(int b = 0; b < 3; ++b) {
            DA_u(a,t) -= (3./2.) * A_uu(a,b,t) * dchi_d(b,t) / chi_guarded;
            DA_u(a,t) -= (1./3.) * g_uu(a,b,t) * (2.*dKhat_d(b,t) + dTheta_d(b,t));
        }
        for(int b = 0; b < 3; ++b)
        for(int c = 0; c < 3; ++c) {
          DA_u(a,t) += Gamma_udd(a,b,c,t) * A_uu(b,c,t);
        }
      }

      // -----------------------------------------------------------------------------------
      // Ricci scalar
      //
      for(int a = 0; a < 3; ++a)
      for(int b = 0; b < 3; ++b) {
        R += oopsi4 * g_uu(a,b,t) * (R_dd(a,b,t) + Rphi_dd(a,b,t));
      }

      // -----------------------------------------------------------------------------------
      // Hamiltonian constraint
      //
      Ht = R + (2./3.)*SQR(K) - AA;// - 16.*M_PI*tmunu.E(m,k,j,i);

      // -----------------------------------------------------------------------------------
      // Finalize advective (Lie) derivatives
      //
      // Shift vector contractions
      for(int a = 0; a < 3; ++a) {
        dbeta += dbeta_du(a,a,t);
      }
      for(int a = 0; a < 3; ++a)
      for(int b = 0; b < 3; ++b) {
        ddbeta_d(a,t) += (1./3.) * ddbeta_ddu(a,b,b,t);
      }

      // Finalize Lchi
      Lchi += (1./6.) * opt.chi_psi_power * chi_guarded * dbeta;

      // Finalize LGam_u (note that this is not a real Lie derivative)
      for(int a = 0; a < 3; ++a) {
        LGam_u(a,t) += (2./3.) * Gamma_u(a,t) * dbeta;
        for(int b = 0; b < 3; ++b) {
          LGam_u(a,t) += g_uu(a,b,t) * ddbeta_d(b,t) - Gamma_u(b,t) * dbeta_du(b,a,t);
          for(int c = 0; c < 3; ++c) {
            LGam_u(a,t) += g_uu(b,c,t) * ddbeta_ddu(b,c,a,t);
          }
        }
      }

      // Finalize Lg_dd and LA_dd
      for(int a = 0; a < 3; ++a)
      for(int b = a; b < 3; ++b) {
        Lg_dd(a,b,t) -= (2./3.) * z4c.g_dd(m,a,b,k,j,i) * dbeta;
        for(int c = 0; c < 3; ++c) {
          Lg_dd(a,b,t) += dbeta_du(a,c,t) * z4c.g_dd(m,b,c,k,j,i);
          Lg_dd(a,b,t) += dbeta_du(b,c,t) * z4c.g_dd(m,a,c,k,j,i);
        }
      }
      for(int a = 0; a < 3; ++a)
      for(int b = a; b < 3; ++b) {
        LA_dd(a,b,t) -= (2./3.) * z4c.vA_dd(m,a,b,k,j,i) * dbeta;
        for(int c = 0; c < 3; ++c) {
          LA_dd(a,b,t) += dbeta_du(b,c,t) * z4c.vA_dd(m,a,c,k,j,i);
          LA_dd(a,b,t) += dbeta_du(a,c,t) * z4c.vA_dd(m,b,c,k,j,i);
        }
      }

      // -----------------------------------------------------------------------------------
      // Assemble RHS
      //
      // Khat, chi, and Theta
      rhs.vKhat(m,k,j,i) = - Ddalpha + z4c.alpha(m,k,j,i)
        * (AA + (1./3.)*SQR(K)) +
        LKhat + opt.damp_kappa1*(1 - opt.damp_kappa2)
        * z4c.alpha(m,k,j,i) * z4c.vTheta(m,k,j,i);
      // Matter term
      if(!is_vacuum) {
        rhs.vKhat(m,k,j,i) += 4.*M_PI * z4c.alpha(m,k,j,i) * (S + tmunu.E(m,k,j,i));
      }
      rhs.chi(m,k,j,i) = Lchi - (1./6.) * opt.chi_psi_power *
        chi_guarded * z4c.alpha(m,k,j,i) * K;
      rhs.vTheta(m,k,j,i) = LTheta + z4c.alpha(m,k,j,i) * (
          0.5*Ht - (2. + opt.damp_kappa2) * opt.damp_kappa1 * z4c.vTheta(m,k,j,i));
      // Matter term
      if(!is_vacuum) {
        rhs.vTheta(m,k,j,i) -= 8.*M_PI * z4c.alpha(m,k,j,i) * tmunu.E(m,k,j,i);
      }
      // If BSSN is enabled, theta is disabled.
      rhs.vTheta(m,k,j,i) *= opt.use_z4c;
      // Gamma's
      for(int a = 0; a < 3; ++a) {
        rhs.vGam_u(m,a,k,j,i) = 2.*z4c.alpha(m,k,j,i)*DA_u(a,t) + LGam_u(a,t);
        rhs.vGam_u(m,a,k,j,i) -= 2.*z4c.alpha(m,k,j,i) * opt.damp_kappa1 *
            (z4c.vGam_u(m,a,k,j,i) - Gamma_u(a,t));
        for(int b = 0; b < 3; ++b) {
          rhs.vGam_u(m,a,k,j,i) -= 2. * A_uu(a,b,t) * dalpha_d(b,t);
          // Matter term
          if(!is_vacuum) {
            rhs.vGam_u(m,a,k,j,i) -= 16.*M_PI * z4c.alpha(m,k,j,i)
                                * g_uu(a,b,t) * tmunu.S_d(m,b,k,j,i);
          }
        }
      }

      // g and A
      for(int a = 0; a < 3; ++a)
      for(int b = a; b < 3; ++b) {
        rhs.g_dd(m,a,b,k,j,i) = - 2. * z4c.alpha(m,k,j,i) * z4c.vA_dd(m,a,b,k,j,i)
                        + Lg_dd(a,b,t);
        rhs.vA_dd(m,a,b,k,j,i) = oopsi4 *
            (-Ddalpha_dd(a,b,t) + z4c.alpha(m,k,j,i) * (R_dd(a,b,t) + Rphi_dd(a,b,t)));
        rhs.vA_dd(m,a,b,k,j,i) -= (1./3.) * z4c.g_dd(m,a,b,k,j,i)
                              * (-Ddalpha + z4c.alpha(m,k,j,i)*R);
        rhs.vA_dd(m,a,b,k,j,i) += z4c.alpha(m,k,j,i) * (K*z4c.vA_dd(m,a,b,k,j,i)
                              - 2.*AA_dd(a,b,t));
        rhs.vA_dd(m,a,b,k,j,i) += LA_dd(a,b,t);
        // Matter term
        if(!is_vacuum) {
          rhs.vA_dd(m,a,b,k,j,i) -= 8.*M_PI * z4c.alpha(m,k,j,i) *
                  (oopsi4*tmunu.S_dd(m,a,b,k,j,i) - (1./3.)*S*z4c.g_dd(m,a,b,k,j,i));
        }
      }
      // lapse function
      Real const f = opt.lapse_oplog * opt.lapse_harmonicf
                  + opt.lapse_harmonic * z4c.alpha(m,k,j,i);
      rhs.alpha(m,k,j,i) = opt.lapse_advect * Lalpha
                        - f * z4c.alpha(m,k,j,i) * z4c.vKhat(m,k,j,i);

      // shift vector
      for(int a = 0; a < 3; ++a) {
        rhs.beta_u(m,a,k,j,i) = opt.shift_ggamma * z4c.vGam_u(m,a,k,j,i)
                              + opt.shift_advect * Lbeta_u(a,t);
        rhs.beta_u(m,a,k,j,i) -= opt.shift_eta * z4c.beta_u(m,a,k,j,i);
        // FORCE beta = 0
        //rhs.beta_u(m,a,k,j,i) = 0;
      }

      // harmonic gauge terms
      for(int a = 0; a < 3; ++a) {
        rhs.beta_u(m,a,k,j,i) += opt.shift_alpha2ggamma *
                            SQR(z4c.alpha(m,k,j,i)) * z4c.vGam_u(m,a,k,j,i);
        for(int b = 0; b < 3; ++b) {
          rhs.beta_u(m,a,k,j,i) += opt.shift_hh * z4c.alpha(m,k,j,i) *
            chi_guarded * (0.5 * z4c.alpha(m,k,j,i) * dchi_d(b,t) - dalpha_d(b,t)) * g_uu(a,b,t);
        }
      }
    });
  });
  // ===================================================================================
  // Add dissipation for stability
  //
  Real &diss = pmy_pack->pz4c->diss;
  auto &u0 = pmy_pack->pz4c->u0;
  auto &u_rhs = pmy_pack->pz4c->u_rhs;
  par_for("K-O Dissipation",
  DevExeSpace(),0,nmb-1,0,nz4c-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(const int m, const int n, const int k, const int j, const int i) {
    Real idx[] = {1/size.d_view(m).dx1, 1/size.d_view(m).dx2, 1/size.d_view(m).dx3};
    for(int a = 0; a < 3; ++a) {
      u_rhs(m,n,k,j,i) += Diss<NGHOST>(a, idx, u0, m, n, k, j, i)*diss;
    }
  });

  return TaskStatus::complete;
}

template TaskStatus Z4c::CalcRHS<2>(Driver *pdriver, int stage);
template TaskStatus Z4c::CalcRHS<3>(Driver *pdriver, int stage);
template TaskStatus Z4c::CalcRHS<4>(Driver *pdriver, int stage);
} // namespace z4c
